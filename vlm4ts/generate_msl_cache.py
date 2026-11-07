"""Unconstrained NASA MSL Cache Generation with Ultra-VETime Teacher.

Generates:
1. Training Curriculum Cache (Partition 1: Timestamps 0 to 36,999):
   - Dense onset, plateau, and offset coverage across 19 anomaly episodes in Partition 1
   - Multi-regime nominal rover telemetry from msl_train.csv and clean Partition 1 segments
   - Stored in cache/msl_train_ultravetime_cache.pt
2. Held-Out Evaluation Cache (Partition 2: Timestamps 37,000 to 73,728):
   - Dense coverage across 17 unseen anomaly episodes and matched nominal periods
   - Stored in cache/msl_eval_head_to_head_cache.pt

Key Features:
- Direct Agg canvas buffer rendering via MSLTelemetryRenderer (zero PNG compression)
- Multi-GPU execution (cuda:0 and cuda:1) when available
- Batching (batch_size=16) for fast GPU parallel inference
- DINOv2-Large (1024-dim self-supervised features) + K=5 nominal operating centroids
- Isolated, cloned tensor payloads (total cache size ~40MB train, ~20MB eval; zero memory leakage)
"""

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from sklearn.cluster import KMeans

from vlm4ts.msl_renderer import MSLTelemetryRenderer


def parse_args():
    parser = argparse.ArgumentParser(description="Unconstrained MSL Cache Generation with Ultra-VETime Teacher")
    parser.add_argument("--data_dir", type=str, default="data/msl", help="Path to MSL data directory")
    parser.add_argument("--output_train_cache", type=str, default="cache/msl_train_ultravetime_cache.pt")
    parser.add_argument("--output_eval_cache", type=str, default="cache/msl_eval_head_to_head_cache.pt")
    parser.add_argument("--window_size", type=int, default=200, help="Sliding window length")
    parser.add_argument("--num_train_windows", type=int, default=3000, help="Target training curriculum windows")
    parser.add_argument("--num_eval_windows", type=int, default=1500, help="Target held-out evaluation windows")
    parser.add_argument("--model_id", type=str, default="facebook/dinov2-large", help="Teacher vision backbone")
    parser.add_argument("--n_regimes", type=int, default=5, help="Number of nominal operational clusters")
    parser.add_argument("--eval_only", action="store_true", default=False, help="Generate only evaluation cache")
    parser.add_argument("--train_only", action="store_true", default=False, help="Generate only training cache")
    return parser.parse_args()


def load_msl_data(data_dir="data/msl"):
    """Loads MSL train and test datasets and computes baseline statistics."""
    train_path = os.path.join(data_dir, "msl_train.csv")
    test_path = os.path.join(data_dir, "msl_test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        candidates = [
            ("data/msl/msl_train.csv", "data/msl/msl_test.csv"),
            ("data/msl_train.csv", "data/msl_test.csv"),
            ("../data/msl/msl_train.csv", "../data/msl/msl_test.csv"),
            ("/kaggle/input/vistad-data/msl/msl_train.csv", "/kaggle/input/vistad-data/msl/msl_test.csv"),
            ("/kaggle/input/vistad-data/msl_train.csv", "/kaggle/input/vistad-data/msl_test.csv"),
        ]
        for tr, te in candidates:
            if os.path.exists(tr) and os.path.exists(te):
                train_path, test_path = tr, te
                break

    print(f"Loading MSL nominal train from: {train_path}")
    df_train = pd.read_csv(train_path)
    train_cols = [c for c in df_train.columns if c not in ["attack", "label", "timestamp"]]
    train_data = df_train[train_cols].values.astype(np.float32)

    base_m = np.mean(train_data, axis=0)
    base_s = np.std(train_data, axis=0)
    base_s[base_s == 0] = 1.0

    print(f"Loading MSL test stream from: {test_path}")
    df_test = pd.read_csv(test_path)
    test_cols = [c for c in df_test.columns if c not in ["attack", "label", "timestamp"]]
    test_data = df_test[test_cols].values.astype(np.float32)
    labels = df_test["attack"].values.astype(np.int64) if "attack" in df_test.columns else np.zeros(len(df_test), dtype=np.int64)

    print(f"Loaded {len(train_data)} train steps (100% nominal), {len(test_data)} test steps across {test_data.shape[1]} channels.")
    print(f"Test anomalies: {labels.sum()} points ({100*labels.mean():.2f}%)")
    return train_data, test_data, labels, base_m, base_s


class UltraVETimeMSLTeacher:
    """Ultra-VETime Teacher for NASA MSL:
    - Backbone: facebook/dinov2-large (24 layers, 1024-dim features)
    - K=5 Nominal Operating Regime Centroids from msl_train.csv
    - 4-Panel Telemetry-Command Dashboard Renderer
    """
    def __init__(self, model_id="facebook/dinov2-large", device="cuda" if torch.cuda.is_available() else "cpu", n_regimes=5):
        self.device = device
        self.model_id = model_id
        self.n_regimes = n_regimes
        print(f"[Ultra-VETime MSL] Loading {model_id} on {self.device}...")
        from transformers import AutoImageProcessor, AutoModel
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        self.renderer = None
        self.regimes_cls = None
        self.regimes_patch = None
        self.nominal_dist_mean = 0.0
        self.nominal_dist_std = 1.0

    def calibrate(self, train_data, base_m, base_s, n_calibration_windows=80, w_size=200):
        """Extracts DINOv2 embeddings across clean nominal windows and clusters K=5 operational regimes."""
        print(f"[Ultra-VETime MSL] Calibrating nominal operating regimes from {n_calibration_windows} train windows...")
        self.renderer = MSLTelemetryRenderer(base_m, base_s)

        step = (len(train_data) - w_size) // n_calibration_windows
        calib_windows = [train_data[i * step : i * step + w_size] for i in range(n_calibration_windows)]
        imgs = [Image.fromarray(self.renderer.render_window(w)) for w in calib_windows]

        cls_feats = []
        patch_feats = []

        batch_size = 16
        for b_start in range(0, len(imgs), batch_size):
            b_imgs = imgs[b_start : b_start + batch_size]
            inputs = self.processor(images=b_imgs, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model(**inputs)
                hs = out.last_hidden_state
                cls_norm = nn.functional.normalize(hs[:, 0, :], p=2, dim=-1)
                grid = hs[:, 1:, :].reshape(-1, 16, 16, 1024)
                col_f = nn.functional.normalize(torch.mean(grid[:, :, 8:, :], dim=1), p=2, dim=-1)

            cls_feats.append(cls_norm.cpu().numpy())
            patch_feats.append(col_f.cpu().numpy())

        cls_matrix = np.concatenate(cls_feats, axis=0)
        patch_matrix = np.concatenate(patch_feats, axis=0)

        kmeans = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(cls_matrix)
        centers = kmeans.cluster_centers_

        self.regimes_cls = torch.from_numpy(centers).float().to(self.device)
        self.regimes_cls = nn.functional.normalize(self.regimes_cls, p=2, dim=-1)

        # Average patch representation per regime
        k = self.n_regimes
        regime_patches = []
        for reg_id in range(k):
            members = patch_matrix[cluster_labels == reg_id]
            if len(members) == 0:
                members = patch_matrix
            p_mean = np.mean(members, axis=0)
            regime_patches.append(torch.from_numpy(p_mean).float().to(self.device))
        self.regimes_patch = torch.stack(regime_patches, dim=0)
        self.regimes_patch = nn.functional.normalize(self.regimes_patch, p=2, dim=-1)

        # Baseline distance calibration
        with torch.no_grad():
            cls_t = torch.from_numpy(cls_matrix).float().to(self.device)
            sims = torch.mm(cls_t, self.regimes_cls.t())
            max_sims, best_regimes = torch.max(sims, dim=1)
            dists = 1.0 - max_sims
            self.nominal_dist_mean = float(torch.mean(dists).cpu())
            self.nominal_dist_std = float(torch.std(dists).cpu() + 1e-6)

        print(f"[Ultra-VETime MSL] Calibrated {k} regime prototypes. Nominal distance μ={self.nominal_dist_mean:.4f}, σ={self.nominal_dist_std:.4f}")

    def score_windows_batch(self, window_data_list):
        """Scores a batch of windows: returns scores, dense_masks, and cls_features."""
        imgs = [Image.fromarray(self.renderer.render_window(w)) for w in window_data_list]
        inp = self.processor(images=imgs, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inp)
            hs = out.last_hidden_state
            cls_f = nn.functional.normalize(hs[:, 0, :], p=2, dim=-1)
            grid = hs[:, 1:, :].reshape(-1, 16, 16, 1024)
            col_f = nn.functional.normalize(torch.mean(grid[:, :, 8:, :], dim=1), p=2, dim=-1)

            # Cosine similarity to nearest regime
            sims_cls = torch.mm(cls_f, self.regimes_cls.t())
            max_sim_cls, best_regime = torch.max(sims_cls, dim=1)
            d_cls = torch.clamp(1.0 - max_sim_cls, min=0.0).cpu().numpy()

            # Patch temporal projection
            dense_masks = []
            scores = []
            x_anchors = np.linspace(0, 199, 8)
            x_query = np.arange(200)

            for b in range(len(imgs)):
                reg_id = best_regime[b]
                p_proto = self.regimes_patch[reg_id]
                p_sim = torch.sum(col_f[b] * p_proto, dim=-1)
                p_dist = torch.clamp(1.0 - p_sim, min=0.0).cpu().numpy()
                dense_mask = np.interp(x_query, x_anchors, p_dist)
                dense_masks.append(dense_mask)

                raw_score = 0.4 * float(d_cls[b]) + 0.6 * float(np.max(p_dist))
                # Sigmoid calibration against nominal stats
                norm_z = (raw_score - self.nominal_dist_mean) / self.nominal_dist_std
                cal_score = 1.0 / (1.0 + np.exp(-norm_z))
                scores.append(float(cal_score))

            return scores, dense_masks, cls_f.cpu()


def generate_msl_curriculum_indices(labels, st_bound, en_bound, target_count, window_size=200):
    """Generates a rich, balanced set of anomaly-crossing and nominal curriculum windows."""
    y_sub = labels[st_bound:en_bound]
    diff = np.diff(np.pad(y_sub, (1, 1), mode="constant"))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    anomaly_starts = []
    # Dense sampling around all events (onset - 150 to end + 20)
    for s, e in zip(starts, ends):
        global_s = st_bound + s
        global_e = st_bound + e
        win_s = max(st_bound, global_s - 180)
        win_e = min(en_bound - window_size, global_e + 20)
        dur = global_e - global_s
        stride = 5 if dur < 50 else (10 if dur < 200 else 20)
        for st in range(win_s, win_e, stride):
            if st + window_size <= en_bound:
                anomaly_starts.append(st)

    anomaly_starts = sorted(list(set(anomaly_starts)))

    # Nominal interval sampling
    nominal_starts = []
    stride_nom = 20 if target_count > 2000 else 40
    for st in range(st_bound, en_bound - window_size, stride_nom):
        if labels[st : st + window_size].max() == 0:
            nominal_starts.append(st)

    # Balance anomaly and nominal windows
    n_anom = len(anomaly_starts)
    n_target_nom = max(500, target_count - n_anom)
    if len(nominal_starts) > n_target_nom:
        sub_indices = np.linspace(0, len(nominal_starts) - 1, n_target_nom, dtype=int)
        nominal_starts = [nominal_starts[i] for i in sub_indices]

    all_starts = sorted(list(set(anomaly_starts + nominal_starts)))
    print(f"Selected {len(all_starts)} windows ({len(anomaly_starts)} anomaly-crossing, {len(nominal_starts)} nominal)")
    return all_starts


def process_window_chunk(start_indices, data, labels, teacher, device_tag="cuda:0", batch_size=16):
    """Processes a chunk of start indices in batches."""
    results = {}
    total = len(start_indices)
    pbar = tqdm(range(0, total, batch_size), desc=f"[{device_tag}] Generating Cache", unit="batch")

    for i in pbar:
        batch_starts = start_indices[i : i + batch_size]
        batch_windows = [data[st : st + 200] for st in batch_starts]
        scores, dense_masks, cls_feats = teacher.score_windows_batch(batch_windows)

        for b, st in enumerate(batch_starts):
            w_data = batch_windows[b]
            w_label = labels[st : st + 200]
            is_anom = int(w_label.max() > 0)
            ratio = float(w_label.mean())

            # Isolated, cloned tensors (prevents saving entire parent array storage)
            results[int(st)] = {
                "x": torch.tensor(w_data, dtype=torch.float32),
                "y_flag": is_anom,
                "y_dense": torch.tensor(w_label, dtype=torch.float32),
                "anomaly_ratio": ratio,
                "z_teacher": cls_feats[b].clone().detach(),
                "teacher_score": float(scores[b]),
                "teacher_dense_mask": torch.tensor(dense_masks[b], dtype=torch.float32)
            }

    return results


def run_caching_pipeline(starts, data, labels, teachers_by_device, split_name="train"):
    """Shards starts across available GPUs and compiles cache payload."""
    devices = list(teachers_by_device.keys())
    print(f"\n--- Running Caching Pipeline for {split_name.upper()} ({len(starts)} windows across {len(devices)} device(s)) ---")

    if len(devices) >= 2:
        mid = len(starts) // 2
        chunk0 = starts[:mid]
        chunk1 = starts[mid:]
        d0, d1 = devices[0], devices[1]
        print(f"Multi-GPU Sharding: {d0} -> {len(chunk0)} windows, {d1} -> {len(chunk1)} windows")

        with ThreadPoolExecutor(max_workers=2) as executor:
            f0 = executor.submit(process_window_chunk, chunk0, data, labels, teachers_by_device[d0], d0)
            f1 = executor.submit(process_window_chunk, chunk1, data, labels, teachers_by_device[d1], d1)
            res0 = f0.result()
            res1 = f1.result()
        merged = {**res0, **res1}
    else:
        d0 = devices[0]
        merged = process_window_chunk(starts, data, labels, teachers_by_device[d0], d0)

    cache_payload = {
        "metadata": {
            "dataset": "msl",
            "entity": "msl_test",
            "teacher_name": "Ultra-VETime Telemetry-Command Teacher",
            "backbone": teachers_by_device[devices[0]].model_id,
            "n_regimes": teachers_by_device[devices[0]].n_regimes,
            "window_size": 200,
            "total_windows": len(merged),
            "split": split_name,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "windows": merged
    }
    return cache_payload


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_train_cache) or "cache", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_eval_cache) or "cache", exist_ok=True)

    train_data, test_data, labels, base_m, base_s = load_msl_data(args.data_dir)
    n_split = 37000  # Midpoint causal split: Partition 1 [0..36999], Partition 2 [37000..73728]

    # Initialize teacher on available devices
    n_gpus = torch.cuda.device_count()
    teachers_by_device = {}
    devices = ["cuda:0", "cuda:1"] if n_gpus >= 2 else (["cuda:0"] if n_gpus == 1 else ["cpu"])
    for d in devices:
        t = UltraVETimeMSLTeacher(model_id=args.model_id, device=d, n_regimes=args.n_regimes)
        t.calibrate(train_data, base_m, base_s, n_calibration_windows=60)
        teachers_by_device[d] = t

    # 1. Training Cache (Partition 1: 0 to 37,000)
    if not args.eval_only:
        print("\nSelecting Partition 1 Training Curriculum Windows...")
        train_starts = generate_msl_curriculum_indices(labels, 0, n_split, args.num_train_windows, args.window_size)
        t0 = time.time()
        train_cache = run_caching_pipeline(train_starts, test_data, labels, teachers_by_device, split_name="train")
        torch.save(train_cache, args.output_train_cache)
        dt = time.time() - t0
        print(f"Saved Training Cache -> {args.output_train_cache} ({len(train_cache['windows'])} windows, {os.path.getsize(args.output_train_cache)/(1024**2):.2f} MB in {dt:.1f}s)")

    # 2. Evaluation Cache (Partition 2: 37,000 to 73,729)
    if not args.train_only:
        print("\nSelecting Partition 2 Held-Out Evaluation Windows...")
        eval_starts = generate_msl_curriculum_indices(labels, n_split, len(test_data), args.num_eval_windows, args.window_size)
        t0 = time.time()
        eval_cache = run_caching_pipeline(eval_starts, test_data, labels, teachers_by_device, split_name="eval_held_out")
        torch.save(eval_cache, args.output_eval_cache)
        dt = time.time() - t0
        print(f"Saved Evaluation Cache -> {args.output_eval_cache} ({len(eval_cache['windows'])} windows, {os.path.getsize(args.output_eval_cache)/(1024**2):.2f} MB in {dt:.1f}s)")

    print("MSL Cache Generation Complete!")


if __name__ == "__main__":
    main()
