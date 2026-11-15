"""Unconstrained NASA MSL Cache Generation with Ultra-VETime Dual-Subsystem Teacher.

Generates:
1. Training Curriculum Cache (Timestamps 0 to 36,999 + Nominal Train):
   - Dense onset, plateau, and offset coverage across the first 18 anomaly episodes in Partition 1
   - Multi-regime nominal rover telemetry from msl_train.csv and clean Partition 1 segments
   - Stored in cache/msl_train_ultravetime_cache.pt
2. Held-Out Evaluation Cache (Timestamps 37,000 to 73,728):
   - Dense coverage across 18 unseen anomaly episodes and matched nominal periods in Partition 2
   - Stored in cache/msl_eval_head_to_head_cache.pt

Key Algorithmic Upgrades:
- Temporal Column Pooling across 16 width patches: col_f = mean(grid, dim=1) [B, 16, 1024]
- Composite distance metric: 0.4 * d_cls + 0.6 * max(d_patch)
- Safety-margin threshold calibration (mu_nom + 2.0 * sigma_nom)
- Multi-GPU parallel execution (cuda:0 and cuda:1) with ThreadPoolExecutor
- Standardized dictionary cache schema: {'metadata': ..., 'windows': {st: {...}}}
"""

import os
import sys
import time
import argparse
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
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

    candidates = [
        (train_path, test_path),
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

    print(f"Loaded {len(train_data):,} train steps (100% nominal), {len(test_data):,} test steps across {test_data.shape[1]} channels.")
    print(f"Test anomalies: {labels.sum():,} points ({100*labels.mean():.2f}%)")
    return train_data, test_data, labels, base_m, base_s


class UltraVETimeMSLTeacher:
    """Ultra-VETime Teacher for NASA MSL:
    - Backbone: facebook/dinov2-large (24 layers, 1024-dim features)
    - Temporal Column-Pooling: extracts genuine chronological progression across 16 width columns
    - K=5 Nominal Operating Regime Prototypes
    - Calibrated composite distance (CLS + peak temporal patch deviation)
    """
    def __init__(self, model_id="facebook/dinov2-large", device=None, n_regimes=5):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
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
        self.cal_threshold = 0.0

    def calibrate(self, train_data, base_m, base_s, n_calibration_windows=60, w_size=200):
        """Calibrates K nominal operating centroids and composite baseline distance statistics."""
        self.renderer = MSLTelemetryRenderer(base_m, base_s)

        step = max(1, (len(train_data) - w_size) // n_calibration_windows)
        cal_indices = [i * step for i in range(n_calibration_windows)]

        imgs = []
        for st in cal_indices:
            w = train_data[st:st + w_size]
            img = self.renderer.render(w)
            imgs.append(img)

        cls_list, col_list = [], []
        batch_size = 12
        for i in range(0, len(imgs), batch_size):
            b_imgs = imgs[i : i + batch_size]
            inp = self.processor(images=b_imgs, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model(**inp)
                hs = out.last_hidden_state
                cls_f = nn.functional.normalize(hs[:, 0, :], p=2, dim=-1)
                grid = hs[:, 1:, :].reshape(-1, 16, 16, 1024)
                col_f = nn.functional.normalize(torch.mean(grid, dim=1), p=2, dim=-1)
                cls_list.append(cls_f)
                col_list.append(col_f)

        cls_feats = torch.cat(cls_list, dim=0)
        col_feats = torch.cat(col_list, dim=0)

        # Fit KMeans on CLS embeddings
        cls_np = cls_feats.cpu().numpy()
        k = min(self.n_regimes, len(cal_indices))
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(cls_np)

        reg_cls, reg_patch = [], []
        for c_id in range(k):
            members = np.where(kmeans.labels_ == c_id)[0]
            if len(members) == 0:
                members = [0]
            c_cls = torch.mean(cls_feats[members], dim=0, keepdim=True)
            c_patch = torch.mean(col_feats[members], dim=0, keepdim=True)
            reg_cls.append(nn.functional.normalize(c_cls, p=2, dim=-1))
            reg_patch.append(nn.functional.normalize(c_patch, p=2, dim=-1))

        self.regimes_cls = torch.cat(reg_cls, dim=0)      # [K, 1024]
        self.regimes_patch = torch.cat(reg_patch, dim=0)  # [K, 16, 1024]

        # Compute full composite score ON CALIBRATION DATA to calibrate threshold
        with torch.no_grad():
            sims = torch.mm(cls_feats, self.regimes_cls.t())
            max_sims, best_reg = torch.max(sims, dim=1)
            d_cls = torch.clamp(1.0 - max_sims, min=0.0).cpu().numpy()

            raw_cal = []
            for b in range(len(imgs)):
                r_id = best_reg[b]
                p_sim = torch.sum(col_feats[b] * self.regimes_patch[r_id], dim=-1)
                p_dist = torch.clamp(1.0 - p_sim, min=0.0).cpu().numpy()
                raw = 0.4 * float(d_cls[b]) + 0.6 * float(np.max(p_dist))
                raw_cal.append(raw)

            self.nominal_dist_mean = float(np.mean(raw_cal))
            self.nominal_dist_std = float(np.std(raw_cal) + 1e-6)
            self.cal_threshold = self.nominal_dist_mean + 2.0 * self.nominal_dist_std

        print(f"[Ultra-VETime MSL] Calibrated {k} regime prototypes. Nominal distance μ={self.nominal_dist_mean:.4f}, σ={self.nominal_dist_std:.4f}, Thresh={self.cal_threshold:.4f}")

    def score_windows_batch(self, window_data_list):
        """Scores a batch of windows: returns scores, dense_masks, and cls_features."""
        imgs = [self.renderer.render(w) for w in window_data_list]
        inp = self.processor(images=imgs, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inp)
            hs = out.last_hidden_state
            cls_f = nn.functional.normalize(hs[:, 0, :], p=2, dim=-1)
            grid = hs[:, 1:, :].reshape(-1, 16, 16, 1024)
            col_f = nn.functional.normalize(torch.mean(grid, dim=1), p=2, dim=-1)

            # Cosine similarity to nearest regime
            sims_cls = torch.mm(cls_f, self.regimes_cls.t())
            max_sim_cls, best_regime = torch.max(sims_cls, dim=1)
            d_cls = torch.clamp(1.0 - max_sim_cls, min=0.0).cpu().numpy()

            # Patch temporal projection
            dense_masks = []
            scores = []
            x_anchors = np.linspace(0, 199, 16)
            x_query = np.arange(200)

            for b in range(len(imgs)):
                reg_id = best_regime[b]
                p_proto = self.regimes_patch[reg_id]
                p_sim = torch.sum(col_f[b] * p_proto, dim=-1)
                p_dist = torch.clamp(1.0 - p_sim, min=0.0).cpu().numpy()
                dense_mask = np.interp(x_query, x_anchors, p_dist)
                dense_masks.append(dense_mask)

                raw_score = 0.4 * float(d_cls[b]) + 0.6 * float(np.max(p_dist))
                # Sigmoid calibration against nominal stats with +2 sigma margin
                norm_z = (raw_score - self.cal_threshold) / self.nominal_dist_std
                cal_score = 1.0 / (1.0 + np.exp(-norm_z))
                scores.append(float(cal_score))

            return scores, dense_masks, cls_f.cpu()


def find_anomaly_episodes(labels):
    """Returns list of (start, end) tuples for contiguous anomaly segments."""
    diff = np.diff(np.pad(labels, (1, 1), mode="constant"))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts, ends))


def sample_msl_curriculum_windows(train_data, test_data, test_labels, num_windows=3000, w_size=200):
    """Samples phase-diverse training curriculum:
    - First 18 anomaly episodes in test_data (t < 37,000)
    - Diverse normal windows from train_data (100% nominal)
    - Diverse normal windows from test_data Partition 1 (t < 37,000)
    """
    episodes = find_anomaly_episodes(test_labels)
    part1_episodes = [(s, e) for (s, e) in episodes if s < 37000]
    print(f"[Sampling MSL Train] Found {len(part1_episodes)} anomaly episodes in Partition 1 (t < 37,000).")

    anomaly_starts = []
    # Dense sampling around all Partition 1 events
    for s, e in part1_episodes:
        dur = e - s + 1
        win_s = max(0, s - 180)
        win_e = min(37000 - w_size, e + 20)
        stride = 5 if dur < 50 else 10
        for st in range(win_s, win_e, stride):
            if st + w_size <= 37000:
                anomaly_starts.append(st)

    anomaly_starts = sorted(list(set(anomaly_starts)))

    # Nominal windows: half from clean train_data, half from clean segments of test partition 1
    n_anom = len(anomaly_starts)
    n_nom_target = max(600, num_windows - n_anom)

    # Clean test partition 1 nominal starts
    test_nom_starts = []
    for st in range(0, 37000 - w_size, 30):
        if test_labels[st:st + w_size].max() == 0:
            test_nom_starts.append(st)

    # Subsample if needed
    if len(test_nom_starts) > n_nom_target:
        sub_idx = np.linspace(0, len(test_nom_starts) - 1, n_nom_target, dtype=int)
        test_nom_starts = [test_nom_starts[i] for i in sub_idx]

    all_starts = sorted(list(set(anomaly_starts + test_nom_starts)))
    print(f"[Sampling MSL Train] Selected {len(all_starts)} windows ({len(anomaly_starts)} anomaly-crossing, {len(test_nom_starts)} nominal).")
    return all_starts


def sample_msl_eval_windows(test_data, test_labels, num_windows=1500, w_size=200):
    """Samples dedicated held-out evaluation windows from Partition 2 (t >= 37,000, 18 unseen episodes)."""
    episodes = find_anomaly_episodes(test_labels)
    part2_episodes = [(s, e) for (s, e) in episodes if s >= 37000]
    print(f"[Sampling MSL Eval] Found {len(part2_episodes)} unseen anomaly episodes in Partition 2 (t >= 37,000).")

    anomaly_starts = []
    en_bound = len(test_data)
    for s, e in part2_episodes:
        dur = e - s + 1
        win_s = max(37000, s - 180)
        win_e = min(en_bound - w_size, e + 20)
        stride = 5 if dur < 50 else 10
        for st in range(win_s, win_e, stride):
            if st + w_size <= en_bound:
                anomaly_starts.append(st)

    anomaly_starts = sorted(list(set(anomaly_starts)))

    n_anom = len(anomaly_starts)
    n_nom_target = max(400, num_windows - n_anom)

    nom_starts = []
    for st in range(37000, en_bound - w_size, 40):
        if test_labels[st:st + w_size].max() == 0:
            nom_starts.append(st)

    if len(nom_starts) > n_nom_target:
        sub_idx = np.linspace(0, len(nom_starts) - 1, n_nom_target, dtype=int)
        nom_starts = [nom_starts[i] for i in sub_idx]

    all_starts = sorted(list(set(anomaly_starts + nom_starts)))
    print(f"[Sampling MSL Eval] Selected {len(all_starts)} windows ({len(anomaly_starts)} anomaly-crossing, {len(nom_starts)} nominal).")
    return all_starts


def process_window_chunk(start_indices, data, labels, teacher, device_tag="cuda:0", batch_size=16):
    """Processes a chunk of start indices in batches."""
    results = {}
    total = len(start_indices)
    pbar = tqdm(range(0, total, batch_size), desc=f"[{device_tag}] Generating MSL Cache", unit="batch")

    for i in pbar:
        batch_starts = start_indices[i : i + batch_size]
        batch_windows = [data[st : st + 200] for st in batch_starts]
        scores, dense_masks, cls_feats = teacher.score_windows_batch(batch_windows)

        for b, st in enumerate(batch_starts):
            w_data = batch_windows[b]
            w_label = labels[st : st + 200]
            is_anom = int(w_label.max() > 0)
            ratio = float(w_label.mean())

            results[int(st)] = {
                "start_idx": int(st),
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
    print(f"\n--- Running MSL Caching Pipeline for {split_name.upper()} ({len(starts)} windows across {len(devices)} device(s)) ---")

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
            "entity": "msl_rover",
            "teacher_name": "T13: Ultra-VETime Telemetry-Command",
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
    os.makedirs(os.path.dirname(args.output_train_cache) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_eval_cache) or ".", exist_ok=True)

    train_data, test_data, labels, base_m, base_s = load_msl_data(args.data_dir)

    # Detect compute hardware (Multi-GPU if available)
    devices = []
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        devices = [f"cuda:{i}" for i in range(min(num_gpus, 2))]
    elif torch.backends.mps.is_available():
        devices = ["mps"]
    else:
        devices = ["cpu"]

    print(f"Executing MSL Cache Generation across device(s): {devices}")

    # Primary teacher for calibration
    primary_teacher = UltraVETimeMSLTeacher(model_id=args.model_id, device=devices[0], n_regimes=args.n_regimes)
    primary_teacher.calibrate(train_data, base_m, base_s)

    teachers_by_device = {devices[0]: primary_teacher}

    # Clone calibrated parameters to secondary GPU if available
    if len(devices) > 1:
        sec_teacher = UltraVETimeMSLTeacher(model_id=args.model_id, device=devices[1], n_regimes=args.n_regimes)
        sec_teacher.renderer = MSLTelemetryRenderer(base_m, base_s)
        sec_teacher.regimes_cls = primary_teacher.regimes_cls.clone().to(devices[1])
        sec_teacher.regimes_patch = primary_teacher.regimes_patch.clone().to(devices[1])
        sec_teacher.nominal_dist_mean = primary_teacher.nominal_dist_mean
        sec_teacher.nominal_dist_std = primary_teacher.nominal_dist_std
        sec_teacher.cal_threshold = primary_teacher.cal_threshold
        teachers_by_device[devices[1]] = sec_teacher

    # 1. Held-Out Evaluation Cache (Partition 2: t >= 37,000)
    if not args.train_only:
        eval_starts = sample_msl_eval_windows(test_data, labels, num_windows=args.num_eval_windows, w_size=args.window_size)
        eval_payload = run_caching_pipeline(eval_starts, test_data, labels, teachers_by_device, split_name="eval")
        torch.save(eval_payload, args.output_eval_cache)
        print(f"Saved MSL Held-Out Eval Cache to: {args.output_eval_cache} ({len(eval_payload['windows'])} windows)")

    # 2. Training Curriculum Cache (Partition 1: t < 37,000)
    if not args.eval_only:
        train_starts = sample_msl_curriculum_windows(train_data, test_data, labels, num_windows=args.num_train_windows, w_size=args.window_size)
        train_payload = run_caching_pipeline(train_starts, test_data, labels, teachers_by_device, split_name="train")
        torch.save(train_payload, args.output_train_cache)
        print(f"Saved MSL Training Cache to: {args.output_train_cache} ({len(train_payload['windows'])} windows)")

    print("\nMSL Cache Generation Pipeline Complete!")


if __name__ == "__main__":
    main()
