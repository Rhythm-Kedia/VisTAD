"""Unconstrained GECCO Cache Generation with Ultra-VETime Dual-Subsystem Teacher.

Generates:
1. Training Curriculum Cache (Timestamps 0 to 69,999):
   - Dense onset, plateau, and recovery coverage across all 29 contamination events in Partition 1
   - Multi-regime nominal cycles across diurnal demand and flow operating modes
   - Stored in cache/gecco_train_ultravetime_cache.pt
2. Held-Out Evaluation Cache (Timestamps 70,000 to 139,565):
   - Dense coverage across all 22 unseen contamination events and matched nominal periods
   - Stored in cache/gecco_eval_head_to_head_cache.pt

Key Features:
- Direct Agg canvas buffer rendering via GeccoDashboardRenderer (zero PNG compression)
- Multi-GPU execution (cuda:0 and cuda:1) when available
- DINOv2-Large (1024-dim self-supervised features) + K=5 nominal operating centroids
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

from vlm4ts.gecco_renderer import GeccoDashboardRenderer, GECCO_SENSORS


def parse_args():
    parser = argparse.ArgumentParser(description="Unconstrained GECCO Cache Generation with Ultra-VETime Teacher")
    parser.add_argument("--data_path", type=str, default="data/gecco/gecco_water.csv", help="Path to GECCO CSV file")
    parser.add_argument("--output_train_cache", type=str, default="cache/gecco_train_ultravetime_cache.pt")
    parser.add_argument("--output_eval_cache", type=str, default="cache/gecco_eval_head_to_head_cache.pt")
    parser.add_argument("--window_size", type=int, default=200, help="Sliding window length")
    parser.add_argument("--num_train_windows", type=int, default=3500, help="Target training curriculum windows")
    parser.add_argument("--num_eval_windows", type=int, default=1200, help="Target held-out evaluation windows")
    parser.add_argument("--model_id", type=str, default="facebook/dinov2-large", help="Teacher vision backbone")
    parser.add_argument("--n_regimes", type=int, default=5, help="Number of nominal operational clusters")
    parser.add_argument("--eval_only", action="store_true", default=False, help="Generate only evaluation cache")
    parser.add_argument("--train_only", action="store_true", default=False, help="Generate only training cache")
    return parser.parse_args()


def load_gecco_data(data_path="data/gecco/gecco_water.csv"):
    """Loads GECCO data and computes nominal baseline statistics from first 16,000 steps."""
    if not os.path.exists(data_path):
        candidates = [
            "data/gecco/gecco_water.csv",
            "data/gecco_water.csv",
            "../data/gecco/gecco_water.csv",
            "/kaggle/input/vistad-data/gecco/gecco_water.csv",
            "/kaggle/input/vistad-data/gecco_water.csv"
        ]
        for c in candidates:
            if os.path.exists(c):
                data_path = c
                break

    print(f"Loading GECCO drinking water data from: {data_path}")
    df = pd.read_csv(data_path)
    labels = df["attack"].values.astype(np.int64) if "attack" in df.columns else np.zeros(len(df), dtype=np.int64)
    data = df[GECCO_SENSORS].values.astype(np.float32)

    # First 16,000 steps are 100% nominal (ground-truth baseline)
    base_slice = data[:16000]
    base_m = np.mean(base_slice, axis=0)
    base_s = np.maximum(np.std(base_slice, axis=0), 1e-4)

    print(f"Loaded {len(data)} steps, {len(GECCO_SENSORS)} channels. Total anomalies: {labels.sum()} ({100*labels.mean():.2f}%)")
    return data, labels, base_m, base_s


class UltraVETimeGeccoTeacher:
    """Ultra-VETime Teacher for GECCO:
    - Backbone: facebook/dinov2-large (24 layers, 1024-dim features)
    - K=5 Nominal Operating Regime Centroids from 16,000-step nominal baseline
    - 4-Panel Dual-Subsystem Dashboard Renderer
    """
    def __init__(self, model_id="facebook/dinov2-large", device="cuda" if torch.cuda.is_available() else "cpu", n_regimes=5):
        self.device = device
        self.model_id = model_id
        self.n_regimes = n_regimes
        print(f"[Ultra-VETime GECCO] Loading {model_id} on {self.device}...")
        from transformers import AutoImageProcessor, AutoModel
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        self.renderer = None
        self.regimes_cls = None
        self.regimes_patch = None
        self.nominal_dist_mean = 0.0
        self.nominal_dist_std = 1.0

    def calibrate(self, data, base_m, base_s, n_calibration_windows=60):
        """Calibrates K nominal operating centroids and baseline distance statistics."""
        self.renderer = GeccoDashboardRenderer(base_m, base_s)
        cal_starts = np.linspace(500, 15500, n_calibration_windows, dtype=int)

        imgs = []
        for st in cal_starts:
            w_data = data[st : st + 200]
            img_arr = self.renderer.render_window(w_data)
            imgs.append(Image.fromarray(img_arr))

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
                col_f = nn.functional.normalize(torch.mean(grid[:, :, 8:, :], dim=1), p=2, dim=-1)
                cls_list.append(cls_f)
                col_list.append(col_f)

        cls_feats = torch.cat(cls_list, dim=0)
        col_feats = torch.cat(col_list, dim=0)

        # Fit KMeans
        cls_np = cls_feats.cpu().numpy()
        k = min(self.n_regimes, len(cal_starts))
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
        self.regimes_patch = torch.cat(reg_patch, dim=0)  # [K, 8, 1024]

        # Compute baseline nominal distance distribution
        with torch.no_grad():
            sims = torch.mm(cls_feats, self.regimes_cls.t())  # [N, K]
            dists = 1.0 - torch.max(sims, dim=1)[0]
            self.nominal_dist_mean = float(torch.mean(dists).cpu())
            self.nominal_dist_std = float(torch.std(dists).cpu() + 1e-6)

        print(f"[Ultra-VETime GECCO] Calibrated {k} regime prototypes. Nominal distance μ={self.nominal_dist_mean:.4f}, σ={self.nominal_dist_std:.4f}")

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


def generate_curriculum_indices(labels, st_bound, en_bound, target_count, window_size=200):
    """Generates a rich, balanced set of anomaly-dense and nominal curriculum windows."""
    # Find contiguous anomaly events within [st_bound, en_bound]
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
        stride = 5 if (global_e - global_s) < 50 else 10
        for st in range(win_s, win_e, stride):
            if st + window_size <= en_bound:
                anomaly_starts.append(st)

    anomaly_starts = sorted(list(set(anomaly_starts)))

    # Nominal interval sampling
    nominal_starts = []
    stride_nom = 35 if target_count > 2000 else 60
    for st in range(st_bound, en_bound - window_size, stride_nom):
        # Check if strictly nominal
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
            "dataset": "gecco",
            "entity": "gecco_water",
            "teacher_name": "T13: Ultra-VETime Dual-Subsystem",
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

    data, labels, base_m, base_s = load_gecco_data(args.data_path)
    n_split = 70000  # Midpoint causal split: Partition 1 [0..69999], Partition 2 [70000..139565]

    # Initialize teacher on available devices
    n_gpus = torch.cuda.device_count()
    teachers_by_device = {}
    devices = ["cuda:0", "cuda:1"] if n_gpus >= 2 else (["cuda:0"] if n_gpus == 1 else ["cpu"])
    for d in devices:
        t = UltraVETimeGeccoTeacher(model_id=args.model_id, device=d, n_regimes=args.n_regimes)
        t.calibrate(data, base_m, base_s, n_calibration_windows=60)
        teachers_by_device[d] = t

    # 1. Training Cache (Partition 1: 0 to 70,000)
    if not args.eval_only:
        print("\nSelecting Partition 1 Training Curriculum Windows...")
        train_starts = generate_curriculum_indices(labels, 0, n_split, args.num_train_windows, args.window_size)
        t0 = time.time()
        train_cache = run_caching_pipeline(train_starts, data, labels, teachers_by_device, split_name="train")
        torch.save(train_cache, args.output_train_cache)
        dt = time.time() - t0
        print(f"Saved Training Cache -> {args.output_train_cache} ({len(train_cache['windows'])} windows, {os.path.getsize(args.output_train_cache)/(1024**2):.2f} MB in {dt:.1f}s)")

    # 2. Evaluation Cache (Partition 2: 70,000 to 139,566)
    if not args.train_only:
        print("\nSelecting Partition 2 Held-Out Evaluation Windows...")
        eval_starts = generate_curriculum_indices(labels, n_split, len(data), args.num_eval_windows, args.window_size)
        t0 = time.time()
        eval_cache = run_caching_pipeline(eval_starts, data, labels, teachers_by_device, split_name="eval_held_out")
        torch.save(eval_cache, args.output_eval_cache)
        dt = time.time() - t0
        print(f"Saved Evaluation Cache -> {args.output_eval_cache} ({len(eval_cache['windows'])} windows, {os.path.getsize(args.output_eval_cache)/(1024**2):.2f} MB in {dt:.1f}s)")

    print("\n--- GECCO Cache Generation Complete ---")


if __name__ == "__main__":
    main()
