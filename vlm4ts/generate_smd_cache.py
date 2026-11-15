"""Unconstrained Multi-Machine Server Machine Dataset (SMD) Cache Generation with Ultra-VETime Teacher.

Generates:
1. Training Curriculum Cache across all 28 machines (50% Train Split):
   - Dense onset, plateau, and offset coverage across all anomaly episodes in the 50% train partition
   - Multi-regime nominal server telemetry from clean train segments
   - Stored in cache/smd_train_ultravetime_cache.pt
2. Held-Out Head-to-Head Evaluation Cache across all 28 machines (Last 40% Test Split):
   - Dense coverage across unseen anomaly episodes in the held-out test slice and matched nominal intervals
   - Stored in cache/smd_eval_head_to_head_cache.pt

Key Features:
- Direct Agg canvas buffer rendering via SMDTelemetryRenderer (zero PNG compression)
- Multi-GPU execution (cuda:0 and cuda:1) when available
- Batching (batch_size=16) for fast GPU parallel inference
- DINOv2-Large (1024-dim self-supervised features) + K=5 nominal operating centroids
- Isolated, cloned tensor payloads (total cache size ~45MB train, ~25MB eval; zero memory leakage)
"""

import os
import sys
import time
import glob
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
from transformers import AutoImageProcessor, AutoModel

from vlm4ts.smd_renderer import SMDTelemetryRenderer


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Machine SMD Cache Generation with Ultra-VETime Teacher")
    parser.add_argument("--data_dir", type=str, default="data/smd", help="Path to SMD data directory")
    parser.add_argument("--output_train_cache", type=str, default="cache/smd_train_ultravetime_cache.pt")
    parser.add_argument("--output_eval_cache", type=str, default="cache/smd_eval_head_to_head_cache.pt")
    parser.add_argument("--window_size", type=int, default=200, help="Sliding window length")
    parser.add_argument("--num_train_windows", type=int, default=3000, help="Target training curriculum windows")
    parser.add_argument("--num_eval_windows", type=int, default=1500, help="Target held-out evaluation windows")
    parser.add_argument("--model_id", type=str, default="facebook/dinov2-large", help="Teacher vision backbone")
    parser.add_argument("--n_regimes", type=int, default=5, help="Number of nominal operational clusters")
    parser.add_argument("--eval_only", action="store_true", default=False, help="Generate only evaluation cache")
    parser.add_argument("--train_only", action="store_true", default=False, help="Generate only training cache")
    return parser.parse_args()


def discover_smd_files(data_dir="data/smd"):
    """Discovers test telemetry files and label files across local and Kaggle paths."""
    candidates = [
        os.path.join(data_dir, "test"),
        "data/smd/test",
        "/kaggle/input/vistad-data/smd/test",
        "/kaggle/input/vistad-data/data_repository/smd/test"
    ]
    test_dir = None
    for c in candidates:
        if os.path.exists(c) and len(glob.glob(os.path.join(c, "*.txt"))) > 0:
            test_dir = c
            break

    if test_dir is None:
        matches = glob.glob("/kaggle/input/**/smd/test", recursive=True)
        if matches:
            test_dir = matches[0]

    if test_dir is None:
        raise FileNotFoundError(f"Could not locate SMD test directory from data_dir: {data_dir}")

    label_dir = test_dir.replace("test", "test_label")
    test_files = sorted(glob.glob(os.path.join(test_dir, "*.txt")))
    print(f"Discovered {len(test_files)} SMD machine telemetry files in {test_dir}")
    return test_files, label_dir


class UltraVETimeSMDTeacher:
    """Ultra-VETime Teacher evaluating 4-panel server subsystem images with DINOv2-Large."""

    def __init__(self, model_id="facebook/dinov2-large", device="cuda"):
        self.device = device
        self.renderer = SMDTelemetryRenderer(dpi=100)
        print(f"[{device}] Loading Ultra-VETime Teacher vision backbone: {model_id}...")
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device)
        self.model.eval()

        self.regimes_cls = None
        self.regimes_patch = None
        self.nominal_dist_mean = 0.5
        self.nominal_dist_std = 0.15

    def calibrate_nominal_regimes(self, nominal_windows_list, k=5):
        """Calibrates K nominal operating clusters from pure nominal server telemetry windows."""
        print(f"[{self.device}] Calibrating {k} nominal operational centroids across {len(nominal_windows_list)} nominal windows...")
        cls_feats = []
        patch_feats = []

        batch_size = 16
        for i in range(0, len(nominal_windows_list), batch_size):
            batch = nominal_windows_list[i : i + batch_size]
            imgs = [Image.fromarray(self.renderer.render_window(w)) for w in batch]
            inp = self.processor(images=imgs, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model(**inp)
                hs = out.last_hidden_state
                cls_f = nn.functional.normalize(hs[:, 0, :], p=2, dim=-1)
                grid = hs[:, 1:, :].reshape(-1, 16, 16, 1024)
                col_f = nn.functional.normalize(torch.mean(grid, dim=1), p=2, dim=-1)

                cls_feats.append(cls_f.cpu().numpy())
                patch_feats.append(col_f.cpu().numpy())

        cls_matrix = np.concatenate(cls_feats, axis=0)
        patch_matrix = np.concatenate(patch_feats, axis=0)

        # KMeans clustering on CLS embeddings
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(cls_matrix)

        regimes_cls = torch.from_numpy(kmeans.cluster_centers_).float().to(self.device)
        self.regimes_cls = nn.functional.normalize(regimes_cls, p=2, dim=-1)

        # Average patch profile per cluster
        regime_patches = []
        for reg_id in range(k):
            members = patch_matrix[cluster_labels == reg_id]
            if len(members) == 0:
                members = patch_matrix
            p_mean = np.mean(members, axis=0)
            regime_patches.append(torch.from_numpy(p_mean).float().to(self.device))
        self.regimes_patch = torch.stack(regime_patches, dim=0)
        self.regimes_patch = nn.functional.normalize(self.regimes_patch, p=2, dim=-1)

        # Distance calibration
        with torch.no_grad():
            cls_t = torch.from_numpy(cls_matrix).float().to(self.device)
            sims = torch.mm(cls_t, self.regimes_cls.t())
            max_sims, _ = torch.max(sims, dim=1)
            dists = 1.0 - max_sims
            self.nominal_dist_mean = float(torch.mean(dists).cpu())
            self.nominal_dist_std = float(torch.std(dists).cpu() + 1e-6)

        print(f"[{self.device}] Calibrated {k} regime prototypes. Nominal distance μ={self.nominal_dist_mean:.4f}, σ={self.nominal_dist_std:.4f}")

    def score_windows_batch(self, window_data_list):
        """Scores a batch of windows: returns calibrated anomaly scores, dense temporal masks, and cls features."""
        imgs = [Image.fromarray(self.renderer.render_window(w)) for w in window_data_list]
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
                norm_z = (raw_score - self.nominal_dist_mean) / self.nominal_dist_std
                cal_score = 1.0 / (1.0 + np.exp(-norm_z))
                scores.append(float(cal_score))

            return scores, dense_masks, cls_f.cpu()


def extract_multi_machine_curriculum(test_files, label_dir, split_type="train", target_total=3000, window_size=200):
    """Samples a rich balanced curriculum of windows across all 28 SMD servers."""
    print(f"\nExtracting multi-machine curriculum windows for {split_type.upper()} across {len(test_files)} machines...")
    per_machine_quota = max(10, target_total // len(test_files))
    collected_records = []

    for tf in test_files:
        m_name = os.path.basename(tf).replace(".txt", "")
        lf = os.path.join(label_dir, os.path.basename(tf))

        raw_data = pd.read_csv(tf, header=None).values.astype(np.float32)
        labels = pd.read_csv(lf, header=None)[0].values.astype(np.float32)

        n = len(raw_data)
        a = int(n * 0.50)  # train bound
        b = int(n * 0.60)  # val bound

        base_m = np.mean(raw_data[:a], axis=0)
        base_s = np.maximum(np.std(raw_data[:a], axis=0), 0.10)
        norm_data = (raw_data - base_m) / base_s

        if split_type == "train":
            st_bound, en_bound = 0, a
        else:  # "eval" -> held-out last 40%
            st_bound, en_bound = b, n

        sub_y = labels[st_bound:en_bound]
        diff = np.diff(np.pad(sub_y, (1, 1), mode="constant"))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        # 1. Anomaly windows
        anom_starts = []
        for s, e in zip(starts, ends):
            global_s = st_bound + s
            global_e = st_bound + e
            win_s = max(st_bound, global_s - 180)
            win_e = min(en_bound - window_size, global_e + 20)
            dur = global_e - global_s
            stride = 6 if dur < 50 else (12 if dur < 200 else 24)
            for st in range(win_s, win_e, stride):
                if st + window_size <= en_bound:
                    anom_starts.append(st)

        anom_starts = sorted(list(set(anom_starts)))

        # 2. Nominal windows
        nom_starts = []
        stride_nom = 25 if split_type == "train" else 40
        for st in range(st_bound, en_bound - window_size, stride_nom):
            if labels[st : st + window_size].max() == 0:
                nom_starts.append(st)

        # Balance anomaly and nominal per machine
        target_anom = min(len(anom_starts), per_machine_quota // 2)
        if len(anom_starts) > target_anom and target_anom > 0:
            idx_sub = np.linspace(0, len(anom_starts) - 1, target_anom, dtype=int)
            anom_starts = [anom_starts[i] for i in idx_sub]

        target_nom = max(per_machine_quota - len(anom_starts), 10)
        if len(nom_starts) > target_nom:
            idx_sub = np.linspace(0, len(nom_starts) - 1, target_nom, dtype=int)
            nom_starts = [nom_starts[i] for i in idx_sub]

        for st in sorted(anom_starts + nom_starts):
            collected_records.append({
                "machine": m_name,
                "st": st,
                "x": norm_data[st : st + window_size],
                "y": labels[st : st + window_size]
            })

    print(f"Total {split_type.upper()} windows extracted: {len(collected_records)} ({sum(int(r['y'].max() > 0) for r in collected_records)} anom, {sum(int(r['y'].max() == 0) for r in collected_records)} nom)")
    return collected_records


def process_records_chunk(records_chunk, teacher, device_tag="cuda:0", batch_size=16):
    """Processes chunk of records with the Ultra-VETime teacher."""
    results = {}
    pbar = tqdm(range(0, len(records_chunk), batch_size), desc=f"[{device_tag}] Scoring Windows", unit="batch")

    for i in pbar:
        batch_recs = records_chunk[i : i + batch_size]
        batch_windows = [r["x"] for r in batch_recs]
        scores, dense_masks, cls_feats = teacher.score_windows_batch(batch_windows)

        for b, rec in enumerate(batch_recs):
            m_name = rec["machine"]
            st = rec["st"]
            key = f"{m_name}_{st}"
            w_label = rec["y"]
            is_anom = int(w_label.max() > 0)

            # Cloned isolated tensors (zero numpy storage memory leaks)
            results[key] = {
                "machine": m_name,
                "st": int(st),
                "x": torch.tensor(rec["x"], dtype=torch.float32).T,  # [38, 200]
                "y_flag": is_anom,
                "y_dense": torch.tensor(w_label, dtype=torch.float32),
                "z_teacher": cls_feats[b].clone().detach(),
                "teacher_score": float(scores[b]),
                "teacher_dense_mask": torch.tensor(dense_masks[b], dtype=torch.float32)
            }

    return results


def run_smd_caching():
    args = parse_args()
    test_files, label_dir = discover_smd_files(args.data_dir)

    devices = []
    if torch.cuda.is_available():
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    elif torch.backends.mps.is_available():
        devices = ["mps"]
    else:
        devices = ["cpu"]
    print(f"Provisioned Compute Devices: {devices}")

    # Build teachers per device
    teachers = {}
    for d in devices:
        teachers[d] = UltraVETimeSMDTeacher(model_id=args.model_id, device=d)

    # 1. Calibrate nominal regimes using nominal train windows
    train_recs = extract_multi_machine_curriculum(test_files, label_dir, split_type="train", target_total=args.num_train_windows)
    pure_nominals = [r["x"] for r in train_recs if r["y"].max() == 0]
    sample_nominals = pure_nominals[:min(250, len(pure_nominals))]

    for d, t in teachers.items():
        t.calibrate_nominal_regimes(sample_nominals, k=args.n_regimes)

    os.makedirs(os.path.dirname(args.output_train_cache) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_eval_cache) or ".", exist_ok=True)

    # 2. Process Training Cache
    if not args.eval_only:
        t0 = time.time()
        print(f"\n--- Generating Training Curriculum Cache ({len(train_recs)} windows) ---")
        if len(devices) >= 2:
            mid = len(train_recs) // 2
            chunk0, chunk1 = train_recs[:mid], train_recs[mid:]
            with ThreadPoolExecutor(max_workers=2) as ex:
                f0 = ex.submit(process_records_chunk, chunk0, teachers[devices[0]], devices[0])
                f1 = ex.submit(process_records_chunk, chunk1, teachers[devices[1]], devices[1])
                res0, res1 = f0.result(), f1.result()
            train_results = {**res0, **res1}
        else:
            train_results = process_records_chunk(train_recs, teachers[devices[0]], devices[0])

        torch.save(train_results, args.output_train_cache)
        dur = time.time() - t0
        mb = os.path.getsize(args.output_train_cache) / (1024 * 1024)
        print(f"Training cache saved to: {args.output_train_cache} ({mb:.2f} MB, {len(train_results)} records) in {dur:.1f}s")

    # 3. Process Evaluation Cache
    if not args.train_only:
        t0 = time.time()
        eval_recs = extract_multi_machine_curriculum(test_files, label_dir, split_type="eval", target_total=args.num_eval_windows)
        print(f"\n--- Generating Head-to-Head Evaluation Cache ({len(eval_recs)} windows) ---")
        if len(devices) >= 2:
            mid = len(eval_recs) // 2
            chunk0, chunk1 = eval_recs[:mid], eval_recs[mid:]
            with ThreadPoolExecutor(max_workers=2) as ex:
                f0 = ex.submit(process_records_chunk, chunk0, teachers[devices[0]], devices[0])
                f1 = ex.submit(process_records_chunk, chunk1, teachers[devices[1]], devices[1])
                res0, res1 = f0.result(), f1.result()
            eval_results = {**res0, **res1}
        else:
            eval_results = process_records_chunk(eval_recs, teachers[devices[0]], devices[0])

        torch.save(eval_results, args.output_eval_cache)
        dur = time.time() - t0
        mb = os.path.getsize(args.output_eval_cache) / (1024 * 1024)
        print(f"Evaluation cache saved to: {args.output_eval_cache} ({mb:.2f} MB, {len(eval_results)} records) in {dur:.1f}s")

    print("\nAll SMD Ultra-VETime teacher caches generated successfully!")


if __name__ == "__main__":
    run_smd_caching()
