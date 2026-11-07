"""Unconstrained SWaT Cache Generation with Ultra-VETime Multi-Stage Teacher.

Generates:
1. Training Curriculum Cache (Timestamps 0 to 224,959):
   - Dense attack onset, plateau, and offset coverage across 21 attacks
   - Multi-regime nominal cycles from SWaT_Dataset_Normal_v1.csv
   - Stored in cache/swat_train_ultravetime_cache.pt
2. Held-Out Evaluation Cache (Timestamps 224,960 to 449,919):
   - Dense coverage across 14 unseen attacks and matched nominal periods
   - Stored in cache/swat_eval_head_to_head_cache.pt

Supports:
- Direct Agg canvas buffer rendering (no PNG compression overhead)
- Multi-GPU execution (cuda:0 and cuda:1) when multiple GPUs are available
- Multi-stage max-envelope supervision from DINOv2-Large (1024-dim features)
"""

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from vlm4ts.renderer import render_vlm4ts_image
from vlm4ts.stage_coupled_selector import SWAT_STAGE_CHANNELS
from vlm4ts.teacher_evaluator import UltraVETimeTeacher


def parse_args():
    parser = argparse.ArgumentParser(description="Unconstrained SWaT Cache Generation with Ultra-VETime Teacher")
    parser.add_argument("--data_dir", type=str, default="data/swat", help="Directory containing SWaT CSV files")
    parser.add_argument("--output_train_cache", type=str, default="cache/swat_train_ultravetime_cache.pt")
    parser.add_argument("--output_eval_cache", type=str, default="cache/swat_eval_head_to_head_cache.pt")
    parser.add_argument("--window_size", type=int, default=200, help="Sliding window length")
    parser.add_argument("--num_train_windows", type=int, default=3500, help="Total training curriculum windows")
    parser.add_argument("--num_eval_windows", type=int, default=1200, help="Total held-out evaluation windows")
    parser.add_argument("--model_id", type=str, default="facebook/dinov2-large", help="Teacher vision backbone")
    parser.add_argument("--n_regimes", type=int, default=5, help="Number of nominal operational clusters per stage")
    parser.add_argument("--eval_only", action="store_true", default=False, help="Generate only evaluation cache")
    parser.add_argument("--train_only", action="store_true", default=False, help="Generate only training cache")
    return parser.parse_args()


def load_swat_data(data_dir="data/swat"):
    """Loads 7-day nominal baseline and full attack test stream."""
    norm_path = os.path.join(data_dir, "SWaT_Dataset_Normal_v1.csv")
    att_path = os.path.join(data_dir, "SWaT_Dataset_Attack_v0.csv")

    if not os.path.exists(norm_path) or not os.path.exists(att_path):
        # Fallback to alternate search paths
        for root in ["data/swat", "data", "../data/swat", "../data", "/kaggle/input/vistad-data/swat"]:
            p1 = os.path.join(root, "SWaT_Dataset_Normal_v1.csv")
            p2 = os.path.join(root, "SWaT_Dataset_Attack_v0.csv")
            if os.path.exists(p1) and os.path.exists(p2):
                norm_path, att_path = p1, p2
                break

    print(f"Loading normal baseline from: {norm_path}")
    df_norm = pd.read_csv(norm_path)
    norm_channels = [c for c in df_norm.columns if c not in ["Timestamp", "Normal/Attack", "attack", "label"]]
    norm_data = df_norm[norm_channels].values.astype(np.float32)

    base_m = np.mean(norm_data, axis=0)
    base_s = np.maximum(np.std(norm_data, axis=0), 0.05)

    print(f"Loading attack test stream from: {att_path}")
    df_att = pd.read_csv(att_path)
    att_cols = [c for c in df_att.columns if c not in ["Timestamp", "Normal/Attack", "attack", "label"]]
    att_data = df_att[att_cols].values.astype(np.float32)

    label_col = [c for c in df_att.columns if c in ["Normal/Attack", "attack", "label"]][0]
    raw_labels = df_att[label_col].values
    if raw_labels.dtype == object:
        labels = (raw_labels != "Normal").astype(np.float32)
    else:
        labels = (raw_labels == 1).astype(np.float32)

    print(f"Loaded {len(norm_data)} normal timestamps, {len(att_data)} attack timestamps across {att_data.shape[1]} channels.")
    return norm_data, att_data, labels, base_m, base_s


def find_attack_episodes(labels):
    """Finds contiguous attack intervals [start, end)."""
    diffs = np.diff(np.pad(labels, (1, 1), mode="constant"))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    return list(zip(starts, ends))


def sample_curriculum_training_windows(train_data, train_labels, norm_data, num_windows=3500, w_size=200):
    """Samples dense phase-diverse attack onsets, plateaus, and multi-regime normal cycles."""
    n_train = len(train_data)
    episodes = find_attack_episodes(train_labels)
    print(f"[Curriculum Sampling] Found {len(episodes)} attack episodes in train partition (0 to {n_train}).")

    anom_starts = []
    # 1. Dense onset transitions: capture attack arriving at every 10-step increment
    onset_offsets = [0, 10, 20, 30, 40, 50, 75, 100, 125, 150, 175, 190]
    for s_ep, e_ep in episodes:
        for offset in onset_offsets:
            st = s_ep - (w_size - offset)
            if 0 <= st and st + w_size <= n_train:
                anom_starts.append(st)

        # 2. Interior plateaus and recoveries
        if e_ep - s_ep > w_size:
            for st in range(s_ep, e_ep - w_size, 30):
                if st + w_size <= n_train:
                    anom_starts.append(st)

        # 3. Offsets / recovery
        for off in [10, 30, 50, 80]:
            st = e_ep - off
            if 0 <= st and st + w_size <= n_train:
                anom_starts.append(st)

    anom_starts = sorted(list(set(anom_starts)))
    target_anom = num_windows // 2
    if len(anom_starts) > target_anom:
        step = len(anom_starts) / target_anom
        anom_starts = [anom_starts[int(i * step)] for i in range(target_anom)]

    # 4. Dense nominal operation windows
    target_norm = num_windows - len(anom_starts)
    norm_starts = []
    step_norm = max(10, (n_train - w_size) // (target_norm + 1))
    for s in range(0, n_train - w_size, step_norm):
        if np.all(train_labels[s : s + w_size] == 0):
            norm_starts.append(s)
        if len(norm_starts) >= target_norm:
            break

    # Combine and interleave
    all_starts = sorted(list(set(anom_starts + norm_starts)))
    print(f"[Curriculum Sampling] Selected {len(all_starts)} training windows ({len(anom_starts)} attack, {len(norm_starts)} nominal).")
    return all_starts


def sample_held_out_evaluation_windows(eval_data, eval_labels, global_offset, num_windows=1200, w_size=200):
    """Samples evaluation windows covering all unseen attacks in the second partition."""
    n_eval = len(eval_data)
    episodes = find_attack_episodes(eval_labels)
    print(f"[Evaluation Sampling] Found {len(episodes)} attack episodes in held-out partition ({global_offset} to {global_offset + n_eval}).")

    anom_starts = []
    onset_offsets = [10, 30, 60, 100, 150, 180]
    for s_ep, e_ep in episodes:
        for offset in onset_offsets:
            st = s_ep - (w_size - offset)
            if 0 <= st and st + w_size <= n_eval:
                anom_starts.append(st)

        if e_ep - s_ep > w_size:
            for st in range(s_ep, e_ep - w_size, 40):
                if st + w_size <= n_eval:
                    anom_starts.append(st)

        for off in [20, 50]:
            st = e_ep - off
            if 0 <= st and st + w_size <= n_eval:
                anom_starts.append(st)

    anom_starts = sorted(list(set(anom_starts)))
    target_anom = num_windows // 2
    if len(anom_starts) > target_anom:
        step = len(anom_starts) / target_anom
        anom_starts = [anom_starts[int(i * step)] for i in range(target_anom)]

    target_norm = num_windows - len(anom_starts)
    norm_starts = []
    step_norm = max(10, (n_eval - w_size) // (target_norm + 1))
    for s in range(0, n_eval - w_size, step_norm):
        if np.all(eval_labels[s : s + w_size] == 0):
            norm_starts.append(s)
        if len(norm_starts) >= target_norm:
            break

    # Convert to global test stream coordinates
    global_starts = sorted([s + global_offset for s in set(anom_starts + norm_starts)])
    print(f"[Evaluation Sampling] Selected {len(global_starts)} held-out evaluation windows.")
    return global_starts


def process_window_chunk(start_indices, full_data, full_labels, base_m, base_s,
                         teacher, device_str="cuda:0"):
    """Worker function to process a slice of windows on an initialized teacher instance."""
    results = {}
    for st in tqdm(start_indices, desc=f"Caching {device_str}"):
        ed = st + 200
        w_data = full_data[st:ed]
        w_labels = full_labels[st:ed]

        # Stage deviation scoring
        z = np.abs((w_data - base_m) / base_s)
        stage_devs = {
            int(sid): float(np.mean(z[:, smeta["channels"]]))
            for sid, smeta in SWAT_STAGE_CHANNELS.items()
        }
        ranked_stages = sorted(stage_devs.items(), key=lambda kv: kv[1], reverse=True)
        s1 = int(ranked_stages[0][0])
        s2 = int(ranked_stages[1][0]) if len(ranked_stages) > 1 else s1

        # Render top-2 stages using fast Agg canvas buffer
        chans1 = SWAT_STAGE_CHANNELS[s1]["channels"][:6]
        chans2 = SWAT_STAGE_CHANNELS[s2]["channels"][:6]
        img1 = render_vlm4ts_image(full_data, st, ed, chans1, base_m, base_s)
        img2 = render_vlm4ts_image(full_data, st, ed, chans2, base_m, base_s)

        # Ultra-VETime forward scoring with feature extraction
        sc1, mask1, _, _, feat1 = teacher.score_window(img1, stage_id=s1, return_mask=True, return_feat=True)
        sc2, mask2, _, _, feat2 = teacher.score_window(img2, stage_id=s2, return_mask=True, return_feat=True)

        envelope_score = max(sc1, sc2)
        primary_feat = feat1 if sc1 >= sc2 else feat2
        primary_mask = mask1 if sc1 >= sc2 else mask2

        results[int(st)] = {
            "start_idx": int(st),
            "end_idx": int(ed),
            "x": torch.tensor(w_data, dtype=torch.float32),
            "y_dense": torch.tensor(w_labels, dtype=torch.float32),
            "y_flag": float(1.0 if np.any(w_labels >= 0.5) else 0.0),
            "anomaly_ratio": float(np.mean(w_labels)),
            "z_teacher": primary_feat.clone().detach().cpu(),  # [1024]
            "teacher_score": float(envelope_score),
            "teacher_dense_mask": torch.tensor(primary_mask, dtype=torch.float32),
            "top_stages": [int(s1), int(s2)],
            "stage_scores": {int(s1): float(sc1), int(s2): float(sc2)}
        }

    return results


def run_caching_pipeline(starts, full_data, full_labels, base_m, base_s,
                         teachers_by_device, split_name="train"):
    """Dispatches window caching across available initialized teachers."""
    devices = list(teachers_by_device.keys())
    print(f"\n--- Running Caching Pipeline for {split_name.upper()} ({len(starts)} windows across {len(devices)} device(s)) ---")

    if len(devices) >= 2:
        mid = len(starts) // 2
        chunk0 = starts[:mid]
        chunk1 = starts[mid:]
        d0, d1 = devices[0], devices[1]
        print(f"Multi-GPU Sharding: {d0} -> {len(chunk0)} windows, {d1} -> {len(chunk1)} windows")

        with ThreadPoolExecutor(max_workers=2) as executor:
            f0 = executor.submit(process_window_chunk, chunk0, full_data, full_labels,
                                 base_m, base_s, teachers_by_device[d0], d0)
            f1 = executor.submit(process_window_chunk, chunk1, full_data, full_labels,
                                 base_m, base_s, teachers_by_device[d1], d1)
            res0 = f0.result()
            res1 = f1.result()

        merged_results = {**res0, **res1}
    else:
        dev = devices[0]
        merged_results = process_window_chunk(starts, full_data, full_labels, base_m, base_s,
                                             teachers_by_device[dev], dev)

    cache_payload = {
        "metadata": {
            "dataset": "swat",
            "teacher_name": "T13: Ultra-VETime Multi-Stage",
            "backbone": teachers_by_device[devices[0]].model_id,
            "n_regimes": teachers_by_device[devices[0]].n_regimes,
            "window_size": 200,
            "total_windows": len(merged_results),
            "split": split_name,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "windows": merged_results
    }
    return cache_payload


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_train_cache) or "cache", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_eval_cache) or "cache", exist_ok=True)

    norm_data, att_data, labels, base_m, base_s = load_swat_data(args.data_dir)
    n_total = len(att_data)
    n_split = n_total // 2  # Standard 50% split (0..224959 = train, 224960..449919 = eval)

    train_data, train_labels = att_data[:n_split], labels[:n_split]
    eval_data, eval_labels = att_data[n_split:], labels[n_split:]

    # Initialize and calibrate teacher(s) across available device(s)
    n_gpus = torch.cuda.device_count()
    teachers_by_device = {}
    if n_gpus >= 2:
        for d in ["cuda:0", "cuda:1"]:
            print(f"[{d}] Initializing Ultra-VETime Teacher backbone ({args.model_id})...")
            t = UltraVETimeTeacher(model_id=args.model_id, device=d, n_regimes=args.n_regimes)
            t.calibrate_multi_regime_prototypes(norm_data, base_m, base_s, n_samples=30)
            teachers_by_device[d] = t
    else:
        d = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"[{d}] Initializing Ultra-VETime Teacher backbone ({args.model_id})...")
        t = UltraVETimeTeacher(model_id=args.model_id, device=d, n_regimes=args.n_regimes)
        t.calibrate_multi_regime_prototypes(norm_data, base_m, base_s, n_samples=30)
        teachers_by_device[d] = t

    # 1. Generate Training Cache
    if not args.eval_only:
        t0 = time.time()
        train_starts = sample_curriculum_training_windows(
            train_data, train_labels, norm_data,
            num_windows=args.num_train_windows, w_size=args.window_size
        )
        train_cache = run_caching_pipeline(
            train_starts, att_data, labels, base_m, base_s,
            teachers_by_device, split_name="train"
        )
        torch.save(train_cache, args.output_train_cache)
        size_mb = os.path.getsize(args.output_train_cache) / (1024 * 1024)
        print(f"\nSaved Training Cache to {args.output_train_cache} ({len(train_cache['windows'])} windows, {size_mb:.2f} MB, {time.time()-t0:.1f}s)")

    # 2. Generate Dedicated Head-to-Head Evaluation Cache
    if not args.train_only:
        t0 = time.time()
        eval_starts = sample_held_out_evaluation_windows(
            eval_data, eval_labels, global_offset=n_split,
            num_windows=args.num_eval_windows, w_size=args.window_size
        )
        eval_cache = run_caching_pipeline(
            eval_starts, att_data, labels, base_m, base_s,
            teachers_by_device, split_name="eval"
        )
        torch.save(eval_cache, args.output_eval_cache)
        size_mb = os.path.getsize(args.output_eval_cache) / (1024 * 1024)
        print(f"\nSaved Evaluation Cache to {args.output_eval_cache} ({len(eval_cache['windows'])} windows, {size_mb:.2f} MB, {time.time()-t0:.1f}s)")

    print("\n" + "=" * 80)
    print(" SWaT ULTRA-VETIME CACHING COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
