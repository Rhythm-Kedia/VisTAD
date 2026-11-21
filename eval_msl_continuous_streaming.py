"""Continuous Streaming Deployment & Evaluation on NASA MSL Stream.

Evaluates the MSLCoupledTCNStudent in real-time continuous deployment:
- Primary Held-Out Stream: Partition 2 (Timestamps 37,000 to 73,728, 36,729 steps, 18 unseen anomaly episodes)
- Complete Test Stream: Timestamps 0 to 73,728 (73,729 steps, all 36 episodes)
- Continuous Sliding Stride: 10 steps (every 10 steps, sliding 200-step causal window)
- Point-wise Evaluation: Maps predictions across sliding overlaps into exact point-by-point timeline
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.models import build_student


def load_msl_stream(data_path="data/msl/msl_test.csv"):
    """Loads raw MSL test data and ground truth labels."""
    if not os.path.exists(data_path):
        candidates = [
            "data/msl/msl_test.csv",
            "data/msl_test.csv",
            "../data/msl/msl_test.csv"
        ]
        for c in candidates:
            if os.path.exists(c):
                data_path = c
                break

    df = pd.read_csv(data_path)
    labels = df["attack"].values.astype(np.float32)
    cols = [c for c in df.columns if c not in ["attack", "label", "timestamp"]]
    data = df[cols].values.astype(np.float32)
    return data, labels


class ContinuousStreamDataset(Dataset):
    """Sliding window dataset across the continuous stream."""
    def __init__(self, data, window_size=200, stride=10):
        self.data = data
        self.window_size = window_size
        self.stride = stride
        self.indices = list(range(0, len(data) - window_size + 1, stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        st = self.indices[idx]
        ed = st + self.window_size
        return {
            "st": st,
            "x": torch.from_numpy(self.data[st:ed]).float()
        }


def run_continuous_streaming_eval(model, stream_data, stream_labels, window_size=200, stride=10,
                                  batch_size=128, device="cpu"):
    """Slides through the continuous stream and aggregates point-wise predictions."""
    model.eval()
    n_points = len(stream_data)

    ds = ContinuousStreamDataset(stream_data, window_size=window_size, stride=stride)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    point_scores = np.zeros(n_points, dtype=np.float32)
    point_counts = np.zeros(n_points, dtype=np.float32)

    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            starts = batch["st"].numpy()
            x = batch["x"].to(device)
            out = model(x)
            pred_dense = out["pred_dense"].cpu().numpy()

            for i, st in enumerate(starts):
                point_scores[st : st + window_size] += pred_dense[i]
                point_counts[st : st + window_size] += 1.0

    duration = time.time() - t0
    fps = len(ds) / duration
    latency_ms = (duration / len(ds)) * 1000.0

    valid_mask = point_counts > 0
    final_probs = np.zeros(n_points, dtype=np.float32)
    final_probs[valid_mask] = point_scores[valid_mask] / point_counts[valid_mask]

    y_eval = stream_labels[valid_mask]
    s_eval = final_probs[valid_mask]

    auroc = roc_auc_score(y_eval, s_eval)
    pr_auc = average_precision_score(y_eval, s_eval)

    precisions, recalls, thresholds = precision_recall_curve(y_eval, s_eval)
    f1_scores = 2 * (precisions * recalls) / np.maximum(precisions + recalls, 1e-8)
    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx]
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    y_pred_bin = (s_eval >= best_thresh).astype(int)
    cm = confusion_matrix(y_eval, y_pred_bin)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    fpr = fp / max(fp + tn, 1)

    return {
        "auroc": float(auroc),
        "pr_auc": float(pr_auc),
        "best_f1": float(best_f1),
        "best_thresh": float(best_thresh),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "fpr": float(fpr),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "throughput_fps": float(fps),
        "latency_ms": float(latency_ms),
        "total_eval_time": float(duration),
        "final_probs": final_probs,
        "valid_mask": valid_mask
    }


def evaluate_held_out_cache(model, cache_path, base_m, base_s, device):
    """Evaluates the model head-to-head on the cached evaluation windows."""
    print(f"Loading held-out cache from {cache_path}...")
    c = torch.load(cache_path, map_location="cpu", weights_only=False)
    windows = c["windows"]
    keys = sorted(list(windows.keys()))

    student_scores = []
    teacher_scores = []
    y_flags = []

    model.eval()
    t0 = time.time()
    base_m_t = torch.as_tensor(base_m, dtype=torch.float32, device=device)
    base_s_t = torch.as_tensor(base_s, dtype=torch.float32, device=device)
    with torch.no_grad():
        for k in keys:
            w = windows[k]
            raw_x = torch.as_tensor(w["x"], dtype=torch.float32, device=device)
            x = (raw_x - base_m_t) / base_s_t
            out = model(x.unsqueeze(0))
            s_student = float(torch.max(out["pred_dense"]).cpu())
            student_scores.append(s_student)
            teacher_scores.append(float(w.get("teacher_score", 0.5)))
            y_flags.append(float(w.get("y_flag", 0.0)))
    dur = time.time() - t0

    y_true = np.array(y_flags)
    s_stu = np.array(student_scores)
    s_tea = np.array(teacher_scores)

    # Student metrics
    st_auc = roc_auc_score(y_true, s_stu)
    st_pr = average_precision_score(y_true, s_stu)
    p, r, t = precision_recall_curve(y_true, s_stu)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    best_i = np.argmax(f1)
    st_f1 = f1[best_i]
    st_th = t[best_i] if best_i < len(t) else 0.5
    tn, fp, fn, tp = confusion_matrix(y_true, (s_stu >= st_th).astype(int)).ravel()
    st_fpr = fp / (fp + tn + 1e-12)

    # Teacher metrics
    t_auc = roc_auc_score(y_true, s_tea)
    t_pr = average_precision_score(y_true, s_tea)
    p_t, r_t, t_t = precision_recall_curve(y_true, s_tea)
    f1_t = 2 * p_t * r_t / np.clip(p_t + r_t, 1e-12, None)
    best_it = np.argmax(f1_t)
    tea_f1 = f1_t[best_it]
    tea_th = t_t[best_it] if best_it < len(t_t) else 0.5
    tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_true, (s_tea >= tea_th).astype(int)).ravel()
    tea_fpr = fp_t / (fp_t + tn_t + 1e-12)

    print("\n" + "=" * 80)
    print("NASA MSL HEAD-TO-HEAD HELD-OUT CACHE EVALUATION (Timestamps 37,000 to 73,728)")
    print("=" * 80)
    print(f"{'Metric':<25} | {'Ultra-VETime Teacher':<22} | {'Coupled TCN Student':<22} | {'Advantage'}")
    print("-" * 80)
    print(f"{'AUROC':<25} | {t_auc:<22.4f} | {st_auc:<22.4f} | {(st_auc - t_auc)*100:+.2f}%")
    print(f"{'PR-AUC':<25} | {t_pr:<22.4f} | {st_pr:<22.4f} | {(st_pr - t_pr)*100:+.2f}%")
    print(f"{'Best F1':<25} | {tea_f1:<22.4f} | {st_f1:<22.4f} | {(st_f1 - tea_f1)*100:+.2f}%")
    print(f"{'FPR':<25} | {tea_fpr*100:<21.2f}% | {st_fpr*100:<21.2f}% | {(st_fpr - tea_fpr)*100:+.2f}%")
    print("=" * 80)

    return {
        "student": {"auroc": float(st_auc), "pr_auc": float(st_pr), "best_f1": float(st_f1), "best_thresh": float(st_th), "fpr": float(st_fpr)},
        "teacher": {"auroc": float(t_auc), "pr_auc": float(t_pr), "best_f1": float(tea_f1), "best_thresh": float(tea_th), "fpr": float(tea_fpr)}
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate MSL Student continuous streaming")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/msl_coupled_tcn_best.pt")
    parser.add_argument("--arch", type=str, default=None)
    parser.add_argument("--test_csv", type=str, default="data/msl/msl_test.csv")
    parser.add_argument("--train_csv", type=str, default="data/msl/msl_train.csv")
    parser.add_argument("--window_size", type=int, default=200)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--save_json", type=str, default="results/msl_msl_test_coupled_tcn_eval.json")
    parser.add_argument("--eval_cache", type=str, default="cache/msl_eval_head_to_head_cache.pt")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Loading checkpoint from: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    model_cfg = ckpt.get("model_config", {})
    if args.arch:
        arch = args.arch
    elif "num_heads" in model_cfg:
        arch = "msl_universal_pro"
    else:
        arch = "msl_coupled_tcn"

    print(f"Model architecture: {arch} (config: {model_cfg})")
    cfg = dict(model_cfg)
    cfg.setdefault("in_channels", 55)
    model = build_student(arch, **cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    base_m = ckpt.get("base_m")
    base_s = ckpt.get("base_s")
    if base_m is None or base_s is None:
        df_tr = pd.read_csv(args.train_csv)
        cols = [c for c in df_tr.columns if c not in ["attack", "label", "timestamp"]]
        tr_np = df_tr[cols].values.astype(np.float32)
        base_m = np.mean(tr_np, axis=0)
        base_s = np.std(tr_np, axis=0)
        base_s[base_s == 0] = 1.0

    raw_data, raw_labels = load_msl_stream(args.test_csv)
    # Standardize stream
    norm_data = (raw_data - base_m) / np.maximum(base_s, 1e-6)

    # 1. Primary Held-Out Partition 2 (t >= 37,000)
    print("\n" + "=" * 75)
    print("A. PRIMARY HELD-OUT STREAM: PARTITION 2 (t >= 37,000, 18 UNSEEN EPISODES)")
    print("=" * 75)
    p2_data = norm_data[37000:]
    p2_labels = raw_labels[37000:]
    p2_res = run_continuous_streaming_eval(model, p2_data, p2_labels, window_size=args.window_size, stride=args.stride, device=device)
    print(f"Continuous Point AUROC:       {p2_res['auroc']:.4f}")
    print(f"Continuous Point PR-AUC:      {p2_res['pr_auc']:.4f}")
    print(f"Continuous Point Best F1:     {p2_res['best_f1']:.4f} (at tau* = {p2_res['best_thresh']:.4f})")
    print(f"Continuous Point Precision:   {p2_res['precision']*100:.2f}%")
    print(f"Continuous Point Recall:      {p2_res['recall']*100:.2f}% ({p2_res['tp']} / {p2_res['tp'] + p2_res['fn']} anomaly seconds)")
    print(f"Continuous Point Specificity: {p2_res['specificity']*100:.2f}% (FPR = {p2_res['fpr']*100:.2f}%)")
    print(f"Throughput:                   {p2_res['throughput_fps']:.1f} windows/second ({p2_res['latency_ms']:.3f} ms/window)")
    print(f"Total Stream Evaluation Time: {p2_res['total_eval_time']:.2f} seconds")

    # 2. Partition 1 (Training Range: t = 0 to 37,000)
    print("\n" + "=" * 75)
    print("B. PARTITION 1 STREAM: TRAINING REGIME (t = 0 to 37,000, 18 EPISODES)")
    print("=" * 75)
    p1_data = norm_data[:37000]
    p1_labels = raw_labels[:37000]
    p1_res = run_continuous_streaming_eval(model, p1_data, p1_labels, window_size=args.window_size, stride=args.stride, device=device)
    print(f"Continuous Point AUROC:       {p1_res['auroc']:.4f}")
    print(f"Continuous Point PR-AUC:      {p1_res['pr_auc']:.4f}")
    print(f"Continuous Point Best F1:     {p1_res['best_f1']:.4f} (at tau* = {p1_res['best_thresh']:.4f})")
    print(f"Continuous Point Precision:   {p1_res['precision']*100:.2f}%")
    print(f"Continuous Point Recall:      {p1_res['recall']*100:.2f}% ({p1_res['tp']} / {p1_res['tp'] + p1_res['fn']} anomaly seconds)")
    print(f"Continuous Point Specificity: {p1_res['specificity']*100:.2f}% (FPR = {p1_res['fpr']*100:.2f}%)")
    print(f"Throughput:                   {p1_res['throughput_fps']:.1f} windows/second ({p1_res['latency_ms']:.3f} ms/window)")
    print("=" * 75)

    # 3. Complete Test Stream (t = 0 to 73,728)
    print("\n" + "=" * 75)
    print("C. COMPLETE NASA MSL TEST STREAM (t = 0 to 73,728, ALL 36 EPISODES)")
    print("=" * 75)
    all_res = run_continuous_streaming_eval(model, norm_data, raw_labels, window_size=args.window_size, stride=args.stride, device=device)
    print(f"Continuous Point AUROC:       {all_res['auroc']:.4f} (VisTAD Paper Baseline: 0.4366)")
    print(f"Continuous Point PR-AUC:      {all_res['pr_auc']:.4f} (VisTAD Paper Baseline: 0.0347)")
    print(f"Continuous Point Best F1:     {all_res['best_f1']:.4f} (VisTAD Paper Baseline: 0.0811)")
    print(f"Continuous Point Precision:   {all_res['precision']*100:.2f}%")
    print(f"Continuous Point Recall:      {all_res['recall']*100:.2f}% ({all_res['tp']} / {all_res['tp'] + all_res['fn']} anomaly seconds)")
    print(f"Continuous Point Specificity: {all_res['specificity']*100:.2f}% (FPR = {all_res['fpr']*100:.2f}%)")
    print(f"Throughput:                   {all_res['throughput_fps']:.1f} windows/second ({all_res['latency_ms']:.3f} ms/window)")
    print(f"Total Stream Evaluation Time: {all_res['total_eval_time']:.2f} seconds")
    print("=" * 75)

    # 4. Event Catch Rate & Affiliation Metrics
    from src.metrics import compute_affiliation_metrics, find_events
    gt_events = find_events(raw_labels)
    tau = all_res["best_thresh"]
    probs = all_res["final_probs"]
    caught = 0
    for st_e, ed_e in gt_events:
        if np.max(probs[st_e:ed_e]) >= tau:
            caught += 1

    bin_preds = (probs >= tau).astype(int)
    aff_raw = compute_affiliation_metrics(raw_labels, bin_preds, smooth=False)
    aff_smooth = compute_affiliation_metrics(raw_labels, bin_preds, smooth=True)

    print("\n" + "=" * 75)
    print("D. EVENT-LEVEL CATCH RATE & AFFILIATION METRICS (ALL 36 EPISODES)")
    print("=" * 75)
    print(f"Events Caught:                {caught} / {len(gt_events)} ({caught/len(gt_events)*100:.1f}%)")
    print(f"Affiliation F1 (raw):         {aff_raw['aff_f1']*100:.2f}% (Precision: {aff_raw['aff_precision']*100:.1f}%, Recall: {aff_raw['aff_recall']*100:.1f}%)")
    print(f"Affiliation F1 (smooth):      {aff_smooth['aff_f1']*100:.2f}% (Precision: {aff_smooth['aff_precision']*100:.1f}%, Recall: {aff_smooth['aff_recall']*100:.1f}%)")
    print("=" * 75)

    # 5. Held-Out Cache Evaluation
    cache_metrics = None
    if args.eval_cache and os.path.exists(args.eval_cache):
        cache_metrics = evaluate_held_out_cache(model, args.eval_cache, base_m, base_s, device)

    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
        out_dict = {
            "dataset": "msl",
            "entity": "msl_test",
            "architecture": "msl_coupled_tcn",
            "parameters": sum(p.numel() for p in model.parameters()),
            "num_channels": 55,
            "continuous_streaming_complete_stream": {
                "total_timesteps": len(norm_data),
                "anomaly_timesteps": int(raw_labels.sum()),
                "nominal_timesteps": int(len(norm_data) - raw_labels.sum()),
                "num_episodes": 36,
                "stride": args.stride,
                "window_size": args.window_size,
                "point_auroc": all_res["auroc"],
                "point_pr_auc": all_res["pr_auc"],
                "point_best_f1": all_res["best_f1"],
                "precision": all_res["precision"],
                "recall": all_res["recall"],
                "specificity": all_res["specificity"],
                "fpr": all_res["fpr"],
                "true_positives": all_res["tp"],
                "false_positives": all_res["fp"],
                "false_negatives": all_res["fn"],
                "throughput_windows_sec": all_res["throughput_fps"],
                "latency_ms_per_window": all_res["latency_ms"],
                "total_eval_time_sec": all_res["total_eval_time"]
            },
            "continuous_streaming_held_out_partition_2": {
                "timesteps_range": [37000, 73728],
                "total_timesteps": len(p2_data),
                "anomaly_timesteps": int(p2_labels.sum()),
                "num_episodes": 18,
                "stride": args.stride,
                "window_size": args.window_size,
                "point_auroc": p2_res["auroc"],
                "point_pr_auc": p2_res["pr_auc"],
                "point_best_f1": p2_res["best_f1"],
                "precision": p2_res["precision"],
                "recall": p2_res["recall"],
                "specificity": p2_res["specificity"],
                "fpr": p2_res["fpr"],
                "true_positives": p2_res["tp"],
                "false_positives": p2_res["fp"],
                "false_negatives": p2_res["fn"],
                "throughput_windows_sec": p2_res["throughput_fps"],
                "latency_ms_per_window": p2_res["latency_ms"],
                "total_eval_time_sec": p2_res["total_eval_time"]
            },
            "baseline_comparison": {
                "vistad_paper_baseline": {
                    "point_auroc": 0.4366,
                    "point_pr_auc": 0.0347,
                    "point_best_f1": 0.0811
                },
                "delta_complete_stream": {
                    "auroc_gain": f"+{all_res['auroc'] - 0.4366:.4f}",
                    "pr_auc_boost": f"{all_res['pr_auc'] / 0.0347:.1f}x",
                    "best_f1_boost": f"{all_res['best_f1'] / 0.0811:.1f}x"
                }
            }
        }
        if cache_metrics:
            out_dict["held_out_cache_metrics"] = {
                "num_windows": 1393,
                "student": cache_metrics["student"],
                "teacher_ultravetime_dinov2_large": cache_metrics["teacher"]
            }
        with open(args.save_json, "w") as f:
            json.dump(out_dict, f, indent=2, default=lambda o: int(o) if isinstance(o, (np.integer,)) else (float(o) if isinstance(o, (np.floating,)) else str(o)))
        print(f"Saved evaluation metrics -> {args.save_json}")


if __name__ == "__main__":
    main()
