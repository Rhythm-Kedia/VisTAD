"""Continuous Streaming Deployment & Evaluation on GECCO Drinking Water Stream.

Evaluates the Subsystem-Coupled Causal TCN Student in real-time continuous deployment:
- Primary Held-Out Stream: Partition 2 (Timestamps 70,000 to 139,565, 69,566 continuous seconds, 22 unseen events)
- Legacy VisTAD Paper Slice: Timestamps 111,652 to 139,565 (27,914 seconds, Event 51 only)
- Complete Stream: Timestamps 0 to 139,565 (139,566 seconds, 51 contamination events)
- Strictly Held-Out Cache: 1,500 windows from cache/gecco_eval_head_to_head_cache.pt
- Continuous Sliding Stride: 10 steps (every 10 seconds, sliding 200-step causal window)
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
from vlm4ts.gecco_renderer import GECCO_SENSORS


def load_gecco_stream(data_path="data/gecco/gecco_water.csv"):
    """Loads raw GECCO stream data and ground truth labels."""
    if not os.path.exists(data_path):
        candidates = [
            "data/gecco/gecco_water.csv",
            "data/gecco_water.csv",
            "../data/gecco/gecco_water.csv"
        ]
        for c in candidates:
            if os.path.exists(c):
                data_path = c
                break

    df = pd.read_csv(data_path)
    labels = df["attack"].values.astype(np.float32)
    data = df[GECCO_SENSORS].values.astype(np.float32)
    return data, labels, df


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
            "x": torch.from_numpy(self.data[st:ed]).float()  # [200, 9]
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
            x = batch["x"].to(device)  # [B, 200, 9]
            out = model(x)
            pred_dense = out["pred_dense"].cpu().numpy()  # [B, 200]

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
    prauc = average_precision_score(y_eval, s_eval)

    precisions, recalls, thresholds = precision_recall_curve(y_eval, s_eval)
    f1_scores = 2 * (precisions * recalls) / np.clip(precisions + recalls, 1e-12, None)
    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx]
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    y_pred_bin = (s_eval >= best_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_eval, y_pred_bin).ravel()
    fpr = fp / (fp + tn + 1e-12)
    tpr = tp / (tp + fn + 1e-12)
    specificity = tn / (tn + fp + 1e-12)

    return {
        "auroc": float(auroc),
        "prauc": float(prauc),
        "best_f1": float(best_f1),
        "best_threshold": float(best_thresh),
        "precision": float(precisions[best_idx]),
        "recall": float(tpr),
        "fpr": float(fpr),
        "specificity": float(specificity),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "total_steps": int(len(y_eval)),
        "anomaly_steps": int(sum(y_eval)),
        "nominal_steps": int(len(y_eval) - sum(y_eval)),
        "duration_sec": float(duration),
        "throughput_fps": float(fps),
        "latency_per_window_ms": float(latency_ms),
        "total_windows": len(ds),
        "final_probs": final_probs,
        "valid_mask": valid_mask
    }


def evaluate_held_out_cache(model, cache_path, base_m, base_s, device):
    """Evaluates the model head-to-head on the cached evaluation windows."""
    print(f"Loading held-out cache from {cache_path}...")
    c = torch.load(cache_path, weights_only=False)
    windows = c["windows"]
    keys = sorted(list(windows.keys()))

    student_scores = []
    teacher_scores = []
    y_flags = []

    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for k in keys:
            w = windows[k]
            raw_x = torch.as_tensor(w["x"], dtype=torch.float32)
            x = (raw_x - base_m) / base_s
            out = model(x.unsqueeze(0).to(device))
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

    # Teacher metrics
    t_auc = roc_auc_score(y_true, s_tea)
    t_pr = average_precision_score(y_true, s_tea)
    p_t, r_t, t_t = precision_recall_curve(y_true, s_tea)
    f1_t = 2 * p_t * r_t / np.clip(p_t + r_t, 1e-12, None)
    best_it = np.argmax(f1_t)
    tea_f1 = f1_t[best_it]
    tea_th = t_t[best_it] if best_it < len(t_t) else 0.5
    tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_true, (s_tea >= tea_th).astype(int)).ravel()

    return {
        "student": {
            "auroc": float(st_auc),
            "prauc": float(st_pr),
            "best_f1": float(st_f1),
            "best_threshold": float(st_th),
            "precision": float(p[best_i]),
            "recall": float(r[best_i]),
            "fpr": float(fp / (fp + tn + 1e-12)),
            "latency_ms": float((dur / len(keys)) * 1000.0)
        },
        "teacher": {
            "auroc": float(t_auc),
            "prauc": float(t_pr),
            "best_f1": float(tea_f1),
            "best_threshold": float(tea_th),
            "precision": float(p_t[best_it]),
            "recall": float(r_t[best_it]),
            "fpr": float(fp_t / (fp_t + tn_t + 1e-12))
        }
    }


def audit_events(df, preds, threshold, st_offset=70000):
    """Audits per-event detection performance."""
    df_copy = df.copy()
    df_copy["event_id"] = (df_copy["attack"] != df_copy["attack"].shift()).cumsum()
    anom_events = df_copy[df_copy["attack"] == 1].groupby("event_id")

    events = []
    for gid, grp in anom_events:
        st = grp.index[0]
        en = grp.index[-1]
        l = len(grp)
        if st >= st_offset:
            events.append((st, en, l))

    pred_bin = (preds >= threshold).astype(int)
    results = []
    caught_events = 0
    total_caught_steps = 0
    total_steps = sum(l for _, _, l in events)

    for i, (st, en, l) in enumerate(events):
        st_rel = st - st_offset
        en_rel = en - st_offset
        ev_p = pred_bin[st_rel:en_rel+1]
        ev_s = preds[st_rel:en_rel+1]
        caught = int(ev_p.sum())
        total_caught_steps += caught
        if caught > 0:
            caught_events += 1
        results.append({
            "event_index": i + 1,
            "start_timestep": int(st),
            "end_timestep": int(en),
            "duration_steps": int(l),
            "caught_steps": int(caught),
            "recall": float(caught / l),
            "max_score": float(np.max(ev_s)),
            "detected": bool(caught > 0)
        })

    return {
        "num_events": len(events),
        "caught_events": caught_events,
        "event_detection_rate": float(caught_events / max(1, len(events))),
        "total_anom_steps": total_steps,
        "total_caught_steps": total_caught_steps,
        "step_recall": float(total_caught_steps / max(1, total_steps)),
        "events": results
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Continuous Streaming on GECCO")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/gecco_coupled_tcn_best.pt")
    parser.add_argument("--data_path", type=str, default="data/gecco/gecco_water.csv")
    parser.add_argument("--eval_cache", type=str, default="cache/gecco_eval_head_to_head_cache.pt")
    parser.add_argument("--save_json", type=str, default="results/gecco_gecco_water_coupled_tcn_eval.json")
    parser.add_argument("--window_size", type=int, default=200)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Selected Compute Device: {device}")

    # Load Model
    print(f"Loading checkpoint from: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("model_config", {})

    base_m = cfg.get("base_mean")
    base_s = cfg.get("base_std")

    model = build_student(
        "gecco_coupled_tcn",
        hidden_dim=cfg.get("hidden_dim", 48),
        base_m=torch.tensor(base_m) if base_m is not None else None,
        base_s=torch.tensor(base_s) if base_s is not None else None
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    full_data, full_labels, df_raw = load_gecco_stream(args.data_path)

    if base_m is not None and base_s is not None:
        print("Standardizing continuous stream using checkpoint baseline parameters...")
        norm_full_data = (full_data - base_m) / (base_s + 1e-6)
    else:
        norm_full_data = full_data

    # 1. Primary Held-Out Evaluation: Partition 2 (Timestamps 70,000 to 139,566)
    p2_data = norm_full_data[70000:]
    p2_labels = full_labels[70000:]
    print(f"\n--- Running Continuous Streaming Evaluation: Partition 2 (Timestamps 70,000 to {len(norm_full_data)}) ---")
    print(f"Stream: {len(p2_data):,} seconds ({p2_labels.sum():.0f} anomalous steps across 22 unseen events, stride={args.stride})")

    res_p2 = run_continuous_streaming_eval(
        model, p2_data, p2_labels,
        window_size=args.window_size, stride=args.stride,
        batch_size=args.batch_size, device=device
    )

    print("\n" + "=" * 75)
    print("GECCO CONTINUOUS STREAMING RESULTS: HELD-OUT PARTITION 2 (70k -> 140k)")
    print("=" * 75)
    print(f"Continuous Point AUROC: {res_p2['auroc']:.4f}")
    print(f"Continuous Point PR-AUC:{res_p2['prauc']:.4f}")
    print(f"Continuous Point Best F1:{res_p2['best_f1']:.4f} (at threshold {res_p2['best_threshold']:.4f})")
    print(f"Continuous Precision:   {res_p2['precision']*100:.2f}%")
    print(f"Continuous Recall:      {res_p2['recall']*100:.2f}% ({res_p2['tp']:,} / {res_p2['anomaly_steps']:,} anomalous seconds caught)")
    print(f"Continuous Specificity: {res_p2['specificity']*100:.2f}% (FPR = {res_p2['fpr']*100:.2f}%)")
    print(f"False Alarms:           {res_p2['fp']:,} false alarm seconds out of {res_p2['nominal_steps']:,} nominal seconds")
    print(f"Throughput:             {res_p2['throughput_fps']:,.1f} windows/sec ({res_p2['latency_per_window_ms']:.3f} ms/window)")
    print("=" * 75)

    # Audit events in Partition 2
    event_audit_p2 = audit_events(df_raw, res_p2["final_probs"], res_p2["best_threshold"], st_offset=70000)
    print(f"\nPartition 2 Per-Event Audit ({event_audit_p2['num_events']} unseen events):")
    print(f"Events Caught: {event_audit_p2['caught_events']} / {event_audit_p2['num_events']} ({event_audit_p2['event_detection_rate']*100:.1f}%)")
    print(f"Steps Caught:  {event_audit_p2['total_caught_steps']} / {event_audit_p2['total_anom_steps']} ({event_audit_p2['step_recall']*100:.1f}%)")

    # 2. Legacy VisTAD Paper Slice Evaluation: Timestamps 111,652 to 139,566
    p_legacy_data = norm_full_data[111652:]
    p_legacy_labels = full_labels[111652:]
    print(f"\n--- Running Continuous Streaming Evaluation: Legacy VisTAD Paper Slice (111,652 to {len(norm_full_data)}) ---")

    res_leg = run_continuous_streaming_eval(
        model, p_legacy_data, p_legacy_labels,
        window_size=args.window_size, stride=args.stride,
        batch_size=args.batch_size, device=device
    )

    print("\n" + "=" * 75)
    print("GECCO CONTINUOUS STREAMING RESULTS: LEGACY VISTAD PAPER SLICE (111.6k -> 140k)")
    print("=" * 75)
    print(f"VisTAD Paper Baseline:  AUROC = 0.4904, Best F1 = 0.0096 (TCN Collapsed)")
    print(f"Coupled TCN AUROC:      {res_leg['auroc']:.4f} (Delta: {(res_leg['auroc'] - 0.4904)*100:+.2f}%)")
    print(f"Coupled TCN PR-AUC:     {res_leg['prauc']:.4f}")
    print(f"Coupled TCN Best F1:    {res_leg['best_f1']:.4f} (Delta: {(res_leg['best_f1'] - 0.0096)*100:+.2f}%)")
    print(f"Coupled TCN Recall:     {res_leg['recall']*100:.2f}% ({res_leg['tp']:,} / {res_leg['anomaly_steps']:,} anomalous seconds caught)")
    print(f"Coupled TCN FPR:        {res_leg['fpr']*100:.2f}% (Specificity: {res_leg['specificity']*100:.2f}%)")
    print("=" * 75)

    # 3. Complete Stream Evaluation (0 to 139,565)
    print(f"\n--- Running Continuous Streaming Evaluation: Complete Stream (0 to {len(norm_full_data)}) ---")
    res_all = run_continuous_streaming_eval(
        model, norm_full_data, full_labels,
        window_size=args.window_size, stride=args.stride,
        batch_size=args.batch_size, device=device
    )
    print("\n" + "=" * 75)
    print("GECCO CONTINUOUS STREAMING RESULTS: COMPLETE STREAM (0 -> 140k)")
    print("=" * 75)
    print(f"Complete Stream AUROC:  {res_all['auroc']:.4f}")
    print(f"Complete Stream PR-AUC: {res_all['prauc']:.4f}")
    print(f"Complete Stream Best F1:{res_all['best_f1']:.4f}")
    print(f"Complete Stream Recall: {res_all['recall']*100:.2f}%")
    print(f"Complete Stream FPR:    {res_all['fpr']*100:.2f}%")
    print("=" * 75)

    # 4. Strictly Held-Out Cache Evaluation
    cache_res = None
    if args.eval_cache and os.path.exists(args.eval_cache):
        b_m_t = torch.tensor(base_m) if base_m is not None else torch.zeros(9)
        b_s_t = torch.tensor(base_s) if base_s is not None else torch.ones(9)
        cache_res = evaluate_held_out_cache(model, args.eval_cache, b_m_t, b_s_t, device)

    # Save Results JSON
    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
        # Helper to clean out non-serializable elements
        def clean_dict(d):
            return {k: v for k, v in d.items() if k not in ["final_probs", "valid_mask"]}

        out_json = {
            "dataset": "gecco",
            "entity": "gecco_water",
            "architecture": "gecco_coupled_tcn",
            "parameters": sum(p.numel() for p in model.parameters()),
            "num_channels": 9,
            "continuous_streaming_held_out_partition_2": clean_dict(res_p2),
            "event_audit_partition_2": event_audit_p2,
            "continuous_streaming_legacy_vistad_slice": clean_dict(res_leg),
            "continuous_streaming_complete_stream": clean_dict(res_all),
            "baseline_comparison": {
                "vistad_paper_baseline": {
                    "point_auroc": 0.4904,
                    "point_pr_auc": 0.0035,
                    "point_best_f1": 0.0096
                },
                "delta_legacy_slice": {
                    "auroc_gain": f"+{res_leg['auroc'] - 0.4904:.4f}",
                    "best_f1_boost": f"{res_leg['best_f1'] / 0.0096:.1f}x"
                }
            }
        }
        if cache_res:
            out_json["held_out_cache_metrics"] = cache_res

        with open(args.save_json, "w") as f:
            json.dump(out_json, f, indent=2, default=lambda o: int(o) if isinstance(o, (np.integer,)) else (float(o) if isinstance(o, (np.floating,)) else str(o)))
        print(f"\nOfficial streaming evaluation metrics exported to: {args.save_json}")


if __name__ == "__main__":
    main()
