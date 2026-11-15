"""Continuous Streaming Evaluation across all 28 Server Machine Dataset (SMD) Entities.

Evaluates the 38-Channel Multi-Subsystem Causal TCN Foundation Student:
- Input: All 38 telemetry metrics across 4 server subsystems (CPU, Memory, Network, Disk)
- Continuous Stream: 708,420 total timesteps across 28 server entities
- Evaluation Slices:
  1. Strictly Held-Out Test Slice (chronological last 40% across all 28 machines: 283,381 steps, 17,738 anomalies)
  2. Complete Continuous Stream (all 28 machines: 708,420 steps, 29,444 anomalies, 327 events)
- Metrics:
  - Point-wise AUROC, PR-AUC, Best F1, Precision, Recall, FPR
  - Continuous Event-level Recall (contiguous failure episodes detected)
  - Latency (ms per window, FPS throughput)
- Outputs: Per-machine breakdown table, overall pooled totals, head-to-head comparison against baseline.
"""

import os
import sys
import time
import glob
import json
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.models.smd_coupled_tcn import SMDSubsystemCoupledTCN


class ContinuousStreamDataset(Dataset):
    def __init__(self, data: np.ndarray, window_size: int = 200, stride: int = 10):
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
            "x": torch.from_numpy(self.data[st:ed]).float().T  # [38, 200]
        }


def compute_event_recall(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, float]:
    """Computes event-level recall across contiguous anomaly episodes."""
    diff = np.diff(np.pad(y_true, (1, 1), mode="constant"))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    total_events = len(starts)
    if total_events == 0:
        return 0, 0, 1.0

    detected_events = 0
    for s, e in zip(starts, ends):
        if np.any(y_pred[s:e] == 1):
            detected_events += 1

    return detected_events, total_events, detected_events / total_events


def evaluate_single_machine(model, raw_data, labels, train_ratio=0.50, val_ratio=0.10,
                            window_size=200, stride=10, batch_size=128, device="cpu"):
    """Evaluates continuous streaming on a single SMD machine and returns continuous score timeline."""
    n = len(raw_data)
    a = int(n * train_ratio)
    b = int(n * (train_ratio + val_ratio))

    # Standardize telemetry using statistics strictly from training partition (0..a)
    base_m = np.mean(raw_data[:a], axis=0)
    base_s = np.maximum(np.std(raw_data[:a], axis=0), 0.10)
    norm_data = (raw_data - base_m) / base_s

    ds = ContinuousStreamDataset(norm_data, window_size=window_size, stride=stride)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    point_scores = np.zeros(n, dtype=np.float32)
    point_counts = np.zeros(n, dtype=np.float32)

    total_infer_time = 0.0
    total_windows = len(ds)

    model.eval()
    with torch.no_grad():
        for batch in loader:
            starts = batch["st"].numpy()
            x = batch["x"].to(device)  # [B, 38, 200]

            t0 = time.time()
            out = model(x)
            total_infer_time += time.time() - t0

            pred_dense = out["pred_dense"].cpu().numpy()  # [B, 200]
            for i, st in enumerate(starts):
                point_scores[st : st + window_size] += pred_dense[i]
                point_counts[st : st + window_size] += 1.0

    # Average overlapping sliding window estimates
    valid_mask = point_counts > 0
    point_scores[valid_mask] /= point_counts[valid_mask]

    # For edge steps not covered by window_size (first few steps if any), fill with first valid score
    if not valid_mask[0]:
        first_valid = np.where(valid_mask)[0][0]
        point_scores[:first_valid] = point_scores[first_valid]

    return point_scores, labels, a, b, total_infer_time, total_windows


def compute_metrics(y_true, y_scores, fixed_thresh=None):
    """Computes comprehensive binary classification metrics."""
    if len(np.unique(y_true)) < 2:
        return {
            "auroc": 0.5, "prauc": float(np.mean(y_true)), "best_f1": 0.0,
            "best_threshold": 0.5, "precision": 0.0, "recall": 0.0, "fpr": 0.0,
            "detected_events": 0, "total_events": 0, "event_recall": 1.0
        }

    auroc = roc_auc_score(y_true, y_scores)
    prauc = average_precision_score(y_true, y_scores)

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * (precisions * recalls) / np.clip(precisions + recalls, 1e-12, None)

    if fixed_thresh is not None:
        thresh = fixed_thresh
        best_f1 = float(2 * (precisions[thresholds >= thresh][0] if np.any(thresholds >= thresh) else 0) * 
                        (recalls[thresholds >= thresh][0] if np.any(thresholds >= thresh) else 0)) # approx
    else:
        best_idx = np.argmax(f1_scores)
        best_f1 = float(f1_scores[best_idx])
        thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

    y_pred = (y_scores >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = float(fp / (fp + tn + 1e-12))
    recall = float(tp / (tp + fn + 1e-12))
    precision = float(tp / (tp + fp + 1e-12))

    det_ev, tot_ev, ev_rec = compute_event_recall(y_true, y_pred)

    return {
        "auroc": float(auroc),
        "prauc": float(prauc),
        "best_f1": best_f1,
        "best_threshold": thresh,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "detected_events": det_ev,
        "total_events": tot_ev,
        "event_recall": float(ev_rec)
    }


def main():
    parser = argparse.ArgumentParser(description="Continuous Streaming Evaluation on Server Machine Dataset")
    parser.add_argument("--data_dir", type=str, default="data/smd")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/smd_coupled_tcn_best.pt")
    parser.add_argument("--window_size", type=int, default=200)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--output_json", type=str, default="results/smd_pooled28_coupled_tcn_eval.json")
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

    # Load model checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    hidden_dim = ckpt.get("model_config", {}).get("hidden_dim", 32)
    model = SMDSubsystemCoupledTCN(in_channels=38, hidden_dim=hidden_dim, teacher_feat_dim=1024)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Architecture: SMDSubsystemCoupledTCN | Hidden Dim: {hidden_dim} | Params: {total_params:,} ({total_params/1e6:.2f}M)")

    test_files = sorted(glob.glob(os.path.join(args.data_dir, "test", "*.txt")))
    print(f"\nDiscovered {len(test_files)} SMD machine streams.")

    all_heldout_scores = []
    all_heldout_labels = []
    all_full_scores = []
    all_full_labels = []

    per_machine_results = []
    total_eval_time = 0.0
    total_windows_evaluated = 0

    print("\n" + "=" * 90)
    print(f"{'Machine':<14} | {'Steps':<7} | {'Held-Out AUROC':<15} | {'Held-Out F1':<12} | {'Full AUROC':<12} | {'Events':<10}")
    print("=" * 90)

    for tf in test_files:
        m_name = os.path.basename(tf).replace(".txt", "")
        lf = os.path.join(args.data_dir, "test_label", os.path.basename(tf))

        raw_data = pd.read_csv(tf, header=None).values.astype(np.float32)
        labels = pd.read_csv(lf, header=None)[0].values.astype(np.float32)

        p_scores, y_true, a, b, infer_time, n_windows = evaluate_single_machine(
            model, raw_data, labels,
            train_ratio=0.50, val_ratio=0.10,
            window_size=args.window_size, stride=args.stride,
            batch_size=args.batch_size, device=device
        )

        total_eval_time += infer_time
        total_windows_evaluated += n_windows

        # Strictly Held-Out Test Slice: from step b (60%) to end (100%)
        ho_scores = p_scores[b:]
        ho_labels = y_true[b:]

        all_heldout_scores.append(ho_scores)
        all_heldout_labels.append(ho_labels)
        all_full_scores.append(p_scores)
        all_full_labels.append(y_true)

        # Per-machine metrics
        ho_m = compute_metrics(ho_labels, ho_scores)
        full_m = compute_metrics(y_true, p_scores)

        per_machine_results.append({
            "machine": m_name,
            "timesteps": len(raw_data),
            "held_out_timesteps": len(ho_labels),
            "held_out_anomalies": int(np.sum(ho_labels)),
            "held_out_auroc": ho_m["auroc"],
            "held_out_prauc": ho_m["prauc"],
            "held_out_f1": ho_m["best_f1"],
            "held_out_threshold": ho_m["best_threshold"],
            "full_timesteps": len(y_true),
            "full_anomalies": int(np.sum(y_true)),
            "full_auroc": full_m["auroc"],
            "full_prauc": full_m["prauc"],
            "full_f1": full_m["best_f1"],
            "full_threshold": full_m["best_threshold"],
            "detected_events": full_m["detected_events"],
            "total_events": full_m["total_events"]
        })

        ev_str = f"{full_m['detected_events']}/{full_m['total_events']}"
        print(f"{m_name:<14} | {len(raw_data):<7} | {ho_m['auroc']:<15.4f} | {ho_m['best_f1']:<12.4f} | {full_m['auroc']:<12.4f} | {ev_str:<10}")

    print("=" * 90)

    # Concatenate across all 28 machines
    concat_ho_scores = np.concatenate(all_heldout_scores)
    concat_ho_labels = np.concatenate(all_heldout_labels)

    concat_full_scores = np.concatenate(all_full_scores)
    concat_full_labels = np.concatenate(all_full_labels)

    print("\nComputing Combined Foundation Metrics Across All 28 Machines...")
    ho_combined = compute_metrics(concat_ho_labels, concat_ho_scores)
    full_combined = compute_metrics(concat_full_labels, concat_full_scores)

    avg_latency_ms = (total_eval_time / max(1, total_windows_evaluated)) * 1000.0
    throughput_fps = total_windows_evaluated / max(1e-6, total_eval_time)

    # Compare against collapsed baseline
    baseline_ho_auroc = 0.6044
    baseline_ho_prauc = 0.0864
    baseline_ho_f1 = 0.1483

    baseline_full_auroc = 0.6194
    baseline_full_prauc = 0.1241
    baseline_full_f1 = 0.1855

    print("\n" + "=" * 90)
    print("ALL 28 SMD MACHINES POOLED CONTINUOUS EVALUATION RESULTS")
    print("=" * 90)
    print(f"1. Strictly Held-Out Test Slice (Last 40% across all 28 machines):")
    print(f"   Timesteps:       {len(concat_ho_labels):,} (Anomalies: {int(np.sum(concat_ho_labels)):,}, Prevalence: {np.mean(concat_ho_labels)*100:.2f}%)")
    print(f"   AUROC:           {ho_combined['auroc']:.4f}  (Baseline: {baseline_ho_auroc:.4f} | Delta: +{(ho_combined['auroc'] - baseline_ho_auroc):.4f})")
    print(f"   PR-AUC:          {ho_combined['prauc']:.4f}  (Baseline: {baseline_ho_prauc:.4f} | Delta: +{(ho_combined['prauc'] - baseline_ho_prauc):.4f})")
    print(f"   Best F1:         {ho_combined['best_f1']:.4f}  (Baseline: {baseline_ho_f1:.4f} | Delta: +{(ho_combined['best_f1'] - baseline_ho_f1):.4f})")
    print(f"   Precision:       {ho_combined['precision']:.4f}")
    print(f"   Point Recall:    {ho_combined['recall']:.4f}")
    print(f"   FPR:             {ho_combined['fpr']*100:.2f}%")
    print(f"   Optimal Thresh:  {ho_combined['best_threshold']:.4f}")
    print(f"   Event Recall:    {ho_combined['detected_events']}/{ho_combined['total_events']} ({ho_combined['event_recall']*100:.1f}%)")

    print(f"\n2. Complete Continuous Stream (All 28 machines 0%..100%):")
    print(f"   Timesteps:       {len(concat_full_labels):,} (Anomalies: {int(np.sum(concat_full_labels)):,}, Prevalence: {np.mean(concat_full_labels)*100:.2f}%)")
    print(f"   AUROC:           {full_combined['auroc']:.4f}  (Baseline: {baseline_full_auroc:.4f} | Delta: +{(full_combined['auroc'] - baseline_full_auroc):.4f})")
    print(f"   PR-AUC:          {full_combined['prauc']:.4f}  (Baseline: {baseline_full_prauc:.4f} | Delta: +{(full_combined['prauc'] - baseline_full_prauc):.4f})")
    print(f"   Best F1:         {full_combined['best_f1']:.4f}  (Baseline: {baseline_full_f1:.4f} | Delta: +{(full_combined['best_f1'] - baseline_full_f1):.4f})")
    print(f"   Precision:       {full_combined['precision']:.4f}")
    print(f"   Point Recall:    {full_combined['recall']:.4f}")
    print(f"   FPR:             {full_combined['fpr']*100:.2f}%")
    print(f"   Event Recall:    {full_combined['detected_events']}/{full_combined['total_events']} ({full_combined['event_recall']*100:.1f}%)")

    print(f"\n3. Inference Latency & Throughput:")
    print(f"   Per-Window Latency: {avg_latency_ms:.3f} ms")
    print(f"   Throughput:         {throughput_fps:,.1f} windows/sec")
    print("=" * 90)

    # Save results to JSON
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    results_payload = {
        "dataset": "Server Machine Dataset (SMD)",
        "num_machines": len(test_files),
        "total_timesteps": int(len(concat_full_labels)),
        "total_anomalies": int(np.sum(concat_full_labels)),
        "model": "SMDSubsystemCoupledTCN",
        "num_params": int(total_params),
        "window_size": args.window_size,
        "stride": args.stride,
        "latency_ms": float(avg_latency_ms),
        "throughput_fps": float(throughput_fps),
        "held_out_slice": {
            "timesteps": int(len(concat_ho_labels)),
            "anomalies": int(np.sum(concat_ho_labels)),
            "metrics": ho_combined,
            "baseline_comparison": {
                "baseline_auroc": baseline_ho_auroc,
                "baseline_prauc": baseline_ho_prauc,
                "baseline_f1": baseline_ho_f1,
                "auroc_delta": float(ho_combined["auroc"] - baseline_ho_auroc),
                "prauc_delta": float(ho_combined["prauc"] - baseline_ho_prauc),
                "f1_delta": float(ho_combined["best_f1"] - baseline_ho_f1)
            }
        },
        "complete_stream": {
            "timesteps": int(len(concat_full_labels)),
            "anomalies": int(np.sum(concat_full_labels)),
            "metrics": full_combined,
            "baseline_comparison": {
                "baseline_auroc": baseline_full_auroc,
                "baseline_prauc": baseline_full_prauc,
                "baseline_f1": baseline_full_f1,
                "auroc_delta": float(full_combined["auroc"] - baseline_full_auroc),
                "prauc_delta": float(full_combined["prauc"] - baseline_full_prauc),
                "f1_delta": float(full_combined["best_f1"] - baseline_full_f1)
            }
        },
        "per_machine": per_machine_results
    }

    with open(args.output_json, "w") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\nEvaluation results exported to: {args.output_json}")


if __name__ == "__main__":
    main()
