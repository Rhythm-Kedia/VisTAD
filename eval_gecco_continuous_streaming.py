"""Continuous Streaming Deployment & Evaluation on GECCO Drinking Water Stream.

Evaluates the Subsystem-Coupled Causal TCN Student in real-time continuous deployment:
- Primary Held-Out Stream: Partition 2 (Timestamps 70,000 to 139,565, 69,566 continuous seconds, 22 unseen events)
- Legacy VisTAD Paper Slice: Timestamps 111,652 to 139,565 (27,914 seconds, Event 51 only)
- Continuous Sliding Stride: 10 steps (every 10 seconds, sliding 200-step causal window)
- Point-wise Evaluation: Maps predictions across sliding overlaps into exact point-by-point timeline
"""

import os
import sys
import time
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

    # Normalize by visit counts to get continuous point-wise probability
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
        "total_windows": len(ds)
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Continuous Streaming on GECCO")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/gecco_coupled_tcn_best.pt")
    parser.add_argument("--data_path", type=str, default="data/gecco/gecco_water.csv")
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
    model = build_student("gecco_coupled_tcn", hidden_dim=cfg.get("hidden_dim", 64)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    full_data, full_labels = load_gecco_stream(args.data_path)

    base_m = cfg.get("base_mean")
    base_s = cfg.get("base_std")
    if base_m is not None and base_s is not None:
        print("Standardizing continuous stream using checkpoint baseline parameters...")
        full_data = (full_data - base_m) / (base_s + 1e-6)

    # 1. Primary Held-Out Evaluation: Partition 2 (Timestamps 70,000 to 139,566)
    p2_data = full_data[70000:]
    p2_labels = full_labels[70000:]
    print(f"\n--- Running Continuous Streaming Evaluation: Partition 2 (Timestamps 70,000 to {len(full_data)}) ---")
    print(f"Stream: {len(p2_data):,} seconds ({p2_labels.sum():.0f} anomalous steps across 22 unseen events, stride={args.stride})")

    res_p2 = run_continuous_streaming_eval(
        model, p2_data, p2_labels,
        window_size=args.window_size, stride=args.stride,
        batch_size=args.batch_size, device=device
    )

    print("\n" + "=" * 75)
    print("GECCO CONTINUOUS STREAMING RESULTS: HELD-OUT PARTITION 2 (70k -> 140k)")
    print("=" * 75)
    print(f"Evaluated Windows:      {res_p2['total_windows']:,} sliding windows ({res_p2['total_steps']:,} seconds)")
    print(f"Continuous Point AUROC: {res_p2['auroc']:.4f}")
    print(f"Continuous Point PR-AUC:{res_p2['prauc']:.4f}")
    print(f"Continuous Point Best F1:{res_p2['best_f1']:.4f} (at threshold {res_p2['best_threshold']:.4f})")
    print(f"Continuous Precision:   {res_p2['precision']*100:.2f}%")
    print(f"Continuous Recall:      {res_p2['recall']*100:.2f}% ({res_p2['tp']:,} / {res_p2['anomaly_steps']:,} anomalous seconds caught)")
    print(f"Continuous Specificity: {res_p2['specificity']*100:.2f}% (FPR = {res_p2['fpr']*100:.2f}%)")
    print(f"False Alarms:           {res_p2['fp']:,} false alarm seconds out of {res_p2['nominal_steps']:,} nominal seconds")
    print(f"Throughput:             {res_p2['throughput_fps']:,.1f} windows/sec ({res_p2['latency_per_window_ms']:.3f} ms/window)")
    print(f"Evaluation Duration:    {res_p2['duration_sec']:.2f} seconds for full ~48-day stream")
    print("=" * 75)

    # 2. Legacy VisTAD Paper Slice Evaluation: Timestamps 111,652 to 139,566
    p_legacy_data = full_data[111652:]
    p_legacy_labels = full_labels[111652:]
    print(f"\n--- Running Continuous Streaming Evaluation: Legacy VisTAD Paper Slice (111,652 to {len(full_data)}) ---")
    print(f"Stream: {len(p_legacy_data):,} seconds ({p_legacy_labels.sum():.0f} anomalous steps, Event 51)")

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


if __name__ == "__main__":
    main()
