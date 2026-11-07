"""Continuous Streaming Evaluation (Stride 10) on SWaT Held-Out Test Stream.

Evaluates Stage-Coupled Causal TCN Student in continuous real-time deployment:
- Timeline: Held-out Partition 2 (Timestamps 224,960 to 449,919, 224,960 seconds)
- Sliding Stride: 10 steps (every 10 seconds, sliding 200-step causal window)
- Point-wise mapping: maps predictions to continuous 1-second timeline
- Metrics: Point-wise AUROC, Point-wise PR-AUC, Point-wise Best F1, Recall, FPR, Latency
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


def load_swat_test_stream(data_dir="data/swat"):
    norm_path = os.path.join(data_dir, "SWaT_Dataset_Normal_v1.csv")
    att_path = os.path.join(data_dir, "SWaT_Dataset_Attack_v0.csv")

    df_norm = pd.read_csv(norm_path)
    norm_chans = [c for c in df_norm.columns if c not in ["Timestamp", "Normal/Attack", "attack", "label"]]
    norm_data = df_norm[norm_chans].values.astype(np.float32)
    base_m = np.mean(norm_data, axis=0)
    base_s = np.maximum(np.std(norm_data, axis=0), 0.05)

    df_att = pd.read_csv(att_path)
    att_cols = [c for c in df_att.columns if c not in ["Timestamp", "Normal/Attack", "attack", "label"]]
    att_data = df_att[att_cols].values.astype(np.float32)
    
    label_col = [c for c in df_att.columns if c in ["Normal/Attack", "attack", "label"]][0]
    raw_labels = df_att[label_col].values
    if raw_labels.dtype == object:
        labels = (raw_labels != "Normal").astype(np.float32)
    else:
        labels = (raw_labels == 1).astype(np.float32)

    return att_data, labels, base_m, base_s


class ContinuousStreamDataset(Dataset):
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
            "x": torch.from_numpy(self.data[st:ed]).float() # [200, 51]
        }


def run_continuous_streaming_eval(model, stream_data, stream_labels, base_m=None, base_s=None,
                                  window_size=200, stride=10, batch_size=128, device="cpu", normalize=False):
    """Slides through the continuous stream and aggregates point-wise predictions."""
    model.eval()
    if normalize and base_m is not None and base_s is not None:
        norm_stream = (stream_data - base_m) / base_s
    else:
        norm_stream = stream_data  # Matches cached training tensor representation
    n_points = len(stream_data)

    ds = ContinuousStreamDataset(norm_stream, window_size=window_size, stride=stride)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    # Accumulator for point-wise scores and visit counts
    point_scores = np.zeros(n_points, dtype=np.float32)
    point_counts = np.zeros(n_points, dtype=np.float32)

    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            starts = batch["st"].numpy()
            x = batch["x"].to(device)  # [B, 200, 51]
            out = model(x)
            pred_dense = out["pred_dense"].cpu().numpy()  # [B, 200]

            for i, st in enumerate(starts):
                point_scores[st : st + window_size] += pred_dense[i]
                point_counts[st : st + window_size] += 1.0

    duration = time.time() - t0
    fps = len(ds) / duration
    latency_ms = (duration / len(ds)) * 1000.0

    # Average overlapping predictions
    mask = point_counts > 0
    point_scores[mask] /= point_counts[mask]

    # Evaluate on the covered timeline
    y_true = stream_labels[mask].astype(int)
    scores = point_scores[mask]

    auroc = float(roc_auc_score(y_true, scores))
    pr_auc = float(average_precision_score(y_true, scores))

    p, r, th = precision_recall_curve(y_true, scores)
    f1s = 2 * p * r / np.clip(p + r, 1e-12, None)
    best_idx = int(np.argmax(f1s))
    best_f1 = float(f1s[best_idx])
    best_th = float(th[min(best_idx, len(th) - 1)])

    preds = (scores >= best_th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    recall = float(tp / max(1, tp + fn))
    spec = float(tn / max(1, tn + fp))
    fpr = float(fp / max(1, fp + tn))
    fnr = float(fn / max(1, fn + tp))

    return {
        "n_points": int(mask.sum()),
        "n_windows": len(ds),
        "stride": stride,
        "auroc": auroc,
        "pr_auc": pr_auc,
        "best_f1": best_f1,
        "best_th": best_th,
        "recall": recall,
        "specificity": spec,
        "fpr": fpr,
        "fnr": fnr,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "duration_sec": duration,
        "fps_windows_per_sec": fps,
        "latency_ms_per_window": latency_ms
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/stage_coupled_tcn_best.pt")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Running continuous evaluation on device: {device}")

    att_data, labels, base_m, base_s = load_swat_test_stream("data/swat")
    n_split = len(att_data) // 2

    # Held-Out Partition 2 (Timestamps 224,960 to 449,919)
    test_data = att_data[n_split:]
    test_labels = labels[n_split:]
    print(f"\nHeld-Out Test Stream (Partition 2): {len(test_data)} timestamps ({int(test_labels.sum())} attack steps, {int((1-test_labels).sum())} nominal steps)")

    print(f"Loading Student Checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, weights_only=False)
    model = build_student("stage_coupled_tcn", stage_dim=32, fusion_dim=128, feat_dim=1024).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    print(f"\nExecuting continuous streaming evaluation (stride={args.stride}, window=200)...")
    res = run_continuous_streaming_eval(
        model, test_data, test_labels, base_m, base_s,
        window_size=200, stride=args.stride, batch_size=128, device=device
    )

    print("\n" + "=" * 80)
    print("   SWAT CONTINUOUS STREAMING EVALUATION RESULTS (STRIDE 10, HELD-OUT TEST)")
    print("=" * 80)
    print(f"Total Stream Timestamps:   {res['n_points']:,} seconds")
    print(f"Sliding Windows Evaluated: {res['n_windows']:,} windows (stride={res['stride']})")
    print(f"Inference Time:            {res['duration_sec']:.2f} seconds ({res['fps_windows_per_sec']:.1f} windows/sec)")
    print(f"Per-Window Latency:        {res['latency_ms_per_window']:.3f} ms")
    print("-" * 80)
    print(f"Continuous AUROC:          {res['auroc']:.4f}")
    print(f"Continuous PR-AUC:         {res['pr_auc']:.4f}")
    print(f"Point-wise Best F1:        {res['best_f1']:.4f} (at tau* = {res['best_th']:.4f})")
    print(f"Recall (Sensitivity):      {res['recall']*100:.2f}% ({res['tp']:,} / {res['tp']+res['fn']:,} true attack steps caught)")
    print(f"Specificity:               {res['specificity']*100:.2f}% ({res['tn']:,} / {res['tn']+res['fp']:,} true nominal steps)")
    print(f"False Alarm Rate (FPR):    {res['fpr']*100:.2f}%")
    print(f"Miss Rate (FNR):           {res['fnr']*100:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
