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
        "total_eval_time": float(duration)
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate MSLCoupledTCNStudent continuous streaming")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/msl_coupled_tcn_best.pt")
    parser.add_argument("--test_csv", type=str, default="data/msl/msl_test.csv")
    parser.add_argument("--train_csv", type=str, default="data/msl/msl_train.csv")
    parser.add_argument("--window_size", type=int, default=200)
    parser.add_argument("--stride", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Loading checkpoint from: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    model = build_student("msl_coupled_tcn", in_channels=55).to(device)
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

    # 2. Complete Test Stream (t = 0 to 73,728)
    print("\n" + "=" * 75)
    print("B. COMPLETE NASA MSL TEST STREAM (t = 0 to 73,728, ALL 36 EPISODES)")
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


if __name__ == "__main__":
    main()
