"""Optimized Top-K Channel Disjunction Ablation Study for the Grand Foundation Model.

Precomputes continuous per-channel probability streams P(t, c) ONCE per dataset,
then evaluates all K in Top-K instantly via vectorized NumPy operations.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.universal_tcn import UniversalFoundationTCN_Pro
from src.data.universal_dataset import load_dataset_telemetry, UniversalWindowDataset, UniversalCollate


def compute_contiguous_event_recall(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, float]:
    """Computes event-level recall across contiguous positive segments."""
    diff = np.diff(np.pad(y_true, (1, 1), mode="constant"))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    total_events = len(starts)
    if total_events == 0:
        return 0, 0, 1.0
    detected = 0
    for s, e in zip(starts, ends):
        if np.any(y_pred[s:e] > 0):
            detected += 1
    return detected, total_events, float(detected / total_events)


def precompute_channel_scores(model: torch.nn.Module, raw_data: np.ndarray, labels: np.ndarray,
                              window_size: int = 200, prefix_len: int = 100,
                              stride: int = 20, batch_size: int = 32, device: str = "cpu") -> tuple[np.ndarray, np.ndarray, float]:
    """Runs a single forward pass over the stream to extract P(t, c) for all timesteps and channels."""
    n, C = raw_data.shape
    ds = UniversalWindowDataset(
        raw_data, labels,
        window_size=window_size,
        prefix_len=prefix_len,
        stride=stride,
        augment_synthetic=False
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=UniversalCollate())
    
    channel_scores = np.zeros((n, C), dtype=np.float32)
    point_counts = np.zeros(n, dtype=np.float32)
    
    model.eval()
    t0 = time.time()
    use_autocast = device.startswith("cuda") if isinstance(device, str) else (device.type == "cuda")

    with torch.no_grad():
        for batch in loader:
            st_indices = batch["st"].numpy()
            x = batch["x"].to(device)
            if use_autocast:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    out = model(x, prefix_len=prefix_len)
            else:
                out = model(x, prefix_len=prefix_len)

            # Extract target window channel probabilities: [B, C, L_target]
            if prefix_len > 0:
                p_chan = out["p_channel"][:, :, prefix_len:].float().cpu().numpy()
            else:
                p_chan = out["p_channel"].float().cpu().numpy()

            p_chan_t = np.transpose(p_chan, (0, 2, 1))  # [B, L_target, C]
            target_len = p_chan_t.shape[1]

            for i, st in enumerate(st_indices):
                ed = min(st + target_len, n)
                cur_l = ed - st
                channel_scores[st:ed] += p_chan_t[i, :cur_l]
                point_counts[st:ed] += 1.0

    valid = point_counts > 0
    channel_scores[valid] /= point_counts[valid, None]
    if not valid[0] and np.any(valid):
        first_valid = np.where(valid)[0][0]
        channel_scores[:first_valid] = channel_scores[first_valid]

    channel_scores = np.nan_to_num(channel_scores, nan=0.0, posinf=1.0, neginf=0.0)
    channel_scores = np.clip(channel_scores, 0.0, 1.0)
    infer_time = time.time() - t0
    return channel_scores, labels.astype(int), infer_time


def evaluate_top_k_metrics(channel_scores: np.ndarray, y_true: np.ndarray, top_k: int | str = 5) -> dict:
    """Computes all metrics given precomputed per-channel stream scores."""
    n, C = channel_scores.shape
    if str(top_k).lower() == "all" or (isinstance(top_k, (int, float)) and int(top_k) >= C):
        k_val = C
    else:
        k_val = max(1, min(int(top_k), C))

    # Sort channels along channel axis and take top-k
    top_p = np.sort(channel_scores, axis=-1)[:, -k_val:]
    p_clamped = np.clip(top_p, 1e-5, 1.0 - 1e-5)
    point_scores = 1.0 - np.prod(1.0 - p_clamped, axis=-1)
    point_scores = np.clip(point_scores, 0.0, 1.0)

    if len(np.unique(y_true)) < 2:
        return {"auroc": 0.5, "prauc": float(np.mean(y_true)), "best_f1": 0.0, "event_recall": 1.0}

    try:
        auroc = float(roc_auc_score(y_true, point_scores))
    except Exception:
        auroc = 0.5

    try:
        prauc = float(average_precision_score(y_true, point_scores))
    except Exception:
        prauc = float(np.mean(y_true))

    precisions, recalls, thresholds = precision_recall_curve(y_true, point_scores)
    f1_scores = 2 * (precisions * recalls) / np.clip(precisions + recalls, 1e-12, None)
    best_idx = np.argmax(f1_scores)
    best_f1 = float(f1_scores[best_idx])
    thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

    y_pred = (point_scores >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = float(fp / (fp + tn + 1e-12))
    recall = float(tp / (tp + fn + 1e-12))
    precision = float(tp / (tp + fp + 1e-12))

    det_ev, tot_ev, ev_rec = compute_contiguous_event_recall(y_true, y_pred)

    return {
        "auroc": auroc,
        "prauc": prauc,
        "best_f1": best_f1,
        "best_threshold": thresh,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "detected_events": det_ev,
        "total_events": tot_ev,
        "event_recall": ev_rec,
        "actual_k": k_val
    }


def main():
    parser = argparse.ArgumentParser(description="Fast Precomputed Top-K Disjunction Ablation")
    parser.add_argument("--ckpt", type=str, default="checkpoints/universal_swat,smd,gecco,msl_to_swat_best.pt")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--datasets", nargs="+", default=["gecco", "msl", "smd", "swat"])
    parser.add_argument("--k_values", nargs="+", default=["1", "2", "3", "5", "8", "10", "all"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--output_json", type=str, default="results/ablations/top_k_ablation_grand_model.json")
    args = parser.parse_args()

    device = args.device
    if not device:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"Top-K Ablation Compute Device: {device}")

    # Load Grand Foundation Model
    print(f"Loading Grand Foundation Model from: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt.get("model_config", {})
    hidden_dim = cfg.get("hidden_dim", 128)
    num_heads = cfg.get("num_heads", 8)
    num_layers = cfg.get("num_layers", 2)
    prefix_len = cfg.get("prefix_len", 100)

    model = UniversalFoundationTCN_Pro(
        hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers,
        dilations=(1, 2, 4, 8, 16, 32), dropout=0.0
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    all_results = {}
    print("\n" + "=" * 105)
    print("STARTING FAST PRECOMPUTED TOP-K PROBABILISTIC DISJUNCTION ABLATION")
    print("=" * 105)

    for ds_name in args.datasets:
        print(f"\n>>> Loading Test Slice for Dataset: [{ds_name.upper()}] <<<")
        raw, labels, train_end, total_C = load_dataset_telemetry(ds_name, args.data_dir)
        test_x = raw[train_end:]
        test_y = labels[train_end:]
        n_anom = int(np.sum(test_y))
        print(f"  Test Slice: {len(test_x):,} timesteps | Channels C: {total_C} | Anomaly steps: {n_anom:,} ({n_anom/len(test_x)*100:.2f}%)")

        print(f"  Precomputing P(t, c) representation with stride={args.stride}, batch_size={args.batch_size}...")
        chan_scores, y_true, infer_time = precompute_channel_scores(
            model, test_x, test_y,
            window_size=200, prefix_len=prefix_len,
            stride=args.stride, batch_size=args.batch_size,
            device=device
        )
        print(f"  Precomputed P(t, c) in {infer_time:.2f}s ({len(test_x)/infer_time:.1f} steps/sec)")

        ds_results = {}
        print("-" * 105)
        print(f"{'Dataset':<8} | {'Top-K':<8} | {'AUROC':<9} | {'PR-AUC':<9} | {'Best F1':<9} | {'Precision':<10} | {'Recall':<9} | {'FPR':<8} | {'Events':<12}")
        print("-" * 105)

        for k_str in args.k_values:
            k_val = "all" if k_str.lower() == "all" else int(k_str)
            m = evaluate_top_k_metrics(chan_scores, y_true, top_k=k_val)
            k_display = f"{total_C} (All)" if k_str.lower() == "all" else f"{k_val}"
            ev_str = f"{m['detected_events']}/{m['total_events']} ({m['event_recall']*100:.1f}%)"
            print(f"{ds_name.upper():<8} | K={k_display:<6} | {m['auroc']:<9.4f} | {m['prauc']:<9.4f} | {m['best_f1']:<9.4f} | {m['precision']*100:6.2f}%   | {m['recall']*100:6.2f}% | {m['fpr']*100:5.2f}% | {ev_str:<12}")

            ds_results[f"K_{k_str}"] = {
                "top_k": k_str,
                "actual_k": m["actual_k"],
                "total_channels": total_C,
                "metrics": m
            }

        all_results[ds_name] = ds_results

    # Export JSON
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print("\n" + "=" * 105)
    print(f"Top-K Ablation payload exported to: {args.output_json}")
    print("=" * 105)


if __name__ == "__main__":
    main()
