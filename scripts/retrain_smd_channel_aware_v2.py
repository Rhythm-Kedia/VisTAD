"""Retraining and Diagnostic Audit of SMD Universal TCN Channel-Aware Student.

Ablates:
1. Model Scale: hidden_dim=128, num_heads=8, ffn_dim=512 (2.05M params, matching Pro)
2. Calibrated Loss: pos_weight=1.0 (unbiased classification) + feature distillation
3. Aggregation Modes on Continuous Stream:
   - Top-5 Probabilistic OR (Original default)
   - Top-1 Probabilistic OR (Highest-risk single channel)
   - Point-wise Channel Max (max_c p_c, matching Residual-TCN)

Saves:
- Checkpoint: checkpoints/smd_channel_aware_audit_v2.pt
- Results: results/smd_channel_aware_audit_v2_eval.json
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.universal_tcn_channel_aware import UniversalTCNChannelAwareStudent
from src.loss import SimplifiedDistillationLoss


def get_flag(w):
    if "y_flag" in w:
        val = w["y_flag"]
        return float(val.item() if hasattr(val, "item") else val)
    y = w.get("y_dense", w.get("y", None))
    if y is not None:
        y_t = torch.as_tensor(y)
        return float((y_t.max() > 0.5).item())
    return 0.0


def get_teacher_score(w):
    s = w.get("teacher_score", 0.5)
    return float(s.item() if hasattr(s, "item") else s)


class UnifiedCachedDataset(Dataset):
    """Cached dataset reading from Ultra-VETime dictionary cache."""
    def __init__(self, windows_dict, keys, in_channels=38):
        self.keys = keys
        self.windows = windows_dict
        self.in_channels = in_channels

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        k = self.keys[idx]
        w = self.windows[k]
        raw_x = torch.as_tensor(w["x"], dtype=torch.float32)
        if raw_x.dim() == 2 and raw_x.shape[0] != self.in_channels and raw_x.shape[1] == self.in_channels:
            raw_x = raw_x.transpose(0, 1)

        y_dense = w.get("y_dense") if "y_dense" in w else w.get("y", np.zeros(200))
        y_flag = get_flag(w)
        z_teacher = w.get("z_teacher", np.zeros(1024))
        teacher_score = get_teacher_score(w)

        return {
            "x": raw_x,
            "y_dense": torch.as_tensor(y_dense, dtype=torch.float32),
            "y_flag": torch.tensor(y_flag, dtype=torch.float32),
            "z_teacher": torch.as_tensor(z_teacher, dtype=torch.float32),
            "teacher_score": torch.tensor(teacher_score, dtype=torch.float32)
        }


def partition_cache(train_cache_path, val_ratio=0.1):
    print(f"Loading curriculum cache from {train_cache_path}...")
    c = torch.load(train_cache_path, map_location="cpu", weights_only=False)
    windows = c["windows"] if isinstance(c, dict) and "windows" in c else c
    valid_keys = [k for k in windows.keys() if isinstance(windows[k], dict) and "x" in windows[k]]

    anom_keys = [k for k in valid_keys if get_flag(windows[k]) > 0.5]
    norm_keys = [k for k in valid_keys if get_flag(windows[k]) <= 0.5]

    n_val_anom = max(1, int(len(anom_keys) * val_ratio))
    n_val_norm = max(1, int(len(norm_keys) * val_ratio))

    train_anom = anom_keys[:-n_val_anom]
    val_anom = anom_keys[-n_val_anom:]

    train_norm = norm_keys[:-n_val_norm]
    val_norm = norm_keys[-n_val_norm:]

    train_keys = sorted(train_anom + train_norm)
    val_keys = sorted(val_anom + val_norm)

    print(f"Curriculum Partition: Total={len(valid_keys)} windows")
    print(f"  Train: {len(train_keys)} windows ({len(train_anom)} anomaly, {len(train_norm)} nominal)")
    print(f"  Val:   {len(val_keys)} windows ({len(val_anom)} anomaly, {len(val_norm)} nominal)")

    return windows, train_keys, val_keys


def evaluate_cache_win_metrics(model, dataloader, device):
    model.eval()
    all_scores, all_flags = [], []
    t0 = time.time()
    with torch.no_grad():
        for batch in dataloader:
            x = batch["x"].to(device)
            out = model(x)
            pred_dense = out["pred_dense"].cpu().numpy()
            y_flag = batch["y_flag"].numpy()
            win_scores = np.max(pred_dense, axis=-1)
            all_scores.extend(win_scores)
            all_flags.extend(y_flag)

    eval_dur = time.time() - t0
    y_true = np.array(all_flags)
    y_scores = np.array(all_scores)

    auroc = roc_auc_score(y_true, y_scores)
    prauc = average_precision_score(y_true, y_scores)

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * (precisions * recalls) / np.clip(precisions + recalls, 1e-12, None)
    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx]
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    y_pred_binary = (y_scores >= best_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    fpr = fp / (fp + tn + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    precision = tp / (tp + fp + 1e-12)

    return {
        "auroc": float(auroc),
        "prauc": float(prauc),
        "best_f1": float(best_f1),
        "best_threshold": float(best_thresh),
        "precision": float(precision),
        "recall": float(recall),
        "fpr": float(fpr),
        "latency_ms": float((eval_dur / len(y_true)) * 1000.0),
        "total_windows": len(y_true)
    }


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
            "x": torch.from_numpy(self.data[st:ed]).float().transpose(0, 1)  # [38, 200]
        }


def compute_point_adjusted_metrics(y_true, y_pred_binary):
    y_adj = y_pred_binary.copy()
    events = []
    in_event = False
    st = 0
    for i, val in enumerate(y_true):
        if val == 1 and not in_event:
            in_event = True
            st = i
        elif val == 0 and in_event:
            in_event = False
            events.append((st, i))
    if in_event:
        events.append((st, len(y_true)))

    for st_e, ed_e in events:
        if np.any(y_pred_binary[st_e:ed_e] == 1):
            y_adj[st_e:ed_e] = 1

    cm = confusion_matrix(y_true, y_adj)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    prec = float(tp / max(tp + fp, 1))
    rec = float(tp / max(tp + fn, 1))
    pa_f1 = float(2 * prec * rec / max(prec + rec, 1e-8))
    return pa_f1


def run_continuous_streaming_eval(model, entities_list, aggregation_mode="top5",
                                  window_size=200, stride=10, batch_size=128, device="cpu"):
    """
    Evaluates continuous streaming across 28 SMD machines.
    aggregation_mode options:
      - 'top5': Original Top-5 Probabilistic OR
      - 'top1': Top-1 Probabilistic OR (single highest-risk channel)
      - 'max':  Point-wise Channel Max (max_c p_c, like Residual-TCN)
    """
    model.eval()
    all_y = []
    all_scores = []
    total_windows = 0
    t0 = time.time()

    with torch.no_grad():
        for stream_data, stream_labels in entities_list:
            n_points = len(stream_data)
            ds = ContinuousStreamDataset(stream_data, window_size=window_size, stride=stride)
            total_windows += len(ds)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

            point_scores = np.zeros(n_points, dtype=np.float32)
            point_counts = np.zeros(n_points, dtype=np.float32)

            for batch in loader:
                starts = batch["st"].numpy()
                x = batch["x"].to(device)
                out = model(x)
                p_chan = out["p_channel"]  # [B, 38, L]

                if aggregation_mode == "top5":
                    top_p, _ = torch.topk(p_chan, k=min(5, p_chan.shape[1]), dim=1)
                    p_clamped = torch.clamp(top_p, 1e-6, 1.0 - 1e-6)
                    pred_dense = (1.0 - torch.prod(1.0 - p_clamped, dim=1)).cpu().numpy()
                elif aggregation_mode == "top1":
                    top_p, _ = torch.topk(p_chan, k=1, dim=1)
                    pred_dense = top_p.squeeze(1).cpu().numpy()
                elif aggregation_mode == "max":
                    pred_dense = torch.max(p_chan, dim=1)[0].cpu().numpy()
                else:
                    raise ValueError(f"Unknown aggregation_mode: {aggregation_mode}")

                for i, st in enumerate(starts):
                    point_scores[st : st + window_size] += pred_dense[i]
                    point_counts[st : st + window_size] += 1.0

            valid_mask = point_counts > 0
            final_probs = np.zeros(n_points, dtype=np.float32)
            final_probs[valid_mask] = point_scores[valid_mask] / point_counts[valid_mask]

            all_y.append(stream_labels[valid_mask])
            all_scores.append(final_probs[valid_mask])

    duration = time.time() - t0
    fps = total_windows / max(duration, 1e-6)
    latency_ms = (duration / max(total_windows, 1)) * 1000.0

    y_eval = np.concatenate(all_y, axis=0)
    s_eval = np.concatenate(all_scores, axis=0)

    auroc = float(roc_auc_score(y_eval, s_eval)) if len(np.unique(y_eval)) > 1 else 0.5
    pr_auc = float(average_precision_score(y_eval, s_eval)) if len(np.unique(y_eval)) > 1 else 0.0

    precisions, recalls, thresholds = precision_recall_curve(y_eval, s_eval)
    f1_scores = 2 * (precisions * recalls) / np.maximum(precisions + recalls, 1e-8)
    best_idx = np.argmax(f1_scores)
    best_f1 = float(f1_scores[best_idx])
    best_thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

    y_pred_bin = (s_eval >= best_thresh).astype(int)
    cm = confusion_matrix(y_eval, y_pred_bin)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    fpr = float(fp / max(fp + tn, 1))
    specificity = float(tn / max(tn + fp, 1))

    pa_f1 = compute_point_adjusted_metrics(y_eval, y_pred_bin)

    tot_events = 0
    caught_events = 0
    for (_, stream_labels), s_arr in zip(entities_list, all_scores):
        diff = np.diff(np.pad(stream_labels[:len(s_arr)], (1, 1), mode="constant"))
        starts_ev = np.where(diff == 1)[0]
        ends_ev = np.where(diff == -1)[0]
        for s_e, e_e in zip(starts_ev, ends_ev):
            tot_events += 1
            if np.any(s_arr[s_e:e_e] >= best_thresh):
                caught_events += 1

    event_recall = f"{caught_events}/{tot_events}" if tot_events > 0 else "N/A"
    event_pct = (caught_events / max(tot_events, 1)) * 100.0

    return {
        "aggregation_mode": aggregation_mode,
        "auroc": auroc,
        "pr_auc": pr_auc,
        "best_f1": best_f1,
        "pa_f1": pa_f1,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "specificity": specificity,
        "event_recall": event_recall,
        "event_pct": event_pct,
        "best_thresh": best_thresh,
        "fps": fps,
        "latency_ms": latency_ms,
        "total_steps": len(y_eval)
    }


def main():
    parser = argparse.ArgumentParser(description="Retrain SMD Channel-Aware v2 and audit aggregation")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--ffn_dim", type=int, default=512)
    parser.add_argument("--pos_weight", type=float, default=1.0)
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/smd_channel_aware_audit_v2.pt")
    parser.add_argument("--results_json", type=str, default="results/smd_channel_aware_audit_v2_eval.json")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_cache_path = "cache/smd_train_ultravetime_cache.pt"
    eval_cache_path = "cache/smd_eval_head_to_head_cache.pt"

    windows_dict, train_keys, val_keys = partition_cache(train_cache_path, val_ratio=0.1)
    train_dataset = UnifiedCachedDataset(windows_dict, train_keys, in_channels=38)
    val_dataset = UnifiedCachedDataset(windows_dict, val_keys, in_channels=38)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print("\nInstantiating Scaled UniversalTCNChannelAwareStudent (v2):")
    print(f"  Channels: 38 | Hidden Dim: {args.hidden_dim} | Heads: {args.num_heads} | FFN: {args.ffn_dim} | Pos Weight: {args.pos_weight}")

    model = UniversalTCNChannelAwareStudent(
        in_channels=38,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ffn_dim=args.ffn_dim,
        top_k=5
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.2f}M params - matches Pro scale)")

    criterion = SimplifiedDistillationLoss(alpha=0.1, gamma=2.0, pos_weight=args.pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
    best_val_f1 = -1.0

    print(f"\n--- Starting SMD Channel-Aware v2 Training ({args.epochs} Epochs) ---")
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot_loss = 0.0
        n_b = 0
        t0 = time.time()
        for batch in train_loader:
            bx = batch["x"].to(device)
            by = batch["y_dense"].to(device)
            bz = batch["z_teacher"].to(device)
            by_flag = batch["y_flag"].to(device)

            optimizer.zero_grad()
            out = model(bx)
            loss, _ = criterion(out["pred_dense"], by, student_feat=out["pred_feature"], teacher_feat=bz, y_flag=by_flag)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tot_loss += loss.item()
            n_b += 1

        scheduler.step()
        train_time = time.time() - t0
        avg_loss = tot_loss / max(1, n_b)

        val_metrics = evaluate_cache_win_metrics(model, val_loader, device)
        is_best = val_metrics["best_f1"] > best_val_f1
        if is_best:
            best_val_f1 = val_metrics["best_f1"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "model_config": {
                    "in_channels": 38,
                    "hidden_dim": args.hidden_dim,
                    "num_heads": args.num_heads,
                    "num_layers": args.num_layers,
                    "ffn_dim": args.ffn_dim,
                    "top_k": 5
                }
            }, args.checkpoint_path)

        flag = " [BEST SAVED]" if is_best else ""
        print(f"Epoch {epoch:02d}/{args.epochs:02d} ({train_time:.1f}s) | Loss: {avg_loss:.4f} | Val AUROC: {val_metrics['auroc']:.4f} | Val F1: {val_metrics['best_f1']:.4f} (Thresh: {val_metrics['best_threshold']:.3f}) | FPR: {val_metrics['fpr']*100:.2f}%{flag}")

    print(f"\nTraining Complete. Checkpoint saved to: {args.checkpoint_path}")

    # Load Best Checkpoint
    best_ckpt = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

    # 1. Evaluate on Head-to-Head Cache
    print(f"\n--- Evaluating on Strictly Held-Out Cache ({eval_cache_path}) ---")
    eval_c = torch.load(eval_cache_path, map_location="cpu", weights_only=False)
    eval_wins = eval_c["windows"] if isinstance(eval_c, dict) and "windows" in eval_c else eval_c
    eval_keys = sorted([k for k in eval_wins.keys() if isinstance(eval_wins[k], dict) and "x" in eval_wins[k]])
    eval_dataset = UnifiedCachedDataset(eval_wins, eval_keys, in_channels=38)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
    cache_metrics = evaluate_cache_win_metrics(model, eval_loader, device)
    print(f"Held-Out Cache AUROC: {cache_metrics['auroc']:.4f} | PR-AUC: {cache_metrics['prauc']:.4f} | Best F1: {cache_metrics['best_f1']:.4f} | FPR: {cache_metrics['fpr']*100:.2f}%")

    # 2. Continuous Streaming across all 28 entities (Partition 2 - last 40%)
    print("\n--- Preparing 28-Entity Continuous Streaming Stream (Partition 2) ---")
    smd_files = sorted(glob.glob("data/smd/test/*.txt"))
    smd_entities = []
    for tf in smd_files:
        lf = os.path.join("data/smd/test_label", os.path.basename(tf))
        raw_m = pd.read_csv(tf, header=None).values.astype(np.float32)
        raw_l = pd.read_csv(lf, header=None)[0].values.astype(np.float32)
        n = len(raw_m)
        a = int(n * 0.50)
        b = int(n * 0.60)
        base_m = np.mean(raw_m[:a], axis=0)
        base_s = np.maximum(np.std(raw_m[:a], axis=0), 0.10)
        norm_m = (raw_m - base_m) / base_s
        p2_d = norm_m[b:]
        p2_l = raw_l[b:]
        smd_entities.append((p2_d, p2_l))

    print(f"Loaded {len(smd_entities)} entities. Commencing multi-mode streaming audit...")

    stream_results = {}
    for mode in ["top5", "top1", "max"]:
        print(f"\nEvaluating Continuous Streaming under [{mode.upper()}] Aggregation...")
        res = run_continuous_streaming_eval(model, smd_entities, aggregation_mode=mode, device=device)
        stream_results[mode] = res
        print(f"  [{mode.upper()}] AUROC: {res['auroc']:.4f} | PR-AUC: {res['pr_auc']:.4f} | Point F1: {res['best_f1']:.4f} | PA-F1: {res['pa_f1']:.4f} | FPR: {res['fpr']*100:.2f}% | Thresh: {res['best_thresh']:.4f} | Event Rec: {res['event_recall']} ({res['event_pct']:.1f}%)")

    # Save summary json
    output_payload = {
        "model": "UniversalTCNChannelAwareStudent_v2",
        "checkpoint": args.checkpoint_path,
        "config": {
            "channels": 38,
            "hidden_dim": args.hidden_dim,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "ffn_dim": args.ffn_dim,
            "pos_weight": args.pos_weight,
            "parameters": n_params
        },
        "held_out_cache_metrics": cache_metrics,
        "continuous_streaming_metrics": stream_results
    }

    os.makedirs(os.path.dirname(args.results_json), exist_ok=True)
    with open(args.results_json, "w") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\nAudit complete. Full results saved to: {args.results_json}")


if __name__ == "__main__":
    main()
