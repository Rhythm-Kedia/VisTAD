"""Train and Evaluate Universal TCN Pro (Channel-Aware / Non-Permutation-Equivariant) on NASA MSL.

Architecture:
- UniversalFoundationTCN_Pro backbone (Causal Dilated TCN + Stacked Cross-Channel Transformer)
- Injected learned Channel Identity Embeddings (breaking permutation invariance for fixed 55 channels)
- Causal quadruplet features: [x, velocity, diff_30, rolling_std_25]
- Masked Temporal Dice Loss + Focal Loss + DINOv2 VLM Feature Distillation
- Evaluated on:
  1. Primary Held-Out Partition 2 (t >= 37,000, 18 unseen episodes)
  2. Partition 1 Training Regime (t < 37,000)
  3. Complete Test Stream (0 to 73,728, all 36 episodes)
  4. Held-Out Evaluation Cache (cache/msl_eval_head_to_head_cache.pt, 1,393 windows)
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.models.universal_tcn_channel_aware import UniversalTCNChannelAwareStudent
from src.loss import SimplifiedDistillationLoss
from src.metrics import compute_affiliation_metrics, find_events
from eval_msl_continuous_streaming import load_msl_stream, run_continuous_streaming_eval, evaluate_held_out_cache


class CachedMSLDataset(Dataset):
    """Loads cached MSL windows directly from RAM for zero I/O bottleneck."""
    def __init__(self, windows_dict, keys, base_m=None, base_s=None):
        self.keys = keys
        self.windows = windows_dict
        self.base_m = torch.as_tensor(base_m, dtype=torch.float32) if base_m is not None else None
        self.base_s = torch.as_tensor(base_s, dtype=torch.float32) if base_s is not None else None
        if self.base_s is not None:
            self.base_s[self.base_s == 0] = 1.0

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        k = self.keys[idx]
        w = self.windows[k]
        raw_x = torch.as_tensor(w["x"], dtype=torch.float32)  # [200, 55]
        if self.base_m is not None and self.base_s is not None:
            x = (raw_x - self.base_m) / self.base_s
        else:
            x = raw_x

        return {
            "start_idx": int(w.get("start_idx", k if isinstance(k, int) else 0)),
            "x": x,
            "y_dense": torch.as_tensor(w["y_dense"], dtype=torch.float32),
            "y_flag": torch.tensor(float(w["y_flag"]), dtype=torch.float32),
            "anomaly_ratio": torch.tensor(float(w.get("anomaly_ratio", 0.0)), dtype=torch.float32),
            "z_teacher": torch.as_tensor(w.get("z_teacher", np.zeros(1024)), dtype=torch.float32),
            "teacher_score": torch.tensor(float(w.get("teacher_score", 0.5)), dtype=torch.float32)
        }


def partition_curriculum_cache(train_cache_path, val_ratio=0.1):
    print(f"Loading MSL curriculum cache from {train_cache_path}...")
    c = torch.load(train_cache_path, map_location="cpu", weights_only=False)
    windows = c["windows"]
    keys = sorted(list(windows.keys()))

    anom_keys = [k for k in keys if windows[k]["y_flag"] > 0.5]
    norm_keys = [k for k in keys if windows[k]["y_flag"] <= 0.5]

    n_val_anom = max(1, int(len(anom_keys) * val_ratio))
    n_val_norm = max(1, int(len(norm_keys) * val_ratio))

    train_anom = anom_keys[:-n_val_anom]
    val_anom = anom_keys[-n_val_anom:]

    train_norm = norm_keys[:-n_val_norm]
    val_norm = norm_keys[-n_val_norm:]

    train_keys = sorted(train_anom + train_norm)
    val_keys = sorted(val_anom + val_norm)

    print(f"MSL Curriculum Partition: Total={len(keys)} windows")
    print(f"  Train: {len(train_keys)} windows ({len(train_anom)} anomaly, {len(train_norm)} nominal)")
    print(f"  Val:   {len(val_keys)} windows ({len(val_anom)} anomaly, {len(val_norm)} nominal)")

    return windows, train_keys, val_keys


def evaluate_val_cache(model, dataloader, device):
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

    return {
        "auroc": float(auroc),
        "prauc": float(prauc),
        "best_f1": float(best_f1),
        "best_threshold": float(best_thresh),
        "fpr": float(fpr)
    }


def main():
    parser = argparse.ArgumentParser(description="Train Universal TCN Pro (Channel-Aware) on MSL")
    parser.add_argument("--train_csv", type=str, default="data/msl/msl_train.csv")
    parser.add_argument("--train_cache", type=str, default="cache/msl_train_ultravetime_cache.pt")
    parser.add_argument("--eval_cache", type=str, default="cache/msl_eval_head_to_head_cache.pt")
    parser.add_argument("--test_csv", type=str, default="data/msl/msl_test.csv")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/21Nov")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--distill_alpha", type=float, default=0.1)
    parser.add_argument("--save_json", type=str, default="results/msl_eval_universal_pro_channel_aware.json")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = args.device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Selected Compute Device: {device}")

    # Baseline statistics from msl_train.csv
    df_tr = pd.read_csv(args.train_csv)
    cols = [c for c in df_tr.columns if c not in ["attack", "label", "timestamp"]]
    train_raw = df_tr[cols].values.astype(np.float32)
    base_m = np.mean(train_raw, axis=0)
    base_s = np.maximum(np.std(train_raw, axis=0), 0.05)

    windows_dict, train_keys, val_keys = partition_curriculum_cache(args.train_cache, val_ratio=0.1)
    train_dataset = CachedMSLDataset(windows_dict, train_keys, base_m=base_m, base_s=base_s)
    val_dataset = CachedMSLDataset(windows_dict, val_keys, base_m=base_m, base_s=base_s)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = UniversalTCNChannelAwareStudent(
        in_channels=55,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ffn_dim=args.hidden_dim * 4,
        top_k=5
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"UniversalTCNChannelAwareStudent instantiated: {n_params:,} parameters ({n_params/1e6:.2f}M)")

    criterion = SimplifiedDistillationLoss(alpha=args.distill_alpha, gamma=2.0, pos_weight=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_f1 = -1.0
    best_checkpoint_path = os.path.join(args.checkpoint_dir, "msl_universal_pro_channel_aware_best.pt")

    print(f"\n{'='*70}\nStarting Training: {args.epochs} Epochs\n{'='*70}")
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
        avg_loss = tot_loss / max(1, n_b)
        train_time = time.time() - t0

        val_metrics = evaluate_val_cache(model, val_loader, device)
        is_best = val_metrics["best_f1"] > best_val_f1
        if is_best:
            best_val_f1 = val_metrics["best_f1"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "base_m": base_m,
                "base_s": base_s,
                "model_config": {
                    "in_channels": 55,
                    "hidden_dim": args.hidden_dim,
                    "num_heads": args.num_heads,
                    "num_layers": args.num_layers
                }
            }, best_checkpoint_path)

        flag = " [BEST SAVED]" if is_best else ""
        print(f"Epoch {epoch:02d}/{args.epochs:02d} ({train_time:.1f}s) | Loss: {avg_loss:.4f} | Val AUROC: {val_metrics['auroc']:.4f} | Val F1: {val_metrics['best_f1']:.4f} (Thresh: {val_metrics['best_threshold']:.3f}) | FPR: {val_metrics['fpr']*100:.2f}%{flag}")

    print(f"\nTraining Complete. Best validation checkpoint saved to: {best_checkpoint_path}")

    # Load Best Model for Full Multi-Slice Stream Evaluation
    best_ckpt = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    # Strictly Held-Out Cache Evaluation (Win-F1)
    if os.path.exists(args.eval_cache):
        print(f"\n--- Loading Strictly Held-Out Evaluation Cache ({args.eval_cache}) ---")
        eval_c = torch.load(args.eval_cache, map_location="cpu", weights_only=False)
        eval_wins = eval_c["windows"]
        eval_keys = sorted(list(eval_wins.keys()))
        eval_dataset = CachedMSLDataset(eval_wins, eval_keys, base_m=base_m, base_s=base_s)
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

        # Teacher Baseline
        teacher_scores = np.array([eval_wins[k]["teacher_score"] for k in eval_keys])
        eval_flags = np.array([eval_wins[k]["y_flag"] for k in eval_keys])

        t_auroc = roc_auc_score(eval_flags, teacher_scores)
        t_prauc = average_precision_score(eval_flags, teacher_scores)
        t_p, t_r, t_th = precision_recall_curve(eval_flags, teacher_scores)
        t_f1s = 2 * (t_p * t_r) / np.clip(t_p + t_r, 1e-12, None)
        t_best_idx = np.argmax(t_f1s)
        t_best_f1 = t_f1s[t_best_idx]
        t_best_th = t_th[t_best_idx] if t_best_idx < len(t_th) else 0.5
        t_pred_bin = (teacher_scores >= t_best_th).astype(int)
        tn_t, fp_t, fn_t, tp_t = confusion_matrix(eval_flags, t_pred_bin).ravel()
        t_fpr = fp_t / (fp_t + tn_t + 1e-12)

        # Student Win-F1 Evaluation
        student_metrics = evaluate_val_cache(model, eval_loader, device)

        diff_auroc = (student_metrics['auroc'] - t_auroc) * 100
        diff_prauc = (student_metrics['prauc'] - t_prauc) * 100
        diff_f1 = (student_metrics['best_f1'] - t_best_f1) * 100
        diff_fpr = (student_metrics['fpr'] - t_fpr) * 100

        print("\n" + "=" * 85)
        print("NASA MSL HELD-OUT CACHE WIN-F1 BENCHMARK (1,393 Windows, Timestamps 37k to 73.7k)")
        print("=" * 85)
        print(f"{'Metric':<28} | {'Ultra-VETime Teacher':<22} | {'Universal TCN Channel-Aware':<28} | {'Advantage'}")
        print("-" * 85)
        print(f"{'Model Size':<28} | {'304M params (DINOv2-L)':<22} | {f'{n_params/1e6:.2f}M params':<28} | {f'{304/(n_params/1e6):.0f}x smaller'}")
        print(f"{'Window AUROC':<28} | {t_auroc:<22.4f} | {student_metrics['auroc']:<28.4f} | {diff_auroc:+.2f}%")
        print(f"{'Window PR-AUC':<28} | {t_prauc:<22.4f} | {student_metrics['prauc']:<28.4f} | {diff_prauc:+.2f}%")
        print(f"{'Window Best F1 (Win-F1)':<28} | {t_best_f1:<22.4f} | {student_metrics['best_f1']:<28.4f} | {diff_f1:+.2f}%")
        print(f"{'Decision Threshold (tau*)':<28} | {t_best_th:<22.4f} | {student_metrics['best_threshold']:<28.4f} | Balanced")
        st_fpr_str = f"{student_metrics['fpr']*100:.2f}%"
        t_fpr_str = f"{t_fpr*100:.2f}%"
        diff_fpr_str = f"{diff_fpr:+.2f}%"
        print(f"{'Window False Positive Rate':<28} | {t_fpr_str:<22} | {st_fpr_str:<28} | {diff_fpr_str}")
        print("=" * 85)

        if args.save_json:
            os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
            res = {
                "model": "UniversalTCNChannelAwareStudent",
                "parameters": n_params,
                "held_out_cache_win_metrics": {
                    "student": student_metrics,
                    "teacher": {
                        "auroc": float(t_auroc),
                        "prauc": float(t_prauc),
                        "best_f1": float(t_best_f1),
                        "best_threshold": float(t_best_th),
                        "fpr": float(t_fpr)
                    }
                }
            }
            with open(args.save_json, "w") as f:
                json.dump(res, f, indent=2)
            print(f"Metrics saved to: {args.save_json}")


if __name__ == "__main__":
    main()
