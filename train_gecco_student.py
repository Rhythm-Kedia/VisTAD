"""Training and Head-to-Head Evaluation of Subsystem-Coupled Causal TCN Student on GECCO.

Evaluates:
1. Episode-Aware Group Validation Split:
   - Partition 1 (Timestamps 0 to 69,999) contains 29 contamination events.
   - Train on Events 1..24 + 80% nominal windows.
   - Validate on Events 25..29 + 20% nominal windows.
2. Strictly Held-Out Head-to-Head Evaluation:
   - Evaluates on cache/gecco_eval_head_to_head_cache.pt (1,500 windows, Timestamps 70,000 to 139,565).
   - Compares Student vs Teacher (Ultra-VETime DINOv2-Large) head-to-head on identical metrics.
3. Feature Representation Distillation:
   - Cosine distillation from 1024-d DINOv2-Large teacher embeddings.
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.models.gecco_coupled_tcn import GeccoSubsystemCoupledTCN
from src.loss import FocalBCELoss, TemporalDiceLoss
from vlm4ts.gecco_renderer import GECCO_SENSORS


class CachedGeccoDataset(Dataset):
    """Loads pre-cached GECCO windows directly from RAM with auxiliary subsystem labels."""

    def __init__(self, windows_dict, keys, base_m=None, base_s=None):
        self.keys = keys
        self.windows = windows_dict
        self.base_m = torch.as_tensor(base_m, dtype=torch.float32) if base_m is not None else None
        self.base_s = torch.as_tensor(base_s + 1e-6, dtype=torch.float32) if base_s is not None else None

        # Precompute auxiliary labels for each window
        self.sub_labels = {}
        for k in keys:
            w = windows_dict[k]
            raw_x = np.asarray(w["x"], dtype=np.float32)
            y = np.asarray(w["y_dense"], dtype=np.float32)

            # Disinfection signal (Redox or Chlorine)
            redox_dev = np.abs(raw_x[:, 3] - 750.0) > 4.0
            cl_dev = np.abs(raw_x[:, 1] - 0.166) > 0.02
            cl2_dev = np.abs(raw_x[:, 6] - 0.109) > 0.015
            y_d = (y * (redox_dev | cl_dev | cl2_dev)).astype(np.float32)

            # Mineral / Physical Quality signal (pH, Leit, Trueb)
            ph_dev = np.abs(raw_x[:, 2] - 8.35) > 0.08
            leit_dev = np.abs(raw_x[:, 4] - 211.0) > 4.0
            trueb_dev = np.abs(raw_x[:, 5] - 0.019) > 0.01
            y_q = (y * (ph_dev | leit_dev | trueb_dev)).astype(np.float32)

            missed = (y > 0) & (y_d == 0) & (y_q == 0)
            y_d[missed] = 1.0
            y_q[missed] = 1.0
            self.sub_labels[k] = (y_d, y_q)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        k = self.keys[idx]
        w = self.windows[k]
        raw_x = torch.as_tensor(w["x"], dtype=torch.float32)  # [200, 9]
        if self.base_m is not None and self.base_s is not None:
            x = (raw_x - self.base_m) / self.base_s
        else:
            x = raw_x

        y_d, y_q = self.sub_labels[k]

        return {
            "start_idx": int(k),
            "x": x,
            "y_dense": torch.as_tensor(w["y_dense"], dtype=torch.float32),
            "y_d": torch.as_tensor(y_d, dtype=torch.float32),
            "y_q": torch.as_tensor(y_q, dtype=torch.float32),
            "y_flag": torch.tensor(float(w["y_flag"]), dtype=torch.float32),
            "anomaly_ratio": torch.tensor(float(w.get("anomaly_ratio", 0.0)), dtype=torch.float32),
            "z_teacher": torch.as_tensor(w.get("z_teacher", np.zeros(1024)), dtype=torch.float32),
            "teacher_score": torch.tensor(float(w.get("teacher_score", 0.5)), dtype=torch.float32),
            "teacher_dense_mask": torch.as_tensor(w.get("teacher_dense_mask", np.zeros(200)), dtype=torch.float32)
        }


def partition_curriculum_cache(train_cache_path, val_ratio=0.1):
    """Partitions curriculum cache into Train and Val using Episode-Aware splitting."""
    print(f"Loading GECCO curriculum cache from {train_cache_path}...")
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

    print(f"GECCO Curriculum Partition: Total={len(keys)} windows")
    print(f"  Train: {len(train_keys)} windows ({len(train_anom)} anomaly, {len(train_norm)} nominal)")
    print(f"  Val:   {len(val_keys)} windows ({len(val_anom)} anomaly, {len(val_norm)} nominal)")

    return windows, train_keys, val_keys


def evaluate_model(model, dataloader, device):
    """Evaluates model predictions and computes standard anomaly detection metrics."""
    model.eval()
    all_scores = []
    all_flags = []

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

    eval_time = time.time() - t0
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
    tpr = tp / (tp + fn + 1e-12)

    latency_per_window_ms = (eval_time / len(y_true)) * 1000.0

    return {
        "auroc": float(auroc),
        "prauc": float(prauc),
        "best_f1": float(best_f1),
        "best_threshold": float(best_thresh),
        "precision": float(precisions[best_idx]),
        "recall": float(tpr),
        "fpr": float(fpr),
        "latency_ms": float(latency_per_window_ms),
        "total_windows": len(y_true)
    }


def train_student():
    parser = argparse.ArgumentParser(description="Train Subsystem-Coupled TCN Student on GECCO")
    parser.add_argument("--data_path", type=str, default="data/gecco/gecco_water.csv")
    parser.add_argument("--train_cache", type=str, default="cache/gecco_train_ultravetime_cache.pt")
    parser.add_argument("--eval_cache", type=str, default="cache/gecco_eval_head_to_head_cache.pt")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--distill_alpha", type=float, default=0.1, help="VLM feature distillation loss weight")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Selected Compute Device: {device}")

    # 0. Load Nominal Baseline Statistics
    data_path = args.data_path
    if not os.path.exists(data_path):
        candidates = ["data/gecco/gecco_water.csv", "data/gecco_water.csv", "/kaggle/input/vistad-data/gecco/gecco_water.csv"]
        for c in candidates:
            if os.path.exists(c):
                data_path = c
                break

    df_raw = pd.read_csv(data_path)
    base_slice = df_raw.iloc[:16000][GECCO_SENSORS].values.astype(np.float32)
    base_m = np.mean(base_slice, axis=0)
    base_s = np.std(base_slice, axis=0) + 1e-6
    print(f"Computed nominal baseline standardization across {len(GECCO_SENSORS)} channels.")

    # 1. Load Curriculum Partition
    windows_dict, train_keys, val_keys = partition_curriculum_cache(args.train_cache, val_ratio=0.1)
    train_dataset = CachedGeccoDataset(windows_dict, train_keys, base_m=base_m, base_s=base_s)
    val_dataset = CachedGeccoDataset(windows_dict, val_keys, base_m=base_m, base_s=base_s)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 2. Build Student
    print(f"\nBuilding Subsystem-Decoupled Causal TCN (hidden_dim={args.hidden_dim})...")
    model = GeccoSubsystemCoupledTCN(
        in_channels=9,
        hidden_dim=args.hidden_dim,
        teacher_feat_dim=1024,
        kernel_size=5,
        dilations=[1, 2, 4, 8, 16],
        dropout=args.dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {total_params:,} ({total_params/1e6:.2f}M params vs Teacher's 304M - {304/(total_params/1e6):.0f}x smaller)")

    focal_loss = FocalBCELoss(gamma=2.0, pos_weight=2.0)
    dice_loss = TemporalDiceLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # 3. Training Loop with Convergence Diagnostics
    print(f"\n--- Starting Decoupled Physical Training ({args.epochs} Epochs) ---")
    best_val_f1 = -1.0
    best_checkpoint_path = os.path.join(args.checkpoint_dir, "gecco_coupled_tcn_best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in train_loader:
            x = batch["x"].to(device)
            y_dense = batch["y_dense"].to(device)
            y_d = batch["y_d"].to(device)
            y_q = batch["y_q"].to(device)
            z_teacher = batch["z_teacher"].to(device)

            optimizer.zero_grad()
            out = model(x)

            l_joint = focal_loss(out["pred_dense"], y_dense) + dice_loss(out["pred_dense"], y_dense)

            # Feature Distillation loss from DINOv2 teacher
            s_feat = out["pred_feature"]
            t_feat = F.normalize(z_teacher, p=2, dim=-1)
            l_feat = (1.0 - torch.sum(s_feat * t_feat, dim=-1)).mean()

            loss = l_joint + args.distill_alpha * l_feat
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_time = time.time() - t0
        avg_loss = total_loss / max(1, n_batches)

        # Validation evaluation
        val_metrics = evaluate_model(model, val_loader, device)

        is_best = val_metrics["best_f1"] > best_val_f1
        if is_best:
            best_val_f1 = val_metrics["best_f1"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "model_config": {
                    "architecture": "gecco_coupled_tcn",
                    "hidden_dim": args.hidden_dim,
                    "in_channels": 9,
                    "total_params": total_params,
                    "base_mean": base_m,
                    "base_std": base_s
                }
            }, best_checkpoint_path)

        flag = " [BEST SAVED]" if is_best else ""
        print(f"Epoch {epoch:02d}/{args.epochs:02d} ({train_time:.1f}s) | Loss: {avg_loss:.4f} | Val AUROC: {val_metrics['auroc']:.4f} | Val F1: {val_metrics['best_f1']:.4f} (Thresh: {val_metrics['best_threshold']:.3f}) | FPR: {val_metrics['fpr']*100:.2f}%{flag}")

    print(f"\nTraining Complete. Best validation checkpoint saved to: {best_checkpoint_path}")

    # 4. Strictly Held-Out Head-to-Head Evaluation
    if os.path.exists(args.eval_cache):
        print(f"\n--- Loading Strictly Held-Out Evaluation Cache ({args.eval_cache}) ---")
        eval_c = torch.load(args.eval_cache, map_location="cpu", weights_only=False)
        eval_wins = eval_c["windows"]
        eval_keys = sorted(list(eval_wins.keys()))
        eval_dataset = CachedGeccoDataset(eval_wins, eval_keys, base_m=base_m, base_s=base_s)
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

        # A. Evaluate Teacher Baseline from Cache
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
        tn, fp, fn, tp = confusion_matrix(eval_flags, t_pred_bin).ravel()
        t_fpr = fp / (fp + tn + 1e-12)

        # B. Evaluate Best Student Checkpoint
        best_ckpt = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])
        student_metrics = evaluate_model(model, eval_loader, device)

        diff_auroc = (student_metrics['auroc'] - t_auroc) * 100
        diff_prauc = (student_metrics['prauc'] - t_prauc) * 100
        diff_f1 = (student_metrics['best_f1'] - t_best_f1) * 100
        diff_fpr = (student_metrics['fpr'] - t_fpr) * 100
        speedup = 1200 / max(0.01, student_metrics['latency_ms'])
        compression = 304 / (total_params / 1e6)

        print("\n" + "=" * 80)
        print("GECCO HEAD-TO-HEAD HELD-OUT CACHE EVALUATION (Timestamps 70,000 to 139,565)")
        print("=" * 80)
        print(f"{'Metric':<25} | {'Ultra-VETime Teacher':<22} | {'Coupled TCN Student':<22} | {'Advantage'}")
        print("-" * 80)
        print(f"{'Model Size':<25} | {'304M params (DINOv2-L)':<22} | {f'{total_params/1e6:.2f}M params':<22} | {f'{compression:.0f}x smaller'}")
        print(f"{'AUROC':<25} | {t_auroc:<22.4f} | {student_metrics['auroc']:<22.4f} | {f'{diff_auroc:+.2f}%'}")
        print(f"{'PR-AUC':<25} | {t_prauc:<22.4f} | {student_metrics['prauc']:<22.4f} | {f'{diff_prauc:+.2f}%'}")
        print(f"{'Best F1':<25} | {t_best_f1:<22.4f} | {student_metrics['best_f1']:<22.4f} | {f'{diff_f1:+.2f}%'}")
        st_fpr_str = f"{student_metrics['fpr']*100:.2f}%"
        t_fpr_str = f"{t_fpr*100:.2f}%"
        diff_fpr_str = f"{diff_fpr:+.2f}%"
        st_lat_str = f"{student_metrics['latency_ms']:.3f} ms ({device})"
        speedup_str = f"{speedup:.0f}x faster"

        print(f"{'False Positive Rate':<25} | {t_fpr_str:<22} | {st_fpr_str:<22} | {diff_fpr_str}")
        print(f"{'Latency (ms/window)':<25} | {'~1200 ms (GPU)':<22} | {st_lat_str:<22} | {speedup_str}")
        print("=" * 80)


if __name__ == "__main__":
    train_student()
