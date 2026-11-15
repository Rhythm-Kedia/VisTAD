"""Multi-Machine Pooled Training of Subsystem-Coupled Causal TCN on Server Machine Dataset.

Trains a unified 38-channel foundation student model across all 28 server entities simultaneously:
- Training Split: Chronological first 50% across all 28 machines (354,201 steps, 7,923 ground truth anomalies)
- Validation Split: Chronological next 10% across all 28 machines (70,838 steps, 3,783 anomalies)
- Held-Out Test Split: Chronological last 40% across all 28 machines (283,381 steps, 17,738 anomalies - strictly preserved)
"""

import os
import sys
import time
import glob
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.models.smd_coupled_tcn import SMDSubsystemCoupledTCN
from src.loss import FocalBCELoss, TemporalDiceLoss


class SMDPooledDataset(Dataset):
    """Zero-I/O in-memory pooled dataset across all 28 SMD servers."""

    def __init__(self, window_records):
        self.records = window_records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        item = {
            "x": torch.as_tensor(rec["x"], dtype=torch.float32),          # [38, 200]
            "y_dense": torch.as_tensor(rec["y"], dtype=torch.float32),    # [200]
            "y_cpu": torch.as_tensor(rec["y_cpu"], dtype=torch.float32),  # [200]
            "y_mem": torch.as_tensor(rec["y_mem"], dtype=torch.float32),  # [200]
            "y_net": torch.as_tensor(rec["y_net"], dtype=torch.float32),  # [200]
            "y_disk": torch.as_tensor(rec["y_disk"], dtype=torch.float32),# [200]
            "y_flag": torch.tensor(float(rec["y_flag"]), dtype=torch.float32),
            "machine": rec["machine"]
        }
        if "z_teacher" in rec:
            item["z_teacher"] = torch.as_tensor(rec["z_teacher"], dtype=torch.float32)
            item["has_teacher"] = torch.tensor(1.0, dtype=torch.float32)
        else:
            item["z_teacher"] = torch.zeros(1024, dtype=torch.float32)
            item["has_teacher"] = torch.tensor(0.0, dtype=torch.float32)
        return item


def extract_smd_pooled_windows(data_dir="data/smd", window_size=200, teacher_cache_path=None):
    """Extracts balanced curriculum windows across the 50% train and 10% val splits of all 28 machines."""
    test_files = sorted(glob.glob(os.path.join(data_dir, "test", "*.txt")))
    print(f"Loading and extracting windows across all {len(test_files)} SMD machines...")

    teacher_cache = None
    if teacher_cache_path and os.path.exists(teacher_cache_path):
        print(f"Loading Ultra-VETime Teacher cache from: {teacher_cache_path}")
        teacher_cache = torch.load(teacher_cache_path, weights_only=False)
        print(f"Loaded {len(teacher_cache)} teacher cache records.")

    train_records = []
    val_records = []

    for tf in test_files:
        m_name = os.path.basename(tf).replace(".txt", "")
        lf = os.path.join(data_dir, "test_label", os.path.basename(tf))
        raw_data = pd.read_csv(tf, header=None).values.astype(np.float32)
        labels = pd.read_csv(lf, header=None)[0].values.astype(np.float32)

        n = len(raw_data)
        a = int(n * 0.50)  # train boundary
        b = int(n * 0.60)  # val boundary

        # Baseline stats for standardization from training split
        base_m = np.mean(raw_data[:a], axis=0)
        base_s = np.maximum(np.std(raw_data[:a], axis=0), 0.10)
        norm_data = (raw_data - base_m) / base_s

        # 1. Training Windows (0..a)
        tr_x = norm_data[:a]
        tr_y = labels[:a]

        diff_tr = np.diff(np.pad(tr_y, (1, 1), mode="constant"))
        tr_starts = np.where(diff_tr == 1)[0]
        tr_ends = np.where(diff_tr == -1)[0]

        anom_indices = []
        for s, e in zip(tr_starts, tr_ends):
            win_s = max(0, s - 180)
            win_e = min(a - window_size, e + 20)
            stride = 6 if (e - s) < 50 else 10
            for st in range(win_s, win_e, stride):
                anom_indices.append(st)

        nom_indices = list(range(0, a - window_size + 1, 50))
        nom_clean = [st for st in nom_indices if tr_y[st : st + window_size].max() == 0]

        tr_all = sorted(list(set(anom_indices + nom_clean)))
        for st in tr_all:
            w_x = tr_x[st : st + window_size].T  # [38, 200]
            w_y = tr_y[st : st + window_size]    # [200]

            # Compute auxiliary subsystem labels based on deviations
            z_w = np.abs(w_x)
            dev_cpu = (z_w[0:8] > 2.0).any(axis=0)
            dev_mem = (z_w[8:15] > 2.0).any(axis=0)
            dev_net = (z_w[15:27] > 2.0).any(axis=0)
            dev_disk = (z_w[27:38] > 2.0).any(axis=0)

            y_c = (w_y * dev_cpu).astype(np.float32)
            y_m = (w_y * dev_mem).astype(np.float32)
            y_n = (w_y * dev_net).astype(np.float32)
            y_d = (w_y * dev_disk).astype(np.float32)

            missed = (w_y > 0) & (y_c == 0) & (y_m == 0) & (y_n == 0) & (y_d == 0)
            y_c[missed] = 1.0
            y_m[missed] = 1.0
            y_n[missed] = 1.0
            y_d[missed] = 1.0

            key = f"{m_name}_{st}"
            rec = {
                "x": w_x, "y": w_y,
                "y_cpu": y_c, "y_mem": y_m, "y_net": y_n, "y_disk": y_d,
                "y_flag": float(w_y.max() > 0),
                "machine": m_name,
                "st": st
            }
            if teacher_cache is not None and key in teacher_cache:
                rec["z_teacher"] = teacher_cache[key]["z_teacher"]
            train_records.append(rec)

        # 2. Validation Windows (a..b)
        va_x = norm_data[a:b]
        va_y = labels[a:b]
        for st in range(0, len(va_x) - window_size + 1, 30):
            w_x = va_x[st : st + window_size].T
            w_y = va_y[st : st + window_size]
            val_records.append({
                "x": w_x, "y": w_y,
                "y_cpu": w_y, "y_mem": w_y, "y_net": w_y, "y_disk": w_y,
                "y_flag": float(w_y.max() > 0),
                "machine": m_name
            })

    print(f"Extraction complete across 28 machines:")
    print(f"  Train: {len(train_records):,} windows ({sum(r['y_flag'] for r in train_records):.0f} anomaly, {len(train_records)-sum(r['y_flag'] for r in train_records):.0f} nominal)")
    print(f"  Val:   {len(val_records):,} windows ({sum(r['y_flag'] for r in val_records):.0f} anomaly, {len(val_records)-sum(r['y_flag'] for r in val_records):.0f} nominal)")

    return train_records, val_records


def evaluate_pooled_model(model, dataloader, device):
    """Evaluates the model across the pooled validation set."""
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

    return {
        "auroc": float(auroc),
        "prauc": float(prauc),
        "best_f1": float(best_f1),
        "best_threshold": float(best_thresh),
        "precision": float(precisions[best_idx]),
        "recall": float(tpr),
        "fpr": float(fpr),
        "latency_ms": float((eval_time / len(y_true)) * 1000.0),
        "total_windows": len(y_true)
    }


def train_smd_pooled():
    parser = argparse.ArgumentParser(description="Train Pooled 38-Channel Causal TCN on SMD")
    parser.add_argument("--data_dir", type=str, default="data/smd")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--teacher_cache", type=str, default=None, help="Path to Ultra-VETime teacher cache")
    parser.add_argument("--w_distill", type=float, default=0.1, help="Distillation feature loss weight")
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

    train_recs, val_recs = extract_smd_pooled_windows(args.data_dir, teacher_cache_path=args.teacher_cache)
    train_ds = SMDPooledDataset(train_recs)
    val_ds = SMDPooledDataset(val_recs)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    print(f"\nBuilding 38-Channel Multi-Subsystem Causal TCN (hidden_dim={args.hidden_dim})...")
    model = SMDSubsystemCoupledTCN(
        in_channels=38,
        hidden_dim=args.hidden_dim,
        teacher_feat_dim=1024,
        kernel_size=5,
        dilations=[1, 2, 4, 8, 16],
        dropout=args.dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {total_params:,} ({total_params/1e6:.2f}M parameters)")

    focal_loss = FocalBCELoss(gamma=2.0, pos_weight=2.0)
    dice_loss = TemporalDiceLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    print(f"\n--- Starting Multi-Machine Pooled Training ({args.epochs} Epochs) ---")
    best_val_f1 = -1.0
    best_checkpoint_path = os.path.join(args.checkpoint_dir, "smd_coupled_tcn_best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in train_loader:
            x = batch["x"].to(device)
            y_dense = batch["y_dense"].to(device)
            y_cpu = batch["y_cpu"].to(device)
            y_mem = batch["y_mem"].to(device)
            y_net = batch["y_net"].to(device)
            y_disk = batch["y_disk"].to(device)

            optimizer.zero_grad()
            out = model(x)

            l_joint = focal_loss(out["pred_dense"], y_dense) + dice_loss(out["pred_dense"], y_dense)
            l_c = focal_loss(out["p_cpu"], y_cpu)
            l_m = focal_loss(out["p_mem"], y_mem)
            l_n = focal_loss(out["p_net"], y_net)
            l_d = focal_loss(out["p_disk"], y_disk)

            loss = l_joint + 0.25 * (l_c + l_m + l_n + l_d)

            if batch["has_teacher"].sum() > 0:
                mask = batch["has_teacher"] > 0
                z_t = batch["z_teacher"].to(device)
                l_distill = nn.functional.mse_loss(out["student_feat"][mask], z_t[mask])
                loss = loss + args.w_distill * l_distill

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_time = time.time() - t0
        avg_loss = total_loss / max(1, n_batches)

        val_metrics = evaluate_pooled_model(model, val_loader, device)
        is_best = val_metrics["best_f1"] > best_val_f1
        if is_best:
            best_val_f1 = val_metrics["best_f1"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "model_config": {
                    "architecture": "smd_coupled_tcn",
                    "hidden_dim": args.hidden_dim,
                    "in_channels": 38,
                    "total_params": total_params,
                    "num_machines": 28
                }
            }, best_checkpoint_path)

        flag = " [BEST SAVED]" if is_best else ""
        print(f"Epoch {epoch:02d}/{args.epochs:02d} ({train_time:.1f}s) | Loss: {avg_loss:.4f} | Val AUROC: {val_metrics['auroc']:.4f} | Val F1: {val_metrics['best_f1']:.4f} (Thresh: {val_metrics['best_threshold']:.3f}) | FPR: {val_metrics['fpr']*100:.2f}%{flag}")

    print(f"\nPooled Training Complete. Best validation checkpoint saved to: {best_checkpoint_path}")


if __name__ == "__main__":
    train_smd_pooled()
