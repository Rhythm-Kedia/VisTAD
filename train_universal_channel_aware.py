"""Unified Training and Win-F1 Evaluation of Universal TCN Pro (Channel-Aware / Non-Permutation-Equivariant) on GECCO, SWaT, and SMD.

Architecture:
- UniversalFoundationTCN_Pro backbone (Causal Dilated TCN + Stacked Cross-Channel Transformer)
- Learned Channel Identity Embeddings (breaking permutation invariance for fixed C channels)
- Causal quadruplet features: [x, velocity, diff_30, rolling_std_25]
- Masked Temporal Dice Loss + Focal Loss + DINOv2 VLM Feature Distillation
- Win-F1 Evaluation on strictly held-out cache against Ultra-VETime Teacher (DINOv2-Large 304M)
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


DATASET_CONFIGS = {
    "gecco": {
        "channels": 9,
        "train_cache": "cache/gecco_train_ultravetime_cache.pt",
        "eval_cache": "cache/gecco_eval_head_to_head_cache.pt",
        "data_csv": "data/gecco/gecco_water.csv",
        "has_baseline_calc": True
    },
    "swat": {
        "channels": 51,
        "train_cache": "cache/swat_train_ultravetime_cache.pt",
        "eval_cache": "cache/swat_eval_head_to_head_cache.pt",
        "data_csv": "data/swat/SWaT_Dataset_Attack_v0.csv",
        "has_baseline_calc": False
    },
    "smd": {
        "channels": 38,
        "train_cache": "cache/smd_train_ultravetime_cache.pt",
        "eval_cache": "cache/smd_eval_head_to_head_cache.pt",
        "data_csv": None,
        "has_baseline_calc": False
    },
    "msl": {
        "channels": 55,
        "train_cache": "cache/msl_train_ultravetime_cache.pt",
        "eval_cache": "cache/msl_eval_head_to_head_cache.pt",
        "data_csv": "data/msl/msl_train.csv",
        "has_baseline_calc": True
    }
}


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
    """Zero-I/O cached dataset supporting flat and nested window dictionaries."""
    def __init__(self, windows_dict, keys, in_channels, base_m=None, base_s=None):
        self.keys = keys
        self.windows = windows_dict
        self.in_channels = in_channels
        self.base_m = torch.as_tensor(base_m, dtype=torch.float32) if base_m is not None else None
        self.base_s = torch.as_tensor(base_s, dtype=torch.float32) if base_s is not None else None
        if self.base_s is not None:
            self.base_s[self.base_s == 0] = 1.0

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        k = self.keys[idx]
        w = self.windows[k]
        raw_x = torch.as_tensor(w["x"], dtype=torch.float32)
        if raw_x.dim() == 2 and raw_x.shape[0] != self.in_channels and raw_x.shape[1] == self.in_channels:
            raw_x = raw_x.transpose(0, 1)  # [in_channels, 200]

        if self.base_m is not None and self.base_s is not None:
            m = self.base_m.view(-1, 1) if self.base_m.numel() == self.in_channels else self.base_m
            s = self.base_s.view(-1, 1) if self.base_s.numel() == self.in_channels else self.base_s
            x = (raw_x - m) / s
        else:
            x = raw_x

        y_dense = w.get("y_dense") if "y_dense" in w else w.get("y", np.zeros(200))
        y_flag = get_flag(w)
        z_teacher = w.get("z_teacher", np.zeros(1024))
        teacher_score = get_teacher_score(w)

        return {
            "x": x,
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


def train_and_eval_dataset(dataset_name: str, epochs: int = 8, batch_size: int = 32, lr: float = 1e-3,
                           checkpoint_dir: str = "checkpoints/21Nov", device: str = None):
    cfg = DATASET_CONFIGS[dataset_name.lower()]
    channels = cfg["channels"]
    train_cache = cfg["train_cache"]
    eval_cache = cfg["eval_cache"]

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(checkpoint_dir, f"{dataset_name.lower()}_universal_pro_channel_aware_best.pt")
    results_json = os.path.join("results", f"{dataset_name.lower()}_eval_universal_pro_channel_aware.json")

    device = device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"\n{'='*85}\nTRAINING UNIVERSAL TCN CHANNEL-AWARE ON {dataset_name.upper()} ({channels} CHANNELS)\n{'='*85}")
    print(f"Device: {device} | Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr}")

    # Baseline statistics
    base_m, base_s = None, None
    if cfg["has_baseline_calc"] and cfg["data_csv"] and os.path.exists(cfg["data_csv"]):
        df_raw = pd.read_csv(cfg["data_csv"])
        if dataset_name.lower() == "gecco":
            cols = ["Tp", "Cl", "pH", "Redox", "Leit", "Trueb", "Cl_2", "Fm", "Fm_2"]
            base_slice = df_raw.iloc[:16000][cols].values.astype(np.float32)
        else:
            cols = [c for c in df_raw.columns if c not in ["attack", "label", "timestamp"]]
            base_slice = df_raw[cols].values.astype(np.float32)
        base_m = np.mean(base_slice, axis=0)
        base_s = np.maximum(np.std(base_slice, axis=0), 0.05)
        print(f"Standardization parameters computed from nominal slice: shape {base_m.shape}")

    windows_dict, train_keys, val_keys = partition_cache(train_cache, val_ratio=0.1)
    train_dataset = UnifiedCachedDataset(windows_dict, train_keys, in_channels=channels, base_m=base_m, base_s=base_s)
    val_dataset = UnifiedCachedDataset(windows_dict, val_keys, in_channels=channels, base_m=base_m, base_s=base_s)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = UniversalTCNChannelAwareStudent(
        in_channels=channels,
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        ffn_dim=256,
        top_k=5
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {n_params:,} ({n_params/1e6:.2f}M params vs Teacher 304M)")

    criterion = SimplifiedDistillationLoss(alpha=0.1, gamma=2.0, pos_weight=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_f1 = -1.0
    print(f"\n--- Starting Physical Training ({epochs} Epochs) ---")
    for epoch in range(1, epochs + 1):
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

        val_metrics = evaluate_cache_win_metrics(model, val_loader, device)
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
                    "in_channels": channels,
                    "hidden_dim": 64,
                    "num_heads": 4,
                    "num_layers": 2
                }
            }, best_checkpoint_path)

        flag = " [BEST SAVED]" if is_best else ""
        print(f"Epoch {epoch:02d}/{epochs:02d} ({train_time:.1f}s) | Loss: {avg_loss:.4f} | Val AUROC: {val_metrics['auroc']:.4f} | Val F1: {val_metrics['best_f1']:.4f} (Thresh: {val_metrics['best_threshold']:.3f}) | FPR: {val_metrics['fpr']*100:.2f}%{flag}")

    print(f"\nTraining Complete. Best checkpoint saved to: {best_checkpoint_path}")

    # Evaluate Held-Out Cache
    if os.path.exists(eval_cache):
        print(f"\n--- Loading Strictly Held-Out Evaluation Cache ({eval_cache}) ---")
        eval_c = torch.load(eval_cache, map_location="cpu", weights_only=False)
        eval_wins = eval_c["windows"] if isinstance(eval_c, dict) and "windows" in eval_c else eval_c
        eval_keys = sorted([k for k in eval_wins.keys() if isinstance(eval_wins[k], dict) and "x" in eval_wins[k]])
        eval_dataset = UnifiedCachedDataset(eval_wins, eval_keys, in_channels=channels, base_m=base_m, base_s=base_s)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

        # Teacher Baseline
        teacher_scores = np.array([get_teacher_score(eval_wins[k]) for k in eval_keys])
        eval_flags = np.array([get_flag(eval_wins[k]) for k in eval_keys])

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

        # Student Win-F1
        best_ckpt = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])
        student_metrics = evaluate_cache_win_metrics(model, eval_loader, device)

        diff_auroc = (student_metrics['auroc'] - t_auroc) * 100
        diff_prauc = (student_metrics['prauc'] - t_prauc) * 100
        diff_f1 = (student_metrics['best_f1'] - t_best_f1) * 100
        diff_fpr = (student_metrics['fpr'] - t_fpr) * 100

        print("\n" + "=" * 90)
        print(f"{dataset_name.upper()} HELD-OUT CACHE WIN-F1 BENCHMARK ({len(eval_keys)} Windows)")
        print("=" * 90)
        print(f"{'Metric':<28} | {'Ultra-VETime Teacher':<22} | {'Universal TCN Channel-Aware':<28} | {'Advantage'}")
        print("-" * 90)
        print(f"{'Model Size':<28} | {'304M params (DINOv2-L)':<22} | {f'{n_params/1e6:.2f}M params':<28} | {f'{304/(n_params/1e6):.0f}x smaller'}")
        print(f"{'Window AUROC':<28} | {t_auroc:<22.4f} | {student_metrics['auroc']:<28.4f} | {diff_auroc:+.2f}%")
        print(f"{'Window PR-AUC':<28} | {t_prauc:<22.4f} | {student_metrics['prauc']:<28.4f} | {diff_prauc:+.2f}%")
        print(f"{'Window Best F1 (Win-F1)':<28} | {t_best_f1:<22.4f} | {student_metrics['best_f1']:<28.4f} | {diff_f1:+.2f}%")
        print(f"{'Decision Threshold (tau*)':<28} | {t_best_th:<22.4f} | {student_metrics['best_threshold']:<28.4f} | Balanced")
        st_fpr_str = f"{student_metrics['fpr']*100:.2f}%"
        t_fpr_str = f"{t_fpr*100:.2f}%"
        diff_fpr_str = f"{diff_fpr:+.2f}%"
        print(f"{'Window False Positive Rate':<28} | {t_fpr_str:<22} | {st_fpr_str:<28} | {diff_fpr_str}")
        print("=" * 90)

        os.makedirs(os.path.dirname(results_json) or ".", exist_ok=True)
        res = {
            "dataset": dataset_name.lower(),
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
        with open(results_json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"Metrics saved to: {results_json}")

        return res


def main():
    parser = argparse.ArgumentParser(description="Train Universal TCN Channel-Aware on specified datasets")
    parser.add_argument("--dataset", type=str, choices=["gecco", "swat", "smd", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/21Nov")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    targets = ["gecco", "swat", "smd"] if args.dataset == "all" else [args.dataset]
    all_results = {}
    for ds in targets:
        all_results[ds] = train_and_eval_dataset(
            ds, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            checkpoint_dir=args.checkpoint_dir, device=args.device
        )

    print("\n" + "=" * 90)
    print("ALL DATASETS TRAINING & WIN-F1 BENCHMARK COMPLETE")
    print("=" * 90)
    for ds, r in all_results.items():
        if r and "held_out_cache_win_metrics" in r:
            st = r["held_out_cache_win_metrics"]["student"]
            te = r["held_out_cache_win_metrics"]["teacher"]
            print(f"| {ds.upper():<8} | Student AUROC: {st['auroc']:.4f} (Teacher: {te['auroc']:.4f}) | Win-F1: {st['best_f1']:.4f} (Teacher: {te['best_f1']:.4f}) | FPR: {st['fpr']*100:.1f}% |")
    print("=" * 90)


if __name__ == "__main__":
    main()
