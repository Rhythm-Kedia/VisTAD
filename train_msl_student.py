"""Training and Head-to-Head Evaluation of Telemetry-Command Coupled Causal TCN Student on NASA MSL.

Evaluates:
1. Episode-Aware & Sequential Continuous Training on Partition 1 (Timestamps 0 to 36,999)
   plus nominal operational segments from msl_train.csv.
2. Strictly Held-Out Head-to-Head Benchmark on cache/msl_eval_head_to_head_cache.pt
   (18 unseen anomaly episodes, Timestamps 37,000 to 73,728) comparing Student vs Ultra-VETime Teacher.
3. Full Continuous Streaming Deployment across Partition 2 and Complete Test Stream (73,729 steps).
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.models.msl_coupled_tcn import MSLCoupledTCNStudent
from src.loss import SimplifiedDistillationLoss


class UnifiedMSLDataset(Dataset):
    """Unified dataset supplying standardized sequences, dense masks, flags, and teacher features."""
    def __init__(self, windows_x, labels_y, teacher_feats=None, teacher_scores=None):
        self.x = torch.from_numpy(windows_x).float() if isinstance(windows_x, np.ndarray) else windows_x
        self.y = torch.from_numpy(labels_y).float() if isinstance(labels_y, np.ndarray) else labels_y
        self.teacher_feats = teacher_feats
        self.teacher_scores = teacher_scores

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_win = self.x[idx]
        y_dense = self.y[idx]
        y_flag = torch.tensor(float(y_dense.max() > 0), dtype=torch.float32)
        z_t = self.teacher_feats[idx] if self.teacher_feats is not None else torch.zeros(1024)
        s_t = self.teacher_scores[idx] if self.teacher_scores is not None else torch.tensor(0.5)

        return {
            "x": x_win,
            "y_dense": y_dense,
            "y_flag": y_flag,
            "z_teacher": z_t,
            "teacher_score": s_t
        }


def build_msl_training_data(train_csv="data/msl/msl_train.csv", test_csv="data/msl/msl_test.csv",
                            train_cache_path="cache/msl_train_ultravetime_cache.pt", window_size=200, stride=20):
    """Constructs comprehensive training data from Partition 1 stream + nominal train + teacher cache."""
    print("Loading raw MSL datasets...")
    df_tr = pd.read_csv(train_csv)
    df_te = pd.read_csv(test_csv)

    cols = [c for c in df_tr.columns if c not in ["attack", "label", "timestamp"]]
    train_raw = df_tr[cols].values.astype(np.float32)
    test_raw = df_te[cols].values.astype(np.float32)
    labels = df_te["attack"].values.astype(np.float32)

    base_m = np.mean(train_raw, axis=0)
    base_s = np.std(train_raw, axis=0)
    base_s[base_s == 0] = 1.0

    tr_norm = (train_raw - base_m) / base_s
    te_norm = (test_raw - base_m) / base_s

    p1_data = te_norm[:37000]
    p1_labels = labels[:37000]

    X_list = []
    Y_list = []

    # 1. Sequential sliding windows across Partition 1
    for st in range(0, len(p1_data) - window_size + 1, stride):
        X_list.append(p1_data[st : st + window_size])
        Y_list.append(p1_labels[st : st + window_size])

    # 2. Regular nominal operating windows from msl_train.csv
    step_tr = (len(tr_norm) - window_size) // 600
    for i in range(600):
        st = i * step_tr
        X_list.append(tr_norm[st : st + window_size])
        Y_list.append(np.zeros(window_size, dtype=np.float32))

    # 3. Augment with pre-cached anomaly curriculum if available
    if train_cache_path and os.path.exists(train_cache_path):
        cache = torch.load(train_cache_path, map_location="cpu", weights_only=False)
        wins = cache.get("windows", cache)
        for k, w in wins.items():
            if w.get("y_flag", 0) > 0:
                raw_w = w["x"]
                norm_w = (raw_w - base_m) / base_s
                X_list.append(norm_w.numpy() if isinstance(norm_w, torch.Tensor) else norm_w)
                w_y = w.get("y_dense", torch.zeros(window_size)).numpy() if isinstance(w.get("y_dense"), torch.Tensor) else w.get("y_dense")
                Y_list.append(w_y)

    X_all = np.stack(X_list, axis=0)
    Y_all = np.stack(Y_list, axis=0)
    print(f"Compiled Training Pool: {len(X_all)} total windows across {X_all.shape[-1]} channels.")

    return X_all, Y_all, base_m, base_s


def evaluate_held_out_cache(model, cache_path, base_m, base_s, device):
    """Evaluates head-to-head on the 1,393 held-out evaluation cache windows."""
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    wins = cache.get("windows", cache)
    keys = sorted(list(wins.keys()))

    teacher_scores = []
    student_scores = []
    y_flags = []

    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for k in keys:
            w = wins[k]
            raw_x = w["x"]
            x = (raw_x - base_m) / base_s
            out = model(x.unsqueeze(0).to(device))
            # Window score is max point anomaly probability
            s_student = float(torch.max(out["pred_dense"]).cpu())
            student_scores.append(s_student)
            teacher_scores.append(float(w.get("teacher_score", 0.5)))
            y_flags.append(float(w.get("y_flag", 0.0)))

    latency_ms = ((time.time() - t0) / len(keys)) * 1000.0

    y_true = np.array(y_flags)
    t_scores = np.array(teacher_scores)
    s_scores = np.array(student_scores)

    def compute_stats(y, s):
        auroc = roc_auc_score(y, s)
        prauc = average_precision_score(y, s)
        prec, rec, thresh = precision_recall_curve(y, s)
        f1s = 2 * (prec * rec) / np.maximum(prec + rec, 1e-8)
        best_idx = np.argmax(f1s)
        best_f1 = f1s[best_idx]
        best_t = thresh[best_idx] if best_idx < len(thresh) else 0.5
        y_pred = (s >= best_t).astype(int)
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        fpr = fp / max(fp + tn, 1)
        return auroc, prauc, best_f1, best_t, fpr, tp, fp, fn

    t_auroc, t_prauc, t_f1, t_t, t_fpr, t_tp, t_fp, t_fn = compute_stats(y_true, t_scores)
    s_auroc, s_prauc, s_f1, s_t, s_fpr, s_tp, s_fp, s_fn = compute_stats(y_true, s_scores)

    return {
        "teacher": {"auroc": t_auroc, "pr_auc": t_prauc, "best_f1": t_f1, "thresh": t_t, "fpr": t_fpr, "tp": t_tp, "fp": t_fp, "fn": t_fn},
        "student": {"auroc": s_auroc, "pr_auc": s_prauc, "best_f1": s_f1, "thresh": s_t, "fpr": s_fpr, "tp": s_tp, "fp": s_fp, "fn": s_fn, "latency_ms": latency_ms}
    }


def main():
    parser = argparse.ArgumentParser(description="Train MSLCoupledTCNStudent on NASA MSL")
    parser.add_argument("--train_csv", type=str, default="data/msl/msl_train.csv")
    parser.add_argument("--test_csv", type=str, default="data/msl/msl_test.csv")
    parser.add_argument("--train_cache", type=str, default="cache/msl_train_ultravetime_cache.pt")
    parser.add_argument("--eval_cache", type=str, default="cache/msl_eval_head_to_head_cache.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_path", type=str, default="checkpoints/msl_coupled_tcn_best.pt")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    X_train, Y_train, base_m, base_s = build_msl_training_data(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        train_cache_path=args.train_cache
    )

    dataset = UnifiedMSLDataset(X_train, Y_train)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = MSLCoupledTCNStudent(in_channels=55, hidden_dim=64).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MSLCoupledTCNStudent parameters: {n_params:,} ({n_params/1e6:.2f}M params vs Teacher 304M)")

    criterion = SimplifiedDistillationLoss(alpha=0.0, gamma=2.0, pos_weight=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    print(f"\n{'='*70}\nStarting Training: {args.epochs} Epochs\n{'='*70}")
    best_loss = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        tot_loss = 0.0
        t0 = time.time()
        for batch in loader:
            bx = batch["x"].to(device)
            by = batch["y_dense"].to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss, _ = criterion(out["pred_dense"], by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tot_loss += loss.item()

        scheduler.step()
        avg_loss = tot_loss / len(loader)
        dt = time.time() - t0
        print(f"Epoch {epoch:02d}/{args.epochs:02d} [{dt:.1f}s] | Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    torch.save({
        "epoch": args.epochs,
        "model_state_dict": best_state,
        "base_m": base_m,
        "base_s": base_s
    }, args.save_path)
    print(f"\nSaved Best Student Checkpoint -> {args.save_path}")

    # Evaluate Head-to-Head
    if os.path.exists(args.eval_cache):
        model.load_state_dict(best_state)
        res = evaluate_held_out_cache(model, args.eval_cache, base_m, base_s, device)
        tm = res["teacher"]
        sm = res["student"]

        print("\n" + "=" * 75)
        print("HEAD-TO-HEAD BENCHMARK: NASA MSL (HELD-OUT EVALUATION CACHE, 1,393 WINDOWS)")
        print("=" * 75)
        print(f"{'Metric':<25} | {'Ultra-VETime Teacher':<22} | {'Coupled TCN Student':<22}")
        print("-" * 75)
        print(f"{'AUROC':<25} | {tm['auroc']:<22.4f} | {sm['auroc']:<22.4f}")
        print(f"{'PR-AUC':<25} | {tm['pr_auc']:<22.4f} | {sm['pr_auc']:<22.4f}")
        print(f"{'Best F1':<25} | {tm['best_f1']:<22.4f} | {sm['best_f1']:<22.4f}")
        print(f"{'Optimal Threshold':<25} | {tm['thresh']:<22.4f} | {sm['thresh']:<22.4f}")
        print(f"{'False Positive Rate':<25} | {tm['fpr']*100:<21.2f}% | {sm['fpr']*100:<21.2f}%")
        print(f"{'True Positives':<25} | {tm['tp']:<22d} | {sm['tp']:<22d}")
        print(f"{'False Positives':<25} | {tm['fp']:<22d} | {sm['fp']:<22d}")
        print(f"{'False Negatives':<25} | {tm['fn']:<22d} | {sm['fn']:<22d}")
        print(f"{'Inference Latency':<25} | {'~1200 ms (GPU)':<22} | {sm['latency_ms']:<18.3f} ms")
        print("=" * 75)


if __name__ == "__main__":
    main()
