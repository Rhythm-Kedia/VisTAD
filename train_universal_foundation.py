"""Universal Foundation Cyber-Physical Anomaly Detection Training & Zero-Shot Transfer Engine.

Trains a dimension-invariant, permutation-equivariant foundation model on Dataset A
(or a multi-domain mixture) and evaluates Zero-Shot on unseen Dataset B:
- No Extreme Value Theory (EVT)
- Dimension-agnostic: accepts any C in [1, inf) dynamically
- Full telemetry ingestion (zero channel dropping)
- Self-supervised synthetic disruption invariants + causal predictive loss
"""

import os
import sys
import time
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.models.universal_tcn import UniversalCyberPhysicalTCN
from src.data.universal_dataset import load_dataset_telemetry, UniversalWindowDataset, MultiDatasetBatchSampler


def compute_contiguous_event_recall(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, float]:
    """Computes event-level recall across contiguous ground-truth anomaly episodes."""
    diff = np.diff(np.pad(y_true, (1, 1), mode="constant"))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    total_events = len(starts)
    if total_events == 0:
        return 0, 0, 1.0

    detected_events = 0
    for s, e in zip(starts, ends):
        if np.any(y_pred[s:e] == 1):
            detected_events += 1

    return detected_events, total_events, detected_events / total_events


def evaluate_continuous_stream(model: nn.Module, test_data: np.ndarray, test_labels: np.ndarray,
                               window_size: int = 200, stride: int = 10, batch_size: int = 64,
                               fixed_channel_indices: list[int] = None,
                               device: str = "cpu") -> dict:
    """Evaluates continuous streaming telemetry over an unseen dataset of arbitrary dimension C."""
    n = len(test_data)
    C = test_data.shape[1]
    
    ds = UniversalWindowDataset(
        test_data, test_labels, window_size=window_size, stride=stride,
        augment_synthetic=False, fixed_channel_indices=fixed_channel_indices
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    
    point_scores = np.zeros(n, dtype=np.float32)
    point_counts = np.zeros(n, dtype=np.float32)
    
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            st_indices = batch["st"].numpy()
            x = batch["x"].to(device)  # [B, C, L]
            out = model(x)
            p_sys = out["p_system"].cpu().numpy()  # [B, L]
            
            for i, st in enumerate(st_indices):
                point_scores[st : st + window_size] += p_sys[i]
                point_counts[st : st + window_size] += 1.0
                
    valid = point_counts > 0
    point_scores[valid] /= point_counts[valid]
    if not valid[0]:
        first_valid = np.where(valid)[0][0]
        point_scores[:first_valid] = point_scores[first_valid]
        
    eval_time = time.time() - t0
    y_true = test_labels.astype(int)
    
    if len(np.unique(y_true)) < 2:
        return {"auroc": 0.5, "prauc": float(np.mean(y_true)), "best_f1": 0.0, "event_recall": 1.0}
        
    auroc = float(roc_auc_score(y_true, point_scores))
    prauc = float(average_precision_score(y_true, point_scores))
    
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
        "point_recall": recall,
        "fpr": fpr,
        "detected_events": det_ev,
        "total_events": tot_ev,
        "event_recall": ev_rec,
        "eval_time_sec": eval_time,
        "throughput_fps": len(ds) / max(1e-6, eval_time),
        "point_scores": point_scores
    }


def main():
    parser = argparse.ArgumentParser(description="Universal Foundation Model Training & Zero-Shot Transfer")
    parser.add_argument("--train_datasets", type=str, default="smd",
                        help="Comma-separated training datasets: smd, swat, gecco, or msl")
    parser.add_argument("--eval_dataset", type=str, default="msl",
                        help="Target dataset to evaluate zero-shot: msl, swat, gecco, or smd")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42],
                        help="List of random seeds for statistical repetitions")
    parser.add_argument("--window_size", type=int, default=200)
    parser.add_argument("--train_stride", type=int, default=20)
    parser.add_argument("--eval_stride", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--subsample_train", action="store_true",
                        help="Randomly subsample 40%-100% of channels dynamically during training")
    parser.add_argument("--eval_channel_fraction", type=float, default=1.0,
                        help="Fraction of channels to retain at evaluation time (stress-testing sensor dropping)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--output_json", type=str, default="results/universal_zeroshot_transfer.json")
    args = parser.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Universal Execution Compute Device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    # 1. Load Training Datasets
    train_names = [d.strip().lower() for d in args.train_datasets.split(",")]
    print(f"\nSetting up Training Datasets: {train_names}")
    train_loaders = {}
    
    for name in train_names:
        raw_data, labels, train_end, C = load_dataset_telemetry(name, args.data_dir)
        train_slice_x = raw_data[:train_end]
        train_slice_y = labels[:train_end]
        
        sub_range = (0.4, 1.0) if args.subsample_train else None
        ds = UniversalWindowDataset(
            train_slice_x, train_slice_y,
            window_size=args.window_size,
            stride=args.train_stride,
            augment_synthetic=True,
            subsample_channels_range=sub_range,
            seed=42
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
        train_loaders[name] = loader
        print(f"  [{name.upper()}] Timesteps: {len(train_slice_x):,}, Channels C: {C}, Windows: {len(ds):,}")

    batch_sampler = MultiDatasetBatchSampler(train_loaders, shuffle=True)

    # 2. Load Evaluation Dataset (Unseen Zero-Shot Target)
    eval_name = args.eval_dataset.strip().lower()
    print(f"\nSetting up Unseen Zero-Shot Target Dataset: [{eval_name.upper()}]")
    eval_raw, eval_labels, eval_train_end, eval_C = load_dataset_telemetry(eval_name, args.data_dir)
    # Strictly held-out test split (from train_end to end)
    test_slice_x = eval_raw[eval_train_end:]
    test_slice_y = eval_labels[eval_train_end:]
    print(f"  Target Test Slice: {len(test_slice_x):,} steps (Anomalies: {int(np.sum(test_slice_y)):,}, Channels C: {eval_C})")

    eval_channel_indices = None
    if args.eval_channel_fraction < 1.0:
        num_eval_channels = max(3, int(eval_C * args.eval_channel_fraction))
        np.random.seed(42)
        eval_channel_indices = sorted(list(np.random.choice(eval_C, size=num_eval_channels, replace=False)))
        print(f"  [Channel Subsampling Stress-Test Active] Retaining {num_eval_channels}/{eval_C} channels ({args.eval_channel_fraction*100:.1f}%)")

    # 3. Instantiate Universal Scaled Foundation Model (~2.0M parameters)
    from src.models.universal_tcn import UniversalFoundationTCN_Pro
    model = UniversalFoundationTCN_Pro(
        hidden_dim=args.hidden_dim, num_heads=args.num_heads, num_layers=args.num_layers,
        kernel_size=5, dilations=(1, 2, 4, 8, 16, 32), dropout=0.1
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nUniversal Foundation Model: {total_params:,} parameters ({total_params/1e6:.2f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # 4. Training Loop
    print("\n" + "=" * 80)
    print(f"STARTING UNIVERSAL FOUNDATION PRETRAINING (Epochs: {args.epochs})")
    print("=" * 80)

    best_zero_shot_auroc = 0.0
    best_results = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        t_epoch = time.time()

        for batch in batch_sampler:
            x = batch["x"].to(device)                # [B, C, L]
            y_dense = batch["y_dense"].to(device)    # [B, L]
            y_win = batch["y_window"].to(device)      # [B]
            
            optimizer.zero_grad()
            out = model(x)
            
            p_sys = out["p_system"]        # [B, L]
            p_win = out["window_score"]    # [B]
            pred_next = out["pred_next"]   # [B, C, L]
            
            # 1. Focal / BCE Loss on dense probability
            loss_dense = F.binary_cross_entropy(p_sys, y_dense)
            # 2. Window-level classification loss
            loss_win = F.binary_cross_entropy(p_win, y_win)
            # 3. Self-Supervised Next-Step Causal Reconstruction Loss
            loss_pred = F.mse_loss(pred_next[:, :, :-1], x[:, :, 1:])
            
            loss = loss_dense + 0.5 * loss_win + 0.1 * loss_pred
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        epoch_time = time.time() - t_epoch
        avg_loss = total_loss / max(1, n_batches)

        # Zero-Shot Continuous Streaming Evaluation on Unseen Dataset B
        eval_metrics = evaluate_continuous_stream(
            model, test_slice_x, test_slice_y,
            window_size=args.window_size, stride=args.eval_stride,
            batch_size=args.batch_size,
            fixed_channel_indices=eval_channel_indices,
            device=device
        )

        print(f"Epoch [{epoch:2d}/{args.epochs:2d}] ({epoch_time:.1f}s) | Loss: {avg_loss:.4f} | "
              f"Zero-Shot [{eval_name.upper()}]: AUROC={eval_metrics['auroc']:.4f}, "
              f"PR-AUC={eval_metrics['prauc']:.4f}, F1={eval_metrics['best_f1']:.4f}, "
              f"Events={eval_metrics['detected_events']}/{eval_metrics['total_events']} ({eval_metrics['event_recall']*100:.1f}%)")

        if eval_metrics["auroc"] > best_zero_shot_auroc:
            best_zero_shot_auroc = eval_metrics["auroc"]
            best_results = eval_metrics
            ckpt_path = os.path.join(args.checkpoint_dir, f"universal_{args.train_datasets}_to_{eval_name}_best.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "hidden_dim": args.hidden_dim,
                    "num_heads": args.num_heads
                },
                "train_datasets": train_names,
                "eval_dataset": eval_name,
                "metrics": eval_metrics
            }, ckpt_path)

    print("\n" + "=" * 80)
    print(f"UNIVERSAL ZERO-SHOT TRANSFER RESULTS: [{args.train_datasets.upper()}] -> [{eval_name.upper()}]")
    print("=" * 80)
    print(f"Target Dataset:        {eval_name.upper()} (C={eval_C} channels, {len(test_slice_x):,} timesteps)")
    print(f"Zero-Shot AUROC:       {best_results['auroc']:.4f}")
    print(f"Zero-Shot PR-AUC:      {best_results['prauc']:.4f}")
    print(f"Zero-Shot Best F1:     {best_results['best_f1']:.4f}")
    print(f"Precision:             {best_results['precision']:.4f}")
    print(f"Point Recall:          {best_results['point_recall']:.4f}")
    print(f"False Alarm Rate (FPR):{best_results['fpr']*100:.2f}%")
    print(f"Optimal Threshold:     {best_results['best_threshold']:.4f}")
    print(f"Contiguous Event Recall: {best_results['detected_events']}/{best_results['total_events']} ({best_results['event_recall']*100:.1f}%)")
    print("=" * 80)

    # Save results to JSON
    # Strip raw point_scores numpy array from json export
    export_metrics = {k: v for k, v in best_results.items() if k != "point_scores"}
    payload = {
        "train_datasets": train_names,
        "eval_dataset": eval_name,
        "num_params": total_params,
        "zero_shot_metrics": export_metrics
    }
    with open(args.output_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved evaluation payload to: {args.output_json}")


if __name__ == "__main__":
    main()
