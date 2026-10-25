import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

for p in [CURRENT_DIR, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.dataset import TimeSeriesDataset, load_dataset_arrays
from src.models import build_student
from src.loss import MultiObjectiveKDLoss
from src.trainer import DistillationTrainer
from src.metrics import compute_point_metrics, roc_auc_score


def split_series(raw, labels, train_frac=0.5, val_frac=0.1):
    n = len(raw)
    a, b = int(n * train_frac), int(n * (train_frac + val_frac))
    return (raw[:a], labels[:a]), (raw[a:b], labels[a:b]), (raw[b:], labels[b:])


def main():
    parser = argparse.ArgumentParser(description="Train Universal Multi-Domain VisTAD Foundation Student")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--k", type=int, default=6, help="Top-K channels (default: 6)")
    parser.add_argument("--w", type=int, default=200, help="Window size")
    parser.add_argument("--stride", type=int, default=100, help="Sliding stride")
    parser.add_argument("--output_ckpt", type=str, default=os.path.join(ROOT_DIR, "checkpoints", "universal_multidomain_cnn_1d_best.pt"))
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print(f"  VisTAD Universal Cross-Domain Foundation Model Training")
    print(f"  Device: {device} | Channels K={args.k} | Window={args.w} | Epochs={args.epochs}")
    print("=" * 72)

    domains = [
        {"dataset": "smd", "entities": ["machine-1-1", "machine-2-1", "machine-3-2", "machine-1-5"]},
        {"dataset": "swat", "entities": ["swat_attack"]},
        {"dataset": "msl", "entities": ["msl_test"]},
        {"dataset": "gecco", "entities": ["gecco_water"]}
    ]

    train_datasets = []
    val_datasets = []
    
    total_cached_windows = 0
    data_dir = os.path.join(ROOT_DIR, "data")
    cache_dir = os.path.join(ROOT_DIR, "cache")

    for dom in domains:
        ds_name = dom["dataset"]
        for ent in dom["entities"]:
            try:
                raw_data, labels = load_dataset_arrays(ds_name, ent, data_dir)
                (tr_x, tr_y), (va_x, va_y), (te_x, te_y) = split_series(raw_data, labels)
                
                bs_steps = min(3000, len(tr_x))
                base_m = np.mean(tr_x[:bs_steps], axis=0)
                # Enforce minimum std floor to eliminate 1e6 division blowup on discrete channels
                base_s = np.maximum(np.std(tr_x[:bs_steps], axis=0), 0.05)

                cache_path = os.path.join(cache_dir, f"{ds_name}_{ent}_teacher_cache.pt")
                teacher_cache = None
                if os.path.exists(cache_path):
                    teacher_cache = torch.load(cache_path, weights_only=True)
                    usable = sum(1 for st in teacher_cache if st + args.w <= len(tr_x))
                    total_cached_windows += usable
                    print(f"  [{ds_name.upper():<5} - {ent:<12}] Train: {len(tr_x):<7} | Usable Teacher Anchors: {usable}")
                else:
                    print(f"  [{ds_name.upper():<5} - {ent:<12}] Train: {len(tr_x):<7} | No Cache")

                train_ds = TimeSeriesDataset(
                    data=tr_x, labels=tr_y, window_size=args.w, stride=args.stride,
                    base_mean=base_m, base_std=base_s,
                    filter_strategy="median_shift", k=args.k,
                    teacher_cache=teacher_cache
                )
                train_datasets.append(train_ds)

                if len(va_x) >= args.w:
                    val_ds = TimeSeriesDataset(
                        data=va_x, labels=va_y, window_size=args.w, stride=args.stride,
                        base_mean=base_m, base_std=base_s,
                        filter_strategy="median_shift", k=args.k,
                        teacher_cache=None
                    )
                    val_datasets.append(val_ds)

            except Exception as e:
                print(f"  Warning: Could not load {ds_name}/{ent}: {e}")

    pooled_train = ConcatDataset(train_datasets)
    pooled_val = ConcatDataset(val_datasets)
    print(f"\nPooled Training Windows: {len(pooled_train)} across 4 domains (Total Teacher Anchors: {total_cached_windows})")
    print(f"Pooled Validation Windows: {len(pooled_val)}")

    train_loader = DataLoader(pooled_train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(pooled_val, batch_size=args.batch_size, shuffle=False)

    student = build_student(
        name="cnn_1d",
        in_channels=args.k,
        hidden_dim=128,
        num_layers=3,
        dropout=0.1,
        teacher_feat_dim=2048
    ).to(device)

    loss_fn = MultiObjectiveKDLoss(
        w_sup=1.0,
        w_feat=0.5,
        w_resp=0.5,
        temperature=2.0,
        decision_threshold=0.60
    )

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    os.makedirs(os.path.dirname(args.output_ckpt), exist_ok=True)
    best_val_auc = -1.0

    print(f"\nStarting Cross-Domain Distillation Training ({args.epochs} epochs)...")
    for epoch in range(1, args.epochs + 1):
        student.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            x = batch["x"].to(device)
            outputs = student(x)
            loss, loss_comps = loss_fn(outputs, batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(1, n_batches)

        # Validation evaluation
        student.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                outputs = student(x)
                val_preds.extend(outputs["score_scalar"].cpu().numpy())
                val_targets.extend(batch["y_flag"].numpy())

        y_true = np.array(val_targets)
        y_score = np.array(val_preds)
        if len(np.unique(y_true)) > 1:
            val_auc = roc_auc_score(y_true, y_score)
        else:
            val_auc = 0.50

        improved = val_auc > best_val_auc
        if improved:
            best_val_auc = val_auc
            torch.save(student.state_dict(), args.output_ckpt)

        print(f"  Epoch [{epoch:2d}/{args.epochs:2d}] Loss: {avg_loss:.4f} | Val AUROC: {val_auc:.4f} {'*' if improved else ''}")

    print("=" * 72)
    print(f"Universal Cross-Domain Foundation Model successfully trained!")
    print(f"Best Val AUROC: {best_val_auc:.4f}")
    print(f"Saved Checkpoint: {args.output_ckpt} ({os.path.getsize(args.output_ckpt) / (1024*1024):.2f} MB)")
    print("=" * 72)


if __name__ == "__main__":
    main()
