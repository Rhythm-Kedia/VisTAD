"""Run Grand Multi-Task Foundation Model Training-Time Top-K Channel Selection Ablation.

Trains Grand Foundation Models (SWaT + SMD + GECCO + MSL) from scratch across:
    K_train in {1, 3, 5, 10, "all"}
with:
- Causal prefix buffer prefix_len = 250 (L_RF - 1)
- Permutation-invariant randomized channel dropping (40% - 100%)
- Channel permutation / shuffling per batch
- Multi-dataset batch sampling
- Full epoch-by-epoch loss tracking (total, dense, win, pred, feat, distill)
- Continuous streaming evaluation on held-out test splits for all 4 benchmarks
"""

import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.models.universal_tcn import UniversalFoundationTCN_Pro
from src.data.universal_dataset import (
    load_dataset_telemetry,
    UniversalWindowDataset,
    MultiDatasetBatchSampler,
    UniversalCollate
)
from train_universal_foundation import evaluate_continuous_stream


def train_single_grand_model(k_val: int | str, train_loaders: dict, test_slices: dict,
                             device: str, epochs: int = 10, lr: float = 1e-3,
                             prefix_len: int = 250, window_size: int = 200,
                             distill_feat_weight: float = 0.5, distill_score_weight: float = 0.2,
                             use_amp: bool = True, checkpoint_dir: str = "checkpoints") -> dict:
    """Trains a single Grand Model with a specific training-time Top-K disjunction value."""
    print("\n" + "=" * 90)
    print(f"TRAINING GRAND FOUNDATION MODEL: K_train = {k_val} (Epochs: {epochs}, Prefix: {prefix_len})")
    print("=" * 90)

    model = UniversalFoundationTCN_Pro(
        hidden_dim=128, num_heads=8, num_layers=2,
        kernel_size=5, dilations=(1, 2, 4, 8, 16, 32), dropout=0.1,
        top_k=k_val
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Instantiated UniversalFoundationTCN_Pro: {total_params:,} parameters ({total_params/1e6:.2f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    if use_amp and (device.startswith("cuda") if isinstance(device, str) else device.type == "cuda"):
        try:
            scaler = torch.amp.GradScaler('cuda')
        except Exception:
            scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    batch_sampler = MultiDatasetBatchSampler(train_loaders, shuffle=True)
    epoch_history = []
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        tot_dense = 0.0
        tot_win = 0.0
        tot_pred = 0.0
        tot_feat = 0.0
        tot_distill = 0.0
        n_batches = 0
        t_epoch = time.time()

        for batch in batch_sampler:
            x = batch["x"].to(device)
            y_dense = batch["y_dense"].to(device)
            y_win = batch["y_window"].to(device)
            b_prefix = batch["prefix_len"]

            optimizer.zero_grad()

            if scaler is not None:
                try:
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        out = model(x, prefix_len=b_prefix)
                except Exception:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        out = model(x, prefix_len=b_prefix)

                p_sys = torch.clamp(out["p_system"].float(), 1e-5, 1.0 - 1e-5)
                p_win = torch.clamp(out["window_score"].float(), 1e-5, 1.0 - 1e-5)
                pred_next = out["pred_next"].float()
                z_student = out["z_student"].float()

                if b_prefix > 0:
                    loss_dense = F.binary_cross_entropy(p_sys[:, b_prefix:], y_dense[:, b_prefix:].float())
                    loss_pred = F.mse_loss(pred_next[:, :, b_prefix:-1], x[:, :, b_prefix+1:].float())
                else:
                    loss_dense = F.binary_cross_entropy(p_sys, y_dense.float())
                    loss_pred = F.mse_loss(pred_next[:, :, :-1], x[:, :, 1:].float())
                loss_win = F.binary_cross_entropy(p_win, y_win.float())

                has_teacher = batch["has_teacher"].to(device)
                if has_teacher.sum() > 0:
                    z_teacher = batch["z_teacher"].to(device).float()
                    t_score = batch["teacher_score"].to(device).float()
                    cos_sim = torch.sum(z_student * z_teacher, dim=-1)
                    loss_feat = torch.sum((1.0 - cos_sim) * has_teacher) / torch.clamp(has_teacher.sum(), min=1.0)
                    loss_distill = torch.sum(((p_win - t_score) ** 2) * has_teacher) / torch.clamp(has_teacher.sum(), min=1.0)
                else:
                    loss_feat = torch.tensor(0.0, device=device)
                    loss_distill = torch.tensor(0.0, device=device)

                loss = loss_dense + 0.5 * loss_win + 0.1 * loss_pred + distill_feat_weight * loss_feat + distill_score_weight * loss_distill

                if not torch.isfinite(loss):
                    optimizer.zero_grad()
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(x, prefix_len=b_prefix)
                p_sys = torch.clamp(out["p_system"], 1e-5, 1.0 - 1e-5)
                p_win = torch.clamp(out["window_score"], 1e-5, 1.0 - 1e-5)
                pred_next = out["pred_next"]
                z_student = out["z_student"]

                if b_prefix > 0:
                    loss_dense = F.binary_cross_entropy(p_sys[:, b_prefix:], y_dense[:, b_prefix:])
                    loss_pred = F.mse_loss(pred_next[:, :, b_prefix:-1], x[:, :, b_prefix+1:])
                else:
                    loss_dense = F.binary_cross_entropy(p_sys, y_dense)
                    loss_pred = F.mse_loss(pred_next[:, :, :-1], x[:, :, 1:])
                loss_win = F.binary_cross_entropy(p_win, y_win)

                has_teacher = batch["has_teacher"].to(device)
                if has_teacher.sum() > 0:
                    z_teacher = batch["z_teacher"].to(device)
                    t_score = batch["teacher_score"].to(device)
                    cos_sim = torch.sum(z_student * z_teacher, dim=-1)
                    loss_feat = torch.sum((1.0 - cos_sim) * has_teacher) / torch.clamp(has_teacher.sum(), min=1.0)
                    loss_distill = torch.sum(((p_win - t_score) ** 2) * has_teacher) / torch.clamp(has_teacher.sum(), min=1.0)
                else:
                    loss_feat = torch.tensor(0.0, device=device)
                    loss_distill = torch.tensor(0.0, device=device)

                loss = loss_dense + 0.5 * loss_win + 0.1 * loss_pred + distill_feat_weight * loss_feat + distill_score_weight * loss_distill

                if not torch.isfinite(loss):
                    optimizer.zero_grad()
                    continue

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            tot_dense += loss_dense.item()
            tot_win += loss_win.item()
            tot_pred += loss_pred.item()
            tot_feat += loss_feat.item()
            tot_distill += loss_distill.item()
            n_batches += 1

        scheduler.step()
        epoch_dur = time.time() - t_epoch
        denom = max(1, n_batches)
        record = {
            "epoch": epoch,
            "train_loss": total_loss / denom,
            "loss_dense": tot_dense / denom,
            "loss_win": tot_win / denom,
            "loss_pred": tot_pred / denom,
            "loss_feat": tot_feat / denom,
            "loss_distill": tot_distill / denom,
            "epoch_time_sec": epoch_dur
        }
        epoch_history.append(record)

        print(f"  [K_train={k_val}] Epoch [{epoch:2d}/{epochs:2d}] ({epoch_dur:.1f}s) | "
              f"Loss: {record['train_loss']:.4f} (dense:{record['loss_dense']:.4f}, win:{record['loss_win']:.4f}, feat:{record['loss_feat']:.4f})")

    total_train_time = time.time() - t_start
    print(f"Finished Training K_train = {k_val} in {total_train_time/60:.2f} mins.")

    # Save checkpoint
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"grand_model_ktrain_{k_val}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "k_val": k_val,
        "prefix_len": prefix_len,
        "window_size": window_size,
        "epoch_history": epoch_history,
        "train_time_sec": total_train_time
    }, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")

    # Evaluate on held-out test slices of all 4 datasets
    print(f"\nEvaluating K_train = {k_val} on all 4 benchmark test slices (prefix_len={prefix_len})...")
    test_eval_results = {}
    for d_name, d_slice in test_slices.items():
        t_eval_start = time.time()
        metrics = evaluate_continuous_stream(
            model, d_slice["x"], d_slice["y"],
            window_size=window_size, prefix_len=prefix_len, stride=10,
            batch_size=64, device=device
        )
        t_eval_end = time.time()
        # Remove point_scores array for clean json
        clean_m = {k: v for k, v in metrics.items() if k != "point_scores"}
        clean_m["eval_time_sec"] = t_eval_end - t_eval_start
        test_eval_results[d_name] = clean_m
        print(f"  [{d_name.upper():5s}] AUROC: {clean_m['auroc']:.4f} | PR-AUC: {clean_m['prauc']:.4f} | "
              f"F1: {clean_m['best_f1']:.4f} | FPR: {clean_m['fpr']*100:.2f}% | "
              f"Events: {clean_m['detected_events']}/{clean_m['total_events']} ({clean_m['event_recall']*100:.1f}%)")

    del model, optimizer, scaler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "k_train": k_val,
        "train_time_sec": total_train_time,
        "epoch_history": epoch_history,
        "test_eval_results": test_eval_results,
        "checkpoint_path": ckpt_path
    }


def main():
    parser = argparse.ArgumentParser(description="Grand Model Training-Time Top-K Ablation Suite")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument("--k_vals", nargs="+", type=str, default=["1", "3", "5", "10", "all"],
                        help="Top-K channel values to ablate during training")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--prefix_len", type=int, default=250)
    parser.add_argument("--window_size", type=int, default=200)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--output_json", type=str, default="results/ablations/top_k_training_ablation_grand.json")
    parser.add_argument("--output_report", type=str, default="results/ablations/TOP_K_TRAINING_ABLATION_REPORT.md")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Top-K Training Ablation Engine Running on: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    dataset_names = ["swat", "smd", "gecco", "msl"]
    print("\n1. Ingesting and Preparing Training Datasets & Test Slices...")
    train_loaders = {}
    test_slices = {}

    for name in dataset_names:
        raw_data, labels, train_end, C = load_dataset_telemetry(name, args.data_dir)
        train_slice_x = raw_data[:train_end]
        train_slice_y = labels[:train_end]
        test_slice_x = raw_data[train_end:]
        test_slice_y = labels[train_end:]

        test_slices[name] = {"x": test_slice_x, "y": test_slice_y, "channels": C}

        # Ingest offline VLM teacher cache if present
        cache_cand = [
            os.path.join(args.cache_dir, f"{name}_train_ultravetime_cache.pt"),
            os.path.join("cache", f"{name}_train_ultravetime_cache.pt"),
            os.path.join("cache", f"{name}_eval_head_to_head_cache.pt")
        ]
        t_cache = None
        for c in cache_cand:
            if os.path.exists(c):
                try:
                    t_cache = torch.load(c, map_location="cpu")
                    n_w = len(t_cache.get("windows", t_cache))
                    print(f"  [{name.upper()}] Loaded VLM Teacher Cache: {c} ({n_w:,} cached windows)")
                    break
                except Exception as e:
                    print(f"  [{name.upper()}] Cache load warning: {e}")

        # Training collator with dynamic subsampling (40%-100%) and batch-level channel permutation
        collate_fn = UniversalCollate(subsample_range=(0.4, 1.0), shuffle_channels=True, seed=42 + len(train_loaders))
        ds = UniversalWindowDataset(
            train_slice_x, train_slice_y,
            window_size=args.window_size,
            prefix_len=args.prefix_len,
            stride=20,
            augment_synthetic=True,
            teacher_cache=t_cache,
            seed=42
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
        train_loaders[name] = loader
        print(f"  [{name.upper():5s}] Train Windows: {len(ds):,}, Test Steps: {len(test_slice_x):,}, Channels: {C}")

    # Parse K values
    parsed_k_vals = []
    for kv in args.k_vals:
        s = kv.strip().lower()
        if s == "all":
            parsed_k_vals.append("all")
        else:
            try:
                parsed_k_vals.append(int(s))
            except ValueError:
                pass

    print(f"\n2. Executing Training-Time Top-K Ablation for K in {parsed_k_vals}...")
    ablation_payload = {
        "k_vals": parsed_k_vals,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "prefix_len": args.prefix_len,
        "window_size": args.window_size,
        "models": {}
    }

    t0_suite = time.time()
    for kv in parsed_k_vals:
        model_result = train_single_grand_model(
            k_val=kv,
            train_loaders=train_loaders,
            test_slices=test_slices,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            prefix_len=args.prefix_len,
            window_size=args.window_size,
            checkpoint_dir=args.checkpoint_dir
        )
        ablation_payload["models"][f"ktrain_{kv}"] = model_result

    total_suite_time = time.time() - t0_suite
    ablation_payload["total_suite_time_sec"] = total_suite_time
    print(f"\nCompleted Full Training Ablation Suite in {total_suite_time/60:.2f} mins.")

    # Save JSON Payload
    with open(args.output_json, "w") as f:
        json.dump(ablation_payload, f, indent=2)
    print(f"Saved full JSON ablation payload to: {args.output_json}")

    # Generate Structured Markdown Report
    lines = [
        "# Grand Foundation Model: Training-Time Top-K Channel Selection Ablation Report",
        "",
        f"- **Datasets Evaluated**: SWaT (C=51), SMD (C=38), GECCO (C=9), NASA MSL (C=55)",
        f"- **Receptive Field Prefix**: $L_{{\\text{{prefix}}}} = {args.prefix_len}$ steps",
        f"- **Training Epochs per Model**: {args.epochs}",
        f"- **Total Suite Execution Time**: {total_suite_time/60:.2f} minutes",
        "",
        "## 1. Cross-Dataset Metric Performance Table (<DATASET>_<METRIC>)",
        "",
        "| $K_{\\text{train}}$ | GECCO AUROC | GECCO F1 | SWaT AUROC | SWaT PR-AUC | SMD AUROC | SMD F1 | MSL AUROC | MSL F1 | MSL Event Rec |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for kv in parsed_k_vals:
        m = ablation_payload["models"][f"ktrain_{kv}"]["test_eval_results"]
        g_auc = m["gecco"]["auroc"]
        g_f1 = m["gecco"]["best_f1"]
        sw_auc = m["swat"]["auroc"]
        sw_pr = m["swat"]["prauc"]
        smd_auc = m["smd"]["auroc"]
        smd_f1 = m["smd"]["best_f1"]
        msl_auc = m["msl"]["auroc"]
        msl_f1 = m["msl"]["best_f1"]
        msl_ev = m["msl"]["event_recall"] * 100

        lines.append(f"| **K={kv}** | {g_auc:.4f} | {g_f1:.4f} | {sw_auc:.4f} | {sw_pr:.4f} | {smd_auc:.4f} | {smd_f1:.4f} | {msl_auc:.4f} | {msl_f1:.4f} | {msl_ev:.1f}% |")

    lines.extend([
        "",
        "## 2. Epoch-by-Epoch Training & Validation Loss Trajectories",
        "",
        "| Model | Epoch | Total Loss | Dense BCE | Window BCE | 1-Step Pred | VLM Feat | Time (s) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ])

    for kv in parsed_k_vals:
        history = ablation_payload["models"][f"ktrain_{kv}"]["epoch_history"]
        for rec in history:
            lines.append(f"| K={kv} | {rec['epoch']} | {rec['train_loss']:.4f} | {rec['loss_dense']:.4f} | {rec['loss_win']:.4f} | {rec['loss_pred']:.4f} | {rec['loss_feat']:.4f} | {rec['epoch_time_sec']:.1f} |")

    with open(args.output_report, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved Markdown summary report to: {args.output_report}")


if __name__ == "__main__":
    main()
