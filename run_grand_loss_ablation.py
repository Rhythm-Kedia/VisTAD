"""Grand Foundation Model Loss Formulation Ablation Suite across Cyber-Physical Benchmarks.

Ablates the loss function on the Grand Foundation Model (SWaT + SMD + GECCO + MSL)
using ALL channels (K = "all") with receptive field prefix buffer L_prefix = 250:
    1. pure_bce:       L = BCE(p_sys, y)
    2. focal:          L = FocalBCE(p_sys, y)
    3. focal_dice:     L = FocalBCE(p_sys, y) + TemporalDice(p_sys, y)
    4. unified_distill L = (FocalBCE + TemporalDice) + 0.10 * CosineDist(z_student, z_teacher)
    5. bce_feat:       L = BCE(p_sys, y) + 0.10 * CosineDist(z_student, z_teacher)

Evaluates on strictly held-out continuous streaming test splits for SWaT, SMD, GECCO, and MSL.
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
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.models.universal_tcn import UniversalFoundationTCN_Pro
from src.loss import UnifiedFoundationLoss
from src.data.universal_dataset import (
    load_dataset_telemetry,
    UniversalWindowDataset,
    MultiDatasetBatchSampler,
    UniversalCollate
)
from train_universal_foundation import evaluate_continuous_stream


def train_single_loss_variant(variant_name: str, train_loaders: dict, test_slices: dict,
                              device: str, epochs: int = 5, lr: float = 1.5e-3,
                              prefix_len: int = 250, window_size: int = 200,
                              eval_stride: int = 20, eval_batch_size: int = 128,
                              use_amp: bool = True, checkpoint_dir: str = "checkpoints") -> dict:
    """Trains a single Grand Model with a specific loss formulation using all channels."""
    print("\n" + "=" * 90)
    print(f"TRAINING GRAND FOUNDATION MODEL: Loss Variant = [{variant_name.upper()}] (All Channels, Epochs: {epochs}, Prefix: {prefix_len})")
    print("=" * 90)

    model = UniversalFoundationTCN_Pro(
        hidden_dim=128, num_heads=8, num_layers=2,
        kernel_size=5, dilations=(1, 2, 4, 8, 16, 32), dropout=0.1,
        top_k="all"
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Instantiated UniversalFoundationTCN_Pro: {total_params:,} parameters ({total_params/1e6:.2f}M) [Top-K: all]")

    loss_fn = UnifiedFoundationLoss(mode=variant_name, alpha=0.1, gamma=2.0, pos_weight=2.0)
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
        tot_loss = 0.0
        tot_cls = 0.0
        tot_dice = 0.0
        tot_feat = 0.0
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

                p_sys = out["p_system"].float()
                z_student = out["z_student"].float()

                # Slice target horizon excluding receptive field warm-up prefix
                target_p = p_sys[:, b_prefix:] if b_prefix > 0 else p_sys
                target_y = y_dense[:, b_prefix:] if b_prefix > 0 else y_dense

                z_teacher = batch["z_teacher"].to(device).float() if "z_teacher" in batch else None
                has_teacher = batch["has_teacher"].to(device) if "has_teacher" in batch else None

                loss, loss_dict = loss_fn(
                    pred_dense=target_p,
                    target_dense=target_y,
                    student_feat=z_student,
                    teacher_feat=z_teacher,
                    has_teacher=has_teacher,
                    y_flag=y_win
                )

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
                p_sys = out["p_system"]
                z_student = out["z_student"]

                target_p = p_sys[:, b_prefix:] if b_prefix > 0 else p_sys
                target_y = y_dense[:, b_prefix:] if b_prefix > 0 else y_dense

                z_teacher = batch["z_teacher"].to(device) if "z_teacher" in batch else None
                has_teacher = batch["has_teacher"].to(device) if "has_teacher" in batch else None

                loss, loss_dict = loss_fn(
                    pred_dense=target_p,
                    target_dense=target_y,
                    student_feat=z_student,
                    teacher_feat=z_teacher,
                    has_teacher=has_teacher,
                    y_flag=y_win
                )

                if not torch.isfinite(loss):
                    optimizer.zero_grad()
                    continue

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            tot_loss += loss_dict["loss_total"]
            tot_cls += loss_dict["loss_cls"]
            tot_dice += loss_dict["loss_dice"]
            tot_feat += loss_dict["loss_feat"]
            n_batches += 1

        scheduler.step()
        epoch_dur = time.time() - t_epoch
        denom = max(1, n_batches)
        record = {
            "epoch": epoch,
            "loss_total": tot_loss / denom,
            "loss_cls": tot_cls / denom,
            "loss_dice": tot_dice / denom,
            "loss_feat": tot_feat / denom,
            "epoch_time_sec": epoch_dur
        }
        epoch_history.append(record)

        print(f"  [{variant_name.upper()}] Epoch [{epoch:2d}/{epochs:2d}] ({epoch_dur:.1f}s) | "
              f"Loss: {record['loss_total']:.4f} (cls:{record['loss_cls']:.4f}, dice:{record['loss_dice']:.4f}, feat:{record['loss_feat']:.4f})")

    total_train_time = time.time() - t_start
    print(f"Finished Training [{variant_name}] in {total_train_time/60:.2f} mins.")

    # Save checkpoint
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"grand_loss_{variant_name}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "loss_variant": variant_name,
        "prefix_len": prefix_len,
        "window_size": window_size,
        "top_k": "all",
        "epoch_history": epoch_history,
        "train_time_sec": total_train_time
    }, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")

    # Continuous Streaming Evaluation on held-out test slices for all 4 benchmarks
    print(f"\nEvaluating [{variant_name}] on all 4 benchmark test slices (prefix_len={prefix_len}, top_k=all, eval_stride={eval_stride})...")
    test_eval_results = {}
    for d_name, d_slice in test_slices.items():
        t_eval_start = time.time()
        metrics = evaluate_continuous_stream(
            model, d_slice["x"], d_slice["y"],
            window_size=window_size, prefix_len=prefix_len, stride=eval_stride,
            batch_size=eval_batch_size, top_k="all", device=device
        )
        t_eval_end = time.time()
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
        "variant": variant_name,
        "train_time_sec": total_train_time,
        "epoch_history": epoch_history,
        "test_eval_results": test_eval_results,
        "checkpoint_path": ckpt_path
    }


def main():
    parser = argparse.ArgumentParser(description="Grand Model Loss Formulation Ablation Suite")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument("--variants", nargs="+", type=str,
                        default=["pure_bce", "focal", "focal_dice", "unified_distill", "bce_feat"],
                        help="Loss variants to ablate")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--train_stride", type=int, default=50)
    parser.add_argument("--eval_stride", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--prefix_len", type=int, default=250)
    parser.add_argument("--window_size", type=int, default=200)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--output_json", type=str, default="results/ablations/grand_loss_ablation_results.json")
    parser.add_argument("--output_report", type=str, default="results/ablations/GRAND_LOSS_ABLATION_REPORT.md")
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
    print(f"Grand Loss Ablation Engine Running on: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    dataset_names = ["swat", "smd", "gecco", "msl"]
    print("\n1. Ingesting Training Datasets & Test Slices...")
    train_loaders = {}
    test_slices = {}

    for name in dataset_names:
        raw_data, labels, train_end, C = load_dataset_telemetry(name, args.data_dir)
        train_slice_x = raw_data[:train_end]
        train_slice_y = labels[:train_end]
        test_slice_x = raw_data[train_end:]
        test_slice_y = labels[train_end:]

        test_slices[name] = {"x": test_slice_x, "y": test_slice_y, "channels": C}

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

        collate_fn = UniversalCollate(subsample_range=(0.4, 1.0), shuffle_channels=True, seed=42 + len(train_loaders))
        ds = UniversalWindowDataset(
            train_slice_x, train_slice_y,
            window_size=args.window_size,
            prefix_len=args.prefix_len,
            stride=args.train_stride,
            augment_synthetic=True,
            teacher_cache=t_cache,
            seed=42
        )
        use_pin = (device == "cuda" or (isinstance(device, str) and device.startswith("cuda")))
        num_workers = args.num_workers if use_pin else 0
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
            collate_fn=collate_fn, num_workers=num_workers, pin_memory=use_pin
        )
        train_loaders[name] = loader
        print(f"  [{name.upper():5s}] Train Windows: {len(ds):,}, Test Steps: {len(test_slice_x):,}, Channels: {C}")

    print(f"\n2. Executing Grand Model Loss Ablation for Variants: {args.variants}...")
    ablation_payload = {
        "variants": args.variants,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "train_stride": args.train_stride,
        "eval_stride": args.eval_stride,
        "prefix_len": args.prefix_len,
        "window_size": args.window_size,
        "top_k": "all",
        "models": {}
    }

    t0_suite = time.time()
    for v in args.variants:
        res = train_single_loss_variant(
            variant_name=v,
            train_loaders=train_loaders,
            test_slices=test_slices,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            prefix_len=args.prefix_len,
            window_size=args.window_size,
            eval_stride=args.eval_stride,
            eval_batch_size=128,
            checkpoint_dir=args.checkpoint_dir
        )
        ablation_payload["models"][v] = res

    total_suite_time = time.time() - t0_suite
    ablation_payload["total_suite_time_sec"] = total_suite_time
    print(f"\nCompleted Full Loss Ablation Suite in {total_suite_time/60:.2f} mins.")

    # Save JSON Payload
    with open(args.output_json, "w") as f:
        json.dump(ablation_payload, f, indent=2)
    print(f"Saved full JSON ablation payload to: {args.output_json}")

    # Generate Structured Markdown Report
    lines = [
        "# Grand Foundation Model: Loss Formulation Ablation Report (All Channels)",
        "",
        f"- **Datasets Evaluated**: SWaT (C=51), SMD (C=38), GECCO (C=9), NASA MSL (C=55)",
        f"- **Channel Pooling**: All Channels ($K = \\text{{all}}$)",
        f"- **Receptive Field Prefix**: $L_{{\\text{{prefix}}}} = {args.prefix_len}$ steps",
        f"- **Training Epochs per Model**: {args.epochs}",
        f"- **Total Suite Execution Time**: {total_suite_time/60:.2f} minutes",
        "",
        "## 1. Cross-Dataset Metric Performance Table (<DATASET>_<METRIC>)",
        "",
        "| Loss Variant | Formulation | SWaT AUROC | SWaT PR-AUC | SMD AUROC | SMD Event Rec | GECCO AUROC | GECCO F1 | MSL AUROC | MSL Event Rec |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    formula_map = {
        "pure_bce": "$\\text{BCE}$",
        "focal": "$\\mathcal{L}_{\\text{focal}}$",
        "focal_dice": "$\\mathcal{L}_{\\text{focal}} + \\mathcal{L}_{\\text{dice}}$",
        "unified_distill": "$(\\mathcal{L}_{\\text{focal}} + \\mathcal{L}_{\\text{dice}}) + 0.1\\mathcal{L}_{\\text{feat}}$",
        "bce_feat": "$\\text{BCE} + 0.1\\mathcal{L}_{\\text{feat}}$"
    }

    for v in args.variants:
        m = ablation_payload["models"][v]["test_eval_results"]
        sw_auc = m["swat"]["auroc"]
        sw_pr = m["swat"]["prauc"]
        smd_auc = m["smd"]["auroc"]
        smd_ev = m["smd"]["event_recall"] * 100
        g_auc = m["gecco"]["auroc"]
        g_f1 = m["gecco"]["best_f1"]
        msl_auc = m["msl"]["auroc"]
        msl_ev = m["msl"]["event_recall"] * 100

        formula_str = formula_map.get(v, v)
        lines.append(f"| **{v}** | {formula_str} | {sw_auc:.4f} | {sw_pr:.4f} | {smd_auc:.4f} | {smd_ev:.1f}% | {g_auc:.4f} | {g_f1:.4f} | {msl_auc:.4f} | {msl_ev:.1f}% |")

    lines.extend([
        "",
        "## 2. Epoch-by-Epoch Convergence Trajectory",
        "",
        "| Variant | Epoch | Total Loss | Cls Loss | Dice Loss | Feat Distill | Time (s) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    ])

    for v in args.variants:
        history = ablation_payload["models"][v]["epoch_history"]
        for rec in history:
            lines.append(f"| {v} | {rec['epoch']} | {rec['loss_total']:.4f} | {rec['loss_cls']:.4f} | {rec['loss_dice']:.4f} | {rec['loss_feat']:.4f} | {rec['epoch_time_sec']:.1f} |")

    with open(args.output_report, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved Markdown summary report to: {args.output_report}")


if __name__ == "__main__":
    main()
