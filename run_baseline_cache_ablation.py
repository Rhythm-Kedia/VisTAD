"""Baseline Student Architectures (1D-CNN & Residual-TCN) × Dual Teacher Cache Manifolds.

Evaluates:
1. 1D-CNN (3 Layers, CI) + New Cache (1024-d Ultra-VETime / DINOv2)
2. 1D-CNN (3 Layers, CI) + Old Cache (2048-d Early VLM)
3. Residual-TCN (5 Layers, CI) + New Cache (1024-d Ultra-VETime / DINOv2)
4. Residual-TCN (5 Layers, CI) + Old Cache (2048-d Early VLM)

Across 4 Datasets: SWaT, GECCO, SMD, MSL
Evaluation: Held-out continuous test splits with Range-Aware Affiliation F1 and AUROC.
Runs locally (CUDA / MPS / CPU) or headlessly on Kaggle GPU.
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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.models.cnn_1d import CNN1DStudent
from src.data.universal_dataset import (
    load_dataset_telemetry,
    UniversalWindowDataset,
    UniversalCollate
)
from train_universal_foundation import evaluate_continuous_stream
from src.metrics import compute_affiliation_metrics


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_old_teacher_cache_path(dataset_name: str, cache_dir: str) -> str:
    name = dataset_name.lower().strip()
    candidates = [
        os.path.join(cache_dir, f"{name}_{name}_water_teacher_cache.pt"),
        os.path.join(cache_dir, f"{name}_{name}_attack_teacher_cache.pt"),
        os.path.join(cache_dir, f"{name}_{name}_test_teacher_cache.pt"),
        os.path.join(cache_dir, f"{name}_machine-1-1_teacher_cache.pt")
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    for f in os.listdir(cache_dir) if os.path.exists(cache_dir) else []:
        if f.startswith(name) and "teacher_cache" in f:
            return os.path.join(cache_dir, f)
    return None


def get_new_teacher_cache_path(dataset_name: str, cache_dir: str) -> str:
    name = dataset_name.lower().strip()
    cand = os.path.join(cache_dir, f"{name}_train_ultravetime_cache.pt")
    if os.path.exists(cand):
        return cand
    cand2 = os.path.join(cache_dir, f"{name}_eval_head_to_head_cache.pt")
    if os.path.exists(cand2):
        return cand2
    return None


def run_baseline_cache_experiment(
    dataset_name: str,
    arch_type: str,            # "cnn_1d" or "res_tcn"
    cache_type: str,           # "new_cache" or "old_cache"
    cache_path: str,
    data_dir: str,
    device: torch.device,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1.5e-3,
    prefix_len: int = 250,
    window_size: int = 200,
    train_stride: int = 50,
    eval_stride: int = 20,
    num_workers: int = 2,
    distill_feat_weight: float = 0.5,
    checkpoint_dir: str = "checkpoints/baseline_ablation"
) -> dict:
    model_key = f"{arch_type}_{cache_type}"
    print("\n" + "=" * 90)
    print(f"RUNNING [{dataset_name.upper()}] | Model: {model_key} ({arch_type} + {cache_type})")
    print(f"  Teacher Cache: {cache_path} | Device: {device}")
    print("=" * 90)

    # 1. Load Data
    raw_data, labels, train_end, C = load_dataset_telemetry(dataset_name, data_dir)
    train_x, train_y = raw_data[:train_end], labels[:train_end]
    test_x, test_y = raw_data[train_end:], labels[train_end:]

    # 2. Ingest Teacher Cache
    t_cache = None
    teacher_dim = 1024
    if cache_path and os.path.exists(cache_path):
        t_cache = torch.load(cache_path, map_location="cpu")
        print(f"  Loaded Teacher Cache: {cache_path} ({len(t_cache.get('windows', t_cache)):,} records)")

    ds_train = UniversalWindowDataset(
        train_x, train_y,
        window_size=window_size, prefix_len=prefix_len, stride=train_stride,
        augment_synthetic=True, teacher_cache=t_cache, seed=42
    )
    teacher_dim = ds_train.teacher_dim
    print(f"  Resolved Teacher Latent Dim: {teacher_dim}-d (Windows: {len(ds_train):,})")

    use_cuda = (device.type == "cuda")
    collate_fn = UniversalCollate(subsample_range=None, shuffle_channels=False)
    loader_train = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True, drop_last=True,
        collate_fn=collate_fn, num_workers=(num_workers if use_cuda else 0), pin_memory=use_cuda
    )

    # 3. Instantiate Model
    if arch_type == "cnn_1d":
        model = CNN1DStudent(
            in_channels=C, hidden_dim=128, num_layers=3, dropout=0.1,
            teacher_feat_dim=teacher_dim, channel_independent=True, use_residual=False, causal=True
        ).to(device)
    elif arch_type == "res_tcn":
        model = CNN1DStudent(
            in_channels=C, hidden_dim=128, num_layers=5, dropout=0.1,
            teacher_feat_dim=teacher_dim, channel_independent=True, use_residual=True, causal=True
        ).to(device)
    else:
        raise ValueError(f"Unknown architecture: {arch_type}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Instantiated {arch_type}: {total_params:,} parameters ({total_params/1e6:.2f}M)")

    # 4. Optimizer & Scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    use_cuda = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    # 5. Training Loop
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        dense_loss_sum = 0.0
        feat_loss_sum = 0.0
        n_batches = 0
        t_ep = time.time()

        for batch in loader_train:
            x = batch["x"].to(device)  # [B, C, L_total]
            y_dense = batch["y_dense"].to(device)  # [B, L_total]
            b_prefix = batch.get("prefix_len", prefix_len)

            optimizer.zero_grad()

            if use_cuda:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    out = model(x)
                p_sys = out.get("pred_dense", out.get("p_system", out.get("logits_dense")))
                z_student = out.get("pred_feature", out.get("z_student", out.get("features", None)))

                p_sys_clamped = torch.clamp(p_sys.float(), 1e-5, 1.0 - 1e-5)
                target_y = y_dense[:, b_prefix:].float() if b_prefix > 0 else y_dense.float()
                pred_y = p_sys_clamped[:, b_prefix:] if (p_sys_clamped.shape[-1] == y_dense.shape[-1] and b_prefix > 0) else p_sys_clamped
                loss_dense = F.binary_cross_entropy(pred_y, target_y)

                has_teacher = batch["has_teacher"].to(device)
                if has_teacher.sum() > 0 and distill_feat_weight > 0.0 and z_student is not None:
                    z_teacher = batch["z_teacher"].to(device).float()
                    z_t_norm = F.normalize(z_teacher, p=2, dim=-1)
                    z_s_norm = F.normalize(z_student.float(), p=2, dim=-1)
                    cos_sim = torch.sum(z_s_norm * z_t_norm, dim=-1)
                    loss_feat = torch.sum((1.0 - cos_sim) * has_teacher) / torch.clamp(has_teacher.sum(), min=1.0)
                else:
                    loss_feat = torch.tensor(0.0, device=device)

                loss = loss_dense + distill_feat_weight * loss_feat
                if not torch.isfinite(loss):
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(x)
                p_sys = out.get("pred_dense", out.get("p_system", out.get("logits_dense")))
                z_student = out.get("pred_feature", out.get("z_student", out.get("features", None)))

                p_sys_clamped = torch.clamp(p_sys.float(), 1e-5, 1.0 - 1e-5)
                target_y = y_dense[:, b_prefix:].float() if b_prefix > 0 else y_dense.float()
                pred_y = p_sys_clamped[:, b_prefix:] if (p_sys_clamped.shape[-1] == y_dense.shape[-1] and b_prefix > 0) else p_sys_clamped
                loss_dense = F.binary_cross_entropy(pred_y, target_y)

                has_teacher = batch["has_teacher"].to(device)
                if has_teacher.sum() > 0 and distill_feat_weight > 0.0 and z_student is not None:
                    z_teacher = batch["z_teacher"].to(device).float()
                    z_t_norm = F.normalize(z_teacher, p=2, dim=-1)
                    z_s_norm = F.normalize(z_student.float(), p=2, dim=-1)
                    cos_sim = torch.sum(z_s_norm * z_t_norm, dim=-1)
                    loss_feat = torch.sum((1.0 - cos_sim) * has_teacher) / torch.clamp(has_teacher.sum(), min=1.0)
                else:
                    loss_feat = torch.tensor(0.0, device=device)

                loss = loss_dense + distill_feat_weight * loss_feat
                if not torch.isfinite(loss):
                    continue

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            dense_loss_sum += loss_dense.item()
            feat_loss_sum += loss_feat.item() if isinstance(loss_feat, torch.Tensor) else loss_feat
            n_batches += 1

        ep_sec = time.time() - t_ep
        avg_loss = total_loss / max(n_batches, 1)
        avg_dense = dense_loss_sum / max(n_batches, 1)
        avg_feat = feat_loss_sum / max(n_batches, 1)
        print(f"  Epoch [{epoch:2d}/{epochs}] ({ep_sec:.1f}s) | Loss: {avg_loss:.4f} (dense:{avg_dense:.4f}, feat:{avg_feat:.4f})")

    # 6. Save Checkpoint
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"{dataset_name}_{model_key}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "arch_type": arch_type,
        "cache_type": cache_type,
        "dataset": dataset_name,
        "teacher_dim": teacher_dim,
        "in_channels": C
    }, ckpt_path)
    print(f"  Saved Checkpoint: {ckpt_path}")

    # 7. Streaming Evaluation on Held-Out Test Split
    print(f"  Evaluating {model_key} on held-out test split ({len(test_x):,} steps, stride={eval_stride})...")
    eval_device_str = "cuda" if use_cuda else ("mps" if device.type == "mps" else "cpu")
    raw_eval = evaluate_continuous_stream(
        model, test_x, test_y,
        window_size=window_size, prefix_len=prefix_len, stride=eval_stride,
        batch_size=128, device=eval_device_str
    )

    clean_metrics = {k: v for k, v in raw_eval.items() if k != "point_scores"}
    metrics = {
        "auroc": float(clean_metrics["auroc"]),
        "prauc": float(clean_metrics["prauc"]),
        "best_f1": float(clean_metrics["best_f1"]),
        "best_threshold": float(clean_metrics.get("best_threshold", 0.5)),
        "aff_f1": float(clean_metrics.get("aff_f1", 0.0)),
        "aff_precision": float(clean_metrics.get("aff_precision", 0.0)),
        "aff_recall": float(clean_metrics.get("aff_recall", 0.0)),
        "mean_tiou": float(clean_metrics.get("mean_tiou", 0.0)),
        "fpr": float(clean_metrics.get("fpr", 0.0)),
        "event_recall": float(clean_metrics.get("event_recall", 0.0))
    }

    print(f"  >>> {model_key.upper()} RESULTS: AUROC={metrics['auroc']:.4f} | PR-AUC={metrics['prauc']:.4f} | Best F1={metrics['best_f1']:.4f} | Aff F1={metrics['aff_f1']:.4f} | Event Rec={metrics['event_recall']*100:.1f}%")

    # Clean GPU memory
    del model, optimizer, loader_train, ds_train
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "dataset": dataset_name,
        "model_key": model_key,
        "arch_type": arch_type,
        "cache_type": cache_type,
        "teacher_dim": teacher_dim,
        "checkpoint_path": ckpt_path,
        "metrics": metrics
    }


def main():
    parser = argparse.ArgumentParser(description="Baseline Student × Dual Teacher Cache Ablation")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument("--datasets", nargs="+", default=["swat", "gecco", "smd", "msl"])
    parser.add_argument("--models", nargs="+", default=["cnn_1d", "res_tcn"], choices=["cnn_1d", "res_tcn"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--train_stride", type=int, default=50)
    parser.add_argument("--eval_stride", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--prefix_len", type=int, default=250)
    parser.add_argument("--window_size", type=int, default=200)
    parser.add_argument("--distill_feat_weight", type=float, default=0.5)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/baseline_ablation")
    parser.add_argument("--output_json", type=str, default="results/ablations/baseline_cache_ablation.json")
    parser.add_argument("--output_report", type=str, default="results/ablations/BASELINE_CACHE_REPORT.md")
    args = parser.parse_args()

    device = get_device()
    print(f"Baseline Cache Ablation Suite Running on: {device}")
    print(f"Datasets: {args.datasets} | Models: {args.models} | Epochs: {args.epochs}")

    t0_suite = time.time()
    all_results = {}

    for ds in args.datasets:
        ds = ds.lower().strip()
        print(f"\n{'#' * 80}\nSTARTING BASELINE ABLATION FOR BENCHMARK: [{ds.upper()}]\n{'#' * 80}")
        
        old_cache = get_old_teacher_cache_path(ds, args.cache_dir)
        new_cache = get_new_teacher_cache_path(ds, args.cache_dir)
        print(f"  New Cache (1024-d): {new_cache}")
        print(f"  Old Cache (2048-d): {old_cache}")

        ds_results = {}
        for arch in args.models:
            # 1. Architecture + New Cache
            if new_cache:
                r_new = run_baseline_cache_experiment(
                    dataset_name=ds, arch_type=arch, cache_type="new_cache",
                    cache_path=new_cache, data_dir=args.data_dir, device=device,
                    epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                    prefix_len=args.prefix_len, window_size=args.window_size,
                    train_stride=args.train_stride, eval_stride=args.eval_stride,
                    num_workers=args.num_workers,
                    distill_feat_weight=args.distill_feat_weight,
                    checkpoint_dir=args.checkpoint_dir
                )
                ds_results[f"{arch}_new_cache"] = r_new

            # 2. Architecture + Old Cache
            if old_cache:
                r_old = run_baseline_cache_experiment(
                    dataset_name=ds, arch_type=arch, cache_type="old_cache",
                    cache_path=old_cache, data_dir=args.data_dir, device=device,
                    epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                    prefix_len=args.prefix_len, window_size=args.window_size,
                    train_stride=args.train_stride, eval_stride=args.eval_stride,
                    num_workers=args.num_workers,
                    distill_feat_weight=args.distill_feat_weight,
                    checkpoint_dir=args.checkpoint_dir
                )
                ds_results[f"{arch}_old_cache"] = r_old

        all_results[ds] = ds_results

    total_suite_sec = time.time() - t0_suite

    # Save JSON Payload
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({
            "total_time_sec": total_suite_sec,
            "datasets": args.datasets,
            "models": args.models,
            "results": all_results
        }, f, indent=2)
    print(f"\nSaved master results JSON to: {args.output_json}")

    # Build Master Markdown Summary Table
    lines = [
        "# Baseline Students (1D-CNN & Res-TCN) × Dual Teacher Cache Manifolds Report",
        "",
        f"- **Datasets Evaluated**: {', '.join([d.upper() for d in args.datasets])}",
        f"- **Architectures**: {', '.join(args.models)}",
        f"- **Teacher Manifolds**: New Cache (1024-d Ultra-VETime) vs. Old Cache (2048-d Early VLM)",
        f"- **Total Suite Runtime**: {total_suite_sec/60:.2f} minutes",
        "",
        "## Master Benchmark Table (with Affiliation F1 & Range-Aware Metrics)",
        "",
        "| Dataset | Model Key | Architecture | Teacher Supervision | AUROC | PR-AUC | Best F1 | Aff F1 | Aff Prec | Aff Rec | FPR (%) | Event Rec (%) |",
        "| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for ds, models_dict in all_results.items():
        for m_key, res in models_dict.items():
            m = res["metrics"]
            arch_display = "1D-CNN (3L)" if res["arch_type"] == "cnn_1d" else "Residual-TCN (5L)"
            c_desc = f"New Cache ({res['teacher_dim']}-d)" if res["cache_type"] == "new_cache" else f"Old Cache ({res['teacher_dim']}-d)"
            lines.append(
                f"| **{ds.upper()}** | `{m_key}` | {arch_display} | {c_desc} | "
                f"{m['auroc']:.4f} | {m['prauc']:.4f} | {m['best_f1']:.4f} | **{m.get('aff_f1', 0.0):.4f}** | "
                f"{m.get('aff_precision', 0.0):.4f} | {m.get('aff_recall', 0.0):.4f} | {m['fpr']*100:.2f}% | {m['event_recall']*100:.1f}% |"
            )

    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
    with open(args.output_report, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved Markdown report to: {args.output_report}")


if __name__ == "__main__":
    main()
