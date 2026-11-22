"""6-Model Cross-Ablation Suite: 2 New Students + 1 Old Student across Dual Teacher Manifolds.

Factorial Grid: Exactly 6 Models per Dataset x 4 Datasets = 24 Total Models
1. old_student_new_cache: Specialized Coupled Domain TCN + New Cache (1024-d Ultra-VETime)
2. old_student_old_cache: Specialized Coupled Domain TCN + Old Cache (2048-d Early VLM)
3. tcn_pro_new_cache: UniversalFoundationTCN_Pro + New Cache (1024-d Ultra-VETime)
4. tcn_pro_old_cache: UniversalFoundationTCN_Pro + Old Cache (2048-d Early VLM)
5. channel_aware_new_cache: UniversalTCN_ChannelAware + New Cache (1024-d Ultra-VETime)
6. channel_aware_old_cache: UniversalTCN_ChannelAware + Old Cache (2048-d Early VLM)

Common Evaluation on Held-Out Test Splits with Affiliation Metrics:
- AUROC, PR-AUC, Best F1, Affiliation F1 (Aff F1), Aff Precision, Aff Recall, Temporal IoU, FPR, Event Recall
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
try:
    from src.models.universal_tcn_channel_aware import UniversalTCN_ChannelAware
except ImportError:
    from src.models.universal_tcn_channel_aware import UniversalTCNChannelAwareStudent as UniversalTCN_ChannelAware

try:
    from src.models.gecco_coupled_tcn import GeccoCoupledTCN
except ImportError:
    from src.models.gecco_coupled_tcn import GeccoSubsystemCoupledTCN as GeccoCoupledTCN

try:
    from src.models.stage_coupled_tcn import StageCoupledTCN
except ImportError:
    from src.models.stage_coupled_tcn import StageCoupledTCNStudent as StageCoupledTCN

try:
    from src.models.smd_coupled_tcn import SMDCoupledTCN
except ImportError:
    from src.models.smd_coupled_tcn import SMDSubsystemCoupledTCN as SMDCoupledTCN

try:
    from src.models.msl_coupled_tcn import MSLCoupledTCN
except ImportError:
    from src.models.msl_coupled_tcn import MSLCoupledTCNStudent as MSLCoupledTCN

from src.data.universal_dataset import (
    load_dataset_telemetry,
    UniversalWindowDataset,
    UniversalCollate
)
from train_universal_foundation import evaluate_continuous_stream
from src.metrics import compute_affiliation_metrics


def get_specialized_student(dataset_name: str, teacher_dim: int, device: str) -> nn.Module:
    """Instantiates the domain-specialized 'Old Student' architecture for a given dataset."""
    name = dataset_name.lower().strip()
    if name == "gecco":
        model = GeccoCoupledTCN(teacher_feat_dim=teacher_dim)
    elif name == "swat":
        model = StageCoupledTCN(feat_dim=teacher_dim)
    elif name == "smd":
        model = SMDCoupledTCN(teacher_feat_dim=teacher_dim)
    elif name == "msl":
        model = MSLCoupledTCN(teacher_feat_dim=teacher_dim)
    else:
        raise ValueError(f"No specialized student architecture for {dataset_name}")
    return model.to(device)


def get_old_teacher_cache_path(dataset_name: str, cache_dir: str) -> str:
    """Discovers the older 2048-d teacher cache for a dataset."""
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
    """Discovers the newer 1024-d Ultra-VETime teacher cache for a dataset."""
    name = dataset_name.lower().strip()
    cand = os.path.join(cache_dir, f"{name}_train_ultravetime_cache.pt")
    if os.path.exists(cand):
        return cand
    cand2 = os.path.join(cache_dir, f"{name}_eval_head_to_head_cache.pt")
    if os.path.exists(cand2):
        return cand2
    return None


def run_single_model_experiment(dataset_name: str, model_key: str, student_type: str,
                                cache_type: str, cache_path: str, data_dir: str, device: str,
                                epochs: int = 10, batch_size: int = 32, lr: float = 1e-3,
                                prefix_len: int = 250, window_size: int = 200,
                                distill_feat_weight: float = 0.5,
                                checkpoint_dir: str = "checkpoints/cross_ablation") -> dict:
    """Trains and evaluates one of the 6 model configurations on the dataset."""
    print("\n" + "=" * 90)
    print(f"RUNNING [{dataset_name.upper()}] | Model: {model_key} ({student_type} + {cache_type})")
    print(f"  Teacher Cache: {cache_path} | Distill Feat Weight: {distill_feat_weight}")
    print("=" * 90)

    # 1. Load Telemetry Split
    raw_data, labels, train_end, C = load_dataset_telemetry(dataset_name, data_dir)
    train_x, train_y = raw_data[:train_end], labels[:train_end]
    test_x, test_y = raw_data[train_end:], labels[train_end:]

    # 2. Ingest Cache
    t_cache = None
    teacher_dim = 1024
    if cache_path and os.path.exists(cache_path):
        t_cache = torch.load(cache_path, map_location="cpu")
        print(f"  Loaded Teacher Cache: {cache_path} ({len(t_cache.get('windows', t_cache)):,} records)")

    ds_train = UniversalWindowDataset(
        train_x, train_y,
        window_size=window_size, prefix_len=prefix_len, stride=20,
        augment_synthetic=True, teacher_cache=t_cache, seed=42
    )
    teacher_dim = ds_train.teacher_dim
    print(f"  Resolved Teacher Latent Dim: {teacher_dim}-d")

    collate_fn = UniversalCollate(subsample_range=None, shuffle_channels=False)
    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)

    # 3. Instantiate Student Model
    if student_type == "old_specialized":
        model = get_specialized_student(dataset_name, teacher_dim, device)
    elif student_type == "tcn_pro":
        model = UniversalFoundationTCN_Pro(
            hidden_dim=128, num_heads=8, num_layers=2,
            kernel_size=5, dilations=(1, 2, 4, 8, 16, 32), dropout=0.1,
            top_k=5, teacher_dim=teacher_dim
        ).to(device)
    elif student_type == "channel_aware":
        model = UniversalTCN_ChannelAware(
            in_channels=C, hidden_dim=128, num_heads=8, num_layers=2,
            kernel_size=5, dilations=(1, 2, 4, 8, 16, 32), dropout=0.05,
            top_k=5, teacher_feat_dim=teacher_dim
        ).to(device)
    else:
        raise ValueError(f"Unknown student_type: {student_type}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Instantiated {student_type}: {total_params:,} parameters ({total_params/1e6:.2f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    use_amp = device.startswith("cuda") if isinstance(device, str) else device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler('cuda') if use_amp else None
    except Exception:
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

    epoch_history = []
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, tot_dense, tot_feat = 0.0, 0.0, 0.0
        n_batches = 0
        t_ep = time.time()

        for batch in loader_train:
            x = batch["x"].to(device)
            y_dense = batch["y_dense"].to(device)
            b_prefix = batch["prefix_len"]

            optimizer.zero_grad()

            if scaler is not None:
                try:
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        if student_type == "old_specialized":
                            out = model(x)
                        else:
                            out = model(x, prefix_len=b_prefix)
                except Exception:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        out = model(x) if student_type == "old_specialized" else model(x, prefix_len=b_prefix)

                # Extract probabilities
                if "p_system" in out:
                    p_sys = out["p_system"]
                elif "pred_dense" in out:
                    p_sys = out["pred_dense"]
                else:
                    p_sys = out["p_channel"].mean(dim=1)

                # Extract features
                if "z_student" in out:
                    z_student = out["z_student"]
                elif "pred_feature" in out:
                    z_student = out["pred_feature"]
                elif "student_feat" in out:
                    z_student = out["student_feat"]
                else:
                    z_student = out.get("feat", None)

                p_sys_clamped = torch.clamp(p_sys.float(), 1e-5, 1.0 - 1e-5)
                # Dense target slice
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
                    optimizer.zero_grad()
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(x) if student_type == "old_specialized" else model(x, prefix_len=b_prefix)
                p_sys = out.get("p_system", out.get("pred_dense", out.get("p_channel").mean(dim=1)))
                z_student = out.get("z_student", out.get("pred_feature", out.get("student_feat", out.get("feat", None))))

                p_sys_clamped = torch.clamp(p_sys, 1e-5, 1.0 - 1e-5)
                target_y = y_dense[:, b_prefix:] if b_prefix > 0 else y_dense
                pred_y = p_sys_clamped[:, b_prefix:] if (p_sys_clamped.shape[-1] == y_dense.shape[-1] and b_prefix > 0) else p_sys_clamped
                loss_dense = F.binary_cross_entropy(pred_y, target_y)

                has_teacher = batch["has_teacher"].to(device)
                if has_teacher.sum() > 0 and distill_feat_weight > 0.0 and z_student is not None:
                    z_teacher = batch["z_teacher"].to(device)
                    z_t_norm = F.normalize(z_teacher, p=2, dim=-1)
                    z_s_norm = F.normalize(z_student, p=2, dim=-1)
                    cos_sim = torch.sum(z_s_norm * z_t_norm, dim=-1)
                    loss_feat = torch.sum((1.0 - cos_sim) * has_teacher) / torch.clamp(has_teacher.sum(), min=1.0)
                else:
                    loss_feat = torch.tensor(0.0, device=device)

                loss = loss_dense + distill_feat_weight * loss_feat
                if not torch.isfinite(loss):
                    optimizer.zero_grad()
                    continue

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            tot_dense += loss_dense.item()
            tot_feat += loss_feat.item()
            n_batches += 1

        scheduler.step()
        ep_dur = time.time() - t_ep
        denom = max(1, n_batches)
        record = {
            "epoch": epoch,
            "train_loss": total_loss / denom,
            "loss_dense": tot_dense / denom,
            "loss_feat": tot_feat / denom,
            "epoch_time_sec": ep_dur
        }
        epoch_history.append(record)
        print(f"  Epoch [{epoch:2d}/{epochs:2d}] ({ep_dur:.1f}s) | Loss: {record['train_loss']:.4f} (dense:{record['loss_dense']:.4f}, feat:{record['loss_feat']:.4f})")

    total_train_sec = time.time() - t_start

    # 4. Continuous Streaming Evaluation on Common Held-Out Test Split
    print(f"  Evaluating {model_key} on held-out test split ({len(test_x):,} steps)...")
    raw_eval = evaluate_continuous_stream(
        model, test_x, test_y,
        window_size=window_size, prefix_len=prefix_len, stride=10,
        batch_size=64, device=device
    )

    clean_metrics = {k: v for k, v in raw_eval.items() if k != "point_scores"}
    print(f"  [RESULT] AUROC={clean_metrics['auroc']:.4f} | PR-AUC={clean_metrics['prauc']:.4f} | "
          f"Best F1={clean_metrics['best_f1']:.4f} | Aff F1={clean_metrics.get('aff_f1', 0.0):.4f} | "
          f"FPR={clean_metrics['fpr']*100:.2f}% | EventRecall={clean_metrics['event_recall']*100:.1f}%")

    # 5. Save Checkpoint
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"{dataset_name}_{model_key}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "dataset": dataset_name,
        "model_key": model_key,
        "student_type": student_type,
        "cache_type": cache_type,
        "teacher_dim": teacher_dim,
        "metrics": clean_metrics,
        "epoch_history": epoch_history,
        "train_time_sec": total_train_sec
    }, ckpt_path)
    print(f"  Saved Checkpoint: {ckpt_path}")

    del model, optimizer, scaler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "dataset": dataset_name,
        "model_key": model_key,
        "student_type": student_type,
        "cache_type": cache_type,
        "teacher_dim": teacher_dim,
        "train_time_sec": total_train_sec,
        "metrics": clean_metrics,
        "checkpoint": ckpt_path,
        "epoch_history": epoch_history
    }


def main():
    parser = argparse.ArgumentParser(description="6-Model x 4-Dataset Cross-Ablation Suite")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument("--datasets", nargs="+", type=str, default=["swat", "smd", "gecco", "msl"],
                        help="Target benchmarks (default: swat smd gecco msl)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/cross_ablation")
    parser.add_argument("--output_json", type=str, default="results/ablations/six_models_cross_ablation.json")
    parser.add_argument("--output_report", type=str, default="results/ablations/SIX_MODELS_CROSS_ABLATION_REPORT.md")
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
    print(f"6-Model Cross-Ablation Engine Running on: {device}")

    # Exactly 6 configurations:
    # 2 New Students x (Old Cache, New Cache) = 4
    # 1 Old Student x (Old Cache, New Cache) = 2
    # Total = 6 models per dataset
    six_configurations = [
        # Old Student (Coupled Domain TCN)
        {"key": "old_student_new_cache", "student": "old_specialized", "cache": "new_cache"},
        {"key": "old_student_old_cache", "student": "old_specialized", "cache": "old_cache"},
        # New Student 1 (UniversalFoundationTCN_Pro)
        {"key": "tcn_pro_new_cache", "student": "tcn_pro", "cache": "new_cache"},
        {"key": "tcn_pro_old_cache", "student": "tcn_pro", "cache": "old_cache"},
        # New Student 2 (UniversalTCN_ChannelAware)
        {"key": "channel_aware_new_cache", "student": "channel_aware", "cache": "new_cache"},
        {"key": "channel_aware_old_cache", "student": "channel_aware", "cache": "old_cache"},
    ]

    all_results = {}
    t0_suite = time.time()

    for ds in args.datasets:
        all_results[ds] = {}
        old_cache = get_old_teacher_cache_path(ds, args.cache_dir)
        new_cache = get_new_teacher_cache_path(ds, args.cache_dir)

        print(f"\n================================================================================")
        print(f"STARTING 6-MODEL SUITE FOR BENCHMARK: [{ds.upper()}]")
        print(f"  New Cache (1024-d): {new_cache}")
        print(f"  Old Cache (2048-d): {old_cache}")
        print(f"================================================================================")

        for cfg in six_configurations:
            c_type = cfg["cache"]
            target_cache = new_cache if c_type == "new_cache" else old_cache

            res = run_single_model_experiment(
                dataset_name=ds,
                model_key=cfg["key"],
                student_type=cfg["student"],
                cache_type=c_type,
                cache_path=target_cache,
                data_dir=args.data_dir,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                distill_feat_weight=0.5,
                checkpoint_dir=args.checkpoint_dir
            )
            all_results[ds][cfg["key"]] = res

    total_suite_sec = time.time() - t0_suite

    # Save JSON Payload
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({
            "total_time_sec": total_suite_sec,
            "datasets": args.datasets,
            "results": all_results
        }, f, indent=2)
    print(f"\nSaved master results JSON to: {args.output_json}")

    # Build Master Markdown Summary Table
    lines = [
        "# 6-Model Cross-Ablation Report: (2 New Students + 1 Old Student) × Dual Teacher Manifolds",
        "",
        f"- **Datasets Evaluated**: {', '.join([d.upper() for d in args.datasets])}",
        f"- **Architectures**: Old Coupled TCN, `UniversalFoundationTCN_Pro`, `UniversalTCN_ChannelAware`",
        f"- **Teacher Manifolds**: New Cache (1024-d Ultra-VETime) vs. Old Cache (2048-d Early VLM)",
        f"- **Total Suite Runtime**: {total_suite_sec/60:.2f} minutes",
        "",
        "## Master Benchmark Table (with Affiliation F1 & Range-Aware Metrics)",
        "",
        "| Dataset | Model Key | Student Architecture | Teacher Supervision | AUROC | PR-AUC | Best F1 | Aff F1 | Aff Prec | Aff Rec | FPR (%) | Event Rec (%) |",
        "| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for ds, models_dict in all_results.items():
        for m_key, res in models_dict.items():
            m = res["metrics"]
            if res["student_type"] == "old_specialized":
                st_name = "Old Coupled TCN"
            elif res["student_type"] == "tcn_pro":
                st_name = "TCN_Pro (Perm-Equiv)"
            else:
                st_name = "ChannelAware (Learned E_ch)"
            c_desc = f"New Cache ({res['teacher_dim']}-d)" if res["cache_type"] == "new_cache" else f"Old Cache ({res['teacher_dim']}-d)"
            lines.append(
                f"| **{ds.upper()}** | `{m_key}` | {st_name} | {c_desc} | "
                f"{m['auroc']:.4f} | {m['prauc']:.4f} | {m['best_f1']:.4f} | **{m.get('aff_f1', 0.0):.4f}** | "
                f"{m.get('aff_precision', 0.0):.4f} | {m.get('aff_recall', 0.0):.4f} | {m['fpr']*100:.2f}% | {m['event_recall']*100:.1f}% |"
            )

    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
    with open(args.output_report, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved Markdown report to: {args.output_report}")


if __name__ == "__main__":
    main()
