"""Master Comprehensive Continuous Streaming Evaluation Registry.

Evaluates 44 student models across the entire held-out test partitions:
- 24 Factorial Cross-Ablation Models (2 New Students + 1 Old Student x Dual Teacher Manifolds)
- 16 Baseline Ablation Models (1D-CNN 3L & Residual-TCN 5L x Dual Teacher Manifolds)
- 4 Best Loss-Tuned Models (Optimal loss ablation checkpoints)
Across all 4 Datasets: SWaT (51D), GECCO (9D), SMD (38D), NASA MSL (55D).

Protocol:
- Stride: 10
- Window: 200
- Prefix: 250 (causal warm-up buffer)
- Point metrics: AUROC, PR-AUC, Best Point-F1, Point Precision, Point Recall, FPR at tau*
- Affiliation metrics: aff_f1, aff_precision, aff_recall
- Event metrics: Contiguous Event Recall (caught/total, percentage)
- Operational metrics: latency (ms), throughput (win/s), parameter count

Outputs:
- results/master_comprehensive_evaluation_registry.json
- results/master_comprehensive_evaluation_summary.md
"""

import os
import sys
import time
import json
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.universal_dataset import load_dataset_telemetry
from src.models.universal_tcn import UniversalFoundationTCN_Pro
try:
    from src.models.universal_tcn_channel_aware import UniversalTCN_ChannelAware
except ImportError:
    from src.models.universal_tcn_channel_aware import UniversalTCNChannelAwareStudent as UniversalTCN_ChannelAware

from src.models.gecco_coupled_tcn import GeccoCoupledTCN
from src.models.stage_coupled_tcn import StageCoupledTCN
from src.models.smd_coupled_tcn import SMDCoupledTCN
from src.models.msl_coupled_tcn import MSLCoupledTCN
from src.models.cnn_1d import CNN1DStudent
from src.models.stage_coupled_tcn import StageCoupledTCNStudent
from src.models.gecco_coupled_tcn import GeccoSubsystemCoupledTCN
from src.models.smd_coupled_tcn import SMDSubsystemCoupledTCN
from src.models.msl_coupled_tcn import MSLCoupledTCNStudent

from train_universal_foundation import evaluate_continuous_stream


def get_specialized_student(dataset_name: str, teacher_dim: int, device: str) -> torch.nn.Module:
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
        raise ValueError(f"Unknown specialized student: {dataset_name}")
    return model.to(device)


def main():
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting Master Comprehensive Evaluation Registry on: {device} ===")

    datasets = ["gecco", "msl", "swat", "smd"]
    data_dir = "data"
    prefix_len = 250
    window_size = 200
    stride = 10

    # 1. Load Data Splits
    splits = {}
    for ds in datasets:
        print(f"Loading {ds.upper()} continuous telemetry...")
        raw_data, labels, train_end, C = load_dataset_telemetry(ds, data_dir)
        test_x, test_y = raw_data[train_end:], labels[train_end:]
        n_anom = int(np.sum(test_y))
        print(f"  {ds.upper()}: C={C}, Total={len(raw_data):,}, TrainEnd={train_end:,}, Test={len(test_x):,} ({n_anom:,} anomalies, {np.mean(test_y)*100:.2f}%)")
        splits[ds] = {
            "test_x": test_x,
            "test_y": test_y,
            "C": C,
            "train_end": train_end
        }

    # 2. Build Evaluation Manifest
    all_runs = []

    for ds in datasets:
        C = splits[ds]["C"]

        # --- A. Cross-Ablation Models (6 per dataset) ---
        cross_base = "checkpoints/22Nov/cross_ablation_24/checkpoints/cross_ablation"
        all_runs.extend([
            {
                "dataset": ds,
                "group": "Cross-Ablation (Specialized Student)",
                "model_key": "old_student_new_cache",
                "cache_type": "New Cache (1024-d Ultra-VETime)",
                "ckpt_path": os.path.join(cross_base, f"{ds}_old_student_new_cache.pt"),
                "loader_fn": lambda d=ds: get_specialized_student(d, 1024, device),
                "batch_size": 128
            },
            {
                "dataset": ds,
                "group": "Cross-Ablation (Specialized Student)",
                "model_key": "old_student_old_cache",
                "cache_type": "Old Cache (2048-d Early VLM)",
                "ckpt_path": os.path.join(cross_base, f"{ds}_old_student_old_cache.pt"),
                "loader_fn": lambda d=ds: get_specialized_student(d, 2048, device),
                "batch_size": 128
            },
            {
                "dataset": ds,
                "group": "Cross-Ablation (Channel-Aware Student)",
                "model_key": "channel_aware_new_cache",
                "cache_type": "New Cache (1024-d Ultra-VETime)",
                "ckpt_path": os.path.join(cross_base, f"{ds}_channel_aware_new_cache.pt"),
                "loader_fn": lambda c=C: UniversalTCN_ChannelAware(
                    in_channels=c, hidden_dim=128, num_heads=8, num_layers=2,
                    kernel_size=5, dilations=(1, 2, 4, 8, 16, 32), dropout=0.05,
                    top_k=5, teacher_feat_dim=1024
                ).to(device),
                "batch_size": 128
            },
            {
                "dataset": ds,
                "group": "Cross-Ablation (Channel-Aware Student)",
                "model_key": "channel_aware_old_cache",
                "cache_type": "Old Cache (2048-d Early VLM)",
                "ckpt_path": os.path.join(cross_base, f"{ds}_channel_aware_old_cache.pt"),
                "loader_fn": lambda c=C: UniversalTCN_ChannelAware(
                    in_channels=c, hidden_dim=128, num_heads=8, num_layers=2,
                    kernel_size=5, dilations=(1, 2, 4, 8, 16, 32), dropout=0.05,
                    top_k=5, teacher_feat_dim=2048
                ).to(device),
                "batch_size": 128
            },
            {
                "dataset": ds,
                "group": "Cross-Ablation (Universal Foundation TCN Pro)",
                "model_key": "tcn_pro_new_cache",
                "cache_type": "New Cache (1024-d Ultra-VETime)",
                "ckpt_path": os.path.join(cross_base, f"{ds}_tcn_pro_new_cache.pt"),
                "loader_fn": lambda: UniversalFoundationTCN_Pro(
                    hidden_dim=128, num_heads=8, num_layers=2,
                    kernel_size=5, dilations=(1, 2, 4, 8, 16, 32), dropout=0.1,
                    top_k=5, teacher_dim=1024
                ).to(device),
                "batch_size": 128
            },
            {
                "dataset": ds,
                "group": "Cross-Ablation (Universal Foundation TCN Pro)",
                "model_key": "tcn_pro_old_cache",
                "cache_type": "Old Cache (2048-d Early VLM)",
                "ckpt_path": os.path.join(cross_base, f"{ds}_tcn_pro_old_cache.pt"),
                "loader_fn": lambda: UniversalFoundationTCN_Pro(
                    hidden_dim=128, num_heads=8, num_layers=2,
                    kernel_size=5, dilations=(1, 2, 4, 8, 16, 32), dropout=0.1,
                    top_k=5, teacher_dim=2048
                ).to(device),
                "batch_size": 128
            },
        ])

        # --- B. Baseline Ablation Models (4 per dataset) ---
        base_dir = "checkpoints/baseline_ablation"
        all_runs.extend([
            {
                "dataset": ds,
                "group": "Baseline Ablation (1D-CNN 3-Layer)",
                "model_key": "cnn_1d_new_cache",
                "cache_type": "New Cache (1024-d Ultra-VETime)",
                "ckpt_path": os.path.join(base_dir, f"{ds}_cnn_1d_new_cache.pt"),
                "loader_fn": lambda c=C: CNN1DStudent(in_channels=c, hidden_dim=128, num_layers=3, teacher_feat_dim=1024, use_residual=False).to(device),
                "batch_size": 128
            },
            {
                "dataset": ds,
                "group": "Baseline Ablation (1D-CNN 3-Layer)",
                "model_key": "cnn_1d_old_cache",
                "cache_type": "Old Cache (2048-d Early VLM)",
                "ckpt_path": os.path.join(base_dir, f"{ds}_cnn_1d_old_cache.pt"),
                "loader_fn": lambda c=C: CNN1DStudent(in_channels=c, hidden_dim=128, num_layers=3, teacher_feat_dim=2048, use_residual=False).to(device),
                "batch_size": 128
            },
            {
                "dataset": ds,
                "group": "Baseline Ablation (Residual-TCN 5-Layer)",
                "model_key": "res_tcn_new_cache",
                "cache_type": "New Cache (1024-d Ultra-VETime)",
                "ckpt_path": os.path.join(base_dir, f"{ds}_res_tcn_new_cache.pt"),
                "loader_fn": lambda c=C: CNN1DStudent(in_channels=c, hidden_dim=128, num_layers=5, teacher_feat_dim=1024, use_residual=True).to(device),
                "batch_size": 128
            },
            {
                "dataset": ds,
                "group": "Baseline Ablation (Residual-TCN 5-Layer)",
                "model_key": "res_tcn_old_cache",
                "cache_type": "Old Cache (2048-d Early VLM)",
                "ckpt_path": os.path.join(base_dir, f"{ds}_res_tcn_old_cache.pt"),
                "loader_fn": lambda c=C: CNN1DStudent(in_channels=c, hidden_dim=128, num_layers=5, teacher_feat_dim=2048, use_residual=True).to(device),
                "batch_size": 128
            },
        ])

        # --- C. Best Loss-Tuned Model (1 per dataset) ---
        loss_ckpts = {
            "swat": ("checkpoints/loss_ablation/Weight_0.00_SWAT_best.pt",
                     lambda: StageCoupledTCNStudent(stage_dim=32, fusion_dim=128, feat_dim=1024, dropout=0.10).to(device)),
            "gecco": ("checkpoints/loss_ablation/Weight_0.25_GECCO_best.pt",
                      lambda: GeccoSubsystemCoupledTCN(in_channels=9, hidden_dim=48).to(device)),
            "smd": ("checkpoints/loss_ablation/Weight_0.10_SMD_best.pt",
                    lambda: SMDSubsystemCoupledTCN(in_channels=38, hidden_dim=32, teacher_feat_dim=1024, dropout=0.10).to(device)),
            "msl": ("checkpoints/loss_ablation/BCE_Only_MSL_best.pt",
                    lambda: MSLCoupledTCNStudent(in_channels=55, hidden_dim=64).to(device)),
        }
        ckpt_p, l_fn = loss_ckpts[ds]
        all_runs.append({
            "dataset": ds,
            "group": "Best Loss-Tuned Student",
            "model_key": "best_loss_tuned",
            "cache_type": "New Cache (1024-d Ultra-VETime)",
            "ckpt_path": ckpt_p,
            "loader_fn": l_fn,
            "batch_size": 128
        })

    print(f"\nTotal planned model evaluations: {len(all_runs)} runs")

    results_registry = {}
    t_global_start = time.time()

    for idx, run in enumerate(all_runs, 1):
        ds = run["dataset"]
        mkey = run["model_key"]
        group = run["group"]
        ckpt_path = run["ckpt_path"]
        cache_type = run["cache_type"]

        print(f"\n[{idx:2d}/{len(all_runs):2d}] {ds.upper()} | {mkey} ({group})")
        print(f"       Checkpoint: {ckpt_path}")

        if not os.path.exists(ckpt_path):
            print(f"       [ERROR] File missing: {ckpt_path}")
            continue

        test_x = splits[ds]["test_x"]
        test_y = splits[ds]["test_y"]

        model = run["loader_fn"]()
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)

        n_params = sum(p.numel() for p in model.parameters())

        t0 = time.time()
        res = evaluate_continuous_stream(
            model, test_x, test_y,
            window_size=window_size, prefix_len=prefix_len, stride=stride,
            batch_size=run["batch_size"], device=device
        )
        eval_dur = time.time() - t0

        clean_m = {k: v for k, v in res.items() if k != "point_scores"}
        clean_m["params"] = n_params
        clean_m["eval_time_sec"] = eval_dur
        clean_m["latency_ms"] = 1000.0 / max(1e-6, clean_m["throughput_fps"])

        print(f"       AUROC={clean_m['auroc']:.4f} | PR-AUC={clean_m['prauc']:.4f} | "
              f"Best-F1={clean_m['best_f1']:.4f} | Aff-F1={clean_m['aff_f1']:.4f} | "
              f"Aff-P={clean_m['aff_precision']:.4f} | Aff-R={clean_m['aff_recall']:.4f} | "
              f"FPR={clean_m['fpr']*100:.2f}% | Ev-Rec={clean_m['event_recall']*100:.1f}% | "
              f"FPS={clean_m['throughput_fps']:.0f} (Lat: {clean_m['latency_ms']:.2f}ms in {eval_dur:.1f}s)")

        if ds not in results_registry:
            results_registry[ds] = {}

        results_registry[ds][mkey] = {
            "dataset": ds,
            "model_key": mkey,
            "group": group,
            "cache_type": cache_type,
            "checkpoint": ckpt_path,
            "params": n_params,
            "metrics": clean_m
        }

        del model, ckpt, state
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_time = time.time() - t_global_start
    print(f"\n=======================================================")
    print(f"Completed all {len(all_runs)} model evaluations in {total_time/60:.2f} minutes ({total_time:.1f}s)!")
    print(f"=======================================================")

    os.makedirs("results", exist_ok=True)
    out_json = "results/master_comprehensive_evaluation_registry.json"
    with open(out_json, "w") as f:
        json.dump({
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": device,
            "total_models": len(all_runs),
            "total_time_sec": total_time,
            "results": results_registry
        }, f, indent=2)
    print(f"Saved JSON registry to: {out_json}")


if __name__ == "__main__":
    main()
