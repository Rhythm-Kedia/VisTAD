"""Evaluates the 4 Standard Baselines and 4 Best Loss-Tuned Models on Continuous Test Streams.

Outputs:
results/eval_missing_target_models.json
"""

import os
import sys
import time
import json
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.universal_dataset import load_dataset_telemetry
from src.models.cnn_1d import CNN1DStudent
from src.models.stage_coupled_tcn import StageCoupledTCNStudent
from src.models.gecco_coupled_tcn import GeccoSubsystemCoupledTCN
from src.models.smd_coupled_tcn import SMDSubsystemCoupledTCN
from src.models.msl_coupled_tcn import MSLCoupledTCNStudent
from train_universal_foundation import evaluate_continuous_stream


def main():
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating missing target models on: {device}")

    eval_plan = [
        # GECCO
        {
            "dataset": "gecco",
            "model_key": "baseline_res_tcn",
            "model_fn": lambda: CNN1DStudent(in_channels=9, num_layers=5, use_residual=True),
            "ckpt_path": "checkpoints/gecco_gecco_water_tcn_backbone_tcn5_residual_gecco_best.pt",
            "batch_size": 256
        },
        {
            "dataset": "gecco",
            "model_key": "best_loss_tuned",
            "model_fn": lambda: GeccoSubsystemCoupledTCN(in_channels=9, hidden_dim=48),
            "ckpt_path": "checkpoints/loss_ablation/Weight_0.25_GECCO_best.pt",
            "batch_size": 256
        },
        # MSL
        {
            "dataset": "msl",
            "model_key": "baseline_res_tcn",
            "model_fn": lambda: CNN1DStudent(in_channels=55, num_layers=5, use_residual=True),
            "ckpt_path": "checkpoints/msl_msl_test_tcn_backbone_tcn5_residual_msl_best.pt",
            "batch_size": 128
        },
        {
            "dataset": "msl",
            "model_key": "best_loss_tuned",
            "model_fn": lambda: MSLCoupledTCNStudent(in_channels=55, hidden_dim=64),
            "ckpt_path": "checkpoints/loss_ablation/BCE_Only_MSL_best.pt",
            "batch_size": 256
        },
        # SWaT
        {
            "dataset": "swat",
            "model_key": "baseline_res_tcn",
            "model_fn": lambda: CNN1DStudent(in_channels=51, num_layers=5, use_residual=True),
            "ckpt_path": "checkpoints/swat_swat_attack_tcn_backbone_tcn5_residual_swat_best.pt",
            "batch_size": 128
        },
        {
            "dataset": "swat",
            "model_key": "best_loss_tuned",
            "model_fn": lambda: StageCoupledTCNStudent(stage_dim=32, fusion_dim=128, feat_dim=1024, dropout=0.10),
            "ckpt_path": "checkpoints/loss_ablation/Weight_0.00_SWAT_best.pt",
            "batch_size": 256
        },
        # SMD
        {
            "dataset": "smd",
            "model_key": "baseline_res_tcn",
            "model_fn": lambda: CNN1DStudent(in_channels=38, num_layers=5, use_residual=True),
            "ckpt_path": "checkpoints/smd_pooled28_tcn_backbone_tcn5_residual_smd_best.pt",
            "batch_size": 128
        },
        {
            "dataset": "smd",
            "model_key": "best_loss_tuned",
            "model_fn": lambda: SMDSubsystemCoupledTCN(in_channels=38, hidden_dim=32, teacher_feat_dim=1024, dropout=0.10),
            "ckpt_path": "checkpoints/loss_ablation/Weight_0.10_SMD_best.pt",
            "batch_size": 256
        },
    ]

    # Pre-cache telemetry splits
    splits = {}
    for ds in ["gecco", "msl", "swat", "smd"]:
        print(f"Loading {ds} telemetry...")
        raw_data, labels, train_end, C = load_dataset_telemetry(ds, "data")
        splits[ds] = (raw_data[train_end:], labels[train_end:])

    all_results = {}
    for item in eval_plan:
        ds = item["dataset"]
        mkey = item["model_key"]
        ckpt_path = item["ckpt_path"]
        print(f"\n---> Evaluating {ds} | {mkey} ({ckpt_path})")

        test_x, test_y = splits[ds]
        model = item["model_fn"]().to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)

        n_params = sum(p.numel() for p in model.parameters())

        t0 = time.time()
        res = evaluate_continuous_stream(
            model, test_x, test_y,
            window_size=200, prefix_len=0, stride=10,
            batch_size=item["batch_size"], device=device
        )
        eval_dur = time.time() - t0

        clean_m = {k: v for k, v in res.items() if k != "point_scores"}
        clean_m["params"] = n_params
        clean_m["total_eval_wall_sec"] = eval_dur

        print(f"  AUROC={clean_m['auroc']:.4f} | PR-AUC={clean_m['prauc']:.4f} | "
              f"Best-F1={clean_m['best_f1']:.4f} | Aff-F1={clean_m['aff_f1']:.4f} | "
              f"Aff-P={clean_m['aff_precision']:.4f} | Aff-R={clean_m['aff_recall']:.4f} | "
              f"FPR={clean_m['fpr']*100:.2f}% | Ev-Rec={clean_m['event_recall']*100:.1f}% | "
              f"FPS={clean_m['throughput_fps']:.0f} ({eval_dur:.1f}s)")

        if ds not in all_results:
            all_results[ds] = {}
        all_results[ds][mkey] = {
            "dataset": ds,
            "model_key": mkey,
            "checkpoint": ckpt_path,
            "params": n_params,
            "metrics": clean_m
        }

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    os.makedirs("results", exist_ok=True)
    out_file = "results/eval_missing_target_models.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved all results to: {out_file}")


if __name__ == "__main__":
    main()
