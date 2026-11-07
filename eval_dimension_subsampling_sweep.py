"""Dimension Subsampling & Sensor Dropping Stress-Test Sweep.

Evaluates how gracefully the Universal Any-Dimension Model performs when an increasing
fraction of sensor dimensions are randomly removed/dropped from the test telemetry.
"""

import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.models.universal_tcn import UniversalCyberPhysicalTCN
from src.data.universal_dataset import load_dataset_telemetry
from train_universal_foundation import evaluate_continuous_stream


def run_subsampling_sweep(ckpt_path: str, dataset_name: str = "swat",
                          fractions: list[float] = [1.0, 0.75, 0.50, 0.25, 0.10],
                          device: str = "cpu"):
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hidden_dim = ckpt.get("model_config", {}).get("hidden_dim", 48)
    num_heads = ckpt.get("model_config", {}).get("num_heads", 4)
    
    model = UniversalCyberPhysicalTCN(hidden_dim=hidden_dim, num_heads=num_heads).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Loading test stream: [{dataset_name.upper()}]")
    raw, labels, train_end, total_C = load_dataset_telemetry(dataset_name)
    test_x = raw[train_end:]
    test_y = labels[train_end:]
    
    print("\n" + "=" * 90)
    print(f"DIMENSION SUBSAMPLING STRESS-TEST ON [{dataset_name.upper()}] (Total Channels: {total_C})")
    print("=" * 90)
    print(f"{'Fraction':<10} | {'Channels':<10} | {'AUROC':<10} | {'PR-AUC':<10} | {'Best F1':<10} | {'FPR':<8} | {'Events Caught':<15}")
    print("-" * 90)

    sweep_results = []
    np.random.seed(42)

    for frac in fractions:
        k = max(3, int(total_C * frac))
        # Deterministic random subset of channels for reproducibility
        keep_indices = sorted(list(np.random.choice(total_C, size=k, replace=False)))
        
        m = evaluate_continuous_stream(
            model, test_x, test_y,
            window_size=200, stride=50, batch_size=64,
            fixed_channel_indices=keep_indices,
            device=device
        )
        
        ev_str = f"{m['detected_events']}/{m['total_events']} ({m['event_recall']*100:.1f}%)"
        print(f"{frac*100:5.0f}%     | {k:2d}/{total_C:<6d} | {m['auroc']:<10.4f} | {m['prauc']:<10.4f} | {m['best_f1']:<10.4f} | {m['fpr']*100:5.2f}% | {ev_str:<15}")
        
        sweep_results.append({
            "fraction": frac,
            "num_channels": k,
            "total_channels": total_C,
            "metrics": {k: v for k, v in m.items() if k != "point_scores"}
        })

    print("=" * 90)
    out_file = f"results/dimension_subsampling_sweep_{dataset_name}.json"
    with open(out_file, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"Saved sweep payload to: {out_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/universal_gecco,msl_to_swat_best.pt")
    parser.add_argument("--dataset", type=str, default="swat")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    
    dev = args.device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    run_subsampling_sweep(args.ckpt, args.dataset, device=dev)
