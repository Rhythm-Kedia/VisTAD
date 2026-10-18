import os
import yaml
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import TimeSeriesDataset, load_dataset_arrays
from src.models import build_student
from src.metrics import compute_point_metrics, compute_affiliation_metrics, compute_dynamic_pot_threshold


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Trained Student on Continuous Benchmark")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name (smd, swat, msl, gecco) (overrides config)")
    parser.add_argument("--entity", type=str, default=None, help="Specific entity to evaluate (overrides config)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file")
    parser.add_argument("--stride", type=int, default=1, help="Evaluation sliding stride (default 1 for continuous)")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
        
    device = cfg["training"].get("device", "cuda") if torch.cuda.is_available() else "cpu"
    
    # 1. Load Data
    dataset_name = args.dataset if args.dataset else cfg["dataset"]["name"]
    entity = args.entity if args.entity else cfg["dataset"].get("entity", "machine-1-1")
    raw_data, labels = load_dataset_arrays(
        dataset_name=dataset_name,
        entity=entity,
        data_dir=cfg["dataset"].get("data_dir", "data")
    )
    bs_steps = cfg["dataset"].get("baseline_steps", 3000)
    base_m = np.mean(raw_data[:bs_steps], axis=0)
    base_s = np.std(raw_data[:bs_steps], axis=0) + 1e-6
    
    num_channels = raw_data.shape[1]
    in_channels = cfg["channel_filter"]["k"] if cfg["channel_filter"]["strategy"] != "none" and num_channels > cfg["channel_filter"]["k"] else num_channels
    
    # 2. Build Model & Load Checkpoint
    model = build_student(
        name=cfg["student"]["architecture"],
        in_channels=in_channels,
        hidden_dim=cfg["student"].get("hidden_dim", 128),
        num_layers=cfg["student"].get("num_layers", 3),
        dropout=cfg["student"].get("dropout", 0.1)
    ).to(device)
    
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = os.path.join(cfg["training"].get("checkpoint_dir", "checkpoints"), f"{cfg['dataset']['name']}_{cfg['student']['architecture']}_best.pt")
        
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded Checkpoint: {ckpt_path}")
    else:
        print(f"Warning: Checkpoint not found at {ckpt_path}. Evaluating initialized weights.")
        
    # 3. Create Continuous Evaluation Dataset
    eval_dataset = TimeSeriesDataset(
        data=raw_data,
        labels=labels,
        window_size=cfg["dataset"]["window_size"],
        stride=args.stride,
        base_mean=base_m,
        base_std=base_s,
        filter_strategy=cfg["channel_filter"]["strategy"],
        k=cfg["channel_filter"]["k"]
    )

    
    eval_loader = DataLoader(eval_dataset, batch_size=128, shuffle=False)
    
    print(f"Running continuous inference on {len(raw_data)} timestamps ({len(eval_dataset)} windows)...")
    model.eval()
    
    # Accumulate continuous point predictions
    point_scores = np.zeros(len(raw_data))
    point_counts = np.zeros(len(raw_data))
    
    w_size = cfg["dataset"]["window_size"]
    with torch.no_grad():
        for batch in eval_loader:
            x = batch["x"].to(device)
            starts = batch["start_idx"].numpy()
            outputs = model(x)
            dense_logits = outputs["logits_dense"].cpu().numpy()  # [B, L]
            
            for b_i, st in enumerate(starts):
                point_scores[st:st+w_size] += dense_logits[b_i]
                point_counts[st:st+w_size] += 1
                
    # Average overlapping predictions
    point_counts = np.maximum(point_counts, 1)
    continuous_scores = point_scores / point_counts
    
    # 4. Dynamic POT Thresholding
    dynamic_threshold = compute_dynamic_pot_threshold(continuous_scores, q=1e-3, level=0.98)
    binary_preds = (continuous_scores >= dynamic_threshold).astype(int)
    
    # 5. Compute Metrics
    point_metrics = compute_point_metrics(labels, continuous_scores, threshold=dynamic_threshold)
    aff_metrics = compute_affiliation_metrics(labels, binary_preds)
    
    print("\n==================================================================")
    print("                    CONTINUOUS BENCHMARK RESULTS                  ")
    print("==================================================================")
    print(f"Dynamic POT Threshold: {dynamic_threshold:.4f}")
    print(f"Point Precision:       {point_metrics['precision']:.4f}")
    print(f"Point Recall:          {point_metrics['recall']:.4f}")
    print(f"Point F1-Score:        {point_metrics['f1']:.4f}")
    print(f"PR-AUC:                {point_metrics['pr_auc']:.4f}")
    print("------------------------------------------------------------------")
    print(f"Affiliation Precision: {aff_metrics['aff_precision']:.4f}")
    print(f"Affiliation Recall:    {aff_metrics['aff_recall']:.4f}")
    print(f"Affiliation F1-Score:  {aff_metrics['aff_f1']:.4f}  <-- [Gold Standard: Zero PA Inflation]")
    print(f"Mean Temporal IoU:     {aff_metrics['mean_tiou']:.4f}")
    print("==================================================================\n")


if __name__ == "__main__":
    main()

