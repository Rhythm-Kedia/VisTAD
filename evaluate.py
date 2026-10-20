import os
import time
import json
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
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name (smd, swat, msl, gecco)")
    parser.add_argument("--entity", type=str, default=None, help="Specific entity to evaluate")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file")
    parser.add_argument("--arch", type=str, default=None, help="Student architecture (cnn_1d, tcn, timesnet, bilstm)")
    parser.add_argument("--num_layers", type=int, default=None, help="Number of model layers")
    parser.add_argument("--stride", type=int, default=10, help="Continuous evaluation sliding stride (default: 10)")
    parser.add_argument("--eval_slice", type=str, default="test", choices=["test", "all"], help="Evaluation slice ('test' for held-out, 'all' for full)")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save evaluation JSON")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
        
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Evaluation Compute Device: {device}")
    
    dataset_name = args.dataset if args.dataset else cfg["dataset"]["name"]
    entity = args.entity if args.entity else cfg["dataset"].get("entity", "machine-1-1")
    raw_data, labels = load_dataset_arrays(
        dataset_name=dataset_name,
        entity=entity,
        data_dir=cfg["dataset"].get("data_dir", "data")
    )
    
    # Slice selection: 'test' slice (last 40% zero-leakage) or full series
    if args.eval_slice == "test":
        split_idx = int(len(raw_data) * 0.6)
        raw_data = raw_data[split_idx:]
        labels = labels[split_idx:]
        print(f"Evaluation slice: HELD-OUT TEST (last 40%, {len(raw_data)} timestamps)")
    else:
        print(f"Evaluation slice: FULL SERIES ({len(raw_data)} timestamps)")
        
    bs_steps = cfg["dataset"].get("baseline_steps", 3000)
    base_m = np.mean(raw_data[:min(bs_steps, len(raw_data))], axis=0)
    base_s = np.std(raw_data[:min(bs_steps, len(raw_data))], axis=0) + 1e-6
    
    num_channels = raw_data.shape[1]
    k = cfg["channel_filter"]["k"]
    in_channels = k if cfg["channel_filter"]["strategy"] != "none" and num_channels > k else num_channels
    
    arch = args.arch if args.arch else cfg["student"]["architecture"]
    num_layers = args.num_layers if args.num_layers is not None else cfg["student"].get("num_layers", 3)
    
    model = build_student(
        name=arch,
        in_channels=in_channels,
        hidden_dim=cfg["student"].get("hidden_dim", 128),
        num_layers=num_layers,
        dropout=cfg["student"].get("dropout", 0.1)
    ).to(device)
    
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = os.path.join(cfg["training"].get("checkpoint_dir", "checkpoints"), f"{dataset_name}_{arch}_best.pt")
        
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded Checkpoint: {ckpt_path}")
    else:
        print(f"Warning: Checkpoint not found at {ckpt_path}. Evaluating initialized weights.")
        
    # Model Parameter Profiling
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    
    w_size = cfg["dataset"]["window_size"]
    eval_dataset = TimeSeriesDataset(
        data=raw_data,
        labels=labels,
        window_size=w_size,
        stride=args.stride,
        base_mean=base_m,
        base_std=base_s,
        filter_strategy=cfg["channel_filter"]["strategy"],
        k=k
    )
    
    bs = 256 if device in ["cuda", "mps"] else 64
    eval_loader = DataLoader(eval_dataset, batch_size=bs, shuffle=False)
    
    print(f"Running continuous inference on {len(raw_data)} timestamps ({len(eval_dataset)} windows, stride={args.stride})...")
    model.eval()
    
    point_scores = np.zeros(len(raw_data), dtype=np.float32)
    point_counts = np.zeros(len(raw_data), dtype=np.float32)
    window_scores = []
    window_labels = []
    
    t0 = time.perf_counter()
    with torch.no_grad():
        for batch in eval_loader:
            x = batch["x"].to(device)
            starts = batch["start_idx"].numpy()
            outputs = model(x)
            dense_logits = outputs["logits_dense"].cpu().numpy()  # [B, L]
            scalar_scores = outputs["score_scalar"].cpu().numpy()
            
            window_scores.extend(scalar_scores)
            window_labels.extend(batch["y_flag"].numpy())
            
            for b_i, st in enumerate(starts):
                point_scores[st:st+w_size] += dense_logits[b_i]
                point_counts[st:st+w_size] += 1
    t1 = time.perf_counter()
    
    elapsed = t1 - t0
    ms_per_window = (elapsed / len(eval_dataset)) * 1000.0
    throughput_pts_sec = len(raw_data) / elapsed
    
    point_counts = np.maximum(point_counts, 1.0)
    continuous_scores = point_scores / point_counts
    
    # Operating Decision Threshold (from teacher config or dynamic POT)
    dec_thresh = cfg["teacher"].get("decision_threshold", 0.50)
    dynamic_threshold = compute_dynamic_pot_threshold(continuous_scores, q=1e-3, level=0.98)
    
    point_m = compute_point_metrics(labels, continuous_scores, threshold=dynamic_threshold)
    binary_preds_pot = (continuous_scores >= dynamic_threshold).astype(int)
    aff_m_pot = compute_affiliation_metrics(labels, binary_preds_pot)
    
    binary_preds_dec = (continuous_scores >= dec_thresh).astype(int)
    aff_m_dec = compute_affiliation_metrics(labels, binary_preds_dec)
    
    print("\n" + "=" * 68)
    print(f"          EVALUATION REPORT: {dataset_name.upper()} ({entity})")
    print("=" * 68)
    print(f"Model Architecture:    {arch} (Layers={num_layers})")
    print(f"Parameter Count:       {param_count:,} params ({model_size_mb:.2f} MB)")
    print(f"Latency Profile:       {ms_per_window:.3f} ms/window | {throughput_pts_sec:,.0f} pts/sec")
    print("-" * 68)
    print("POINT-LEVEL CONTINUOUS METRICS:")
    print(f"  AUROC:               {point_m['auroc']:.4f}")
    print(f"  PR-AUC:              {point_m['pr_auc']:.4f}")
    print(f"  Best Point F1:       {point_m['best_f1']:.4f} (at threshold {point_m['best_thresh']:.4f})")
    print("-" * 68)
    print(f"DYNAMIC POT THRESHOLD ({dynamic_threshold:.4f}):")
    print(f"  Point Precision:     {point_m['precision']:.4f}")
    print(f"  Point Recall:        {point_m['recall']:.4f}")
    print(f"  Point F1-Score:      {point_m['f1']:.4f}")
    print(f"  Affiliation Prec:    {aff_m_pot['aff_precision']:.4f}")
    print(f"  Affiliation Recall:  {aff_m_pot['aff_recall']:.4f}")
    print(f"  Affiliation F1:      {aff_m_pot['aff_f1']:.4f}")
    print(f"  Mean Temporal IoU:   {aff_m_pot['mean_tiou']:.4f}")
    print("-" * 68)
    print(f"CONFIG OPERATING THRESHOLD ({dec_thresh:.4f}):")
    print(f"  Affiliation Prec:    {aff_m_dec['aff_precision']:.4f}")
    print(f"  Affiliation Recall:  {aff_m_dec['aff_recall']:.4f}")
    print(f"  Affiliation F1:      {aff_m_dec['aff_f1']:.4f}")
    print(f"  Mean Temporal IoU:   {aff_m_dec['mean_tiou']:.4f}")
    print("=" * 68 + "\n")
    
    os.makedirs(args.output_dir, exist_ok=True)
    out_filename = f"{dataset_name}_{entity}_{arch}_eval.json"
    out_path = os.path.join(args.output_dir, out_filename)
    results = {
        "dataset": dataset_name,
        "entity": entity,
        "architecture": arch,
        "num_layers": num_layers,
        "slice": args.eval_slice,
        "parameters": param_count,
        "model_size_mb": model_size_mb,
        "latency_ms_per_window": ms_per_window,
        "throughput_pts_sec": throughput_pts_sec,
        "point_metrics": point_m,
        "dynamic_pot_threshold": dynamic_threshold,
        "affiliation_metrics_pot": aff_m_pot,
        "affiliation_metrics_config": aff_m_dec
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation artifact to: {out_path}")


if __name__ == "__main__":
    main()
