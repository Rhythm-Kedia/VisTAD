import os
import time
import json
import yaml
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve

from src.dataset import TimeSeriesDataset, load_dataset_arrays
from src.models import build_student
from src.metrics import compute_point_metrics, compute_affiliation_metrics, compute_dynamic_pot_threshold, compute_vus_metrics

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
    parser.add_argument("--thresh_mode", type=str, default="val", choices=["val", "pot", "config"],
                        help="Operating decision threshold mode: 'val' (validation F1 search tau*), 'pot' (Dynamic POT), or 'config'")
    parser.add_argument("--tag", type=str, default=None, help="Experiment tag suffix")
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
    full_data, full_labels = load_dataset_arrays(
        dataset_name=dataset_name,
        entity=entity,
        data_dir=cfg["dataset"].get("data_dir", "data")
    )
    
    n_total = len(full_data)
    val_start = int(n_total * 0.50)
    test_start = int(n_total * 0.60)
    val_data = full_data[val_start:test_start]
    val_labels = full_labels[val_start:test_start]

    # Slice selection: 'test' slice (last 40% zero-leakage) or full series
    if args.eval_slice == "test":
        raw_data = full_data[test_start:]
        labels = full_labels[test_start:]
        print(f"Evaluation slice: HELD-OUT TEST (last 40%, {len(raw_data)} timestamps)")
    else:
        raw_data = full_data
        labels = full_labels
        print(f"Evaluation slice: FULL SERIES ({len(raw_data)} timestamps)")
        
    bs_steps = cfg["dataset"].get("baseline_steps", 3000)
    base_m = np.mean(full_data[:min(bs_steps, n_total)], axis=0)
    base_s = np.std(full_data[:min(bs_steps, n_total)], axis=0) + 1e-6
    
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
    
    # 1. Validation Threshold Search (tau*) on validation segment [0.50N, 0.60N]
    tau_star = 0.50
    if len(val_data) > w_size:
        val_dataset = TimeSeriesDataset(
            data=val_data,
            labels=val_labels,
            window_size=w_size,
            stride=args.stride,
            base_mean=base_m,
            base_std=base_s,
            filter_strategy=cfg["channel_filter"]["strategy"],
            k=k
        )
        val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
        val_point_scores = np.zeros(len(val_data), dtype=np.float32)
        val_point_counts = np.zeros(len(val_data), dtype=np.float32)
        with torch.no_grad():
            for batch in val_loader:
                vx = batch["x"].to(device)
                vstarts = batch["start_idx"].numpy()
                vout = model(vx)
                vdense = vout["logits_dense"].cpu().numpy()
                for i, st in enumerate(vstarts):
                    val_point_scores[st:st+w_size] += vdense[i]
                    val_point_counts[st:st+w_size] += 1
        val_scores = val_point_scores / np.maximum(val_point_counts, 1.0)
        val_pos = int(np.sum(val_labels > 0))
        if 0 < val_pos < len(val_labels):
            p_v, r_v, th_v = precision_recall_curve(val_labels, val_scores)
            f1_v = 2 * p_v * r_v / np.clip(p_v + r_v, 1e-12, None)
            best_idx = np.argmax(f1_v)
            tau_star = float(th_v[min(best_idx, len(th_v) - 1)])
        else:
            tau_star = float(np.percentile(val_scores, 98.0))
            
        # FPR-Constrained SLA Thresholding (measured on nominal validation points)
        val_nom = val_scores[val_labels == 0]
        if len(val_nom) > 0:
            tau_fpr05 = float(np.percentile(val_nom, 95.0))
            tau_fpr01 = float(np.percentile(val_nom, 99.0))
        else:
            tau_fpr05 = float(np.percentile(val_scores, 95.0))
            tau_fpr01 = float(np.percentile(val_scores, 99.0))
    else:
        tau_fpr05 = 0.50
        tau_fpr01 = 0.50

    # 2. Operating Decision Thresholds
    dec_thresh = cfg["teacher"].get("decision_threshold", 0.50)
    dynamic_threshold = compute_dynamic_pot_threshold(continuous_scores, q=1e-3, level=0.98)
    
    # Validation calibrated tau* (primary submission standard)
    point_m_val = compute_point_metrics(labels, continuous_scores, threshold=tau_star)
    binary_preds_val = (continuous_scores >= tau_star).astype(int)
    aff_m_val = compute_affiliation_metrics(labels, binary_preds_val, smooth=False)
    aff_m_val_smooth = compute_affiliation_metrics(labels, binary_preds_val, smooth=True)

    # FPR-Constrained Metrics (SLA false alarm <= 5%)
    point_m_fpr05 = compute_point_metrics(labels, continuous_scores, threshold=tau_fpr05)
    aff_m_fpr05 = compute_affiliation_metrics(labels, (continuous_scores >= tau_fpr05).astype(int), smooth=False)

    # Dynamic POT
    point_m_pot = compute_point_metrics(labels, continuous_scores, threshold=dynamic_threshold)
    binary_preds_pot = (continuous_scores >= dynamic_threshold).astype(int)
    aff_m_pot = compute_affiliation_metrics(labels, binary_preds_pot, smooth=False)
    aff_m_pot_smooth = compute_affiliation_metrics(labels, binary_preds_pot, smooth=True)

    # Config threshold
    point_m_dec = compute_point_metrics(labels, continuous_scores, threshold=dec_thresh)
    binary_preds_dec = (continuous_scores >= dec_thresh).astype(int)
    aff_m_dec = compute_affiliation_metrics(labels, binary_preds_dec, smooth=False)
    aff_m_dec_smooth = compute_affiliation_metrics(labels, binary_preds_dec, smooth=True)

    vus_m = compute_vus_metrics(labels, continuous_scores)

    active_thresh = tau_star if args.thresh_mode == "val" else (dynamic_threshold if args.thresh_mode == "pot" else dec_thresh)
    active_point_m = point_m_val if args.thresh_mode == "val" else (point_m_pot if args.thresh_mode == "pot" else point_m_dec)
    active_aff_m = aff_m_val if args.thresh_mode == "val" else (aff_m_pot if args.thresh_mode == "pot" else aff_m_dec)
    active_aff_smooth = aff_m_val_smooth if args.thresh_mode == "val" else (aff_m_pot_smooth if args.thresh_mode == "pot" else aff_m_dec_smooth)
    
    print("\n" + "=" * 68)
    print(f"          EVALUATION REPORT: {dataset_name.upper()} ({entity})")
    print("=" * 68)
    print(f"Model Architecture:    {arch} (Layers={num_layers})")
    print(f"Parameter Count:       {param_count:,} params ({model_size_mb:.2f} MB)")
    print(f"Latency Profile:       {ms_per_window:.3f} ms/window | {throughput_pts_sec:,.0f} pts/sec")
    print(f"Active Threshold Mode: {args.thresh_mode.upper()} (threshold = {active_thresh:.4f})")
    print("-" * 68)
    print("POINT-LEVEL CONTINUOUS METRICS:")
    print(f"  AUROC:               {active_point_m['auroc']:.4f}")
    print(f"  PR-AUC:              {active_point_m['pr_auc']:.4f}")
    print(f"  VUS-ROC:             {vus_m['vus_roc']:.4f}")
    print(f"  VUS-PR:              {vus_m['vus_pr']:.4f}")
    print(f"  Best Point F1:       {active_point_m['best_f1']:.4f} (at threshold {active_point_m['best_thresh']:.4f})")
    print("-" * 68)
    print(f"ACTIVE OPERATIONAL METRICS ({args.thresh_mode.upper()}: {active_thresh:.4f}):")
    print(f"  Point Precision:     {active_point_m['precision']:.4f}")
    print(f"  Point Recall:        {active_point_m['recall']:.4f}")
    print(f"  Point F1-Score:      {active_point_m['f1']:.4f}")
    print(f"  Raw Affiliation Prec:{active_aff_m['aff_precision']:.4f}")
    print(f"  Raw Affiliation Rec: {active_aff_m['aff_recall']:.4f}")
    print(f"  Raw Affiliation F1:  {active_aff_m['aff_f1']:.4f}")
    print(f"  Smoothed Aff-Prec:   {active_aff_smooth['aff_precision']:.4f}")
    print(f"  Smoothed Aff-Rec:    {active_aff_smooth['aff_recall']:.4f}")
    print(f"  Smoothed Aff-F1:     {active_aff_smooth['aff_f1']:.4f}")
    print(f"  Mean Temporal IoU:   {active_aff_m['mean_tiou']:.4f}")
    print("-" * 68)
    print(f"SUMMARY OF OPERATING THRESHOLDS:")
    print(f"  Validation tau* ({tau_star:.4f}):      Aff-F1={aff_m_val['aff_f1']:.4f}, Point-F1={point_m_val['f1']:.4f}")
    print(f"  FPR<=0.05 SLA   ({tau_fpr05:.4f}):      Aff-F1={aff_m_fpr05['aff_f1']:.4f}, Point-F1={point_m_fpr05['f1']:.4f}")
    print(f"  Dynamic POT     ({dynamic_threshold:.4f}):      Aff-F1={aff_m_pot['aff_f1']:.4f}, Point-F1={point_m_pot['f1']:.4f}")
    print(f"  Config Default  ({dec_thresh:.4f}):      Aff-F1={aff_m_dec['aff_f1']:.4f}, Point-F1={point_m_dec['f1']:.4f}")
    print("=" * 68 + "\n")
    
    os.makedirs(args.output_dir, exist_ok=True)
    tag_suffix = f"_{args.tag}" if args.tag else ""
    out_filename = f"{dataset_name}_{entity}_{arch}{tag_suffix}_eval.json"
    out_path = os.path.join(args.output_dir, out_filename)
    results = {
        "dataset": dataset_name,
        "entity": entity,
        "architecture": arch,
        "tag": args.tag,
        "num_layers": num_layers,
        "slice": args.eval_slice,
        "thresh_mode": args.thresh_mode,
        "parameters": param_count,
        "model_size_mb": model_size_mb,
        "latency_ms_per_window": ms_per_window,
        "throughput_pts_sec": throughput_pts_sec,
        "point_metrics": active_point_m,
        "vus_metrics": vus_m,
        "active_threshold": active_thresh,
        "tau_star_threshold": tau_star,
        "tau_fpr05_threshold": tau_fpr05,
        "tau_fpr01_threshold": tau_fpr01,
        "dynamic_pot_threshold": dynamic_threshold,
        "config_threshold": dec_thresh,
        "affiliation_metrics_val": aff_m_val,
        "affiliation_metrics_val_smooth": aff_m_val_smooth,
        "affiliation_metrics_fpr05": aff_m_fpr05,
        "point_metrics_fpr05": point_m_fpr05,
        "affiliation_metrics_pot": aff_m_pot,
        "affiliation_metrics_pot_smooth": aff_m_pot_smooth,
        "affiliation_metrics_config": aff_m_dec,
        "affiliation_metrics_config_smooth": aff_m_dec_smooth
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation artifact to: {out_path}")


if __name__ == "__main__":
    main()
