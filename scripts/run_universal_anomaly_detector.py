"""Standalone Universal Cyber-Physical Anomaly Detection CLI.

Drop-in inference on ANY multivariate telemetry stream (any dimension C in [1, inf)):
- Ingests CSV or NPY files
- Automated column type profiling and causal online standardization
- Continuous streaming anomaly score generation
- Per-sensor root-cause attribution (top-K contributing channels per anomaly)
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.universal_tcn import UniversalFoundationTCN_Pro
from src.data.universal_dataset import UniversalWindowDataset
from torch.utils.data import DataLoader


def run_inference(input_file: str, checkpoint_path: str, output_csv: str = None,
                  window_size: int = 200, stride: int = 10, batch_size: int = 64,
                  device: str = "cpu"):
    print(f"Loading Universal Foundation Checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hidden_dim = ckpt.get("model_config", {}).get("hidden_dim", 128)
    num_heads = ckpt.get("model_config", {}).get("num_heads", 8)
    num_layers = ckpt.get("model_config", {}).get("num_layers", 2)
    
    if "channel_transformer.layers.0.mha.in_proj_weight" in ckpt["model_state_dict"]:
        model = UniversalFoundationTCN_Pro(hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers).to(device)
    else:
        from src.models.universal_tcn import UniversalCyberPhysicalTCN_v1
        model = UniversalCyberPhysicalTCN_v1(hidden_dim=hidden_dim, num_heads=num_heads).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Ingesting Telemetry File: {input_file}")
    if input_file.endswith(".csv") or input_file.endswith(".txt"):
        df = pd.read_csv(input_file)
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                        if not c.lower().startswith("time") and not c.lower().startswith("unnamed")]
        raw_data = df[numeric_cols].values.astype(np.float32)
        channel_names = list(numeric_cols)
    else:
        raw_data = np.load(input_file).astype(np.float32)
        channel_names = [f"channel_{i}" for i in range(raw_data.shape[1])]

    T, C = raw_data.shape
    print(f"Telemetry Profile: {T:,} timesteps, {C} continuous channels")

    # Causal online normalization using initial 20% baseline
    calib_end = max(100, int(T * 0.20))
    base_m = np.mean(raw_data[:calib_end], axis=0)
    base_s = np.maximum(np.std(raw_data[:calib_end], axis=0), 0.05)
    std_data = (raw_data - base_m) / base_s

    ds = UniversalWindowDataset(std_data, window_size=window_size, stride=stride, augment_synthetic=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    point_scores = np.zeros(T, dtype=np.float32)
    point_counts = np.zeros(T, dtype=np.float32)
    channel_risk_matrix = np.zeros((T, C), dtype=np.float32)

    print("Executing streaming neural inference across all channels...")
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            st_indices = batch["st"].numpy()
            x = batch["x"].to(device)  # [B, C, L]
            out = model(x)
            
            p_sys = out["p_system"].cpu().numpy()    # [B, L]
            p_ch = out["p_channel"].cpu().numpy()     # [B, C, L]
            
            for i, st in enumerate(st_indices):
                point_scores[st : st + window_size] += p_sys[i]
                channel_risk_matrix[st : st + window_size] += p_ch[i].T
                point_counts[st : st + window_size] += 1.0

    valid = point_counts > 0
    point_scores[valid] /= point_counts[valid]
    channel_risk_matrix[valid] /= point_counts[valid, np.newaxis]
    
    if not valid[0]:
        first_valid = np.where(valid)[0][0]
        point_scores[:first_valid] = point_scores[first_valid]
        channel_risk_matrix[:first_valid] = channel_risk_matrix[first_valid]

    duration = time.time() - t0
    throughput = len(ds) / max(1e-6, duration)
    print(f"Inference Completed: {duration:.2f}s ({throughput:,.1f} windows/sec)")

    # Identify top-3 contributing root-cause channels for each timestep
    top_root_causes = []
    for t in range(T):
        if point_scores[t] > 0.5:
            top_ch_idx = np.argsort(channel_risk_matrix[t])[::-1][:3]
            causes = [f"{channel_names[idx]}({channel_risk_matrix[t, idx]:.2f})" for idx in top_ch_idx]
            top_root_causes.append("; ".join(causes))
        else:
            top_root_causes.append("nominal")

    # Build output dataframe
    out_df = pd.DataFrame({
        "timestep": np.arange(T),
        "anomaly_probability": point_scores,
        "is_anomaly_alert": (point_scores >= 0.5).astype(int),
        "root_cause_attribution": top_root_causes
    })

    if output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        out_df.to_csv(output_csv, index=False)
        print(f"Saved anomaly detection output to: {output_csv}")

    num_alerts = int((point_scores >= 0.5).sum())
    print(f"\nOperational Summary:")
    print(f"  Total Timesteps:     {T:,}")
    print(f"  Anomalous Timesteps: {num_alerts:,} ({num_alerts/T*100:.2f}%)")
    print(f"  Peak Anomaly Score:  {point_scores.max():.4f}")
    return out_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Universal Cyber-Physical Anomaly Detector")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV or NPY telemetry file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/universal_gecco,msl_to_swat_best.pt",
                        help="Path to trained universal foundation checkpoint")
    parser.add_argument("--output", type=str, default="results/universal_predictions.csv",
                        help="Path to save output CSV predictions")
    parser.add_argument("--window_size", type=int, default=200)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    dev = args.device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    run_inference(args.input, args.checkpoint, args.output, args.window_size, args.stride, device=dev)
