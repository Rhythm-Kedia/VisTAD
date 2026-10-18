"""
prepare_swat.py: Formats raw iTrust SWaT physical sensor telemetry for VisTAD.

Steps performed:
1. Locates the attack CSV or Excel file.
2. Strips whitespace from column names.
3. Maps 'Normal/Attack' string column ('Attack' -> 1, 'Normal' -> 0) to 'attack'.
4. Removes Timestamp / non-sensor metadata.
5. Saves clean CSV ready for load_dataset_arrays(dataset_name='swat').
"""

import os
import glob
import argparse
import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare SWaT physical telemetry for VisTAD")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to raw SWaT CSV/XLSX file, or directory containing it (auto-detects if omitted)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/swat/SWaT_Dataset_Attack_v0.csv",
        help="Path to save cleaned CSV"
    )
    return parser.parse_args()


def find_swat_file(input_path=None):
    if input_path and os.path.isfile(input_path):
        return input_path
    
    candidates = []
    search_dirs = [input_path] if input_path and os.path.isdir(input_path) else [
        "data/swat",
        "datasets/swat",
        "/kaggle/input",
        "/kaggle/input/swat-dataset",
        "/kaggle/input/swat",
        "."
    ]
    
    for d in search_dirs:
        if not os.path.exists(d):
            continue
        for root, _, files in os.walk(d):
            for f in files:
                f_lower = f.lower()
                if ("swat" in f_lower or "attack" in f_lower) and (f_lower.endswith(".csv") or f_lower.endswith(".xlsx")):
                    candidates.append(os.path.join(root, f))
                    
    if not candidates:
        raise FileNotFoundError(
            "Could not automatically locate SWaT attack file. "
            "Please specify --input path/to/SWaT_Dataset_Attack_v0.csv"
        )
        
    # Prefer files with 'attack' in the name
    attack_candidates = [c for c in candidates if "attack" in os.path.basename(c).lower()]
    return attack_candidates[0] if attack_candidates else candidates[0]


def main():
    args = parse_args()
    raw_path = find_swat_file(args.input)
    print(f"Reading raw SWaT file: {raw_path}")
    
    if raw_path.endswith(".xlsx") or raw_path.endswith(".xls"):
        df = pd.read_excel(raw_path)
    else:
        df = pd.read_csv(raw_path)
        
    print(f"Raw shape: {df.shape}")
    
    # Strip whitespace from column headers
    df.columns = [str(c).strip() for c in df.columns]
    
    # Identify label column
    label_col = None
    for col in ["Normal/Attack", "normal/attack", "Attack", "attack", "Label", "label"]:
        if col in df.columns:
            label_col = col
            break
            
    if label_col is None:
        raise ValueError(f"Could not find ground-truth attack column in: {list(df.columns)}")
        
    print(f"Detected label column: '{label_col}'")
    
    # Map labels to binary 0/1
    raw_labels = df[label_col].astype(str).str.strip().str.upper()
    binary_labels = (raw_labels == "ATTACK") | (raw_labels == "1") | (raw_labels == "1.0")
    df["attack"] = binary_labels.astype(int)
    
    if label_col != "attack":
        df = df.drop(columns=[label_col])
        
    # Drop timestamp columns
    ts_cols = [c for c in df.columns if c.lower() in ["timestamp", "time", "date", "datetime"]]
    if ts_cols:
        print(f"Dropping timestamp column(s): {ts_cols}")
        df = df.drop(columns=ts_cols)
        
    # Ensure numeric features
    feature_cols = [c for c in df.columns if c != "attack"]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[feature_cols] = df[feature_cols].ffill().bfill().fillna(0.0)
    
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    df.to_csv(args.output, index=False)
    
    n_attacks = int((df['attack'] == 1).sum())
    attack_pct = (n_attacks / len(df)) * 100
    print(f"Successfully processed SWaT dataset!")
    print(f"Saved clean CSV to: {args.output}")
    print(f"Telemetry channels: {len(feature_cols)} (Expected 51)")
    print(f"Total timestamps:   {len(df)}")
    print(f"Attack timesteps:   {n_attacks} ({attack_pct:.2f}%)")


if __name__ == "__main__":
    main()
