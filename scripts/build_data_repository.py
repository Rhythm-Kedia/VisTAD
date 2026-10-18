"""
build_data_repository.py: Builds a unified, clean, and Kaggle-ready data repository
containing all 4 primary benchmarks: SWaT, SMD, MSL, and GECCO.

Outputs:
1. data/            - Local project data directory for immediate local testing.
2. data_repository/ - Self-contained directory ready to upload as a Kaggle Dataset.
3. data_repository.zip - Zip archive for 1-click drag-and-drop upload to Kaggle.
"""

import os
import glob
import json
import shutil
import zipfile
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np


def process_swat_attack(raw_path, out_paths):
    print(f"\n[SWaT] Processing Attack Dataset: {raw_path}")
    try:
        df = pd.read_excel(raw_path, header=1, engine="calamine")
    except Exception:
        df = pd.read_excel(raw_path, header=1, engine="openpyxl")
        
    df.columns = [str(c).strip() for c in df.columns]
    
    label_col = None
    for col in ["Normal/Attack", "normal/attack", "Attack", "attack", "Label", "label"]:
        if col in df.columns:
            label_col = col
            break
            
    if label_col is None:
        raise ValueError(f"Could not find attack label column in {df.columns}")
        
    print(f"[SWaT] Found label column: '{label_col}'")
    raw_labels = df[label_col].astype(str).str.strip().str.upper()
    df["attack"] = ((raw_labels == "ATTACK") | (raw_labels == "1") | (raw_labels == "1.0")).astype(int)
    
    if label_col != "attack":
        df = df.drop(columns=[label_col])
        
    ts_cols = [c for c in df.columns if c.lower() in ["timestamp", "time", "date", "datetime"]]
    if ts_cols:
        print(f"[SWaT] Dropping timestamp column(s): {ts_cols}")
        df = df.drop(columns=ts_cols)
        
    feat_cols = [c for c in df.columns if c != "attack"]
    for col in feat_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[feat_cols] = df[feat_cols].ffill().bfill().fillna(0.0)
    
    for p in out_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        df.to_csv(p, index=False)
        print(f"[SWaT] Wrote clean CSV -> {p}")
        
    n_attack = int((df["attack"] == 1).sum())
    print(f"[SWaT] Shape: {df.shape} ({len(feat_cols)} sensors, {len(df)} timesteps, {n_attack} attack steps: {n_attack/len(df):.2%})")


def process_swat_normal(raw_path, out_paths):
    print(f"\n[SWaT] Processing Normal Dataset: {raw_path}")
    try:
        df = pd.read_excel(raw_path, header=1, engine="calamine")
    except Exception:
        df = pd.read_excel(raw_path, header=1, engine="openpyxl")
        
    df.columns = [str(c).strip() for c in df.columns]
    
    for col in ["Normal/Attack", "normal/attack", "Attack", "attack", "Label", "label"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    df["attack"] = 0
    
    ts_cols = [c for c in df.columns if c.lower() in ["timestamp", "time", "date", "datetime"]]
    if ts_cols:
        print(f"[SWaT Normal] Dropping timestamp column(s): {ts_cols}")
        df = df.drop(columns=ts_cols)
        
    feat_cols = [c for c in df.columns if c != "attack"]
    for col in feat_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[feat_cols] = df[feat_cols].ffill().bfill().fillna(0.0)
    
    for p in out_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        df.to_csv(p, index=False)
        print(f"[SWaT Normal] Wrote clean CSV -> {p}")
        
    print(f"[SWaT Normal] Shape: {df.shape} ({len(feat_cols)} sensors, {len(df)} nominal timesteps)")


def download_file(url, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    # Use curl with browser user-agent to bypass CDN user-agent filters
    cmd = ["curl", "-sL", "-A", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)", "-o", dst, url]
    subprocess.run(cmd, check=True)


def fetch_smd_dataset(target_dirs):
    print("\n[SMD] Querying OmniAnomaly GitHub repository for Server Machine Dataset entities...")
    api_url = "https://api.github.com/repos/NetManAIOps/OmniAnomaly/contents/ServerMachineDataset/test"
    req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
    
    try:
        with urllib.request.urlopen(req) as resp:
            items = json.loads(resp.read().decode())
            machines = sorted([f["name"].replace(".txt", "") for f in items if f["name"].endswith(".txt")])
    except Exception as e:
        print(f"[SMD] GitHub API check failed ({e}). Using standard 28 entity list.")
        machines = [
            f"machine-{g}-{i}"
            for g, count in [(1, 8), (2, 9), (3, 11)]
            for i in range(1, count + 1)
        ]
        
    print(f"[SMD] Found {len(machines)} entities: {machines[:4]} ... {machines[-2:]}")
    
    tasks = []
    base_raw = "https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/ServerMachineDataset"
    
    for m in machines:
        t_url = f"{base_raw}/test/{m}.txt"
        l_url = f"{base_raw}/test_label/{m}.txt"
        for t_dir in target_dirs:
            tasks.append((t_url, os.path.join(t_dir, "smd", "test", f"{m}.txt")))
            tasks.append((l_url, os.path.join(t_dir, "smd", "test_label", f"{m}.txt")))
            
    print(f"[SMD] Downloading {len(tasks)} files concurrently across {len(target_dirs)} target directories...")
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(download_file, url, dst) for url, dst in tasks]
        for f in futures:
            f.result()
            
    print("[SMD] Download complete.")


def process_gecco_dataset(target_dirs):
    print("\n[GECCO] Processing GECCO 2018 Water Quality Anomaly Detection dataset...")
    raw_gecco = "/tmp/raw_downloads/gecco.csv"
    if not (os.path.exists(raw_gecco) and os.path.getsize(raw_gecco) > 1000000):
        url = "https://zenodo.org/api/records/3884398/files/1_gecco2018_water_quality.csv/content"
        download_file(url, raw_gecco)
        
    df = pd.read_csv(raw_gecco)
    drop_cols = [c for c in ["Unnamed: 0", "Time", "time"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    if "EVENT" in df.columns:
        df["attack"] = df["EVENT"].astype(bool).astype(int)
        df = df.drop(columns=["EVENT"])
    else:
        df["attack"] = 0
        
    sensor_cols = [c for c in df.columns if c != "attack"]
    for col in sensor_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[sensor_cols] = df[sensor_cols].ffill().bfill().fillna(0.0)
    
    n_attack = int((df["attack"] == 1).sum())
    print(f"[GECCO] Shape: {df.shape} ({len(sensor_cols)} sensors, {len(df)} steps, {n_attack} anomalies: {n_attack/len(df):.2%})")
    
    for t_dir in target_dirs:
        out_dir = os.path.join(t_dir, "gecco")
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(os.path.join(out_dir, "gecco_water.csv"), index=False)
        df.to_csv(os.path.join(out_dir, "gecco.csv"), index=False)
        print(f"[GECCO] Wrote -> {out_dir}/gecco_water.csv & gecco.csv")


def process_msl_dataset(target_dirs):
    print("\n[MSL] Processing NASA Mars Science Laboratory (MSL) Telemetry dataset...")
    raw_dir = "/tmp/raw_downloads"
    test_x_p = os.path.join(raw_dir, "MSL_test.npy")
    test_y_p = os.path.join(raw_dir, "MSL_test_label.npy")
    train_x_p = os.path.join(raw_dir, "MSL_train.npy")
    
    base_hf = "https://huggingface.co/datasets/talgonen/scarce_ts/resolve/main/MSL"
    if not os.path.exists(test_x_p):
        download_file(f"{base_hf}/MSL_test.npy", test_x_p)
    if not os.path.exists(test_y_p):
        download_file(f"{base_hf}/MSL_test_label.npy", test_y_p)
    if not os.path.exists(train_x_p):
        download_file(f"{base_hf}/MSL_train.npy", train_x_p)
        
    test_x = np.load(test_x_p)
    test_y = np.load(test_y_p)
    train_x = np.load(train_x_p)
    
    sensor_cols = [f"sensor_{i:02d}" for i in range(test_x.shape[1])]
    
    # Build test dataframe
    df_test = pd.DataFrame(test_x, columns=sensor_cols)
    df_test["attack"] = test_y.astype(int)
    
    # Build train dataframe
    df_train = pd.DataFrame(train_x, columns=sensor_cols)
    df_train["attack"] = 0
    
    n_attack = int((df_test["attack"] == 1).sum())
    print(f"[MSL] Test Shape: {df_test.shape} ({len(sensor_cols)} sensors, {len(df_test)} steps, {n_attack} anomalies: {n_attack/len(df_test):.2%})")
    print(f"[MSL] Train Shape: {df_train.shape} ({len(sensor_cols)} sensors, {len(df_train)} steps)")
    
    for t_dir in target_dirs:
        out_dir = os.path.join(t_dir, "msl")
        os.makedirs(out_dir, exist_ok=True)
        df_test.to_csv(os.path.join(out_dir, "msl_test.csv"), index=False)
        df_test.to_csv(os.path.join(out_dir, "msl.csv"), index=False)
        df_train.to_csv(os.path.join(out_dir, "msl_train.csv"), index=False)
        shutil.copy2(test_x_p, os.path.join(out_dir, "MSL_test.npy"))
        shutil.copy2(test_y_p, os.path.join(out_dir, "MSL_test_label.npy"))
        shutil.copy2(train_x_p, os.path.join(out_dir, "MSL_train.npy"))
        print(f"[MSL] Wrote -> {out_dir}/ (msl_test.csv, msl.csv, msl_train.csv, and raw .npy files)")


def create_dataset_card(repo_dir):
    readme_content = """# VisTAD Unified Benchmark Datasets

This dataset archive contains the preprocessed, verified benchmark telemetry datasets for **VisTAD** (VLM-Guided Distillation for Time-Series Anomaly Detection).

## Directory Structure
```text
.
├── smd/                           # Server Machine Dataset (38 channels, 28 entities)
│   ├── test/                      # Telemetry matrices (comma/space-separated floats, no headers)
│   │   ├── machine-1-1.txt
│   │   ├── machine-1-2.txt
│   │   └── ... (28 machines)
│   └── test_label/                # Ground truth binary labels (0 = nominal, 1 = anomaly)
│       ├── machine-1-1.txt
│       └── ...
├── swat/                          # Secure Water Treatment (51 physical sensors & actuators)
│   ├── SWaT_Dataset_Attack_v0.csv # Physical attack telemetry with binary 'attack' column
│   ├── SWaT_Dataset_Normal_v1.csv # 7-day nominal continuous telemetry baseline ('attack'=0)
│   ├── swat_attack.csv            # Convenient alias for attack telemetry
│   └── swat_normal.csv            # Convenient alias for normal telemetry
├── msl/                           # NASA Mars Science Laboratory (55 telemetry sensors/actuators)
│   ├── msl_test.csv               # Processed test telemetry with binary 'attack' column
│   ├── msl.csv                    # Alias for msl_test.csv
│   ├── msl_train.csv              # Nominal training telemetry ('attack'=0)
│   ├── MSL_test.npy               # Standard benchmark numpy array [73729, 55]
│   ├── MSL_test_label.npy         # Benchmark binary label array [73729]
│   └── MSL_train.npy              # Nominal training numpy array [58317, 55]
└── gecco/                         # GECCO 2018 Water Quality Anomaly Detection (9 sensors)
    ├── gecco_water.csv            # Continuous water telemetry with binary 'attack' column
    └── gecco.csv                  # Convenient alias
```

## Dataset Specifications

### 1. SWaT (Secure Water Treatment)
- **Origin**: iTrust, Centre for Research in Cyber Security, Singapore University of Technology and Design.
- **Channels**: 51 physical process sensors and actuators across 6 stages (P1-P6).
- **Ground Truth**: Column `attack` (1 = cyber-physical attack active, 0 = normal).
- **Timesteps**: 449,919 steps (Attack: 54,621 steps, 12.14%).

### 2. SMD (Server Machine Dataset)
- **Origin**: NetMan AIOps OmniAnomaly Benchmark (collected from 28 machines over 5 weeks).
- **Channels**: 38 continuous metrics per machine (CPU, Memory, Network, Disk).
- **Ground Truth**: Stored in `test_label/` with exact 1-to-1 timestep alignment.

### 3. MSL (Mars Science Laboratory Rover)
- **Origin**: NASA Jet Propulsion Laboratory (JPL) Telemanom benchmark.
- **Channels**: 55 continuous telemetry channels (actuators, sensors, power modules).
- **Ground Truth**: Column `attack` (1 = labeled anomaly, 0 = normal).
- **Timesteps**: 73,729 test steps (Anomalies: 7,766 steps, 10.53%).

### 4. GECCO (Drinking Water Quality)
- **Origin**: Thüringer Fernwasserversorgung & SPOTSeven Lab (GECCO Industrial Challenge 2018).
- **Channels**: 9 water quality sensors: Temperature (Tp), Chlorine (Cl, Cl_2), pH, Redox, Conductivity (Leit), Turbidity (Trueb), Flow Rate (Fm, Fm_2).
- **Ground Truth**: Column `attack` (1 = operational disturbance/anomaly, 0 = nominal).
- **Timesteps**: 139,566 continuous 1-minute steps (Anomalies: 1,726 steps, 1.24%).

## How to Mount in Kaggle
1. Upload this folder or `data_repository.zip` to Kaggle Datasets as **`vistad-datasets`**.
2. In your Kaggle Notebook, click **+ Add Data** and attach `vistad-datasets`.
3. In VisTAD, `src/dataset.py` auto-detects datasets mounted under `/kaggle/input/` automatically!
"""
    readme_path = os.path.join(repo_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"[Manifest] Created dataset documentation -> {readme_path}")


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    repo_dir = os.path.join(base_dir, "data_repository")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(repo_dir, exist_ok=True)
    
    # 1. Process SWaT
    onedrive_dir = os.path.join(base_dir, "OneDrive_1_18-10-2026")
    swat_attack_xlsx = os.path.join(onedrive_dir, "SWaT_Dataset_Attack_v0.xlsx")
    swat_normal_xlsx = os.path.join(onedrive_dir, "SWaT_Dataset_Normal_v1.xlsx")
    
    existing_swat_attack = os.path.join(data_dir, "swat", "SWaT_Dataset_Attack_v0.csv")
    if os.path.exists(existing_swat_attack):
        print(f"\n[SWaT] Found existing CSV at {existing_swat_attack}, ensuring copies and aliases exist...")
        for p in [
            os.path.join(repo_dir, "swat", "SWaT_Dataset_Attack_v0.csv"),
            os.path.join(data_dir, "swat", "swat_attack.csv"),
            os.path.join(repo_dir, "swat", "swat_attack.csv"),
        ]:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            if not os.path.exists(p):
                shutil.copy2(existing_swat_attack, p)
    elif os.path.exists(swat_attack_xlsx):
        process_swat_attack(
            swat_attack_xlsx,
            [
                os.path.join(data_dir, "swat", "SWaT_Dataset_Attack_v0.csv"),
                os.path.join(repo_dir, "swat", "SWaT_Dataset_Attack_v0.csv"),
                os.path.join(data_dir, "swat", "swat_attack.csv"),
                os.path.join(repo_dir, "swat", "swat_attack.csv"),
            ]
        )
        
    existing_swat_normal = os.path.join(data_dir, "swat", "SWaT_Dataset_Normal_v1.csv")
    if os.path.exists(existing_swat_normal):
        for p in [
            os.path.join(repo_dir, "swat", "SWaT_Dataset_Normal_v1.csv"),
            os.path.join(data_dir, "swat", "swat_normal.csv"),
            os.path.join(repo_dir, "swat", "swat_normal.csv"),
        ]:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            if not os.path.exists(p):
                shutil.copy2(existing_swat_normal, p)
    elif os.path.exists(swat_normal_xlsx):
        process_swat_normal(
            swat_normal_xlsx,
            [
                os.path.join(data_dir, "swat", "SWaT_Dataset_Normal_v1.csv"),
                os.path.join(repo_dir, "swat", "SWaT_Dataset_Normal_v1.csv"),
                os.path.join(data_dir, "swat", "swat_normal.csv"),
                os.path.join(repo_dir, "swat", "swat_normal.csv"),
            ]
        )
        
    # 2. Fetch SMD dataset
    fetch_smd_dataset([data_dir, repo_dir])
    
    # 3. Process GECCO dataset
    process_gecco_dataset([data_dir, repo_dir])
    
    # 4. Process MSL dataset
    process_msl_dataset([data_dir, repo_dir])
    
    # 5. Create manifest
    create_dataset_card(repo_dir)
    
    # 6. Create Kaggle ready zip archive
    zip_path = os.path.join(base_dir, "data_repository.zip")
    print(f"\n[Packaging] Building Kaggle zip archive: {zip_path}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(repo_dir):
            for file in files:
                abs_f = os.path.join(root, file)
                rel_f = os.path.relpath(abs_f, repo_dir)
                z.write(abs_f, arcname=rel_f)
                
    zip_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"[Packaging] Finished! Archive size: {zip_mb:.1f} MB.")
    print("=" * 70)
    print(f"Data Repository ready for Kaggle at:")
    print(f"  Folder:  {repo_dir}")
    print(f"  Archive: {zip_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
