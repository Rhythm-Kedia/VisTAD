"""
build_data_repository.py: Builds a unified, clean, deduplicated, and Kaggle-ready data repository
containing all 4 primary benchmarks (SWaT, SMD, MSL, GECCO) plus pre-computed Teacher Distillation Caches.

Outputs:
1. data/            - Local project data directory for immediate local testing.
2. data_repository/ - Clean, deduplicated directory ready to upload as a Kaggle Dataset.
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
        # Canonical single file: gecco_water.csv
        df.to_csv(os.path.join(out_dir, "gecco_water.csv"), index=False)
        print(f"[GECCO] Wrote -> {out_dir}/gecco_water.csv")


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
    
    df_test = pd.DataFrame(test_x, columns=sensor_cols)
    df_test["attack"] = test_y.astype(int)
    
    df_train = pd.DataFrame(train_x, columns=sensor_cols)
    df_train["attack"] = 0
    
    n_attack = int((df_test["attack"] == 1).sum())
    print(f"[MSL] Test Shape: {df_test.shape} ({len(sensor_cols)} sensors, {len(df_test)} steps, {n_attack} anomalies: {n_attack/len(df_test):.2%})")
    print(f"[MSL] Train Shape: {df_train.shape} ({len(sensor_cols)} sensors, {len(df_train)} steps)")
    
    for t_dir in target_dirs:
        out_dir = os.path.join(t_dir, "msl")
        os.makedirs(out_dir, exist_ok=True)
        # Canonical single files
        df_test.to_csv(os.path.join(out_dir, "msl_test.csv"), index=False)
        df_train.to_csv(os.path.join(out_dir, "msl_train.csv"), index=False)
        shutil.copy2(test_x_p, os.path.join(out_dir, "MSL_test.npy"))
        shutil.copy2(test_y_p, os.path.join(out_dir, "MSL_test_label.npy"))
        shutil.copy2(train_x_p, os.path.join(out_dir, "MSL_train.npy"))
        print(f"[MSL] Wrote -> {out_dir}/ (msl_test.csv, msl_train.csv, and raw .npy files)")


def sync_teacher_cache(cache_src, target_dirs):
    print(f"\n[Cache] Syncing teacher cache from {cache_src}...")
    pt_files = sorted(glob.glob(os.path.join(cache_src, "*_teacher_cache.pt")))
    if not pt_files:
        print("[Cache] No teacher cache files found to sync.")
        return
        
    for t_dir in target_dirs:
        out_dir = os.path.join(t_dir, "cache")
        os.makedirs(out_dir, exist_ok=True)
        for pt in pt_files:
            dst = os.path.join(out_dir, os.path.basename(pt))
            shutil.copy2(pt, dst)
        print(f"[Cache] Copied {len(pt_files)} teacher cache (.pt) files -> {out_dir}/")


def create_dataset_card(repo_dir):
    readme_content = """# VisTAD Unified Benchmark Datasets

This dataset archive contains the preprocessed, verified, deduplicated benchmark telemetry datasets for **VisTAD** (VLM-Guided Distillation for Time-Series Anomaly Detection), including pre-computed Teacher Distillation Caches.

## Directory Structure
```text
.
├── cache/                         # Pre-computed Teacher Distillation Caches (.pt)
│   ├── smd_machine-1-1_teacher_cache.pt
│   ├── smd_machine-1-2_teacher_cache.pt
│   └── ... (28 SMD entities)
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
│   └── SWaT_Dataset_Normal_v1.csv # 7-day nominal continuous telemetry baseline ('attack'=0)
├── msl/                           # NASA Mars Science Laboratory (55 telemetry sensors/actuators)
│   ├── msl_test.csv               # Processed test telemetry with binary 'attack' column
│   ├── msl_train.csv              # Nominal training telemetry ('attack'=0)
│   ├── MSL_test.npy               # Standard benchmark numpy array [73729, 55]
│   ├── MSL_test_label.npy         # Benchmark binary label array [73729]
│   └── MSL_train.npy              # Nominal training numpy array [58317, 55]
└── gecco/                         # GECCO 2018 Water Quality Anomaly Detection (9 sensors)
    └── gecco_water.csv            # Continuous water telemetry with binary 'attack' column
```

---

## Benchmark Specifications

| Benchmark | Domain | Channels | Timesteps | Anomalies | Source |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SMD** | Server Infrastructure | 38 | ~28k / machine | 4.2% – 12.8% | [OmniAnomaly GitHub](https://github.com/NetManAIOps/OmniAnomaly) |
| **SWaT** | Cyber-Physical Water | 51 | 449,919 | 12.13% | [iTrust Labs SUTD](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/) |
| **MSL** | Spacecraft Rover | 55 | 73,729 | 10.53% | [NASA JPL Telemanom](https://github.com/khundman/telemanom) |
| **GECCO** | Drinking Water Quality | 9 | 139,566 | 1.24% | [Zenodo 3884398](https://zenodo.org/records/3884398) |

---

## How to Use in Kaggle

1. Upload this archive (`data_repository.zip`) as a Kaggle Dataset named **`vistad-datasets`**.
2. Attach it to your Kaggle Notebook via **+ Add Data**.
3. In `vistad_kaggle_pipeline.ipynb`, **Step 3** automatically mounts all 4 benchmarks and syncs the pre-computed teacher caches into the local workspace.
"""
    readme_path = os.path.join(repo_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"[Manifest] Created dataset documentation -> {readme_path}")


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    repo_dir = os.path.join(base_dir, "data_repository")
    cache_dir = os.path.join(base_dir, "cache")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(repo_dir, exist_ok=True)
    
    # Clean up redundant legacy duplicate files if present
    for d in [data_dir, repo_dir]:
        for dup in [
            os.path.join(d, "swat", "swat_attack.csv"),
            os.path.join(d, "swat", "swat_normal.csv"),
            os.path.join(d, "gecco", "gecco.csv"),
            os.path.join(d, "msl", "msl.csv"),
        ]:
            if os.path.exists(dup):
                os.remove(dup)
                
    # 1. Process SWaT
    onedrive_dir = os.path.join(base_dir, "OneDrive_1_18-10-2026")
    swat_attack_xlsx = os.path.join(onedrive_dir, "SWaT_Dataset_Attack_v0.xlsx")
    swat_normal_xlsx = os.path.join(onedrive_dir, "SWaT_Dataset_Normal_v1.xlsx")
    
    existing_swat_attack = os.path.join(data_dir, "swat", "SWaT_Dataset_Attack_v0.csv")
    if os.path.exists(existing_swat_attack):
        print(f"\n[SWaT] Found existing CSV at {existing_swat_attack}, ensuring copy in data_repository...")
        target_p = os.path.join(repo_dir, "swat", "SWaT_Dataset_Attack_v0.csv")
        os.makedirs(os.path.dirname(target_p), exist_ok=True)
        if not os.path.exists(target_p):
            shutil.copy2(existing_swat_attack, target_p)
    elif os.path.exists(swat_attack_xlsx):
        process_swat_attack(
            swat_attack_xlsx,
            [
                os.path.join(data_dir, "swat", "SWaT_Dataset_Attack_v0.csv"),
                os.path.join(repo_dir, "swat", "SWaT_Dataset_Attack_v0.csv"),
            ]
        )
        
    existing_swat_normal = os.path.join(data_dir, "swat", "SWaT_Dataset_Normal_v1.csv")
    if os.path.exists(existing_swat_normal):
        target_p = os.path.join(repo_dir, "swat", "SWaT_Dataset_Normal_v1.csv")
        os.makedirs(os.path.dirname(target_p), exist_ok=True)
        if not os.path.exists(target_p):
            shutil.copy2(existing_swat_normal, target_p)
    elif os.path.exists(swat_normal_xlsx):
        process_swat_normal(
            swat_normal_xlsx,
            [
                os.path.join(data_dir, "swat", "SWaT_Dataset_Normal_v1.csv"),
                os.path.join(repo_dir, "swat", "SWaT_Dataset_Normal_v1.csv"),
            ]
        )
        
    # 2. Fetch SMD dataset
    fetch_smd_dataset([data_dir, repo_dir])
    
    # 3. Process GECCO dataset
    process_gecco_dataset([data_dir, repo_dir])
    
    # 4. Process MSL dataset
    process_msl_dataset([data_dir, repo_dir])
    
    # 5. Sync Teacher Cache files into data_repository
    if os.path.exists(cache_dir):
        sync_teacher_cache(cache_dir, [repo_dir])
        
    # 6. Create manifest
    create_dataset_card(repo_dir)
    
    # 7. Create Kaggle ready zip archive
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
