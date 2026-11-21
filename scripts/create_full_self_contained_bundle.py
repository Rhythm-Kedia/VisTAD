import os
import zipfile
import json
import glob

zip_path = "vistad_cross_ablation_complete_standalone.zip"

readme_text = """# VisTAD: 6-Model x 4-Dataset Complete Standalone Package

This package is **100% self-contained**: it includes the complete codebase, runner notebook, driver scripts, all 4 benchmark telemetry datasets (SWaT, SMD, GECCO, MSL), and all precomputed dual-manifold teacher distillation caches (1024-d Ultra-VETime and 2048-d Early VLM).

No external downloads, API keys, or pre-existing datasets are required.

---

## 1. What This Reproduces
A comprehensive factorial ablation across **4 Datasets** and **6 Model Configurations** (24 total training & streaming evaluation runs):

| Model Key | Student Backbone | Inductive Bias | Teacher Cache Manifold |
| :--- | :--- | :--- | :--- |
| `old_student_new_cache` | Domain-Coupled TCN | Coupled physical channels | New Cache (1024-d Ultra-VETime / DINOv2) |
| `old_student_old_cache` | Domain-Coupled TCN | Coupled physical channels | Old Cache (2048-d Early VLM) |
| `tcn_pro_new_cache` | `UniversalFoundationTCN_Pro` | Permutation-equivariant universal | New Cache (1024-d Ultra-VETime / DINOv2) |
| `tcn_pro_old_cache` | `UniversalFoundationTCN_Pro` | Permutation-equivariant universal | Old Cache (2048-d Early VLM) |
| `channel_aware_new_cache` | `UniversalTCN_ChannelAware` | Learned channel embedding ($E_{\\text{channel}}$) | New Cache (1024-d Ultra-VETime / DINOv2) |
| `channel_aware_old_cache` | `UniversalTCN_ChannelAware` | Learned channel embedding ($E_{\\text{channel}}$) | Old Cache (2048-d Early VLM) |

### Evaluated Metrics
- **Range-Aware Affiliation F1 (`Aff F1`)**, Affiliation Precision, Affiliation Recall, Mean Temporal IoU (`tIoU`)
- Point-Adjusted / Pointwise AUROC, PR-AUC, Best F1, False Positive Rate (FPR), and Event Recall

---

## 2. Running on Kaggle (Step-by-Step for New Users)

If you have a blank Kaggle account with no existing datasets or models:

### Step 1: Upload the Data & Cache as a Kaggle Dataset
1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets) and click **"New Dataset"**.
2. Drag and drop this zip file (`vistad_cross_ablation_complete_standalone.zip`) or upload the `data/` and `cache/` folders.
3. Title it `vistad-standalone-bundle` and click **Create** (make it Private or Public as you prefer).

### Step 2: Upload the Runner Notebook
1. Go to [kaggle.com/code](https://www.kaggle.com/code) and click **"New Notebook"**.
2. Click **File -> Upload Notebook** and select `kaggle_cross_compatibility.ipynb` (extracted from this zip).
3. In the right-hand **Notebook options** panel:
   - **Accelerator**: GPU T4 x2 (or GPU P100)
   - **Internet**: On
4. In the **Input** section (top right):
   - Click **Add Input**.
   - Search for `vistad-standalone-bundle` (the dataset you just created in Step 1) and click **Add**.
5. Click **Run All** (or click **Save Version -> Save & Run All (Commit)** for background execution).

---

## 3. Running Locally or on Any Cloud GPU (Colab, RunPod, Vast.ai, Linux Server)

If running on a machine with a GPU:

```bash
# 1. Unzip the package
unzip vistad_cross_ablation_complete_standalone.zip
cd vistad_cross_ablation_complete_standalone

# 2. Install minimal dependencies
pip install torch torchvision numpy pandas scikit-learn

# 3. Execute the 6-Model x 4-Dataset ablation suite
python run_cross_student_teacher_ablation.py \\
    --data_dir data \\
    --cache_dir cache \\
    --datasets swat gecco smd msl \\
    --epochs 10 \\
    --batch_size 32 \\
    --checkpoint_dir checkpoints/cross_ablation \\
    --output_json results/cross_student_teacher_ablation.json \\
    --output_report results/CROSS_STUDENT_TEACHER_REPORT.md
```

---

## 4. Expected Outputs
Upon completion, the run exports:
- `results/CROSS_STUDENT_TEACHER_REPORT.md`: Formatted Markdown benchmark table comparing all 24 configurations on Aff F1, AUROC, Best F1, and Event Recall.
- `results/cross_student_teacher_ablation.json`: Full numerical metric dictionary.
- `checkpoints/cross_ablation/*.pt`: Saved weights for each trained student checkpoint.
"""

def main():
    print(f"Building complete standalone bundle: {zip_path}...")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # 1. Documentation & Metadata
        zf.writestr("README.md", readme_text)
        zf.write("kaggle_cross_compatibility.ipynb", arcname="kaggle_cross_compatibility.ipynb")
        
        meta = {
            "id": "your-username/cross-compatibility-ablation",
            "title": "cross-compatibility-ablation",
            "code_file": "kaggle_cross_compatibility.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_tpu": "false",
            "enable_internet": "true",
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": []
        }
        zf.writestr("kernel-metadata.json", json.dumps(meta, indent=2))
        
        # 2. Driver Scripts
        zf.write("run_cross_student_teacher_ablation.py", arcname="run_cross_student_teacher_ablation.py")
        zf.write("train_universal_foundation.py", arcname="train_universal_foundation.py")
        
        # 3. Codebase: src/
        for root, dirs, files in os.walk("src"):
            for f in files:
                if f.endswith(".py"):
                    p = os.path.join(root, f)
                    zf.write(p, arcname=p)
        print("Embedded codebase and scripts.")
        
        # 4. Teacher Caches (cache/)
        essential_caches = [
            "swat_swat_attack_teacher_cache.pt",
            "swat_train_ultravetime_cache.pt",
            "swat_eval_head_to_head_cache.pt",
            "gecco_gecco_water_teacher_cache.pt",
            "gecco_train_ultravetime_cache.pt",
            "gecco_eval_head_to_head_cache.pt",
            "smd_machine-1-1_teacher_cache.pt",
            "smd_train_ultravetime_cache.pt",
            "smd_eval_head_to_head_cache.pt",
            "msl_msl_test_teacher_cache.pt",
            "msl_train_ultravetime_cache.pt",
            "msl_eval_head_to_head_cache.pt"
        ]
        for c in essential_caches:
            cp = os.path.join("cache", c)
            if os.path.exists(cp):
                print(f"Adding cache: {c} ({os.path.getsize(cp)/(1024*1024):.1f} MB)")
                zf.write(cp, arcname=f"cache/{c}")
            else:
                print(f"WARNING: Cache file {cp} not found!")

        # 5. Raw Telemetry Data (data/)
        data_files = [
            "data/gecco/gecco_water.csv",
            "data/swat/SWaT_Dataset_Attack_v0.csv",
            "data/swat/SWaT_Dataset_Normal_v1.csv",
            "data/msl/msl_test.csv",
            "data/msl/msl_train.csv"
        ]
        for df in data_files:
            if os.path.exists(df):
                print(f"Adding data: {df} ({os.path.getsize(df)/(1024*1024):.1f} MB)")
                zf.write(df, arcname=df)
            else:
                print(f"WARNING: Data file {df} not found!")
                
        # SMD test and label files
        smd_test_files = sorted(glob.glob("data/smd/test/*.txt"))
        smd_lbl_files = sorted(glob.glob("data/smd/test_label/*.txt"))
        print(f"Adding SMD data: {len(smd_test_files)} test files, {len(smd_lbl_files)} label files")
        for f in smd_test_files + smd_lbl_files:
            zf.write(f, arcname=f)

    final_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"\nBundle created successfully: {zip_path} ({final_size_mb:.2f} MB)")

if __name__ == "__main__":
    main()
