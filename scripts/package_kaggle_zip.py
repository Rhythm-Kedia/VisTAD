import os
import zipfile
import json

zip_path = "kaggle_cross_ablation_bundle.zip"

readme_lines = [
    "# VisTAD: 6-Model x 4-Dataset Cross-Ablation Study (Kaggle Package)",
    "",
    "This package contains everything required to reproduce the **6-Model x 4-Dataset Student Architecture x Teacher Cache Cross-Ablation** on Kaggle GPU.",
    "",
    "---",
    "",
    "## What This Experiment Runs",
    "Across **4 Datasets** (SWaT, SMD, GECCO, MSL), evaluates **6 model configurations** (24 total runs) on common held-out continuous test streams:",
    "",
    "1. `old_student_new_cache`: Domain-Coupled TCN + New Cache (1024-d Ultra-VETime / DINOv2)",
    "2. `old_student_old_cache`: Domain-Coupled TCN + Old Cache (2048-d Early VLM)",
    "3. `tcn_pro_new_cache`: UniversalFoundationTCN_Pro + New Cache (1024-d Ultra-VETime)",
    "4. `tcn_pro_old_cache`: UniversalFoundationTCN_Pro + Old Cache (2048-d Early VLM)",
    "5. `channel_aware_new_cache`: UniversalTCN_ChannelAware (Learned E_ch) + New Cache (1024-d)",
    "6. `channel_aware_old_cache`: UniversalTCN_ChannelAware (Learned E_ch) + Old Cache (2048-d)",
    "",
    "### Metrics Computed",
    "- **Range-Aware Affiliation F1 (Aff F1)**, Affiliation Precision, Affiliation Recall, Mean Temporal IoU (tIoU)",
    "- Point-Adjusted / Pointwise AUROC, PR-AUC, Best F1, False Positive Rate (FPR), and Event Recall",
    "",
    "---",
    "",
    "## How to Run on Kaggle",
    "",
    "### Method A: Kaggle Web UI (Zero Setup)",
    "1. Go to https://www.kaggle.com/code and click 'New Notebook'.",
    "2. Click File -> Upload Notebook and select `kaggle_cross_compatibility.ipynb`.",
    "3. In the right-hand panel (Notebook options):",
    "   - Accelerator: GPU T4 x2 (or GPU P100)",
    "   - Internet: On",
    "4. In the Input section, click 'Add Input' and attach these two Kaggle Datasets:",
    "   - `rhythmkedia/vistad-data` (Telemetry data for SWaT, SMD, GECCO, MSL)",
    "   - `rhythmkedia/vistad-novelty-caches` (1024-d and 2048-d teacher distillation caches)",
    "5. Click 'Run All' (or 'Save Version -> Save & Run All (Commit)' for headless background execution).",
    "",
    "---",
    "",
    "### Method B: Kaggle CLI (Headless Push)",
    "If you have the kaggle CLI configured (~/.kaggle/kaggle.json):",
    "1. Place `kaggle_cross_compatibility.ipynb` and `kernel-metadata.json` in a folder.",
    "2. Edit `kernel-metadata.json` to change the username in 'id' to your own Kaggle username (e.g., 'your-username/cross-compatibility-ablation').",
    "3. Push headlessly:",
    "   kaggle kernels push -p .",
    "   kaggle kernels status <your-username>/cross-compatibility-ablation",
    "4. When finished, retrieve the outputs:",
    "   kaggle kernels output <your-username>/cross-compatibility-ablation -p ./results/",
    "",
    "---",
    "",
    "## Output Artifacts",
    "When complete, the run exports into /kaggle/working/:",
    "- `results/ablations/CROSS_STUDENT_TEACHER_REPORT.md` (Markdown summary table with Aff F1 rankings)",
    "- `results/ablations/cross_student_teacher_ablation.json` (Full metric dictionary for all 24 models)",
    "- `checkpoints/cross_ablation/*.pt` (PyTorch model checkpoints for every configuration)",
    "",
    "---",
    "",
    "## Package Contents",
    "- `kaggle_cross_compatibility.ipynb`: Self-contained notebook with embedded codebase",
    "- `kernel-metadata.json`: Metadata configuration for headless Kaggle CLI execution",
    "- `run_cross_student_teacher_ablation.py`: Standalone Python driver script",
    "- `train_universal_foundation.py`: Training engine & streaming evaluation routines",
    "- `src/`: Complete source tree (models, datasets, affiliation metrics)",
    "- `README.md`: This instruction guide"
]
readme_content = "\n".join(readme_lines) + "\n"

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
    # 1. Notebook & README & Metadata
    zf.writestr("README.md", readme_content)
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
        "dataset_sources": [
            "rhythmkedia/vistad-data",
            "rhythmkedia/vistad-novelty-caches"
        ],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": []
    }
    zf.writestr("kernel-metadata.json", json.dumps(meta, indent=2))
    
    # 2. Driver scripts
    zf.write("run_cross_student_teacher_ablation.py", arcname="run_cross_student_teacher_ablation.py")
    zf.write("train_universal_foundation.py", arcname="train_universal_foundation.py")
    
    # 3. src tree
    for root, dirs, files in os.walk("src"):
        for f in files:
            if f.endswith(".py"):
                p = os.path.join(root, f)
                zf.write(p, arcname=p)

print(f"Created {zip_path}: {os.path.getsize(zip_path)/1024:.1f} KB")
