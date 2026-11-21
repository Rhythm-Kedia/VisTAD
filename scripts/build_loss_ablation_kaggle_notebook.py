"""
Builds self-contained Kaggle runner notebook for the Loss Component and Weight Ablation Suite:

Tracks:
1. Workspace Setup & Multi-GPU Diagnostics
2. Dataset & Cache Auto-Discovery (/kaggle/input/vistad-data)
3. Experiment 1: Component Ablation Suite (Full vs w/o Focal vs w/o Dice vs w/o Feat vs BCE)
4. Experiment 2: Feature Weight Grid Sweep (w_feat in [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0, 2.0])
5. Experiment 3: Universal Foundation Distillation Ablation (lambda_feat in [0.0, 0.5])
6. Final Grand Milestone Export (JSONs + Markdown Report to /kaggle/working/results)
"""

import json
import io
import os
import zipfile
import base64

def main():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        # 1. src package
        for root, dirs, files in os.walk('src'):
            for f in files:
                if f.endswith('.py'):
                    p = os.path.join(root, f)
                    zf.write(p, arcname=p)

        # 2. Scripts
        py_files = [
            'run_loss_and_weight_ablation.py',
            'train_universal_foundation.py',
            'train_gecco_student.py',
            'train_msl_student.py',
            'train_stage_coupled_student.py',
            'train_smd_pooled_student.py',
            'vlm4ts/gecco_renderer.py',
            'vlm4ts/generate_gecco_cache.py'
        ]
        for f in py_files:
            if os.path.exists(f):
                zf.write(f, arcname=f)

    zip_bytes = buf.getvalue()
    b64_str = base64.b64encode(zip_bytes).decode('ascii')
    print(f"Code bundle created: {len(zip_bytes)/1024:.1f} KB (b64 string: {len(b64_str)/1024:.1f} KB)")

    nb_cells = [
        # Cell 0: Header
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# VisTAD: Loss Component & Feature Weight Ablation Study (Kaggle GPU)\n",
                "\n",
                "**Experimental Objectives:**\n",
                "1. **Experiment 1 (Loss Component Ablation):** Isolates the contribution of Focal Loss, Temporal Dice Loss, and Feature Distillation Loss. Demonstrates why removing any single term triggers optimization instability.\n",
                "2. **Experiment 2 (Feature Weight Grid Sweep):** Sweeps $w_{\\text{feat}} \\in [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0, 2.0]$ to prove that $w_{\\text{feat}} = 0.1$ balances the gradient forces $1:1$.\n",
                "3. **Experiment 3 (Universal Foundation Distillation Ablation):** Evaluates zero-shot cross-domain transfer with vs. without teacher feature distillation ($\\lambda_{\\text{feat}} = 0.0$ vs. $0.5$).\n",
                "4. **Grand Artifact Export:** Emits full epoch training trajectories, confusion matrices, and metrics to structured JSON files.\n"
            ]
        },
        # Cell 1: Workspace Setup
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# --- Step 1: Workspace Setup & Codebase Unpacking ---\n",
                "import os, sys, shutil, subprocess, base64, io, zipfile\n",
                "\n",
                "repo_dir = '/kaggle/working/UniFoundation'\n",
                "os.makedirs(repo_dir, exist_ok=True)\n",
                "os.chdir(repo_dir)\n",
                "\n",
                "print('Unpacking self-contained ablation suite codebase...')\n",
                f'b64_bundle = "{b64_str}"\n',
                "raw_zip = base64.b64decode(b64_bundle)\n",
                "with zipfile.ZipFile(io.BytesIO(raw_zip)) as zf:\n",
                "    zf.extractall('.')\n",
                "print('Codebase unpacked successfully.')\n",
                "print(f'Extracted top-level files: {sorted(os.listdir(\".\"))}')\n"
            ]
        },
        # Cell 2: GPU Diagnostics
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# --- Step 2: GPU Accelerator Diagnostics ---\n",
                "import torch\n",
                "print(f'PyTorch Version: {torch.__version__}')\n",
                "print(f'CUDA Available: {torch.cuda.is_available()}')\n",
                "if torch.cuda.is_available():\n",
                "    n_gpus = torch.cuda.device_count()\n",
                "    print(f'Detected {n_gpus} GPU(s):')\n",
                "    for i in range(n_gpus):\n",
                "        prop = torch.cuda.get_device_properties(i)\n",
                "        print(f'  [GPU {i}] {prop.name} | VRAM: {prop.total_memory / 1e9:.2f} GB')\n"
            ]
        },
        # Cell 3: Data & Cache Auto-Discovery
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# --- Step 3: Multi-Dataset Auto-Discovery & Path Linking ---\n",
                "import os, glob, shutil\n",
                "os.makedirs('data/swat', exist_ok=True)\n",
                "os.makedirs('data/smd/test', exist_ok=True)\n",
                "os.makedirs('data/smd/test_label', exist_ok=True)\n",
                "os.makedirs('data/gecco', exist_ok=True)\n",
                "os.makedirs('data/msl', exist_ok=True)\n",
                "os.makedirs('cache', exist_ok=True)\n",
                "os.makedirs('results/ablations', exist_ok=True)\n",
                "\n",
                "print('Searching for datasets and caches in /kaggle/input/ ...')\n",
                "# Link caches from input datasets\n",
                "for root, dirs, files in os.walk('/kaggle/input'):\n",
                "    for f in files:\n",
                "        if f.endswith('.pt') and 'cache' in f:\n",
                "            src_p = os.path.join(root, f)\n",
                "            dst_p = os.path.join('cache', f)\n",
                "            if not os.path.exists(dst_p):\n",
                "                try:\n",
                "                    os.symlink(src_p, dst_p)\n",
                "                    print(f'Linked cache: {f}')\n",
                "                except Exception:\n",
                "                    shutil.copy2(src_p, dst_p)\n",
                "                    print(f'Copied cache: {f}')\n",
                "        elif f == 'gecco_water.csv':\n",
                "            dst_p = 'data/gecco/gecco_water.csv'\n",
                "            if not os.path.exists(dst_p):\n",
                "                shutil.copy2(os.path.join(root, f), dst_p)\n",
                "        elif 'SWaT_Dataset' in f:\n",
                "            dst_p = os.path.join('data/swat', f)\n",
                "            if not os.path.exists(dst_p):\n",
                "                shutil.copy2(os.path.join(root, f), dst_p)\n",
                "\n",
                "print(f'Available caches: {sorted(os.listdir(\"cache\"))}')\n"
            ]
        },
        # Cell 4: Execute Loss Component and Weight Ablation
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# --- Step 4: Execute Loss Component & Weight Ablation Suite ---\n",
                "import os, subprocess, time\n",
                "os.environ['PYTHONPATH'] = '.'\n",
                "\n",
                "cmd = [\n",
                "    sys.executable, 'run_loss_and_weight_ablation.py',\n",
                "    '--dataset', 'all',\n",
                "    '--cache_dir', 'cache',\n",
                "    '--data_dir', 'data',\n",
                "    '--output_dir', 'results/ablations',\n",
                "    '--epochs', '15',\n",
                "    '--batch_size', '64',\n",
                "    '--lr', '1e-3'\n",
                "]\n",
                "\n",
                "print('Starting Loss and Weight Ablation Execution...')\n",
                "t0 = time.time()\n",
                "res = subprocess.run(cmd, capture_output=False)\n",
                "elapsed = time.time() - t0\n",
                "print(f'Ablation Suite Finished in {elapsed:.1f}s with exit code {res.returncode}')\n"
            ]
        },
        # Cell 5: Universal Foundation Loss Ablation
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# --- Step 5: Universal Foundation Model Distillation Ablation ---\n",
                "# Compares zero-shot transfer with teacher feature distillation vs standalone\n",
                "import os, subprocess, json\n",
                "os.environ['PYTHONPATH'] = '.'\n",
                "\n",
                "uni_ablation_results = []\n",
                "for lambda_f in [0.0, 0.5]:\n",
                "    run_tag = f'uni_distill_lambda_{lambda_f:.1f}'\n",
                "    print(f'\\n[Universal Foundation] Training LODO Fold 1 with lambda_feat = {lambda_f}...')\n",
                "    cmd = [\n",
                "        sys.executable, 'train_universal_foundation.py',\n",
                "        '--train_datasets', 'smd,gecco,msl',\n",
                "        '--eval_dataset', 'gecco',\n",
                "        '--epochs', '5',\n",
                "        '--batch_size', '32',\n",
                "        '--distill_feat_weight', str(lambda_f),\n",
                "        '--distill_score_weight', '0.0',\n",
                "        '--checkpoint_dir', f'checkpoints/{run_tag}',\n",
                "        '--output_json', f'results/ablations/{run_tag}.json'\n",
                "    ]\n",
                "    subprocess.run(cmd, check=False)\n",
                "    json_out = f'results/ablations/{run_tag}.json'\n",
                "    if os.path.exists(json_out):\n",
                "        with open(json_out) as f:\n",
                "            data = json.load(f)\n",
                "            data['lambda_feat'] = lambda_f\n",
                "            uni_ablation_results.append(data)\n",
                "\n",
                "uni_summary_path = 'results/ablations/universal_foundation_distill_ablation.json'\n",
                "with open(uni_summary_path, 'w') as f:\n",
                "    json.dump(uni_ablation_results, f, indent=2)\n",
                "print(f'[Universal Foundation] Ablation summary saved to: {uni_summary_path}')\n"
            ]
        },
        # Cell 6: Grand Milestone Export
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# --- Step 6: Grand Milestone Checkpoint & Export ---\n",
                "import os, glob, shutil, json\n",
                "\n",
                "out_export = '/kaggle/working/results'\n",
                "os.makedirs(out_export, exist_ok=True)\n",
                "\n",
                "# Copy all JSON results and Markdown reports to permanent results directory\n",
                "for f in glob.glob('results/**/*.json', recursive=True):\n",
                "    dst = os.path.join(out_export, os.path.basename(f))\n",
                "    shutil.copy2(f, dst)\n",
                "    print(f'Exported JSON: {os.path.basename(f)}')\n",
                "\n",
                "for f in glob.glob('results/**/*.md', recursive=True):\n",
                "    dst = os.path.join(out_export, os.path.basename(f))\n",
                "    shutil.copy2(f, dst)\n",
                "    print(f'Exported Markdown: {os.path.basename(f)}')\n",
                "\n",
                "print('\\nGrand Milestone Export Completed. All outputs preserved for download.')\n"
            ]
        }
    ]

    nb_dict = {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    out_nb = "kaggle_loss_ablation.ipynb"
    with open(out_nb, "w") as f:
        json.dump(nb_dict, f, indent=1)

    print(f"[Done] Generated self-contained Kaggle notebook: {out_nb} ({os.path.getsize(out_nb)/1024:.1f} KB)")


if __name__ == "__main__":
    main()
