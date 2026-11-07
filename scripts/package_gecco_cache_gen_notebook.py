"""Packages VisTAD codebase into a self-contained Kaggle notebook for GECCO unconstrained cache generation."""

import os
import json
import base64
import io
import zipfile


def build_codebase_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for root_dir in ["src", "vlm4ts", "configs"]:
            for dirpath, _, filenames in os.walk(root_dir):
                if "__pycache__" in dirpath or ".DS_Store" in dirpath:
                    continue
                for f in filenames:
                    if f.endswith((".py", ".yaml", ".md", ".json")) and not f.startswith("."):
                        fpath = os.path.join(dirpath, f)
                        z.write(fpath, fpath)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def make_notebook(b64_bundle):
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# GECCO Unconstrained Cache Generation: Ultra-VETime Dual-Subsystem Teacher\n",
                    "### High-Density Curriculum Caching & Dedicated Head-to-Head Evaluation Cache\n",
                    "- **Domain**: GECCO 2018 Industrial Challenge (Drinking Water Quality Telemetry, 139,566 steps, 9 channels)\n",
                    "- **Backbone**: `facebook/dinov2-large` (1024-dim self-supervised visual patch features)\n",
                    "- **Dual-Subsystem 4-Panel Dashboard**: Chemical Disinfection (ORP/pH/Cl) + Physical Quality (Leit/Trueb) on dedicated scales\n",
                    "- **Multi-Regime Operational Centroids**: $K=5$ KMeans nominal operating clusters from 16,000-step baseline\n",
                    "- **Rendering**: Fast direct Agg canvas buffer copy (zero PNG compression bottleneck)\n",
                    "- **Multi-GPU**: Shards candidate windows across `cuda:0` and `cuda:1` if dual GPUs are provisioned"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# --- Step 1: Workspace Setup & Codebase Unpacking ---\n",
                    "import os, sys, shutil, subprocess, base64, io, zipfile\n",
                    "\n",
                    "repo_dir = '/kaggle/working/VisTAD'\n",
                    "os.makedirs(repo_dir, exist_ok=True)\n",
                    "os.chdir(repo_dir)\n",
                    "\n",
                    "print('Unpacking self-contained VisTAD codebase (src, vlm4ts, configs, data)...')\n",
                    f"b64_bundle = \"{b64_bundle}\"\n",
                    "raw_zip = base64.b64decode(b64_bundle)\n",
                    "with zipfile.ZipFile(io.BytesIO(raw_zip), 'r') as z:\n",
                    "    z.extractall(repo_dir)\n",
                    "print('VisTAD codebase unpacked successfully.')\n",
                    "print('Extracted top-level dirs:', os.listdir(repo_dir))\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# --- Step 2: GPU Accelerator & Multi-GPU Diagnostics ---\n",
                    "import torch\n",
                    "print(f'PyTorch Version: {torch.__version__}')\n",
                    "print(f'CUDA Available: {torch.cuda.is_available()}')\n",
                    "num_gpus = torch.cuda.device_count()\n",
                    "print(f'Detected {num_gpus} GPU(s):')\n",
                    "for i in range(num_gpus):\n",
                    "    props = torch.cuda.get_device_properties(i)\n",
                    "    print(f'  GPU {i}: {props.name} ({props.total_memory / (1024**3):.1f} GB VRAM)')\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# --- Step 3: GECCO Dataset Auto-Discovery ---\n",
                    "import os, glob, shutil\n",
                    "os.makedirs('data/gecco', exist_ok=True)\n",
                    "gecco_target = 'data/gecco/gecco_water.csv'\n",
                    "\n",
                    "candidates = glob.glob('/kaggle/input/**/gecco_water.csv', recursive=True)\n",
                    "if not candidates:\n",
                    "    candidates = glob.glob('/kaggle/input/**/gecco/*.csv', recursive=True)\n",
                    "if candidates:\n",
                    "    print(f'Found GECCO dataset candidate: {candidates[0]}')\n",
                    "    if not os.path.exists(gecco_target):\n",
                    "        shutil.copy2(candidates[0], gecco_target)\n",
                    "else:\n",
                    "    print('WARNING: Could not find gecco_water.csv in /kaggle/input!')\n",
                    "print(f'GECCO data file verified at {gecco_target}: {os.path.getsize(gecco_target)/(1024**2):.2f} MB')\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# --- Step 4: Execute GECCO Unconstrained Cache Generation ---\n",
                    "import os, subprocess, time\n",
                    "os.environ['PYTHONPATH'] = '.'\n",
                    "\n",
                    "cmd = [\n",
                    "    'python', 'vlm4ts/generate_gecco_cache.py',\n",
                    "    '--data_path', 'data/gecco/gecco_water.csv',\n",
                    "    '--output_train_cache', 'cache/gecco_train_ultravetime_cache.pt',\n",
                    "    '--output_eval_cache', 'cache/gecco_eval_head_to_head_cache.pt',\n",
                    "    '--window_size', '200',\n",
                    "    '--num_train_windows', '3500',\n",
                    "    '--num_eval_windows', '1500',\n",
                    "    '--model_id', 'facebook/dinov2-large',\n",
                    "    '--n_regimes', '5'\n",
                    "]\n",
                    "print('Executing unconstrained caching command:', ' '.join(cmd))\n",
                    "t0 = time.time()\n",
                    "proc = subprocess.run(cmd, check=True)\n",
                    "duration = time.time() - t0\n",
                    "print(f'Caching completed successfully in {duration:.1f}s ({duration/60:.2f} min).')\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# --- Step 5: Verify Cache Statistics & Teacher Baseline Evaluation ---\n",
                    "import torch, os\n",
                    "import numpy as np\n",
                    "from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve\n",
                    "\n",
                    "for cache_path, label in [\n",
                    "    ('cache/gecco_train_ultravetime_cache.pt', 'TRAINING CURRICULUM CACHE'),\n",
                    "    ('cache/gecco_eval_head_to_head_cache.pt', 'HEAD-TO-HEAD EVALUATION CACHE')\n",
                    "]:\n",
                    "    if not os.path.exists(cache_path):\n",
                    "        print(f'WARNING: {cache_path} not found!')\n",
                    "        continue\n",
                    "    c = torch.load(cache_path, weights_only=False)\n",
                    "    wins = c['windows']\n",
                    "    meta = c.get('metadata', {})\n",
                    "    y_flags = [wins[k]['y_flag'] for k in wins]\n",
                    "    t_scores = [wins[k]['teacher_score'] for k in wins]\n",
                    "    feat_dim = wins[list(wins.keys())[0]]['z_teacher'].shape\n",
                    "    \n",
                    "    print(f'=== {label} ===')\n",
                    "    print(f'  Path:             {cache_path} ({os.path.getsize(cache_path)/(1024**2):.2f} MB)')\n",
                    "    print(f'  Teacher:          {meta.get(\"teacher_name\")} ({meta.get(\"backbone\")})')\n",
                    "    print(f'  Total Windows:    {len(wins)}')\n",
                    "    print(f'  Anomalous:        {sum(y_flags)} ({100*sum(y_flags)/len(wins):.1f}%)')\n",
                    "    print(f'  Nominal:          {len(wins) - sum(y_flags)} ({100*(len(wins)-sum(y_flags))/len(wins):.1f}%)')\n",
                    "    print(f'  Feature Dim:      {feat_dim}')\n",
                    "    \n",
                    "    try:\n",
                    "        auroc = roc_auc_score(y_flags, t_scores)\n",
                    "        prauc = average_precision_score(y_flags, t_scores)\n",
                    "        p, r, th = precision_recall_curve(y_flags, t_scores)\n",
                    "        f1s = 2 * p * r / np.clip(p + r, 1e-12, None)\n",
                    "        best_f1 = np.max(f1s)\n",
                    "        best_th = th[np.argmax(f1s)]\n",
                    "        print(f'  Teacher Baseline Performance on {label}:')\n",
                    "        print(f'    AUROC:          {auroc:.4f}')\n",
                    "        print(f'    PR-AUC:         {prauc:.4f}')\n",
                    "        print(f'    Best F1:        {best_f1:.4f} (at threshold {best_th:.4f})')\n",
                    "    except Exception as e:\n",
                    "        print(f'    Metrics computation note: {e}')\n",
                    "    print()\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# --- Step 6: Stage Outputs for Direct Kaggle Download ---\n",
                    "import shutil\n",
                    "output_dir = '/kaggle/working'\n",
                    "for cp in ['cache/gecco_train_ultravetime_cache.pt', 'cache/gecco_eval_head_to_head_cache.pt']:\n",
                    "    if os.path.exists(cp):\n",
                    "        target = os.path.join(output_dir, os.path.basename(cp))\n",
                    "        print(f'Copying {cp} -> {target}')\n",
                    "        shutil.copy2(cp, target)\n",
                    "print('All cache artifacts staged in /kaggle/working for output retrieval.')\n"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    return nb


def main():
    print("Building codebase zip bundle with GECCO telemetry...")
    bundle_b64 = build_codebase_zip()
    print(f"Base64 bundle size: {len(bundle_b64) / (1024**2):.2f} MB")

    out_nb_path = "kaggle_gecco_cache_gen.ipynb"
    nb = make_notebook(bundle_b64)
    with open(out_nb_path, "w") as f:
        json.dump(nb, f, indent=2)
    print(f"Generated standalone Kaggle notebook -> {out_nb_path} ({os.path.getsize(out_nb_path)/(1024**2):.2f} MB)")


if __name__ == "__main__":
    main()
