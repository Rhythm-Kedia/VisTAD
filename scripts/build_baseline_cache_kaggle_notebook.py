"""Builds self-contained Kaggle & Local runner notebook for Baseline Students (1D-CNN & Res-TCN) × Dual Caches."""

import json
import io
import os
import zipfile
import base64


def main():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        # 1. src package
        for root, dirs, files in os.walk("src"):
            for f in files:
                if f.endswith(".py"):
                    p = os.path.join(root, f)
                    zf.write(p, arcname=p)

        # 2. Driver scripts
        py_files = [
            "run_baseline_cache_ablation.py",
            "train_universal_foundation.py"
        ]
        for f in py_files:
            if os.path.exists(f):
                zf.write(f, arcname=f)

    zip_bytes = buf.getvalue()
    b64_str = base64.b64encode(zip_bytes).decode("ascii")
    print(f"Code bundle created: {len(zip_bytes)/1024:.1f} KB (b64 string: {len(b64_str)/1024:.1f} KB)")

    nb_cells = [
        # Cell 0: Header
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# VisTAD: Baseline Students (1D-CNN & Res-TCN) × Dual Teacher Cache Manifolds\n",
                "\n",
                "**Experimental Goal:**\n",
                "Evaluates conventional time-series baselines (1D-CNN and Residual-TCN) across dual teacher distillation manifolds:\n",
                "1. **New Cache (1024-d Ultra-VETime / DINOv2)**\n",
                "2. **Old Cache (2048-d Early VLM)**\n",
                "\n",
                "Evaluated across all 4 benchmarks: **SWaT, GECCO, SMD, MSL** with Range-Aware **Affiliation F1 (`Aff F1`)** and threshold-free **AUROC**."
            ]
        },
        # Cell 1: Unpack code bundle
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# --- Step 1: Workspace Setup & Codebase Unpacking ---\n",
                "import os, sys, shutil, subprocess, base64, io, zipfile\n",
                "\n",
                "is_kaggle = os.path.exists('/kaggle')\n",
                "repo_dir = '/kaggle/working/BaselineAblation' if is_kaggle else os.path.abspath('BaselineAblation')\n",
                "os.makedirs(repo_dir, exist_ok=True)\n",
                "os.chdir(repo_dir)\n",
                "\n",
                "print(f'Working directory: {os.getcwd()}')\n",
                "print('Unpacking self-contained Baseline Ablation codebase...')\n",
                f'b64_bundle = "{b64_str}"\n',
                "raw_zip = base64.b64decode(b64_bundle)\n",
                "with zipfile.ZipFile(io.BytesIO(raw_zip), 'r') as zf:\n",
                "    zf.extractall('.')\n",
                "print(f'Extracted top-level files: {sorted(os.listdir(\".\"))}')\n"
            ]
        },
        # Cell 2: Accelerator Diagnostics
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# --- Step 2: Accelerator Diagnostics ---\n",
                "import torch\n",
                "print(f'PyTorch Version: {torch.__version__}')\n",
                "print(f'CUDA Available: {torch.cuda.is_available()}')\n",
                "if torch.cuda.is_available():\n",
                "    n_gpus = torch.cuda.device_count()\n",
                "    print(f'Detected {n_gpus} CUDA GPU(s):')\n",
                "    for i in range(n_gpus):\n",
                "        prop = torch.cuda.get_device_properties(i)\n",
                "        print(f'  [GPU {i}] {prop.name} | VRAM: {prop.total_memory / 1e9:.2f} GB')\n",
                "elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():\n",
                "    print('Apple Silicon MPS accelerator available.')\n",
                "else:\n",
                "    print('Running on CPU.')\n"
            ]
        },
        # Cell 3: Data & Cache Linking
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# --- Step 3: Multi-Dataset & Dual-Cache Auto-Discovery ---\n",
                "import os, glob, shutil\n",
                "\n",
                "for ds in ['swat', 'smd/test', 'smd/test_label', 'gecco', 'msl']:\n",
                "    os.makedirs(os.path.join('data', ds), exist_ok=True)\n",
                "os.makedirs('cache', exist_ok=True)\n",
                "os.makedirs('checkpoints/baseline_ablation', exist_ok=True)\n",
                "os.makedirs('results/ablations', exist_ok=True)\n",
                "\n",
                "# Link Data Files from Kaggle or local parent directories\n",
                "search_roots = ['/kaggle/input', '..', '.']\n",
                "for root_dir in search_roots:\n",
                "    if not os.path.exists(root_dir): continue\n",
                "    # SWaT\n",
                "    for p in glob.glob(os.path.join(root_dir, '**/SWaT_Dataset_*.csv'), recursive=True):\n",
                "        target = os.path.join('data/swat', os.path.basename(p))\n",
                "        if not os.path.exists(target): shutil.copy2(p, target)\n",
                "    # SMD\n",
                "    for p in glob.glob(os.path.join(root_dir, '**/smd/test/*.txt'), recursive=True):\n",
                "        target = os.path.join('data/smd/test', os.path.basename(p))\n",
                "        if not os.path.exists(target): shutil.copy2(p, target)\n",
                "    for p in glob.glob(os.path.join(root_dir, '**/smd/test_label/*.txt'), recursive=True):\n",
                "        target = os.path.join('data/smd/test_label', os.path.basename(p))\n",
                "        if not os.path.exists(target): shutil.copy2(p, target)\n",
                "    # GECCO\n",
                "    for p in glob.glob(os.path.join(root_dir, '**/gecco_water.csv'), recursive=True):\n",
                "        target = os.path.join('data/gecco', os.path.basename(p))\n",
                "        if not os.path.exists(target): shutil.copy2(p, target)\n",
                "    # MSL\n",
                "    for p in glob.glob(os.path.join(root_dir, '**/msl_*.csv'), recursive=True):\n",
                "        target = os.path.join('data/msl', os.path.basename(p))\n",
                "        if not os.path.exists(target): shutil.copy2(p, target)\n",
                "    # Teacher Caches\n",
                "    for p in glob.glob(os.path.join(root_dir, '**/*.pt'), recursive=True):\n",
                "        fname = os.path.basename(p)\n",
                "        if 'cache' in fname.lower() or 'ultravetime' in fname.lower():\n",
                "            target = os.path.join('cache', fname)\n",
                "            if not os.path.exists(target):\n",
                "                try: os.symlink(p, target)\n",
                "                except OSError: shutil.copy2(p, target)\n",
                "\n",
                "print(f'Linked data: SWaT={len(os.listdir(\"data/swat\"))}, SMD={len(os.listdir(\"data/smd/test\"))}, GECCO={len(os.listdir(\"data/gecco\"))}, MSL={len(os.listdir(\"data/msl\"))}')\n",
                "print(f'Discovered caches in cache/: {sorted(os.listdir(\"cache\"))}')\n"
            ]
        },
        # Cell 4: Execute Experiment
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# --- Step 4: Execute Baseline Cache Ablation Suite ---\n",
                "import os, sys, subprocess, time\n",
                "os.environ['PYTHONPATH'] = '.'\n",
                "\n",
                "# Configure models to run: 'cnn_1d', 'res_tcn', or both\n",
                "models_to_run = ['cnn_1d', 'res_tcn']  # or ['cnn_1d']\n",
                "datasets_to_run = ['swat', 'gecco', 'smd', 'msl']\n",
                "\n",
                "cmd = [\n",
                "    sys.executable, 'run_baseline_cache_ablation.py',\n",
                "    '--data_dir', 'data',\n",
                "    '--cache_dir', 'cache',\n",
                "    '--datasets'] + datasets_to_run + [\n",
                "    '--models'] + models_to_run + [\n",
                "    '--epochs', '5',\n",
                "    '--lr', '0.0015',\n",
                "    '--train_stride', '50',\n",
                "    '--eval_stride', '20',\n",
                "    '--num_workers', '2',\n",
                "    '--batch_size', '32',\n",
                "    '--checkpoint_dir', 'checkpoints/baseline_ablation',\n",
                "    '--output_json', 'results/ablations/baseline_cache_ablation.json',\n",
                "    '--output_report', 'results/ablations/BASELINE_CACHE_REPORT.md'\n",
                "]\n",
                "\n",
                "print(f'Executing command: {\" \".join(cmd)}')\n",
                "t0 = time.time()\n",
                "process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n",
                "for line in iter(process.stdout.readline, ''):\n",
                "    print(line, end='', flush=True)\n",
                "process.stdout.close()\n",
                "rc = process.wait()\n",
                "print(f'Process exited with code: {rc} (Elapsed: {(time.time() - t0)/60:.2f} mins)')\n",
                "if rc != 0:\n",
                "    raise RuntimeError(f'Baseline ablation failed with code {rc}')\n"
            ]
        },
        # Cell 5: Export Artifacts
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# --- Step 5: Export Checkpoints & Reports to /kaggle/working/ ---\n",
                "import os, shutil\n",
                "\n",
                "if os.path.exists('/kaggle/working'):\n",
                "    out_ckpt = '/kaggle/working/checkpoints/baseline_ablation'\n",
                "    out_res = '/kaggle/working/results/ablations'\n",
                "    os.makedirs(out_ckpt, exist_ok=True)\n",
                "    os.makedirs(out_res, exist_ok=True)\n",
                "\n",
                "    src_ck = 'checkpoints/baseline_ablation'\n",
                "    if os.path.exists(src_ck):\n",
                "        for f in os.listdir(src_ck):\n",
                "            shutil.copy2(os.path.join(src_ck, f), os.path.join(out_ckpt, f))\n",
                "\n",
                "    src_rep = 'results/ablations'\n",
                "    if os.path.exists(src_rep):\n",
                "        for f in os.listdir(src_rep):\n",
                "            shutil.copy2(os.path.join(src_rep, f), os.path.join(out_res, f))\n",
                "    print('Exported all checkpoints and reports to /kaggle/working/')\n"
            ]
        },
        # Cell 6: Render Report
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# --- Step 6: Render Markdown Summary Table ---\n",
                "import os\n",
                "from IPython.display import display, Markdown\n",
                "\n",
                "rep_path = 'results/ablations/BASELINE_CACHE_REPORT.md'\n",
                "if os.path.exists(rep_path):\n",
                "    with open(rep_path) as f:\n",
                "        display(Markdown(f.read()))\n",
                "else:\n",
                "    print(f'Report not found at {rep_path}')\n"
            ]
        }
    ]

    nb_json = {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    out_file = "kaggle_baseline_cache_ablation.ipynb"
    with open(out_file, "w") as f:
        json.dump(nb_json, f, indent=2)

    print(f"Successfully generated notebook: {out_file} ({os.path.getsize(out_file)/1024:.1f} KB)")


if __name__ == "__main__":
    main()
