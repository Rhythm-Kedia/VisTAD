"""Packages VisTAD codebase into a self-contained Kaggle notebook for SWaT unconstrained cache generation."""

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
        for root_file in ["train.py", "evaluate.py", "TODO.md"]:
            if os.path.exists(root_file):
                z.write(root_file, root_file)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def make_notebook(b64_bundle):
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# SWaT Unconstrained Cache Generation: Ultra-VETime Multi-Stage Teacher\n",
                    "### High-Density Curriculum Caching & Dedicated Head-to-Head Evaluation Cache\n",
                    "- **Backbone**: `facebook/dinov2-large` (1024-dim self-supervised visual patch features)\n",
                    "- **Multi-Regime Operational Centroids**: $K=5$ KMeans nominal operating clusters per stage\n",
                    "- **Multi-Stage Fusion**: Top-2 physical plant stages evaluated per window\n",
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
                    "print('Unpacking self-contained VisTAD codebase (src, vlm4ts, configs)...')\n",
                    f"b64_bundle = \"{b64_bundle}\"\n",
                    "raw_zip = base64.b64decode(b64_bundle)\n",
                    "with zipfile.ZipFile(io.BytesIO(raw_zip), 'r') as z:\n",
                    "    z.extractall(repo_dir)\n",
                    "print('VisTAD codebase unpacked successfully.')\n",
                    "print('Extracted files:', os.listdir(repo_dir))\n"
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
                    "# --- Step 3: SWaT Dataset Auto-Discovery ---\n",
                    "import os, glob, shutil\n",
                    "os.makedirs('data/swat', exist_ok=True)\n",
                    "\n",
                    "dataset_paths = glob.glob('/kaggle/input/**/SWaT_Dataset_*.csv', recursive=True)\n",
                    "print(f'Found {len(dataset_paths)} SWaT dataset candidates in /kaggle/input:')\n",
                    "for p in dataset_paths:\n",
                    "    fname = os.path.basename(p)\n",
                    "    target = os.path.join('data/swat', fname)\n",
                    "    if not os.path.exists(target):\n",
                    "        print(f'  Linking {p} -> {target}')\n",
                    "        try:\n",
                    "            os.symlink(p, target)\n",
                    "        except OSError:\n",
                    "            shutil.copy2(p, target)\n",
                    "\n",
                    "print('SWaT data directory contents:', os.listdir('data/swat'))\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# --- Step 4: Execute SWaT Unconstrained Cache Generation ---\n",
                    "import os, subprocess, time\n",
                    "os.environ['PYTHONPATH'] = '.'\n",
                    "\n",
                    "cmd = [\n",
                    "    'python', 'vlm4ts/generate_cache.py',\n",
                    "    '--data_dir', 'data/swat',\n",
                    "    '--output_train_cache', 'cache/swat_train_ultravetime_cache.pt',\n",
                    "    '--output_eval_cache', 'cache/swat_eval_head_to_head_cache.pt',\n",
                    "    '--window_size', '200',\n",
                    "    '--num_train_windows', '3500',\n",
                    "    '--num_eval_windows', '1200',\n",
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
                    "# --- Step 5: Verify Cache Statistics & Ground Truth Distributions ---\n",
                    "import torch, os\n",
                    "import numpy as np\n",
                    "\n",
                    "for cache_path, label in [\n",
                    "    ('cache/swat_train_ultravetime_cache.pt', 'TRAINING CURRICULUM CACHE'),\n",
                    "    ('cache/swat_eval_head_to_head_cache.pt', 'HEAD-TO-HEAD EVALUATION CACHE')\n",
                    "]:\n",
                    "    if not os.path.exists(cache_path):\n",
                    "        print(f'WARNING: {cache_path} not found!')\n",
                    "        continue\n",
                    "    size_mb = os.path.getsize(cache_path) / (1024 * 1024)\n",
                    "    c = torch.load(cache_path, weights_only=False)\n",
                    "    windows = c['windows']\n",
                    "    flags = [w['y_flag'] for w in windows.values()]\n",
                    "    scores = [w['teacher_score'] for w in windows.values()]\n",
                    "    anom_count = sum(1 for f in flags if f > 0.5)\n",
                    "    norm_count = len(flags) - anom_count\n",
                    "    print(f'\\n================== {label} ==================')\n",
                    "    print(f'File Size: {size_mb:.2f} MB | Total Windows: {len(windows)}')\n",
                    "    print(f'Attack Windows: {anom_count} ({anom_count/len(windows)*100:.1f}%) | Nominal: {norm_count}')\n",
                    "    print(f'Teacher Anomaly Score: min={min(scores):.4f}, median={np.median(scores):.4f}, max={max(scores):.4f}')\n",
                    "    first_w = next(iter(windows.values()))\n",
                    "    print(f'Telemetry Tensor x: {tuple(first_w[\"x\"].shape)}')\n",
                    "    print(f'Feature Target z_teacher: {tuple(first_w[\"z_teacher\"].shape)}')\n",
                    "    print(f'Temporal Mask teacher_dense_mask: {tuple(first_w[\"teacher_dense_mask\"].shape)}')\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# --- Step 6: Package Output Bundle for Retrieval ---\n",
                    "import os, zipfile, glob\n",
                    "\n",
                    "bundle_path = '/kaggle/working/swat_ultravetime_cache_bundle.zip'\n",
                    "print(f'Packaging cache files into {bundle_path}...')\n",
                    "with zipfile.ZipFile(bundle_path, 'w', zipfile.ZIP_DEFLATED) as z:\n",
                    "    for f in glob.glob('cache/*.pt'):\n",
                    "        print(f'  Adding {f} ({os.path.getsize(f)/(1024*1024):.2f} MB)')\n",
                    "        z.write(f, os.path.relpath(f, '.'))\n",
                    "print(f'Bundle created successfully ({os.path.getsize(bundle_path)/(1024*1024):.2f} MB).')\n",
                    "print('Output directory contents:', os.listdir('/kaggle/working'))\n"
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
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    return nb

def main():
    print("Building self-contained VisTAD codebase zip...")
    b64 = build_codebase_zip()
    print(f"Codebase zip built ({len(b64)} b64 chars).")
    nb = make_notebook(b64)
    out_file = "kaggle_cache_gen.ipynb"
    with open(out_file, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Saved Kaggle notebook to {out_file} ({os.path.getsize(out_file)/(1024):.1f} KB).")

if __name__ == "__main__":
    main()
