"""Packages VisTAD codebase into a self-contained Kaggle notebook for NASA MSL unconstrained cache generation."""

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
                    "# NASA MSL Unconstrained Cache Generation: Ultra-VETime Telemetry-Command Teacher\n",
                    "### High-Density Curriculum Caching & Dedicated Head-to-Head Evaluation Cache\n",
                    "- **Domain**: NASA Mars Science Laboratory Rover (MSL, 73,729 steps, 55 channels)\n",
                    "- **Backbone**: `facebook/dinov2-large` (1024-dim self-supervised visual patch features)\n",
                    "- **4-Panel Telemetry-Command Dashboard**: Channel 0 Analog Telemetry (+-3 sigma bounds) + Command Activation Context + Volatility + Deviation Energy\n",
                    "- **Multi-Regime Operational Centroids**: $K=5$ KMeans nominal operating clusters from 58,317 clean train steps\n",
                    "- **Rendering**: Fast direct Agg canvas buffer copy (zero PNG compression bottleneck)\n",
                    "- **Multi-GPU & Batching**: Shards candidate windows across `cuda:0` and `cuda:1` with batch size 16"
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
                    "# --- Step 3: MSL Dataset Auto-Discovery (Direct Input Mount) ---\n",
                    "import os, glob, pandas as pd\n",
                    "\n",
                    "c_tr = glob.glob('/kaggle/input/**/msl_train.csv', recursive=True)\n",
                    "c_te = glob.glob('/kaggle/input/**/msl_test.csv', recursive=True)\n",
                    "if c_tr and c_te:\n",
                    "    msl_dir = os.path.dirname(c_tr[0])\n",
                    "    print(f'Found MSL data directory: {msl_dir}')\n",
                    "    print(f'Train path: {c_tr[0]}')\n",
                    "    print(f'Test path:  {c_te[0]}')\n",
                    "else:\n",
                    "    raise FileNotFoundError('Could not locate msl_train.csv or msl_test.csv in /kaggle/input!')\n",
                    "\n",
                    "df_tr = pd.read_csv(c_tr[0])\n",
                    "df_te = pd.read_csv(c_te[0])\n",
                    "print(f'MSL Train shape: {df_tr.shape}, MSL Test shape: {df_te.shape}')\n",
                    "print(f'MSL Test anomalies: {df_te[\"attack\"].sum()} / {len(df_te)} ({100*df_te[\"attack\"].mean():.2f}%)')\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# --- Step 4: Run Ultra-VETime MSL Cache Generation ---\n",
                    "import sys\n",
                    "sys.path.insert(0, '/kaggle/working/VisTAD')\n",
                    "\n",
                    "from vlm4ts.generate_msl_cache import main\n",
                    "\n",
                    "sys.argv = [\n",
                    "    'generate_msl_cache.py',\n",
                    "    '--data_dir', msl_dir,\n",
                    "    '--output_train_cache', '/kaggle/working/msl_train_ultravetime_cache.pt',\n",
                    "    '--output_eval_cache', '/kaggle/working/msl_eval_head_to_head_cache.pt',\n",
                    "    '--num_train_windows', '3000',\n",
                    "    '--num_eval_windows', '1500',\n",
                    "    '--model_id', 'facebook/dinov2-large',\n",
                    "    '--n_regimes', '5'\n",
                    "]\n",
                    "\n",
                    "print('Starting MSL cache generation...')\n",
                    "main()\n",
                    "print('MSL cache generation completed successfully!')\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# --- Step 5: Clean Codebase and Verify Exported Caches ---\n",
                    "import os, torch, shutil\n",
                    "\n",
                    "# Clean extracted codebase so only final caches remain in /kaggle/working\n",
                    "if os.path.exists('/kaggle/working/VisTAD'):\n",
                    "    shutil.rmtree('/kaggle/working/VisTAD')\n",
                    "\n",
                    "train_c = '/kaggle/working/msl_train_ultravetime_cache.pt'\n",
                    "eval_c = '/kaggle/working/msl_eval_head_to_head_cache.pt'\n",
                    "\n",
                    "for p in [train_c, eval_c]:\n",
                    "    if os.path.exists(p):\n",
                    "        sz = os.path.getsize(p) / (1024**2)\n",
                    "        dat = torch.load(p, map_location='cpu', weights_only=False)\n",
                    "        n_wins = len(dat.get('windows', dat))\n",
                    "        print(f'File: {os.path.basename(p)} | Size: {sz:.2f} MB | Windows: {n_wins}')\n",
                    "    else:\n",
                    "        print(f'MISSING: {p}')\n",
                    "\n",
                    "print('Final Kaggle working directory contents:', os.listdir('/kaggle/working'))\n"
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
                "name": "python",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    return nb


def main():
    print("Packing codebase into base64 bundle...")
    b64_bundle = build_codebase_zip()
    print(f"Codebase bundle size: {len(b64_bundle) / 1024:.1f} KB")

    nb = make_notebook(b64_bundle)
    out_path = "kaggle_msl_cache_gen.ipynb"
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=2)
    print(f"Generated standalone notebook: {out_path} ({os.path.getsize(out_path) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
