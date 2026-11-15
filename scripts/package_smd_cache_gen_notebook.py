"""Packages VisTAD codebase into a self-contained Kaggle notebook for Server Machine Dataset (SMD) cache generation & distillation."""

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
        for root_file in ["train_smd_pooled_student.py", "eval_smd_continuous_streaming.py", "TODO.md"]:
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
                    "# Server Machine Dataset (SMD) Ultra-VETime Teacher Cache Generation\n",
                    "### 38-Channel Multi-Subsystem Teacher Caching & Distillation Across All 28 Servers\n",
                    "- **Domain**: 28 Large-Scale Compute Cluster Servers (708,420 timesteps, 38 metrics, 327 failure events)\n",
                    "- **4 Subsystem Visual Dashboard**: CPU Dynamics (0..7), Memory & Paging (8..14), Network Traffic (15..26), Disk I/O (27..37)\n",
                    "- **Teacher Backbone**: `facebook/dinov2-large` (1024-dim self-supervised representations)\n",
                    "- **Multi-Regime Operational Centroids**: $K=5$ nominal operating clusters\n",
                    "- **Zero Disk I/O Rendering**: Direct Agg memory buffer extraction (<10 ms/window)\n",
                    "- **Multi-GPU Execution**: Parallel batching across provisioned GPUs"
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
                    "# --- Step 2: GPU Accelerator Diagnostics ---\n",
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
                    "# --- Step 3: SMD Dataset Auto-Discovery & Path Linking ---\n",
                    "import os, glob, shutil\n",
                    "os.makedirs('data/smd/test', exist_ok=True)\n",
                    "os.makedirs('data/smd/test_label', exist_ok=True)\n",
                    "\n",
                    "smd_test_candidates = glob.glob('/kaggle/input/**/smd/test/*.txt', recursive=True)\n",
                    "print(f'Found {len(smd_test_candidates)} SMD test files in /kaggle/input')\n",
                    "for p in smd_test_candidates:\n",
                    "    fname = os.path.basename(p)\n",
                    "    target = os.path.join('data/smd/test', fname)\n",
                    "    if not os.path.exists(target):\n",
                    "        try:\n",
                    "            os.symlink(p, target)\n",
                    "        except OSError:\n",
                    "            shutil.copy2(p, target)\n",
                    "\n",
                    "smd_label_candidates = glob.glob('/kaggle/input/**/smd/test_label/*.txt', recursive=True)\n",
                    "print(f'Found {len(smd_label_candidates)} SMD test label files in /kaggle/input')\n",
                    "for p in smd_label_candidates:\n",
                    "    fname = os.path.basename(p)\n",
                    "    target = os.path.join('data/smd/test_label', fname)\n",
                    "    if not os.path.exists(target):\n",
                    "        try:\n",
                    "            os.symlink(p, target)\n",
                    "        except OSError:\n",
                    "            shutil.copy2(p, target)\n",
                    "\n",
                    "print('Local data/smd/test count:', len(os.listdir('data/smd/test')))\n",
                    "print('Local data/smd/test_label count:', len(os.listdir('data/smd/test_label')))\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# --- Step 4: Execute Ultra-VETime Teacher Cache Generation ---\n",
                    "import os, subprocess, time\n",
                    "os.environ['PYTHONPATH'] = '.'\n",
                    "\n",
                    "cmd = [\n",
                    "    'python', 'vlm4ts/generate_smd_cache.py',\n",
                    "    '--data_dir', 'data/smd',\n",
                    "    '--output_train_cache', 'cache/smd_train_ultravetime_cache.pt',\n",
                    "    '--output_eval_cache', 'cache/smd_eval_head_to_head_cache.pt',\n",
                    "    '--window_size', '200',\n",
                    "    '--num_train_windows', '3000',\n",
                    "    '--num_eval_windows', '1500',\n",
                    "    '--model_id', 'facebook/dinov2-large',\n",
                    "    '--n_regimes', '5'\n",
                    "]\n",
                    "print('Executing SMD cache generation command:', ' '.join(cmd))\n",
                    "t0 = time.time()\n",
                    "proc = subprocess.run(cmd, check=True)\n",
                    "dur = time.time() - t0\n",
                    "print(f'SMD Teacher Caching completed successfully in {dur:.1f}s ({dur/60:.2f} min).')\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# --- Step 5: Verify Teacher Cache Statistics ---\n",
                    "import torch, os\n",
                    "import numpy as np\n",
                    "\n",
                    "for c_path, name in [\n",
                    "    ('cache/smd_train_ultravetime_cache.pt', 'SMD TRAINING CURRICULUM CACHE'),\n",
                    "    ('cache/smd_eval_head_to_head_cache.pt', 'SMD HEAD-TO-HEAD EVALUATION CACHE')\n",
                    "]:\n",
                    "    if not os.path.exists(c_path):\n",
                    "        print(f'WARNING: {c_path} not found!')\n",
                    "        continue\n",
                    "    sz_mb = os.path.getsize(c_path) / (1024 * 1024)\n",
                    "    c = torch.load(c_path, weights_only=False)\n",
                    "    scores = [v['teacher_score'] for v in c.values()]\n",
                    "    flags = [v['y_flag'] for v in c.values()]\n",
                    "    anoms = sum(flags)\n",
                    "    print(f'\\n================== {name} ==================')\n",
                    "    print(f'File Size: {sz_mb:.2f} MB | Total Records: {len(c)}')\n",
                    "    print(f'Anomaly Windows: {anoms} ({anoms/len(c)*100:.1f}%) | Nominal: {len(c)-anoms}')\n",
                    "    print(f'Teacher Anomaly Score: min={min(scores):.4f}, median={np.median(scores):.4f}, max={max(scores):.4f}')\n",
                    "    sample = next(iter(c.values()))\n",
                    "    print(f'Tensor x: {tuple(sample[\"x\"].shape)}')\n",
                    "    print(f'Feature Target z_teacher: {tuple(sample[\"z_teacher\"].shape)}')\n",
                    "    print(f'Dense Mask teacher_dense_mask: {tuple(sample[\"teacher_dense_mask\"].shape)}')\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# --- Step 6: Package Cache Files for Local Retrieval ---\n",
                    "import os, zipfile, glob\n",
                    "\n",
                    "out_zip = '/kaggle/working/smd_ultravetime_cache_bundle.zip'\n",
                    "print(f'Packaging cache files into {out_zip}...')\n",
                    "with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as z:\n",
                    "    for f in glob.glob('cache/smd_*.pt'):\n",
                    "        z.write(f, os.path.basename(f))\n",
                    "        print(f'  Added {f} ({os.path.getsize(f)/(1024*1024):.2f} MB)')\n",
                    "\n",
                    "print(f'Cache package ready: {out_zip} ({os.path.getsize(out_zip)/(1024*1024):.2f} MB)')\n",
                    "# Also copy files to root working directory for direct download\n",
                    "for f in glob.glob('cache/smd_*.pt'):\n",
                    "    shutil.copy2(f, f'/kaggle/working/{os.path.basename(f)}')\n"
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
    print("Building codebase base64 bundle...")
    b64_bundle = build_codebase_zip()
    print(f"Codebase bundle size: {len(b64_bundle)/(1024):.1f} KB")

    print("Generating kaggle_smd_cache_gen.ipynb...")
    nb = make_notebook(b64_bundle)
    out_path = "kaggle_smd_cache_gen.ipynb"
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=2)
    print(f"Self-contained Kaggle notebook saved to: {out_path} ({os.path.getsize(out_path)/(1024):.1f} KB)")


if __name__ == "__main__":
    main()
