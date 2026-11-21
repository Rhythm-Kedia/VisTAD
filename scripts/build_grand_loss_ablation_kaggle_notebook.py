"""
Builds self-contained Kaggle runner notebook for the Grand Model Loss Formulation Ablation Suite.

Ablates:
1. pure_bce:       L = BCE(p_sys, y)
2. focal:          L = FocalBCE(p_sys, y)
3. focal_dice:     L = FocalBCE(p_sys, y) + TemporalDice(p_sys, y)
4. unified_distill:L = (FocalBCE + TemporalDice) + 0.10 * CosineDist(z_student, z_teacher)
5. bce_feat:       L = BCE(p_sys, y) + 0.10 * CosineDist(z_student, z_teacher)

Evaluates on SWaT, SMD, GECCO, MSL using all channels (K = "all") and L_prefix = 250.
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
            'run_grand_loss_ablation.py',
            'train_universal_foundation.py'
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
                "# VisTAD: Grand Model Loss Formulation Ablation Study (Kaggle GPU)\n",
                "\n",
                "**Experimental Objectives:**\n",
                "1. **Clean Unified Objective:** Replaces multi-hyperparameter clutter with the unified 2-term objective: $\\mathcal{L} = \\mathcal{L}_{\\text{seg}} + \\alpha \\mathcal{L}_{\\text{feat}}$.\n",
                "2. **Ablation Matrix (5 Variants):**\n",
                "   - `pure_bce`: Standard point-wise Binary Cross-Entropy.\n",
                "   - `focal`: Focal BCE (gamma=2.0, pos_weight=2.0) downweighting easy nominals.\n",
                "   - `focal_dice`: Focal BCE + Continuous Temporal Soft-Dice (enforcing contiguous episodes).\n",
                "   - `unified_distill`: (Focal + Dice) + 0.10 $\\times$ Feature Distillation (the master student objective).\n",
                "   - `bce_feat`: Standard BCE + 0.10 $\\times$ Feature Distillation.\n",
                "3. **All Channels ($K=\\text{all}$):** Evaluates dense multi-sensor consensus across all channels dynamically.\n",
                "4. **Receptive Field Warm-Up ($L_{\\text{prefix}} = 250$):** Zero cold-start edge starvation.\n",
                "5. **Artifact Persistence:** Exports all checkpoints and structured `<DATASET>_<METRIC>` tables to `/kaggle/working/`.\n"
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
                "print('Unpacking self-contained Grand Loss ablation codebase...')\n",
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
                "\n",
                "for ds in ['swat', 'smd/test', 'smd/test_label', 'gecco', 'msl']:\n",
                "    os.makedirs(os.path.join('data', ds), exist_ok=True)\n",
                "os.makedirs('cache', exist_ok=True)\n",
                "os.makedirs('checkpoints', exist_ok=True)\n",
                "os.makedirs('results/ablations', exist_ok=True)\n",
                "\n",
                "# 1. SWaT\n",
                "swat_files = glob.glob('/kaggle/input/**/SWaT_Dataset_*.csv', recursive=True) + glob.glob('/kaggle/input/**/swat/*.csv', recursive=True)\n",
                "for p in set(swat_files):\n",
                "    fname = os.path.basename(p)\n",
                "    target = os.path.join('data/swat', fname)\n",
                "    if not os.path.exists(target):\n",
                "        try: os.symlink(p, target)\n",
                "        except OSError: shutil.copy2(p, target)\n",
                "print(f'Linked {len(os.listdir(\"data/swat\"))} SWaT files.')\n",
                "\n",
                "# 2. SMD\n",
                "smd_test_files = glob.glob('/kaggle/input/**/smd/test/*.txt', recursive=True)\n",
                "for p in smd_test_files:\n",
                "    target = os.path.join('data/smd/test', os.path.basename(p))\n",
                "    if not os.path.exists(target):\n",
                "        try: os.symlink(p, target)\n",
                "        except OSError: shutil.copy2(p, target)\n",
                "\n",
                "smd_label_files = glob.glob('/kaggle/input/**/smd/test_label/*.txt', recursive=True)\n",
                "for p in smd_label_files:\n",
                "    target = os.path.join('data/smd/test_label', os.path.basename(p))\n",
                "    if not os.path.exists(target):\n",
                "        try: os.symlink(p, target)\n",
                "        except OSError: shutil.copy2(p, target)\n",
                "print(f'Linked {len(os.listdir(\"data/smd/test\"))} SMD test files.')\n",
                "\n",
                "# 3. GECCO\n",
                "gecco_files = glob.glob('/kaggle/input/**/gecco_water.csv', recursive=True) + glob.glob('/kaggle/input/**/gecco/*.csv', recursive=True)\n",
                "for p in set(gecco_files):\n",
                "    target = os.path.join('data/gecco', os.path.basename(p))\n",
                "    if not os.path.exists(target):\n",
                "        try: os.symlink(p, target)\n",
                "        except OSError: shutil.copy2(p, target)\n",
                "print(f'Linked {len(os.listdir(\"data/gecco\"))} GECCO files.')\n",
                "\n",
                "# 4. MSL\n",
                "msl_files = glob.glob('/kaggle/input/**/msl_*.csv', recursive=True) + glob.glob('/kaggle/input/**/msl/*.csv', recursive=True)\n",
                "for p in set(msl_files):\n",
                "    target = os.path.join('data/msl', os.path.basename(p))\n",
                "    if not os.path.exists(target):\n",
                "        try: os.symlink(p, target)\n",
                "        except OSError: shutil.copy2(p, target)\n",
                "print(f'Linked {len(os.listdir(\"data/msl\"))} MSL files.')\n",
                "\n",
                "# 5. Offline VLM Teacher Caches\n",
                "for root, dirs, files in os.walk('/kaggle/input'):\n",
                "    for f in files:\n",
                "        if f.endswith('.pt') and 'cache' in f:\n",
                "            src_p = os.path.join(root, f)\n",
                "            dst_p = os.path.join('cache', f)\n",
                "            if not os.path.exists(dst_p):\n",
                "                try: os.symlink(src_p, dst_p)\n",
                "                except Exception: shutil.copy2(src_p, dst_p)\n",
                "print(f'Available caches: {sorted(os.listdir(\"cache\"))}')\n"
            ]
        },
        # Cell 4: Execute Grand Model Loss Ablation
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# --- Step 4: Execute Grand Model Loss Ablation Suite ---\n",
                "import os, sys, subprocess, time\n",
                "os.environ['PYTHONPATH'] = '.'\n",
                "\n",
                "cmd = [\n",
                "    sys.executable, 'run_grand_loss_ablation.py',\n",
                "    '--data_dir', 'data',\n",
                "    '--cache_dir', 'cache',\n",
                "    '--variants', 'pure_bce', 'focal', 'focal_dice', 'unified_distill', 'bce_feat',\n",
                "    '--epochs', '5',\n",
                "    '--lr', '0.0015',\n",
                "    '--train_stride', '50',\n",
                "    '--eval_stride', '20',\n",
                "    '--num_workers', '2',\n",
                "    '--batch_size', '32',\n",
                "    '--prefix_len', '250',\n",
                "    '--window_size', '200',\n",
                "    '--checkpoint_dir', 'checkpoints',\n",
                "    '--output_json', 'results/ablations/grand_loss_ablation_results.json',\n",
                "    '--output_report', 'results/ablations/GRAND_LOSS_ABLATION_REPORT.md'\n",
                "]\n",
                "\n",
                "print(f'Executing command: {\" \".join(cmd)}')\n",
                "t0 = time.time()\n",
                "process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n",
                "for line in iter(process.stdout.readline, ''):\n",
                "    print(line, end='', flush=True)\n",
                "process.stdout.close()\n",
                "rc = process.wait()\n",
                "print(f'Grand Loss Ablation process exited with return code: {rc} (Elapsed: {(time.time() - t0)/60:.2f} mins)')\n",
                "if rc != 0:\n",
                "    raise RuntimeError(f'Grand Loss Ablation failed with return code {rc}')\n"
            ]
        },
        # Cell 5: Export Checkpoints & Reports to /kaggle/working/
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# --- Step 5: Export Checkpoints & Reports to /kaggle/working/ ---\n",
                "import os, shutil\n",
                "\n",
                "out_ckpt_dir = '/kaggle/working/checkpoints'\n",
                "out_res_dir = '/kaggle/working/results'\n",
                "os.makedirs(out_ckpt_dir, exist_ok=True)\n",
                "os.makedirs(out_res_dir, exist_ok=True)\n",
                "\n",
                "# Copy checkpoints\n",
                "for f in os.listdir('checkpoints'):\n",
                "    src_p = os.path.join('checkpoints', f)\n",
                "    dst_p = os.path.join(out_ckpt_dir, f)\n",
                "    shutil.copy2(src_p, dst_p)\n",
                "    print(f'Exported checkpoint: {dst_p} ({os.path.getsize(dst_p)/1e6:.2f} MB)')\n",
                "\n",
                "# Copy results\n",
                "for root, dirs, files in os.walk('results'):\n",
                "    for f in files:\n",
                "        src_p = os.path.join(root, f)\n",
                "        dst_p = os.path.join(out_res_dir, f)\n",
                "        shutil.copy2(src_p, dst_p)\n",
                "        print(f'Exported result artifact: {dst_p}')\n",
                "\n",
                "report_path = 'results/ablations/GRAND_LOSS_ABLATION_REPORT.md'\n",
                "if os.path.exists(report_path):\n",
                "    print('\\n' + '=' * 80)\n",
                "    print('FINAL GRAND LOSS ABLATION REPORT:')\n",
                "    print('=' * 80)\n",
                "    with open(report_path, 'r') as rf:\n",
                "        print(rf.read())\n"
            ]
        }
    ]

    nb_payload = {
        "cells": nb_cells,
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
                "nbformat": 4,
                "nbformat_minor": 2,
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    out_file = "kaggle_grand_loss_ablation.ipynb"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(nb_payload, f, indent=2)
    print(f"Successfully generated Kaggle notebook: {out_file} ({os.path.getsize(out_file)/1024:.1f} KB)")


if __name__ == '__main__':
    main()
