"""
Builds self-contained Kaggle runner notebook for the 2x2 Student Architecture x Teacher Cache Cross-Ablation:

1. Old Student + Old Cache (Coupled Domain TCN + 2048-d Early VLM)
2. Old Student + New Cache (Coupled Domain TCN + 1024-d Ultra-VETime / DINOv2)
3. New Student + Old Cache (UniversalFoundationTCN_Pro + 2048-d Early VLM)
4. New Student + New Cache (UniversalFoundationTCN_Pro + 1024-d Ultra-VETime / DINOv2)
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
            'run_cross_student_teacher_ablation.py',
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
                "# VisTAD: 2×2 Student Architecture × Teacher Manifold Quality Cross-Ablation\n",
                "\n",
                "**Experimental Hypothesis:**\n",
                "Isolates the performance contributions of:\n",
                "1. **Student Architecture Inductive Biases**: Domain-Coupled Subsystem TCNs vs. Scaled Permutation-Equivariant Universal Foundation (`UniversalFoundationTCN_Pro`).\n",
                "2. **Teacher Manifold Supervision**: 2048-d Early Multimodal VLM features vs. 1024-d Ultra-VETime / DINOv2-Large foundation embeddings.\n",
                "\n",
                "**Factorial Matrix:**\n",
                "- **Cell (1,1)**: Old Student + Old Cache (2048-d)\n",
                "- **Cell (1,2)**: Old Student + New Cache (1024-d)\n",
                "- **Cell (2,1)**: New Student + Old Cache (2048-d)\n",
                "- **Cell (2,2)**: New Student + New Cache (1024-d)\n"
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
                "repo_dir = '/kaggle/working/CrossAblation'\n",
                "os.makedirs(repo_dir, exist_ok=True)\n",
                "os.chdir(repo_dir)\n",
                "\n",
                "print('Unpacking self-contained Cross-Ablation codebase...')\n",
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
                "# --- Step 3: Multi-Dataset & Dual-Cache Auto-Discovery ---\n",
                "import os, glob, shutil\n",
                "\n",
                "for ds in ['swat', 'smd/test', 'smd/test_label', 'gecco', 'msl']:\n",
                "    os.makedirs(os.path.join('data', ds), exist_ok=True)\n",
                "os.makedirs('cache', exist_ok=True)\n",
                "os.makedirs('checkpoints/cross_ablation', exist_ok=True)\n",
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
                "# 5. Teacher Caches\n",
                "cache_files = glob.glob('/kaggle/input/**/*.pt', recursive=True)\n",
                "for p in set(cache_files):\n",
                "    if 'cache' in p.lower() or 'ultravetime' in p.lower():\n",
                "        target = os.path.join('cache', os.path.basename(p))\n",
                "        if not os.path.exists(target):\n",
                "            try: os.symlink(p, target)\n",
                "            except OSError: shutil.copy2(p, target)\n",
                "print(f'Discovered and linked {len(os.listdir(\"cache\"))} caches in cache/: {sorted(os.listdir(\"cache\"))}')\n"
            ]
        },
        # Cell 4: Execute Cross-Ablation Suite
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# --- Step 4: Execute 6-Model Cross-Ablation Suite ---\n",
                "import os, sys, subprocess, time\n",
                "os.environ['PYTHONPATH'] = '.'\n",
                "\n",
                "cmd = [\n",
                "    sys.executable, 'run_cross_student_teacher_ablation.py',\n",
                "    '--data_dir', 'data',\n",
                "    '--cache_dir', 'cache',\n",
                "    '--datasets', 'swat', 'gecco', 'smd', 'msl',\n",
                "    '--epochs', '10',\n",
                "    '--batch_size', '32',\n",
                "    '--checkpoint_dir', 'checkpoints/cross_ablation',\n",
                "    '--output_json', 'results/ablations/cross_student_teacher_ablation.json',\n",
                "    '--output_report', 'results/ablations/CROSS_STUDENT_TEACHER_REPORT.md'\n",
                "]\n",
                "\n",
                "print(f'Executing command: {\" \".join(cmd)}')\n",
                "t0 = time.time()\n",
                "process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n",
                "for line in iter(process.stdout.readline, ''):\n",
                "    print(line, end='', flush=True)\n",
                "process.stdout.close()\n",
                "rc = process.wait()\n",
                "print(f'Cross-Ablation process exited with return code: {rc} (Elapsed: {(time.time() - t0)/60:.2f} mins)')\n",
                "if rc != 0:\n",
                "    raise RuntimeError(f'Cross-Ablation failed with return code {rc}')\n"
            ]
        },
        # Cell 5: Export Checkpoints & Reports
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# --- Step 5: Export Checkpoints & Reports to /kaggle/working/ ---\n",
                "import os, shutil\n",
                "\n",
                "out_ckpt = '/kaggle/working/checkpoints/cross_ablation'\n",
                "out_res = '/kaggle/working/results/ablations'\n",
                "os.makedirs(out_ckpt, exist_ok=True)\n",
                "os.makedirs(out_res, exist_ok=True)\n",
                "\n",
                "# Copy checkpoints\n",
                "src_ck = 'checkpoints/cross_ablation'\n",
                "if os.path.exists(src_ck):\n",
                "    for f in os.listdir(src_ck):\n",
                "        p = os.path.join(src_ck, f)\n",
                "        shutil.copy2(p, os.path.join(out_ckpt, f))\n",
                "        print(f'Exported: {f} ({os.path.getsize(p)/1e6:.2f} MB)')\n",
                "\n",
                "# Copy results\n",
                "src_re = 'results/ablations'\n",
                "if os.path.exists(src_re):\n",
                "    for f in os.listdir(src_re):\n",
                "        p = os.path.join(src_re, f)\n",
                "        shutil.copy2(p, os.path.join(out_res, f))\n",
                "        print(f'Exported result artifact: {f}')\n",
                "\n",
                "report_p = 'results/ablations/CROSS_STUDENT_TEACHER_REPORT.md'\n",
                "if os.path.exists(report_p):\n",
                "    print('\\n' + '=' * 80)\n",
                "    print('FINAL 2x2 FACTORIAL CROSS-ABLATION REPORT:')\n",
                "    print('=' * 80)\n",
                "    with open(report_p, 'r') as rf:\n",
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

    out_file = "kaggle_cross_compatibility.ipynb"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(nb_payload, f, indent=2)
    print(f"Successfully generated Kaggle notebook: {out_file} ({os.path.getsize(out_file)/1024:.1f} KB)")


if __name__ == '__main__':
    main()
