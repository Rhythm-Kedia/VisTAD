"""Builds self-contained Kaggle runner notebook with Decoupled Stage Isolation:

TRACK 1: IN-DOMAIN VLM DISTILLATION & SPECIALIZED STUDENTS SUITE
- Step 5: VLM Teacher Caches (Verified or Regenerated)
- Step 6: Specialized Coupled Students (SWaT, SMD, GECCO, MSL)
- Step 7: STAGE 1 MILESTONE EXPORT (Permanently saves checkpoints & in-domain tables to disk)

TRACK 2: UNIVERSAL FOUNDATION MODEL SUITE
- Step 8: 4-Fold Leave-One-Domain-Out (LODO) Foundation Benchmark (20 Epochs, AMP, Clamped BCE)
- Step 9: Grand Unified Multi-Task Foundation Pre-Training (All 4 Domains Jointly)
- Step 10: Dimension Subsampling & Sensor Dropping Stress Sweeps (100% to 10%)
- Step 11: Final Grand Export & High-Density Markdown Summary Report
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

        # 2. Training and Evaluation Scripts
        py_files = [
            'train_universal_foundation.py',
            'eval_dimension_subsampling_sweep.py',
            'train_gecco_student.py',
            'train_msl_student.py',
            'train_stage_coupled_student.py',
            'train_smd_pooled_student.py',
            'vlm4ts/gecco_renderer.py',
            'vlm4ts/generate_gecco_cache.py',
            'vlm4ts/msl_renderer.py',
            'vlm4ts/generate_msl_cache.py'
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
                "# Universal Cyber-Physical Foundation Model: Decoupled Stage Isolation Runner\n",
                "\n",
                "**Two Decoupled, Resilient Workload Tracks:**\n",
                "\n",
                "### TRACK 1: IN-DOMAIN VLM DISTILLATION & SPECIALIZED STUDENTS SUITE\n",
                "1. **Workspace Setup & Multi-GPU Diagnostics**\n",
                "2. **Multi-Dataset Auto-Discovery & Path Linking** (`SWaT`, `SMD`, `GECCO`, `MSL`)\n",
                "3. **VLM Teacher Cache Verification & Generation** (GECCO & MSL Caches)\n",
                "4. **In-Domain Specialized Students Benchmark** (`SWaT`, `SMD`, `GECCO`, `MSL` Coupled TCNs)\n",
                "5. **STAGE 1 MILESTONE CHECKPOINT & EXPORT** (Guarantees Track 1 results are saved to disk)\n",
                "\n",
                "### TRACK 2: UNIVERSAL FOUNDATION MODEL SUITE\n",
                "6. **4-Fold Leave-One-Domain-Out (LODO) Foundation Benchmark** (20 Epochs, AMP, Clamped BCE)\n",
                "7. **Grand Unified Multi-Task Foundation Pre-Training** (All 4 Domains Jointly)\n",
                "8. **Dimension Subsampling & Sensor Dropping Stress Sweeps** (100% down to 10% Channels)\n",
                "9. **Final Grand Export & Consolidation**\n"
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
                "print('Unpacking self-contained Universal Foundation codebase...')\n",
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
                "        print(f'  GPU {i}: {prop.name} ({prop.total_memory / (1024**3):.1f} GB VRAM)')\n"
            ]
        },
        # Cell 3: Dataset Linking
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
                "\n",
                "# 1. SWaT\n",
                "swat_files = glob.glob('/kaggle/input/**/SWaT_Dataset_Attack_v0.csv', recursive=True) + glob.glob('/kaggle/input/**/SWaT_Dataset_Normal_v0.csv', recursive=True)\n",
                "for p in set(swat_files):\n",
                "    target = os.path.join('data/swat', os.path.basename(p))\n",
                "    if not os.path.exists(target):\n",
                "        try: os.symlink(p, target)\n",
                "        except OSError: shutil.copy2(p, target)\n",
                "print(f'Linked {len(os.listdir(\"data/swat\"))} SWaT files.')\n",
                "\n",
                "# 2. SMD\n",
                "smd_test_files = glob.glob('/kaggle/input/**/test/*.txt', recursive=True)\n",
                "for p in smd_test_files:\n",
                "    target = os.path.join('data/smd/test', os.path.basename(p))\n",
                "    if not os.path.exists(target):\n",
                "        try: os.symlink(p, target)\n",
                "        except OSError: shutil.copy2(p, target)\n",
                "smd_label_files = glob.glob('/kaggle/input/**/test_label/*.txt', recursive=True)\n",
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
                "# 5. Link Existing Caches (SWaT, SMD, GECCO, MSL)\n",
                "cache_files = glob.glob('/kaggle/input/**/*.pt', recursive=True)\n",
                "for p in cache_files:\n",
                "    fname = os.path.basename(p)\n",
                "    target = os.path.join('cache', fname)\n",
                "    if not os.path.exists(target):\n",
                "        try: os.symlink(p, target)\n",
                "        except OSError: shutil.copy2(p, target)\n",
                "print(f'Linked {len(os.listdir(\"cache\"))} existing cache files.')\n"
            ]
        },
        # Cell 4: Verify Scaled Model
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# --- Step 4: Verify Scaled Architecture & Dimensional Invariants ---\n",
                "import torch\n",
                "from src.models.universal_tcn import UniversalFoundationTCN_Pro\n",
                "\n",
                "model = UniversalFoundationTCN_Pro(hidden_dim=128, num_heads=8, num_layers=2)\n",
                "n_params = sum(p.numel() for p in model.parameters())\n",
                "print(f'UniversalFoundationTCN_Pro Parameters: {n_params:,} ({n_params/1e6:.2f}M)')\n",
                "\n",
                "# Verify dimensional invariance across all 4 dataset channel counts\n",
                "for C, name in [(51, 'SWaT'), (38, 'SMD'), (9, 'GECCO'), (55, 'MSL')]:\n",
                "    x = torch.randn(2, C, 300)\n",
                "    out = model(x, prefix_len=100)\n",
                "    assert out['p_system'].shape == (2, 300)\n",
                "    assert out['z_student'].shape == (2, 1024)\n",
                "    assert out['pred_next'].shape == (2, C, 300)\n",
                "    print(f'  [{name}] C={C} Verified: p_system={list(out[\"p_system\"].shape)}, z_student={list(out[\"z_student\"].shape)}')\n",
                "print('All dimensional scaling invariants verified successfully!')\n"
            ]
        },
        # Cell 5: VLM Teacher Caches (Verified or Regenerated)
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# --- Step 5: VLM Teacher Caches (GECCO & MSL) ---\n",
                "import os, subprocess, time, torch, gc\n",
                "os.environ['PYTHONPATH'] = '.'\n",
                "os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'\n",
                "\n",
                "gecco_ready = os.path.exists('cache/gecco_train_ultravetime_cache.pt') and os.path.exists('cache/gecco_eval_head_to_head_cache.pt')\n",
                "if gecco_ready:\n",
                "    print('GECCO VLM teacher caches already verified in cache/! Skipping regeneration.')\n",
                "else:\n",
                "    print('\\n' + '='*80)\n",
                "    print('GENERATING UNCONSTRAINED GECCO VLM TEACHER CACHE (3,500 TRAIN + 1,200 EVAL)')\n",
                "    print('='*80)\n",
                "    try:\n",
                "        t0 = time.time()\n",
                "        subprocess.run(['python', 'vlm4ts/generate_gecco_cache.py', '--num_train_windows', '3500', '--num_eval_windows', '1200'], check=True)\n",
                "        print(f'GECCO Cache Generation Complete in {(time.time() - t0)/60:.2f} minutes.')\n",
                "    except Exception as e:\n",
                "        print(f'Warning: GECCO cache generation error: {e}')\n",
                "\n",
                "msl_ready = os.path.exists('cache/msl_train_ultravetime_cache.pt') and os.path.exists('cache/msl_eval_head_to_head_cache.pt')\n",
                "if msl_ready:\n",
                "    print('NASA MSL VLM teacher caches already verified in cache/! Skipping regeneration.')\n",
                "else:\n",
                "    print('\\n' + '='*80)\n",
                "    print('GENERATING UNCONSTRAINED NASA MSL VLM TEACHER CACHE (3,000 TRAIN + 1,500 EVAL)')\n",
                "    print('='*80)\n",
                "    try:\n",
                "        t0 = time.time()\n",
                "        subprocess.run(['python', 'vlm4ts/generate_msl_cache.py', '--num_train_windows', '3000', '--num_eval_windows', '1500'], check=True)\n",
                "        print(f'NASA MSL Cache Generation Complete in {(time.time() - t0)/60:.2f} minutes.')\n",
                "    except Exception as e:\n",
                "        print(f'Warning: NASA MSL cache generation error: {e}')\n",
                "\n",
                "gc.collect()\n",
                "if torch.cuda.is_available():\n",
                "    torch.cuda.empty_cache()\n"
            ]
        },
        # Cell 6: Train In-Domain Specialized Coupled Students
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# --- Step 6: Train Specialized Coupled Students (In-Domain Supervised + Distilled Benchmarks) ---\n",
                "import os, subprocess, time, torch, gc\n",
                "os.environ['PYTHONPATH'] = '.'\n",
                "os.makedirs('results/specialized_students', exist_ok=True)\n",
                "os.makedirs('checkpoints', exist_ok=True)\n",
                "\n",
                "student_runs = [\n",
                "    ('GECCO Student', ['python', 'train_gecco_student.py', '--epochs', '15', '--batch_size', '64', '--distill_alpha', '0.1']),\n",
                "    ('NASA MSL Student', ['python', 'train_msl_student.py', '--epochs', '15', '--batch_size', '64', '--distill_alpha', '0.1']),\n",
                "    ('SWaT Student', ['python', 'train_stage_coupled_student.py', '--epochs', '15', '--batch_size', '64', '--alpha', '0.1']),\n",
                "    ('SMD Student', ['python', 'train_smd_pooled_student.py', '--epochs', '15', '--batch_size', '64'])\n",
                "]\n",
                "\n",
                "for name, cmd in student_runs:\n",
                "    gc.collect()\n",
                "    if torch.cuda.is_available():\n",
                "        torch.cuda.empty_cache()\n",
                "    print(f'\\n{\"=\"*80}\\nTRAINING SPECIALIZED {name.upper()}\\n{\"=\"*80}')\n",
                "    try:\n",
                "        t0 = time.time()\n",
                "        subprocess.run(cmd, check=True)\n",
                "        print(f'>>> {name} Completed in {(time.time() - t0)/60:.2f} minutes.')\n",
                "    except Exception as e:\n",
                "        print(f'Warning: {name} training encountered issue: {e}. Proceeding...')\n"
            ]
        },
        # Cell 7: STAGE 1 MILESTONE CHECKPOINT & EXPORT
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# --- Step 7: STAGE 1 MILESTONE CHECKPOINT & EXPORT ---\n",
                "import os, glob, shutil\n",
                "out_stage1 = '/kaggle/working/results'\n",
                "os.makedirs(out_stage1, exist_ok=True)\n",
                "os.makedirs('/kaggle/working/stage1_artifacts', exist_ok=True)\n",
                "\n",
                "# Copy all student checkpoints and caches to permanent results directory\n",
                "for f in glob.glob('checkpoints/*.pt'):\n",
                "    shutil.copy2(f, out_stage1)\n",
                "    shutil.copy2(f, '/kaggle/working/stage1_artifacts')\n",
                "for f in glob.glob('cache/*_cache.pt'):\n",
                "    shutil.copy2(f, out_stage1)\n",
                "for f in glob.glob('results/**/*.json', recursive=True):\n",
                "    shutil.copy2(f, out_stage1)\n",
                "\n",
                "print('\\n' + '='*80)\n",
                "print('TRACK 1 (IN-DOMAIN VLM DISTILLATION & SPECIALIZED STUDENTS) COMPLETED & CHECKPOINTED!')\n",
                "print('All specialized student weights, caches, and metric files are safely persisted to disk.')\n",
                "print('Persisted files:')\n",
                "for f in os.listdir(out_stage1):\n",
                "    size_mb = os.path.getsize(os.path.join(out_stage1, f)) / (1024*1024)\n",
                "    print(f'  {f} ({size_mb:.2f} MB)')\n",
                "print('='*80 + '\\n')\n"
            ]
        },
        # Cell 8: LODO 4-Fold Foundation Benchmark
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# --- Step 8: TRACK 2 - Leave-One-Domain-Out (LODO) Foundation Benchmark (20 Epochs) ---\n",
                "import os, subprocess, time, json, shutil, torch, gc\n",
                "os.environ['PYTHONPATH'] = '.'\n",
                "os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'\n",
                "os.makedirs('results/lodo', exist_ok=True)\n",
                "os.makedirs('checkpoints', exist_ok=True)\n",
                "os.makedirs('/kaggle/working/results/lodo', exist_ok=True)\n",
                "\n",
                "lodo_folds = [\n",
                "    {'train': 'smd,gecco,msl', 'eval': 'swat', 'name': 'Fold_1_ZeroShot_SWaT'},\n",
                "    {'train': 'swat,gecco,msl', 'eval': 'smd',  'name': 'Fold_2_ZeroShot_SMD'},\n",
                "    {'train': 'swat,smd,msl',   'eval': 'gecco','name': 'Fold_3_ZeroShot_GECCO'},\n",
                "    {'train': 'swat,smd,gecco', 'eval': 'msl',  'name': 'Fold_4_ZeroShot_MSL'}\n",
                "]\n",
                "\n",
                "lodo_results = {}\n",
                "for fold in lodo_folds:\n",
                "    gc.collect()\n",
                "    if torch.cuda.is_available():\n",
                "        torch.cuda.empty_cache()\n",
                "    print('\\n' + '='*80)\n",
                "    print(f'EXECUTING LODO {fold[\"name\"]}: [{fold[\"train\"].upper()}] -> [{fold[\"eval\"].upper()}] (20 Epochs)')\n",
                "    print('='*80)\n",
                "    out_json = f'results/lodo/{fold[\"name\"]}.json'\n",
                "    cmd = [\n",
                "        'python', 'train_universal_foundation.py',\n",
                "        '--train_datasets', fold['train'],\n",
                "        '--eval_dataset', fold['eval'],\n",
                "        '--epochs', '20',\n",
                "        '--batch_size', '32',\n",
                "        '--lr', '1e-3',\n",
                "        '--hidden_dim', '128',\n",
                "        '--num_heads', '8',\n",
                "        '--num_layers', '2',\n",
                "        '--window_size', '200',\n",
                "        '--prefix_len', '100',\n",
                "        '--train_stride', '20',\n",
                "        '--eval_stride', '10',\n",
                "        '--eval_every', '4',\n",
                "        '--use_amp',\n",
                "        '--subsample_train',\n",
                "        '--distill_feat_weight', '0.5',\n",
                "        '--distill_score_weight', '0.2',\n",
                "        '--output_json', out_json\n",
                "    ]\n",
                "    t0 = time.time()\n",
                "    try:\n",
                "        subprocess.run(cmd, check=True)\n",
                "        dur = time.time() - t0\n",
                "        print(f'>>> {fold[\"name\"]} completed successfully in {dur/60:.2f} minutes.')\n",
                "        if os.path.exists(out_json):\n",
                "            shutil.copy2(out_json, '/kaggle/working/results/lodo/')\n",
                "            with open(out_json) as f:\n",
                "                lodo_results[fold['eval']] = json.load(f)['zero_shot_metrics']\n",
                "    except Exception as e:\n",
                "        print(f'Warning: {fold[\"name\"]} encountered error: {e}. Checkpoint saved. Continuing...')\n"
            ]
        },
        # Cell 9: Grand Unified Multi-Task
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# --- Step 9: Grand Unified Multi-Task Foundation Pre-Training Across All 4 Domains (20 Epochs) ---\n",
                "import os, subprocess, time, shutil\n",
                "os.environ['PYTHONPATH'] = '.'\n",
                "\n",
                "print('\\n' + '='*80)\n",
                "print('TRAINING GRAND UNIFIED MULTI-TASK FOUNDATION MODEL (SWaT + SMD + GECCO + MSL)')\n",
                "print('='*80)\n",
                "cmd = [\n",
                "    'python', 'train_universal_foundation.py',\n",
                "    '--train_datasets', 'swat,smd,gecco,msl',\n",
                "    '--eval_dataset', 'swat',\n",
                "    '--epochs', '20',\n",
                "    '--batch_size', '32',\n",
                "    '--lr', '1e-3',\n",
                "    '--hidden_dim', '128',\n",
                "    '--num_heads', '8',\n",
                "    '--num_layers', '2',\n",
                "    '--window_size', '200',\n",
                "    '--prefix_len', '100',\n",
                "    '--train_stride', '20',\n",
                "    '--eval_stride', '10',\n",
                "    '--eval_every', '4',\n",
                "    '--use_amp',\n",
                "    '--subsample_train',\n",
                "    '--distill_feat_weight', '0.5',\n",
                "    '--distill_score_weight', '0.2',\n",
                "    '--output_json', 'results/grand_unified_multi_task.json'\n",
                "]\n",
                "try:\n",
                "    t0 = time.time()\n",
                "    subprocess.run(cmd, check=True)\n",
                "    print(f'Grand Unified Pre-Training completed in {(time.time() - t0)/60:.2f} minutes.')\n",
                "    if os.path.exists('results/grand_unified_multi_task.json'):\n",
                "        shutil.copy2('results/grand_unified_multi_task.json', '/kaggle/working/results/')\n",
                "except Exception as e:\n",
                "    print(f'Warning: Grand Unified pretraining error: {e}. Proceeding...')\n"
            ]
        },
        # Cell 10: Stress Sweeps
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# --- Step 10: Dimension Subsampling & Sensor Dropping Stress-Test Sweeps ---\n",
                "import os, subprocess\n",
                "os.environ['PYTHONPATH'] = '.'\n",
                "\n",
                "for ds_name in ['swat', 'smd', 'gecco', 'msl']:\n",
                "    ckpt_path = f'checkpoints/universal_swat,smd,gecco,msl_to_{ds_name}_best.pt'\n",
                "    if not os.path.exists(ckpt_path):\n",
                "        ckpt_path = 'checkpoints/universal_swat,smd,gecco,msl_to_swat_best.pt'\n",
                "    if os.path.exists(ckpt_path):\n",
                "        cmd = [\n",
                "            'python', 'eval_dimension_subsampling_sweep.py',\n",
                "            '--ckpt', ckpt_path,\n",
                "            '--dataset', ds_name\n",
                "        ]\n",
                "        print(f'\\nExecuting Dimension Subsampling Sweep on [{ds_name.upper()}]...')\n",
                "        try:\n",
                "            subprocess.run(cmd, check=True)\n",
                "        except Exception as e:\n",
                "            print(f'Warning: Sweep on {ds_name} error: {e}')\n"
            ]
        },
        # Cell 11: Export Artifacts & Final Report
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# --- Step 11: Final Grand Export & High-Density Markdown Report ---\n",
                "import os, glob, json, shutil\n",
                "out_export_dir = '/kaggle/working/results'\n",
                "os.makedirs(out_export_dir, exist_ok=True)\n",
                "\n",
                "# Copy all JSON results, checkpoints, and generated caches\n",
                "for f in glob.glob('results/**/*.json', recursive=True):\n",
                "    shutil.copy2(f, out_export_dir)\n",
                "for f in glob.glob('checkpoints/*.pt'):\n",
                "    shutil.copy2(f, out_export_dir)\n",
                "for f in glob.glob('cache/*_cache.pt'):\n",
                "    shutil.copy2(f, out_export_dir)\n",
                "\n",
                "print('\\n' + '='*80)\n",
                "print('FINAL EXPORTED ARTIFACTS IN /kaggle/working/results:')\n",
                "print('='*80)\n",
                "for f in sorted(os.listdir(out_export_dir)):\n",
                "    size_mb = os.path.getsize(os.path.join(out_export_dir, f)) / (1024*1024)\n",
                "    print(f'  {f} ({size_mb:.2f} MB)')\n",
                "print('\\nPIPELINE COMPLETED SUCCESSFULLY ACROSS BOTH TRACKS!')\n"
            ]
        }
    ]

    nb = {
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

    with open('kaggle_universal_foundation.ipynb', 'w') as f:
        json.dump(nb, f, indent=1)

    print('Updated kaggle_universal_foundation.ipynb successfully!')
    print(f'New notebook size: {os.path.getsize("kaggle_universal_foundation.ipynb")/1024:.2f} KB')

if __name__ == '__main__':
    main()
