# VisTAD: Vision-Language Model Guided Knowledge Distillation for Time-Series Anomaly Detection

> **Note on Student Modeling:** We do not work intensively with per-dataset fine-tuned students; our primary focus and investigation was on NASA MSL, where our architecture and decoupled modeling yield significant performance improvements.

## Environment Setup

```bash
pip install -r requirements.txt
```

---

## Reproducing Results

All evaluation scripts strictly utilize pre-generated teacher representations and evaluation windows located in `cache/`.

### 1. Reproducing Continuous Chronological Streaming Evaluation
To run full chronological streaming evaluation on the held-out test partitions:

```bash
# 1. NASA MSL Spacecraft (Dual-stream decoupled analog/command flags)
python eval_msl_continuous_streaming.py

# 2. GECCO Drinking Water Quality (Dual-subsystem chemical/hydraulic)
python eval_gecco_continuous_streaming.py

# 3. SMD Server Cluster (38-channel multi-tier telemetry)
python eval_smd_continuous_streaming.py
```

### 2. Reproducing Table 1: Teacher Model Cross-Benchmark Comparison
To reproduce the symmetric cross-benchmark comparison of the **Older VLM Teacher (Qwen2.5-VL-3B)** vs. the **Newer Visual Reasoning Teacher (Ultra-VETime DINOv2-Large)** across all 4 datasets (SWaT, GECCO, SMD, NASA MSL) using the latest evaluation caches:

```bash
python scripts/reproduce_table1_teacher_comparison.py
```

This script evaluates:
* **SWaT SCADA**: `cache/swat_eval_head_to_head_cache.pt` ($N=1,082$ windows) vs. historical cache ($N=5,000$ windows).
* **GECCO Drinking Water**: `cache/gecco_eval_head_to_head_cache.pt` ($N=1,464$ windows) vs. historical cache ($N=2,000$ windows).
* **SMD Server Cluster**: `cache/smd_eval_head_to_head_cache.pt` ($N=1,484$ windows across all 28 servers) vs. historical all-28 pooled caches `cache/smd_machine-*_teacher_cache.pt` ($N=13,209$ windows pooled; $N=3,619$ held-out slice; $N=532$ on `machine-1-1`).
* **NASA MSL Spacecraft**: `cache/msl_eval_head_to_head_cache.pt` ($N=1,393$ windows) vs. historical cache ($N=1,000$ windows).

### 3. Teacher Cache Artifacts Overview

For the latest Ultra-VETime (DINOv2-Large) teacher, each benchmark has **two separate caches**:
1. **Training Cache (`*_train_ultravetime_cache.pt`)**: Used during student training to provide 1024-d teacher embeddings for feature distillation ($\mathcal{L}_{\text{feat}}$).
2. **Evaluation Cache (`*_eval_head_to_head_cache.pt`)**: Used on held-out test partitions to evaluate teacher scores and benchmark student anomaly detection head-to-head.

| Benchmark | Older Teacher Cache (Qwen2.5-VL-3B, 2048-d) | Ultra-VETime Training Cache (DINOv2-L, 1024-d) | Ultra-VETime Evaluation Cache (DINOv2-L, 1024-d) | Evaluation Scope |
| :--- | :--- | :--- | :--- | :--- |
| **NASA MSL** | `cache/msl_msl_test_teacher_cache.pt` | `cache/msl_train_ultravetime_cache.pt` | `cache/msl_eval_head_to_head_cache.pt` | Partition 2 Held-Out (1,393 windows) |
| **SWaT** | `cache/swat_swat_attack_teacher_cache.pt` | `cache/swat_train_ultravetime_cache.pt` | `cache/swat_eval_head_to_head_cache.pt` | Partition 2 Held-Out (1,082 windows) |
| **GECCO** | `cache/gecco_gecco_water_teacher_cache.pt` | `cache/gecco_train_ultravetime_cache.pt` | `cache/gecco_eval_head_to_head_cache.pt` | Partition 2 Held-Out (1,464 windows) |
| **SMD** | `cache/smd_machine-1-1_teacher_cache.pt` .. `machine-3-11` | `cache/smd_train_ultravetime_cache.pt` | `cache/smd_eval_head_to_head_cache.pt` | All 28 Servers Pooled (1,484 windows) |

---

## Train Student Models

```bash
# 1. NASA MSL: Dual-Stream Decoupled TCN Student (Primary focus)
python train_msl_student.py

# 2. Universal Foundation Any-Dimension Model (Channel Tokenization & Cross-Attention)
python train_universal_foundation.py

# 3. SWaT: 6-Stage Subsystem Coupled TCN
python train_stage_coupled_student.py

# 4. GECCO: Water Quality Dual-Subsystem Coupled TCN
python train_gecco_student.py

# 5. SMD: Pooled 28-Server 4-Tier Subsystem TCN
python train_smd_pooled_student.py
```

---

## Running Ablation Studies

The suite includes dedicated runner scripts and self-contained notebooks to reproduce all empirical ablations across loss objectives, distillation weights, teacher manifolds, and channel selection:

### 1. Loss Component & Distillation Weight Sweeps
Evaluates BCE-Only, Full VisTAD (Focal + Temporal Dice), and distillation weight variations ($\lambda_{\text{feat}} \in [0.00, 2.00]$):

```bash
# Run weight sweep for a specific benchmark (e.g. NASA MSL, SWaT, GECCO, SMD):
python run_loss_and_weight_ablation.py --dataset msl
python run_loss_and_weight_ablation.py --dataset swat

# Run the grand loss variant suite (pure BCE, focal, focal+dice, unified distill, BCE+feat):
python run_grand_loss_ablation.py --variants pure_bce focal focal_dice unified_distill bce_feat --epochs 5
```

### 2. Baseline Architecture & Cache Ablation
Compares 1D-CNN (3L) vs. Residual-TCN (5L) under Old (Qwen2.5-VL) vs. New (Ultra-VETime DINOv2) teacher manifolds:

```bash
python run_baseline_cache_ablation.py
```

### 3. Cross-Compatibility (Student Architecture × Teacher Cache)
Evaluates the $2 \times 2$ matrix of student inductive biases (generic channel-independent vs. domain-coupled / channel-aware) across both teacher caches:

```bash
python run_cross_student_teacher_ablation.py
```

### 4. Top-K Channel Selection Training
Evaluates training with variance- and z-score-filtered channel subsets ($K \in \{1, 3, 5, 10, \text{all}\}$):

```bash
python run_top_k_training_ablation.py
```

### 5. Comprehensive Checkpoint Evaluation Registry
To evaluate and audit all saved checkpoints across all benchmarks:

```bash
python scripts/eval_master_comprehensive_registry.py
```

---

## Launching the Demo App

A live interactive test interface is provided:

```bash
cd streamlit_app
streamlit run app.py
```

*What to do in the demo?*
- **Domain Selector:** Choose between SMD, SWaT, MSL, or GECCO.
- **Threshold Calibration Slider:** Tune your decision threshold for the score given by the models.
- **Select smoothing parameters:** Choose minimum continuous anomaly scores and window clustering to eliminate isolated false positives.

---

## Checkpoints and Submission Artifacts

- **Production Checkpoints:** The best model checkpoints for each domain and the universal foundation model are located in `checkpoints/release/`.
- **Course Submission Deliverables:** The final course presentation deliverables are located directly at the project root:
  - `Null_Terminators_IE643_CourseProject_Poster.pdf` (Official A1 Poster)
  - `Null_Terminators_IE643_NoveltyAssessment_report.pdf` (Novelty Assessment Report)
