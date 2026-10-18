# VLM-Guided Distillation for Time Series Anomaly Detection (TSAD)

This codebase implements a lightweight Student model (1D-CNN) trained via Knowledge Distillation from a frozen Vision-Language Model (VLM) Teacher (Qwen2.5-VL-3B).

The pipeline extracts multi-channel windows, renders them as dual-horizon line plots for the VLM, caches the teacher guidance offline, and distills that knowledge into a fast sub-millisecond student.

---

## Repository Structure

```text
project/
├── config.yaml              # Master configuration file (dataset, model, loss weights)
├── data/                    # Dataset storage (SMD, SWaT, MSL, GECCO)
├── cache/                   # Offline teacher predictions and feature embeddings
├── checkpoints/             # Saved PyTorch student model weights
│
├── src/
│   ├── dataset.py           # Slicing (L=200), Top-K sensor filtering, PyTorch Dataset
│   ├── models/              # Student model architectures
│   │   ├── __init__.py      # Model factory: build_student(...)
│   │   ├── cnn_1d.py        # 1D-CNN (primary proposed student)
│   │   ├── bilstm.py        # BiLSTM (Khan & Das baseline)
│   │   └── timesnet.py      # TimesNet (multi-periodic 2D-conv baseline)
│   ├── loss.py              # Multi-objective KD loss (L_sup, L_feat, L_resp)
│   ├── trainer.py           # PyTorch training and validation loop
│   ├── metrics.py           # Point metrics, Affiliation-F1, Temporal IoU, POT
│   └── teacher_vlm.py       # Dual-horizon canvas plotting and VLM prompt parser
│
├── run_teacher_cache.py     # Step 1: Run VLM teacher and cache outputs
├── train.py                 # Step 2: Train student model with distillation
└── evaluate.py              # Step 3: Evaluate trained student on continuous stream
```

---

## Environment Setup

Always use the default project virtual environment:

```bash
source ~/.venvs/default/bin/activate
cd project
```

Dependencies required: `torch`, `torchvision`, `transformers`, `qwen-vl-utils`, `pandas`, `numpy`, `matplotlib`, `pyyaml`, `scikit-learn`.

---

## How to Run & Execution Workflow

### Step 1: Cache Teacher Guidance (`run_teacher_cache.py`)

**What it does:**  
Extracts representative curriculum windows from the dataset. Renders dual-horizon line charts. Queries the frozen Qwen2.5-VL teacher. Saves the teacher's anomaly scores and 2048-dim latent features to disk.

**Why run it:**  
Running the 3B VLM during training takes ~8.5 seconds per window. Caching runs the VLM once offline. Student training can then run in milliseconds without GPU bottlenecks.

**How to run:**

```bash
# 1. Full teacher caching with Qwen2.5-VL (requires GPU)
python run_teacher_cache.py --num_windows 60

# 2. Mock mode for quick pipeline testing (no GPU or model download needed)
python run_teacher_cache.py --mock --num_windows 20
```

Outputs are saved to `cache/<dataset>_teacher_cache.pt`.

---

### Step 2: Train Student Model (`train.py`)

**What it does:**  
Trains the chosen lightweight student model. Computes the composite distillation loss:
- $\mathcal{L}_{\text{sup}}$: Ground-truth continuous anomaly density matching (BCE).
- $\mathcal{L}_{\text{feat}}$: Alignment between student features and teacher 2048-dim features (MSE).
- $\mathcal{L}_{\text{resp}}$: Soft likelihood matching against teacher scores (KL/BCE).

**Why run it:**  
Transfers visual reasoning into a lightweight student. The trained student runs in $<1\text{ ms}$ for real-time monitoring.

**How to run:**

```bash
# Train default model (1D-CNN on SMD from config.yaml)
python train.py

# Quick test run (few epochs, smaller batch)
python train.py training.epochs=5 training.batch_size=128
```

Checkpoints are saved to `checkpoints/<dataset>_<model>_best.pt`.

---

### Step 3: Evaluate on Continuous Stream (`evaluate.py`)

**What it does:**  
Loads the trained student checkpoint. Slides across the full continuous telemetry sequence (stride $S=1$ or $S=100$). Averages overlapping predictions into a continuous 1D anomaly score curve. Automatically computes a dynamic threshold using Peaks-Over-Threshold (POT). Calculates Affiliation-F1, Point metrics, and Temporal IoU.

**Why run it:**  
Proves real-world detection capability on the continuous timeline. Uses Affiliation-F1 to eliminate the Point-Adjustment (PA) inflation trap.

**How to run:**

```bash
# Full continuous streaming evaluation (stride = 1)
python evaluate.py --stride 1

# Faster coarse evaluation (stride = 100)
python evaluate.py --stride 100

# Evaluate a specific checkpoint
python evaluate.py --checkpoint checkpoints/smd_cnn_1d_best.pt --stride 50
```

---

## Running Ablation Studies (No Code Edits Needed)

You can hot-swap any component directly via command-line arguments.

### 1. Student Architecture Ablations

```bash
# Baseline: 1D-CNN
python train.py student.architecture=cnn_1d

# Ablation A: BiLSTM (Khan & Das baseline)
python train.py student.architecture=bilstm

# Ablation B: TimesNet (2D-variation baseline)
python train.py student.architecture=timesnet
```

### 2. Dataset Swapping

```bash
# Train on SMD (38 channels)
python train.py dataset.name=smd dataset.entity=machine-1-1

# Train on SWaT (51 channels)
python train.py dataset.name=swat

# Train on MSL (55 channels)
python train.py dataset.name=msl
```

### 3. Channel Filter (Top-K) Ablations

```bash
# Use Top-6 by z-score deviation (default)
python train.py channel_filter.strategy=top_k_zscore channel_filter.k=6

# Use Top-8 by variance
python train.py channel_filter.strategy=top_k_variance channel_filter.k=8

# Disable filtering (pass all channels directly)
python train.py channel_filter.strategy=none
```

### 4. Loss Function Ablations

Setting any loss weight to `0.0` turns off that objective.

```bash
# Full multi-objective KD (Default: sup + feat + resp)
python train.py training.loss_weights.w_sup=1.0 training.loss_weights.w_feat=0.5 training.loss_weights.w_resp=1.0

# Response KD only (Hinton style: no feature alignment)
python train.py training.loss_weights.w_feat=0.0

# Feature KD only (FitNets style: no soft response)
python train.py training.loss_weights.w_resp=0.0

# Supervised only (No teacher distillation)
python train.py training.loss_weights.w_feat=0.0 training.loss_weights.w_resp=0.0
```

---

## Summary of Key Knobs in `config.yaml`

| Section | Key | Values | Meaning |
| :--- | :--- | :--- | :--- |
| `dataset` | `name` | `smd`, `swat`, `msl`, `gecco` | Target telemetry dataset. |
| `dataset` | `window_size` | `200` | Local window length ($L$). |
| `dataset` | `stride` | `100` | Training window stride ($50\%$ overlap). |
| `channel_filter` | `strategy` | `median_shift`, `top_k_zscore`, `top_k_variance`, `none` | Active sensor selection logic. |
| `channel_filter` | `k` | `6` | Number of channels to plot and feed. |
| `student` | `architecture` | `cnn_1d`, `bilstm`, `timesnet` | Student backbone architecture. |
| `student` | `hidden_dim` | `128` | Latent layer capacity. |
| `training` | `loss_weights` | `w_sup`, `w_feat`, `w_resp` | Relative weights for the three losses. |

---

## Benchmark Datasets & Automated Setup

VisTAD supports four major multivariate benchmarks, complete with pre-computed teacher caches:
- **SMD**: 38 continuous channels across 28 server entities ([NetManAIOps/OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly)).
- **SWaT**: 51 physical sensors & actuators across 6 water treatment stages ([iTrust SUTD](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/)).
- **MSL**: 55 telemetry channels from NASA's Curiosity Mars rover ([NASA JPL Telemanom](https://github.com/khundman/telemanom)).
- **GECCO**: 9 drinking water quality monitoring sensors ([Zenodo Record 3884398](https://zenodo.org/records/3884398)).

### 1-Click Dataset Builder
To automatically fetch all benchmarks, clean SWaT, sync teacher caches, and package the Kaggle-ready archive (`data_repository.zip`):

```bash
python scripts/build_data_repository.py
```

### Running Distillation on Any Benchmark
Each benchmark has a ready-to-use profile under `configs/`:

```bash
# 1. Offline VLM Teacher Caching (or use pre-computed cache in cache/)
python run_teacher_cache.py --config configs/smd.yaml    # Server machines
python run_teacher_cache.py --config configs/swat.yaml   # Cyber-physical water
python run_teacher_cache.py --config configs/msl.yaml    # Spacecraft rover
python run_teacher_cache.py --config configs/gecco.yaml  # Water quality

# 2. Student Training (Channel-Independent 1D-CNN)
python train.py --config configs/<dataset>.yaml

# 3. Continuous Streaming POT Evaluation
python evaluate.py --config configs/<dataset>.yaml
```


