# VLM4TS Engine: Vision-Language Model Guided Time Series Anomaly Detection

This module implements a two-stage multimodal framework for time-series anomaly detection inspired by **VLM4TS** (*AAAI 2026 Oral*, [arXiv:2506.06836](https://arxiv.org/abs/2506.06836)). It replaces naive single-axis rolling-window VLM prompting with structured, context-aware visual reasoning specifically designed for complex non-spike anomalies (such as industrial cyber-physical attacks in SWaT).

---

## Architecture Overview

```
                      Raw Multivariate Time Series (e.g. SWaT 51 channels)
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
        Continuous Analog Sensors               Discrete Actuator States
        (Levels, Flows, Pressures)              (Pumps, Motorized Valves)
                    │                                       │
                    └───────────────────┬───────────────────┘
                                        ▼
                  Multi-Scale Visual Renderer (renderer.py)
         ┌──────────────────────────────────────────────────────────┐
         │ Panel 1: Historical Macro Baseline (Steps -800 to 0)     │
         │ Panel 2: Micro Target Inspection Window (Steps 0 to 200) │
         │   • Analog channels with baseline confidence bands       │
         │   • Actuators on discrete step-state axes                │
         └──────────────────────────────┬───────────────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
      Stage 1: ViT4TS Screener               Stage 2: VLM4TS Reasoner
      (vit_screener.py)                      (vlm_reasoner.py)
      • Lightweight Vision Transformer       • Frontier Multimodal VLM
      • Fast visual embedding extraction       (Qwen2.5-VL / ChartGemma)
      • Rapid sliding-window screening       • Invariant violation reasoning
      • Filters candidates with zero LLM     • Flatlines & stuck-at detection
        token overhead                       • Outputs continuous likelihood,
                                               localized intervals, dense
                                               guidance mask, & latent features
                                        │
                                        ▼
                        Multi-Objective Student Distillation
```

---

## Key Solutions to Prior Deficiencies

1. **Elimination of Phase-Locking Bias ($t = 50$)**:
   - `generate_cache.py` samples windows across uniform phase distributions: early onset (10%), mid-window (25%, 50%), late onset (75%), and full-window sustained plateaus (100% attack).
2. **Elimination of the $\pm 3\sigma$ Spike Prompt Trap**:
   - The reasoning prompt in `vlm_reasoner.py` directs the model to detect physical invariant violations (pump ON but zero flow), sensor flatlines / freezes, and level offsets inside the normal band.
3. **Decoupled Visual Subplots**:
   - `renderer.py` separates continuous analog sensors from discrete actuator states, preventing visual collision.
4. **Token & Latency Efficiency**:
   - `ViT4TSScreener` can rapidly pre-screen thousands of sliding windows, only calling the VLM on flagged candidate regions.

---

## Quickstart

### 1. Test the Visual Renderer
```python
from src.dataset import load_dataset_arrays
from vlm4ts.renderer import render_vlm4ts_image

raw_data, labels = load_dataset_arrays("swat", "swat_attack", "data")
img = render_vlm4ts_image(raw_data, start_idx=2000, end_idx=2200, dims=[1, 5, 7, 16, 18, 28])
img.save("sample_vlm4ts_render.png")
```

### 2. Generate Phase-Diverse Teacher Cache
```bash
python -m vlm4ts.generate_cache --dataset swat --entity swat_attack --num_windows 500
```

### 3. Benchmark Legacy Teacher vs. VLM4TS Teacher
```bash
python -m vlm4ts.benchmark_teachers --dataset swat --entity swat_attack
```

### 4. Reproducing Evaluation (All 44 Models)
To evaluate all 44 models (24 factorial cross-ablation models, 16 baseline models, and 4 loss-tuned models) strictly over the entire held-out continuous test partition without point adjustment:

```bash
# Ensure ~/.venv is activated (Python 3.12 with MPS / CUDA support)
source ~/.venv/bin/activate

# Execute continuous streaming inference over entire test partitions:
python scripts/eval_master_comprehensive_registry.py
```

- **Evaluated Test Streams**:
  - `SWaT (51D)`: Entire Partition 2 ($t \in [224,960, 449,919]$, 224,960 continuous seconds, 14 unseen attacks).
  - `GECCO (9D)`: Entire Partition 2 ($t \in [69,783, 139,566]$, 69,783 continuous seconds, 22 unseen events).
  - `SMD (38D)`: Chronological last 50% across all 28 servers (354,210 continuous timesteps, 122 unseen events).
  - `NASA MSL (55D)`: Chronological test stream ($t \in [36,864, 73,729]$, 36,865 continuous steps, 17 unseen episodes).
- **Metric Outputs**:
  - Point-wise: Continuous AUROC, PR-AUC, Best Point-$F_1$, Point Precision, Point Recall, FPR at $\tau^*$.
  - Range-aware: Affiliation $F_1$ (`Aff-F1`), Aff-Precision, Aff-Recall (without artificial point-adjustment credit).
  - Event-level: Contiguous Event Recall ($\text{detected} / \text{total}$ events and percentage).
  - Operational: Parameter count, Latency per window (ms), Throughput (win/s), Speedup vs Teacher.
- **Results Registry**: Saved directly to `results/master_comprehensive_evaluation_registry.json`.
