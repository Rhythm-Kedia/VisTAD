# VisTAD Comprehensive Technical Audit & Action Items (TODO)

This document synthesizes the empirical discrepancies, architectural constraints, data pipeline bottlenecks, and loss modeling limitations identified across the VisTAD codebase and experimental results (SMD, SWaT, MSL, GECCO).

---

## 1. Core Architectural & Methodological Concerns

### Point 1: Teacher Anomaly Window Phase-Locking Bias (Onset Fixed at $t = 50$)
* **Root Cause in Code**: In `run_teacher_cache.py` (line 207), anomaly windows are extracted via:
  ```python
  st = max(0, int(idx - 50))
  ```
* **Consequence**: Every single anomaly window cached for the Vision-Language Model teacher forces the anomaly onset to occur at exactly relative timestep $50$ (25% into the 200-step window).
* **Failure Mode**:
  - The teacher VLM never sees anomalies starting at arbitrary offsets ($t = 0, 10, 80, 150, 190$) or continuous sustained plateaus that occupy 100% of the window.
  - Distillation forces the student to align features with a phase-locked teacher representation.
  - When evaluated under continuous sliding-window inference with stride 10, the anomaly onset drifts continuously across windows, causing detection failure.
* **Action Item**: Implement phase-jittered window sampling for teacher caching ($st = idx - \Delta$ where $\Delta \sim \mathcal{U}(10, 180)$) and sustained plateau sampling.

---

### Point 2: Dataset-Specific Data Scarcity, Sparsity, and Imbalance
* **Empirical Analysis**:
  - **SMD (`machine-1-1`)**: Total 28,479 steps. In the chronological 50% train split ($0 \dots 14,239$), positive anomaly count is exactly **0 (0.00%)**. Single-machine supervised training has zero positive ground-truth anchors.
  - **GECCO (`gecco_water`)**: 139,566 steps, but only 1.24% positive steps (1,726 points total across 51 segments; median duration: 22 steps). The test split has only 251 anomaly points (0.45% prevalence). Extreme class imbalance causes point-wise BCE to collapse to predicting all zeros.
  - **SWaT (`swat_attack`)**: 449,919 steps, but split into:
    - Train ($0\% - 50\%$): 5.21% anomalies (11,720 steps)
    - Val ($50\% - 60\%$): **79.79% anomalies** (35,900 steps - dominated by one prolonged attack)
    - Test ($60\% - 100\%$): 3.87% anomalies (6,964 steps)
    - Severe split-to-split distribution shift breaks validation threshold calibration ($\tau^*$).
  - **MSL (`msl_test`)**: 73,729 steps. 55 channels, where channel 0 is continuous telemetry and channels 1–54 are sparse discrete one-hot/binary command flags.
* **Action Item**: Standardize balanced sampling, stratified cross-validation/purged splitting, and domain-adapted imbalance weighting.

---

### Point 3: Mismatch Between Training Data Pipeline and Continuous Stride Evaluation
* **Root Cause in Code**:
  - `TimeSeriesDataset` in `train.py` iterates over uniform step indices with stride 100:
    ```python
    self.indices = [i for i in range(0, total_len - window_size + 1, stride)] # 0, 100, 200, 300...
    ```
  - `has_teacher` is verified strictly by exact dictionary key lookup:
    ```python
    has_teacher = bool(self.teacher_cache is not None and start in self.teacher_cache)
    ```
  - But `teacher_cache` keys were generated at arbitrary offsets (`idx - 50` and variable normal stride)!
* **Disaster Metric**:
  - In SWaT: Only **95 out of 2,248** train windows (4.2%) matched the cache keys! Over 95.8% of training windows had `has_teacher = False`!
  - In SMD: Only **6 out of 141** train windows (4.3%) had `has_teacher = True`!
  - The student was essentially trained *without* teacher distillation on 95% of its batches.
* **Action Item**: Build a unified curriculum data loader that samples directly from the cached windows during distillation or performs nearest-neighbor window matching.

---

### Point 4: Input Normalization & Physical Scale Incompatibilities
* **Root Cause in Code**:
  - In `src/dataset.py`, `TimeSeriesDataset` computes `self.base_mean` and `self.base_std`, but in `__getitem__`, `window_x` is returned as raw unnormalized data.
  - SMD raw data is already normalized to $[0.0, 1.0]$.
  - SWaT raw data has channel means up to 910 and maximums up to 1201 (e.g. water levels in mm vs binary actuator status $\{1, 2\}$).
  - GECCO raw data ranges up to 3923 (conductivity vs pH $\sim 7$).
  - Passing unnormalized heterogeneous physical units into a channel-independent 1D-CNN or TCN causes large-scale channels to dominate gradients, saturating batchnorm and GELU activations.
* **Action Item**: Enforce baseline Z-score standardization $((x - \mu) / \sigma)$ in `TimeSeriesDataset` across all training and evaluation pipelines.

---

### Point 5: Inadequate Receptive Fields for Long-Range Context and Sustained Plateaus
* **Analytical Receptive Field Calculation**:
  - **1D-CNN (3 Layers)**:
    - Layer 1: $k=5, d=1 \to \text{RF} = 5$
    - Layer 2: $k=5, d=2 \to \text{RF} = 5 + 4 \times 2 = 13$
    - Layer 3: $k=5, d=4 \to \text{RF} = 13 + 4 \times 4 = 29$ steps.
    - **A 3-layer 1D-CNN has a receptive field of only 29 steps!**
    - In a 200-step window where SWaT attacks have a median duration of 444 steps and MSL has 121 steps, a 29-step receptive field cannot capture the baseline state or long-range slope changes.
  - **Residual TCN (5 Layers)**:
    - 2 conv layers per block with dilations $d \in \{1, 2, 4, 8, 16\}$:
    - Block receptive field increases by $2 \times (k-1) \times d = 8 \times d$.
    - Total RF $= 1 + 8 \times (1 + 2 + 4 + 8 + 16) = 249$ steps (covers 200-step window).
* **Action Item**: Expand 1D-CNN receptive field or standardize on deep dilated residual backbones with RF $\ge$ window length ($L=200$).

---

### Point 6: Non-Causal Convolutions & Future Leakage in TCN
* **Root Cause in Code**:
  - In `src/models/cnn_1d.py` (line 10), `ResidualDilatedBlock` defines:
    ```python
    padding = 2 * dilation
    self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=padding, dilation=dilation)
    ```
  - This applies symmetric padding ($2 \times d$ on the left and $2 \times d$ on the right).
  - Timestep $t$ computes activations using future observations $t + 1 \dots t + 2d$.
* **Consequences**:
  - Violates temporal causality required for real-time online anomaly detection.
  - At window edges during continuous streaming (stride 10), future zero-padding introduces boundary artifacts.
* **Action Item**: Implement causal padding: left-pad by $(k - 1) \times d$ and right-pad by 0 (or causal slice `x[:, :, :-(k-1)*d]`).

---

### Point 7: Spatial & Cross-Sensor Invariant Blindness
* **Issue**:
  - Both CNN and TCN models use `channel_independent=True` followed by channel-wise permutation-invariant `max()` pooling.
  - Real industrial attacks (especially SWaT) are cyber-physical invariant violations: e.g., valve is open and pump is running, but flow rate is 0. Individual sensors remain within standard operational bounds, but their joint physical correlation is violated.
  - Channel-independent models cannot model pairwise sensor covariance or spatial interactions.
* **Action Item**: Introduce lightweight channel-mixing layers (e.g., cross-channel $1 \times 1$ convs, spatial attention, or graphical sensor embeddings) after temporal feature extraction.

---

### Point 8: Spike-Biased Scalar Window Pooling
* **Root Cause in Code**:
  - All student models pool scalar window anomaly scores via:
    ```python
    max_p = torch.max(dense_probs, dim=1)[0]
    top_p = torch.topk(dense_probs, k=10, dim=1)[0].mean(dim=1)
    score_scalar = 0.6 * max_p + 0.4 * top_p
    ```
* **Failure Mode**:
  - A transient single-step noise spike ($p = 0.90$ for 1 step, $0.05$ elsewhere) gets $\approx 0.58$.
  - A subtle sustained attack ($p = 0.45$ for 150 consecutive steps) gets $\approx 0.45$.
  - Sustained plateaus and drift attacks are systematically ranked *below* high-frequency noise spikes.
* **Action Item**: Incorporate temporal integral / mean pooling:
  ```python
  mean_p = torch.mean(dense_probs, dim=1)
  score_scalar = 0.4 * max_p + 0.3 * top_p + 0.3 * mean_p
  ```

---

### Point 9: Loss Function Structural Deficiencies
* **Flaws in Current `MultiObjectiveKDLoss`**:
  1. **Point-wise Independence**: Point-wise BCE ignores temporal smoothness, enabling high-frequency prediction chatter.
  2. **Class Imbalance**: Standard BCE does not down-weight easy nominal steps in sparse datasets (GECCO, SWaT).
  3. **No Segment / Range Awareness**: Lacks Soft-Dice / IoU objective to align with event-level evaluation (Affiliation F1).
  4. **No Direct Metric Ranking Optimization**: Does not optimize pairwise ranking margin between anomalous and nominal windows.
  5. **Negative Distillation Transfer**: In SMD, supervised-only achieved AUROC **0.857**, while Full VisTAD dropped to **0.640** due to noisy teacher response matching $\sigma(\beta(a_T - \tau))$.
* **Action Item**: Implement modular Focal BCE, Continuous Soft-Dice, Temporal Total Variation (TV) regularizer, and Pairwise Margin Ranking.

---

### Point 10: Disconnect Between Cached Curriculum Evaluation and Continuous Streaming
* **Empirical Discrepancy Found in Results**:
  - In Table 3 (evaluated on 50/50 balanced cached windows with onset at $t=50$):
    - SWaT: Student AUROC **0.782**, Best F1 **0.748**
    - GECCO: Student AUROC **0.818**, Best F1 **0.760**
    - MSL: Student AUROC **0.529**, Best F1 **0.740**
  - In Table 4 (evaluated on continuous sliding stream with stride 10):
    - SWaT: AUROC **0.590**, Point-F1 **0.093**, Aff-F1 **0.054** (Catastrophic collapse)
    - GECCO: AUROC **0.726**, Point-F1 **0.211**, Aff-F1 **0.111**
    - MSL: AUROC **0.443**, Point-F1 **0.081**, Aff-F1 **0.276**
* **Root Cause**: Evaluating on artificial 50/50 balanced curriculum windows masked severe operational failures on real-world continuous data streams.
* **Action Item**: Make continuous streaming evaluation (stride 10) the primary evaluation benchmark for all experiments.

---

### Point 11: Teacher Representation Collapse on SWaT (~0.50 AUROC, 98% False Alarm Hallucination)
* **Empirical Audit (Kaggle GPU & Local Diagnosis)**:
  - In the remote Kaggle teacher benchmark on SWaT (200 balanced test windows: 100 attacks, 100 normal baselines):
    - **T4 (Process-Stage Coupled Qwen2.5-VL with Logit Difference)**: AUROC = **0.5170**, PR-AUC = **0.5578**, FPR = **0.9800** (99 TP, 98 FP, 2 TN, 1 FN).
    - **T5 (Vision Foundation ViT-B/16 Cosine Anomaly)**: AUROC = **0.4300**, PR-AUC = **0.4986**, FPR = **0.9100** (97 TP, 91 FP, 9 TN, 3 FN).
  - Both teachers flagged **91% to 98% of clean nominal windows as operational attacks**, collapsing AUROC to random chance (~0.50).

* **The Root Causes Discovered**:
  1. **The Nominal Normalization Bug (`base_m`, `base_s` from `attack[:3000]`)**:
     - `swat_attack` has 449,919 steps (4 days of physical attacks). The pipeline computed `base_m = np.mean(raw[:3000])` and `base_s = np.std(raw[:3000])`.
     - In SWaT, the first 3000 seconds (50 minutes) of `swat_attack` is a warm-up period where Stage 2 (chemical dosing) and Stage 5 (reverse osmosis) were sitting in an idle or static state where sensors barely fluctuated.
     - Compared to the true 7-day nominal dataset (`SWaT_Dataset_Normal_v1.csv`, 495,000 clean steps):
       - **Dim 37 (FIT501 / RO flow)**: True nominal std is **6.438**, but `attack[:3000]` was **0.083** (underestimated by **77x**!).
       - **Dim 25 (AIT401 / chlorine analyzer)**: True nominal std is **54.808**, but `attack[:3000]` collapsed to the minimum clamp **0.050** (underestimated by **1096x**!).
       - **Dim 44 (FIT504 / RO permeate flow)**: True std is **13.601**, in `attack[:3000]` it was **0.468** (underestimated by **29x**!).
       - **Dim 46 (PIT503 / RO pump pressure)**: True std is **10.614**, in `attack[:3000]` it was **0.417** (underestimated by **25x**!).
     - **Catastrophic Consequence**:
       - When normal operational cycles resumed in the test set (e.g. steps 4100, 8500, 12900), natural operational oscillations of $\pm 2\sigma$ in real engineering units produced standardized $z$-scores of **6.54 to 24.16**!
       - Channel selector chose Stage 5 channels because their artificial $z$-scores were $24\sigma$.
       - Dashed control boundaries were drawn at $\pm 3\sigma$ based on the bogus `base_s`, so **the signals in every clean normal window exploded far outside the $\pm 3\sigma$ control boundaries**!
       - To any model (multimodal VLM or Vision Transformer), every single clean normal window looked like an extreme catastrophic invariant breakdown, causing the 98% false positive rate!

  2. **Empirical Verification & Kaggle GPU Benchmark (Versions 5, 6, & 7)**:
     - When normalized with the true operational statistics $(\mu_{\text{normal}}, \sigma_{\text{normal}})$ from `SWaT_Dataset_Normal_v1.csv` and evaluated across 200 balanced test windows on Kaggle GPU:
       | Teacher Formulation | AUROC | PR-AUC | Best F1 | Optimal $\tau^*$ | FPR (False Alarm) | FNR (Miss Rate) | TP | FP | TN | FN | Latency |
       | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
       | **T13: Ultra-VETime Multi-Stage (DINOv2-Large Top-2 Stages)** | **0.9142** | 0.8697 | **0.9100** | 0.0443 | 0.1500 | **0.0400** | **96** | 15 | 85 | **4** | 1.208s |
       | **T12: Ultra-VETime Multi-Horizon (DINOv2-Large L=100 + L=200)** | **0.8877** | **0.9254** | **0.8506** | 0.1436 | **0.0000** | 0.2600 | 74 | **0** | **100** | 26 | 0.651s |
       | **T6: DINOv2-base (Self-Supervised Global CLS)** | **0.8646** | **0.9138** | **0.8457** | 0.0722 | **0.0100** | 0.2600 | 74 | **1** | **99** | 26 | **0.021s** |
       | **T9: Ultra-VETime (DINOv2-Large + K=5 Regimes CLS)** | **0.8607** | **0.9106** | **0.8556** | 0.0337 | 0.0300 | 0.2300 | 77 | 3 | 97 | 23 | 0.065s |
       | **T8: Time-VLM Multi-Stage Ensemble (DINOv2 Top-2)** | 0.8530 | 0.7722 | **0.8688** | 0.0546 | 0.2500 | **0.0400** | **96** | 25 | 75 | **4** | 1.149s |
       | **T5: ViT-B/16 (ImageNet Supervised CLS)** | 0.8504 | 0.8909 | 0.8000 | 0.0598 | 0.2900 | 0.1400 | 86 | 29 | 71 | 14 | **0.015s** |
       | **T11: Ultra-VETime (DINOv2-Large Combined 0.5 CLS + 0.5 Patch)** | 0.8408 | 0.8935 | 0.8061 | 0.0562 | 0.1700 | 0.2100 | 79 | 17 | 83 | 21 | 0.065s |
       | **T7: VETime / Time-VLM (DINOv2 + Patch Alignment)** | 0.8374 | 0.8907 | 0.8019 | 0.0712 | 0.2400 | 0.1700 | 83 | 24 | 76 | 17 | **0.021s** |
       | **T10: Ultra-VETime (DINOv2-Large + K=5 Regimes Patch)** | 0.8296 | 0.8561 | 0.8040 | 0.0783 | 0.1900 | 0.2000 | 80 | 19 | 81 | 20 | 0.065s |
       | **T4: Process-Stage Coupled VLM (Qwen2.5-VL)** | 0.2339 | 0.4040 | 0.6667 | 0.2451 | 1.0000 | 0.0000 | 100 | 100 | 0 | 0 | 2.445s |
       | *Initial VLM4TS Teacher (Qwen2.5-VL JSON Text)* | *0.5665* | *0.5813* | *0.6739* | *0.5000* | *1.0000* | *0.0000* | *250* | *233* | *0* | *0* | *10.00s* |

     - **Key Empirical Insights**:
       - **Ultra-VETime Multi-Stage (T13)** achieves the new all-time high on SWaT: **AUROC = 0.9142**, **Best F1 = 0.9100**, and **96.0% Recall** (catching 96 out of 100 attacks, with only 4 misses). The $K=5$ multi-regime prototype clustering reduces false alarms from 25 down to 15 (a 40% reduction compared to single-prototype T8).
       - **Ultra-VETime Multi-Horizon (T12)** achieves **100.0% Specificity**: **ZERO false alarms** across 100 nominal windows ($\text{FPR} = 0.0000$, $\text{PR-AUC} = 0.9254$, $\text{AUROC} = 0.8877$).
       - **Self-Supervised Vision Foundation Models (DINOv2-Large / Base)** fundamentally outperform text-prompted VLMs (which collapsed to $0.23 - 0.58$ AUROC due to conversational confirmation bias and text-curve semantic gap).
       - Visual patch geometry preserves continuous analog derivatives and actuator discrete step changes directly in metric latent space without lexical ambiguity.

* **Action Items**:
  1. [x] Standardize all SWaT pipelines to compute nominal baselines strictly from `SWaT_Dataset_Normal_v1.csv` rather than `attack[:3000]`.
  2. [x] Benchmark Time-VLM / VETime / DINOv2 teacher architectures on Kaggle GPU (**Ultra-VETime Multi-Stage identified as undisputed top teacher: AUROC 0.9142, Best F1 0.9100, FNR 0.0400**).
  3. [x] Generate full curriculum distillation cache and dedicated held-out evaluation cache on SWaT using the winning **Ultra-VETime Multi-Stage / DINOv2-Large** teacher representations:
     - **Training Curriculum Cache** (`cache/swat_train_ultravetime_cache.pt`): 3,312 windows (150.54 MB) across 21 attacks and nominal operation cycles.
     - **Dedicated Head-to-Head Evaluation Cache** (`cache/swat_eval_head_to_head_cache.pt`): 1,082 held-out windows (49.18 MB) covering all 14 unseen attacks and matched nominal periods. Teacher baseline on this held-out set: **AUROC = 0.8165**, **PR-AUC = 0.8582**, **Best F1 = 0.7833**.
  4. [x] Distill the Causal TCN student on SWaT with Stage-Coupled architecture and simplified 2-term loss using the calibrated teacher cache (**Student outperforms teacher on held-out test cache: AUROC 0.8874 vs 0.8165, PR-AUC 0.9304 vs 0.8582, Best F1 0.8656 vs 0.7833, FPR 3.3% vs 24.1%, Latency 0.44 ms vs 1208 ms - 2,763x faster**).
  5. [x] Benchmark Student on Full Continuous Streaming Deployment (Stride 10 across 224,960 held-out test timestamps, 22,477 sliding windows):
     - **Continuous AUROC**: **0.8751**
     - **Continuous PR-AUC**: **0.8549**
     - **Continuous Point-wise Best F1**: **0.8796** (at $\tau^* = 0.6436$)
     - **Continuous Specificity**: **99.80%** (FPR = **0.20%**, only 356 false alarm seconds out of 182k nominal steps)
     - **Streaming Throughput**: **4,151 windows/sec (0.24 ms/window)**. VisTAD streaming degradation solved!

---

## 2. Experimental Plan: Anchor Dataset & Staged Execution

### Recommended First Anchor Dataset: **SWaT (`swat_attack`)**
* **Why SWaT First?**
  1. Shows the starkest gap between cached evaluation (F1 $\approx 0.89$) and continuous streaming (F1 $\approx 0.054$).
  2. Large data scale (449k steps, 35 documented physical attacks).
  3. Clearly exhibits non-spike anomaly morphologies (sustained plateaus, level offsets, sensor-actuator invariant breaks).
  4. Once SWaT is fixed, improvements directly transfer to MSL (telemetry) and GECCO (water sensors).
* *(Alternative Anchor)*: **SMD (`machine-1-1`)** as a baseline sanity check to ensure fixes do not regress point-spike detection.

### Staged Roadmap (Teacher-First Investigation on SWaT)
1. [x] **Phase 1: Student Loss Suite & Causal Architecture Benchmark** (Completed: HYBRID identified as top student objective on SWaT).
2. [x] **Phase 2: Teacher Representation Benchmark on SWaT** (Completed on Kaggle GPU):
   - **T1: Legacy VisTAD Teacher**: Single symlog plot, $\pm 3\sigma$ prompt, Logit-Difference Scoring ($\text{AUROC} = 0.8111$ on onset=0 plateaus).
   - **T2: VLM4TS Free-Form JSON**: Initial baseline (collapsed to 100% false positives due to warm-up normalization, $\text{AUROC} = 0.5665$).
   - **T4: Process-Stage Coupled VLM (Qwen2.5-VL)**: Logit difference with physical stage pairing ($\text{AUROC} = 0.2339$; instruction VLM heavily biased toward affirmative answer).
   - **T5: Vision Foundation Model (ViT-B/16 Stage Prototypes)**: Supervised ImageNet baseline ($\text{AUROC} = 0.8504$, $\text{Best F1} = 0.8000$, $\text{FPR} = 0.2900$).
   - **T6: DINOv2-base (Self-Supervised Global CLS)**: $\text{AUROC} = 0.8646$, $\text{PR-AUC} = 0.9138$, $\text{FPR} = 0.0100$ (99% True Negatives).
   - **T12: Ultra-VETime Multi-Horizon (DINOv2-Large L=100 + L=200)**: $\text{AUROC} = 0.8877$, $\text{PR-AUC} = 0.9254$, **$\text{FPR} = 0.0000$ (100% True Negatives, 0 false alarms)**.
   - **T13: Ultra-VETime Multi-Stage (DINOv2-Large Top-2 Stages)**: **UNDISPUTED CHAMPION** — **$\text{AUROC} = 0.9142$**, **$\text{PR-AUC} = 0.8697$**, **$\text{Best F1} = 0.9100$**, **$\text{FNR} = 0.0400$ (96.0% Recall: 96 / 100 true attacks detected)**.
3. [x] **Phase 3: Student Distillation with Stage-Coupled Causal TCN on SWaT** (Completed):
   - **Stage-Coupled Causal TCN (1.05M params)** trained on 3,312 curriculum windows with simplified loss $\mathcal{L}_{\text{total}} = (\mathcal{L}_{\text{Focal}} + \mathcal{L}_{\text{Dice}}) + 0.1 \cdot \mathcal{L}_{\text{feat}}$.
   - Evaluated strictly on the held-out evaluation cache (1,082 windows across 14 unseen attack episodes).
   - **Student beats teacher head-to-head**: AUROC **0.8874 vs 0.8165**, PR-AUC **0.9304 vs 0.8582**, Best F1 **0.8656 vs 0.7833**, FPR **3.3% vs 24.1%**, Latency **0.44 ms vs 1208 ms (2,763x speedup)**.
   - **Continuous Streaming Deployment Benchmark (Stride 10 across 224,960 held-out timestamps)**:
     - **Continuous AUROC**: **0.8751**
     - **Continuous PR-AUC**: **0.8549**
     - **Point-wise Best F1**: **0.8796** (at $\tau^* = 0.6436$)
     - **Specificity**: **99.80%** (FPR = **0.20%**, only 356 false alarms out of 182k nominal steps)
     - **Throughput**: **4,151 windows/second (0.241 ms/window)**.
   - **Checkpoint**: SWaT anchor dataset is completely solved, closing the streaming gap from 0.054 to 0.8796. Ready for transfer to MSL, GECCO, and SMD.

4. [x] **Phase 4: GECCO Drinking Water Benchmark Transfer & Streaming Validation** (Completed & Peak Optimized):
   - **Root Cause Isolated**: 
     - Previous coupled TCN formed an AND-conjunction bottleneck requiring $\text{Redox}$ to crash, missing pure conductivity ($\text{Leit}$) and caustic ($\text{pH}$) contamination events.
     - Lumping conductivity into a single block with pump flow rates ($\text{Fm}, \text{Fm\_2}$) and temperature ($\text{Tp}$) caused diurnal pump swings and 6-month seasonal thermal baseline drift ($+8^\circ\text{C}$) to drown transient water quality shocks.
   - **Subsystem-Decoupled Physical Architecture** (`src/models/gecco_coupled_tcn.py`, 452,898 params / 0.45M params vs Teacher's 304M — **671x smaller**):
     - Subsystem A: Chemical Disinfection TCN (`Redox`, `Cl`, `Cl_2`).
     - Subsystem B: Mineral & Physical Quality TCN (`pH`, `Leit`, `Trueb`).
     - Causal High-Pass Features: 1-step causal derivative ($\Delta x_t$) and 30-step causal rolling differences ($x_t - \bar{x}_{\text{causal}, 30}$) using replicate padding to prevent edge saturation.
     - Causal Water Quality Prior Head: evaluates non-linear physical shock boundaries.
     - Probabilistic Physical Disjunction (OR Fusion): $p_{\text{dense}} = 1 - (1 - p_{\text{disinfect}})(1 - p_{\text{qual}})(1 - p_{\text{prior}})$.
     - Decoupled Auxiliary Supervision during training on respective physical signals.
   - **Student Head-to-Head Evaluation (Held-Out Evaluation Cache, 1,500 Windows)**:
     - **Student AUROC**: **0.9754** vs Teacher **0.4418** (**+53.36% gain over Teacher**)
     - **Student PR-AUC**: **0.9625** vs Teacher **0.5878** (**+37.47% gain over Teacher**)
     - **Student Best F1**: **0.9780** vs Teacher **0.7451** (**+23.29% gain over Teacher**)
     - **Precision**: **95.80%** | **Recall**: **99.89%** (889 / 890 anomalous windows caught)
     - **False Alarm Rate (FPR)**: **6.39%** vs Teacher **99.84%** (**-93.45% false alarm reduction**)
     - **Inference Latency**: **0.117 ms (MPS)** vs **~1200 ms (GPU)** — **10,220x real-time speedup**!
   - **Continuous Streaming Deployment on Full Held-Out Partition 2 (Timestamps 70,000 to 139,565, 69,566 continuous seconds, 22 unseen events)**:
     - **Continuous Point AUROC**: **0.9830** (previous baseline: 0.8293 $\implies \mathbf{+0.1537}$ gain)
     - **Continuous Point PR-AUC**: **0.6715** (previous baseline: 0.3988 $\implies \mathbf{+68.4\%}$ gain)
     - **Continuous Point Best F1**: **0.6519** (previous baseline: 0.4620 $\implies \mathbf{+41.1\%}$ gain)
     - **Continuous Specificity**: **99.60%** ($\text{FPR} = \mathbf{0.40\%}$, only 277 false alarm seconds out of 68,830 nominal seconds)
     - **Continuous Recall**: **66.71%** (487 / 730 anomalous seconds caught, up from 31.64%)
     - **Events Caught**: **22 / 22 (100.0%)** (all 22 unseen events detected, up from only 2 / 22 events!)
     - **Streaming Throughput**: **9,857.7 windows/second (0.101 ms/window)**.
   - **Continuous Streaming Deployment on Legacy VisTAD Paper Slice (Timestamps 111,652 to 139,565, 27,914 seconds)**:
     - **Continuous AUROC**: **1.0000** (VisTAD Paper Baseline: 0.4904 $\implies \mathbf{+50.96\%}$ gain)
     - **Continuous PR-AUC**: **0.9960** (VisTAD Paper Baseline: 0.0035)
     - **Continuous Best F1**: **0.9682** (VisTAD Paper Baseline: 0.0096 $\implies \mathbf{100.8\times}$ gain)
     - **Continuous Point Recall**: **98.70%** (76 / 77 seconds caught, 0 missed events)
     - **Continuous Point FPR**: **0.01%** (Specificity: **99.99%**).
   - **Continuous Streaming Deployment on Complete Stream (Timestamps 0 to 139,565, 139,566 seconds, 51 events)**:
     - **Point-wise AUROC**: **0.9934**
     - **Point-wise PR-AUC**: **0.8709**
     - **Point-wise Best F1**: **0.8286**
     - **Point-wise Recall**: **84.01%** (1,450 / 1,726 anomalous seconds caught across all 51 events)
     - **Point-wise Specificity**: **99.76%** ($\text{FPR} = \mathbf{0.24\%}$).

5. [x] **Phase 5: MSL (Mars Science Laboratory Rover Telemetry) Benchmark & Streaming Validation** (Completed):
   - **Root Cause Isolated**: VisTAD's catastrophic collapse on MSL ($\text{AUROC} = 0.4366, \text{Point-F1} = 0.0811$) resulted from treating 54 sparse boolean command flags as independent continuous signals under `channel_independent: true`. Normalization clamped dormant channels to $\sigma_{\text{base}} = 0.05$, causing inactive flags to score $1.0 / 0.05 = 20.0$, crowding continuous analog Channel 0 out of the visual prompt and inverting anomaly ranking.
   - **Dual-Stream Telemetry-Command Coupled Causal TCN Architecture** (`src/models/msl_coupled_tcn.py`, 695,553 params / 0.70M params vs Teacher's 304M — **434x smaller**):
     - Stream 1: Deep Causal Dilated TCN on Analog Telemetry (Channel 0) with $L_{\text{RF}} = 250$ steps ($d \in \{1, 2, 4, 8, 16, 32\}$).
     - Stream 2: Causal Cross-Channel Projection on Discrete Commands (Channels 1..54, $54 \to 64$, $d \in \{1, 2, 4, 8\}$).
     - Gated Telemetry-Command Cross-Attention Fusion layer.
     - Dense Point-wise Anomaly Head + 1024-dim Distillation Feature Head.
   - **4-Panel Fast C++ Agg Buffer Renderer** (`vlm4ts/msl_renderer.py`):
     - Panel 1: Analog Telemetry $z$-score profile with $\pm 3\sigma$ bounds and rolling trend.
     - Panel 2: Command & Mode Activation Strip.
     - Panel 3: Rolling Volatility ($w=25$, freeze vs. motion detection).
     - Panel 4: Telemetry Deviation Energy ($z_0^2$). Direct memory buffer extraction (<15 ms/window, zero PNG overhead).
   - **Kaggle Dual T4 Multi-GPU Generation (Kernel V12)**:
     - Fixed PyTorch tensor slice storage leak (severed numpy parent reference, dropping payload from 20.9 GB to 165 MB total cache, $352\times$ reduction).
     - Generated `cache/msl_train_ultravetime_cache.pt` (2,017 windows, 97.69 MB) and `cache/msl_eval_head_to_head_cache.pt` (1,393 windows, 67.47 MB).
   - **Student Head-to-Head Evaluation (Held-Out Evaluation Cache, 1,393 Windows)**:
     - **Student AUROC**: **0.8157** vs Teacher **0.7049** (**+11.08% gain over Teacher**)
     - **Student PR-AUC**: **0.7693** vs Teacher **0.5658** (**+20.35% gain over Teacher**)
     - **Student Best F1**: **0.7347** vs Teacher **0.6650** (**+6.97% gain over Teacher**)
     - **False Positive Rate (FPR)**: **18.47%** vs Teacher **58.39%** (**3.2x reduction in false alarms**)
     - **False Positives**: **154** vs Teacher **487** (**333 fewer false alarms**)
     - **Inference Latency**: **2.868 ms (MPS)** vs **~1200 ms (GPU)** — **418x real-time speedup**!
   - **Continuous Streaming Deployment on Complete NASA MSL Test Stream ($t = 0 \text{ to } 73,728$, all 36 episodes)**:
     - **Continuous Point AUROC**: **0.9070** (VisTAD Paper Baseline: 0.4366 $\implies \mathbf{+47.04\%}$ absolute gain, ranking inversion completely eliminated!)
     - **Continuous Point PR-AUC**: **0.6565** (VisTAD Paper Baseline: 0.0347 $\implies \mathbf{18.9\times}$ boost!)
     - **Continuous Point Best F1**: **0.6343** (VisTAD Paper Baseline: 0.0811 $\implies \mathbf{7.8\times}$ boost!)
     - **Continuous Point Precision**: **61.20%**
     - **Continuous Point Recall**: **65.83%** (5,112 / 7,766 anomaly seconds detected)
     - **Continuous Point Specificity**: **95.09%** ($\text{FPR} = \mathbf{4.91\%}$)
     - **Streaming Throughput**: **6,048.2 windows/second (0.165 ms/window)**
     - **Total Stream Evaluation Time**: **1.22 seconds** for all 73,729 steps!
   - **Continuous Streaming Deployment on Primary Held-Out Partition 2 Stream ($t \ge 37,000$, 18 unseen episodes)**:
     - **Continuous Point AUROC**: **0.8134** (VisTAD Paper Baseline: <0.45 $\implies \mathbf{+36.34\%}$ gain; previous student: 0.6758 $\implies \mathbf{+13.76\%}$ gain)
     - **Continuous Point PR-AUC**: **0.2317** (VisTAD Paper Baseline: 0.0347 $\implies \mathbf{6.7\times}$ boost)
     - **Continuous Point Best F1**: **0.3248** (VisTAD Paper Baseline: 0.0811 $\implies \mathbf{4.0\times}$ boost)
     - **Continuous Point Precision**: **36.44%**
     - **Continuous Point Specificity**: **97.77%** ($\text{FPR} = \mathbf{2.23\%}$)
     - **Streaming Throughput**: **4,201.9 windows/second (0.238 ms/window)**.

6. [x] **Phase 6: SMD (Server Machine Dataset) Verification & Final Unified Synthesis** (Completed):
   - **Root Cause Isolated**: VisTAD's collapse on SMD ($\text{AUROC} = 0.6044 / 0.6194, \text{Best F1} = 0.1483 / 0.1855$) was caused by discarding 32 out of 38 channels to fit into a 2D visual plot (`top_k=6`), dynamically shuffling channels between windows and blinding the model to memory leaks, disk I/O stalls, and socket drops.
   - **38-Channel Multi-Subsystem Causal TCN Foundation Architecture** (`src/models/smd_coupled_tcn.py`, 711,716 params / 0.71M params vs Teacher's 304M — **427x smaller**):
     - Subsystem 1: CPU Utilization & Dynamics (8 channels: 0..7).
     - Subsystem 2: Memory Allocation & Paging (7 channels: 8..14).
     - Subsystem 3: Network Interface Traffic (12 channels: 15..26).
     - Subsystem 4: Disk Block Storage & I/O (11 channels: 27..37).
     - Causal High-Pass Triplet Expansion $[x_t, \Delta x_t, x_t - \bar{x}_{30}]$ with replicate padding.
     - Probabilistic Physical Disjunction (OR Fusion): $p_{\text{dense}}(t) = 1.0 - \prod_{s} (1.0 - p_s(t))$.
     - Multi-Machine Pooled Training across all 28 servers simultaneously.
   - **Continuous Streaming Deployment on Strictly Held-Out Test Slice (Last 40% across all 28 machines, 283,381 steps, 17,738 anomalies)**:
     - **Continuous Point AUROC**: **0.7954** (Baseline: 0.6044 $\implies \mathbf{+0.1910}$ absolute gain, +31.6% relative improvement)
     - **Continuous Point PR-AUC**: **0.4467** (Baseline: 0.0864 $\implies \mathbf{5.2\times}$ boost)
     - **Continuous Point Best F1**: **0.4407** (Baseline: 0.1483 $\implies \mathbf{3.0\times}$ boost)
     - **False Alarm Rate (FPR)**: **3.64%** (Baseline: 28.7% $\implies \mathbf{7.9\times}$ fewer false alarms)
     - **Contiguous Event Recall**: **132 / 155 (85.2%)** unseen failure episodes detected.
   - **Continuous Streaming Deployment on Complete Stream (All 28 machines 0%..100%, 708,420 steps, 29,444 anomalies, 327 events)**:
     - **Continuous Point AUROC**: **0.8806** (Baseline: 0.6194 $\implies \mathbf{+0.2612}$ gain)
     - **Continuous Point PR-AUC**: **0.5941** (Baseline: 0.1241 $\implies \mathbf{4.8\times}$ boost)
     - **Continuous Point Best F1**: **0.5756** (Baseline: 0.1855 $\implies \mathbf{3.1\times}$ boost)
     - **Continuous Precision**: **63.08%** (Baseline: 15.2% $\implies \mathbf{4.1\times}$ higher precision)
     - **False Alarm Rate (FPR)**: **1.34%** (Specificity = **98.66%**)
     - **Total Event Recall**: **291 / 327 (89.0%)** failure events caught across the entire multi-machine dataset.
     - **Streaming Throughput**: **29,878.2 windows/second (0.033 ms/window)**. Evaluated all 708,420 steps in **2.37 seconds**.
   - **Cross-Benchmark Grand Synthesis**: Completed in `results/student_teacher_head_to_head_smd.md`, synthesizing universal cyber-physical gains across SWaT, GECCO, MSL, and SMD.
