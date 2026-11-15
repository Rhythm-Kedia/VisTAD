import os
import sys
import time
import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

for p in [CURRENT_DIR, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.models import build_student
from src.dataset import select_top_k_channels
from src.metrics import compute_dynamic_pot_threshold, smooth_binary_predictions, find_events


class VisTADServingEngine:
    """Production-grade edge inference engine for VisTAD foundation and student models."""

    # Explicit, structured model catalog organized by (Target Domain, Paradigm, Distillation, Architecture)
    MODEL_CATALOG = {
        # --- SWaT Domain Models ---
        "swat_zero_shot": {
            "name": "Zero-Shot Foundation (Cross-Domain)",
            "target": "SWaT",
            "paradigm": "Zero-Shot Foundation",
            "distilled": True,
            "arch_name": "Universal-TCN-Pro (Any-Dimension)",
            "file": "universal_smd,gecco,msl_to_swat_best.pt",
            "arch": "universal_foundation_pro",
            "k": None,
            "trained_on": "SMD + GECCO + MSL (Zero SWaT Samples Seen)",
            "distill_type": "DINOv2 ViT-L/14 (1024-dim geometric manifold)",
            "default_threshold": 0.1000,
            "desc": "Evaluates unseen 51-sensor SWaT plant. Model was never trained on any SWaT data."
        },
        "swat_coupled_distilled": {
            "name": "Subsystem-Coupled TCN (Distilled)",
            "target": "SWaT",
            "paradigm": "In-Domain Specialized Student",
            "distilled": True,
            "arch_name": "StageCoupledTCN (6 Physical Stages)",
            "file": "stage_coupled_tcn_best.pt",
            "arch": "stage_coupled_tcn",
            "k": None,
            "input_scaling": "raw",
            "trained_on": "SWaT Water Treatment SCADA (In-Domain)",
            "distill_type": "Coupled Multi-Subsystem DINOv2 Distillation",
            "default_threshold": 0.5000,
            "desc": "6-stage coupled SCADA student modeling physical water flow topology across all 51 sensors."
        },
        "swat_tcn_distilled": {
            "name": "Standard Residual-TCN (Distilled)",
            "target": "SWaT",
            "paradigm": "In-Domain Specialized Student",
            "distilled": True,
            "arch_name": "Residual-TCN 5L (Top-6 Filtered)",
            "file": "swat_swat_attack_tcn_loss_no_resp_swat_best.pt",
            "arch": "tcn",
            "k": 6,
            "layers": 5,
            "trained_on": "SWaT Water Treatment SCADA (In-Domain)",
            "distill_type": "Teacher Feature Distillation Loss (No response loss)",
            "default_threshold": 0.1750,
            "desc": "5-layer dilated residual TCN distilled from DINOv2 features."
        },
        "swat_tcn_pure": {
            "name": "Standard Residual-TCN (Pure Supervised)",
            "target": "SWaT",
            "paradigm": "In-Domain Specialized Student",
            "distilled": False,
            "arch_name": "Residual-TCN 5L (Top-6 Filtered)",
            "file": "swat_swat_attack_tcn_loss_sup_only_swat_best.pt",
            "arch": "tcn",
            "k": 6,
            "layers": 5,
            "trained_on": "SWaT Water Treatment SCADA (In-Domain)",
            "distill_type": "None (Pure Supervised Discrete 0/1 Labels)",
            "default_threshold": 0.2000,
            "desc": "Ablation baseline trained strictly on binary ground truth labels without teacher distillation."
        },

        # --- GECCO Drinking Water Models ---
        "gecco_zero_shot": {
            "name": "Zero-Shot Foundation (Cross-Domain)",
            "target": "GECCO",
            "paradigm": "Zero-Shot Foundation",
            "distilled": True,
            "arch_name": "Universal-TCN-Pro (Any-Dimension)",
            "file": "universal_swat,smd,msl_to_gecco_best.pt",
            "arch": "universal_foundation_pro",
            "k": None,
            "trained_on": "SWaT + SMD + MSL (Zero GECCO Samples Seen)",
            "distill_type": "DINOv2 ViT-L/14 (1024-dim geometric manifold)",
            "default_threshold": 0.3500,
            "desc": "Evaluates unseen 9-sensor drinking water facility without target training."
        },
        "gecco_coupled_distilled": {
            "name": "Subsystem-Coupled TCN (Distilled)",
            "target": "GECCO",
            "paradigm": "In-Domain Specialized Student",
            "distilled": True,
            "arch_name": "GeccoCoupledTCN (Phys/Chem Subsystems)",
            "file": "gecco_coupled_tcn_best.pt",
            "arch": "gecco_coupled_tcn",
            "hidden_dim": 48,
            "k": None,
            "trained_on": "GECCO Drinking Water Quality (In-Domain)",
            "distill_type": "Subsystem DINOv2 Feature Alignment",
            "default_threshold": 0.9155,
            "desc": "Coupled physical-chemical subsystem student across all 9 water quality metrics."
        },
        "gecco_tcn_distilled": {
            "name": "Standard Residual-TCN (Distilled)",
            "target": "GECCO",
            "paradigm": "In-Domain Specialized Student",
            "distilled": True,
            "arch_name": "Residual-TCN 5L (Top-6 Filtered)",
            "file": "gecco_gecco_water_tcn_loss_no_resp_gecco_best.pt",
            "arch": "tcn",
            "k": 6,
            "layers": 5,
            "trained_on": "GECCO Drinking Water Quality (In-Domain)",
            "distill_type": "Teacher Feature Distillation Loss",
            "default_threshold": 0.0620,
            "desc": "5-layer dilated residual TCN distilled from DINOv2."
        },
        "gecco_tcn_pure": {
            "name": "Standard Residual-TCN (Pure Supervised)",
            "target": "GECCO",
            "paradigm": "In-Domain Specialized Student",
            "distilled": False,
            "arch_name": "Residual-TCN 5L (Top-6 Filtered)",
            "file": "gecco_gecco_water_tcn_loss_sup_only_gecco_best.pt",
            "arch": "tcn",
            "k": 6,
            "layers": 5,
            "trained_on": "GECCO Drinking Water Quality (In-Domain)",
            "distill_type": "None (Pure Supervised Discrete 0/1 Labels)",
            "default_threshold": 0.1500,
            "desc": "Ablation baseline trained strictly on binary ground truth labels without teacher distillation."
        },

        # --- SMD Cloud Servers Models ---
        "smd_zero_shot": {
            "name": "Zero-Shot Foundation (Cross-Domain)",
            "target": "SMD",
            "paradigm": "Zero-Shot Foundation",
            "distilled": True,
            "arch_name": "Universal-TCN-Pro (Any-Dimension)",
            "file": "universal_swat,gecco,msl_to_smd_best.pt",
            "arch": "universal_foundation_pro",
            "k": None,
            "trained_on": "SWaT + GECCO + MSL (Zero SMD Samples Seen)",
            "distill_type": "DINOv2 ViT-L/14 (1024-dim geometric manifold)",
            "default_threshold": 0.5000,
            "desc": "Evaluates unseen 38-metric cloud server telemetry without server training."
        },
        "smd_coupled_distilled": {
            "name": "Subsystem-Coupled TCN (Distilled)",
            "target": "SMD",
            "paradigm": "In-Domain Specialized Student",
            "distilled": True,
            "arch_name": "SMDCoupledTCN (4 Resource Subsystems)",
            "file": "smd_coupled_tcn_best.pt",
            "arch": "smd_coupled_tcn",
            "hidden_dim": 32,
            "k": None,
            "trained_on": "SMD Server Telemetry (In-Domain 28 Entities)",
            "distill_type": "Multi-Resource Subsystem Distillation",
            "default_threshold": 0.8917,
            "desc": "4-resource coupled student (CPU, memory, network, disk) across 38 metrics."
        },
        "smd_tcn_distilled": {
            "name": "Standard Residual-TCN (Distilled)",
            "target": "SMD",
            "paradigm": "In-Domain Specialized Student",
            "distilled": True,
            "arch_name": "Residual-TCN 5L (Top-6 Filtered)",
            "file": "smd_pooled28_tcn_smd_pooled28_foundation_best.pt",
            "arch": "tcn",
            "k": 6,
            "layers": 5,
            "trained_on": "SMD Server Telemetry (In-Domain)",
            "distill_type": "Teacher Feature Distillation Loss",
            "default_threshold": 0.2000,
            "desc": "5-layer dilated residual TCN distilled from DINOv2."
        },
        "smd_tcn_pure": {
            "name": "Standard Residual-TCN (Pure Supervised)",
            "target": "SMD",
            "paradigm": "In-Domain Specialized Student",
            "distilled": False,
            "arch_name": "Residual-TCN 5L (Top-6 Filtered)",
            "file": "smd_pooled28_tcn_loss_sup_only_smd_best.pt",
            "arch": "tcn",
            "k": 6,
            "layers": 5,
            "trained_on": "SMD Server Telemetry (In-Domain)",
            "distill_type": "None (Pure Supervised Discrete 0/1 Labels)",
            "default_threshold": 0.2500,
            "desc": "Ablation baseline trained strictly on binary ground truth labels without teacher distillation."
        },

        # --- NASA MSL Mars Rover Models ---
        "msl_zero_shot": {
            "name": "Zero-Shot Foundation (Cross-Domain)",
            "target": "MSL",
            "paradigm": "Zero-Shot Foundation",
            "distilled": True,
            "arch_name": "Universal-TCN-Pro (Any-Dimension)",
            "file": "universal_swat,smd,gecco_to_msl_best.pt",
            "arch": "universal_foundation_pro",
            "k": None,
            "trained_on": "SWaT + SMD + GECCO (Zero MSL Samples Seen)",
            "distill_type": "DINOv2 ViT-L/14 (1024-dim geometric manifold)",
            "default_threshold": 0.2500,
            "desc": "Evaluates unseen 55-channel deep-space spacecraft rover telemetry."
        },
        "msl_coupled_distilled": {
            "name": "Subsystem-Coupled TCN (Distilled)",
            "target": "MSL",
            "paradigm": "In-Domain Specialized Student",
            "distilled": True,
            "arch_name": "MSLCoupledTCN (Analog/Discrete)",
            "file": "msl_coupled_tcn_best.pt",
            "arch": "msl_coupled_tcn",
            "k": None,
            "trained_on": "NASA MSL Mars Rover Telemetry (In-Domain)",
            "distill_type": "Subsystem Analog/Discrete Coupling",
            "default_threshold": 0.8100,
            "desc": "Dual-subsystem student coupling analog sensor dynamics and discrete command channels."
        },
        "msl_tcn_distilled": {
            "name": "Standard Residual-TCN (Distilled)",
            "target": "MSL",
            "paradigm": "In-Domain Specialized Student",
            "distilled": True,
            "arch_name": "Residual-TCN 5L (Top-6 Filtered)",
            "file": "msl_msl_test_tcn_loss_no_resp_msl_best.pt",
            "arch": "tcn",
            "k": 6,
            "layers": 5,
            "trained_on": "NASA MSL Mars Rover Telemetry (In-Domain)",
            "distill_type": "Teacher Feature Distillation Loss",
            "default_threshold": 0.6500,
            "desc": "5-layer dilated residual TCN distilled from DINOv2."
        },
        "msl_tcn_pure": {
            "name": "Standard Residual-TCN (Pure Supervised)",
            "target": "MSL",
            "paradigm": "In-Domain Specialized Student",
            "distilled": False,
            "arch_name": "Residual-TCN 5L (Top-6 Filtered)",
            "file": "msl_msl_test_tcn_loss_sup_only_msl_best.pt",
            "arch": "tcn",
            "k": 6,
            "layers": 5,
            "trained_on": "NASA MSL Mars Rover Telemetry (In-Domain)",
            "distill_type": "None (Pure Supervised Discrete 0/1 Labels)",
            "default_threshold": 0.6500,
            "desc": "Ablation baseline trained strictly on binary ground truth labels without teacher distillation."
        },

        # --- Grand Unified Foundation Model (All 4 Domains) ---
        "grand_unified": {
            "name": "Grand Unified Multi-Domain Foundation",
            "target": "All",
            "paradigm": "Grand Unified Foundation",
            "distilled": True,
            "arch_name": "Universal-TCN-Pro (Any-Dimension)",
            "file": "universal_swat,smd,gecco,msl_to_swat_best.pt",
            "arch": "universal_foundation_pro",
            "k": None,
            "trained_on": "SWaT + SMD + GECCO + MSL (All 4 Domains Jointly)",
            "distill_type": "Universal Multi-Task DINOv2 Distillation",
            "default_threshold": 0.2000,
            "desc": "Pre-trained jointly across all cyber-physical water, aerospace, and IT server domains."
        },

        # --- Legacy Lightweight Baselines ---
        "legacy_cnn_cross": {
            "name": "Legacy 1D-CNN Baseline (Top-6 Filtered)",
            "target": "All",
            "paradigm": "Legacy Baseline",
            "distilled": True,
            "arch_name": "1D-CNN 3L (Top-6 Filtered)",
            "file": "universal_multidomain_cnn3.pt",
            "arch": "cnn_1d",
            "k": 6,
            "layers": 3,
            "trained_on": "Legacy Pooled Telemetry",
            "distill_type": "Channel-Independent 1D-CNN Distillation",
            "default_threshold": 0.2000,
            "desc": "Earlier 3-layer 1D-CNN baseline operating on top-6 variance-filtered channels."
        }
    }

    # Backward compatibility map
    AVAILABLE_MODELS = {k: v for k, v in MODEL_CATALOG.items()}

    def __init__(self, device=None):
        if device is None:
            self.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.current_model = None
        self.current_model_key = None
        self.model_info = None
        self.ckpt_base_m = None
        self.ckpt_base_s = None

    @classmethod
    def find_models(cls, target_domain="SWaT", paradigm=None, distilled=None):
        """Filters model catalog by target domain, evaluation paradigm, and distillation regimen."""
        matches = {}
        target_clean = target_domain.upper()
        for key, info in cls.MODEL_CATALOG.items():
            info_target = info.get("target", "").upper()
            if info_target not in [target_clean, "ALL"]:
                continue
            if paradigm is not None and info.get("paradigm") != paradigm:
                continue
            if distilled is not None and info.get("distilled") != distilled:
                continue
            matches[key] = info
        return matches

    def _resolve_checkpoint_path(self, filename):
        """Authoritatively locates checkpoint in repository root checkpoints/ before fallbacks."""
        # Check explicit root directory first to prevent shadowing by outdated files
        root_ckpt = os.path.normpath(os.path.join(ROOT_DIR, "checkpoints", filename))
        if os.path.exists(root_ckpt):
            return root_ckpt

        search_dirs = [
            os.path.join(ROOT_DIR, "results/results"),
            os.path.join(ROOT_DIR, "results"),
            os.path.join(CURRENT_DIR, "checkpoints")
        ]
        for sdir in search_dirs:
            p = os.path.normpath(os.path.join(sdir, filename))
            if os.path.exists(p):
                return p

        return None

    def load_model(self, model_key):
        """Loads model checkpoint into memory with caching and strict architecture verification."""
        if model_key not in self.MODEL_CATALOG:
            # Check if passed as a display name or legacy key
            found = None
            for k, info in self.MODEL_CATALOG.items():
                if info.get("name") == model_key or info.get("file") == model_key:
                    found = k
                    break
            if found is None:
                raise ValueError(f"Unknown model key '{model_key}'. Choose from: {list(self.MODEL_CATALOG.keys())}")
            model_key = found

        if self.current_model_key == model_key and self.current_model is not None:
            return True

        info = self.MODEL_CATALOG[model_key]
        raw_file = info["file"]

        ckpt_path = self._resolve_checkpoint_path(raw_file)
        if ckpt_path is None:
            raise FileNotFoundError(f"Checkpoint file '{raw_file}' for '{model_key}' was not found in checkpoints/.")

        # Build appropriate model architecture
        arch = info["arch"]
        build_kwargs = {}
        if "hidden_dim" in info:
            build_kwargs["hidden_dim"] = info["hidden_dim"]
        if "layers" in info:
            build_kwargs["num_layers"] = info["layers"]
        if info.get("k") is not None:
            build_kwargs["in_channels"] = info["k"]

        model = build_student(name=arch, **build_kwargs).to(self.device)

        # Load weights safely
        state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state_dict = state["model_state_dict"]
            self.ckpt_base_m = state.get("base_m")
            self.ckpt_base_s = state.get("base_s")
        else:
            state_dict = state
            self.ckpt_base_m = None
            self.ckpt_base_s = None

        # Adapt legacy flat CNN1D backbone keys if loading early flat Sequential weights
        if any(k.startswith("backbone.") and k.split(".")[1].isdigit() for k in state_dict.keys()):
            has_flat_cnn = any(k.startswith("backbone.4.") or k.startswith("backbone.1.running_mean") for k in state_dict.keys())
            if has_flat_cnn:
                remapped = {}
                for k, v in state_dict.items():
                    if k.startswith("backbone."):
                        parts = k.split(".")
                        if parts[1].isdigit():
                            idx = int(parts[1])
                            layer_idx = idx // 4
                            sub_type = idx % 4
                            rest = ".".join(parts[2:])
                            if sub_type == 0:
                                remapped[f"backbone.{layer_idx}.conv.{rest}"] = v
                            elif sub_type == 1:
                                remapped[f"backbone.{layer_idx}.bn.{rest}"] = v
                            else:
                                remapped[k] = v
                        else:
                            remapped[k] = v
                    else:
                        remapped[k] = v
                state_dict = remapped

        model.load_state_dict(state_dict)
        model.eval()

        self.current_model = model
        self.current_model_key = model_key
        self.model_info = info
        return True

    def process_telemetry(self, raw_data, window_size=200, stride=10, 
                          threshold_mode="pot", fixed_threshold=0.25, pot_q=1e-3,
                          apply_smoothing=True, min_duration=3, merge_gap=10,
                          base_mean=None, base_std=None):
        """Runs end-to-end continuous inference and range-aware event localization."""
        if self.current_model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
            
        raw_arr = np.asarray(raw_data, dtype=np.float32)
        if raw_arr.ndim == 1:
            raw_arr = raw_arr.reshape(-1, 1)
        if len(raw_arr) == 0:
            raise ValueError("Input telemetry stream is empty.")
            
        n_steps, n_channels = raw_arr.shape
        arch = self.model_info.get("arch", "")
        k = self.model_info.get("k")
        
        t0 = time.perf_counter()
        
        # 1. Baseline normalization with minimum std floor
        # Priority:
        #   A. Checkpoint native training baseline (exact training distribution match)
        #   B. Explicit reference baseline passed from caller (clean train/normal split)
        #   C. Rolling slice fallback
        if self.ckpt_base_m is not None and self.ckpt_base_s is not None and len(self.ckpt_base_m) == n_channels:
            base_m = np.asarray(self.ckpt_base_m, dtype=np.float32)
            base_s = np.asarray(self.ckpt_base_s, dtype=np.float32)
            base_s = np.where(base_s == 0, 1.0, base_s)
            base_s = np.maximum(base_s, 0.05)
        elif base_mean is not None and base_std is not None:
            base_m = np.asarray(base_mean, dtype=np.float32)
            base_s = np.asarray(base_std, dtype=np.float32)
            base_s = np.where(base_s == 0, 1.0, base_s)
            base_s = np.maximum(base_s, 0.05)
        else:
            bs_len = min(2000, n_steps)
            base_m = np.mean(raw_arr[:bs_len], axis=0)
            base_s = np.maximum(np.std(raw_arr[:bs_len], axis=0), 0.05)
        
        # 2. Adaptive Channel Selection & Input Scaling
        input_scaling = self.model_info.get("input_scaling", "zscore")
        if k is not None and n_channels > k:
            force_0 = bool(self.model_info.get("target") == "MSL")
            selected_channels = select_top_k_channels(
                raw_arr, base_m, base_s, strategy="median_shift", k=k,
                force_include_channel_0=force_0
            )
            selected_channels = [int(c) for c in selected_channels]
            sub_data = raw_arr[:, selected_channels]
            sub_mean = base_m[selected_channels]
            sub_std = base_s[selected_channels]
            norm_data = (sub_data - sub_mean) / sub_std
        elif input_scaling == "raw":
            selected_channels = list(range(n_channels))
            norm_data = raw_arr.copy()
        else:
            # Modern Any-Dimension Models: Ingest all C channels with Z-score normalization
            selected_channels = list(range(n_channels))
            norm_data = (raw_arr - base_m) / base_s
        
        num_used_channels = norm_data.shape[1]

        # 3. Vectorized Sliding-Window Slicing with tail-coverage
        if n_steps < window_size:
            starts = np.array([0])
            norm_data = np.pad(norm_data, ((0, window_size - n_steps), (0, 0)), mode="edge")
        else:
            starts = np.arange(0, n_steps - window_size + 1, stride)
            if starts[-1] + window_size < n_steps:
                starts = np.append(starts, n_steps - window_size)
            
        windows = np.array([norm_data[st:st + window_size].T for st in starts])  # [N, C, L]
        tensor_batches = torch.from_numpy(windows).float()
        
        # 4. Fast Model Inference (Overlap-Averaged Scoring)
        point_scores = np.zeros(n_steps, dtype=np.float32)
        point_counts = np.zeros(n_steps, dtype=np.float32)
        
        has_channel_attr = (arch == "universal_foundation_pro")
        if has_channel_attr:
            channel_point_scores = np.zeros((num_used_channels, n_steps), dtype=np.float32)
        else:
            channel_point_scores = None

        batch_size = 64 if self.device in ["cuda", "mps"] else 32
        with torch.no_grad():
            for b_i in range(0, len(windows), batch_size):
                b_x = tensor_batches[b_i:b_i + batch_size].to(self.device)
                outputs = self.current_model(b_x)
                
                # Extract dense continuous probabilities
                if isinstance(outputs, dict):
                    if "p_system" in outputs:
                        dense_probs = outputs["p_system"].cpu().numpy()
                    elif "pred_dense" in outputs:
                        dense_probs = outputs["pred_dense"].cpu().numpy()
                    elif "logits_dense" in outputs:
                        dense_probs = torch.sigmoid(outputs["logits_dense"]).cpu().numpy()
                    else:
                        first_k = next(iter(outputs.keys()))
                        dense_probs = outputs[first_k].cpu().numpy()
                else:
                    dense_probs = torch.sigmoid(outputs).cpu().numpy()
                
                if dense_probs.ndim == 1:
                    dense_probs = dense_probs.reshape(len(b_x), -1)

                if has_channel_attr and "p_channel" in outputs:
                    chan_probs = outputs["p_channel"].cpu().numpy()  # [B, C, L]
                else:
                    chan_probs = None
                
                b_starts = starts[b_i:b_i + batch_size]
                for i, st in enumerate(b_starts):
                    end_idx = min(st + window_size, n_steps)
                    valid_len = end_idx - st
                    point_scores[st:end_idx] += dense_probs[i][:valid_len]
                    point_counts[st:end_idx] += 1
                    if chan_probs is not None and channel_point_scores is not None:
                        channel_point_scores[:, st:end_idx] += chan_probs[i][:, :valid_len]
                        
        t1 = time.perf_counter()
        
        point_counts = np.maximum(point_counts, 1.0)
        continuous_scores = np.nan_to_num(point_scores / point_counts, nan=0.0)
        continuous_scores = np.clip(continuous_scores, 0.0, 1.0)
        
        if channel_point_scores is not None:
            channel_continuous = np.nan_to_num(channel_point_scores / point_counts[np.newaxis, :], nan=0.0)
            channel_continuous = np.clip(channel_continuous, 0.0, 1.0)
        else:
            channel_continuous = None

        # 5. Threshold Selection & Localization
        if threshold_mode == "pot":
            operating_threshold = compute_dynamic_pot_threshold(continuous_scores, q=pot_q, level=0.98)
        else:
            operating_threshold = float(fixed_threshold)
            
        raw_binary = (continuous_scores >= operating_threshold).astype(int)
        
        if apply_smoothing:
            smoothed_binary = smooth_binary_predictions(
                raw_binary, min_duration=min_duration, merge_gap=merge_gap
            )
        else:
            smoothed_binary = raw_binary
            
        detected_events = find_events(smoothed_binary)
        
        # 6. Extract root cause sensor attributions per detected event
        event_details = []
        for st_idx, ed_idx in detected_events:
            dur = ed_idx - st_idx
            peak_val = float(np.max(continuous_scores[st_idx:ed_idx])) if dur > 0 else float(continuous_scores[st_idx])
            
            top_sensors = []
            if channel_continuous is not None and dur > 0:
                mean_sensor_scores = np.mean(channel_continuous[:, st_idx:ed_idx], axis=1)
                top_indices = np.argsort(mean_sensor_scores)[::-1][:5]
                top_sensors = [(int(ch), float(mean_sensor_scores[ch])) for ch in top_indices]
            elif dur > 0:
                # Fallback: sensor deviation from baseline during incident
                z_dev = np.abs((raw_arr[st_idx:ed_idx] - base_m) / base_s).mean(axis=0)
                top_indices = np.argsort(z_dev)[::-1][:5]
                top_sensors = [(int(ch), float(z_dev[ch])) for ch in top_indices]

            event_details.append({
                "start": int(st_idx),
                "end": int(ed_idx),
                "duration": int(dur),
                "peak_prob": peak_val,
                "top_sensors": top_sensors
            })

        total_time_ms = (t1 - t0) * 1000.0
        ms_per_window = total_time_ms / max(1, len(starts))
        
        return {
            "continuous_scores": continuous_scores,
            "channel_attributions": channel_continuous,
            "threshold": operating_threshold,
            "binary_predictions": smoothed_binary,
            "raw_binary_predictions": raw_binary,
            "detected_events": detected_events,
            "event_details": event_details,
            "selected_channels": selected_channels,
            "inference_time_ms": total_time_ms,
            "ms_per_window": ms_per_window,
            "n_windows": len(starts),
            "n_events": len(detected_events)
        }
