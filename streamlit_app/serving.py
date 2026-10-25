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
    """Production-grade edge inference engine for VisTAD student models."""

    AVAILABLE_MODELS = {
        "Universal Foundation (Zero-Shot Cross-Domain)": {
            "path": "universal_multidomain_cnn3.pt",
            "aliases": ["universal_multidomain_cnn_1d_best.pt"],
            "k": 6,
            "arch": "cnn_1d",
            "layers": 3,
            "domain": "Cross-Domain Telemetry (SMD + SWaT + MSL + GECCO)",
            "default_threshold": 0.20
        },
        "SMD Cloud Computing Expert": {
            "path": "smd_pooled28_tcn_smd_pooled28_foundation_best.pt",
            "aliases": ["smd_pooled28_foundation_tcn5.pt", "smd_pooled28_foundation_cnn3.pt"],
            "k": 6,
            "arch": "tcn",
            "layers": 5,
            "domain": "IT Ops / Server Infrastructure (Residual-TCN 5L Foundation)",
            "default_threshold": 0.20
        },
        "SWaT Water Treatment SCADA Expert": {
            "path": "swat_swat_attack_tcn_loss_no_resp_swat_best.pt",
            "aliases": ["swat_swat_attack_tcn_backbone_tcn5_residual_swat_best.pt", "swat_attack_residual_tcn5.pt"],
            "k": 6,
            "arch": "tcn",
            "layers": 5,
            "domain": "Industrial SCADA Control Systems (Residual-TCN 5L)",
            "default_threshold": 0.1750
        },
        "MSL Mars Rover Telemetry Expert": {
            "path": "msl_msl_test_tcn_loss_no_resp_msl_best.pt",
            "aliases": ["msl_msl_test_tcn_filter_top_k_variance_msl_best.pt", "msl_msl_test_cnn_1d_backbone_cnn3_msl_best.pt"],
            "k": 6,
            "arch": "tcn",
            "layers": 5,
            "domain": "Aerospace Deep-Space Telemetry (Residual-TCN 5L)",
            "default_threshold": 0.6500
        },
        "GECCO Water Quality Expert": {
            "path": "gecco_gecco_water_cnn_1d_gecco_cnn3_best.pt",
            "aliases": ["gecco_water_cnn3.pt", "gecco_gecco_water_cnn_1d_backbone_cnn3_gecco_best.pt"],
            "k": 9,
            "arch": "cnn_1d",
            "layers": 3,
            "domain": "Environmental / Water Quality (CI 1D-CNN 3L)",
            "default_threshold": 0.0620
        }
    }

    ALIASES = {
        "Universal Cross-Domain Foundation (SMD+SWaT+MSL+GECCO)": "Universal Foundation (Zero-Shot Cross-Domain)",
        "SMD Cloud Computing Foundation (28 Entities)": "SMD Cloud Computing Expert"
    }

    def __init__(self, device=None):
        if device is None:
            self.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.current_model = None
        self.current_model_name = None
        self.model_info = None

    def load_model(self, model_name):
        """Loads model checkpoint into memory with caching and robust path resolution."""
        model_name = self.ALIASES.get(model_name, model_name)
        
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_name}. Choose from: {list(self.AVAILABLE_MODELS.keys())}")
        
        if self.current_model_name == model_name and self.current_model is not None:
            return True
            
        info = self.AVAILABLE_MODELS[model_name]
        raw_path = info["path"]
        
        search_dirs = [
            os.path.join(CURRENT_DIR, "checkpoints"),
            os.path.join(ROOT_DIR, "checkpoints")
        ]
        
        candidate_names = [raw_path, os.path.basename(raw_path)] + info.get("aliases", [])
        
        ckpt_path = None
        if os.path.isabs(raw_path) and os.path.exists(raw_path):
            ckpt_path = raw_path
        else:
            for sdir in search_dirs:
                for fname in candidate_names:
                    p = os.path.normpath(os.path.join(sdir, fname))
                    if os.path.exists(p):
                        ckpt_path = p
                        break
                if ckpt_path is not None:
                    break
                
        if ckpt_path is None:
            raise FileNotFoundError(f"Checkpoint not found for '{model_name}'.")
            
        model = build_student(
            name=info["arch"],
            in_channels=info["k"],
            hidden_dim=128,
            num_layers=info["layers"],
            dropout=0.0
        ).to(self.device)
        
        state_dict = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        
        self.current_model = model
        self.current_model_name = model_name
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
        k = self.model_info["k"]
        
        t0 = time.perf_counter()
        
        # 1. Baseline normalization with minimum std floor
        if base_mean is not None and base_std is not None:
            base_m = np.asarray(base_mean, dtype=np.float32)
            base_s = np.maximum(np.asarray(base_std, dtype=np.float32), 0.05)
        else:
            bs_len = min(2000, n_steps)
            base_m = np.mean(raw_arr[:bs_len], axis=0)
            base_s = np.maximum(np.std(raw_arr[:bs_len], axis=0), 0.05)
        
        # 2. Adaptive Channel Selection
        force_0 = bool(self.current_model_name and "msl" in self.current_model_name.lower())
        if n_channels > k:
            selected_channels = select_top_k_channels(
                raw_arr, base_m, base_s, strategy="median_shift", k=k,
                force_include_channel_0=force_0
            )
        else:
            selected_channels = list(range(n_channels))
            
        selected_channels = [int(c) for c in selected_channels]
        sub_data = raw_arr[:, selected_channels]
        sub_mean = base_m[selected_channels]
        sub_std = base_s[selected_channels]
        norm_data = (sub_data - sub_mean) / sub_std
        
        # 3. Vectorized Sliding-Window Slicing with tail-coverage
        if n_steps < window_size:
            starts = np.array([0])
            norm_data = np.pad(norm_data, ((0, window_size - n_steps), (0, 0)), mode="edge")
        else:
            starts = np.arange(0, n_steps - window_size + 1, stride)
            if starts[-1] + window_size < n_steps:
                starts = np.append(starts, n_steps - window_size)
            
        windows = np.array([norm_data[st:st + window_size].T for st in starts])  # [N, K, L]
        tensor_batches = torch.from_numpy(windows).float()
        
        # 4. Fast Model Inference (Overlap-Averaged Scoring)
        point_scores = np.zeros(n_steps, dtype=np.float32)
        point_counts = np.zeros(n_steps, dtype=np.float32)
        
        batch_size = 256 if self.device in ["cuda", "mps"] else 64
        with torch.no_grad():
            for b_i in range(0, len(windows), batch_size):
                b_x = tensor_batches[b_i:b_i + batch_size].to(self.device)
                outputs = self.current_model(b_x)
                dense_logits = outputs["logits_dense"].cpu().numpy()  # [B, L]
                
                b_starts = starts[b_i:b_i + batch_size]
                for i, st in enumerate(b_starts):
                    end_idx = min(st + window_size, n_steps)
                    valid_len = end_idx - st
                    point_scores[st:end_idx] += dense_logits[i][:valid_len]
                    point_counts[st:end_idx] += 1
                        
        t1 = time.perf_counter()
        
        point_counts = np.maximum(point_counts, 1.0)
        continuous_scores = point_scores / point_counts
        
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
        
        total_time_ms = (t1 - t0) * 1000.0
        ms_per_window = total_time_ms / max(1, len(starts))
        
        return {
            "continuous_scores": continuous_scores,
            "threshold": operating_threshold,
            "binary_predictions": smoothed_binary,
            "raw_binary_predictions": raw_binary,
            "detected_events": detected_events,
            "selected_channels": selected_channels,
            "inference_time_ms": total_time_ms,
            "ms_per_window": ms_per_window,
            "n_windows": len(starts),
            "n_events": len(detected_events)
        }
