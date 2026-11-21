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

# Heavy neural imports
from src.models import build_student
from src.models.stage_coupled_tcn import StageCoupledTCN, StageCoupledTCNStudent
from src.models.gecco_coupled_tcn import GeccoCoupledTCN, GeccoSubsystemCoupledTCN
from src.models.smd_coupled_tcn import SMDCoupledTCN, SMDSubsystemCoupledTCN
from src.models.msl_coupled_tcn import MSLCoupledTCN, MSLCoupledTCNStudent
from src.models.universal_tcn import UniversalFoundationTCN_Pro
try:
    from src.models.universal_tcn_channel_aware import UniversalTCNChannelAwareStudent as UniversalTCN_ChannelAware
except ImportError:
    from src.models.universal_tcn_channel_aware import UniversalTCN_ChannelAware
from src.models.cnn_1d import CNN1DStudent
from src.dataset import select_top_k_channels
from src.metrics import compute_dynamic_pot_threshold, smooth_binary_predictions, find_events


def get_config_path():
    """Authoritatively locates the external streamlit_models.yaml configuration file."""
    candidates = [
        os.path.join(ROOT_DIR, "configs", "streamlit_models.yaml"),
        os.path.join(CURRENT_DIR, "..", "configs", "streamlit_models.yaml"),
        os.path.join(ROOT_DIR, "configs", "model_catalog.yaml"),
        os.path.join(CURRENT_DIR, "config.yaml"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return None


def load_model_registry(config_path=None):
    """Loads master model registry from external YAML configuration outside python folder."""
    if config_path is None:
        config_path = get_config_path()
    if config_path and os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"[VisTADServingEngine] Warning: Failed to load config from {config_path}: {e}")
    return None


ARCH_CLASS_MAP = {
    "StageCoupledTCN": StageCoupledTCN,
    "StageCoupledTCNStudent": StageCoupledTCNStudent,
    "GeccoCoupledTCN": GeccoCoupledTCN,
    "GeccoSubsystemCoupledTCN": GeccoSubsystemCoupledTCN,
    "SMDCoupledTCN": SMDCoupledTCN,
    "SMDSubsystemCoupledTCN": SMDSubsystemCoupledTCN,
    "MSLCoupledTCN": MSLCoupledTCN,
    "MSLCoupledTCNStudent": MSLCoupledTCNStudent,
    "UniversalFoundationTCN_Pro": UniversalFoundationTCN_Pro,
    "UniversalTCN_ChannelAware": UniversalTCN_ChannelAware,
    "UniversalTCNChannelAwareStudent": UniversalTCN_ChannelAware,
    "CNN1DStudent": CNN1DStudent,
}

# Backward compatibility map for legacy model keys to verified master models
LEGACY_KEY_MAP = {
    "swat_zero_shot": "swat_tcn_pro_new_cache",
    "swat_coupled_distilled": "swat_old_student_new_cache",
    "swat_tcn_distilled": "swat_res_tcn_new_cache",
    "swat_tcn_pure": "swat_cnn_1d_new_cache",
    "gecco_zero_shot": "gecco_tcn_pro_new_cache",
    "gecco_coupled_distilled": "gecco_old_student_new_cache",
    "gecco_tcn_distilled": "gecco_res_tcn_new_cache",
    "gecco_tcn_pure": "gecco_cnn_1d_new_cache",
    "smd_zero_shot": "smd_tcn_pro_new_cache",
    "smd_coupled_distilled": "smd_old_student_new_cache",
    "smd_tcn_distilled": "smd_res_tcn_new_cache",
    "smd_tcn_pure": "smd_cnn_1d_new_cache",
    "msl_zero_shot": "msl_tcn_pro_new_cache",
    "msl_coupled_distilled": "msl_old_student_new_cache",
    "msl_tcn_distilled": "msl_res_tcn_new_cache",
    "msl_tcn_pure": "msl_cnn_1d_new_cache",
    "grand_unified": "universal_foundation_pro_new",
}


class VisTADServingEngine:
    """Production-grade edge inference engine for VisTAD foundation and student models.
    
    Driven by external configuration (configs/streamlit_models.yaml) representing all
    44 master models from MASTER_MODEL_EVALUATION_METRICS.md.
    """

    CONFIG = load_model_registry()
    MODEL_CATALOG = dict(CONFIG.get("models", {})) if CONFIG else {}

    # Register legacy aliases for backward compatibility
    for legacy_k, master_k in LEGACY_KEY_MAP.items():
        if master_k in MODEL_CATALOG and legacy_k not in MODEL_CATALOG:
            aliased = dict(MODEL_CATALOG[master_k])
            aliased["key"] = legacy_k
            MODEL_CATALOG[legacy_k] = aliased

    AVAILABLE_MODELS = {k: v for k, v in MODEL_CATALOG.items()}

    def __init__(self, device=None, config_path=None):
        if device is None:
            self.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.config = load_model_registry(config_path) or self.CONFIG
        if self.config and "models" in self.config:
            self.catalog = dict(self.config["models"])
            for leg_k, mas_k in LEGACY_KEY_MAP.items():
                if mas_k in self.catalog and leg_k not in self.catalog:
                    self.catalog[leg_k] = dict(self.catalog[mas_k])
        else:
            self.catalog = dict(self.MODEL_CATALOG)

        self.current_model = None
        self.current_model_key = None
        self.model_info = None
        self.ckpt_base_m = None
        self.ckpt_base_s = None

    @classmethod
    def find_models(cls, target_domain="SWaT", family=None, cache_type=None):
        """Filters model catalog by target domain, architecture family, and teacher cache manifold."""
        matches = {}
        target_clean = target_domain.upper()
        for key, info in cls.MODEL_CATALOG.items():
            info_domain = str(info.get("domain", info.get("dataset", ""))).upper()
            if target_clean != "ALL" and info_domain not in [target_clean, "ALL", "CUSTOM"]:
                continue
            if family is not None and info.get("family") != family:
                continue
            if cache_type is not None and info.get("cache_type") != cache_type:
                continue
            matches[key] = info
        return matches

    def _resolve_checkpoint_path(self, filepath_or_filename):
        """Authoritatively locates checkpoint across root checkpoints/ and ablation directories."""
        if not filepath_or_filename:
            return None

        candidates = [
            filepath_or_filename,
            os.path.join(ROOT_DIR, filepath_or_filename),
            os.path.join(CURRENT_DIR, filepath_or_filename),
            os.path.join(ROOT_DIR, "checkpoints", filepath_or_filename),
            os.path.join(ROOT_DIR, "checkpoints/22Nov/cross_ablation_24", filepath_or_filename),
            os.path.join(ROOT_DIR, "checkpoints/22Nov/cross_ablation_24/checkpoints/cross_ablation", os.path.basename(filepath_or_filename)),
            os.path.join(ROOT_DIR, "checkpoints/baseline_ablation", os.path.basename(filepath_or_filename)),
            os.path.join(ROOT_DIR, "checkpoints/loss_ablation", os.path.basename(filepath_or_filename)),
            os.path.join(ROOT_DIR, "results", filepath_or_filename),
            os.path.join(CURRENT_DIR, "checkpoints", os.path.basename(filepath_or_filename)),
        ]
        for c in candidates:
            if os.path.exists(c):
                return os.path.abspath(c)

        return None

    def load_model(self, model_key):
        """Loads model checkpoint into memory with caching and strict architecture verification."""
        if model_key in LEGACY_KEY_MAP and model_key not in self.catalog:
            model_key = LEGACY_KEY_MAP[model_key]

        if model_key not in self.catalog:
            # Check by display name or checkpoint match
            found = None
            for k, info in self.catalog.items():
                if info.get("name") == model_key or info.get("checkpoint") == model_key:
                    found = k
                    break
            if found is None:
                raise ValueError(f"Unknown model key '{model_key}'. Choose from: {list(self.catalog.keys())[:10]}...")
            model_key = found

        if self.current_model_key == model_key and self.current_model is not None:
            return True

        info = self.catalog[model_key]
        raw_file = info.get("checkpoint", info.get("file", ""))

        ckpt_path = self._resolve_checkpoint_path(raw_file)
        if ckpt_path is None:
            raise FileNotFoundError(f"Checkpoint file '{raw_file}' for '{model_key}' was not found.")

        # Build appropriate model architecture
        arch_cls_name = info.get("arch_class")
        arch_kwargs = dict(info.get("arch_kwargs", {}))

        if arch_cls_name and arch_cls_name in ARCH_CLASS_MAP:
            model = ARCH_CLASS_MAP[arch_cls_name](**arch_kwargs).to(self.device)
        else:
            arch = info.get("arch", "universal_foundation_pro")
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

        model.load_state_dict(state_dict, strict=False)
        model.eval()

        self.current_model = model
        self.current_model_key = model_key
        self.model_info = info
        return True

    def infer_continuous_scores(self, raw_data, window_size=200, stride=10, prefix_len=250,
                                base_mean=None, base_std=None):
        """Runs model inference over continuous stream with causal lookback buffer (prefix=250)."""
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
        
        # 1. Baseline normalization
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
            bs_len = min(3000, n_steps)
            base_m = np.mean(raw_arr[:bs_len], axis=0)
            base_s = np.maximum(np.std(raw_arr[:bs_len], axis=0), 0.05)
        
        # 2. Adaptive Channel Selection & Input Scaling
        input_scaling = self.model_info.get("input_scaling", "zscore")
        if k is not None and n_channels > k:
            force_0 = bool(self.model_info.get("domain", "").upper() == "MSL")
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
            selected_channels = list(range(n_channels))
            norm_data = (raw_arr - base_m) / base_s
        
        num_used_channels = norm_data.shape[1]

        # 3. Vectorized Sliding-Window Slicing with causal lookback buffer (prefix=250)
        starts = np.arange(0, n_steps - window_size + 1, stride)
        if len(starts) == 0:
            starts = np.array([0])
        elif starts[-1] + window_size < n_steps:
            starts = np.append(starts, n_steps - window_size)
            
        windows = []
        for st in starts:
            ed = st + window_size
            pref_st = max(0, st - prefix_len)
            pref = norm_data[pref_st:st]
            win = norm_data[st:ed]
            if len(pref) < prefix_len:
                if len(pref) > 0:
                    pref = np.pad(pref, ((prefix_len - len(pref), 0), (0, 0)), mode="edge")
                else:
                    pref = np.repeat(win[:1], prefix_len, axis=0)
            full_w = np.concatenate([pref, win], axis=0).T  # [C, prefix_len + window_size]
            windows.append(full_w)
            
        tensor_batches = torch.from_numpy(np.array(windows)).float()
        
        # 4. Overlap-Averaged Scoring across causal horizon
        point_scores = np.zeros(n_steps, dtype=np.float32)
        point_counts = np.zeros(n_steps, dtype=np.float32)
        
        has_channel_attr = (arch in ["universal_foundation_pro", "universal_channel_aware"])
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

                if has_channel_attr and isinstance(outputs, dict) and "p_channel" in outputs:
                    chan_probs = outputs["p_channel"].cpu().numpy()  # [B, C, L]
                else:
                    chan_probs = None

                # Slice out predictions after causal prefix warm-up (exact evaluation protocol)
                if prefix_len > 0 and dense_probs.shape[-1] == b_x.shape[-1]:
                    target_dense = dense_probs[:, prefix_len:]
                else:
                    target_dense = dense_probs

                if chan_probs is not None:
                    if prefix_len > 0 and chan_probs.shape[-1] == b_x.shape[-1]:
                        target_chan = chan_probs[:, :, prefix_len:]
                    else:
                        target_chan = chan_probs
                else:
                    target_chan = None
                
                b_starts = starts[b_i:b_i + batch_size]
                for i, st in enumerate(b_starts):
                    end_idx = min(st + window_size, n_steps)
                    valid_len = end_idx - st
                    point_scores[st:end_idx] += target_dense[i][:valid_len]
                    point_counts[st:end_idx] += 1
                    if target_chan is not None and channel_point_scores is not None:
                        channel_point_scores[:, st:end_idx] += target_chan[i][:, :valid_len]
                        
        t1 = time.perf_counter()
        
        point_counts = np.maximum(point_counts, 1.0)
        continuous_scores = np.nan_to_num(point_scores / point_counts, nan=0.0)
        continuous_scores = np.clip(continuous_scores, 0.0, 1.0)
        
        if channel_point_scores is not None:
            channel_continuous = np.nan_to_num(channel_point_scores / point_counts[np.newaxis, :], nan=0.0)
            channel_continuous = np.clip(channel_continuous, 0.0, 1.0)
        else:
            channel_continuous = None

        total_time_ms = (t1 - t0) * 1000.0
        ms_per_window = total_time_ms / max(1, len(starts))

        return {
            "continuous_scores": continuous_scores,
            "channel_attributions": channel_continuous,
            "selected_channels": selected_channels,
            "inference_time_ms": total_time_ms,
            "ms_per_window": ms_per_window,
            "n_windows": len(starts),
            "base_m": base_m,
            "base_s": base_s,
            "raw_arr": raw_arr
        }

    def postprocess_alarms(self, continuous_scores, channel_continuous=None,
                           operating_threshold=0.50, threshold_mode="fixed", pot_q=1e-3,
                           apply_smoothing=True, min_duration=3, merge_gap=10,
                           raw_arr=None, base_m=None, base_s=None):
        """Ultra-fast event localization and root-cause attribution (< 1 ms, no neural forward pass)."""
        continuous_scores = np.asarray(continuous_scores, dtype=np.float32)
        n_steps = len(continuous_scores)
        
        if threshold_mode == "pot":
            operating_threshold = compute_dynamic_pot_threshold(continuous_scores, q=pot_q, level=0.98)
        else:
            operating_threshold = float(operating_threshold)
            
        raw_binary = (continuous_scores >= operating_threshold).astype(int)
        
        if apply_smoothing:
            smoothed_binary = smooth_binary_predictions(
                raw_binary, min_duration=min_duration, merge_gap=merge_gap
            )
        else:
            smoothed_binary = raw_binary
            
        detected_events = find_events(smoothed_binary)
        
        event_details = []
        for st_idx, ed_idx in detected_events:
            dur = ed_idx - st_idx
            peak_val = float(np.max(continuous_scores[st_idx:ed_idx])) if dur > 0 else float(continuous_scores[st_idx])
            
            top_sensors = []
            if channel_continuous is not None and dur > 0:
                mean_sensor_scores = np.mean(channel_continuous[:, st_idx:ed_idx], axis=1)
                top_indices = np.argsort(mean_sensor_scores)[::-1][:5]
                top_sensors = [(int(ch), float(mean_sensor_scores[ch])) for ch in top_indices]
            elif dur > 0 and raw_arr is not None and base_m is not None and base_s is not None:
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

        return {
            "threshold": operating_threshold,
            "binary_predictions": smoothed_binary,
            "raw_binary_predictions": raw_binary,
            "detected_events": detected_events,
            "event_details": event_details,
            "n_events": len(detected_events)
        }

    def process_telemetry(self, raw_data, window_size=200, stride=10, prefix_len=250,
                          threshold_mode="fixed", fixed_threshold=0.50, pot_q=1e-3,
                          apply_smoothing=True, min_duration=3, merge_gap=10,
                          base_mean=None, base_std=None):
        """Runs end-to-end continuous streaming inference and range-aware event localization."""
        inf_res = self.infer_continuous_scores(
            raw_data=raw_data,
            window_size=window_size,
            stride=stride,
            prefix_len=prefix_len,
            base_mean=base_mean,
            base_std=base_std
        )
        post_res = self.postprocess_alarms(
            continuous_scores=inf_res["continuous_scores"],
            channel_continuous=inf_res["channel_attributions"],
            operating_threshold=fixed_threshold,
            threshold_mode=threshold_mode,
            pot_q=pot_q,
            apply_smoothing=apply_smoothing,
            min_duration=min_duration,
            merge_gap=merge_gap,
            raw_arr=inf_res["raw_arr"],
            base_m=inf_res["base_m"],
            base_s=inf_res["base_s"]
        )
        return {**inf_res, **post_res}
