"""Universal Multi-Dataset Telemetry Ingestion, Online Standardization, and Synthetic Disruption Suite.

Supports arbitrary dimension C in [1, inf) across cyber-physical, server, and space telemetry:
- SWaT (51 channels, industrial water treatment)
- SMD (38 channels, 28 internet compute servers)
- NASA MSL (55 channels, Curiosity Mars rover telemetry)
- GECCO 2018 (9 channels, drinking water quality)
- Generic user-provided CSV/NPY files with unknown channel count
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class SyntheticAnomalyGenerator:
    """Injects universal domain-invariant physical disruptions on nominal telemetry windows.
    
    Teaches neural models the mathematical syntax of cyber-physical failure without
    overfitting to any single dataset's specific ground-truth label distributions.
    """
    def __init__(self, p_disrupt: float = 0.5, seed: int = 42):
        self.p_disrupt = p_disrupt
        self.rng = np.random.default_rng(seed)

    def __call__(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        """
        x: [C, L] normalized telemetry window
        Returns:
            x_aug: [C, L] augmented telemetry window
            y_dense: [L] binary target indicating perturbed timesteps
            y_window: 1 if disrupted, 0 if nominal
        """
        C, L = x.shape
        x_aug = x.copy()
        y_dense = np.zeros(L, dtype=np.float32)
        
        if self.rng.random() > self.p_disrupt:
            return x_aug, y_dense, 0

        # Choose disruption type
        # 0: Velocity Shock, 1: Static Plateau Freeze, 2: Channel Desynchronization, 3: Trend Drift
        disrupt_type = self.rng.integers(0, 4)
        
        # Select subset of channels to disrupt (1 to at most half of channels)
        num_target_channels = self.rng.integers(1, max(2, C // 2 + 1))
        target_channels = self.rng.choice(C, size=num_target_channels, replace=False)
        
        # Select disruption interval [s, e]
        dur = self.rng.integers(5, min(L // 2, 60))
        st = self.rng.integers(10, L - dur - 5)
        ed = st + dur
        
        if disrupt_type == 0:
            # Velocity Shock: sudden high-frequency spike / oscillation
            shock_mag = self.rng.uniform(2.5, 5.0)
            sign = self.rng.choice([-1.0, 1.0])
            for ch in target_channels:
                noise = self.rng.normal(0.0, 0.5, size=dur)
                x_aug[ch, st:ed] += (shock_mag * sign + noise)
            y_dense[st:ed] = 1.0

        elif disrupt_type == 1:
            # Static Plateau Freeze: channel flatlines while system is active
            for ch in target_channels:
                x_aug[ch, st:ed] = x_aug[ch, max(0, st - 1)]
            y_dense[st:ed] = 1.0

        elif disrupt_type == 2:
            # Channel Desynchronization: phase-shift or temporal shuffle on target channels
            shift = self.rng.integers(5, 25)
            for ch in target_channels:
                if st >= shift:
                    x_aug[ch, st:ed] = x[ch, (st - shift):(ed - shift)]
                else:
                    x_aug[ch, st:ed] = x[ch, (st + shift):(ed + shift)]
            y_dense[st:ed] = 1.0

        elif disrupt_type == 3:
            # Trend Drift: unexpected linear ramp or bias injection
            drift_rate = self.rng.uniform(0.05, 0.15) * self.rng.choice([-1.0, 1.0])
            ramp = np.arange(dur) * drift_rate
            for ch in target_channels:
                x_aug[ch, st:ed] += ramp
            y_dense[st:ed] = 1.0

        return x_aug, y_dense, 1


class UniversalWindowDataset(Dataset):
    """Sliding-window dataset for an arbitrary C-channel multivariate time series with causal prefix buffer."""
    def __init__(self, data: np.ndarray, labels: np.ndarray = None, window_size: int = 200, 
                 prefix_len: int = 250, stride: int = 10, augment_synthetic: bool = False, 
                 subsample_channels_range: tuple = None, 
                 fixed_channel_indices: list[int] = None,
                 teacher_cache: dict = None, seed: int = 42):
        self.data = data.astype(np.float32)  # [T, C]
        self.labels = labels.astype(np.float32) if labels is not None else np.zeros(len(data), dtype=np.float32)
        self.window_size = window_size
        self.prefix_len = prefix_len
        self.stride = stride
        self.augment_synthetic = augment_synthetic
        self.subsample_channels_range = subsample_channels_range
        self.fixed_channel_indices = fixed_channel_indices
        self.rng = np.random.default_rng(seed)
        self.synth_gen = SyntheticAnomalyGenerator(p_disrupt=0.5, seed=seed) if augment_synthetic else None
        
        # Ingest and normalize teacher cache if provided
        self.teacher_windows = {}
        if teacher_cache is not None:
            raw_wins = teacher_cache.get("windows", teacher_cache)
            if isinstance(raw_wins, dict):
                for k, v in raw_wins.items():
                    if isinstance(v, dict):
                        st_idx = v.get("start_idx", None)
                        if st_idx is None:
                            if isinstance(k, (int, np.integer)):
                                st_idx = int(k)
                            elif isinstance(k, str) and "_" in k:
                                parts = k.split("_")
                                try:
                                    st_idx = int(parts[-1].lstrip("t"))
                                except ValueError:
                                    pass
                        if st_idx is not None:
                            self.teacher_windows[int(st_idx)] = v

        self.cached_starts = np.array(sorted(self.teacher_windows.keys())) if len(self.teacher_windows) > 0 else np.array([], dtype=int)
        
        # Auto-detect teacher feature dimensionality (e.g. 1024-d Ultra-VETime vs 2048-d early VLM)
        self.teacher_dim = 1024
        if len(self.teacher_windows) > 0:
            first_w = next(iter(self.teacher_windows.values()))
            cand = first_w.get("z_teacher", first_w.get("z_feature", None))
            if cand is not None:
                self.teacher_dim = int(cand.shape[0]) if hasattr(cand, "shape") else len(cand)

        self.indices = list(range(0, len(self.data) - self.window_size + 1, self.stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        st = self.indices[idx]
        ed = st + self.window_size
        
        # 1. Target window: [L_target, C]
        target_x = self.data[st:ed]
        target_y_dense = self.labels[st:ed]
        
        # 2. Historical Receptive Field Prefix: [L_prefix, C]
        if self.prefix_len > 0:
            if st >= self.prefix_len:
                prefix_x = self.data[st - self.prefix_len : st]
            else:
                prefix_x = self.data[0 : st]
                pad_len = self.prefix_len - len(prefix_x)
                if len(prefix_x) > 0:
                    prefix_x = np.pad(prefix_x, ((pad_len, 0), (0, 0)), mode="edge")
                else:
                    prefix_x = np.repeat(target_x[:1], self.prefix_len, axis=0)
            
            full_x = np.concatenate([prefix_x, target_x], axis=0).T  # [C, L_total]
            full_y_dense = np.concatenate([np.zeros(self.prefix_len, dtype=np.float32), target_y_dense], axis=0)
        else:
            full_x = target_x.T  # [C, L_target]
            full_y_dense = target_y_dense

        window_y = 1.0 if np.any(target_y_dense > 0) else 0.0

        # Fixed channel indices (e.g. at test time for sensor-dropping stress testing)
        if self.fixed_channel_indices is not None:
            full_x = full_x[self.fixed_channel_indices, :]

        # Synthetic augmentation (injected into target horizon only)
        if self.augment_synthetic and window_y == 0.0:
            target_region = full_x[:, self.prefix_len:]
            aug_region, synth_dense, synth_win = self.synth_gen(target_region)
            full_x[:, self.prefix_len:] = aug_region
            full_y_dense[self.prefix_len:] = np.maximum(full_y_dense[self.prefix_len:], synth_dense)
            window_y = float(max(window_y, synth_win))

        # Teacher supervision lookup (polymorphic across 1024-d and 2048-d caches)
        has_teacher = 0.0
        z_teacher = torch.zeros(self.teacher_dim, dtype=torch.float32)
        teacher_score = 0.0

        target_st = None
        if st in self.teacher_windows:
            target_st = st
        elif len(self.cached_starts) > 0:
            idx = np.searchsorted(self.cached_starts, st)
            candidates = []
            if idx < len(self.cached_starts):
                candidates.append(self.cached_starts[idx])
            if idx > 0:
                candidates.append(self.cached_starts[idx - 1])
            if len(candidates) > 0:
                best = min(candidates, key=lambda c: abs(c - st))
                if abs(best - st) <= max(15, self.stride):
                    target_st = best

        if target_st is not None and target_st in self.teacher_windows:
            w_info = self.teacher_windows[target_st]
            feat_cand = w_info.get("z_teacher", w_info.get("z_feature", None))
            score_cand = w_info.get("teacher_score", w_info.get("anomaly_likelihood", 0.0))
            if feat_cand is not None:
                if torch.is_tensor(feat_cand):
                    z_teacher = feat_cand.clone().detach().float()
                else:
                    z_teacher = torch.tensor(feat_cand, dtype=torch.float32)
                teacher_score = float(score_cand)
                has_teacher = 1.0

        return {
            "st": st,
            "x": torch.from_numpy(full_x).float(),                   # [C, L_total]
            "y_dense": torch.from_numpy(full_y_dense).float(),       # [L_total]
            "y_window": torch.tensor(window_y).float(),              # scalar
            "prefix_len": self.prefix_len,
            "z_teacher": z_teacher,                                  # [teacher_dim]
            "teacher_score": torch.tensor(teacher_score).float(),    # scalar
            "has_teacher": torch.tensor(has_teacher).float(),        # scalar
            "num_channels": full_x.shape[0]
        }


class UniversalCollate:
    """Batch collator that applies coordinated channel subsampling and permutation across all samples in a batch.
    
    Guarantees all B samples in the batch share the exact same retained and permuted channel indices,
    producing a well-formed [B, C_sub, L] tensor without artificial padding.
    """
    def __init__(self, subsample_range: tuple = None, shuffle_channels: bool = False, seed: int = 42):
        self.subsample_range = subsample_range
        self.shuffle_channels = shuffle_channels
        self.rng = np.random.default_rng(seed)

    def __call__(self, batch: list[dict]) -> dict:
        base_c = batch[0]["x"].shape[0]
        
        if self.subsample_range is not None:
            min_f, max_f = self.subsample_range
            frac = self.rng.uniform(min_f, max_f)
            num_keep = max(3, int(base_c * frac))
            if self.shuffle_channels:
                keep_idx = list(self.rng.permutation(base_c)[:num_keep])
            else:
                keep_idx = sorted(self.rng.choice(base_c, size=num_keep, replace=False))
            x_list = [item["x"][keep_idx, :] for item in batch]
        else:
            if self.shuffle_channels:
                perm_idx = list(self.rng.permutation(base_c))
                x_list = [item["x"][perm_idx, :] for item in batch]
            else:
                x_list = [item["x"] for item in batch]

        return {
            "st": torch.tensor([item["st"] for item in batch], dtype=torch.long),
            "x": torch.stack(x_list, dim=0),
            "y_dense": torch.stack([item["y_dense"] for item in batch], dim=0),
            "y_window": torch.stack([item["y_window"] for item in batch], dim=0),
            "prefix_len": batch[0]["prefix_len"],
            "z_teacher": torch.stack([item["z_teacher"] for item in batch], dim=0),
            "teacher_score": torch.stack([item["teacher_score"] for item in batch], dim=0),
            "has_teacher": torch.stack([item["has_teacher"] for item in batch], dim=0),
            "num_channels": x_list[0].shape[0]
        }


def load_dataset_telemetry(dataset_name: str, data_dir: str = "data") -> tuple[np.ndarray, np.ndarray, int, int]:
    """Loads and standardizes telemetry for any named benchmark dataset without future lookahead.
    
    Returns:
        raw_data: [T, C] continuous standardized telemetry
        labels: [T] binary ground-truth labels
        train_end_idx: integer index separating training/calibration split from test
        num_channels: C
    """
    dataset_name = dataset_name.lower().strip()
    
    def extract_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
        label_col = None
        for cand in ["attack", "EVENT", "label", "Normal/Attack", "anomaly"]:
            if cand in df.columns:
                label_col = cand
                break
        
        feature_cols = [c for c in df.columns if c != label_col and not c.lower().startswith("timestamp") and not c.lower().startswith("time") and not c.startswith("Unnamed")]
        x = df[feature_cols].values.astype(np.float32)
        
        if label_col is not None:
            if df[label_col].dtype == object:
                y = (df[label_col].astype(str).str.strip().str.lower() != "normal").values.astype(np.float32)
            else:
                y = (df[label_col].values > 0).astype(np.float32)
        else:
            y = np.zeros(len(x), dtype=np.float32)
            
        return x, y, feature_cols

    if dataset_name == "swat":
        attack_file = os.path.join(data_dir, "swat", "SWaT_Dataset_Attack_v0.csv")
        normal_file = os.path.join(data_dir, "swat", "SWaT_Dataset_Normal_v1.csv")
        
        df_atk = pd.read_csv(attack_file)
        raw_atk, lbl_atk, feat_cols = extract_xy(df_atk)
        
        if os.path.exists(normal_file):
            df_norm = pd.read_csv(normal_file)
            raw_norm = df_norm[[c for c in feat_cols if c in df_norm.columns]].values.astype(np.float32)
            base_m = np.mean(raw_norm, axis=0)
            base_s = np.maximum(np.std(raw_norm, axis=0), 0.05)
            std_atk = (raw_atk - base_m) / base_s
            return std_atk, lbl_atk, int(len(std_atk) * 0.50), std_atk.shape[1]
        else:
            train_end = int(len(raw_atk) * 0.50)
            base_m = np.mean(raw_atk[:train_end], axis=0)
            base_s = np.maximum(np.std(raw_atk[:train_end], axis=0), 0.05)
            std_atk = (raw_atk - base_m) / base_s
            return std_atk, lbl_atk, train_end, std_atk.shape[1]

    elif dataset_name == "gecco":
        csv_file = os.path.join(data_dir, "gecco", "gecco_water.csv")
        df = pd.read_csv(csv_file)
        raw, lbl, _ = extract_xy(df)
        train_end = int(len(raw) * 0.50)
        base_m = np.mean(raw[:train_end], axis=0)
        base_s = np.maximum(np.std(raw[:train_end], axis=0), 0.05)
        std_data = (raw - base_m) / base_s
        return std_data, lbl, train_end, std_data.shape[1]

    elif dataset_name == "msl":
        test_file = os.path.join(data_dir, "msl", "msl_test.csv")
        train_file = os.path.join(data_dir, "msl", "msl_train.csv")
        df_test = pd.read_csv(test_file)
        raw_test, lbl_test, feat_cols = extract_xy(df_test)
        
        if os.path.exists(train_file):
            df_train = pd.read_csv(train_file)
            raw_train = df_train[[c for c in feat_cols if c in df_train.columns]].values.astype(np.float32)
            base_m = np.mean(raw_train, axis=0)
            base_s = np.maximum(np.std(raw_train, axis=0), 0.05)
        else:
            train_end = int(len(raw_test) * 0.50)
            base_m = np.mean(raw_test[:train_end], axis=0)
            base_s = np.maximum(np.std(raw_test[:train_end], axis=0), 0.05)
            
        std_test = (raw_test - base_m) / base_s
        return std_test, lbl_test, int(len(std_test) * 0.50), std_test.shape[1]

    elif dataset_name == "smd":
        # Load concatenated or primary SMD stream
        test_files = sorted(glob.glob(os.path.join(data_dir, "smd", "test", "*.txt")))
        all_raw = []
        all_lbl = []
        for tf in test_files:
            lf = os.path.join(data_dir, "smd", "test_label", os.path.basename(tf))
            raw = pd.read_csv(tf, header=None).values.astype(np.float32)
            lbl = pd.read_csv(lf, header=None)[0].values.astype(np.float32)
            
            # Normalize each server using strictly its first 50%
            a = int(len(raw) * 0.50)
            base_m = np.mean(raw[:a], axis=0)
            base_s = np.maximum(np.std(raw[:a], axis=0), 0.10)
            all_raw.append((raw - base_m) / base_s)
            all_lbl.append(lbl)
            
        std_concat = np.concatenate(all_raw, axis=0)
        lbl_concat = np.concatenate(all_lbl, axis=0)
        return std_concat, lbl_concat, int(len(std_concat) * 0.50), std_concat.shape[1]

    else:
        # Generic user file
        if os.path.isfile(dataset_name):
            df = pd.read_csv(dataset_name) if dataset_name.endswith(".csv") else pd.DataFrame(np.load(dataset_name))
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            raw = df[numeric_cols].values.astype(np.float32)
            lbl = np.zeros(len(raw), dtype=np.float32)
            train_end = int(len(raw) * 0.50)
            base_m = np.mean(raw[:train_end], axis=0)
            base_s = np.maximum(np.std(raw[:train_end], axis=0), 0.05)
            std_data = (raw - base_m) / base_s
            return std_data, lbl, train_end, std_data.shape[1]
        raise ValueError(f"Unrecognized dataset name or file path: {dataset_name}")


class MultiDatasetBatchSampler:
    """Iterates across multiple datasets with varying channel counts C without padding.
    
    Each yielded batch is drawn from a single dataset, ensuring native [B, C, L] tensors
    without artificial padding or channel masking.
    """
    def __init__(self, dataloaders: dict[str, DataLoader], shuffle: bool = True):
        self.dataloaders = dataloaders
        self.dataset_keys = list(dataloaders.keys())
        self.shuffle = shuffle
        self.iters = {k: iter(dl) for k, dl in dataloaders.items()}
        self.total_batches = sum(len(dl) for dl in dataloaders.values())

    def __iter__(self):
        self.iters = {k: iter(dl) for k, dl in self.dataloaders.items()}
        # Create an interleaved sequence of dataset indices
        schedule = []
        for k in self.dataset_keys:
            schedule.extend([k] * len(self.dataloaders[k]))
        if self.shuffle:
            np.random.shuffle(schedule)
            
        for k in schedule:
            try:
                batch = next(self.iters[k])
            except StopIteration:
                self.iters[k] = iter(self.dataloaders[k])
                batch = next(self.iters[k])
            batch["dataset_name"] = k
            yield batch

    def __len__(self):
        return self.total_batches
