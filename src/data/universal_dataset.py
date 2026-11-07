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
    """Sliding-window dataset for an arbitrary C-channel multivariate time series."""
    def __init__(self, data: np.ndarray, labels: np.ndarray = None, window_size: int = 200, 
                 stride: int = 10, augment_synthetic: bool = False, seed: int = 42):
        self.data = data.astype(np.float32)  # [T, C]
        self.labels = labels.astype(np.float32) if labels is not None else np.zeros(len(data), dtype=np.float32)
        self.window_size = window_size
        self.stride = stride
        self.augment_synthetic = augment_synthetic
        self.synth_gen = SyntheticAnomalyGenerator(p_disrupt=0.5, seed=seed) if augment_synthetic else None
        
        self.indices = list(range(0, len(self.data) - self.window_size + 1, self.stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        st = self.indices[idx]
        ed = st + self.window_size
        
        # [C, L]
        window_x = self.data[st:ed].T.copy()
        window_y_dense = self.labels[st:ed].copy()
        window_y = 1.0 if np.any(window_y_dense > 0) else 0.0

        # If synthetic augmentation is enabled and window is originally nominal, inject disruption
        if self.augment_synthetic and window_y == 0.0:
            window_x, synth_dense, synth_win = self.synth_gen(window_x)
            window_y_dense = np.maximum(window_y_dense, synth_dense)
            window_y = float(max(window_y, synth_win))

        return {
            "st": st,
            "x": torch.from_numpy(window_x).float(),             # [C, L]
            "y_dense": torch.from_numpy(window_y_dense).float(), # [L]
            "y_window": torch.tensor(window_y).float(),          # scalar
            "num_channels": window_x.shape[0]
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
