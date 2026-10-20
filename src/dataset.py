import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def split_series(raw, labels, train_frac=0.5, val_frac=0.1):
    n = len(raw)
    a, b = int(n * train_frac), int(n * (train_frac + val_frac))
    return (raw[:a], labels[:a]), (raw[a:b], labels[a:b]), (raw[b:], labels[b:])

def select_top_k_channels(window_data, base_mean=None, base_std=None, ctx_data=None, strategy="median_shift", k=6, force_include_channel_0=False):
    """Filters channels to Top-K avoiding visual clutter and dead-channel division explosions."""
    num_channels = window_data.shape[1]
    if strategy == "none" or num_channels <= k:
        return list(range(num_channels))
    
    # 1. Primary Strategy: Robust Median-Shift (No zero-division, 50% breakdown point)
    if strategy == "median_shift":
        if ctx_data is not None and len(ctx_data) >= 10:
            reference = np.median(ctx_data, axis=0)
        elif base_mean is not None:
            reference = base_mean
        else:
            reference = np.median(window_data, axis=0)
            
        deltas = np.max(np.abs(window_data - reference), axis=0)
        if base_std is not None:
            deltas = deltas / np.maximum(base_std, 1e-4)
        top_k_idx = list(np.argsort(deltas)[::-1][:k])
        if force_include_channel_0 and 0 not in top_k_idx:
            top_k_idx = [0] + top_k_idx[:k - 1]
        return sorted(list(top_k_idx))

    # 2. Legacy fallback: Top-K Z-Score with robust standard deviation floor
    elif strategy == "top_k_zscore":
        denom = (base_std if base_std is not None else np.std(window_data, axis=0)) + 0.05
        ref = base_mean if base_mean is not None else np.mean(window_data, axis=0)
        z_scores = np.abs((window_data - ref) / denom)
        max_z = np.max(z_scores, axis=0)
        top_k_idx = list(np.argsort(max_z)[::-1][:k])
        if force_include_channel_0 and 0 not in top_k_idx:
            top_k_idx = [0] + top_k_idx[:k - 1]
        return sorted(list(top_k_idx))
        
    elif strategy == "top_k_variance":
        variances = np.var(window_data, axis=0)
        top_k_idx = list(np.argsort(variances)[::-1][:k])
        if force_include_channel_0 and 0 not in top_k_idx:
            top_k_idx = [0] + top_k_idx[:k - 1]
        return sorted(list(top_k_idx))
    
    default_idx = list(range(min(k, num_channels)))
    if force_include_channel_0 and 0 not in default_idx:
        default_idx = [0] + default_idx[:k - 1]
    return sorted(default_idx)


class TimeSeriesDataset(Dataset):
    """Sliding-window dataset for multivariate time series anomaly detection."""
    
    def __init__(self, data, labels, window_size=200, stride=100, 
                 base_mean=None, base_std=None, filter_strategy="none", k=6,
                 teacher_cache=None):
        self.window_size = window_size
        self.stride = stride
        self.filter_strategy = filter_strategy
        self.k = k
        self.teacher_cache = teacher_cache
        
        self.data = np.asarray(data, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.float32) if labels is not None else np.zeros(len(data), dtype=np.float32)
        
        self.base_mean = base_mean if base_mean is not None else np.mean(self.data[:min(3000, len(self.data))], axis=0)
        self.base_std = base_std if base_std is not None else np.std(self.data[:min(3000, len(self.data))], axis=0) + 1e-6
        
        total_len = len(self.data)
        self.indices = [i for i in range(0, total_len - window_size + 1, stride)]
        
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        start = self.indices[idx]
        end = start + self.window_size
        
        window_x = self.data[start:end]
        window_y = self.labels[start:end]
        
        if self.filter_strategy != "none":
            ctx_start = max(0, start - 800)
            ctx_data = self.data[ctx_start:start] if start > 0 else None
            active_dims = select_top_k_channels(window_x, self.base_mean, self.base_std, 
                                                ctx_data=ctx_data,
                                                strategy=self.filter_strategy, k=self.k)
            window_x = window_x[:, active_dims]
            
        gt_ratio = float(np.mean(window_y))
        gt_flag = 1.0 if np.any(window_y >= 0.5) else 0.0
        
        sample = {
            "start_idx": start,
            "x": torch.tensor(window_x.T, dtype=torch.float32),
            "y_dense": torch.tensor(window_y, dtype=torch.float32),
            "y_ratio": torch.tensor(gt_ratio, dtype=torch.float32),
            "y_flag": torch.tensor(gt_flag, dtype=torch.float32)
        }
        
        has_teacher = bool(self.teacher_cache is not None and start in self.teacher_cache)
        sample["has_teacher"] = torch.tensor(1.0 if has_teacher else 0.0, dtype=torch.float32)
        
        if has_teacher:
            t_data = self.teacher_cache[start]
            sample["a_teacher"] = torch.tensor(t_data.get("anomaly_likelihood", 0.0), dtype=torch.float32)
            sample["z_teacher"] = t_data.get("z_feature", torch.zeros(2048, dtype=torch.float32))
            tdm = t_data.get("teacher_dense_mask")
            if isinstance(tdm, torch.Tensor):
                if len(tdm) != self.window_size:
                    tdm = F.interpolate(tdm.unsqueeze(0).unsqueeze(0).float(),
                                        size=self.window_size, mode="nearest").squeeze(0).squeeze(0)
                sample["y_teacher_dense"] = tdm.float()
            else:
                sample["y_teacher_dense"] = torch.zeros(self.window_size, dtype=torch.float32)
        else:
            sample["a_teacher"] = torch.tensor(0.0, dtype=torch.float32)
            sample["z_teacher"] = torch.zeros(2048, dtype=torch.float32)
            sample["y_teacher_dense"] = torch.zeros(self.window_size, dtype=torch.float32)
            
        return sample


def load_dataset_arrays(dataset_name="smd", entity="machine-1-1", data_dir="data"):
    """Loads raw train/test arrays for supported datasets."""
    paths_to_check = [
        os.path.join(data_dir, dataset_name),
        os.path.join("data", dataset_name),
        os.path.join("datasets", dataset_name),
        os.path.join("..", "datasets", dataset_name),
        os.path.join("/kaggle/input", dataset_name),
    ]
    
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if os.path.basename(root).lower() == dataset_name.lower():
                paths_to_check.append(root)
            elif "test" in dirs and "test_label" in dirs:
                paths_to_check.append(root)
                
    target_dir = None
    for p in paths_to_check:
        if os.path.exists(p):
            target_dir = p
            break
            
    if target_dir is None:
        raise FileNotFoundError(
            f"Could not locate dataset '{dataset_name}' in any checked paths: {paths_to_check}. "
            f"On Kaggle, ensure the dataset is attached under /kaggle/input/ or in the project directory."
        )
            
    if dataset_name == "smd":
        test_file = os.path.join(target_dir, "test", f"{entity}.txt")
        label_file = os.path.join(target_dir, "test_label", f"{entity}.txt")
        test_df = pd.read_csv(test_file, header=None)
        test_labels = pd.read_csv(label_file, header=None)[0].values
        return test_df.values, test_labels
    else:
        csv_path = os.path.join(target_dir, f"{entity}.csv")
        if not os.path.exists(csv_path):
            candidates = glob.glob(os.path.join(target_dir, "*.csv"))
            for c in candidates:
                base = os.path.splitext(os.path.basename(c))[0].lower()
                ent_lower = entity.lower().replace("-", "_")
                if base == ent_lower or ent_lower in base or base in ent_lower:
                    csv_path = c
                    break
                ent_tokens = set(ent_lower.split("_"))
                base_tokens = set(base.split("_"))
                if "attack" in ent_tokens and "attack" in base_tokens:
                    csv_path = c
                    break
                if "normal" in ent_tokens and "normal" in base_tokens:
                    csv_path = c
                    break
            if not os.path.exists(csv_path) and len(candidates) == 1:
                csv_path = candidates[0]
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Could not locate CSV file for entity '{entity}' in {target_dir}")
        df = pd.read_csv(csv_path)
        labels = df["attack"].values if "attack" in df.columns else np.zeros(len(df))
        data = df.drop(columns=[c for c in ["attack", "label", "timestamp"] if c in df.columns]).values
        return data, labels
