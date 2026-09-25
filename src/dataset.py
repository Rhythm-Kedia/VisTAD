import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


def select_top_k_channels(window_data, base_mean, base_std, strategy="top_k_zscore", k=6):
    """Filters channels to Top-K to avoid visual spaghetti / dimensional bloat."""
    num_channels = window_data.shape[1]
    if strategy == "none" or num_channels <= k:
        return list(range(num_channels))
    
    if strategy == "top_k_zscore":
        z_scores = np.abs((window_data - base_mean) / base_std)
        max_z = np.max(z_scores, axis=0)
        top_k_idx = np.argsort(max_z)[::-1][:k]
        return sorted(list(top_k_idx))
        
    elif strategy == "top_k_variance":
        variances = np.var(window_data, axis=0)
        top_k_idx = np.argsort(variances)[::-1][:k]
        return sorted(list(top_k_idx))
    
    return list(range(min(k, num_channels)))


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
            active_dims = select_top_k_channels(window_x, self.base_mean, self.base_std, 
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
            sample["y_teacher_dense"] = t_data.get("teacher_dense_mask", torch.zeros(self.window_size, dtype=torch.float32))
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
        df = pd.read_csv(csv_path)
        labels = df["attack"].values if "attack" in df.columns else np.zeros(len(df))
        data = df.drop(columns=[c for c in ["attack", "label", "timestamp"] if c in df.columns]).values
        return data, labels
