"""Unified Multi-Dataset Loss Component and Weight Ablation Runner.

Executes systematic loss ablations across all four cyber-physical benchmarks:
1. SWaT (51 channels, Water Treatment Plant) -> StageCoupledTCNStudent
2. GECCO (9 channels, Drinking Water Distribution) -> GeccoSubsystemCoupledTCN
3. NASA MSL (55 channels, Spacecraft Telemetry & Commands) -> MSLCoupledTCNStudent
4. SMD (38 channels, IT Server Telemetry) -> SMDSubsystemCoupledTCN

Enforces the Fair Evaluation Protocol:
- Strictly Causal Nominal Standardization (base_m, base_s with variance clamp epsilon >= 0.05)
- Stratified Curriculum Train/Val Splits by y_flag
- Dual-Tier Reporting: Window-Level Incident Recall (Tier 1) + Continuous Point-Level Segmentation (Tier 2)
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.models.gecco_coupled_tcn import GeccoSubsystemCoupledTCN
from src.models.stage_coupled_tcn import StageCoupledTCNStudent
from src.models.msl_coupled_tcn import MSLCoupledTCNStudent
from src.models.smd_coupled_tcn import SMDSubsystemCoupledTCN
from src.loss import FocalBCELoss, TemporalDiceLoss


class CachedAblationDataset(Dataset):
    """Loads pre-cached windows directly from RAM with nominal baseline standardization."""
    def __init__(self, windows_dict, keys, base_m=None, base_s=None):
        self.keys = keys
        self.windows = windows_dict
        self.base_m = torch.as_tensor(base_m, dtype=torch.float32) if base_m is not None else None
        self.base_s = torch.as_tensor(base_s + 1e-6, dtype=torch.float32) if base_s is not None else None

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        k = self.keys[idx]
        w = self.windows[k]
        raw_x = torch.as_tensor(w["x"], dtype=torch.float32)
        if self.base_m is not None and self.base_s is not None:
            x = (raw_x - self.base_m) / self.base_s
        else:
            x = raw_x

        y_dense = torch.as_tensor(w["y_dense"], dtype=torch.float32)
        y_flag = float(w.get("y_flag", float(y_dense.max() > 0.5)))
        z_teacher = torch.as_tensor(w.get("z_teacher", np.zeros(1024)), dtype=torch.float32)

        return {
            "start_idx": int(w.get("start_idx", 0)),
            "x": x,
            "y_dense": y_dense,
            "y_flag": torch.tensor(y_flag, dtype=torch.float32),
            "z_teacher": z_teacher
        }


def compute_metrics(y_true, y_pred_prob):
    """Computes comprehensive continuous and thresholded classification metrics."""
    auroc = float(roc_auc_score(y_true, y_pred_prob)) if len(np.unique(y_true)) > 1 else 0.5
    prauc = float(average_precision_score(y_true, y_pred_prob)) if len(np.unique(y_true)) > 1 else 0.0

    p, r, th = precision_recall_curve(y_true, y_pred_prob)
    f1_arr = np.where((p + r) > 0, 2 * p * r / (p + r + 1e-9), 0.0)
    best_idx = int(np.argmax(f1_arr))
    best_f1 = float(f1_arr[best_idx])
    best_th = float(th[best_idx]) if best_idx < len(th) else 0.5

    pred_bin = (y_pred_prob >= best_th).astype(int)
    cm = confusion_matrix(y_true, pred_bin, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = float(fp / max(1, fp + tn))
    precision = float(tp / max(1, tp + fp))
    recall = float(tp / max(1, tp + fn))

    intersection = np.sum((y_pred_prob >= best_th) & (y_true == 1))
    union = np.sum(((y_pred_prob >= best_th) | (y_true == 1)))
    iou = float(intersection / max(1, union))

    return {
        "auroc": auroc,
        "prauc": prauc,
        "best_f1": best_f1,
        "best_threshold": best_th,
        "fpr": fpr,
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
    }


def evaluate(model, loader, device):
    """Evaluates model over DataLoader, returning both window-level and dense point-level metrics."""
    model.eval()
    all_targets, all_preds = [], []
    all_win_scores, all_flags = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y_dense = batch["y_dense"].numpy()
            y_flag = batch["y_flag"].numpy()
            out = model(x)
            pred_dense = out["pred_dense"].cpu().numpy()

            all_targets.append(y_dense.ravel())
            all_preds.append(pred_dense.ravel())

            # For MSL, pred_scalar balances max and top-10 mean; otherwise max-pool
            if "pred_scalar" in out and out["pred_scalar"] is not None:
                win_scores = out["pred_scalar"].cpu().numpy()
            else:
                win_scores = np.max(pred_dense, axis=-1)
            all_win_scores.extend(win_scores)
            all_flags.extend(y_flag)

    y_true_pt = np.concatenate(all_targets)
    y_pred_pt = np.concatenate(all_preds)
    pt_metrics = compute_metrics(y_true_pt, y_pred_pt)

    y_true_win = np.array(all_flags)
    y_scores_win = np.array(all_win_scores)
    win_metrics = compute_metrics(y_true_win, y_scores_win)

    win_metrics["point_auroc"] = pt_metrics["auroc"]
    win_metrics["point_f1"] = pt_metrics["best_f1"]
    win_metrics["point_prauc"] = pt_metrics["prauc"]
    win_metrics["point_fpr"] = pt_metrics["fpr"]
    return win_metrics


def build_student_model(dataset_name, device):
    """Instantiates the specialized physical student for a target domain."""
    d = dataset_name.lower()
    if d == "gecco":
        return GeccoSubsystemCoupledTCN(
            in_channels=9, hidden_dim=48, teacher_feat_dim=1024,
            kernel_size=5, dilations=[1, 2, 4, 8, 16], dropout=0.10
        ).to(device)
    elif d == "swat":
        return StageCoupledTCNStudent(
            stage_dim=32, fusion_dim=128, feat_dim=1024, dropout=0.10
        ).to(device)
    elif d == "msl":
        return MSLCoupledTCNStudent(
            in_channels=55, hidden_dim=64, teacher_feat_dim=1024,
            dropout=0.10
        ).to(device)
    elif d == "smd":
        return SMDSubsystemCoupledTCN(
            in_channels=38, hidden_dim=32, teacher_feat_dim=1024,
            dropout=0.10
        ).to(device)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def load_dataset_caches_and_standardization(dataset_name, cache_dir, data_dir):
    """Loads caches and computes strictly causal nominal standardization parameters."""
    d = dataset_name.lower()
    train_cache_path = os.path.join(cache_dir, f"{d}_train_ultravetime_cache.pt")
    eval_cache_path = os.path.join(cache_dir, f"{d}_eval_head_to_head_cache.pt")

    if not os.path.exists(train_cache_path) or not os.path.exists(eval_cache_path):
        raise FileNotFoundError(f"Missing cache for {dataset_name}: {train_cache_path} or {eval_cache_path}")

    print(f"[{d.upper()}] Loading Train Cache: {train_cache_path}")
    raw_tr = torch.load(train_cache_path, weights_only=False)
    tr_dict = raw_tr["windows"] if isinstance(raw_tr, dict) and "windows" in raw_tr else raw_tr

    print(f"[{d.upper()}] Loading Eval Cache: {eval_cache_path}")
    raw_ev = torch.load(eval_cache_path, weights_only=False)
    ev_dict = raw_ev["windows"] if isinstance(raw_ev, dict) and "windows" in raw_ev else raw_ev

    # Canonicalize cache window shapes to (L=200, C)
    c_map = {"gecco": 9, "swat": 51, "msl": 55, "smd": 38}
    expected_c = c_map.get(d, 9)
    for target_dict in [tr_dict, ev_dict]:
        for k, w in target_dict.items():
            arr = w["x"]
            if torch.is_tensor(arr):
                arr = arr.cpu().numpy()
            if arr.shape[0] == expected_c and arr.shape[1] != expected_c:
                w["x"] = np.transpose(arr, (1, 0))
            else:
                w["x"] = arr

    # Compute baseline mean & std from nominal data
    base_m, base_s = None, None
    if d == "gecco":
        cand = [os.path.join(data_dir, "gecco", "gecco_water.csv"), os.path.join(data_dir, "gecco_water.csv")]
        for c in cand:
            if os.path.exists(c):
                import pandas as pd
                sensors = ['Tp', 'Cl', 'pH', 'Redox', 'Leit', 'Trueb', 'Cl_2', 'Fm', 'Fm_2']
                df = pd.read_csv(c)
                nom_slice = df.iloc[:16000][sensors].values.astype(np.float32)
                base_m = np.mean(nom_slice, axis=0)
                base_s = np.std(nom_slice, axis=0) + 1e-6
                break
    elif d == "swat":
        cand = [os.path.join(data_dir, "swat", "SWaT_Dataset_Normal_v1.csv"), os.path.join(data_dir, "SWaT_Dataset_Normal_v1.csv")]
        for c in cand:
            if os.path.exists(c):
                import pandas as pd
                df = pd.read_csv(c)
                chans = [col for col in df.columns if col not in ["Timestamp", "Normal/Attack", "attack", "label"]]
                nom_slice = df[chans].values.astype(np.float32)
                base_m = np.mean(nom_slice, axis=0)
                base_s = np.std(nom_slice, axis=0) + 1e-6
                break
    elif d == "msl":
        cand = [os.path.join(data_dir, "msl", "msl_train.csv"), os.path.join(data_dir, "msl_train.csv"),
                os.path.join(data_dir, "msl", "MSL_train.npy"), os.path.join(data_dir, "MSL_train.npy")]
        for c in cand:
            if os.path.exists(c):
                if c.endswith(".csv"):
                    import pandas as pd
                    df = pd.read_csv(c)
                    cols = [col for col in df.columns if col not in ["attack", "label", "timestamp"]]
                    nom_slice = df[cols].values.astype(np.float32)
                else:
                    nom_slice = np.load(c).astype(np.float32)
                base_m = np.mean(nom_slice, axis=0)
                base_s = np.std(nom_slice, axis=0)
                base_s[base_s == 0] = 1.0
                break
    elif d == "smd":
        # SMD curriculum cache is already standardized per machine in generate_smd_cache.py
        base_m = None
        base_s = None

    # Fallback to nominal cache windows if raw file is not mounted
    if d != "smd" and (base_m is None or base_s is None):
        print(f"[{d.upper()}] Estimating causal baseline standardization from nominal cache windows...")
        norm_wins = [w["x"] for w in tr_dict.values() if float(w.get("y_flag", 0.0)) == 0.0]
        if len(norm_wins) > 0:
            concat_nom = np.concatenate(norm_wins, axis=0)
            base_m = np.mean(concat_nom, axis=0)
            base_s = np.std(concat_nom, axis=0)
            base_s[base_s == 0] = 1.0
            base_s = np.maximum(base_s, 1e-6)
        else:
            all_wins = np.concatenate([w["x"] for w in tr_dict.values()], axis=0)
            base_m = np.mean(all_wins, axis=0)
            base_s = np.std(all_wins, axis=0)
            base_s[base_s == 0] = 1.0
            base_s = np.maximum(base_s, 1e-6)

    return tr_dict, ev_dict, base_m, base_s


def train_single_ablation_run(
    dataset_name,
    loss_config,
    train_loader,
    val_loader,
    eval_loader,
    device,
    epochs=15,
    lr=1e-3,
    run_name="ablation_run"
):
    """Trains a model under an explicit loss configuration and records epoch dynamics."""
    model = build_student_model(dataset_name, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    focal_loss = FocalBCELoss(gamma=2.0, pos_weight=2.0)
    dice_loss = TemporalDiceLoss()
    bce_loss = nn.BCELoss()

    use_focal = loss_config.get("use_focal", True)
    use_dice = loss_config.get("use_dice", True)
    w_feat = loss_config.get("w_feat", 0.1)

    history = []
    best_val_f1 = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_sup_loss, total_feat_loss = 0.0, 0.0, 0.0
        n_batches = 0

        for batch in train_loader:
            x = batch["x"].to(device)
            y_dense = batch["y_dense"].to(device)
            y_flag = batch["y_flag"].to(device)
            z_teacher = batch["z_teacher"].to(device)

            optimizer.zero_grad()
            out = model(x)
            pred_dense = torch.clamp(out["pred_dense"], 1e-6, 1.0 - 1e-6)

            # 1. Supervised segmentation loss (strictly masked on positive windows)
            if use_focal and use_dice:
                l_sup = focal_loss(pred_dense, y_dense) + dice_loss(pred_dense, y_dense, y_flag=y_flag)
            elif use_focal and not use_dice:
                l_sup = focal_loss(pred_dense, y_dense)
            elif not use_focal and use_dice:
                l_sup = bce_loss(pred_dense, y_dense) + dice_loss(pred_dense, y_dense, y_flag=y_flag)
            else:
                l_sup = bce_loss(pred_dense, y_dense)

            # 2. Feature distillation loss
            if w_feat > 0.0:
                s_feat = out.get("pred_feature", out.get("student_feat", None))
                if s_feat is None:
                    raise KeyError(f"Model output missing feature projection (keys: {list(out.keys())})")
                t_feat = F.normalize(z_teacher, p=2, dim=-1)
                l_feat = (1.0 - torch.sum(s_feat * t_feat, dim=-1)).mean()
            else:
                l_feat = torch.tensor(0.0, device=device)

            loss = l_sup + w_feat * l_feat
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_sup_loss += l_sup.item()
            total_feat_loss += l_feat.item() if w_feat > 0.0 else 0.0
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(1, n_batches)
        avg_sup = total_sup_loss / max(1, n_batches)
        avg_feat = total_feat_loss / max(1, n_batches)

        val_res = evaluate(model, val_loader, device)
        if val_res["best_f1"] > best_val_f1:
            best_val_f1 = val_res["best_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        history.append({
            "epoch": epoch,
            "loss_total": avg_loss,
            "loss_sup": avg_sup,
            "loss_feat": avg_feat,
            "val_auroc": val_res["auroc"],
            "val_f1": val_res["best_f1"],
            "val_prauc": val_res["prauc"],
            "val_fpr": val_res["fpr"]
        })

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    test_res = evaluate(model, eval_loader, device)

    # Save best model checkpoint
    ckpt_dir = os.path.join("checkpoints", "loss_ablation")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{run_name}_best.pt")
    torch.save({
        "run_name": run_name,
        "dataset": dataset_name,
        "model_state_dict": best_state,
        "best_val_f1": best_val_f1,
        "test_metrics": test_res,
        "loss_config": loss_config
    }, ckpt_path)

    return {
        "run_name": run_name,
        "dataset": dataset_name,
        "loss_config": loss_config,
        "history": history,
        "best_val_f1": best_val_f1,
        "test_metrics": test_res
    }


def run_ablation_for_dataset(dataset_name, cache_dir, data_dir, output_dir, device, epochs=15, batch_size=64, lr=1e-3):
    """Executes Experiment 1 (Components) and Experiment 2 (Weights) on a specific dataset."""
    d = dataset_name.lower()
    print("\n" + "#" * 80)
    print(f"STARTING COMPREHENSIVE ABLATION SUITE: [{d.upper()}]")
    print("#" * 80)

    # Dataset-aligned epoch schedule
    dataset_epochs = {
        "msl": 8,
        "smd": 10,
        "gecco": 15,
        "swat": 15
    }
    actual_epochs = dataset_epochs.get(d, epochs)
    print(f"[{d.upper()}] Calibrated Training Horizon: {actual_epochs} epochs")

    tr_dict, ev_dict, base_m, base_s = load_dataset_caches_and_standardization(d, cache_dir, data_dir)
    tr_keys = sorted(list(tr_dict.keys()))
    ev_keys = sorted(list(ev_dict.keys()))

    anom_keys = [k for k in tr_keys if float(tr_dict[k].get("y_flag", 0.0)) > 0.5]
    norm_keys = [k for k in tr_keys if float(tr_dict[k].get("y_flag", 0.0)) <= 0.5]
    val_ratio = 0.10
    n_val_anom = max(1, int(len(anom_keys) * val_ratio))
    n_val_norm = max(1, int(len(norm_keys) * val_ratio))
    val_keys = sorted(anom_keys[-n_val_anom:] + norm_keys[-n_val_norm:])
    sub_tr_keys = sorted(anom_keys[:-n_val_anom] + norm_keys[:-n_val_norm])

    train_ds = CachedAblationDataset(tr_dict, sub_tr_keys, base_m=base_m, base_s=base_s)
    val_ds = CachedAblationDataset(tr_dict, val_keys, base_m=base_m, base_s=base_s)
    eval_ds = CachedAblationDataset(ev_dict, ev_keys, base_m=base_m, base_s=base_s)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)

    print(f"[{d.upper()}] Partitions: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(eval_ds)}")

    # 1. Experiment 1: Component Ablation
    print(f"\n[{d.upper()}] --- Experiment 1: Loss Component Ablation ---")
    component_configs = [
        {"run_name": f"Full_VisTAD_{d.upper()}", "use_focal": True, "use_dice": True, "w_feat": 0.1,
         "description": "Full VisTAD Formulation: Focal Loss + Temporal Dice + 0.1 * L_feat"},
        {"run_name": f"Without_Focal_{d.upper()}", "use_focal": False, "use_dice": True, "w_feat": 0.1,
         "description": "Ablated Focal: Standard BCE + Temporal Dice + 0.1 * L_feat"},
        {"run_name": f"Without_Dice_{d.upper()}", "use_focal": True, "use_dice": False, "w_feat": 0.1,
         "description": "Ablated Dice: Focal Loss + 0.1 * L_feat"},
        {"run_name": f"Without_Feature_{d.upper()}", "use_focal": True, "use_dice": True, "w_feat": 0.0,
         "description": "Ablated Distillation: Focal Loss + Temporal Dice, w_feat=0"},
        {"run_name": f"BCE_Only_{d.upper()}", "use_focal": False, "use_dice": False, "w_feat": 0.0,
         "description": "Naive Point-Wise BCE Only Baseline"}
    ]

    comp_results = []
    for cfg in component_configs:
        print(f"  -> Running: {cfg['run_name']}")
        res = train_single_ablation_run(
            d, cfg, train_loader, val_loader, eval_loader, device,
            epochs=actual_epochs, lr=lr, run_name=cfg["run_name"]
        )
        print(f"     Test Win-AUROC: {res['test_metrics']['auroc']:.4f} | Win-F1: {res['test_metrics']['best_f1']:.4f} | "
              f"Point-AUROC: {res['test_metrics']['point_auroc']:.4f} | Point-F1: {res['test_metrics']['point_f1']:.4f}")
        comp_results.append(res)

    os.makedirs(output_dir, exist_ok=True)
    comp_path = os.path.join(output_dir, f"loss_component_ablation_{d}.json")
    with open(comp_path, "w") as f:
        json.dump(comp_results, f, indent=2)
    # Also save to results/
    with open(f"results/loss_component_ablation_{d}.json", "w") as f:
        json.dump(comp_results, f, indent=2)

    # 2. Experiment 2: Weight Grid Sweep
    print(f"\n[{d.upper()}] --- Experiment 2: Feature Loss Weight Sweep (w_feat) ---")
    weights_to_test = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0, 2.0]
    weight_results = []
    for w in weights_to_test:
        cfg = {
            "run_name": f"Weight_{w:.2f}_{d.upper()}",
            "use_focal": True, "use_dice": True, "w_feat": w,
            "description": f"Feature Distillation Loss Weight w_feat = {w:.2f}"
        }
        res = train_single_ablation_run(
            d, cfg, train_loader, val_loader, eval_loader, device,
            epochs=actual_epochs, lr=lr, run_name=cfg["run_name"]
        )
        print(f"     w={w:.2f} -> Win-AUROC: {res['test_metrics']['auroc']:.4f} | Win-F1: {res['test_metrics']['best_f1']:.4f} | "
              f"Point-AUROC: {res['test_metrics']['point_auroc']:.4f}")
        weight_results.append(res)

    weight_path = os.path.join(output_dir, f"loss_weight_sweep_{d}.json")
    with open(weight_path, "w") as f:
        json.dump(weight_results, f, indent=2)
    # Also save to results/
    with open(f"results/loss_weight_sweep_{d}.json", "w") as f:
        json.dump(weight_results, f, indent=2)

    return comp_results, weight_results


def main():
    parser = argparse.ArgumentParser(description="Multi-Dataset VisTAD Loss & Weight Ablation Suite")
    parser.add_argument("--dataset", type=str, default="all", choices=["all", "gecco", "swat", "msl", "smd"],
                        help="Target dataset or 'all' to run across all 4 benchmarks")
    parser.add_argument("--cache_dir", type=str, default="cache", help="Directory containing pre-computed caches")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing raw datasets")
    parser.add_argument("--output_dir", type=str, default="results/ablations", help="Directory for JSON outputs")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs per ablation run")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Executing VisTAD Multi-Dataset Ablation Suite on Device: {device}")

    target_datasets = ["gecco", "swat", "msl", "smd"] if args.dataset == "all" else [args.dataset]
    master_comp = {}
    master_weight = {}

    for d in target_datasets:
        try:
            c_res, w_res = run_ablation_for_dataset(
                d, args.cache_dir, args.data_dir, args.output_dir, device,
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr
            )
            master_comp[d] = c_res
            master_weight[d] = w_res
        except Exception as e:
            print(f"Error executing ablation on {d}: {e}")

    # Generate Grand Master Markdown Report
    report_path = os.path.join(args.output_dir, "LOSS_AND_WEIGHT_ABLATION_MASTER_REPORT.md")
    with open(report_path, "w") as f:
        f.write("# VisTAD: Multi-Dataset Loss Component & Distillation Weight Ablation Study\n\n")
        f.write("**Execution Environment:** " + device.upper() + "\n\n")

        for d, results in master_comp.items():
            f.write(f"## 1. Loss Component Ablation Matrix: {d.upper()}\n\n")
            f.write("| Variant / Configuration | Window AUROC | Window Best F1 | Point AUROC | Point Best F1 | False Alarm Rate (FPR) |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
            for r in results:
                m = r["test_metrics"]
                f.write(f"| **{r['run_name']}** | **{m['auroc']:.4f}** | **{m['best_f1']:.4f}** | "
                        f"{m.get('point_auroc', 0.0):.4f} | {m.get('point_f1', 0.0):.4f} | {m['fpr']*100:.2f}% |\n")
            f.write("\n")

        for d, results in master_weight.items():
            f.write(f"## 2. Feature Loss Weight Grid Sweep (w_feat): {d.upper()}\n\n")
            f.write("| Distillation Weight ($w_{\\text{feat}}$) | Window AUROC | Window Best F1 | Point AUROC | Effective Force Ratio |\n")
            f.write("| :---: | :---: | :---: | :---: | :---: |\n")
            for r in results:
                m = r["test_metrics"]
                w = r["loss_config"]["w_feat"]
                f.write(f"| $w = {w:.2f}$ | **{m['auroc']:.4f}** | **{m['best_f1']:.4f}** | "
                        f"{m.get('point_auroc', 0.0):.4f} | {w*10.0:.1f} : 1.0 |\n")
            f.write("\n")

    print(f"\n[Saved] Master Report written to: {report_path}")


if __name__ == "__main__":
    main()
