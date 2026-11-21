"""Side-by-side continuous streaming evaluation of Channel-Aware Universal TCN Pro
vs. Hand-Crafted Specialized Coupled TCN Students on Partition 2 (Real Test Sets).

Evaluates:
- GECCO Partition 2 (69,566 steps, 22 unseen events)
- MSL Partition 2 (36,729 steps, 18 unseen episodes)
- SWaT Partition 2 (224,959 steps, 14 unseen attacks)
- SMD Partition 2 (283,381 steps, 28 machines, 155 unseen failure events)

Outputs:
- results/channel_aware_vs_specialized_partition2.json
"""

import os
import sys
import time
import glob
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.stage_coupled_tcn import StageCoupledTCNStudent
from src.models.gecco_coupled_tcn import GeccoSubsystemCoupledTCN
from src.models.msl_coupled_tcn import MSLCoupledTCNStudent
from src.models.smd_coupled_tcn import SMDSubsystemCoupledTCN
from src.models.universal_tcn_channel_aware import UniversalTCNChannelAwareStudent


class ContinuousStreamDataset(Dataset):
    def __init__(self, data, window_size=200, stride=10):
        self.data = data
        self.window_size = window_size
        self.stride = stride
        self.indices = list(range(0, len(data) - window_size + 1, stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        st = self.indices[idx]
        ed = st + self.window_size
        return {
            "st": st,
            "x": torch.from_numpy(self.data[st:ed]).float().transpose(0, 1)
        }


def compute_point_adjusted_metrics(y_true, y_pred_binary):
    y_adj = y_pred_binary.copy()
    events = []
    in_event = False
    st = 0
    for i, val in enumerate(y_true):
        if val == 1 and not in_event:
            in_event = True
            st = i
        elif val == 0 and in_event:
            in_event = False
            events.append((st, i))
    if in_event:
        events.append((st, len(y_true)))

    for st_e, ed_e in events:
        if np.any(y_pred_binary[st_e:ed_e] == 1):
            y_adj[st_e:ed_e] = 1

    cm = confusion_matrix(y_true, y_adj)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    prec = float(tp / max(tp + fp, 1))
    rec = float(tp / max(tp + fn, 1))
    pa_f1 = float(2 * prec * rec / max(prec + rec, 1e-8))
    return pa_f1


def run_streaming_eval_entities(model, entities_list, window_size=200, stride=10, batch_size=128, device="cpu"):
    model.eval()
    all_y = []
    all_scores = []
    total_windows = 0
    t0 = time.time()

    with torch.no_grad():
        for stream_data, stream_labels in entities_list:
            n_points = len(stream_data)
            ds = ContinuousStreamDataset(stream_data, window_size=window_size, stride=stride)
            total_windows += len(ds)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

            point_scores = np.zeros(n_points, dtype=np.float32)
            point_counts = np.zeros(n_points, dtype=np.float32)

            for batch in loader:
                starts = batch["st"].numpy()
                x = batch["x"].to(device)
                out = model(x)
                if "pred_dense" in out:
                    pred_dense = out["pred_dense"].cpu().numpy()
                elif "logits_dense" in out:
                    pred_dense = out["logits_dense"].cpu().numpy()
                elif "dense_probs" in out:
                    pred_dense = out["dense_probs"].cpu().numpy()
                else:
                    first_k = list(out.keys())[0]
                    pred_dense = out[first_k].cpu().numpy()

                for i, st in enumerate(starts):
                    point_scores[st : st + window_size] += pred_dense[i]
                    point_counts[st : st + window_size] += 1.0

            valid_mask = point_counts > 0
            final_probs = np.zeros(n_points, dtype=np.float32)
            final_probs[valid_mask] = point_scores[valid_mask] / point_counts[valid_mask]

            all_y.append(stream_labels[valid_mask])
            all_scores.append(final_probs[valid_mask])

    duration = time.time() - t0
    fps = total_windows / max(duration, 1e-6)
    latency_ms = (duration / max(total_windows, 1)) * 1000.0

    y_eval = np.concatenate(all_y, axis=0)
    s_eval = np.concatenate(all_scores, axis=0)

    auroc = float(roc_auc_score(y_eval, s_eval)) if len(np.unique(y_eval)) > 1 else 0.5
    pr_auc = float(average_precision_score(y_eval, s_eval)) if len(np.unique(y_eval)) > 1 else 0.0

    precisions, recalls, thresholds = precision_recall_curve(y_eval, s_eval)
    f1_scores = 2 * (precisions * recalls) / np.maximum(precisions + recalls, 1e-8)
    best_idx = np.argmax(f1_scores)
    best_f1 = float(f1_scores[best_idx])
    best_thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

    y_pred_bin = (s_eval >= best_thresh).astype(int)
    cm = confusion_matrix(y_eval, y_pred_bin)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    fpr = float(fp / max(fp + tn, 1))
    specificity = float(tn / max(tn + fp, 1))

    # Point-Adjusted F1
    pa_f1 = compute_point_adjusted_metrics(y_eval, y_pred_bin)

    # Event audit
    tot_events = 0
    caught_events = 0
    for (_, stream_labels), s_arr in zip(entities_list, all_scores):
        diff = np.diff(np.pad(stream_labels[:len(s_arr)], (1, 1), mode="constant"))
        starts_ev = np.where(diff == 1)[0]
        ends_ev = np.where(diff == -1)[0]
        for s_e, e_e in zip(starts_ev, ends_ev):
            tot_events += 1
            if np.any(s_arr[s_e:e_e] >= best_thresh):
                caught_events += 1

    event_recall = f"{caught_events}/{tot_events}" if tot_events > 0 else "N/A"
    event_pct = (caught_events / max(tot_events, 1)) * 100.0

    return {
        "auroc": auroc,
        "pr_auc": pr_auc,
        "best_f1": best_f1,
        "pa_f1": pa_f1,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "specificity": specificity,
        "event_recall": event_recall,
        "event_pct": event_pct,
        "best_thresh": best_thresh,
        "fps": fps,
        "latency_ms": latency_ms
    }


def load_model_by_spec(ckpt_path, ds_name, model_type, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    cfg = ckpt.get("model_config", {})

    c_map = {"gecco": 9, "msl": 55, "swat": 51, "smd": 38}
    channels = c_map.get(ds_name.lower())

    if model_type == "channel_aware":
        model = UniversalTCNChannelAwareStudent(
            in_channels=cfg.get("in_channels", channels),
            hidden_dim=cfg.get("hidden_dim", 64),
            num_heads=cfg.get("num_heads", 4),
            num_layers=cfg.get("num_layers", 2),
            dropout=0.05
        ).to(device)
    elif model_type == "specialized":
        if ds_name == "gecco":
            model = GeccoSubsystemCoupledTCN(in_channels=9, hidden_dim=cfg.get("hidden_dim", 48)).to(device)
        elif ds_name == "msl":
            model = MSLCoupledTCNStudent(in_channels=55, hidden_dim=cfg.get("hidden_dim", 64)).to(device)
        elif ds_name == "swat":
            model = StageCoupledTCNStudent(stage_dim=32, fusion_dim=128, feat_dim=1024, dropout=0.10).to(device)
        elif ds_name == "smd":
            model = SMDSubsystemCoupledTCN(in_channels=38, hidden_dim=32, teacher_feat_dim=1024, dropout=0.10).to(device)
        else:
            raise ValueError(f"Unknown dataset {ds_name}")
    else:
        raise ValueError(f"Unknown model_type {model_type}")

    model.load_state_dict(state, strict=False)
    return model


def main():
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. GECCO
    df_g = pd.read_csv("data/gecco/gecco_water.csv")
    g_chans = [c for c in df_g.columns if c not in ["Time", "attack", "label", "timestamp"]]
    g_raw = df_g[g_chans].values.astype(np.float32)
    g_labels = df_g["attack"].values.astype(np.float32)
    g_nom = g_raw[:60000]
    g_m = np.mean(g_nom, axis=0)
    g_s = np.std(g_nom, axis=0) + 1e-6
    g_norm = (g_raw - g_m) / g_s
    gecco_p2_data = g_norm[70000:]
    gecco_p2_labels = g_labels[70000:]

    # 2. MSL
    df_m = pd.read_csv("data/msl/msl_test.csv")
    m_chans = [c for c in df_m.columns if c not in ["Time", "attack", "label", "timestamp"]]
    m_raw = df_m[m_chans].values.astype(np.float32)
    m_labels = df_m["attack"].values.astype(np.float32)
    df_m_tr = pd.read_csv("data/msl/msl_train.csv")
    m_tr_raw = df_m_tr[m_chans].values.astype(np.float32)
    m_m = np.mean(m_tr_raw, axis=0)
    m_s = np.std(m_tr_raw, axis=0)
    m_s[m_s == 0] = 1.0
    m_norm = (m_raw - m_m) / np.maximum(m_s, 1e-6)
    msl_p2_data = m_norm[37000:]
    msl_p2_labels = m_labels[37000:]

    # 3. SWaT
    df_s_norm = pd.read_csv("data/swat/SWaT_Dataset_Normal_v1.csv")
    s_chans = [c for c in df_s_norm.columns if c not in ["Timestamp", "Normal/Attack", "attack", "label"]]
    s_norm_raw = df_s_norm[s_chans].values.astype(np.float32)
    s_m = np.mean(s_norm_raw, axis=0)
    s_s = np.std(s_norm_raw, axis=0) + 1e-6
    df_s_att = pd.read_csv("data/swat/SWaT_Dataset_Attack_v0.csv")
    s_att_raw = df_s_att[s_chans].values.astype(np.float32)
    raw_s_labels = df_s_att["attack"].values if "attack" in df_s_att.columns else df_s_att["Normal/Attack"].values
    s_labels = (raw_s_labels != "Normal").astype(np.float32) if raw_s_labels.dtype == object else (raw_s_labels == 1).astype(np.float32)
    s_norm = (s_att_raw - s_m) / s_s
    swat_p2_data = s_norm[224960:]
    swat_p2_labels = s_labels[224960:]

    # 4. SMD (28 entities)
    smd_files = sorted(glob.glob("data/smd/test/*.txt"))
    smd_entities = []
    for tf in smd_files:
        lf = os.path.join("data/smd/test_label", os.path.basename(tf))
        raw_m = pd.read_csv(tf, header=None).values.astype(np.float32)
        raw_l = pd.read_csv(lf, header=None)[0].values.astype(np.float32)
        n = len(raw_m)
        a = int(n * 0.50)
        b = int(n * 0.60)
        base_m = np.mean(raw_m[:a], axis=0)
        base_s = np.maximum(np.std(raw_m[:a], axis=0), 0.10)
        norm_m = (raw_m - base_m) / base_s
        p2_d = norm_m[b:]
        p2_l = raw_l[b:]
        smd_entities.append((p2_d, p2_l))

    benchmarks = [
        {
            "dataset": "GECCO",
            "name": "gecco",
            "entities": [(gecco_p2_data, gecco_p2_labels)],
            "specialized_ckpt": "checkpoints/gecco_coupled_tcn_best.pt",
            "channel_aware_ckpt": "checkpoints/21Nov/gecco_universal_pro_channel_aware_best.pt"
        },
        {
            "dataset": "SWaT",
            "name": "swat",
            "entities": [(swat_p2_data, swat_p2_labels)],
            "specialized_ckpt": "checkpoints/stage_coupled_tcn_best.pt",
            "channel_aware_ckpt": "checkpoints/21Nov/swat_universal_pro_channel_aware_best.pt"
        },
        {
            "dataset": "SMD",
            "name": "smd",
            "entities": smd_entities,
            "specialized_ckpt": "checkpoints/smd_coupled_tcn_best.pt",
            "channel_aware_ckpt": "checkpoints/21Nov/smd_universal_pro_channel_aware_best.pt"
        },
        {
            "dataset": "MSL",
            "name": "msl",
            "entities": [(msl_p2_data, msl_p2_labels)],
            "specialized_ckpt": "checkpoints/21Nov/msl_coupled_tcn_best.pt",
            "channel_aware_ckpt": "checkpoints/21Nov/msl_universal_pro_channel_aware_best.pt"
        }
    ]

    results = {}
    print("\n" + "=" * 90)
    print("RUNNING PARTITION 2 CONTINUOUS STREAMING HEAD-TO-HEAD EVALUATION")
    print("=" * 90)

    for bm in benchmarks:
        ds = bm["dataset"]
        name = bm["name"]
        print(f"\n>>> Benchmarking {ds} Partition 2...")

        # 1. Specialized Student
        print(f"  [1/2] Loading Specialized Student: {bm['specialized_ckpt']}")
        model_spec = load_model_by_spec(bm["specialized_ckpt"], name, "specialized", device)
        m_spec = run_streaming_eval_entities(model_spec, bm["entities"], device=device)

        # 2. Channel-Aware Universal TCN Pro Student
        print(f"  [2/2] Loading Channel-Aware Universal TCN Pro: {bm['channel_aware_ckpt']}")
        model_ca = load_model_by_spec(bm["channel_aware_ckpt"], name, "channel_aware", device)
        m_ca = run_streaming_eval_entities(model_ca, bm["entities"], device=device)

        results[name] = {
            "dataset": ds,
            "specialized": m_spec,
            "channel_aware": m_ca
        }

        print(f"  {ds} Specialized:   AUROC={m_spec['auroc']:.4f} | PR-AUC={m_spec['pr_auc']:.4f} | Point-F1={m_spec['best_f1']:.4f} | PA-F1={m_spec['pa_f1']:.4f} | FPR={m_spec['fpr']*100:.2f}% | Events={m_spec['event_recall']}")
        print(f"  {ds} Channel-Aware: AUROC={m_ca['auroc']:.4f} | PR-AUC={m_ca['pr_auc']:.4f} | Point-F1={m_ca['best_f1']:.4f} | PA-F1={m_ca['pa_f1']:.4f} | FPR={m_ca['fpr']*100:.2f}% | Events={m_ca['event_recall']}")

    # Save JSON
    out_json = "results/channel_aware_vs_specialized_partition2.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll continuous streaming metrics saved to {out_json}")


if __name__ == "__main__":
    main()
