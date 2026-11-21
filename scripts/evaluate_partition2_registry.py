"""Evaluates all checkpoints on the complete test partition (Partition 2) across all benchmarks.

Evaluates:
- GECCO Partition 2: t in [70,000, 139,565] (69,566 steps, 22 unseen events, 730 anomaly steps)
- MSL Partition 2: t in [37,000, 73,728] (36,729 steps, 18 unseen episodes, 1,536 anomaly steps)
- SWaT Partition 2: t in [224,960, 449,919] (224,959 steps, 14 unseen attacks, 42,864 anomaly steps)
- SMD Partition 2: Chronological last 40% across all 28 servers (283,381 steps, 17,738 anomaly steps)

Outputs:
results/PARTITION_2_TEST_EVALUATION_REGISTRY.md
Notes/21Nov/PARTITION_2_TEST_EVALUATION_REGISTRY.md
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
from src.models.cnn_1d import CNN1DStudent


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
        # Shape: [Channels, L]
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
    """Evaluates continuous streaming over one or more independent entities (e.g., 28 SMD servers)."""
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

    # Point-Adjusted F1
    pa_f1 = compute_point_adjusted_metrics(y_eval, y_pred_bin)

    # Event audit across entities
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

    return {
        "auroc": auroc,
        "pr_auc": pr_auc,
        "best_f1": best_f1,
        "pa_f1": pa_f1,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "event_recall": event_recall,
        "best_thresh": best_thresh,
        "fps": fps,
        "latency_ms": latency_ms
    }


def load_model(ckpt_path, ds_name, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    bname = os.path.basename(ckpt_path).lower()

    if "coupled" in bname or "stage" in bname or "vistad" in bname or "weight" in bname or "without" in bname or "bce_only" in bname:
        if ds_name == "gecco":
            model = GeccoSubsystemCoupledTCN(in_channels=9, hidden_dim=48).to(device)
        elif ds_name == "msl":
            model = MSLCoupledTCNStudent(in_channels=55, hidden_dim=64).to(device)
        elif ds_name == "swat":
            model = StageCoupledTCNStudent(stage_dim=32, fusion_dim=128, feat_dim=1024, dropout=0.10).to(device)
        elif ds_name == "smd":
            model = SMDSubsystemCoupledTCN(in_channels=38, hidden_dim=32, teacher_feat_dim=1024, dropout=0.10).to(device)
        else:
            raise ValueError(f"Unknown dataset {ds_name}")
    elif "cnn_1d" in bname or "cnn3" in bname:
        c_map = {"gecco": 9, "msl": 55, "swat": 51, "smd": 38}
        model = CNN1DStudent(in_channels=c_map.get(ds_name, 9), num_layers=3, use_residual=False).to(device)
    elif "tcn" in bname or "filter" in bname or "residual" in bname:
        c_map = {"gecco": 9, "msl": 55, "swat": 51, "smd": 38}
        model = CNN1DStudent(in_channels=c_map.get(ds_name, 9), num_layers=5, use_residual=True).to(device)
    else:
        if ds_name == "gecco":
            model = GeccoSubsystemCoupledTCN(in_channels=9, hidden_dim=48).to(device)
        elif ds_name == "msl":
            model = MSLCoupledTCNStudent(in_channels=55, hidden_dim=64).to(device)
        elif ds_name == "swat":
            model = StageCoupledTCNStudent(stage_dim=32, fusion_dim=128, feat_dim=1024, dropout=0.10).to(device)
        elif ds_name == "smd":
            model = SMDSubsystemCoupledTCN(in_channels=38, hidden_dim=32, teacher_feat_dim=1024, dropout=0.10).to(device)
        else:
            raise ValueError(f"Unknown dataset {ds_name}")

    try:
        model.load_state_dict(state)
    except Exception:
        model.load_state_dict(state, strict=False)
    return model


def format_cell(num_val, max_val, str_rep):
    if num_val is None or max_val is None:
        return str_rep
    if abs(num_val - max_val) < 1e-6 and max_val > 0.0:
        return f"**{str_rep}**"
    return str_rep


def main():
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Partition 2 Evaluation on: {device}")

    # 1. GECCO Stream
    print("\n--- Loading GECCO Partition 2 Stream ---")
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
    print(f"GECCO P2: {len(gecco_p2_data)} steps, {int(np.sum(gecco_p2_labels))} anomaly seconds ({np.mean(gecco_p2_labels)*100:.2f}%)")

    # 2. NASA MSL Stream
    print("\n--- Loading NASA MSL Partition 2 Stream ---")
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
    print(f"MSL P2: {len(msl_p2_data)} steps, {int(np.sum(msl_p2_labels))} anomaly steps ({np.mean(msl_p2_labels)*100:.2f}%)")

    # 3. SWaT Stream
    print("\n--- Loading SWaT Partition 2 Stream ---")
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
    print(f"SWaT P2: {len(swat_p2_data)} steps, {int(np.sum(swat_p2_labels))} anomaly seconds ({np.mean(swat_p2_labels)*100:.2f}%)")

    # 4. SMD Multi-Server Stream (28 Machines)
    print("\n--- Loading SMD Partition 2 Stream (28 Server Entities) ---")
    smd_files = sorted(glob.glob("data/smd/test/*.txt"))
    smd_entities = []
    tot_smd_steps = 0
    tot_smd_anom = 0
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
        tot_smd_steps += len(p2_d)
        tot_smd_anom += int(np.sum(p2_l))
    print(f"SMD P2: {tot_smd_steps} steps across 28 servers, {tot_smd_anom} anomaly steps ({tot_smd_anom/max(1, tot_smd_steps)*100:.2f}%)")

    benchmarks = [
        ("SWAT", [(swat_p2_data, swat_p2_labels)], "Partition 2 (Timestamps 224,960 to 449,919 — 224,959 continuous seconds, 14 unseen attacks)", [
            "stage_coupled_tcn_best.pt",
            "swat_swat_attack_cnn_1d_backbone_cnn3_swat_best.pt",
            "swat_swat_attack_tcn_swat_tcn5_best.pt",
            "swat_swat_attack_tcn_backbone_tcn5_residual_swat_best.pt",
            "swat_swat_attack_tcn_test_swat_hybrid_best.pt"
        ], "checkpoints/loss_ablation/*SWAT*.pt"),
        ("GECCO", [(gecco_p2_data, gecco_p2_labels)], "Partition 2 (Timestamps 70,000 to 139,566 — 69,566 continuous seconds, 22 unseen events)", [
            "gecco_coupled_tcn_best.pt",
            "gecco_gecco_water_cnn_1d_gecco_cnn3_best.pt",
            "gecco_gecco_water_cnn_1d_backbone_cnn3_gecco_best.pt",
            "gecco_gecco_water_tcn_backbone_tcn5_residual_gecco_best.pt",
            "gecco_gecco_water_tcn_filter_median_shift_gecco_best.pt",
            "gecco_gecco_water_tcn_filter_none_gecco_best.pt",
            "gecco_gecco_water_tcn_filter_top_k_variance_gecco_best.pt"
        ], "checkpoints/loss_ablation/*GECCO*.pt"),
        ("MSL", [(msl_p2_data, msl_p2_labels)], "Partition 2 (Timestamps 37,000 to 73,728 — 36,729 continuous steps, 18 unseen anomaly episodes)", [
            "msl_coupled_tcn_best.pt",
            "msl_msl_test_cnn_1d_backbone_cnn3_msl_best.pt",
            "msl_msl_test_tcn_msl_tcn5_best.pt",
            "msl_msl_test_tcn_backbone_tcn5_residual_msl_best.pt"
        ], "checkpoints/loss_ablation/*MSL*.pt"),
        ("SMD", smd_entities, f"Partition 2 (Chronological Last 40% across all 28 servers — {tot_smd_steps:,} continuous steps, 17,738 anomalies)", [
            "smd_coupled_tcn_best.pt",
            "smd_machine-1-1_tcn_smd_single_m1_1_best.pt",
            "smd_pooled28_cnn_1d_backbone_cnn3_smd_best.pt",
            "smd_pooled28_tcn_backbone_tcn5_residual_smd_best.pt"
        ], "checkpoints/loss_ablation/*SMD*.pt")
    ]

    output_lines = []
    output_lines.append("# Partition 2 (Complete Held-Out Test Stream) Evaluation Registry")
    output_lines.append("")
    output_lines.append(f"**Generated on:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append(f"**Execution Device:** Apple Silicon ({device.upper()}) / Python 3.12 (`~/.venv`)")
    output_lines.append("**Evaluation Paradigm:** Continuous Sliding-Window Streaming (Window $L=200$, Stride $S=10$, Multi-Window Overlap Averaging)")
    output_lines.append("")
    output_lines.append("This document records the evaluation of all saved model checkpoints (both recent loss ablations and pre-existing baselines) directly over the **complete held-out Partition 2 test stream** of each dataset. There is 0% temporal overlap between these evaluation streams and the training regime.")
    output_lines.append("")
    output_lines.append("> [!NOTE]")
    output_lines.append("> In all metric columns below, **bold formatting is applied exclusively to the maximum value** achieved within each benchmark.")
    output_lines.append("")
    output_lines.append("---")
    output_lines.append("")

    for ds_name, entities_list, p2_desc, older_ckpts, ablation_pattern in benchmarks:
        tot_pts = sum(len(d) for d, _ in entities_list)
        tot_anom = sum(int(np.sum(l)) for _, l in entities_list)

        print(f"\n=======================================================")
        print(f"Evaluating {ds_name} Models on Complete Partition 2 Stream")
        print(f"=======================================================")
        output_lines.append(f"## {ds_name} Benchmark: Partition 2 Stream Evaluation")
        output_lines.append(f"- **Evaluated Test Stream**: `{p2_desc}`")
        output_lines.append(f"- **Total Timesteps**: {tot_pts:,} seconds/steps")
        output_lines.append(f"- **Ground-Truth Anomalies**: {tot_anom:,} steps ({tot_anom/max(1, tot_pts)*100:.2f}% base rate)")
        output_lines.append("")
        output_lines.append("| Checkpoint Filename | Paradigm / Origin | File Timestamp | Continuous Point AUROC | Continuous Point PR-AUC | Point-Adjusted F1 (PA-F1) | Point Best $F_1$ | Precision | Recall | False Positive Rate (FPR) | Event Recall | Optimal $\\tau^*$ | Throughput |")
        output_lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

        # Gather checkpoints
        abl_ckpts = sorted(glob.glob(ablation_pattern))
        old_ckpts_paths = [os.path.join("checkpoints", f) for f in older_ckpts if os.path.exists(os.path.join("checkpoints", f))]

        all_target_ckpts = old_ckpts_paths + abl_ckpts

        results_list = []
        for ckpt_path in all_target_ckpts:
            bname = os.path.basename(ckpt_path)
            mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(ckpt_path)))
            is_abl = "loss_ablation" in ckpt_path
            paradigm = "Loss Ablation (21 Nov)" if is_abl else ("Specialized Student" if "coupled" in bname else "Standard Baseline")

            print(f"  -> Evaluating: {bname} ...", end=" ", flush=True)
            try:
                model = load_model(ckpt_path, ds_name.lower(), device)
                res = run_streaming_eval_entities(model, entities_list, window_size=200, stride=10, batch_size=128, device=device)
                res["bname"] = bname
                res["mtime"] = mtime
                res["paradigm"] = paradigm
                results_list.append(res)
                print(f"AUROC={res['auroc']:.4f}, PR-AUC={res['pr_auc']:.4f}, PA-F1={res['pa_f1']:.4f}, F1={res['best_f1']:.4f}, FPR={res['fpr']*100:.2f}%")
            except Exception as e:
                print(f"ERROR: {e}")

        # Compute maximums
        max_auc = max((r["auroc"] for r in results_list), default=0.0)
        max_pr = max((r["pr_auc"] for r in results_list), default=0.0)
        max_pa_f1 = max((r["pa_f1"] for r in results_list), default=0.0)
        max_f1 = max((r["best_f1"] for r in results_list), default=0.0)
        max_prec = max((r["precision"] for r in results_list), default=0.0)
        max_rec = max((r["recall"] for r in results_list), default=0.0)

        for r in results_list:
            s_auc = format_cell(r["auroc"], max_auc, f"{r['auroc']:.4f}")
            s_pr = format_cell(r["pr_auc"], max_pr, f"{r['pr_auc']:.4f}")
            s_pa_f1 = format_cell(r["pa_f1"], max_pa_f1, f"{r['pa_f1']:.4f}")
            s_f1 = format_cell(r["best_f1"], max_f1, f"{r['best_f1']:.4f}")
            s_prec = format_cell(r["precision"], max_prec, f"{r['precision']*100:.2f}%")
            s_rec = format_cell(r["recall"], max_rec, f"{r['recall']*100:.2f}%")
            s_fpr = f"{r['fpr']*100:.2f}%"
            s_tau = f"{r['best_thresh']:.4f}"
            s_fps = f"{r['fps']:.0f} win/s"

            output_lines.append(f"| `{r['bname']}` | {r['paradigm']} | {r['mtime']} | {s_auc} | {s_pr} | {s_pa_f1} | {s_f1} | {s_prec} | {s_rec} | {s_fpr} | {r['event_recall']} | {s_tau} | {s_fps} |")

        output_lines.append("")

    output_lines.append("---")
    output_lines.append("")
    output_lines.append("## How to Re-Run This Partition 2 Registry")
    output_lines.append("To re-execute this complete evaluation over all Partition 2 continuous streams and update this document:")
    output_lines.append("```bash")
    output_lines.append("~/.venv/bin/python scripts/evaluate_partition2_registry.py")
    output_lines.append("```")
    output_lines.append("")

    content = "\n".join(output_lines)
    with open("results/PARTITION_2_TEST_EVALUATION_REGISTRY.md", "w") as f:
        f.write(content)
    with open("Notes/21Nov/PARTITION_2_TEST_EVALUATION_REGISTRY.md", "w") as f:
        f.write(content)

    print("\nPartition 2 evaluation complete! Saved to:")
    print("  - results/PARTITION_2_TEST_EVALUATION_REGISTRY.md")
    print("  - Notes/21Nov/PARTITION_2_TEST_EVALUATION_REGISTRY.md")


if __name__ == "__main__":
    main()
