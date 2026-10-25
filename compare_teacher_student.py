import os
import glob
import json
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, f1_score, precision_score, recall_score

from src.dataset import load_dataset_arrays, select_top_k_channels
from src.models import build_student


def parse_args():
    parser = argparse.ArgumentParser(description="Head-to-Head Teacher vs. Student Evaluation")
    parser.add_argument("--cache_dir", type=str, default="cache", help="Path to teacher caches")
    parser.add_argument("--output_md", type=str, default="results/teacher_vs_student_comparison.md")
    parser.add_argument("--output_csv", type=str, default="results/teacher_vs_student_comparison.csv")
    return parser.parse_args()


def eval_scores(y_true, scores, default_thresh=0.5):
    y = np.asarray(y_true > 0, dtype=int)
    s = np.asarray(scores, dtype=float)
    if y.min() == y.max():
        return {
            "auroc": float("nan"), "auprc": float("nan"), "best_f1": float("nan"),
            "best_thresh": default_thresh, "prec_op": float("nan"), "rec_op": float("nan"),
            "f1_op": float("nan"), "n_pos": int(y.sum()), "n_total": len(y)
        }
    auroc = float(roc_auc_score(y, s))
    auprc = float(average_precision_score(y, s))
    p, r, t = precision_recall_curve(y, s)
    f1s = 2 * p * r / np.clip(p + r, 1e-12, None)
    best_idx = int(np.argmax(f1s))
    best_f1 = float(f1s[best_idx])
    best_thresh = float(t[min(best_idx, len(t) - 1)])
    
    pred_bin = (s >= default_thresh).astype(int)
    prec_op = float(precision_score(y, pred_bin, zero_division=0))
    rec_op = float(recall_score(y, pred_bin, zero_division=0))
    f1_op = float(f1_score(y, pred_bin, zero_division=0))
    
    return {
        "auroc": auroc,
        "auprc": auprc,
        "best_f1": best_f1,
        "best_thresh": best_thresh,
        "prec_op": prec_op,
        "rec_op": rec_op,
        "f1_op": f1_op,
        "n_pos": int(y.sum()),
        "n_total": len(y)
    }


def find_latest_checkpoint(dataset, entity, arch):
    if entity:
        candidates = glob.glob(f"checkpoints/{dataset}_{entity}_{arch}*_best.pt")
        if candidates:
            candidates.sort(key=os.path.getmtime, reverse=True)
            return candidates[0]
    candidates = glob.glob(f"checkpoints/{dataset}_*_{arch}*_best.pt")
    if not candidates:
        candidates = glob.glob(f"checkpoints/{dataset}_*_best.pt")
    if candidates:
        candidates.sort(key=os.path.getmtime, reverse=True)
        return candidates[0]
    return None


def df_to_markdown(df):
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def main():
    args = parse_args()
    os.makedirs("results", exist_ok=True)
    
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Teacher vs. Student Evaluator | Device: {device} | Cache: {args.cache_dir}\n")
    
    benchmarks = [
        {"name": "SWaT", "dataset": "swat", "entity": "swat_attack", "config": "configs/swat.yaml", "thresh": 0.63, "k": 6, "w": 200},
        {"name": "SMD (Pooled 28 on machine-1-1)", "dataset": "smd", "entity": "machine-1-1", "config": "configs/smd.yaml", "thresh": 0.63, "k": 6, "w": 200},
        {"name": "GECCO", "dataset": "gecco", "entity": "gecco_water", "config": "configs/gecco.yaml", "thresh": 0.50, "k": 9, "w": 200},
        {"name": "MSL", "dataset": "msl", "entity": "msl_test", "config": "configs/msl.yaml", "thresh": 0.68, "k": 6, "w": 200}
    ]
    
    table_rows = []
    
    for bench in benchmarks:
        name = bench["name"]
        ds = bench["dataset"]
        ent = bench["entity"]
        w = bench["w"]
        k = bench["k"]
        thresh = bench["thresh"]
        
        cache_path = os.path.join(args.cache_dir, f"{ds}_{ent}_teacher_cache.pt")
        if not os.path.exists(cache_path):
            print(f"Skipping {name}: Cache not found at {cache_path}")
            continue
            
        # Find best student checkpoint (prefer 1D-CNN foundation student, fallback to TCN)
        if ds == "smd":
            ckpt = (find_latest_checkpoint(ds, "pooled28", "cnn_1d") or
                    find_latest_checkpoint(ds, "pooled28", "tcn") or
                    find_latest_checkpoint(ds, ent, "cnn_1d") or
                    find_latest_checkpoint(ds, ent, "tcn"))
        else:
            ckpt = find_latest_checkpoint(ds, ent, "cnn_1d") or find_latest_checkpoint(ds, ent, "tcn")
        if not ckpt or not os.path.exists(ckpt):
            ckpt = find_latest_checkpoint(ds, ent, "")
        if not ckpt or not os.path.exists(ckpt):
            print(f"Skipping {name}: No checkpoint found for {ds}")
            continue
            
        print(f"Evaluating {name}...")
        print(f"  Teacher Cache: {cache_path}")
        print(f"  Student Checkpoint: {ckpt}")
        
        # Determine student architecture from checkpoint path
        arch = "cnn_1d" if "cnn_1d" in ckpt else ("tcn" if "tcn" in ckpt else "cnn_1d")
        num_layers = 3 if arch == "cnn_1d" else (5 if arch == "tcn" else 3)
        
        # Load raw data
        raw_data, labels = load_dataset_arrays(ds, ent, "data")
        bs_steps = 3000
        base_m = np.mean(raw_data[:bs_steps], axis=0)
        base_s = np.maximum(np.std(raw_data[:bs_steps], axis=0), 0.05)
        num_ch = raw_data.shape[1]
        in_ch = k if num_ch > k else num_ch
        
        # Load model
        student = build_student(arch, in_channels=in_ch, hidden_dim=128, num_layers=num_layers).to(device)
        student.load_state_dict(torch.load(ckpt, map_location=device))
        student.eval()
        
        # Load teacher cache
        teacher_cache = torch.load(cache_path, weights_only=False)
        starts = sorted(teacher_cache.keys())
        
        t_scores, s_scores, y_flags = [], [], []
        test_slice_start = int(len(raw_data) * 0.6)
        
        t_test, s_test, y_test = [], [], []
        
        for st in starts:
            ed = st + w
            if ed > len(raw_data):
                continue
            t_entry = teacher_cache[st]
            t_score = t_entry.get("anomaly_likelihood", t_entry.get("anomaly_confidence", 0.0))
            
            # Student forward
            w_x = raw_data[st:ed]
            if num_ch > k:
                ctx_st = max(0, st - 800)
                ctx = raw_data[ctx_st:st] if st > 0 else None
                top_k = select_top_k_channels(w_x, base_m, base_s, ctx, strategy="median_shift", k=k,
                                              force_include_channel_0=(ds == "msl"))
                w_x = w_x[:, top_k]
                norm_x = (w_x - base_m[top_k]) / base_s[top_k]
            else:
                norm_x = (w_x - base_m) / base_s
                
            tx = torch.from_numpy(norm_x.T).unsqueeze(0).float().to(device)
            with torch.no_grad():
                s_score = student(tx)["score_scalar"].item()
                
            w_y = labels[st:ed]
            y_flag = 1 if np.any(w_y >= 0.5) else 0
            
            t_scores.append(t_score)
            s_scores.append(s_score)
            y_flags.append(y_flag)
            
            # Check if in held-out test slice (zero leakage)
            if st >= test_slice_start:
                t_test.append(t_score)
                s_test.append(s_score)
                y_test.append(y_flag)
                
        # Overall metrics on all cached windows
        t_m = eval_scores(np.array(y_flags), np.array(t_scores), default_thresh=thresh)
        s_m = eval_scores(np.array(y_flags), np.array(s_scores), default_thresh=thresh)
        
        # Test slice metrics
        has_test_pos = len(y_test) > 0 and (np.sum(y_test) > 0) and (np.sum(y_test) < len(y_test))
        if has_test_pos:
            t_test_m = eval_scores(np.array(y_test), np.array(t_test), default_thresh=thresh)
            s_test_m = eval_scores(np.array(y_test), np.array(s_test), default_thresh=thresh)
            test_auc_t, test_auc_s = t_test_m["auroc"], s_test_m["auroc"]
            test_f1_t, test_f1_s = t_test_m["best_f1"], s_test_m["best_f1"]
        else:
            test_auc_t, test_auc_s = float("nan"), float("nan")
            test_f1_t, test_f1_s = float("nan"), float("nan")
            
        delta_auc = s_m["auroc"] - t_m["auroc"]
        delta_f1 = s_m["best_f1"] - t_m["best_f1"]
        
        table_rows.append({
            "Benchmark": name,
            "Domain": ds.upper(),
            "Windows": len(y_flags),
            "Teacher AUROC": round(t_m["auroc"], 4),
            "Student AUROC": round(s_m["auroc"], 4),
            "Δ AUROC": f"{'+' if delta_auc >= 0 else ''}{delta_auc:.4f}",
            "Teacher AUPRC": round(t_m["auprc"], 4),
            "Student AUPRC": round(s_m["auprc"], 4),
            "Teacher Best F1": round(t_m["best_f1"], 4),
            "Student Best F1": round(s_m["best_f1"], 4),
            "Δ F1": f"{'+' if delta_f1 >= 0 else ''}{delta_f1:.4f}",
            "Test Slice AUROC (T / S)": f"{test_auc_t:.3f} / {test_auc_s:.3f}" if not np.isnan(test_auc_t) else "N/A",
            "Student Checkpoint": os.path.basename(ckpt)
        })

    if table_rows:
        df = pd.DataFrame(table_rows)
        df.to_csv(args.output_csv, index=False)
        
        md_content = "# Teacher vs. Student Head-to-Head Comparison (Table 1)\n\n"
        md_content += df_to_markdown(df) + "\n"
        with open(args.output_md, "w") as f:
            f.write(md_content)
            
        print("\n" + "=" * 80)
        print("          TEACHER vs. STUDENT BENCHMARK COMPARISON (TABLE 1)          ")
        print("=" * 80)
        print(df_to_markdown(df))
        print(f"\nSaved artifacts:\n  - {args.output_md}\n  - {args.output_csv}\n")


if __name__ == "__main__":
    main()
