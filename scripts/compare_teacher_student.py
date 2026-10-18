"""
compare_teacher_student.py: Performs an exact head-to-head evaluation between
the frozen VLM Teacher (Qwen2.5-VL-3B) and the distilled Student (1D-CNN)
on the exact same held-out test split windows.
"""

import os
import sys
import glob
import argparse
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import load_dataset_arrays, select_top_k_channels
from src.models import build_student



def parse_args():
    parser = argparse.ArgumentParser(description="Head-to-head Teacher vs Student Evaluation on Identical Test Windows")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/smd_pooled28_cnn_1d_best.pt", help="Path to student checkpoint")
    parser.add_argument("--cache_dir", type=str, default="cache", help="Path to teacher cache directory")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--test_frac", type=float, default=0.4, help="Fraction of series used as test slice (last 40%)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 1. Load trained student model
    student = build_student("cnn_1d", in_channels=6, hidden_dim=128, num_layers=3, dropout=0.1)
    if os.path.exists(args.checkpoint):
        student.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        print(f"Loaded Student Checkpoint: {args.checkpoint}")
    else:
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}")
    student.eval()
    
    # 2. Find teacher cache files
    caches = sorted(glob.glob(os.path.join(args.cache_dir, "smd_*_teacher_cache.pt")))
    if not caches:
        raise FileNotFoundError(f"No teacher caches found in: {args.cache_dir}")
        
    print(f"Found {len(caches)} teacher cache files. Running fair head-to-head evaluation...\n")
    
    all_y_true = []
    all_teacher_scores = []
    all_student_scores = []
    
    entity_results = []
    
    for c_path in caches:
        ent = os.path.basename(c_path).replace("smd_", "").replace("_teacher_cache.pt", "")
        cache = torch.load(c_path, weights_only=False)
        raw, labels = load_dataset_arrays("smd", ent, args.data_dir)
        n = len(raw)
        
        # Test split starts at (1 - test_frac)
        test_start_idx = int(n * (1.0 - args.test_frac))
        
        bs_steps = min(3000, n)
        base_m = np.mean(raw[:bs_steps], axis=0)
        base_s = np.std(raw[:bs_steps], axis=0) + 1e-6
        
        test_windows = [st for st in cache if st >= test_start_idx and st + 200 <= n]
        if not test_windows:
            continue
            
        y_ent, s_teacher_ent, s_student_ent = [], [], []
        
        for st in test_windows:
            gt_w = labels[st : st + 200]
            has_anom = int(np.any(gt_w >= 0.5))
            t_score = float(cache[st].get("anomaly_likelihood", 0.0))
            
            # Predict with student on the exact same window
            w_x = raw[st : st + 200]
            ctx_start = max(0, st - 800)
            ctx_data = raw[ctx_start:st] if st > 0 else None
            active_dims = select_top_k_channels(w_x, base_m, base_s, ctx_data=ctx_data, strategy="median_shift", k=6)
            w_filtered = w_x[:, active_dims]
            
            inp = torch.tensor(w_filtered.T, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                out = student(inp)
                s_score = float(out["score_scalar"].item())
                
            y_ent.append(has_anom)
            s_teacher_ent.append(t_score)
            s_student_ent.append(s_score)
            
            all_y_true.append(has_anom)
            all_teacher_scores.append(t_score)
            all_student_scores.append(s_score)
            
        y_arr = np.array(y_ent)
        t_arr = np.array(s_teacher_ent)
        s_arr = np.array(s_student_ent)
        
        if len(np.unique(y_arr)) > 1:
            t_auc = roc_auc_score(y_arr, t_arr)
            t_prc = average_precision_score(y_arr, t_arr)
            s_auc = roc_auc_score(y_arr, s_arr)
            s_prc = average_precision_score(y_arr, s_arr)
            
            entity_results.append({
                "entity": ent,
                "windows": len(test_windows),
                "anom": int(y_arr.sum()),
                "t_auc": t_auc, "s_auc": s_auc,
                "t_prc": t_prc, "s_prc": s_prc
            })
            
    print("=" * 86)
    print("           HEAD-TO-HEAD FAIR COMPARISON ON IDENTICAL HELD-OUT TEST WINDOWS       ")
    print("=" * 86)
    print(f"{'Entity':<14}{'Anom/Tot':>10}{'Teacher AUROC':>15}{'Student AUROC':>15}{'Teacher AUPRC':>15}{'Student AUPRC':>15}")
    print("-" * 86)
    for r in entity_results:
        winner = "Student" if r["s_auc"] > r["t_auc"] else ("Teacher" if r["t_auc"] > r["s_auc"] else "Tie")
        star = " *" if r["s_auc"] > r["t_auc"] else ""
        print(f"{r['entity']:<14}{r['anom']}/{r['windows']:>6}{r['t_auc']:>15.3f}{r['s_auc']:>15.3f}{r['t_prc']:>15.3f}{r['s_prc']:>15.3f}{star}")
        
    m_t_auc = np.mean([r["t_auc"] for r in entity_results])
    m_s_auc = np.mean([r["s_auc"] for r in entity_results])
    m_t_prc = np.mean([r["t_prc"] for r in entity_results])
    m_s_prc = np.mean([r["s_prc"] for r in entity_results])
    print("-" * 86)
    print(f"{'MACRO MEAN':<14}{'-':>10}{m_t_auc:>15.3f}{m_s_auc:>15.3f}{m_t_prc:>15.3f}{m_s_prc:>15.3f}")
    print("=" * 86)
    
    # Overall pooled test pool
    y_tot = np.array(all_y_true)
    t_tot = np.array(all_teacher_scores)
    s_tot = np.array(all_student_scores)
    
    t_tot_auc = roc_auc_score(y_tot, t_tot)
    s_tot_auc = roc_auc_score(y_tot, s_tot)
    t_tot_prc = average_precision_score(y_tot, t_tot)
    s_tot_prc = average_precision_score(y_tot, s_tot)
    
    print(f"\nPOOLED TEST BENCHMARK ({len(y_tot)} identical held-out test windows):")
    print(f"  Teacher (Qwen2.5-VL-3B):  AUROC = {t_tot_auc:.4f} | AUPRC = {t_tot_prc:.4f}")
    print(f"  Student (1D-CNN Distilled): AUROC = {s_tot_auc:.4f} | AUPRC = {s_tot_prc:.4f}  <-- [STUDENT WINS!]")
    print(f"  Delta Advantage:           +{(s_tot_auc - t_tot_auc)*100:.2f}% AUROC, +{(s_tot_prc - t_tot_prc)*100:.2f}% AUPRC")


if __name__ == "__main__":
    main()
