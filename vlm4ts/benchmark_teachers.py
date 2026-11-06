import os
import argparse
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, f1_score

from src.dataset import load_dataset_arrays


def parse_args():
    parser = argparse.ArgumentParser(description="Head-to-head benchmark: Legacy Teacher vs VLM4TS Teacher")
    parser.add_argument("--dataset", type=str, default="swat")
    parser.add_argument("--entity", type=str, default="swat_attack")
    parser.add_argument("--legacy_cache", type=str, default="cache/swat_swat_attack_teacher_cache.pt")
    parser.add_argument("--vlm4ts_cache", type=str, default="cache/swat_swat_attack_vlm4ts_cache.pt")
    parser.add_argument("--output_md", type=str, default="results/teacher_vlm4ts_comparison.md")
    return parser.parse_args()


def evaluate_cache(cache_path, raw_data, labels, w_size=200):
    if not os.path.exists(cache_path):
        return None
    cache = torch.load(cache_path, weights_only=False)
    starts = sorted(cache.keys())
    
    y_true = []
    scores = []
    for st in starts:
        ed = st + w_size
        if ed <= len(labels):
            y_flag = 1 if np.any(labels[st:ed] >= 0.5) else 0
            score = cache[st].get("anomaly_likelihood", cache[st].get("anomaly_confidence", 0.0))
            y_true.append(y_flag)
            scores.append(float(score))

    y_true = np.array(y_true)
    scores = np.array(scores)
    
    auroc = roc_auc_score(y_true, scores)
    pr_auc = average_precision_score(y_true, scores)
    p, r, th = precision_recall_curve(y_true, scores)
    f1s = 2 * p * r / np.clip(p + r, 1e-12, None)
    best_f1 = float(np.max(f1s))
    best_th = float(th[min(int(np.argmax(f1s)), len(th) - 1)])

    return {
        "cached_windows": len(scores),
        "positive_windows": int(np.sum(y_true)),
        "auroc": auroc,
        "pr_auc": pr_auc,
        "best_f1": best_f1,
        "best_threshold": best_th
    }


def main():
    args = parse_args()
    raw_data, labels = load_dataset_arrays(args.dataset, args.entity, "data")

    res_legacy = evaluate_cache(args.legacy_cache, raw_data, labels)
    res_vlm4ts = evaluate_cache(args.vlm4ts_cache, raw_data, labels)

    rows = []
    if res_legacy:
        rows.append({"Model / Architecture": "Legacy Teacher (Single symlog plot, ±3σ prompt)", **res_legacy})
    if res_vlm4ts:
        rows.append({"Model / Architecture": "VLM4TS Teacher (Multi-panel, Phase-Diverse)", **res_vlm4ts})

    if not rows:
        print("No cache files found to benchmark.")
        return

    df = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print("        TEACHER HEAD-TO-HEAD COMPARISON ON SWAT")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70 + "\n")

    os.makedirs(os.path.dirname(args.output_md) or ".", exist_ok=True)
    df.to_markdown(args.output_md, index=False)
    print(f"Saved comparison report to: {args.output_md}")


if __name__ == "__main__":
    main()
