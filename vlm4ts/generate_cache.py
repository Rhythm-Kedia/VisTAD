import os
import argparse
import numpy as np
import torch
from tqdm import tqdm

from src.dataset import load_dataset_arrays, select_top_k_channels
from vlm4ts.renderer import render_vlm4ts_image
from vlm4ts.vit_screener import ViT4TSScreener
from vlm4ts.vlm_reasoner import VLM4TSReasoner


def parse_args():
    parser = argparse.ArgumentParser(description="Generate VLM4TS Teacher Cache with Phase Diversity")
    parser.add_argument("--dataset", type=str, default="swat", help="Dataset name (swat, smd, msl, gecco)")
    parser.add_argument("--entity", type=str, default="swat_attack", help="Entity name")
    parser.add_argument("--num_windows", type=int, default=500, help="Total curriculum windows to inspect and cache")
    parser.add_argument("--window_size", type=int, default=200, help="Sliding window size")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="Multimodal VLM backbone")
    parser.add_argument("--output_path", type=str, default=None, help="Output cache .pt file path")
    parser.add_argument("--use_vit_screener", action="store_true", default=False, help="Enable ViT4TS fast screening")
    return parser.parse_args()


def select_phase_diverse_windows(raw_data, labels, num_windows=500, w_size=200, train_frac=0.5):
    """Extracts curriculum windows across diverse temporal phases (onset at 10%, 25%, 50%, 75%, and full plateaus)."""
    n_train = int(len(raw_data) * train_frac)
    train_labels = labels[:n_train]
    anom_indices = np.where(train_labels == 1)[0]
    
    half_w = max(1, num_windows // 2)
    anom_starts = []

    # Sample across 4 distinct phase offsets: 20 (early), 50 (standard), 100 (mid), 160 (late)
    phase_offsets = [20, 50, 100, 160]
    
    if len(anom_indices) > 0:
        step_stride = max(1, len(anom_indices) // (half_w // len(phase_offsets) + 1))
        for idx in anom_indices[::step_stride]:
            for offset in phase_offsets:
                st = max(0, int(idx - offset))
                if st + w_size <= n_train:
                    anom_starts.append(st)

    anom_starts = sorted(list(set(anom_starts)))[:half_w]

    # Select clean normal anchors (labels strictly 0)
    normal_step = max(5, (n_train - 3000 - w_size) // max(1, half_w))
    normal_starts = [
        s for s in range(3000, n_train - w_size, normal_step)
        if np.all(train_labels[s : s + w_size] == 0)
    ][:half_w]

    # Interleave anomaly and normal windows
    selected = []
    for a, n in zip(anom_starts, normal_starts):
        selected.extend([a, n])
    if len(anom_starts) > len(normal_starts):
        selected.extend(anom_starts[len(normal_starts):])
    elif len(normal_starts) > len(anom_starts):
        selected.extend(normal_starts[len(anom_starts):])

    return selected[:num_windows]


def main():
    args = parse_args()
    os.makedirs("cache", exist_ok=True)
    out_path = args.output_path or f"cache/{args.dataset}_{args.entity}_vlm4ts_cache.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Generating VLM4TS Teacher Cache for {args.dataset.upper()} ({args.entity}) on {device}")
    
    raw_data, labels = load_dataset_arrays(args.dataset, args.entity, "data")
    n_total, c_total = raw_data.shape
    base_m = np.mean(raw_data[:min(3000, n_total)], axis=0)
    base_s = np.maximum(np.std(raw_data[:min(3000, n_total)], axis=0), 0.05)

    starts = select_phase_diverse_windows(raw_data, labels, num_windows=args.num_windows, w_size=args.window_size)
    print(f"Sampled {len(starts)} phase-diverse windows (early onset, plateaus, late onset, and clean baselines).")

    # Optional Stage 1: ViT4TS coarse screening
    if args.use_vit_screener:
        screener = ViT4TSScreener(device=device)
        print("Calibrating ViT4TS Screener on baseline windows...")
        cal_imgs = [
            render_vlm4ts_image(raw_data, st, st + args.window_size, list(range(min(6, c_total))), base_m, base_s)
            for st in starts[:10]
        ]
        screener.calibrate_nominal_baseline(cal_imgs)

    reasoner = VLM4TSReasoner(model_id=args.model_id, device=device)
    
    teacher_cache = {}
    print(f"Running VLM4TS multimodal reasoning over {len(starts)} candidate windows...")
    for st in tqdm(starts):
        ed = st + args.window_size
        top_k = select_top_k_channels(raw_data[st:ed], base_m, base_s, strategy="median_shift", k=min(6, c_total))
        
        img = render_vlm4ts_image(raw_data, st, ed, top_k, base_mean=base_m, base_std=base_s)
        res = reasoner.inspect_window(img, window_size=args.window_size)
        
        teacher_cache[st] = {
            "is_anomaly": res["is_anomaly"],
            "anomaly_confidence": res["anomaly_confidence"],
            "anomaly_likelihood": res["anomaly_likelihood"],
            "anomaly_type": res["anomaly_type"],
            "intervals": res["intervals"],
            "reason": res["reason"],
            "z_feature": res["z_feature"],
            "teacher_dense_mask": res["teacher_dense_mask"],
            "top_dims": top_k
        }

    torch.save(teacher_cache, out_path)
    print(f"\nSuccessfully generated and saved VLM4TS cache with {len(teacher_cache)} windows: {out_path}")


if __name__ == "__main__":
    main()
