import os
import yaml
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch

from src.dataset import load_dataset_arrays, select_top_k_channels
from src.teacher_vlm import (
    render_dual_horizon_image, render_annotated_plot, SYSTEM_PROMPT,
    extract_telemetry_metrics, extract_logit_scores_and_features,
    LOC_SYSTEM, LOC_PROMPT, parse_intervals
)

import csv
import datetime
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_curve, matthews_corrcoef)

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def eval_localization(cache, labels, starts, w_size, key):
    G, P, F, ious, hits, n_anom = [], [], [], [], 0, 0
    for st in starts:
        if st not in cache:
            continue
        g = labels[st:st + w_size] >= 0.5
        p = cache[st][key].numpy() > 0 if key in cache[st] else np.zeros(w_size, dtype=bool)   # unflagged = no mask
        G.append(g); P.append(p)
        F.append(np.full(w_size, bool(cache[st]["is_anomaly"])))          # whole-window baseline
        if g.any():
            n_anom += 1; hits += int((g & p).any())
            u = (g | p).sum(); ious.append((g & p).sum() / u if u else 0.0)
    g, p, f = np.concatenate(G), np.concatenate(P), np.concatenate(F)

    def prf(x):
        tp = (g & x).sum(); fp = (~g & x).sum(); fn = (g & ~x).sum()
        pr = tp / max(1, tp + fp); rc = tp / max(1, tp + fn)
        return pr, rc, 2 * pr * rc / max(1e-9, pr + rc)

    a, b = prf(p), prf(f)
    print(f"[{key:<18}] step P/R/F1 = {a[0]:.3f}/{a[1]:.3f}/{a[2]:.3f} | hit rate {hits / max(1, n_anom):.1%} | "
          f"mean IoU {np.mean(ious) if ious else float('nan'):.3f}")
    return b

def report_metrics(y_true, scores, threshold, tag, rule_scores=None, log_path=None, extra=None):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    pred = (scores >= threshold).astype(int)

    tp = int(((pred == 1) & (y_true == 1)).sum()); fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum()); fn = int(((pred == 0) & (y_true == 1)).sum())
    n = tp + fp + tn + fn
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn); spec = tn / max(1, tn + fp)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    acc = (tp + tn) / max(1, n); bal_acc = (rec + spec) / 2
    mcc = matthews_corrcoef(y_true, pred) if len(set(pred)) > 1 and len(set(y_true)) > 1 else float("nan")
    prev = y_true.mean() if n else float("nan")
    yes_f1 = 2 * prev / (1 + prev) if prev > 0 else float("nan")      # always-YES baseline: precision=prev, recall=1

    two_class = len(set(y_true)) == 2
    auroc = roc_auc_score(y_true, scores) if two_class else float("nan")
    auprc = average_precision_score(y_true, scores) if two_class else float("nan")
    best_f1, best_thr = float("nan"), float("nan")
    if two_class:
        p, r, t = precision_recall_curve(y_true, scores)
        f = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-9)
        if len(f):
            best_f1, best_thr = float(f.max()), float(t[f.argmax()])

    print("\n" + "=" * 70)
    print(f"  RUN SUMMARY  [{tag}]  windows={n}  positives={int(y_true.sum())} ({prev:.1%})")
    print("=" * 70)
    print(f"Threshold {threshold:.2f}:  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"  Precision {prec:.3f} | Recall {rec:.3f} | F1 {f1:.3f} | Specificity {spec:.3f}")
    print(f"  Accuracy {acc:.3f} | Balanced acc {bal_acc:.3f} | MCC {mcc:.3f}")
    print(f"Threshold-free:  AUROC {auroc:.3f} | AUPRC {auprc:.3f} (chance = {prev:.3f})")
    print(f"  Best-F1 threshold {best_thr:.3f} -> F1 {best_f1:.3f}  (tuned on this data, optimistic)")
    print(f"Baseline 'always YES':  F1 {yes_f1:.3f}  <- your F1 must clearly beat this")
    rule_aucs = {}
    if rule_scores and two_class:
        for name, vals in rule_scores.items():
            vals = np.asarray(vals, dtype=float)
            if not np.isnan(vals).any():
                rule_aucs[name] = roc_auc_score(y_true, vals)
                print(f"Rule '{name}' AUROC: {rule_aucs[name]:.3f}")
    print("=" * 70 + "\n")

    if log_path:
        row = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "tag": tag,
               "n": n, "pos_rate": round(prev, 4), "thr": threshold, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
               "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
               "specificity": round(spec, 4), "bal_acc": round(bal_acc, 4), "mcc": round(mcc, 4),
               "auroc": round(auroc, 4), "auprc": round(auprc, 4), "always_yes_f1": round(yes_f1, 4),
               "best_f1": round(best_f1, 4), "best_thr": round(best_thr, 4)}
        row.update({f"auroc_{k}": round(v, 4) for k, v in rule_aucs.items()})
        if extra: row.update(extra)
        new = not os.path.exists(log_path)
        with open(log_path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if new: w.writeheader()
            w.writerow(row)
        print(f"Appended to {log_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Run Frozen VLM Teacher & Cache Distillation Guidance")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--entity", type=str, default=None, help="Entity/machine name (overrides config)")
    parser.add_argument("--num_windows", type=int, default=80, help="Number of curriculum windows to cache")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for parallel VLM teacher inference (split across GPUs)")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold for logit classification (overrides config)")
    parser.add_argument("--mock", action="store_true", help="Generate mock teacher cache without loading 3B model (for fast testing)")
    parser.add_argument("--display-images", action="store_true",help="Save the full and VLM-view images for the first few windows to --debug-dir")
    parser.add_argument("--display-count", type=int, default=6, help="How many windows to save when --display-images is set")
    parser.add_argument("--debug-dir", type=str, default="debug_images",help="Folder for the saved debug images")
    return parser.parse_args()



def main():
    
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # CLI overrides config
    if args.entity:
        cfg["dataset"]["entity"] = args.entity
    entity = cfg["dataset"].get("entity", "machine-1-1")
    print(f"Entity: {entity}")

    cache_dir = cfg["teacher"].get("cache_dir", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{cfg['dataset']['name']}_{entity}_teacher_cache.pt")

    debug_dir = os.path.join(args.debug_dir, entity)
    if args.display_images:
        os.makedirs(debug_dir, exist_ok=True)

    raw_data, labels = load_dataset_arrays(
        dataset_name=cfg["dataset"]["name"],
        entity=entity,
        data_dir=cfg["dataset"].get("data_dir", "data")
    )
    
    w_size = cfg["dataset"]["window_size"]
    
    bs = cfg["dataset"].get("baseline_steps", 3000)
    base_m = np.mean(raw_data[:bs], axis=0)
    base_s = np.std(raw_data[:bs], axis=0)
    pos = base_s[base_s > 1e-8]
    floor = 0.05 * np.median(pos) if len(pos) else 1.0     # 5% of a typical channel's std
    base_s = np.maximum(base_s, max(floor, 1e-6))
    print(f"Baseline std: min={base_s.min():.4f} median={np.median(base_s):.4f} max={base_s.max():.4f} (floor={floor:.4f})")
    
    print(f"Loaded {len(raw_data)} steps of {cfg['dataset']['name']}.")
    
    # 2. Extract targeted curriculum windows (balanced interleaved anomaly + clean normal)
    half_w = max(1, args.num_windows // 2)
    anom_indices = np.where(labels == 1)[0]
    anom_starts = []
    
    if len(anom_indices) > 0:
        step_stride = max(1, len(anom_indices) // half_w)
        for idx in anom_indices[::step_stride]:
            st = max(0, int(idx - 50))
            if st + w_size <= len(raw_data):
                anom_starts.append(st)
        anom_starts = sorted(list(set(anom_starts)))[:half_w]

    # select clean normal anchors (windows where labels are strictly 0, after warmup baseline_steps)
    baseline_steps = cfg["dataset"].get("baseline_steps", 3000)
    normal_step = max(5, (len(raw_data) - baseline_steps - w_size) // max(1, half_w * 2))
    all_normal_starts = [
        s for s in range(baseline_steps, len(raw_data) - w_size, normal_step)
        if np.all(labels[s : s + w_size] == 0)
    ]
    normal_anchors = []
    if len(all_normal_starts) > 0:
        normal_stride = max(1, len(all_normal_starts) // half_w)
        normal_anchors = all_normal_starts[::normal_stride][:half_w]

    # Interleave so every batch immediately exercises both anomaly and normal detection
    curriculum_starts = []
    for a, n in zip(anom_starts, normal_anchors):
        curriculum_starts.extend([a, n])
    if len(anom_starts) > len(normal_anchors):
        curriculum_starts.extend(anom_starts[len(normal_anchors):])
    elif len(normal_anchors) > len(anom_starts):
        curriculum_starts.extend(normal_anchors[len(anom_starts):])
    curriculum_starts = curriculum_starts[:args.num_windows]
    
    print(f"Selected {len(curriculum_starts)} balanced curriculum windows (interleaved: {len(anom_starts)} anomaly, {len(normal_anchors)} normal).")
    
    ctx_len = cfg["teacher"].get("macro_context_len", 800)

    def get_top_dims(st):
                end = st + w_size
                ctx_slice = raw_data[max(0, st - ctx_len):st]
                return select_top_k_channels(
                    raw_data[st:end], base_mean=base_m, base_std=base_s, ctx_data=ctx_slice,
                    strategy=cfg["channel_filter"].get("strategy", "median_shift"),
                    k=cfg["channel_filter"].get("k", 6),
                )
                
    if os.path.exists(cache_file):
        print(f"Found existing teacher cache at {cache_file}. Loading to resume...")
        teacher_cache = torch.load(cache_file)
        print(f"Loaded {len(teacher_cache)} existing cached windows.")
    else:
        teacher_cache = {}
    
    
    if args.mock:
        print("Running in MOCK mode: Simulating teacher outputs for pipeline validation...")
        for st in curriculum_starts:
            if st in teacher_cache:
                continue
            end = st + w_size
            gt_w = labels[st:end]
            has_anom = bool(np.any(gt_w == 1))
            ratio = float(np.mean(gt_w == 1))
            
            # simulate calibrated teacher probability and dense mask
            sim_likelihood = min(1.0, max(0.0, ratio + np.random.normal(0, 0.05))) if has_anom else 0.03
            dense_mask = np.zeros(w_size, dtype=np.float32)
            if has_anom:
                dense_mask[np.where(gt_w == 1)[0]] = sim_likelihood
                
            teacher_cache[st] = {
                "is_anomaly": has_anom,
                "anomaly_confidence": 3 if has_anom else 1,
                "anomaly_likelihood": sim_likelihood,
                "z_feature": torch.randn(cfg["student"].get("teacher_feat_dim", 2048)),
                "anomalous_timesteps": [20, 80] if has_anom else [],
                "affected_dimensions": ["Dim 1", "Dim 5"] if has_anom else [],
                "reason": "Simulated teacher SRE diagnosis",
                "teacher_dense_mask": torch.tensor(dense_mask, dtype=torch.float32)
            }
            torch.save(teacher_cache, cache_file)
            
    else:
        # Check GPU Availability and Print Diagnostics
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            num_gpus = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[ACCELERATION] CUDA Active: Found {num_gpus} GPU(s). Primary: {gpu_name} ({total_vram:.2f} GB VRAM)")
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            device_map = {"": 0}
            primary_device = "cuda:0"
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        else:
            print("[ACCELERATION] ⚠️ WARNING: CUDA NOT DETECTED! Running on CPU.")
            print("[ACCELERATION] In Kaggle, enable GPU via Settings -> Accelerator -> GPU T4 x2 or P100.")
            torch_dtype = torch.float32
            device_map = "cpu"
            primary_device = "cpu"

        print(f"Loading {cfg['teacher']['model_id']} with token budget capping and left-padding...")
        
        model_id = cfg["teacher"]["model_id"]
        # token budget: cap visual tokens to ~252 per image (down from 1,008) to slash GPU compute by ~4x
        min_px = cfg["teacher"].get("min_pixels", 128 * 28 * 28)
        max_px = cfg["teacher"].get("max_pixels", 256 * 28 * 28)
        processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels=min_px,
            max_pixels=max_px
        )
        processor.tokenizer.padding_side = "left"
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map=device_map, attn_implementation="sdpa"
        )
        model.eval()
        
        models = [model]
        devices = [primary_device]
        
        if has_cuda:
            vram_used = torch.cuda.memory_allocated(0) / 1e9
            print(f"[ACCELERATION] Primary Worker loaded on cuda:0 ({vram_used:.2f} GB VRAM).")
            if num_gpus >= 2:
                print(f"[ACCELERATION] Initializing Dual-GPU Data Parallelism: loading Worker 2 on cuda:1...")
                model_1 = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_id, torch_dtype=torch_dtype, device_map={"": 1}, attn_implementation="sdpa"
                )
                model_1.eval()
                models.append(model_1)
                devices.append("cuda:1")
                vram_1 = torch.cuda.memory_allocated(1) / 1e9
                print(f"[ACCELERATION] Worker 2 loaded on cuda:1 ({vram_1:.2f} GB VRAM). Dual-GPU Active!")

        def run_forward_worker(model_ref, dev, msgs):
            texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
            image_inputs, video_inputs = process_vision_info(msgs)
            inp = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
            inp = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in inp.items()}
            with torch.inference_mode():
                base_outputs = model_ref.model(**inp)
                last_hidden = base_outputs[0]
                mask = inp["attention_mask"].unsqueeze(-1).to(dtype=last_hidden.dtype)
                z_T = ((last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)).float().cpu()
                if (inp["attention_mask"][:, -1] == 1).all():
                    last_tok = last_hidden[:, -1:, :]
                else:
                    last_pos = inp["attention_mask"].sum(dim=1) - 1
                    last_tok = last_hidden[torch.arange(len(last_hidden), device=dev), last_pos:last_pos+1, :]
                logits = model_ref.lm_head(last_tok).squeeze(1).float().cpu()
            return logits, z_T
        
                # Second processor for the localization pass: larger image budget so tick labels stay readable
        loc_processor = AutoProcessor.from_pretrained(model_id, min_pixels=min_px, max_pixels=512 * 28 * 28)
        loc_processor.tokenizer.padding_side = "left"

        def vlm_localize(model_ref, dev, imgs, max_new_tokens=48):
            msgs = [[{"role": "system", "content": [{"type": "text", "text": LOC_SYSTEM}]},
                     {"role": "user", "content": [{"type": "image", "image": im},
                                                  {"type": "text", "text": LOC_PROMPT}]}] for im in imgs]
            texts = [loc_processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
            image_inputs, video_inputs = process_vision_info(msgs)
            inp = loc_processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
            inp = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in inp.items()}
            with torch.inference_mode():
                out = model_ref.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
            return loc_processor.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        
        decision_threshold = args.threshold if args.threshold is not None else cfg["teacher"].get("decision_threshold", 0.63)
        pending_starts = [st for st in curriculum_starts if st not in teacher_cache]
        print(f"Curriculum Queue: {len(pending_starts)} windows to process (batch_size={args.batch_size}, threshold={decision_threshold}).")
        
        for b_idx in range(0, len(pending_starts), args.batch_size):
            t_batch_start = time.perf_counter()
            batch_starts = pending_starts[b_idx : b_idx + args.batch_size]
            batch_messages = []
            batch_windows = []
            batch_top_dims = []
            
            t0 = time.perf_counter()
            for st in batch_starts:
                end = st + w_size
                window_slice = raw_data[st:end]
                
                ctx_start = max(0, st - ctx_len)
                ctx_slice = raw_data[ctx_start:st]
                
                filter_strategy = cfg["channel_filter"].get("strategy", "median_shift")
                k_channels = cfg["channel_filter"].get("k", 6)
                top_dims = select_top_k_channels(
                    window_slice, 
                    base_mean=base_m, 
                    base_std=base_s, 
                    ctx_data=ctx_slice, 
                    strategy=filter_strategy, 
                    k=k_channels
                )
                channel_names = [f"Dim {d}" for d in top_dims]
                batch_windows.append(window_slice)
                batch_top_dims.append(top_dims)
                
                peak_z, sustained_steps, mean_z = extract_telemetry_metrics(
                    window_slice, base_m, base_s, top_dims
                )
                
                # in-memory rendering: returns a PIL Image directly without disk I/O, z-normalized
                img = render_dual_horizon_image(
                    raw_data, st, end, top_dims,
                    base_mean=base_m, base_std=base_s,
                    ctx_len=ctx_len
                )
                
                if args.display_images and st in curriculum_starts[:args.display_count]:
                    img.save(os.path.join(args.debug_dir, f"debug_{st}_full.png"))
                    s = min(1.0, (max_px / (img.width * img.height)) ** 0.5)
                    vlm_view = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))))
                    vlm_view.save(os.path.join(args.debug_dir, f"debug_{st}_vlm_view.png"))
                        
                # if st in curriculum_starts[:6]:
                #     img.save(f"debug_{st}_full.png")
                #     s = (max_px / (img.width * img.height)) ** 0.5
                #     img.resize((int(img.width * s), int(img.height * s))).save(f"debug_{st}_vlm_view.png")
                
                grounded_prompt = (
                    f"ACTIVE CHANNELS TELEMETRY:\n"
                    f"- Active Channels: {channel_names}\n"
                    f"- Peak absolute z-score: {peak_z:.1f}σ\n"
                    f"- Max consecutive steps beyond ±3σ: {sustained_steps} steps\n"
                    f"- Active channels mean deviation: {mean_z:.2f}σ\n\n"
                    f"TASK:\n"
                    f"Compare Panel 2 to Panel 1. Does Panel 2 display a genuine operational anomaly (a sustained plateau of 15+ steps or coordinated breakdown) breaking from the baseline in Panel 1?\n"
                    f"Brief isolated pulses or routine background noise matching Panel 1 are NORMAL.\n"
                    f"Answer strictly with YES or NO.\n"
                    f"Answer:"
                )
                batch_messages.append([
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": grounded_prompt}]}
                ])
            t_plot = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            if len(models) >= 2 and len(batch_messages) >= 2:
                half = (len(batch_messages) + 1) // 2
                msgs_0 = batch_messages[:half]
                msgs_1 = batch_messages[half:]
                with ThreadPoolExecutor(max_workers=2) as executor:
                    fut_0 = executor.submit(run_forward_worker, models[0], devices[0], msgs_0)
                    fut_1 = executor.submit(run_forward_worker, models[1], devices[1], msgs_1)
                    logits_0, z_0 = fut_0.result()
                    logits_1, z_1 = fut_1.result()
                last_logits = torch.cat([logits_0, logits_1], dim=0)
                z_T_batch = torch.cat([z_0, z_1], dim=0)
            else:
                last_logits, z_T_batch = run_forward_worker(models[0], devices[0], batch_messages)

            if has_cuda:
                torch.cuda.synchronize()
            t_gpu = (time.perf_counter() - t0) * 1000
            t_prep = 0.0

            t0 = time.perf_counter()
            batch_results = extract_logit_scores_and_features(
                processor=processor,
                temperature=1.0, threshold=decision_threshold, window_size=w_size,
                active_dims_list=batch_top_dims, w_slices=batch_windows,
                base_m=base_m, base_s=base_s,
                last_logits=last_logits, z_T_batch=z_T_batch
            )
            
                
            for i_b, st in enumerate(batch_starts):
                batch_results[i_b]["top_dims"] = [int(d) for d in batch_top_dims[i_b]]
                teacher_cache[st] = batch_results[i_b]
                
            # persist incrementally after each batch
            torch.save(teacher_cache, cache_file)
            t_ext_save = (time.perf_counter() - t0) * 1000
            
            b_elapsed = time.perf_counter() - t_batch_start
            per_w = b_elapsed / len(batch_starts)
            done_count = len(teacher_cache)
            print(f"[{done_count}/{len(curriculum_starts)}] Cached batch of {len(batch_starts)} windows in {b_elapsed:.2f}s ({per_w:.2f}s/w) | "
                  f"Plot: {t_plot:.0f}ms | Prep: {t_prep:.0f}ms | GPU: {t_gpu:.0f}ms | Ext/Save: {t_ext_save:.0f}ms")

                # Localization pass: only windows the teacher flagged and that haven't been localized yet
                
                
        todo = [st for st in curriculum_starts
                if st in teacher_cache and teacher_cache[st]["is_anomaly"] and "vlm_intervals" not in teacher_cache[st]]
        print(f"Localization pass on {len(todo)} flagged windows...")
        loc_bs = max(1, args.batch_size // 2)                    # larger images and generation need more memory
        
    
        for i in range(0, len(todo), loc_bs):
            chunk = todo[i:i + loc_bs]
            imgs = []
            for st in chunk:
                dims = teacher_cache[st].get("top_dims")
                if dims is None:                      # old cache entry
                    dims = [int(d) for d in get_top_dims(st)]
                    teacher_cache[st]["top_dims"] = dims
                imgs.append(render_dual_horizon_image(
                    raw_data, st, st + w_size, dims,
                    base_mean=base_m, base_std=base_s, ctx_len=ctx_len,
                    px=(840, 476), tick_step=20))
            texts = vlm_localize(models[0], devices[0], imgs)
            for st, t in zip(chunk, texts):
                iv = parse_intervals(t, w_size)
                m = np.zeros(w_size, dtype=np.float32)
                for s, e in iv:
                    m[s:e] = 1.0
                teacher_cache[st]["vlm_intervals"] = iv
                teacher_cache[st]["vlm_raw"] = t                 # raw model text, for inspection
                teacher_cache[st]["vlm_mask"] = torch.from_numpy(m)
            torch.save(teacher_cache, cache_file)

    print(f"Teacher Cache saved successfully ({len(teacher_cache)} windows) to: {cache_file}")

    # Automated Teacher Diagnostic Audit against Ground Truth
    print("\n" + "=" * 70)
    print("                 TEACHER ACCURACY & QUALITY AUDIT                 ")
    print("=" * 70)
    
    
    decision_threshold = args.threshold if args.threshold is not None else cfg["teacher"].get("decision_threshold", 0.63)
    audit_windows = [st for st in curriculum_starts if st in teacher_cache]
    y_true, y_score, y_peak, y_sus = [], [], [], []
    for st in audit_windows:
        v = teacher_cache[st]
        y_true.append(int(np.any(labels[st: st + w_size] >= 0.5)))
        y_score.append(float(v.get("anomaly_likelihood", 0.0)))
        y_peak.append(v.get("peak_z", np.nan))
        y_sus.append(v.get("max_sustained", np.nan))

    if y_true:
        report_metrics(
            y_true, y_score, decision_threshold,
            tag=f"{cfg['dataset']['name']}/{cfg['dataset'].get('entity', '')}",
            rule_scores={"peak_z": y_peak, "max_sustained": y_sus},
        )
        
        for key in ("teacher_dense_mask", "vlm_mask"):
            base = eval_localization(teacher_cache, labels, audit_windows, w_size, key)
        print(f"[{'whole-window':<18}] step P/R/F1 = {base[0]:.3f}/{base[1]:.3f}/{base[2]:.3f}   <- baseline to beat")
    else:
        print("No windows to audit.")
        
    if args.display_images:
        shown = [st for st in curriculum_starts[:args.display_count] if st in teacher_cache]
        for st in shown:
            v = teacher_cache[st]
            dims = v.get("top_dims") or [int(d) for d in get_top_dims(st)]
            gt_w = labels[st:st + w_size]
            title = (f"window {st} | GT anomaly: {bool(np.any(gt_w >= 0.5))} | "
                     f"teacher p={v.get('anomaly_likelihood', 0):.2f} flagged={v['is_anomaly']} | "
                     f"VLM intervals: {v.get('vlm_intervals', [])}")
            render_annotated_plot(
                raw_data, st, st + w_size, dims, gt_w, base_m, base_s,
                vlm_intervals=v.get("vlm_intervals", []),
                teacher_mask=v.get("teacher_dense_mask"),
                title=title,
                save_path=os.path.join(args.debug_dir, f"debug_{st}_annotated.png"))
        # print(f"Saved {len(shown)} annotated plots to {args.debug_dir}")

if __name__ == "__main__":
    main()
