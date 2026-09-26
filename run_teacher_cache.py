import os
import yaml
import argparse
import time
import numpy as np
import torch

from src.dataset import load_dataset_arrays, select_top_k_channels
from src.teacher_vlm import render_dual_horizon_image, parse_vlm_json_output, SYSTEM_PROMPT

def parse_args():
    parser = argparse.ArgumentParser(description="Run Frozen VLM Teacher & Cache Distillation Guidance")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--num_windows", type=int, default=50, help="Number of curriculum windows to cache")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for parallel VLM teacher inference")
    parser.add_argument("--mock", action="store_true", help="Generate mock teacher cache without loading 3B model (for fast testing)")
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
        
    cache_dir = cfg["teacher"].get("cache_dir", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{cfg['dataset']['name']}_teacher_cache.pt")
    
    # 1. Load Raw Data
    raw_data, labels = load_dataset_arrays(
        dataset_name=cfg["dataset"]["name"],
        entity=cfg["dataset"].get("entity", "machine-1-1"),
        data_dir=cfg["dataset"].get("data_dir", "data")
    )
    
    w_size = cfg["dataset"]["window_size"]
    base_m = np.mean(raw_data[:cfg["dataset"].get("baseline_steps", 3000)], axis=0)
    base_s = np.std(raw_data[:cfg["dataset"].get("baseline_steps", 3000)], axis=0) + 1e-6
    
    print(f"Loaded {len(raw_data)} steps of {cfg['dataset']['name']}.")
    
    # 2. Extract targeted curriculum windows (balanced anomaly + clean normal)
    half_w = max(1, args.num_windows // 2)
    anom_indices = np.where(labels == 1)[0]
    curriculum_starts = []
    
    if len(anom_indices) > 0:
        step_stride = max(1, len(anom_indices) // half_w)
        for idx in anom_indices[::step_stride]:
            st = max(0, int(idx - 50))
            if st + w_size <= len(raw_data):
                curriculum_starts.append(st)

    # select clean normal anchors (windows where labels are strictly 0, after warmup baseline_steps)
    baseline_steps = cfg["dataset"].get("baseline_steps", 3000)
    all_normal_starts = [
        s for s in range(baseline_steps, len(raw_data) - w_size, w_size)
        if np.all(labels[s : s + w_size] == 0)
    ]
    if len(all_normal_starts) > 0:
        normal_stride = max(1, len(all_normal_starts) // half_w)
        normal_anchors = all_normal_starts[::normal_stride][:half_w]
        curriculum_starts.extend(normal_anchors)

    curriculum_starts = sorted(list(set(curriculum_starts)))[:args.num_windows]
    
    print(f"Selected {len(curriculum_starts)} balanced curriculum windows for teacher caching.")
    
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
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
        
        model_id = cfg["teacher"]["model_id"]
        # token budget cap: limits visual tokens from ~810 to ~360, accelerating prefill by >50%
        processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels=256 * 28 * 28,
            max_pixels=512 * 28 * 28
        )
        processor.tokenizer.padding_side = "left"
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map=device_map
        )
        model.eval()
        
        if has_cuda:
            vram_used = torch.cuda.memory_allocated() / 1e9
            print(f"[ACCELERATION] Model weights successfully resident in VRAM: {vram_used:.2f} GB")
        
        ctx_len = cfg["teacher"].get("macro_context_len", 800)
        pending_starts = [st for st in curriculum_starts if st not in teacher_cache]
        print(f"Curriculum Queue: {len(pending_starts)} windows to process (batch_size={args.batch_size}).")
        
        for b_idx in range(0, len(pending_starts), args.batch_size):
            t_batch_start = time.time()
            batch_starts = pending_starts[b_idx : b_idx + args.batch_size]
            batch_messages = []
            
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
                
                # in-memory rendering: returns a PIL Image directly without disk I/O
                img = render_dual_horizon_image(raw_data, st, end, top_dims, ctx_len=ctx_len)
                
                grounded_prompt = (
                    f"{SYSTEM_PROMPT}\n\n"
                    f"CURRENT WINDOW TELEMETRY INFO:\n"
                    f"- Active Channels Plotted in Canvas: {channel_names}\n"
                    f"- Evaluator Guidance: Compare Panel 2 to Panel 1. If all signals remain steady and quiet within low baseline ranges (<= 0.25) or repeat harmless heartbeat pulses, mark as normal (confidence 1). If any channel surges or is sustained above 0.40, or displays coordinated multi-channel step shifts or turbulence, mark as an anomaly (confidence 2 or 3).\n"
                    f"- Strictly report affected_dimensions ONLY from {channel_names}."
                )
                batch_messages.append([{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": grounded_prompt}]}])

            texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch_messages]
            image_inputs, video_inputs = process_vision_info(batch_messages)
            inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
            inputs = inputs.to(primary_device)
            
            with torch.inference_mode():
                # Step 1: Single forward pass to extract prompt embeddings z_T
                prompt_outputs = model(**inputs, output_hidden_states=True)
                prompt_hidden = prompt_outputs.hidden_states[-1]
                mask = inputs.attention_mask.unsqueeze(-1).to(dtype=prompt_hidden.dtype)
                z_T_batch = ((prompt_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)).float().cpu()
                del prompt_outputs, prompt_hidden

                # Step 2: Fast autoregressive text generation without retaining hidden states
                gen_outputs = model.generate(
                    **inputs,
                    max_new_tokens=90,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_hidden_states=False
                )
                
            prompt_len = inputs.input_ids.shape[1]
            for i_b, st in enumerate(batch_starts):
                out_ids = gen_outputs.sequences[i_b][prompt_len:]
                out_txt = processor.decode(out_ids, skip_special_tokens=True)
                parsed = parse_vlm_json_output(out_txt, window_size=w_size)
                
                z_T = z_T_batch[i_b]
                
                teacher_cache[st] = {
                    "is_anomaly": parsed["is_anomaly"],
                    "anomaly_confidence": parsed.get("anomaly_confidence", 1),
                    "anomaly_likelihood": parsed["anomaly_likelihood"],
                    "z_feature": z_T,
                    "anomalous_timesteps": parsed.get("anomalous_timesteps", []),
                    "affected_dimensions": parsed.get("affected_dimensions", []),
                    "reason": parsed.get("reason", ""),
                    "teacher_dense_mask": parsed.get("teacher_dense_mask", torch.zeros(w_size, dtype=torch.float32))
                }
                
            # persist incrementally after each batch
            torch.save(teacher_cache, cache_file)
            b_elapsed = time.time() - t_batch_start
            per_w = b_elapsed / len(batch_starts)
            done_count = len(teacher_cache)
            print(f"[{done_count}/{len(curriculum_starts)}] Cached batch of {len(batch_starts)} windows ({batch_starts}) in {b_elapsed:.2f}s ({per_w:.2f}s/window).")

    print(f"Teacher Cache saved successfully ({len(teacher_cache)} windows) to: {cache_file}")

    # Automated Teacher Diagnostic Audit against Ground Truth
    print("\n" + "=" * 70)
    print("                 TEACHER ACCURACY & QUALITY AUDIT                 ")
    print("=" * 70)
    tp, fp, tn, fn = 0, 0, 0, 0
    audit_windows = [st for st in curriculum_starts if st in teacher_cache]
    for st in audit_windows:
        v = teacher_cache[st]
        gt_window = labels[st : st + w_size]
        is_gt_anom = bool(np.any(gt_window >= 0.5))
        pred_anom = bool(v.get("is_anomaly", False))
        score = v.get("anomaly_likelihood", 0.0)
        conf = v.get("anomaly_confidence", 1)
        
        if is_gt_anom and pred_anom:
            tp += 1
            tag = "TP [Hit]"
        elif not is_gt_anom and not pred_anom:
            tn += 1
            tag = "TN [Clean]"
        elif not is_gt_anom and pred_anom:
            fp += 1
            tag = "FP [False Alarm]"
        else:
            fn += 1
            tag = "FN [Missed]"
            
        print(f"Window @ {st:05d} | GT: {'ANOM' if is_gt_anom else 'NORM':4s} | Pred: {'ANOM' if pred_anom else 'NORM':4s} (Score={score:.2f}, Conf={conf}) | {tag}")

    total = tp + fp + tn + fn
    acc = (tp + tn) / max(1, total)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-6, prec + rec)

    print("-" * 70)
    print(f"Summary: TP={tp}, FP={fp}, TN={tn}, FN={fn} | Total Windows: {total}")
    print(f"Teacher Accuracy:  {acc:.1%}")
    print(f"Teacher Precision: {prec:.1%}")
    print(f"Teacher Recall:    {rec:.1%}")
    print(f"Teacher F1-Score:  {f1:.3f}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
