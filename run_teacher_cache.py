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
            device_map = "auto"
        else:
            print("[ACCELERATION] ⚠️ WARNING: CUDA NOT DETECTED! Running on CPU.")
            print("[ACCELERATION] In Kaggle, enable GPU via Settings -> Accelerator -> GPU T4 x2 or P100.")
            torch_dtype = torch.float32
            device_map = "cpu"

        print(f"Loading {cfg['teacher']['model_id']} with device_map='{device_map}'...")
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
        
        model_id = cfg["teacher"]["model_id"]
        processor = AutoProcessor.from_pretrained(model_id)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map=device_map
        )
        model.eval()
        
        if has_cuda:
            vram_used = torch.cuda.memory_allocated() / 1e9
            print(f"[ACCELERATION] Model weights successfully resident in VRAM: {vram_used:.2f} GB")
        
        ctx_len = cfg["teacher"].get("macro_context_len", 800)
        for i, st in enumerate(curriculum_starts):
            if st in teacher_cache:
                print(f"[{i+1}/{len(curriculum_starts)}] Window @ {st} already cached. Skipping.")
                continue

            t_win_start = time.time()
            end = st + w_size
            window_slice = raw_data[st:end]
            
            # compute local macro-context baseline for steady operational reference
            ctx_start = max(0, st - ctx_len)
            ctx_slice = raw_data[ctx_start:st]
            if len(ctx_slice) >= 100:
                cur_base_m = np.mean(ctx_slice, axis=0)
                cur_base_s = np.std(ctx_slice, axis=0) + 1e-6
            else:
                cur_base_m = base_m
                cur_base_s = base_s
                
            top_dims = select_top_k_channels(window_slice, cur_base_m, cur_base_s, strategy="top_k_zscore", k=6)
            
            # compute max z-score departure across selected channels
            z_scores = np.abs((window_slice[:, top_dims] - cur_base_m[top_dims]) / cur_base_s[top_dims])
            max_z = float(np.max(z_scores))
            channel_names = [f"Dim {d}" for d in top_dims]
            
            img_path = render_dual_horizon_image(
                raw_data, st, end, top_dims, cur_base_m, base_std=cur_base_s,
                save_path="temp_teacher.png", 
                ctx_len=ctx_len
            )
            
            # multimodal grounding: provide explicit active channel names and comparative guidance
            grounded_prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"CURRENT WINDOW DIAGNOSTIC TELEMETRY:\n"
                f"- Active Channels Plotted in Canvas: {channel_names}\n"
                f"- Peak Deviation in Target Window: {max_z:.2f} standard deviations (σ) from nominal baseline.\n"
                f"- Comparative Guidance: Periodic pulses, sawtooth rhythms, or baseline spikes that recur in Panel 1 are normal operational telemetry (is_anomaly: false, likelihood <= 0.15). Only flag an anomaly if Panel 2 exhibits an unprecedented disruption absent in Panel 1.\n"
                f"- Strictly report affected_dimensions ONLY from {channel_names}."
            )
            
            messages = [{"role": "user", "content": [{"type": "image", "image": img_path}, {"type": "text", "text": grounded_prompt}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            img_in, vid_in = process_vision_info(messages)
            inputs = processor(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt")
            if has_cuda:
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(model.device)
            
            with torch.no_grad():
                gen_ids = model.generate(**inputs, max_new_tokens=200, do_sample=False)
                feat_out = model(**inputs, output_hidden_states=True)
                z_T = feat_out.hidden_states[-1].mean(dim=1).squeeze(0).detach().cpu()
                
            out_ids = [o[len(inp):] for inp, o in zip(inputs.input_ids, gen_ids)]
            out_txt = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
            parsed = parse_vlm_json_output(out_txt, window_size=w_size)
            
            teacher_cache[st] = {
                "is_anomaly": parsed["is_anomaly"],
                "anomaly_likelihood": parsed["anomaly_likelihood"],
                "z_feature": z_T,
                "anomalous_timesteps": parsed.get("anomalous_timesteps", []),
                "affected_dimensions": parsed.get("affected_dimensions", []),
                "reason": parsed.get("reason", ""),
                "teacher_dense_mask": parsed.get("teacher_dense_mask", torch.zeros(w_size, dtype=torch.float32))
            }
            # persist incrementally after each window so progress is never lost
            torch.save(teacher_cache, cache_file)
            if os.path.exists(img_path):
                os.remove(img_path)
            elapsed = time.time() - t_win_start
            print(f"[{i+1}/{len(curriculum_starts)}] Cached window @ {st} in {elapsed:.2f}s: Score={parsed['anomaly_likelihood']:.2f}, IsAnom={parsed['is_anomaly']} | Reason: {parsed.get('reason', '')[:50]}")

    print(f"Teacher Cache saved successfully ({len(teacher_cache)} windows) to: {cache_file}")

if __name__ == "__main__":
    main()
