"""Teacher Representation Benchmark: T4 (Process-Stage Coupled VLM) vs T5 (Vision Foundation Model) on SWaT."""

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from PIL import Image
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    confusion_matrix, f1_score
)
from tqdm import tqdm

from src.dataset import load_dataset_arrays
from vlm4ts.renderer import render_vlm4ts_image
from vlm4ts.stage_coupled_selector import select_coupled_stage_channels, SWAT_STAGE_CHANNELS


def sample_balanced_eval_windows(raw_data, labels, n_attacks=100, n_normals=100, window_size=200):
    """Samples phase-diverse attack windows and clean nominal baseline windows."""
    n_total = len(labels)
    anom_indices = np.where(labels == 1)[0]
    
    # Stratified attack windows across onset offsets: 0, 20, 50, 100, 150
    phase_offsets = [0, 20, 50, 100, 150]
    attack_starts = []
    step = max(1, len(anom_indices) // (n_attacks // len(phase_offsets) + 1))
    for idx in anom_indices[::step]:
        for off in phase_offsets:
            st = max(0, int(idx - off))
            if st + window_size <= n_total and np.any(labels[st : st + window_size] == 1):
                attack_starts.append(st)
    attack_starts = sorted(list(set(attack_starts)))[:n_attacks]
    
    # Clean normal windows (strictly 0 across entire window)
    normal_candidates = [
        s for s in range(3000, 25000, max(5, (25000 - 3000) // (n_normals * 2)))
        if s + window_size <= n_total and np.all(labels[s : s + window_size] == 0)
    ]
    normal_starts = normal_candidates[:n_normals]
    
    return attack_starts, normal_starts


class ProcessCoupledVLMTeacher:
    """T4: Process-Stage Coupled VLM Teacher with Logit-Difference Scoring."""
    def __init__(self, model_id="Qwen/Qwen2.5-VL-3B-Instruct", device="cuda" if torch.cuda.is_available() else "cpu", temperature=1.0):
        # Auto-detect pre-mounted Kaggle model sources
        kaggle_paths = [
            "/kaggle/input/qwen2.5-vl/transformers/3b-instruct/2",
            "/kaggle/input/qwen2.5-vl/transformers/3b-instruct/1",
            "/kaggle/input/qwen-2.5-vl/transformers/3b-instruct/2",
            "/kaggle/input/qwen-2.5-vl/transformers/3b-instruct/1",
        ]
        for p in kaggle_paths:
            if os.path.exists(p):
                print(f"[T4] Found local Kaggle model mount at {p}")
                model_id = p
                break

        self.model_id = model_id
        self.device = device
        self.temperature = temperature
        self.model = None
        self.processor = None
        self.pos_token_ids = []
        self.neg_token_ids = []

    def load_model(self):
        if self.model is not None:
            return
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        print(f"[T4] Loading VLM backbone: {self.model_id} on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None
        )
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        self.model.eval()

        # Cache YES and NO token IDs for logit difference scoring
        tok = self.processor.tokenizer
        for w in ["YES", "Yes", " YES", " Yes"]:
            t_id = tok.encode(w, add_special_tokens=False)[-1]
            if t_id not in self.pos_token_ids:
                self.pos_token_ids.append(t_id)
        for w in ["NO", "No", " NO", " No"]:
            t_id = tok.encode(w, add_special_tokens=False)[-1]
            if t_id not in self.neg_token_ids:
                self.neg_token_ids.append(t_id)
        print(f"[T4] Logit tokens configured: YES={self.pos_token_ids}, NO={self.neg_token_ids}")

    def score_window(self, pil_image: Image.Image, stage_name: str) -> float:
        """Computes continuous anomaly probability via a single forward pass without autoregressive decoding."""
        self.load_model()
        from qwen_vl_utils import process_vision_info

        prompt = (
            f"You are an expert industrial reliability engineer inspecting telemetry from {stage_name}.\n"
            "The image shows a 2x2 multi-scale telemetry dashboard:\n"
            "- Left column: MACRO BASELINE (prior context showing normal operating behavior).\n"
            "- Right column: MICRO TARGET WINDOW under inspection.\n"
            "- Top row: Analog process metrics (flow rates, levels, pressure).\n"
            "- Bottom row: Discrete actuators (valves, pumps: state 1 vs 2).\n\n"
            "DIAGNOSTIC CRITERIA:\n"
            "- NORMAL: Telemetry behaves consistently with the macro baseline cycles and physical expectations (e.g. pumps turning on cause tank levels to change accordingly, signals remain within bounds).\n"
            "- ANOMALY / ATTACK: Invariant violations (e.g. pump active but level does not respond, or level changes without inflow), sensor freeze / stuck-at faults, sustained abnormal plateaus, or sharp out-of-control deviations.\n\n"
            "Does the micro target window display a genuine physical operational anomaly or attack? Answer strictly with YES or NO."
        )

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt}
            ]}
        ]

        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            if "attention_mask" in inputs:
                last_pos = inputs["attention_mask"].sum(dim=1) - 1
                last_logits = outputs.logits[torch.arange(inputs["attention_mask"].shape[0], device=outputs.logits.device), last_pos, :]
            else:
                last_logits = outputs.logits[:, -1, :]

            l_pos = torch.logsumexp(last_logits[:, self.pos_token_ids], dim=-1)
            l_neg = torch.logsumexp(last_logits[:, self.neg_token_ids], dim=-1)
            logit_diff = (l_pos - l_neg) / max(1e-4, self.temperature)
            prob = torch.sigmoid(logit_diff).item()

        return float(prob)


class VisionFoundationTeacher:
    """T5: Vision Foundation Model (ViT-B/16 / Vision-MAE representation) without text prompt bias."""
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"[T5] Initializing Vision Foundation Model (ViT-B/16) on {self.device}...")
        try:
            weights = models.ViT_B_16_Weights.DEFAULT
            vit = models.vit_b_16(weights=weights)
        except Exception:
            vit = models.vit_b_16(weights=None)
            
        vit.heads = nn.Identity()  # 768-dim latent patch pooling
        self.model = vit.to(device).eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.stage_prototypes = {}

    def calibrate_stage_prototypes(self, raw_data, base_m, base_s):
        """Calibrates nominal visual prototype for each physical stage from clean operational windows."""
        self.stage_prototypes = {}
        cal_starts = list(range(5000, 15000, 1000))
        for stage_id, meta in SWAT_STAGE_CHANNELS.items():
            chans = meta["channels"][:6]
            imgs = [
                render_vlm4ts_image(raw_data, s, s + 200, chans, base_m, base_s)
                for s in cal_starts
            ]
            tensors = torch.stack([self.transform(img) for img in imgs], dim=0).to(self.device)
            with torch.no_grad():
                feats = self.model(tensors)
                feats = nn.functional.normalize(feats, p=2, dim=-1)
                proto = nn.functional.normalize(torch.mean(feats, dim=0, keepdim=True), p=2, dim=-1)
                self.stage_prototypes[stage_id] = proto
        print(f"[T5] Calibrated nominal visual prototypes for all {len(self.stage_prototypes)} physical stages.")

    def score_window(self, pil_image: Image.Image, stage_id: int = 1) -> float:
        """Computes visual anomaly score based on cosine distance from calibrated stage prototype."""
        t = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(t)
            feat = nn.functional.normalize(feat, p=2, dim=-1)
            proto = self.stage_prototypes.get(stage_id, None)
            if proto is not None:
                sim = torch.sum(feat * proto).item()
                return float(max(0.0, 1.0 - sim))
        return 0.5


class TimeVLMTeacher:
    """Time-VLM / VETime-inspired Teacher with DINOv2 Backbone and Patch-Level Temporal Alignment."""
    def __init__(self, model_id="facebook/dinov2-base", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model_id = model_id
        print(f"[Time-VLM/VETime] Initializing DINOv2 Backbone ({model_id}) on {self.device}...")
        from transformers import AutoImageProcessor, AutoModel
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        self.stage_cls_prototypes = {}
        self.stage_patch_prototypes = {}

    def calibrate_stage_prototypes(self, raw_data, base_m, base_s):
        """Calibrates nominal CLS and spatial patch prototypes for each physical stage from clean operational windows."""
        self.stage_cls_prototypes = {}
        self.stage_patch_prototypes = {}
        cal_starts = list(range(5000, 15000, 1000))

        for stage_id, meta in SWAT_STAGE_CHANNELS.items():
            chans = meta["channels"][:6]
            imgs = [
                render_vlm4ts_image(raw_data, s, s + 200, chans, base_m, base_s)
                for s in cal_starts
            ]
            inputs = self.processor(images=imgs, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model(**inputs)
                hs = out.last_hidden_state  # [B, 257, 768]
                cls_feats = nn.functional.normalize(hs[:, 0, :], p=2, dim=-1)
                
                # Grid: 16 rows x 16 cols (columns 8..15 correspond to Micro Target Window steps 0..200)
                grid = hs[:, 1:, :].reshape(-1, 16, 16, 768)
                col_feats = nn.functional.normalize(torch.mean(grid[:, :, 8:, :], dim=1), p=2, dim=-1) # [B, 8, 768]

                proto_cls = nn.functional.normalize(torch.mean(cls_feats, dim=0, keepdim=True), p=2, dim=-1)
                proto_patch = nn.functional.normalize(torch.mean(col_feats, dim=0, keepdim=True), p=2, dim=-1) # [1, 8, 768]

                self.stage_cls_prototypes[stage_id] = proto_cls
                self.stage_patch_prototypes[stage_id] = proto_patch

        print(f"[Time-VLM/VETime] Calibrated CLS & Patch prototypes for all {len(self.stage_cls_prototypes)} physical stages.")

    def score_window(self, pil_image: Image.Image, stage_id: int = 1, return_mask=False):
        """Computes CLS distance, Patch-level temporal distance, and dense 200-step point-wise anomaly mask."""
        inputs = self.processor(images=[pil_image], return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
            hs = out.last_hidden_state
            cls_f = nn.functional.normalize(hs[:, 0, :], p=2, dim=-1)
            grid = hs[:, 1:, :].reshape(-1, 16, 16, 768)
            col_f = nn.functional.normalize(torch.mean(grid[:, :, 8:, :], dim=1), p=2, dim=-1) # [1, 8, 768]

            proto_cls = self.stage_cls_prototypes.get(stage_id, None)
            proto_patch = self.stage_patch_prototypes.get(stage_id, None)

            if proto_cls is not None and proto_patch is not None:
                d_cls = float(max(0.0, 1.0 - torch.sum(cls_f * proto_cls).item()))
                
                # Cosine similarity for each of the 8 temporal columns
                col_sim = torch.sum(col_f * proto_patch, dim=-1).squeeze(0) # [8]
                col_dist = torch.clamp(1.0 - col_sim, min=0.0).cpu().numpy() # [8]
                d_patch = float(np.max(col_dist))

                # Interpolate 8 column distances to 200 temporal points (VETime dense mask)
                x_anchors = np.linspace(0, 199, 8)
                x_query = np.arange(200)
                dense_mask = np.interp(x_query, x_anchors, col_dist)
                
                # Score combines global context and local temporal peak
                score = 0.4 * d_cls + 0.6 * d_patch
                if return_mask:
                    return float(score), dense_mask, float(d_cls), float(d_patch)
                return float(score), float(d_cls), float(d_patch)

        if return_mask:
            return 0.5, np.full(200, 0.5), 0.5, 0.5
        return 0.5, 0.5, 0.5


class UltraVETimeTeacher:
    """Ultra-VETime Teacher:
    - Backbone: facebook/dinov2-large (1024-dim self-supervised visual patch embeddings, 24 layers)
    - Multi-Regime Operational Clustering (K=5 regime centroids per stage from nominal baseline)
    - Patch-Level Temporal Alignment (columns 8..15 in 16x16 patch grid)
    """
    def __init__(self, model_id="facebook/dinov2-large", device="cuda" if torch.cuda.is_available() else "cpu", n_regimes=5):
        self.device = device
        self.model_id = model_id
        self.n_regimes = n_regimes
        print(f"[Ultra-VETime] Initializing DINOv2-Large Backbone ({model_id}) on {self.device}...")
        from transformers import AutoImageProcessor, AutoModel
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        self.stage_regimes_cls = {}
        self.stage_regimes_patch = {}

    def calibrate_multi_regime_prototypes(self, normal_data, base_m, base_s, n_samples=30):
        """Calibrates K distinct operational regime centroids per physical stage using KMeans clustering."""
        from sklearn.cluster import KMeans
        self.stage_regimes_cls = {}
        self.stage_regimes_patch = {}
        cal_starts = list(range(4000, 24000, max(1, 20000 // n_samples)))

        for stage_id, meta in SWAT_STAGE_CHANNELS.items():
            chans = meta["channels"][:6]
            imgs = [
                render_vlm4ts_image(normal_data, s, s + 200, chans, base_m, base_s)
                for s in cal_starts
            ]
            cls_list = []
            col_list = []
            batch_size = 10
            for i in range(0, len(imgs), batch_size):
                b_imgs = imgs[i : i + batch_size]
                inp = self.processor(images=b_imgs, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    out = self.model(**inp)
                    hs = out.last_hidden_state  # [B, 257, 1024]
                    cls_f = nn.functional.normalize(hs[:, 0, :], p=2, dim=-1)  # [B, 1024]
                    grid = hs[:, 1:, :].reshape(-1, 16, 16, 1024)
                    col_f = nn.functional.normalize(torch.mean(grid[:, :, 8:, :], dim=1), p=2, dim=-1)  # [B, 8, 1024]
                    cls_list.append(cls_f)
                    col_list.append(col_f)

            cls_feats = torch.cat(cls_list, dim=0)
            col_feats = torch.cat(col_list, dim=0)

            cls_np = cls_feats.cpu().numpy()
            k = min(self.n_regimes, len(cal_starts))
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(cls_np)

            regime_cls_list = []
            regime_patch_list = []
            for c_id in range(k):
                members = np.where(kmeans.labels_ == c_id)[0]
                if len(members) == 0:
                    members = [0]
                c_cls = torch.mean(cls_feats[members], dim=0, keepdim=True)
                c_patch = torch.mean(col_feats[members], dim=0, keepdim=True)
                regime_cls_list.append(nn.functional.normalize(c_cls, p=2, dim=-1))
                regime_patch_list.append(nn.functional.normalize(c_patch, p=2, dim=-1))

            self.stage_regimes_cls[stage_id] = torch.cat(regime_cls_list, dim=0)  # [K, 1024]
            self.stage_regimes_patch[stage_id] = torch.cat(regime_patch_list, dim=0)  # [K, 8, 1024]

        print(f"[Ultra-VETime] Calibrated {self.n_regimes}-regime centroids for all {len(self.stage_regimes_cls)} physical stages.")

    def score_window(self, pil_image: Image.Image, stage_id: int = 1, return_mask=False, return_feat=False):
        """Computes minimum distance to closest nominal operating regime."""
        inputs = self.processor(images=[pil_image], return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
            hs = out.last_hidden_state
            cls_f = nn.functional.normalize(hs[:, 0, :], p=2, dim=-1)  # [1, 1024]
            grid = hs[:, 1:, :].reshape(-1, 16, 16, 1024)
            col_f = nn.functional.normalize(torch.mean(grid[:, :, 8:, :], dim=1), p=2, dim=-1)  # [1, 8, 1024]

            regimes_cls = self.stage_regimes_cls.get(stage_id, None)
            regimes_patch = self.stage_regimes_patch.get(stage_id, None)

            if regimes_cls is not None and regimes_patch is not None:
                # Minimum distance to nearest operational regime in CLS
                sims_cls = torch.sum(cls_f * regimes_cls, dim=-1)  # [K]
                best_cls_idx = torch.argmax(sims_cls).item()
                d_cls = float(max(0.0, 1.0 - sims_cls[best_cls_idx].item()))

                # Minimum distance across regimes in patch space
                col_sims = torch.sum(col_f * regimes_patch, dim=-1)  # [K, 8]
                col_dists = torch.clamp(1.0 - col_sims, min=0.0).cpu().numpy()  # [K, 8]
                regime_patch_max = np.max(col_dists, axis=1)  # [K]
                best_patch_idx = int(np.argmin(regime_patch_max))
                d_patch = float(regime_patch_max[best_patch_idx])

                col_dist = col_dists[best_patch_idx]
                x_anchors = np.linspace(0, 199, 8)
                x_query = np.arange(200)
                dense_mask = np.interp(x_query, x_anchors, col_dist)

                score = 0.5 * d_cls + 0.5 * d_patch
                if return_mask and return_feat:
                    return float(score), dense_mask, float(d_cls), float(d_patch), cls_f.squeeze(0).cpu()
                elif return_mask:
                    return float(score), dense_mask, float(d_cls), float(d_patch)
                elif return_feat:
                    return float(score), float(d_cls), float(d_patch), cls_f.squeeze(0).cpu()
                return float(score), float(d_cls), float(d_patch)

        feat_zero = torch.zeros(1024, dtype=torch.float32)
        if return_mask and return_feat:
            return 0.5, np.full(200, 0.5), 0.5, 0.5, feat_zero
        elif return_mask:
            return 0.5, np.full(200, 0.5), 0.5, 0.5
        elif return_feat:
            return 0.5, 0.5, 0.5, feat_zero
        return 0.5, 0.5, 0.5


def evaluate_predictions(y_true, scores, name="Teacher"):
    y_true = np.array(y_true, dtype=int)
    scores = np.array(scores, dtype=float)

    auroc = float(roc_auc_score(y_true, scores))
    pr_auc = float(average_precision_score(y_true, scores))
    
    p, r, th = precision_recall_curve(y_true, scores)
    f1s = 2 * p * r / np.clip(p + r, 1e-12, None)
    best_idx = int(np.argmax(f1s))
    best_f1 = float(f1s[best_idx])
    best_th = float(th[min(best_idx, len(th) - 1)])

    preds = (scores >= best_th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    fpr = float(fp / max(1, fp + tn))
    fnr = float(fn / max(1, fn + tp))

    return {
        "Teacher Formulation": name,
        "AUROC": auroc,
        "PR-AUC": pr_auc,
        "Best F1": best_f1,
        "Optimal tau*": best_th,
        "FPR (False Alarm Rate)": fpr,
        "FNR (Miss Rate)": fnr,
        "True Positives": int(tp),
        "False Positives": int(fp),
        "True Negatives": int(tn),
        "False Negatives": int(fn)
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Teacher Formulations on SWaT Ground-Truth Telemetry")
    parser.add_argument("--n_eval", type=int, default=100, help="Number of attacks and normal windows (total = 2 * n_eval)")
    parser.add_argument("--eval_vlm", action="store_true", default=False, help="Whether to evaluate the slow Qwen2.5-VL model")
    parser.add_argument("--output_md", type=str, default="results/teacher_t4_t5_comparison.md")
    parser.add_argument("--output_csv", type=str, default="results/teacher_t4_t5_comparison.csv")
    args = parser.parse_args()

    print("=" * 75)
    print("      SWAT TIME-SERIES VISION FOUNDATION BENCHMARK (Time-VLM / VETime)")
    print("=" * 75)

    print("Loading SWaT datasets (swat_normal for baseline calibration, swat_attack for evaluation)...")
    try:
        normal_data, _ = load_dataset_arrays("swat", "swat_normal", "data")
        base_m = np.mean(normal_data, axis=0)
        base_s = np.maximum(np.std(normal_data, axis=0), 0.05)
        print(f"Loaded normal operational baseline ({len(normal_data)} steps) successfully.")
    except Exception as e:
        print(f"Warning: could not load swat_normal ({e}), falling back to attack[:3000].")
        normal_data = None
        base_m = None
        base_s = None

    raw_data, labels = load_dataset_arrays("swat", "swat_attack", "data")
    if base_m is None:
        base_m = np.mean(raw_data[:3000], axis=0)
        base_s = np.maximum(np.std(raw_data[:3000], axis=0), 0.05)

    attack_starts, normal_starts = sample_balanced_eval_windows(
        raw_data, labels, n_attacks=args.n_eval, n_normals=args.n_eval, window_size=200
    )
    all_eval_windows = [(st, 1) for st in attack_starts] + [(st, 0) for st in normal_starts]
    print(f"Evaluation set: {len(attack_starts)} attacks, {len(normal_starts)} normals (Total: {len(all_eval_windows)})")

    # Render stage-coupled images
    print("\nRendering process-stage coupled dual-horizon images...")
    rendered_data = []
    for st, y in tqdm(all_eval_windows, desc="Rendering"):
        coupled_dims, stage_id = select_coupled_stage_channels(raw_data[st : st + 200], base_m, base_s, max_channels=6)
        sname = SWAT_STAGE_CHANNELS[stage_id]["name"]
        img = render_vlm4ts_image(raw_data, st, st + 200, coupled_dims, base_m, base_s)
        rendered_data.append((img, stage_id, sname, y, st))

    y_true = [item[3] for item in rendered_data]
    results_list = []

    # --- 1. Evaluate T5: Vision Foundation Baseline (ViT-B/16 ImageNet Supervised) ---
    print("\n" + "-" * 75)
    print("Evaluating Model 1 (T5): ViT-B/16 (ImageNet Supervised Baseline)")
    print("-" * 75)
    t5_model = VisionFoundationTeacher()
    cal_data = normal_data if normal_data is not None else raw_data
    t5_model.calibrate_stage_prototypes(cal_data, base_m, base_s)

    t5_scores = []
    t0 = time.time()
    for img, stage_id, _, _, _ in tqdm(rendered_data, desc="T5 ViT-B/16"):
        t5_scores.append(t5_model.score_window(img, stage_id=stage_id))
    t5_duration = time.time() - t0
    t5_latency = t5_duration / len(rendered_data)
    res_t5 = evaluate_predictions(y_true, t5_scores, name="T5: ViT-B/16 (ImageNet Supervised CLS)")
    res_t5["Latency (s/window)"] = f"{t5_latency:.3f}s"
    results_list.append(res_t5)

    # --- 2. Evaluate Time-VLM / VETime Backbone (DINOv2-base) ---
    print("\n" + "-" * 75)
    print("Evaluating Model 2 & 3: Time-VLM / VETime DINOv2-base (CLS vs Patch Alignment)")
    print("-" * 75)
    time_vlm = TimeVLMTeacher()
    time_vlm.calibrate_stage_prototypes(cal_data, base_m, base_s)

    t6_dinov2_cls = []
    t7_vetime_patch = []
    t0 = time.time()
    for img, stage_id, _, _, _ in tqdm(rendered_data, desc="DINOv2 Inference"):
        score, mask, d_cls, d_patch = time_vlm.score_window(img, stage_id=stage_id, return_mask=True)
        t6_dinov2_cls.append(d_cls)
        t7_vetime_patch.append(d_patch)
    tv_duration = time.time() - t0
    tv_latency = tv_duration / len(rendered_data)

    res_t6 = evaluate_predictions(y_true, t6_dinov2_cls, name="T6: DINOv2-base (Self-Supervised Global CLS)")
    res_t6["Latency (s/window)"] = f"{tv_latency:.3f}s"
    results_list.append(res_t6)

    res_t7 = evaluate_predictions(y_true, t7_vetime_patch, name="T7: VETime / Time-VLM (DINOv2 + Micro-Target Patch Alignment)")
    res_t7["Latency (s/window)"] = f"{tv_latency:.3f}s"
    results_list.append(res_t7)

    # --- 3. Evaluate T8: Multi-Stage Physical Invariant Envelope (Top-2 Stages) ---
    print("\n" + "-" * 75)
    print("Evaluating Model 4 (T8): Time-VLM Multi-Stage Ensemble (Top-2 Stage Envelope)")
    print("-" * 75)
    t8_scores = []
    t0 = time.time()
    for _, top_stage_id, _, _, st in tqdm(rendered_data, desc="Multi-Stage Inference"):
        # Select second stage by evaluating z deviations
        z = np.abs((raw_data[st : st + 200] - base_m) / base_s)
        stage_scores = {sid: float(np.mean(z[:, smeta["channels"]])) for sid, smeta in SWAT_STAGE_CHANNELS.items()}
        ranked_stages = sorted(stage_scores.items(), key=lambda kv: kv[1], reverse=True)
        s1 = ranked_stages[0][0]
        s2 = ranked_stages[1][0] if len(ranked_stages) > 1 else s1

        # Render top 2 stages
        img1 = render_vlm4ts_image(raw_data, st, st + 200, SWAT_STAGE_CHANNELS[s1]["channels"][:6], base_m, base_s)
        img2 = render_vlm4ts_image(raw_data, st, st + 200, SWAT_STAGE_CHANNELS[s2]["channels"][:6], base_m, base_s)

        _, _, _, d1_patch = time_vlm.score_window(img1, stage_id=s1, return_mask=True)
        _, _, _, d2_patch = time_vlm.score_window(img2, stage_id=s2, return_mask=True)
        t8_scores.append(max(d1_patch, d2_patch))
    t8_duration = time.time() - t0
    t8_latency = t8_duration / len(rendered_data)

    res_t8 = evaluate_predictions(y_true, t8_scores, name="T8: Time-VLM Multi-Stage Ensemble (DINOv2 Patch Top-2)")
    res_t8["Latency (s/window)"] = f"{t8_latency:.3f}s"
    results_list.append(res_t8)

    # --- 4. Evaluate Ultra-VETime (DINOv2-Large + K=5 Regimes) ---
    print("\n" + "-" * 75)
    print("Evaluating Model 5, 6, 7: Ultra-VETime (DINOv2-Large + K=5 Regimes)")
    print("-" * 75)
    ultra_vetime = UltraVETimeTeacher(model_id="facebook/dinov2-large", n_regimes=5)
    ultra_vetime.calibrate_multi_regime_prototypes(cal_data, base_m, base_s, n_samples=30)

    t9_ultra_cls = []
    t10_ultra_patch = []
    t11_ultra_combined = []
    t0 = time.time()
    for img, stage_id, _, _, _ in tqdm(rendered_data, desc="Ultra-VETime Large"):
        score, mask, d_cls, d_patch = ultra_vetime.score_window(img, stage_id=stage_id, return_mask=True)
        t9_ultra_cls.append(d_cls)
        t10_ultra_patch.append(d_patch)
        t11_ultra_combined.append(score)
    ultra_duration = time.time() - t0
    ultra_latency = ultra_duration / len(rendered_data)

    res_t9 = evaluate_predictions(y_true, t9_ultra_cls, name="T9: Ultra-VETime (DINOv2-Large + K=5 Regimes CLS)")
    res_t9["Latency (s/window)"] = f"{ultra_latency:.3f}s"
    results_list.append(res_t9)

    res_t10 = evaluate_predictions(y_true, t10_ultra_patch, name="T10: Ultra-VETime (DINOv2-Large + K=5 Regimes Patch)")
    res_t10["Latency (s/window)"] = f"{ultra_latency:.3f}s"
    results_list.append(res_t10)

    res_t11 = evaluate_predictions(y_true, t11_ultra_combined, name="T11: Ultra-VETime (DINOv2-Large + K=5 Multi-Regime Combined)")
    res_t11["Latency (s/window)"] = f"{ultra_latency:.3f}s"
    results_list.append(res_t11)

    # --- 5. Evaluate Ultra-VETime Multi-Horizon (L=100 + L=200 Ensemble) ---
    print("\n" + "-" * 75)
    print("Evaluating Model 8 (T12): Ultra-VETime Multi-Horizon (L=100 + L=200)")
    print("-" * 75)
    t12_scores = []
    t0 = time.time()
    for idx, (img, stage_id, sname, y, st) in enumerate(tqdm(rendered_data, desc="Multi-Horizon Large")):
        coupled_dims, _ = select_coupled_stage_channels(raw_data[st : st + 200], base_m, base_s, max_channels=6)
        img_100 = render_vlm4ts_image(raw_data, st + 100, st + 200, coupled_dims, base_m, base_s, ctx_len=800)
        score_100, _, _, _ = ultra_vetime.score_window(img_100, stage_id=stage_id, return_mask=True)
        t12_scores.append(max(t11_ultra_combined[idx], score_100))
    t12_duration = time.time() - t0
    t12_latency = t12_duration / len(rendered_data)

    res_t12 = evaluate_predictions(y_true, t12_scores, name="T12: Ultra-VETime Multi-Horizon (DINOv2-Large L=100 + L=200)")
    res_t12["Latency (s/window)"] = f"{(ultra_latency + t12_latency):.3f}s"
    results_list.append(res_t12)

    # --- 6. Evaluate Ultra-VETime Multi-Stage Ensemble (Top-2 Stages with DINOv2-Large) ---
    print("\n" + "-" * 75)
    print("Evaluating Model 9 (T13): Ultra-VETime Multi-Stage (Top-2 Stages + DINOv2-Large)")
    print("-" * 75)
    t13_scores = []
    t0 = time.time()
    for _, top_stage_id, _, _, st in tqdm(rendered_data, desc="Multi-Stage Large"):
        z = np.abs((raw_data[st : st + 200] - base_m) / base_s)
        stage_scores = {sid: float(np.mean(z[:, smeta["channels"]])) for sid, smeta in SWAT_STAGE_CHANNELS.items()}
        ranked_stages = sorted(stage_scores.items(), key=lambda kv: kv[1], reverse=True)
        s1 = ranked_stages[0][0]
        s2 = ranked_stages[1][0] if len(ranked_stages) > 1 else s1

        img1 = render_vlm4ts_image(raw_data, st, st + 200, SWAT_STAGE_CHANNELS[s1]["channels"][:6], base_m, base_s)
        img2 = render_vlm4ts_image(raw_data, st, st + 200, SWAT_STAGE_CHANNELS[s2]["channels"][:6], base_m, base_s)

        sc1, _, _, _ = ultra_vetime.score_window(img1, stage_id=s1, return_mask=True)
        sc2, _, _, _ = ultra_vetime.score_window(img2, stage_id=s2, return_mask=True)
        t13_scores.append(max(sc1, sc2))
    t13_duration = time.time() - t0
    t13_latency = t13_duration / len(rendered_data)

    res_t13 = evaluate_predictions(y_true, t13_scores, name="T13: Ultra-VETime Multi-Stage (DINOv2-Large Top-2 Stages)")
    res_t13["Latency (s/window)"] = f"{t13_latency:.3f}s"
    results_list.append(res_t13)

    # --- 4. Optional VLM evaluation ---
    if args.eval_vlm:
        print("\n" + "-" * 75)
        print("Evaluating Teacher 4 (T4): Process-Stage Coupled VLM (Logit-Difference)")
        print("-" * 75)
        t4_model = ProcessCoupledVLMTeacher()
        t4_scores = []
        t0 = time.time()
        for img, _, sname, _, _ in tqdm(rendered_data, desc="T4 VLM"):
            t4_scores.append(t4_model.score_window(img, stage_name=sname))
        t4_duration = time.time() - t0
        t4_latency = t4_duration / len(rendered_data)
        res_t4 = evaluate_predictions(y_true, t4_scores, name="T4: Process-Stage Coupled VLM (Logit-Difference)")
        res_t4["Latency (s/window)"] = f"{t4_latency:.3f}s"
        results_list.append(res_t4)

    # Metrics compilation
    df = pd.DataFrame(results_list)
    print("\n" + "=" * 90)
    print("               SWAT TIME-SERIES VISION FOUNDATION TEACHER BENCHMARK")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90 + "\n")

    os.makedirs(os.path.dirname(args.output_md) or ".", exist_ok=True)
    df.to_markdown(args.output_md, index=False)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved results to {args.output_md} and {args.output_csv}")


if __name__ == "__main__":
    main()
