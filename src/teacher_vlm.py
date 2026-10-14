import os
import io
import json
import re
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from PIL import Image
import torch

SYSTEM_PROMPT = (
    "You are an expert SRE telemetry analyst performing binary anomaly detection.\n"
    "You are inspecting a 2-panel line plot of z-normalized telemetry (red dashed lines = nominal ±3σ control boundaries):\n"
    "- PANEL 1 (TOP): Prior historical context (baseline operating state).\n"
    "- PANEL 2 (BOTTOM): Target window under inspection (relative steps 0 to 200).\n\n"
    "- The y-axis is linear within ±3σ and logarithmic beyond, so tall spikes are compressed.\n"
    "DIAGNOSTIC PRINCIPLES:\n"
    "1. NORMAL BEHAVIOR:\n"
    "   - Signals stay inside nominal baseline (inside ±3σ).\n"
    "   - Periodic pulses or brief isolated blips that immediately return to 0 (matching recurring patterns in Panel 1).\n"
    "2. ANOMALY:\n"
    "   - Sustained elevation or plateau above +3σ or below -3σ persisting for many steps.\n"
    "   - Coordinated multi-channel step shifts or breakdown disrupting the baseline.\n\n"
    "Answer strictly with 'YES' if Panel 2 displays a genuine operational anomaly, or 'NO' if it represents normal baseline operations."
)


def extract_telemetry_metrics(w_slice, base_m, base_s, top_dims):
    """Computes peak z-score, maximum consecutive steps exceeding 3σ, and mean deviation."""
    w_z = (w_slice[:, top_dims] - base_m[top_dims]) / base_s[top_dims]
    peak_z = float(np.max(np.abs(w_z)))
    mean_z = float(np.mean(w_z))
    runs = []
    for d in range(w_z.shape[1]):
        over = np.abs(w_z[:, d]) >= 3.0
        max_run = 0
        cur = 0
        for val in over:
            cur = cur + 1 if val else 0
            if cur > max_run:
                max_run = cur
        runs.append(max_run)
    max_sustained = max(runs) if runs else 0
    return peak_z, max_sustained, mean_z

def render_dual_horizon_image(data, start_idx, end_idx, dims, base_mean=None, base_std=None,
                              save_path=None, ctx_len=800, px=(588, 336), dpi=100, tick_step=50):
    """Dual-panel plot sized to the VLM pixel budget (588x336 <= 256*28*28).
    Shared symlog y-axis: linear inside +-3 sigma, logarithmic beyond, so spikes of 25 and 1200 sigma both fit."""
    ctx_start = max(0, start_idx - ctx_len)
    ctx_data = data[ctx_start:start_idx]
    local_data = data[start_idx:end_idx]
    w_len = len(local_data)

    mu = base_mean[dims] if base_mean is not None else np.mean(ctx_data[:, dims], axis=0)
    sigma = base_std[dims] if base_std is not None else np.std(ctx_data[:, dims], axis=0)
    sigma = np.where(sigma < 1e-4, 1.0, sigma)
    ctx_plot = (ctx_data[:, dims] - mu) / sigma
    local_plot = (local_data[:, dims] - mu) / sigma

    peak = max(np.abs(ctx_plot).max(initial=0.0), np.abs(local_plot).max(initial=0.0))
    ymax = float(max(10.0, 1.1 * peak))

    fig, (ax_macro, ax_micro) = plt.subplots(
        2, 1, figsize=(px[0] / dpi, px[1] / dpi), dpi=dpi, sharey=True,
        gridspec_kw={"height_ratios": [1, 1.3]})
    colors = plt.cm.tab10(np.arange(len(dims)) % 10)

    n_ctx = len(ctx_data)
    for i in range(len(dims)):
        ax_macro.plot(np.arange(-n_ctx, 0), ctx_plot[:, i], color=colors[i], linewidth=1.0, alpha=0.85)
        ax_micro.plot(np.arange(w_len), local_plot[:, i], color=colors[i], linewidth=1.5)

    ax_macro.set_xlim(-ctx_len, 0)
    ax_macro.set_xticks(np.arange(-ctx_len, 1, 200))
    ax_macro.set_title("PANEL 1: BASELINE (steps before the target window)",
                       fontsize=8, fontweight="bold", color="navy")
    ax_micro.set_xlim(0, w_len)
    
    ax_micro.xaxis.set_major_locator(MultipleLocator(tick_step))
    ax_micro.xaxis.set_minor_locator(MultipleLocator(10))
    
    ax_micro.set_title(f"PANEL 2: TARGET WINDOW UNDER INSPECTION (steps 0 to {w_len})",
                       fontsize=8, fontweight="bold", color="darkgreen")
    ax_micro.set_xlabel("step", fontsize=8)

    for ax in (ax_macro, ax_micro):
        ax.set_yscale("symlog", linthresh=3)
        ax.set_ylim(-ymax, ymax)
        ax.axhspan(-3, 3, color="green", alpha=0.08)              # visible normal band
        ax.axhline(3.0, color="crimson", linestyle="--", linewidth=0.9)
        ax.axhline(-3.0, color="crimson", linestyle="--", linewidth=0.9)
        ax.axhline(0.0, color="gray", linewidth=0.6, alpha=0.5)
        ax.grid(True, which="major", linestyle=":", alpha=0.4)
        ax.set_ylabel("z-score", fontsize=8)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    fig.canvas.draw()
    rgba_arr = np.asarray(fig.canvas.buffer_rgba())
    img = Image.fromarray(rgba_arr[:, :, :3])
    plt.close(fig)
    if save_path:
        img.save(save_path)
        return save_path
    return img


def parse_vlm_json_output(output_text, window_size=200):
    """Parses, sanitizes, and calibrates JSON response from the VLM."""
    clean_str = output_text.strip()
    if "```json" in clean_str:
        clean_str = clean_str.split("```json")[1].split("```")[0].strip()
    elif "```" in clean_str:
        clean_str = clean_str.split("```")[1].split("```")[0].strip()
    elif "{" in clean_str and "}" in clean_str:
        s_brace = clean_str.find("{")
        e_brace = clean_str.rfind("}") + 1
        clean_str = clean_str[s_brace:e_brace].strip()
    elif "{" in clean_str:
        s_brace = clean_str.find("{")
        clean_str = clean_str[s_brace:].strip()
        if not clean_str.endswith("}"):
            clean_str = clean_str + "\n}"
        
    try:
        parsed = json.loads(clean_str)
        if "reason" not in parsed:
            parsed["reason"] = (
                parsed.get("temporal_continuity_analysis")
                or parsed.get("macro_comparison")
                or parsed.get("observation", "")
            )
    except Exception:
        conf_match = re.search(r'"anomaly_confidence"\s*:\s*([1-3])', clean_str)
        conf = int(conf_match.group(1)) if conf_match else None
        
        is_anom_match = re.search(r'"is_anomaly"\s*:\s*(true|false)', clean_str, re.IGNORECASE)
        if is_anom_match:
            is_anom = (is_anom_match.group(1).lower() == "true")
        elif conf is not None:
            is_anom = (conf >= 2)
        else:
            is_anom = False
        
        likelihood_match = re.search(r'"anomaly_likelihood"\s*:\s*([0-9.]+)', clean_str)
        if likelihood_match:
            likelihood = float(likelihood_match.group(1))
        elif conf is not None:
            likelihood = 0.95 if conf == 3 else (0.75 if conf == 2 else 0.05)
        else:
            likelihood = 0.75 if is_anom else 0.05
            
        times_match = re.search(r'"anomalous_timesteps"\s*:\s*\[([^\]]*)\]', clean_str)
        times = []
        if times_match and times_match.group(1).strip():
            raw_nums = [x.strip() for x in times_match.group(1).split(",") if x.strip()]
            for rn in raw_nums:
                try:
                    times.append(int(float(rn)))
                except ValueError:
                    pass
            
        dims_match = re.search(r'"affected_dimensions"\s*:\s*\[([^\]]*)\]', clean_str)
        dims = []
        if dims_match and dims_match.group(1).strip():
            dims = [d.strip().strip('"').strip("'") for d in dims_match.group(1).split(",") if d.strip()]            
        
        analysis_match = re.search(r'"temporal_continuity_analysis"\s*:\s*"([^"]*)', clean_str)
        comp_match = re.search(r'"macro_comparison"\s*:\s*"([^"]*)', clean_str)
        obs_match = re.search(r'"observation"\s*:\s*"([^"]*)', clean_str)
        reason = (
            analysis_match.group(1) if analysis_match
            else (comp_match.group(1) if comp_match
            else (obs_match.group(1) if obs_match else clean_str[:80]))
        )
        
        parsed = {
            "is_anomaly": is_anom,
            "anomaly_confidence": conf,
            "anomaly_likelihood": likelihood,
            "anomalous_timesteps": times,
            "affected_dimensions": dims,
            "reason": reason
        }

    # calibrate boolean flag with confidence if provided
    if "anomaly_confidence" in parsed and parsed["anomaly_confidence"] is not None:
        try:
            c = int(parsed["anomaly_confidence"])
            if c == 1:
                parsed["is_anomaly"] = False
                parsed["anomaly_likelihood"] = min(parsed.get("anomaly_likelihood", 0.0), 0.15)
            elif c >= 2:
                parsed["is_anomaly"] = True
                parsed["anomaly_likelihood"] = max(parsed.get("anomaly_likelihood", 0.75), 0.75)
        except (ValueError, TypeError):
            pass

    # clamp likelihood
    parsed["anomaly_likelihood"] = float(np.clip(parsed.get("anomaly_likelihood", 0.0), 0.0, 1.0))
    
    # reconcile boolean flag with likelihood
    if parsed["anomaly_likelihood"] >= 0.5:
        parsed["is_anomaly"] = True
    elif parsed["anomaly_likelihood"] < 0.2:
        parsed["is_anomaly"] = False
        
    # sanitize timesteps to relative range [0, window_size]
    timesteps = parsed.get("anomalous_timesteps", [])
    valid_ts = []
    if parsed["is_anomaly"] and isinstance(timesteps, list):
        if len(timesteps) >= 2:
            t0 = int(np.clip(min(timesteps), 0, window_size))
            t1 = int(np.clip(max(timesteps), 0, window_size))
            if t0 < t1:
                valid_ts = [t0, t1]
            elif t1 < t0:
                valid_ts = [t1, t0]
            else:
                valid_ts = [max(0, t0 - 10), min(window_size, t0 + 10)]
        elif len(timesteps) == 1:
            try:
                t0 = int(np.clip(int(float(timesteps[0])), 0, window_size))
                valid_ts = [max(0, t0 - 10), min(window_size, t0 + 10)]
            except (ValueError, TypeError):
                pass
    parsed["anomalous_timesteps"] = valid_ts
    
    if not parsed["is_anomaly"]:
        parsed["anomalous_timesteps"] = []
        parsed["affected_dimensions"] = []

    # construct dense temporal guidance mask
    dense_mask = np.zeros(window_size, dtype=np.float32)
    if parsed["is_anomaly"]:
        if len(valid_ts) == 2:
            dense_mask[valid_ts[0]:valid_ts[1]] = parsed["anomaly_likelihood"]
        else:
            dense_mask[int(window_size * 0.1):int(window_size * 0.9)] = parsed["anomaly_likelihood"]
    parsed["teacher_dense_mask"] = torch.tensor(dense_mask, dtype=torch.float32)

    return parsed


def extract_logit_scores_and_features(outputs=None, inputs=None, processor=None, temperature=1.0, threshold=0.63,
                                     window_size=200, active_dims_list=None, w_slices=None,
                                     base_m=None, base_s=None,
                                     last_logits=None, z_T_batch=None):
    """Extracts calibrated soft probability a_T, pooled z_T, and dense guidance mask in a single pass."""
    if last_logits is None:
        logits = outputs.logits
        attn_mask = inputs.attention_mask
        B, S, V = logits.shape
        # Locate last valid non-padded token position for each batch item
        if (attn_mask[:, -1] == 1).all():
            last_logits = logits[:, -1, :]
        else:
            last_pos = attn_mask.sum(dim=1) - 1
            last_logits = logits[torch.arange(B, device=logits.device), last_pos, :]
    else:
        B = last_logits.shape[0]
        
    # Affirmative and negative token IDs in Qwen2/Qwen2.5-VL vocab
    # 14004: 'YES', 9454: 'Yes', 14080: ' YES', 7414: ' Yes'
    # 8996: 'NO', 2753: 'No', 5664: ' NO', 2308: ' No'
    pos_ids = [14004, 9454, 14080, 7414]
    neg_ids = [8996, 2753, 5664, 2308]
    
    # If processor provided, supplement with any extra dynamic encodings
    if processor is not None and hasattr(processor, "tokenizer"):
        tok = processor.tokenizer
        for w in ["YES", "Yes", " YES", " Yes"]:
            t_id = tok.encode(w, add_special_tokens=False)[-1]
            if t_id not in pos_ids:
                pos_ids.append(t_id)
        for w in ["NO", "No", " NO", " No"]:
            t_id = tok.encode(w, add_special_tokens=False)[-1]
            if t_id not in neg_ids:
                neg_ids.append(t_id)

    l_pos = torch.logsumexp(last_logits[:, pos_ids], dim=-1)
    l_neg = torch.logsumexp(last_logits[:, neg_ids], dim=-1)
    logit_diff = (l_pos - l_neg) / max(1e-4, temperature)
    a_T_batch = torch.sigmoid(logit_diff).float().cpu()
    
    # Multimodal feature pooling over prompt attention mask
    if z_T_batch is None:
        hidden = outputs.hidden_states[-1]
        attn_mask = inputs.attention_mask
        mask_expanded = attn_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        z_T_batch = ((hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1.0)).float().cpu()
    
    results = []
    for b in range(B):
        a_T = float(a_T_batch[b].item())
        is_anom = bool(a_T >= threshold)
        conf = 2 if is_anom else 1
        
        # Dense mask and active timesteps computation
        dense_mask = np.zeros(window_size, dtype=np.float32)
        valid_ts = []
        if is_anom and w_slices is not None and base_m is not None and base_s is not None and active_dims_list is not None:
            w_sl = w_slices[b]
            dims = active_dims_list[b]
            w_z = np.abs((w_sl[:, dims] - base_m[dims]) / base_s[dims])
            exceeding = np.any(w_z >= 2.5, axis=1)
            ex_indices = np.where(exceeding)[0]
            if len(ex_indices) >= 2:
                t0, t1 = int(ex_indices[0]), int(ex_indices[-1]) + 1
                valid_ts = [t0, t1]
                dense_mask[t0:t1] = a_T
            else:
                dense_mask[int(window_size * 0.1):int(window_size * 0.9)] = a_T
                valid_ts = [int(window_size * 0.1), int(window_size * 0.9)]
        elif is_anom:
            dense_mask[int(window_size * 0.1):int(window_size * 0.9)] = a_T
            valid_ts = [int(window_size * 0.1), int(window_size * 0.9)]
            
        dims_names = [f"Dim {d}" for d in (active_dims_list[b] if active_dims_list is not None else [])] if is_anom else []
        
        results.append({
            "is_anomaly": is_anom,
            "anomaly_confidence": conf,
            "anomaly_likelihood": a_T,
            "z_feature": z_T_batch[b],
            "anomalous_timesteps": valid_ts,
            "affected_dimensions": dims_names,
            "reason": f"Direct Logit score: {a_T:.3f} (YES={l_pos[b].item():.2f}, NO={l_neg[b].item():.2f})",
            "teacher_dense_mask": torch.tensor(dense_mask, dtype=torch.float32)
        })
        
    return results

LOC_SYSTEM = "You are a reliability engineer inspecting sensor telemetry plots. Reply with JSON only."

LOC_PROMPT = (
    "The image shows z-normalized sensor channels (one colored line per channel, 0 = typical level). "
    "The green band between the red dashed lines is the normal range (±3 standard deviations).\n"
    "- PANEL 1 (top): normal baseline behavior before the target window.\n"
    "- PANEL 2 (bottom): the target window. Its x-axis shows steps 0 to 200 with labels every 20 steps.\n\n"
    "An anomaly was detected in PANEL 2. Using the x-axis labels of PANEL 2, list the step ranges where "
    "the anomalous behavior happens (lines outside the band, a level shift, or channels breaking from "
    "PANEL 1 together). Reply with JSON only, for example {\"intervals\": [[60, 110], [150, 170]]}. "
    "Use 0 <= start < end <= 200. Reply {\"intervals\": []} if you see none."
)


def parse_intervals(text, w=200, min_len=3, merge_gap=5):
    """Extracts [start, end] pairs from the model text, clips them, sorts them and merges close ones."""
    pairs = re.findall(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]", text)
    iv = []
    for a, b in pairs:
        s, e = int(np.clip(float(a), 0, w)), int(np.clip(float(b), 0, w))
        if e - s >= min_len:
            iv.append([s, e])
    iv.sort()
    merged = []
    for s, e in iv:
        if merged and s - merged[-1][1] <= merge_gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged

from matplotlib.patches import Patch

def _runs(mask):
    """Returns [(start, end), ...] for contiguous True runs in a boolean array."""
    m = np.concatenate([[0], np.asarray(mask).astype(int), [0]])
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def render_annotated_plot(data, start_idx, end_idx, dims, gt, base_mean, base_std,
                          vlm_intervals=None, teacher_mask=None, title="", save_path=None,
                          figsize=(12, 6.5), dpi=110):
    """Target-window plot with ground truth (red shading) and a comparison strip:
    GT vs VLM intervals vs teacher dense mask."""
    local = data[start_idx:end_idx]
    w = len(local)
    mu = base_mean[dims]
    sigma = np.where(base_std[dims] < 1e-4, 1.0, base_std[dims])
    z = (local[:, dims] - mu) / sigma
    ymax = float(max(10.0, 1.1 * np.abs(z).max()))
    gt = np.asarray(gt)[:w] >= 0.5

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=figsize, dpi=dpi, sharex=True,
                                  gridspec_kw={"height_ratios": [4, 1.2]})
    colors = plt.cm.tab10(np.arange(len(dims)) % 10)
    for i, d in enumerate(dims):
        ax.plot(np.arange(w), z[:, i], color=colors[i], linewidth=1.2, label=f"Dim {d}")

    ax.set_yscale("symlog", linthresh=3)
    ax.set_ylim(-ymax, ymax)
    ax.axhspan(-3, 3, color="green", alpha=0.08)
    ax.axhline(3, color="crimson", linestyle="--", linewidth=0.9)
    ax.axhline(-3, color="crimson", linestyle="--", linewidth=0.9)
    for s, e in _runs(gt):
        ax.axvspan(s, e, color="red", alpha=0.18)
    for s, e in (vlm_intervals or []):
        ax.axvline(s, color="royalblue", linestyle="-", linewidth=1.2)
        ax.axvline(e, color="royalblue", linestyle="-", linewidth=1.2)
    ax.set_ylabel("z-score (symlog)")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(True, which="major", linestyle=":", alpha=0.4)
    ax.legend(loc="upper left", fontsize=7, ncol=min(6, len(dims)))

    # comparison strip
    rows = [("Ground truth", [(s, e - s) for s, e in _runs(gt)], "red", 2)]
    rows.append(("VLM intervals", [(s, e - s) for s, e in (vlm_intervals or [])], "royalblue", 1))
    if teacher_mask is not None:
        tm = np.asarray(teacher_mask)[:w] > 0
        rows.append(("Teacher mask", [(s, e - s) for s, e in _runs(tm)], "darkorange", 0))
    for _, spans, c, y in rows:
        if spans:
            ax2.broken_barh(spans, (y - 0.4, 0.8), facecolors=c, alpha=0.8)
    ax2.set_yticks([r[3] for r in rows])
    ax2.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax2.set_ylim(-0.6, 2.6)
    ax2.set_xlim(0, w)
    ax2.set_xlabel("step in window")
    ax2.xaxis.set_major_locator(MultipleLocator(20))
    ax2.grid(True, axis="x", linestyle=":", alpha=0.4)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
        return save_path
    fig.canvas.draw()
    img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
    plt.close(fig)
    return img