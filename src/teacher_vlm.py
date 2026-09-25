import os
import json
import re
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import torch

SYSTEM_PROMPT = (
    "You are an expert site reliability engineer (SRE) performing precision time series anomaly detection on server metrics.\n\n"
    "The image contains TWO separate panels:\n"
    "- PANEL 1 (TOP): Macro historical context showing past metric behavior. DO NOT flag historical spikes from Panel 1.\n"
    "- PANEL 2 (BOTTOM): Target window under inspection (relative steps 0 to 200).\n"
    "  * Solid curves: Actual sensor values for the active channels.\n"
    "  * Colored dashed lines: Baseline nominal level (μ0).\n"
    "  * Shaded tolerance bands: Normal operational range (±2.5σ). Curves inside this band are NORMAL.\n\n"
    "ANOMALY EVALUATION PRINCIPLES:\n"
    "1. NORMAL OPERATION (is_anomaly: false, likelihood <= 0.15):\n"
    "   - Curves fluctuate inside or near the shaded ±2.5σ tolerance band.\n"
    "   - Routine periodic background pulses or stationary noise.\n"
    "2. TRUE ANOMALY (is_anomaly: true, likelihood >= 0.80):\n"
    "   - Sudden massive spikes breaking clearly OUTSIDE the ±2.5σ tolerance band.\n"
    "   - Prolonged severe divergence or uncharacteristic multi-channel disruptions.\n\n"
    "CALIBRATION GUIDELINES:\n"
    "- 0.00 - 0.20: Normal operation. Curves remain inside the shaded ±2.5σ tolerance band.\n"
    "  --> is_anomaly: false, anomalous_timesteps: [], affected_dimensions: []\n"
    "- 0.25 - 0.45: Suspicious or mild noise. Minor transient deviation that quickly recovers.\n"
    "  --> is_anomaly: false, anomalous_timesteps: [], affected_dimensions: []\n"
    "- 0.50 - 0.74: Moderate anomaly. Sustained departure breaking out of the tolerance band.\n"
    "  --> is_anomaly: true, anomalous_timesteps: [<start_step>, <end_step>], affected_dimensions: [...]\n"
    "- 0.75 - 1.00: Severe breakdown or extreme novel spikes. Dramatic breakout far beyond the tolerance band.\n"
    "  --> is_anomaly: true, anomalous_timesteps: [<start_step>, <end_step>], affected_dimensions: [...]\n\n"
    "CRITICAL CONSTRAINTS:\n"
    "1. Only flag anomalies that are actively occurring in PANEL 2. Do NOT flag historical anomalies from Panel 1.\n"
    "2. 'anomalous_timesteps' must be relative integers strictly within [0, 200]. If no anomaly in Panel 2, use [].\n\n"
    "OUTPUT FORMAT: Return a valid JSON object with this exact key order:\n"
    "{\n"
    "  \"observation\": \"<one-sentence observation of whether curves remain inside or break out of the shaded ±2.5σ tolerance band>\",\n"
    "  \"is_anomaly\": <true/false>,\n"
    "  \"anomaly_likelihood\": <float between 0.0 and 1.0>,\n"
    "  \"anomalous_timesteps\": [<start_0_to_200>, <end_0_to_200>],\n"
    "  \"affected_dimensions\": [\"Dim X\", ...]\n"
    "}"
)


def render_dual_horizon_image(data, start_idx, end_idx, dims, base_mean, base_std=None, save_path="temp_dual.png", ctx_len=800):
    """Renders dual-horizon canvas with explicit panel demarcation and matching baseline colors."""
    ctx_start = max(0, start_idx - ctx_len)
    ctx_data = data[ctx_start:end_idx]
    local_data = data[start_idx:end_idx]
    w_len = len(local_data)
    
    fig, (ax_macro, ax_micro) = plt.subplots(2, 1, figsize=(8.5, 5.0), dpi=100, gridspec_kw={"height_ratios": [1, 1.4]})
    
    # top panel: macro context (read-only reference)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(dims), 1)))
    for i, d in enumerate(dims):
        ax_macro.plot(range(len(ctx_data)), ctx_data[:, d], label=f"Dim {d}", color=colors[i], linewidth=0.9, alpha=0.8)
    loc_start = start_idx - ctx_start
    ax_macro.axvspan(loc_start, loc_start + w_len, color="gold", alpha=0.35, label="Target Window Location")
    ax_macro.set_title("PANEL 1: MACRO CONTEXT (Past History - READ ONLY REFERENCE, DO NOT FLAG)", fontsize=9, fontweight="bold", color="darkred")
    ax_macro.grid(True, linestyle=":", alpha=0.5)
    ax_macro.set_ylabel("Val", fontsize=8)
    ax_macro.set_xlabel("Historical Timesteps (Past Reference)", fontsize=8)
    
    # bottom panel: micro target window
    for i, d in enumerate(dims):
        ax_micro.plot(range(w_len), local_data[:, d], label=f"Dim {d}", color=colors[i], linewidth=1.2)
        ax_micro.axhline(base_mean[d], color=colors[i], linestyle="--", linewidth=0.8, alpha=0.6)
        if base_std is not None:
            std_d = base_std[d]
            ax_micro.fill_between(range(w_len), base_mean[d] - 2.5 * std_d, base_mean[d] + 2.5 * std_d, color=colors[i], alpha=0.08)
            
    ax_micro.set_xlim(0, w_len)
    ax_micro.xaxis.set_major_locator(MultipleLocator(50))
    ax_micro.xaxis.set_minor_locator(MultipleLocator(10))
    ax_micro.grid(True, which="major", linestyle="--", alpha=0.6)
    ax_micro.grid(True, which="minor", linestyle=":", alpha=0.3)
    ax_micro.set_title(f"PANEL 2: TARGET WINDOW (Steps 0 to {w_len}, Shaded = Normal ±2.5σ Band, Excursions = Anomaly)", fontsize=9, fontweight="bold", color="darkgreen")
    ax_micro.set_xlabel(f"Relative Step Index within Target Window (0 - {w_len})", fontsize=8)
    ax_micro.set_ylabel("Val", fontsize=8)
    ax_micro.legend(loc="upper right", ncol=min(4, len(dims)), fontsize=7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close("all")
    return save_path


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
        
    try:
        parsed = json.loads(clean_str)
        if "reason" not in parsed and "observation" in parsed:
            parsed["reason"] = parsed["observation"]
    except Exception:
        is_anom_match = re.search(r'"is_anomaly"\s*:\s*(true|false)', clean_str, re.IGNORECASE)
        is_anom = (is_anom_match.group(1).lower() == "true") if is_anom_match else False
        
        likelihood_match = re.search(r'"anomaly_likelihood"\s*:\s*([0-9.]+)', clean_str)
        if likelihood_match:
            likelihood = float(likelihood_match.group(1))
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
        
        obs_match = re.search(r'"observation"\s*:\s*"([^"]*)', clean_str)
        reason_match = re.search(r'"reason"\s*:\s*"([^"]*)', clean_str)
        reason = obs_match.group(1) if obs_match else (reason_match.group(1) if reason_match else clean_str[:80])
        
        parsed = {
            "is_anomaly": is_anom,
            "anomaly_likelihood": likelihood,
            "anomalous_timesteps": times,
            "affected_dimensions": dims,
            "reason": reason
        }

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
    if parsed["is_anomaly"] and isinstance(timesteps, list) and len(timesteps) >= 2:
        t0 = int(np.clip(timesteps[0], 0, window_size))
        t1 = int(np.clip(timesteps[1], 0, window_size))
        if t0 < t1:
            valid_ts = [t0, t1]
        elif t1 < t0:
            valid_ts = [t1, t0]
        else:
            valid_ts = [max(0, t0 - 10), min(window_size, t0 + 10)]
    parsed["anomalous_timesteps"] = valid_ts
    
    if not parsed["is_anomaly"]:
        parsed["anomalous_timesteps"] = []
        parsed["affected_dimensions"] = []

    # construct dense temporal guidance mask
    dense_mask = np.zeros(window_size, dtype=np.float32)
    if parsed["is_anomaly"] and len(valid_ts) == 2:
        dense_mask[valid_ts[0]:valid_ts[1]] = parsed["anomaly_likelihood"]
    parsed["teacher_dense_mask"] = torch.tensor(dense_mask, dtype=torch.float32)

    return parsed
