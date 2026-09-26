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
    "You are an expert server telemetry and SRE analyst performing precision anomaly detection.\n"
    "You are inspecting a 2-panel line plot of z-normalized metric channels (0 = nominal baseline mean, ±1 = 1 std dev, red dashed lines = ±3 std dev threshold):\n"
    "- PANEL 1 (TOP): Prior historical context over past 800 steps (baseline reference).\n"
    "- PANEL 2 (BOTTOM): Candidate target window under inspection (relative steps 0 to 200).\n\n"
    "DIAGNOSTIC PRINCIPLES:\n"
    "1. NORMAL BEHAVIOR (anomaly_confidence: 1, anomaly_likelihood: 0.00 - 0.15):\n"
    "   - All signals fluctuate within nominal range (comfortably inside the red dashed ±3σ lines).\n"
    "   - Periodic pulses or spikes that ALSO appear regularly in Panel 1 (benign background operations/heartbeats).\n"
    "2. ANOMALY (anomaly_confidence: 2 or 3, anomaly_likelihood: 0.75 - 1.00):\n"
    "   - Any channel breaching the red dashed ±3σ lines with a NEW spike, plateau, or offset NOT present in Panel 1.\n"
    "   - Coordinated multi-channel step shifts, persistent baseline offsets, or turbulence that disrupts the historical pattern.\n"
    "   - Sustained elevation or drop continuing across the window.\n\n"
    "Respond ONLY with a valid JSON object with NO extra text or markdown:\n"
    "{\n"
    "  \"anomaly_confidence\": <1 (normal) or 2/3 (anomaly)>,\n"
    "  \"is_anomaly\": <true if anomaly_confidence >= 2, else false>,\n"
    "  \"anomaly_likelihood\": <float 0.00 to 1.00>,\n"
    "  \"affected_dimensions\": [\"Dim X\", ...],\n"
    "  \"anomalous_timesteps\": [<start_step>, <end_step>],\n"
    "  \"temporal_continuity_analysis\": \"<concise reason under 8 words>\"\n"
    "}"
)


def render_dual_horizon_image(data, start_idx, end_idx, dims, base_mean=None, base_std=None, save_path=None, ctx_len=800):
    """Renders dual-horizon canvas where Panel 1 is strictly prior history and curves are uncluttered."""
    ctx_start = max(0, start_idx - ctx_len)
    # strictly prior history up to start_idx to prevent target window leakage into the reference panel
    ctx_data = data[ctx_start:start_idx]
    local_data = data[start_idx:end_idx]
    w_len = len(local_data)
    
    # z-normalize per-channel using baseline statistics to highlight anomalies
    if base_mean is not None:
        mu = base_mean[dims]
    else:
        mu = np.mean(ctx_data[:, dims], axis=0)

    if base_std is not None:
        sigma = base_std[dims]
    else:
        sigma = np.std(ctx_data[:, dims], axis=0)

    sigma = np.where(sigma < 1e-4, 1.0, sigma)
    ctx_plot = (ctx_data[:, dims] - mu) / sigma
    local_plot = (local_data[:, dims] - mu) / sigma

    fig, (ax_macro, ax_micro) = plt.subplots(2, 1, figsize=(9.0, 5.2), dpi=100, gridspec_kw={"height_ratios": [1, 1.3]})
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(dims), 1)))
    
    # top panel: strictly prior historical context (reference baseline)
    for i, d in enumerate(dims):
        ax_macro.plot(range(len(ctx_data)), ctx_plot[:, i], label=f"Dim {d}", color=colors[i], linewidth=0.9, alpha=0.85)
    ax_macro.set_xlim(0, max(1, len(ctx_data)))
    ax_macro.set_title("PANEL 1: PRIOR HISTORICAL CONTEXT (Past 800 Steps BEFORE Target Window - Baseline)", fontsize=9, fontweight="bold", color="navy")
    ax_macro.grid(True, linestyle=":", alpha=0.5)
    ax_macro.set_ylabel("Z-Score (std dev)", fontsize=8)
    ax_macro.set_xlabel("Historical Time Steps Prior to Target Window", fontsize=8)
    ax_macro.legend(loc="upper right", ncol=min(6, len(dims)), fontsize=7)
    
    # bottom panel: candidate target window under inspection
    for i, d in enumerate(dims):
        ax_micro.plot(range(w_len), local_plot[:, i], label=f"Dim {d}", color=colors[i], linewidth=1.3)
    ax_micro.set_xlim(0, w_len)
    ax_micro.xaxis.set_major_locator(MultipleLocator(50))
    ax_micro.xaxis.set_minor_locator(MultipleLocator(10))
    ax_micro.grid(True, which="major", linestyle="--", alpha=0.6)
    ax_micro.grid(True, which="minor", linestyle=":", alpha=0.3)
    ax_micro.set_title(f"PANEL 2: CANDIDATE TARGET WINDOW UNDER INSPECTION (Steps 0 to {w_len})", fontsize=9, fontweight="bold", color="darkgreen")
    ax_micro.set_xlabel(f"Relative Step Index within Target Window (0 - {w_len})", fontsize=8)
    ax_micro.set_ylabel("Z-Score (std dev)", fontsize=8)

    # visual anomaly threshold anchors (0 mean and ±3 sigma lines)
    for ax in (ax_macro, ax_micro):
        ax.axhline(0.0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
        ax.axhline(3.0, color="crimson", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.axhline(-3.0, color="crimson", linestyle="--", linewidth=0.8, alpha=0.6)
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(min(y_min, -4.0), max(y_max, 4.0))
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    img.load()
    buf.close()
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
