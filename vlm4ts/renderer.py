import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from PIL import Image


def detect_channel_types(data, dims):
    """Categorizes channels into discrete actuators (few unique states) vs continuous analog sensors."""
    discrete_dims = []
    analog_dims = []
    for d in dims:
        unique_vals = np.unique(data[:, d])
        if len(unique_vals) <= 5:
            discrete_dims.append(d)
        else:
            analog_dims.append(d)
    return analog_dims, discrete_dims


def render_vlm4ts_image(data, start_idx, end_idx, dims, base_mean=None, base_std=None,
                        ctx_len=800, px=(672, 392), dpi=100, channel_names=None):
    """Multi-scale, multi-panel visual renderer for VLM4TS time-series inspection.
    
    Separates continuous analog sensors from discrete actuators to eliminate visual
    collision, and displays a historical macro baseline alongside the micro target window.
    """
    ctx_start = max(0, start_idx - ctx_len)
    ctx_data = data[ctx_start:start_idx]
    local_data = data[start_idx:end_idx]
    w_len = len(local_data)
    n_ctx = len(ctx_data)

    if base_mean is None:
        base_mean = np.mean(data[:min(3000, len(data))], axis=0)
    if base_std is None:
        base_std = np.maximum(np.std(data[:min(3000, len(data))], axis=0), 0.05)

    analog_dims, discrete_dims = detect_channel_types(data, dims)
    if not analog_dims:
        analog_dims = dims[:min(4, len(dims))]
        discrete_dims = dims[min(4, len(dims)):]

    # Standardize analog channels
    mu_a = base_mean[analog_dims]
    sig_a = np.maximum(base_std[analog_dims], 0.05)
    ctx_analog = (ctx_data[:, analog_dims] - mu_a) / sig_a if n_ctx > 0 else np.zeros((1, len(analog_dims)))
    loc_analog = (local_data[:, analog_dims] - mu_a) / sig_a

    fig = plt.figure(figsize=(px[0] / dpi, px[1] / dpi), dpi=dpi)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 0.8], width_ratios=[1, 1.4], hspace=0.35, wspace=0.15)

    ax_macro_a = fig.add_subplot(gs[0, 0])
    ax_micro_a = fig.add_subplot(gs[0, 1], sharey=ax_macro_a)
    ax_macro_d = fig.add_subplot(gs[1, 0])
    ax_micro_d = fig.add_subplot(gs[1, 1], sharey=ax_macro_d)

    colors = plt.cm.tab10(np.arange(len(dims)) % 10)

    # 1. Analog Macro Baseline
    if n_ctx > 0:
        for idx, d in enumerate(analog_dims):
            name = channel_names[d] if channel_names and d < len(channel_names) else f"Dim {d}"
            ax_macro_a.plot(np.arange(-n_ctx, 0), ctx_analog[:, idx], color=colors[idx % 10],
                            linewidth=1.0, alpha=0.75, label=name)
        ax_macro_a.set_xlim(-ctx_len, 0)
    ax_macro_a.set_title("MACRO BASELINE (Prior Context)", fontsize=8, fontweight="bold", color="navy")
    ax_macro_a.set_ylabel("Standardized (z-score)", fontsize=7)
    ax_macro_a.grid(True, linestyle=":", alpha=0.4)
    ax_macro_a.axhspan(-2, 2, color="green", alpha=0.06)

    # 2. Analog Micro Target Window
    for idx, d in enumerate(analog_dims):
        name = channel_names[d] if channel_names and d < len(channel_names) else f"Dim {d}"
        ax_micro_a.plot(np.arange(w_len), loc_analog[:, idx], color=colors[idx % 10],
                        linewidth=1.4, label=name)
    ax_micro_a.set_xlim(0, w_len)
    ax_micro_a.set_title(f"MICRO TARGET WINDOW (Steps 0-{w_len})", fontsize=8, fontweight="bold", color="darkgreen")
    ax_micro_a.grid(True, linestyle=":", alpha=0.4)
    ax_micro_a.axhspan(-2, 2, color="green", alpha=0.06)
    ax_micro_a.xaxis.set_major_locator(MultipleLocator(50))
    ax_micro_a.xaxis.set_minor_locator(MultipleLocator(10))
    if len(analog_dims) <= 6:
        ax_micro_a.legend(loc="upper right", fontsize=6, framealpha=0.6)

    # 3. Discrete / Actuator Macro & Micro Panels
    if discrete_dims:
        for idx, d in enumerate(discrete_dims):
            c_idx = (len(analog_dims) + idx) % 10
            name = channel_names[d] if channel_names and d < len(channel_names) else f"Actuator {d}"
            if n_ctx > 0:
                ax_macro_d.step(np.arange(-n_ctx, 0), ctx_data[:, d], color=colors[c_idx],
                                linewidth=1.1, alpha=0.8, where="post")
            ax_micro_d.step(np.arange(w_len), local_data[:, d], color=colors[c_idx],
                            linewidth=1.3, label=name, where="post")
        ax_macro_d.set_xlim(-ctx_len, 0)
        ax_macro_d.set_ylabel("Actuator State", fontsize=7)
        ax_macro_d.grid(True, linestyle=":", alpha=0.4)
        ax_micro_d.set_xlim(0, w_len)
        ax_micro_d.set_xlabel("Time Step in Window", fontsize=8)
        ax_micro_d.grid(True, linestyle=":", alpha=0.4)
        ax_micro_d.xaxis.set_major_locator(MultipleLocator(50))
        if len(discrete_dims) <= 4:
            ax_micro_d.legend(loc="upper right", fontsize=6, framealpha=0.6)
    else:
        # If all channels are analog, use lower row for rate-of-change / gradient
        diff_loc = np.diff(loc_analog, axis=0, prepend=loc_analog[:1])
        diff_ctx = np.diff(ctx_analog, axis=0, prepend=ctx_analog[:1]) if n_ctx > 0 else np.zeros((1, len(analog_dims)))
        if n_ctx > 0:
            for idx in range(len(analog_dims)):
                ax_macro_d.plot(np.arange(-n_ctx, 0), diff_ctx[:, idx], color=colors[idx % 10],
                                linewidth=0.8, alpha=0.6)
        for idx in range(len(analog_dims)):
            ax_micro_d.plot(np.arange(w_len), diff_loc[:, idx], color=colors[idx % 10],
                            linewidth=1.0, alpha=0.85)
        ax_macro_d.set_xlim(-ctx_len, 0)
        ax_macro_d.set_ylabel("Rate of Change (Δz)", fontsize=7)
        ax_macro_d.grid(True, linestyle=":", alpha=0.4)
        ax_micro_d.set_xlim(0, w_len)
        ax_micro_d.set_xlabel("Time Step in Window", fontsize=8)
        ax_micro_d.grid(True, linestyle=":", alpha=0.4)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    return img
