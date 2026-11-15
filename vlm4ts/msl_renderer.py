"""Fast C++ Agg memory renderer for NASA MSL rover telemetry.

Renders a 3-panel visual dashboard:
1. Panel 1: Primary Analog Telemetry (Channel 0) with nominal baseline bounds (+-3 sigma).
2. Panel 2: Command & Mode Activation Strip (Active discrete channels 1..54).
3. Panel 3: Instantaneous Deviation Energy and Rolling Volatility.

Zero disk I/O, zero PNG compression overhead, direct memory buffer extraction.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class MSLTelemetryRenderer:
    def __init__(self, base_m, base_s, active_discrete_channels=None, img_size=(1024, 768)):
        self.base_m = np.asarray(base_m, dtype=np.float32)
        self.base_s = np.asarray(base_s, dtype=np.float32)
        self.base_s[self.base_s == 0] = 1.0
        self.img_size = img_size
        
        if active_discrete_channels is None:
            self.active_discrete = [5, 6, 7, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 23, 27, 28, 29, 31, 33, 35, 39, 41, 43, 45, 46, 47, 48, 49, 53, 54]
        else:
            self.active_discrete = list(active_discrete_channels)

    def render(self, window_data):
        """Renders 200-step window into RGB PIL Image via direct memory buffer.
        
        Args:
            window_data: np.ndarray of shape (200, 55).
        Returns:
            PIL.Image (RGB, 1024x768).
        """
        dpi = 100
        fig_w = self.img_size[0] / dpi
        fig_h = self.img_size[1] / dpi
        
        fig, axes = plt.subplots(3, 1, figsize=(fig_w, fig_h), dpi=dpi,
                                 gridspec_kw={"height_ratios": [2.2, 1.2, 1.2]},
                                 constrained_layout=True)
        
        t = np.arange(len(window_data))
        ch0 = window_data[:, 0]
        m0 = self.base_m[0]
        s0 = self.base_s[0]
        
        # Panel 1: Analog Telemetry (ch0)
        ax0 = axes[0]
        ax0.plot(t, ch0, color="#1f77b4", linewidth=2.0, label="Analog Telemetry (Ch 0)")
        # Rolling 15-step mean
        if len(ch0) >= 15:
            roll_m = np.convolve(ch0, np.ones(15)/15, mode="same")
            ax0.plot(t, roll_m, color="#ff7f0e", linestyle="--", linewidth=1.5, label="15-step Trend")
            
        # Nominal bounds
        ax0.axhline(m0, color="#2ca02c", linestyle=":", linewidth=1.2, label=f"Nominal Mean ({m0:.2f})")
        ax0.fill_between(t, m0 - 3 * s0, m0 + 3 * s0, color="#2ca02c", alpha=0.15, label=r"Nominal $\pm 3\sigma$")
        ax0.axhline(m0 - 3 * s0, color="#d62728", linestyle="--", alpha=0.5, linewidth=1.0)
        ax0.axhline(m0 + 3 * s0, color="#d62728", linestyle="--", alpha=0.5, linewidth=1.0)
        
        ax0.set_title("NASA MSL Rover Telemetry: Analog Sensor Profile", fontsize=11, fontweight="bold")
        ax0.set_ylabel("Standardized Value", fontsize=9)
        ax0.set_xlim(0, len(window_data) - 1)
        ax0.grid(True, linestyle="--", alpha=0.3)
        ax0.legend(loc="upper right", fontsize=8, framealpha=0.8)
        
        # Panel 2: Command & Mode Activation Strip
        ax1 = axes[1]
        disc_data = window_data[:, self.active_discrete]
        # Sum of active commands over time
        cmd_activity = np.sum(disc_data > 0, axis=1)
        ax1.plot(t, cmd_activity, color="#9467bd", linewidth=1.8, label="Active Command Count")
        ax1.fill_between(t, 0, cmd_activity, color="#9467bd", alpha=0.25)
        
        # Overlay top 3 active individual channels
        col_list = ["#8c564b", "#e377c2", "#7f7f7f"]
        top_ch_idx = np.argsort(np.sum(disc_data > 0, axis=0))[::-1][:3]
        for idx, col in zip(top_ch_idx, col_list):
            ch_num = self.active_discrete[idx]
            if np.max(disc_data[:, idx]) > 0:
                ax1.step(t, disc_data[:, idx] * 0.8, color=col, linestyle=":", alpha=0.7, label=f"Cmd Ch {ch_num}")
                
        ax1.set_title("Spacecraft Command & Mode Activation Context", fontsize=10, fontweight="bold")
        ax1.set_ylabel("Active Commands", fontsize=9)
        ax1.set_xlim(0, len(window_data) - 1)
        ax1.grid(True, linestyle="--", alpha=0.3)
        ax1.legend(loc="upper right", fontsize=8, framealpha=0.8)
        
        # Panel 3: Deviation Energy & Rolling Volatility
        ax2 = axes[2]
        z_sq = ((ch0 - m0) / s0) ** 2
        ax2.bar(t, z_sq, color="#d62728", alpha=0.6, width=1.0, label=r"Deviation Energy $z_0^2$")
        ax2.axhline(9.0, color="#d62728", linestyle="--", linewidth=1.2, label=r"$3\sigma$ Threshold ($z^2=9$)")
        
        ax2.set_title("Telemetry Deviation Energy & Variance Anomaly", fontsize=10, fontweight="bold")
        ax2.set_xlabel("Window Timestep (t)", fontsize=9)
        ax2.set_ylabel(r"$z^2$ Score", fontsize=9)
        ax2.set_xlim(0, len(window_data) - 1)
        ax2.grid(True, linestyle="--", alpha=0.3)
        ax2.legend(loc="upper right", fontsize=8, framealpha=0.8)
        
        # Direct buffer copy
        canvas = fig.canvas
        canvas.draw()
        buf = np.asarray(canvas.buffer_rgba())
        img = Image.fromarray(buf[:, :, :3]).copy()
        plt.close(fig)
        return img

    def render_window(self, window_data):
        """Returns rendered RGB image as np.ndarray (H, W, 3)."""
        return np.asarray(self.render(window_data))
