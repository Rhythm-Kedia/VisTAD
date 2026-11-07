"""Fast C++ Agg memory renderer for NASA MSL rover telemetry.

Renders a 4-panel visual dashboard (600x600 RGB):
1. Panel 1: Primary Analog Telemetry (Channel 0) with nominal baseline bounds (+-3 sigma).
2. Panel 2: Command & Mode Activation Strip (Active discrete channels 1..54).
3. Panel 3: Rolling Volatility / Standard Deviation across time.
4. Panel 4: Telemetry Deviation Energy (z_0^2) bar chart.

Zero disk I/O, zero PNG compression overhead, direct memory buffer extraction.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class MSLTelemetryRenderer:
    def __init__(self, base_m: np.ndarray, base_s: np.ndarray, active_discrete_channels=None, dpi: int = 100):
        self.base_m = np.asarray(base_m, dtype=np.float32)
        self.base_s = np.asarray(base_s, dtype=np.float32)
        self.base_s[self.base_s == 0] = 1.0
        self.dpi = dpi
        
        if active_discrete_channels is None:
            self.active_discrete = [5, 6, 7, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 23, 27, 28, 29, 31, 33, 35, 39, 41, 43, 45, 46, 47, 48, 49, 53, 54]
        else:
            self.active_discrete = list(active_discrete_channels)

    def render_window(self, window_data: np.ndarray) -> np.ndarray:
        """Renders 200-step window into RGB numpy array (H, W, 3)."""
        t = np.arange(len(window_data))
        ch0 = window_data[:, 0]
        m0 = self.base_m[0]
        s0 = self.base_s[0]
        z0 = (ch0 - m0) / s0

        fig, axes = plt.subplots(2, 2, figsize=(6, 6), dpi=self.dpi)

        # Panel 1: Analog Telemetry (ch0)
        ax1 = axes[0, 0]
        ax1.plot(t, z0, color="#1f77b4", linewidth=1.5, label="Telemetry (z)")
        if len(z0) >= 15:
            roll_m = np.convolve(z0, np.ones(15)/15, mode="same")
            ax1.plot(t, roll_m, color="#ff7f0e", linestyle="--", linewidth=1.2, label="Trend")
        ax1.axhline(3.0, color="crimson", linestyle="--", alpha=0.6, lw=0.8)
        ax1.axhline(-3.0, color="crimson", linestyle="--", alpha=0.6, lw=0.8)
        ax1.set_title("Panel 1: Analog Telemetry Profile", fontsize=8, fontweight="bold")
        ax1.set_ylim(-6, 6)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=6, loc="upper right")

        # Panel 2: Command & Mode Activation Context
        ax2 = axes[0, 1]
        disc_data = window_data[:, self.active_discrete]
        cmd_activity = np.sum(disc_data > 0, axis=1)
        ax2.plot(t, cmd_activity, color="#9467bd", linewidth=1.4, label="Active Cmds")
        ax2.fill_between(t, 0, cmd_activity, color="#9467bd", alpha=0.25)
        ax2.set_title("Panel 2: Command Context", fontsize=8, fontweight="bold")
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=6, loc="upper right")

        # Panel 3: Rolling Volatility (w=25)
        ax3 = axes[1, 0]
        w = 25
        c1 = np.cumsum(np.insert(z0, 0, 0))
        c2 = np.cumsum(np.insert(z0**2, 0, 0))
        m = (c1[w:] - c1[:-w]) / w
        v = (c2[w:] - c2[:-w]) / w - m**2
        r_std = np.sqrt(np.maximum(0, v))
        pad = w // 2
        padded_std = np.pad(r_std, (pad, len(z0) - len(r_std) - pad), mode="edge")
        ax3.plot(t, padded_std, color="#2ca02c", linewidth=1.4, label="Rolling Volatility")
        ax3.axhline(1.5, color="crimson", linestyle="--", alpha=0.6, lw=0.8)
        ax3.set_title("Panel 3: Volatility & Freezes", fontsize=8, fontweight="bold")
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=6, loc="upper right")

        # Panel 4: Telemetry Deviation Energy (z_0^2)
        ax4 = axes[1, 1]
        z_sq = z0 ** 2
        ax4.plot(t, z_sq, color="#d62728", linewidth=1.2, label="Energy (z0^2)")
        ax4.axhline(9.0, color="crimson", linestyle="--", alpha=0.6, lw=0.8, label="3-Sigma Threshold")
        ax4.set_title("Panel 4: Deviation Energy", fontsize=8, fontweight="bold")
        ax4.set_ylim(0, max(25.0, float(np.percentile(z_sq, 98)) * 1.2))
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=6, loc="upper right")

        fig.subplots_adjust(left=0.10, right=0.95, top=0.92, bottom=0.08, hspace=0.35, wspace=0.35)
        fig.canvas.draw()
        rgba_buffer = fig.canvas.buffer_rgba()
        img_array = np.asarray(rgba_buffer)[:, :, :3]
        plt.close(fig)
        return img_array
