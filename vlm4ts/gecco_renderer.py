"""Optimized high-speed visual renderer for GECCO drinking water quality anomaly detection.

Renders dual-subsystem 4-panel dashboards directly from matplotlib Agg RGBA memory
buffers without PNG compression bottlenecks.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GECCO_SENSORS = ["Tp", "Cl", "pH", "Redox", "Leit", "Trueb", "Cl_2", "Fm", "Fm_2"]
SUB_A_COLS = ["Redox", "pH", "Cl", "Cl_2"]
SUB_B_QUALITY = ["Leit", "Trueb"]
SUB_B_HYDRAULIC = ["Fm", "Fm_2", "Tp"]
QUALITY_COLS = ["Redox", "pH", "Cl_2", "Cl", "Leit", "Trueb"]

COLOR_MAP = {
    "Redox": "#d95f02",
    "pH": "#1f78b4",
    "Cl": "#33a02c",
    "Cl_2": "#b2df8a",
    "Leit": "#7570b3",
    "Trueb": "#e7298a",
    "Fm": "#666666",
    "Fm_2": "#999999",
    "Tp": "#e6ab02"
}


class GeccoDashboardRenderer:
    """Renders 4-panel multi-subsystem water telemetry dashboards into RGB numpy arrays."""

    def __init__(self, base_mean: np.ndarray, base_std: np.ndarray, dpi: int = 100):
        self.sensor_to_idx = {name: i for i, name in enumerate(GECCO_SENSORS)}
        self.base_mean = base_mean.astype(np.float32)
        self.base_std = (base_std + 1e-6).astype(np.float32)
        self.dpi = dpi

    def render_window(self, window_data: np.ndarray) -> np.ndarray:
        """Renders window_data (L, 9) into RGB array (H, W, 3)."""
        z_scores = (window_data - self.base_mean) / self.base_std
        t_len = len(window_data)
        t_axis = np.arange(t_len)

        fig, axes = plt.subplots(2, 2, figsize=(6, 6), dpi=self.dpi)

        # Panel 1: Subsystem A (Chemical Disinfection: Redox, pH, Cl, Cl_2)
        ax1 = axes[0, 0]
        for col in SUB_A_COLS:
            idx = self.sensor_to_idx[col]
            ax1.plot(t_axis, z_scores[:, idx], label=col, color=COLOR_MAP[col], lw=1.2)
        ax1.axhline(3.0, color="crimson", linestyle="--", alpha=0.5, lw=0.8)
        ax1.axhline(-3.0, color="crimson", linestyle="--", alpha=0.5, lw=0.8)
        ax1.set_title("Panel 1: Chemical Equilibrium", fontsize=8, fontweight="bold")
        ax1.set_ylim(-6, 6)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=6, loc="upper right")

        # Panel 2: Physical Water Quality (Conductivity & Turbidity)
        ax2 = axes[1, 0]
        for col in SUB_B_QUALITY:
            idx = self.sensor_to_idx[col]
            ax2.plot(t_axis, z_scores[:, idx], label=col, color=COLOR_MAP[col], lw=1.2)
        ax2.axhline(3.0, color="crimson", linestyle="--", alpha=0.5, lw=0.8)
        ax2.axhline(-3.0, color="crimson", linestyle="--", alpha=0.5, lw=0.8)
        ax2.set_title("Panel 2: Physical Quality (Leit/Trueb)", fontsize=8, fontweight="bold")
        ax2.set_ylim(-6, 6)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=6, loc="upper right")

        # Panel 3: Operational & Hydraulic Context (Fm, Fm_2, Tp)
        ax3 = axes[0, 1]
        for col in SUB_B_HYDRAULIC:
            idx = self.sensor_to_idx[col]
            ax3.plot(t_axis, z_scores[:, idx], label=col, color=COLOR_MAP[col], lw=1.0)
        ax3.set_title("Panel 3: Hydraulic & Thermal Context", fontsize=8, fontweight="bold")
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=6, loc="upper right")

        # Panel 4: Water Quality Deviation Energy (z_i^2)
        ax4 = axes[1, 1]
        energies = []
        bar_colors = []
        for col in QUALITY_COLS:
            idx = self.sensor_to_idx[col]
            energies.append(np.mean(z_scores[:, idx] ** 2))
            bar_colors.append(COLOR_MAP[col])
        ax4.bar(range(len(QUALITY_COLS)), energies, color=bar_colors, alpha=0.85)
        ax4.axhline(9.0, color="crimson", linestyle="--", alpha=0.6, label="3σ Bound")
        ax4.set_xticks(range(len(QUALITY_COLS)))
        ax4.set_xticklabels(QUALITY_COLS, rotation=45, fontsize=6)
        ax4.set_title("Panel 4: Water Quality Deviation Energy", fontsize=8, fontweight="bold")
        ax4.set_ylim(0, max(25.0, float(np.max(energies)) * 1.15))
        ax4.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.canvas.draw()
        rgba_buffer = fig.canvas.buffer_rgba()
        img_array = np.asarray(rgba_buffer)[:, :, :3]
        plt.close(fig)
        return img_array
