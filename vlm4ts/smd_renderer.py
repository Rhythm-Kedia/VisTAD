"""Fast C++ Agg memory renderer for Server Machine Dataset (SMD) Telemetry.

Renders a 4-panel visual dashboard (600x600 RGB):
1. Panel 1: CPU Utilization & Process Dynamics (Channels 0..7)
2. Panel 2: Memory Allocation & Page Faults (Channels 8..14)
3. Panel 3: Network Interface Traffic & Socket Drops (Channels 15..26)
4. Panel 4: Disk Block Storage & I/O Wait Times (Channels 27..37)

Zero disk I/O, zero PNG compression overhead, direct memory buffer extraction.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# 38-Channel Subsystem Index Partitions
SUB_CPU_INDICES = list(range(0, 8))     # 8 ch
SUB_MEM_INDICES = list(range(8, 15))    # 7 ch
SUB_NET_INDICES = list(range(15, 27))   # 12 ch
SUB_DISK_INDICES = list(range(27, 38))  # 11 ch


class SMDTelemetryRenderer:
    """High-speed memory renderer for 38-channel server telemetry."""

    def __init__(self, dpi: int = 100):
        self.dpi = dpi
        # Predefined visual palettes for the 4 infrastructure subsystems
        self.cpu_colors = ["#1f77b4", "#aec7e8", "#2ca02c", "#98df8a", "#ff7f0e", "#ffbb78", "#9467bd", "#c5b0d5"]
        self.mem_colors = ["#d62728", "#ff9896", "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f"]
        self.net_colors = ["#17becf", "#9edae5", "#bcbd22", "#dbdb8d", "#393b79", "#5254a3", "#6b6ecf", "#9c9ede", "#637939", "#8ca252", "#b5cf6b", "#cedb9c"]
        self.disk_colors = ["#8c6d31", "#bd9e39", "#e7ba52", "#e7cb94", "#843c39", "#ad494a", "#d6616b", "#e7969c", "#7b4173", "#a55194", "#ce6dbd"]

    def render_window(self, window_data: np.ndarray) -> np.ndarray:
        """Renders 200-step x 38-channel window into RGB numpy array (600, 600, 3).
        
        Args:
            window_data: np.ndarray of shape [200, 38] or [38, 200]
        Returns:
            np.ndarray of shape [H, W, 3] with dtype uint8
        """
        if window_data.shape[0] == 38 and window_data.shape[1] != 38:
            data = window_data.T
        else:
            data = window_data

        t = np.arange(len(data))
        fig, axes = plt.subplots(2, 2, figsize=(6, 6), dpi=self.dpi)

        # Panel 1: CPU Subsystem (0..7)
        ax1 = axes[0, 0]
        for idx, ch in enumerate(SUB_CPU_INDICES):
            ax1.plot(t, data[:, ch], color=self.cpu_colors[idx % len(self.cpu_colors)], lw=1.0, alpha=0.85)
        # Highlight mean CPU load trend
        cpu_mean = np.mean(data[:, SUB_CPU_INDICES], axis=1)
        ax1.plot(t, cpu_mean, color="#000000", lw=1.8, linestyle="--", label="CPU Mean")
        ax1.set_title("Panel 1: CPU Dynamics (Ch 0-7)", fontsize=8, fontweight="bold")
        ax1.grid(True, alpha=0.25)
        ax1.set_ylim(-1.0, 5.0)

        # Panel 2: Memory & Paging Subsystem (8..14)
        ax2 = axes[0, 1]
        for idx, ch in enumerate(SUB_MEM_INDICES):
            ax2.plot(t, data[:, ch], color=self.mem_colors[idx % len(self.mem_colors)], lw=1.0, alpha=0.85)
        mem_used = data[:, 8]
        ax2.plot(t, mem_used, color="#800000", lw=1.6, label="Mem Used")
        ax2.set_title("Panel 2: Memory & Paging (Ch 8-14)", fontsize=8, fontweight="bold")
        ax2.grid(True, alpha=0.25)
        ax2.set_ylim(-1.0, 5.0)

        # Panel 3: Network Interface Subsystem (15..26)
        ax3 = axes[1, 0]
        for idx, ch in enumerate(SUB_NET_INDICES):
            ax3.plot(t, data[:, ch], color=self.net_colors[idx % len(self.net_colors)], lw=1.0, alpha=0.85)
        net_agg = np.max(data[:, SUB_NET_INDICES], axis=1)
        ax3.plot(t, net_agg, color="#004d40", lw=1.6, linestyle=":", label="Peak Net")
        ax3.set_title("Panel 3: Network Traffic (Ch 15-26)", fontsize=8, fontweight="bold")
        ax3.grid(True, alpha=0.25)
        ax3.set_ylim(-1.0, 5.0)

        # Panel 4: Disk Block Storage Subsystem (27..37)
        ax4 = axes[1, 1]
        for idx, ch in enumerate(SUB_DISK_INDICES):
            ax4.plot(t, data[:, ch], color=self.disk_colors[idx % len(self.disk_colors)], lw=1.0, alpha=0.85)
        disk_max = np.max(data[:, SUB_DISK_INDICES], axis=1)
        ax4.plot(t, disk_max, color="#4a148c", lw=1.6, label="Disk Stall")
        ax4.set_title("Panel 4: Disk I/O & Storage (Ch 27-37)", fontsize=8, fontweight="bold")
        ax4.grid(True, alpha=0.25)
        ax4.set_ylim(-1.0, 5.0)

        fig.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.08, hspace=0.32, wspace=0.28)
        fig.canvas.draw()
        rgba_buffer = fig.canvas.buffer_rgba()
        img_array = np.asarray(rgba_buffer)[:, :, :3]
        plt.close(fig)
        return img_array
