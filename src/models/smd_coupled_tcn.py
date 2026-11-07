"""Multi-Subsystem Causal Dilated TCN Student for Server Machine Dataset (SMD).

Ingests all 38 native server metrics across 4 coupled infrastructure subsystems:
- Subsystem 1: CPU Utilization & Process Dynamics (8 channels: 0..7)
- Subsystem 2: Memory Allocation & Page Faults (7 channels: 8..14)
- Subsystem 3: Network Interface Traffic & Socket Drops (12 channels: 15..26)
- Subsystem 4: Disk Block Storage & I/O Wait (11 channels: 27..37)

Temporal Receptive Field:
- k=5, dilations=[1, 2, 4, 8, 16] -> L_RF = 1 + 4 * (1 + 2 + 4 + 8 + 16) = 125 steps.
- Covers transient spikes (2-10 steps) and sustained saturation plateaus (400-700 steps) with zero future leakage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 38-Channel Subsystem Index Partitions
SUB_CPU_INDICES = list(range(0, 8))     # 8 channels: CPU user, system, idle, load, context switches
SUB_MEM_INDICES = list(range(8, 15))    # 7 channels: Mem used, free, buff, cache, swap, page faults
SUB_NET_INDICES = list(range(15, 27))   # 12 channels: In/out bytes, packets, drops, errors (eth0/eth1)
SUB_DISK_INDICES = list(range(27, 38))  # 11 channels: Read/write reqs, merged, sectors, I/O wait time


class CausalConv1d(nn.Module):
    """1D causal convolution with replicate left-padding to prevent edge saturation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        return self.conv(F.pad(x, (self.pad, 0), mode="replicate"))


class CausalResidualBlock(nn.Module):
    """Dilated causal residual block with GroupNorm and GELU."""

    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation=dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation=dilation)
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.drop(out)
        out = self.norm2(self.conv2(out))
        return self.act(out + res)


class ServerSubsystemEncoder(nn.Module):
    """Encoder for a specific server subsystem with causal dilated convolutions."""

    def __init__(self, in_channels: int, hidden_dim: int = 32, kernel_size: int = 5,
                 dilations: list[int] | None = None, dropout: float = 0.1):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8, 16]  # L_RF = 125
        self.input_proj = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList([
            CausalResidualBlock(hidden_dim, kernel_size=kernel_size, dilation=d, dropout=dropout)
            for d in dilations
        ])
        self.head = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim // 2, 1, kernel_size=1)
        )
        # Initialize output bias negative (-2.2) so initial anomaly probability is ~0.10
        nn.init.constant_(self.head[-1].bias, -2.2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, in_channels, L]
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        p = torch.sigmoid(self.head(h).squeeze(1))  # [B, L]
        return p, h


class CausalServerPhysicalPrior(nn.Module):
    """Causal physical shock prior evaluating high-pass deviations across server subsystems."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 38, L] - normalized server telemetry in [0, 1]
        pad_30 = F.pad(x, (29, 0), mode="replicate")
        causal_avg = F.avg_pool1d(pad_30, kernel_size=30, stride=1)
        diff = torch.abs(x - causal_avg)  # [B, 38, L]

        # Subsystem shocks
        shock_cpu = torch.max(diff[:, 0:8, :], dim=1)[0]
        shock_mem = torch.max(diff[:, 8:15, :], dim=1)[0]
        shock_net = torch.max(diff[:, 15:27, :], dim=1)[0]
        shock_disk = torch.max(diff[:, 27:38, :], dim=1)[0]

        shock_max = torch.maximum(
            torch.maximum(shock_cpu, shock_mem),
            torch.maximum(shock_net, shock_disk)
        )  # [B, L]

        # Calibrated probability: shocks > 0.35 trigger physical alarm
        p_prior = torch.sigmoid(15.0 * (shock_max - 0.35))
        return p_prior


class SMDSubsystemCoupledTCN(nn.Module):
    """Multi-Subsystem Causal Dilated TCN for Server Machine Dataset."""

    def __init__(self, in_channels: int = 38, hidden_dim: int = 32, teacher_feat_dim: int = 1024,
                 kernel_size: int = 5, dilations: list[int] | None = None, dropout: float = 0.1):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8, 16]

        self.idx_cpu = SUB_CPU_INDICES      # 8 ch
        self.idx_mem = SUB_MEM_INDICES      # 7 ch
        self.idx_net = SUB_NET_INDICES      # 12 ch
        self.idx_disk = SUB_DISK_INDICES    # 11 ch

        # Each subsystem stream takes (channels * 3) features: [x, delta_1, diff_30]
        self.enc_cpu = ServerSubsystemEncoder(len(self.idx_cpu) * 3, hidden_dim, kernel_size, dilations, dropout)
        self.enc_mem = ServerSubsystemEncoder(len(self.idx_mem) * 3, hidden_dim, kernel_size, dilations, dropout)
        self.enc_net = ServerSubsystemEncoder(len(self.idx_net) * 3, hidden_dim, kernel_size, dilations, dropout)
        self.enc_disk = ServerSubsystemEncoder(len(self.idx_disk) * 3, hidden_dim, kernel_size, dilations, dropout)

        self.physical_prior = CausalServerPhysicalPrior()

        # Latent projection to 1024-dim student feature
        self.feat_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, teacher_feat_dim)
        )

    def extract_stream_features(self, x_sub: torch.Tensor) -> torch.Tensor:
        # x_sub: [B, C, L]
        pad_1 = F.pad(x_sub, (1, 0), mode="replicate")
        delta = pad_1[:, :, 1:] - pad_1[:, :, :-1]

        pad_30 = F.pad(x_sub, (29, 0), mode="replicate")
        causal_avg = F.avg_pool1d(pad_30, kernel_size=30, stride=1)
        diff_avg = x_sub - causal_avg

        return torch.cat([x_sub, delta, diff_avg], dim=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Handle input shape: [B, L, C] -> [B, C, L]
        if x.dim() == 3 and x.shape[1] != 38 and x.shape[2] == 38:
            x = x.transpose(1, 2)
        elif x.dim() == 2:
            x = x.unsqueeze(0).transpose(1, 2)

        # 1. Extract Triplet Features for Each Subsystem
        f_cpu = self.extract_stream_features(x[:, self.idx_cpu, :])
        f_mem = self.extract_stream_features(x[:, self.idx_mem, :])
        f_net = self.extract_stream_features(x[:, self.idx_net, :])
        f_disk = self.extract_stream_features(x[:, self.idx_disk, :])

        # 2. Causal Dilated Encoders
        p_cpu, h_cpu = self.enc_cpu(f_cpu)
        p_mem, h_mem = self.enc_mem(f_mem)
        p_net, h_net = self.enc_net(f_net)
        p_disk, h_disk = self.enc_disk(f_disk)

        # 3. Causal Physical Prior
        p_prior = self.physical_prior(x)

        # 4. Probabilistic Physical Disjunction (OR Fusion across 4 server subsystems)
        # Server anomaly occurs if CPU saturates OR Memory leaks OR Network fails OR Disk stalls
        pred_dense = 1.0 - (1.0 - p_cpu) * (1.0 - p_mem) * (1.0 - p_net) * (1.0 - p_disk)

        # 5. Global Window Anomaly Score
        window_score = torch.max(pred_dense, dim=-1)[0]

        # 6. Global Feature Embedding
        h_cat = torch.cat([h_cpu, h_mem, h_net, h_disk], dim=1)
        h_pool = torch.mean(h_cat, dim=-1)
        feat = self.feat_head(h_pool)
        student_feat = F.normalize(feat, p=2, dim=-1)

        return {
            "pred_dense": pred_dense,
            "p_cpu": p_cpu,
            "p_mem": p_mem,
            "p_net": p_net,
            "p_disk": p_disk,
            "p_prior": p_prior,
            "window_score": window_score,
            "logits": window_score,
            "student_feat": student_feat,
            "z_student": student_feat
        }
