"""Subsystem-Coupled Causal Dilated TCN for GECCO Drinking Water Anomaly Detection.

Couples 2 physical water treatment subsystems across 9 telemetry channels:
- Subsystem A (Chemical Disinfection - 4 channels): Redox, pH, Cl, Cl_2
- Subsystem B (Hydraulic & Physical Quality - 5 channels): Leit, Trueb, Fm, Fm_2, Tp

Receptive Field:
- k=5, dilations=[1, 2, 4, 8, 16] -> L_RF = 1 + 4 * (1 + 2 + 4 + 8 + 16) = 125 steps.
- Perfectly matched to GECCO's 18-29 step transient contamination pulses without lookahead leakage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Channel mappings matching GECCO_SENSORS = ["Tp", "Cl", "pH", "Redox", "Leit", "Trueb", "Cl_2", "Fm", "Fm_2"]
SUB_A_INDICES = [3, 2, 1, 6]  # Redox (3), pH (2), Cl (1), Cl_2 (6)
SUB_B_INDICES = [4, 5, 7, 8, 0]  # Leit (4), Trueb (5), Fm (7), Fm_2 (8), Tp (0)


class CausalConv1d(nn.Module):
    """1D causal convolution with left-padding to prevent lookahead leakage."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class CausalResidualBlock(nn.Module):
    """Dilated causal residual block with LayerNorm and GELU."""

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


class SubsystemEncoder(nn.Module):
    """Encoder for a physical plant subsystem with causal dilated convolutions."""

    def __init__(self, in_channels: int, hidden_dim: int = 64, kernel_size: int = 5,
                 dilations: list[int] | None = None, dropout: float = 0.1):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8, 16]  # L_RF = 125
        self.input_proj = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList([
            CausalResidualBlock(hidden_dim, kernel_size=kernel_size, dilation=d, dropout=dropout)
            for d in dilations
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_channels, L]
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return h


class GeccoSubsystemCoupledTCN(nn.Module):
    """Subsystem-Coupled Causal TCN for GECCO drinking water quality anomaly detection."""

    def __init__(self, in_channels: int = 9, hidden_dim: int = 64, teacher_feat_dim: int = 1024,
                 kernel_size: int = 5, dilations: list[int] | None = None, dropout: float = 0.1, **kwargs):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8, 16]

        self.sub_a_indices = SUB_A_INDICES
        self.sub_b_indices = SUB_B_INDICES

        # Subsystem encoders
        self.encoder_a = SubsystemEncoder(len(self.sub_a_indices), hidden_dim, kernel_size, dilations, dropout)
        self.encoder_b = SubsystemEncoder(len(self.sub_b_indices), hidden_dim, kernel_size, dilations, dropout)

        # Cross-subsystem fusion (128 -> 64)
        self.fusion_proj = nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1)
        self.fusion_blocks = nn.ModuleList([
            CausalResidualBlock(hidden_dim, kernel_size=3, dilation=d, dropout=dropout)
            for d in [1, 2]
        ])

        # Point-wise dense temporal anomaly head (64 -> 1)
        self.dense_head = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim // 2, 1, kernel_size=1)
        )

        # Global feature distillation head (matches DINOv2-Large 1024-dim embedding)
        self.feat_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, teacher_feat_dim)
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Handle input shape: [B, L, C] -> [B, C, L]
        if x.dim() == 3 and x.shape[1] != 9 and x.shape[2] == 9:
            x = x.transpose(1, 2)
        elif x.dim() == 2:
            x = x.unsqueeze(0).transpose(1, 2)

        # Split into physical subsystems
        x_a = x[:, self.sub_a_indices, :]
        x_b = x[:, self.sub_b_indices, :]

        # Encode subsystems
        h_a = self.encoder_a(x_a)
        h_b = self.encoder_b(x_b)

        # Cross-subsystem fusion
        h_cat = torch.cat([h_a, h_b], dim=1)
        h_fused = self.fusion_proj(h_cat)
        for fb in self.fusion_blocks:
            h_fused = fb(h_fused)

        # Dense point-wise prediction: probabilities in [0, 1]
        dense_logits = self.dense_head(h_fused).squeeze(1)  # [B, L]
        pred_dense = torch.sigmoid(dense_logits)           # [B, L]

        # Global window anomaly score: max-pool across time
        window_score, _ = torch.max(pred_dense, dim=-1)    # [B]

        # Feature representation pooled across time: normalized to unit sphere
        h_pool = torch.mean(h_fused, dim=-1)
        feat = self.feat_head(h_pool)                      # [B, 1024]
        student_feat = F.normalize(feat, p=2, dim=-1)      # [B, 1024]

        return {
            "pred_dense": pred_dense,
            "window_score": window_score,
            "logits": window_score,
            "pred_feature": student_feat,
            "student_feat": student_feat,
            "z_student": student_feat
        }
