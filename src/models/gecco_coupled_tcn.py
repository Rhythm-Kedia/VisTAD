"""Subsystem-Decoupled Causal TCN with Physical Disjunction for GECCO Water Quality.

Couples 2 physical water treatment subsystems across 9 telemetry channels:
- Subsystem A (Chemical Disinfection - 3 channels): Redox (3), Cl (1), Cl_2 (6)
- Subsystem B (Mineral & Physical Quality - 3 channels): pH (2), Leit (4), Trueb (5)
- Context Channels (Hydraulic & Thermal - 3 channels): Tp (0), Fm (7), Fm_2 (8)

Receptive Field:
- k=5, dilations=[1, 2, 4, 8, 16] -> L_RF = 1 + 4 * (1 + 2 + 4 + 8 + 16) = 125 steps.
- Perfectly matched to GECCO's 18-29 step transient contamination pulses without lookahead leakage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Channel mappings matching GECCO_SENSORS = ["Tp", "Cl", "pH", "Redox", "Leit", "Trueb", "Cl_2", "Fm", "Fm_2"]
SUB_DISINFECT_INDICES = [3, 1, 6]  # Redox (3), Cl (1), Cl_2 (6)
SUB_QUAL_INDICES = [2, 4, 5]       # pH (2), Leit (4), Trueb (5)
SUB_CTX_INDICES = [0, 7, 8]        # Tp (0), Fm (7), Fm_2 (8)


class CausalConv1d(nn.Module):
    """1D causal convolution with left-padding to prevent lookahead leakage."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        return self.conv(F.pad(x, (self.pad, 0)))


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


class SubsystemEncoder(nn.Module):
    """Encoder for a physical plant subsystem with causal dilated convolutions."""

    def __init__(self, in_channels: int, hidden_dim: int = 48, kernel_size: int = 5,
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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, in_channels, L]
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        p = torch.sigmoid(self.head(h).squeeze(1))
        return p, h


class CausalWaterQualityPrior(nn.Module):
    """Causal physical equilibrium prior evaluating non-linear boundary shocks."""

    def __init__(self, base_m: torch.Tensor | None = None, base_s: torch.Tensor | None = None):
        super().__init__()
        # Expected nominal baselines from winter training partition:
        # Redox ~ 750 mV, Cl ~ 0.166, Cl_2 ~ 0.109, pH ~ 8.35, Leit ~ 211
        if base_m is not None:
            self.register_buffer("base_m", torch.as_tensor(base_m, dtype=torch.float32))
            self.register_buffer("base_s", torch.as_tensor(base_s, dtype=torch.float32))
        else:
            self.register_buffer("base_m", torch.tensor([6.70, 0.166, 8.35, 750.6, 211.3, 0.019, 0.109, 1522.0, 874.0]))
            self.register_buffer("base_s", torch.tensor([0.13, 0.007, 0.04, 1.80, 2.46, 0.004, 0.005, 164.0, 85.9]))

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        # x_norm: [B, 9, L]
        # Reconstruct unnormalized telemetry: raw = x_norm * base_s + base_m
        b_m = self.base_m.view(1, 9, 1)
        b_s = self.base_s.view(1, 9, 1)
        x_raw = x_norm * b_s + b_m

        redox = x_raw[:, 3, :]
        cl = x_raw[:, 1, :]
        ph = x_raw[:, 2, :]
        leit = x_raw[:, 4, :]
        trueb = x_raw[:, 5, :]
        cl_2 = x_raw[:, 6, :]

        # Causal moving baseline (30 steps) for high-pass filtration with replicate padding
        pad_30_ph = F.pad(ph.unsqueeze(1), (29, 0), mode="replicate")
        causal_ph = F.avg_pool1d(pad_30_ph, kernel_size=30, stride=1).squeeze(1)
        diff_ph = torch.abs(ph - causal_ph)

        pad_30_leit = F.pad(leit.unsqueeze(1), (29, 0), mode="replicate")
        causal_leit = F.avg_pool1d(pad_30_leit, kernel_size=30, stride=1).squeeze(1)
        diff_leit = torch.abs(leit - causal_leit)

        # Standardized physical deviation indicators
        dev_redox = F.relu((750.0 - redox) / 5.0)
        dev_cl = F.relu((0.160 - cl) / 0.01)
        dev_cl2 = F.relu((0.100 - cl_2) / 0.01)
        dev_leit = diff_leit / 2.0
        dev_ph = diff_ph / 0.05
        dev_trueb = F.relu((trueb - 0.03) / 0.01)

        # Disjunction over physical indicators
        phys_max = torch.maximum(
            torch.maximum(dev_redox, torch.maximum(dev_cl, dev_cl2)),
            torch.maximum(torch.maximum(dev_leit, dev_ph), dev_trueb)
        )
        p_prior = torch.sigmoid(1.5 * (phys_max - 3.0))
        return p_prior


class GeccoSubsystemCoupledTCN(nn.Module):
    """Subsystem-Decoupled Causal TCN with Physical Disjunction for Drinking Water."""

    def __init__(self, in_channels: int = 9, hidden_dim: int = 48, teacher_feat_dim: int = 1024,
                 kernel_size: int = 5, dilations: list[int] | None = None, dropout: float = 0.1,
                 base_m: torch.Tensor | None = None, base_s: torch.Tensor | None = None):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8, 16]

        self.idx_disinfect = SUB_DISINFECT_INDICES  # [3, 1, 6]
        self.idx_qual = SUB_QUAL_INDICES            # [2, 4, 5]
        self.idx_ctx = SUB_CTX_INDICES              # [0, 7, 8]

        # Each subsystem stream takes 3 channels * 3 representations = 9 input channels:
        # [x_norm, causal_delta_1, causal_diff_30]
        self.encoder_disinfect = SubsystemEncoder(9, hidden_dim, kernel_size, dilations, dropout)
        self.encoder_qual = SubsystemEncoder(9, hidden_dim, kernel_size, dilations, dropout)

        # Physical equilibrium prior
        self.physical_prior = CausalWaterQualityPrior(base_m=base_m, base_s=base_s)

        # Global feature distillation head (matches DINOv2-Large 1024-dim embedding)
        self.feat_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, teacher_feat_dim)
        )

    def extract_causal_stream_features(self, x_sub: torch.Tensor) -> torch.Tensor:
        # x_sub: [B, C, L]
        # 1-step causal derivative
        pad_1 = F.pad(x_sub, (1, 0), mode="replicate")
        delta = pad_1[:, :, 1:] - pad_1[:, :, :-1]

        # 30-step causal rolling deviation (removes slow seasonal drift)
        pad_30 = F.pad(x_sub, (29, 0), mode="replicate")
        causal_avg = F.avg_pool1d(pad_30, kernel_size=30, stride=1)
        diff_avg = x_sub - causal_avg

        return torch.cat([x_sub, delta, diff_avg], dim=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Handle input shape: [B, L, C] -> [B, C, L]
        if x.dim() == 3 and x.shape[1] != 9 and x.shape[2] == 9:
            x = x.transpose(1, 2)
        elif x.dim() == 2:
            x = x.unsqueeze(0).transpose(1, 2)

        # 1. Physical Subsystem Inputs
        feat_disinfect = self.extract_causal_stream_features(x[:, self.idx_disinfect, :])
        feat_qual = self.extract_causal_stream_features(x[:, self.idx_qual, :])

        # 2. Causal Dilated Encoders
        p_disinfect, h_disinfect = self.encoder_disinfect(feat_disinfect)
        p_qual, h_qual = self.encoder_qual(feat_qual)

        # 3. Causal Physical Prior
        p_prior = self.physical_prior(x)

        # 4. Probabilistic Physical Disjunction (OR Fusion)
        # Anomaly occurs if Disinfection fails OR Water Quality degrades OR Physical Prior trips
        pred_dense = 1.0 - (1.0 - p_disinfect) * (1.0 - p_qual) * (1.0 - p_prior)

        # 5. Global Window Score: max-pool across time
        window_score = torch.max(pred_dense, dim=-1)[0]

        # 6. Global Feature Embedding: pooled across time and projected to 1024-d
        h_cat = torch.cat([h_disinfect, h_qual], dim=1)
        h_pool = torch.mean(h_cat, dim=-1)
        feat = self.feat_head(h_pool)
        student_feat = F.normalize(feat, p=2, dim=-1)

        return {
            "pred_dense": pred_dense,
            "p_disinfect": p_disinfect,
            "p_qual": p_qual,
            "p_prior": p_prior,
            "window_score": window_score,
            "logits": window_score,
            "student_feat": student_feat,
            "z_student": student_feat
        }
