"""Dual-Stream Telemetry-Command Coupled Causal TCN with Causal Volatility Prior for NASA MSL.

Solves the dual failure modes on NASA MSL:
1. Mode 1 (Command-Telemetry Mismatch / Plateaus):
   - Stream 1: Causal dilated TCN on Analog Telemetry (Channel 0) with RF >= 200.
     Features: [z0, tanh(z0/5), diff, std25].
   - Stream 2: Causal cross-channel projection on Discrete Commands (Channels 1..54 + active count).
   - Gated Telemetry-Command Cross-Attention Fusion layer.
   - Dense point-wise anomaly head + 1024-d latent feature distillation head.
2. Mode 2 (Actuator Chattering / Limit-Cycle Oscillation / High-Frequency Transients):
   - Causal Volatility Prior: O(1) causal rolling standard deviation with calibrated sigmoid gating.
3. Dual-Prior Probabilistic OR Fusion:
   - pred_dense = 1 - (1 - p_tcn) * (1 - p_vol).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """1D convolution with strict left-padding (zero future leakage)."""
    def __init__(self, in_channels, out_channels, kernel_size=5, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation)

    def forward(self, x):
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class CausalResidualBlock(nn.Module):
    """Dilated causal residual block with GroupNorm and GELU."""
    def __init__(self, channels, kernel_size=5, dilation=1, dropout=0.05):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv2 = CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=1)
        self.norm2 = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        res = x
        out = F.gelu(self.norm1(self.conv1(x)))
        out = self.dropout(out)
        out = self.norm2(self.conv2(out))
        return F.gelu(out + res)


class CausalVolatilityPrior(nn.Module):
    """Causal rolling volatility prior for detecting high-frequency telemetry chattering/oscillation."""
    def __init__(self, window_size=50, threshold=0.60, k=3.0):
        super().__init__()
        self.window_size = window_size
        self.register_buffer("threshold", torch.tensor(threshold))
        self.register_buffer("k", torch.tensor(k))

    def forward(self, ch0):
        # ch0: (B, 1, L)
        w = self.window_size
        pad_ch0 = F.pad(ch0, (w - 1, 0), mode="replicate")
        m = F.avg_pool1d(pad_ch0, kernel_size=w, stride=1)
        m2 = F.avg_pool1d(pad_ch0**2, kernel_size=w, stride=1)
        std = torch.sqrt(torch.clamp(m2 - m**2, min=0.0) + 1e-8)
        p_vol = torch.sigmoid(self.k * (std - self.threshold))
        return p_vol.squeeze(1)


class MSLCoupledTCNStudent(nn.Module):
    """Dual-Stream Telemetry-Command Coupled Causal TCN with Causal Volatility Prior for NASA MSL."""
    def __init__(self, in_channels=55, hidden_dim=64, teacher_feat_dim=1024, dropout=0.05,
                 vol_window=50, vol_threshold=0.60, vol_k=3.0):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        
        # Stream 1: Causal Analog Telemetry Extractor (4 features: z0, tanh(z0/5), diff, std25)
        self.analog_in = nn.Conv1d(4, hidden_dim, kernel_size=1)
        analog_dilations = [1, 2, 4, 8, 16, 32]
        self.analog_blocks = nn.ModuleList([
            CausalResidualBlock(hidden_dim, kernel_size=5, dilation=d, dropout=dropout)
            for d in analog_dilations
        ])
        
        # Stream 2: Discrete Commands (54 channels + 1 active count = 55 features)
        cmd_channels = in_channels
        self.cmd_in = nn.Conv1d(cmd_channels, hidden_dim, kernel_size=1)
        cmd_dilations = [1, 2, 4, 8]
        self.cmd_blocks = nn.ModuleList([
            CausalResidualBlock(hidden_dim, kernel_size=3, dilation=d, dropout=dropout)
            for d in cmd_dilations
        ])
        
        # Telemetry-Command Gated Fusion
        self.gate_conv = nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1)
        self.fusion_conv = CausalConv1d(hidden_dim * 2, hidden_dim, kernel_size=3, dilation=1)
        self.fusion_norm = nn.GroupNorm(1, hidden_dim)
        self.fusion_refine = CausalResidualBlock(hidden_dim, kernel_size=3, dilation=1, dropout=dropout)
        
        # Dense Point-wise Anomaly Head
        self.dense_head = nn.Sequential(
            CausalConv1d(hidden_dim, 32, kernel_size=3, dilation=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(32, 1, kernel_size=1)
        )
        
        # Latent Feature Head (Distillation matching DINOv2-Large 1024-d)
        self.feat_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, teacher_feat_dim)
        )
        
        # Branch 2: Causal Volatility Prior
        self.vol_prior = CausalVolatilityPrior(window_size=vol_window, threshold=vol_threshold, k=vol_k)

    def extract_analog_features(self, ch0, w=25):
        """Extracts 4 causal analog telemetry features with zero future leakage."""
        z0 = ch0
        z_tanh = torch.tanh(z0 / 5.0)
        diff = F.pad(z0[:, :, 1:] - z0[:, :, :-1], (1, 0), mode="replicate")
        pad_x = F.pad(z0, (w - 1, 0), mode="replicate")
        m = F.avg_pool1d(pad_x, kernel_size=w, stride=1)
        m2 = F.avg_pool1d(pad_x**2, kernel_size=w, stride=1)
        std = torch.sqrt(torch.clamp(m2 - m**2, min=0.0) + 1e-8)
        return torch.cat([z0, z_tanh, diff, std], dim=1)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, in_channels, L) or (B, L, in_channels).
        Returns:
            dict with:
                pred_dense: (B, L) dual-prior point-wise anomaly probabilities in [0, 1]
                p_tcn: (B, L) deep coupled TCN point probabilities
                p_vol: (B, L) causal volatility prior point probabilities
                pred_scalar: (B,) window anomaly score in [0, 1]
                pred_feature: (B, 1024) L2-normalized latent feature
                dense_logits: (B, L) raw TCN logits
        """
        if x.dim() == 3 and x.shape[1] != self.in_channels and x.shape[2] == self.in_channels:
            x = x.transpose(1, 2)
            
        b, c, l = x.shape
        
        # Split analog telemetry (ch0) vs discrete commands (ch1..54)
        x_analog_raw = x[:, 0:1, :]
        x_cmd_raw = x[:, 1:, :]
        
        # Process Analog Stream
        x_analog = self.extract_analog_features(x_analog_raw, w=25)
        h_analog = self.analog_in(x_analog)
        for blk in self.analog_blocks:
            h_analog = blk(h_analog)
            
        # Process Command Stream (augmented with active command count)
        cmd_count = torch.sum(x_cmd_raw > 0, dim=1, keepdim=True).float()
        x_cmd = torch.cat([x_cmd_raw, cmd_count], dim=1)
        h_cmd = self.cmd_in(x_cmd)
        for blk in self.cmd_blocks:
            h_cmd = blk(h_cmd)
            
        # Gated Cross-Attention Fusion
        concat = torch.cat([h_analog, h_cmd], dim=1)
        gate = torch.sigmoid(self.gate_conv(concat))
        fused = F.gelu(self.fusion_norm(self.fusion_conv(concat)))
        h_total = self.fusion_refine(fused + gate * h_cmd)
        
        # Deep TCN point-wise anomaly probability
        dense_logits = self.dense_head(h_total).squeeze(1)
        p_tcn = torch.sigmoid(dense_logits)
        
        # Causal Volatility Prior point-wise probability
        p_vol = self.vol_prior(x_analog_raw)
        
        # Dual-Prior Probabilistic OR Fusion
        pred_dense = 1.0 - (1.0 - p_tcn) * (1.0 - p_vol)
        
        # Balanced Integral Window Pooling
        max_p = torch.max(pred_dense, dim=1)[0]
        k_top = min(10, l)
        top_p = torch.topk(pred_dense, k=k_top, dim=1)[0].mean(dim=1)
        pred_scalar = 0.5 * max_p + 0.5 * top_p
        
        # Global Feature Pooling & Projection
        global_pool = torch.mean(h_total, dim=2)
        pred_feat = F.normalize(self.feat_head(global_pool), p=2, dim=-1)
        
        return {
            "pred_dense": pred_dense,
            "p_tcn": p_tcn,
            "p_vol": p_vol,
            "pred_scalar": pred_scalar,
            "pred_feature": pred_feat,
            "dense_logits": dense_logits,
        }
