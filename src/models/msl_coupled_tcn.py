"""Dual-Stream Telemetry-Command Coupled Causal TCN for NASA MSL.

Solves the channel-independent breakdown on NASA MSL:
- Stream 1: Deep causal dilated TCN on Analog Telemetry (Channel 0) with RF >= 200.
- Stream 2: Causal cross-channel projection on Discrete Commands (Channels 1..54).
- Gated Telemetry-Command Cross-Attention Fusion layer.
- Dense anomaly head + 1024-d latent feature distillation head.
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
    """Dilated causal residual block with LayerNorm and GELU."""
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


class MSLCoupledTCNStudent(nn.Module):
    """Dual-Stream Telemetry-Command Coupled Causal TCN for NASA MSL."""
    def __init__(self, in_channels=55, hidden_dim=64, teacher_feat_dim=1024, dropout=0.05):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        
        # Stream 1: Analog Telemetry (ch0)
        self.analog_in = nn.Conv1d(1, hidden_dim, kernel_size=1)
        analog_dilations = [1, 2, 4, 8, 16, 32]
        self.analog_blocks = nn.ModuleList([
            CausalResidualBlock(hidden_dim, kernel_size=5, dilation=d, dropout=dropout)
            for d in analog_dilations
        ])
        
        # Stream 2: Discrete Commands (ch1..54)
        cmd_channels = in_channels - 1
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

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, in_channels, L) or (B, L, in_channels).
        Returns:
            dict with:
                pred_dense: (B, L) point-wise anomaly probabilities in [0, 1]
                pred_scalar: (B,) window anomaly score in [0, 1]
                pred_feature: (B, 1024) L2-normalized latent feature
                dense_logits: (B, L) raw logits
        """
        if x.dim() == 3 and x.shape[1] != self.in_channels and x.shape[2] == self.in_channels:
            x = x.transpose(1, 2)
            
        b, c, l = x.shape
        
        # Split analog telemetry (ch0) vs discrete commands (ch1..54)
        x_analog = x[:, 0:1, :]
        x_cmd = x[:, 1:, :]
        
        # Process Analog Stream
        h_analog = self.analog_in(x_analog)
        for blk in self.analog_blocks:
            h_analog = blk(h_analog)
            
        # Process Command Stream
        h_cmd = self.cmd_in(x_cmd)
        for blk in self.cmd_blocks:
            h_cmd = blk(h_cmd)
            
        # Gated Cross-Attention Fusion
        concat = torch.cat([h_analog, h_cmd], dim=1)
        gate = torch.sigmoid(self.gate_conv(concat))
        fused = F.gelu(self.fusion_norm(self.fusion_conv(concat)))
        h_total = self.fusion_refine(fused + gate * h_cmd)
        
        # Dense point-wise anomaly probability
        dense_logits = self.dense_head(h_total).squeeze(1)
        pred_dense = torch.sigmoid(dense_logits)
        
        # Balanced Integral Window Pooling
        max_p = torch.max(pred_dense, dim=1)[0]
        k_top = min(10, l)
        top_p = torch.topk(pred_dense, k=k_top, dim=1)[0].mean(dim=1)
        mean_p = torch.mean(pred_dense, dim=1)
        pred_scalar = 0.4 * max_p + 0.3 * top_p + 0.3 * mean_p
        
        # Global Feature Pooling & Projection
        global_pool = torch.mean(h_total, dim=2)
        pred_feat = F.normalize(self.feat_head(global_pool), p=2, dim=-1)
        
        return {
            "pred_dense": pred_dense,
            "pred_scalar": pred_scalar,
            "pred_feature": pred_feat,
            "dense_logits": dense_logits,
        }
