"""Universal TCN Pro with Channel Identity Embeddings (Non-Permutation-Equivariant for Single-Dataset Deployment).

Retains the scalable UniversalFoundationTCN_Pro inductive biases:
1. Multi-scale causal feature extractor per channel: [x, velocity, diff_30, rolling_std_25]
2. Channel-independent causal dilated residual temporal backbone (RF >= 250 steps, zero future leakage)
3. Stacked Pre-LN Cross-Channel Relational Transformer (2 layers, 8 heads, D=128, FFN=512)
4. Non-Permutation-Equivariant Channel Identity Embedding:
   Injects learned E_channel in R^{1 x C x D} so attention explicitly identifies each physical sensor
   (e.g., distinguishing continuous motor drive Channel 0 from discrete command lines 1..54 on NASA MSL).
5. Per-channel attribution + Top-K Probabilistic OR Disjunction system risk aggregation.
6. 1024-dim DINOv2 VLM latent feature alignment head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1dResidualBlock(nn.Module):
    """Causal dilated residual block with left-only padding (zero future leakage)."""
    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1, dropout: float = 0.05):
        super().__init__()
        self.pad_len = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.norm2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = F.pad(x, (self.pad_len, 0), mode="constant", value=0.0)
        out = self.norm1(F.gelu(self.conv1(out)))
        out = F.pad(out, (self.pad_len, 0), mode="constant", value=0.0)
        out = self.dropout(self.norm2(self.conv2(out)))
        return F.gelu(res + out)


class ChannelIndependentTemporalBackbone(nn.Module):
    """Shared causal temporal feature extractor operating on [B * C, 4, L]."""
    def __init__(self, in_features: int = 4, hidden_dim: int = 128, kernel_size: int = 5,
                 dilations: tuple = (1, 2, 4, 8, 16, 32), dropout: float = 0.05):
        super().__init__()
        self.in_proj = nn.Conv1d(in_features, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList([
            CausalConv1dResidualBlock(hidden_dim, kernel_size=kernel_size, dilation=d, dropout=dropout)
            for d in dilations
        ])
        self.out_norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, x_features: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x_features)
        for block in self.blocks:
            h = block(h)
        return self.out_norm(h)


class CrossChannelTransformerLayer(nn.Module):
    """Pre-LN Multi-Head Self-Attention + FFN across the C channels."""
    def __init__(self, hidden_dim: int = 128, num_heads: int = 8, ffn_dim: int = 512, dropout: float = 0.05):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = self.norm1(x)
        attn_out, _ = self.mha(norm_x, norm_x, norm_x)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class UniversalTCNChannelAwareStudent(nn.Module):
    """Universal TCN Pro with Channel Identity Embeddings for Single-Dataset Deployment."""

    def __init__(self, in_channels: int = 55, hidden_dim: int = 128, num_heads: int = 8,
                 num_layers: int = 2, ffn_dim: int | None = None, kernel_size: int = 5,
                 dilations: tuple = (1, 2, 4, 8, 16, 32), dropout: float = 0.05,
                 top_k: int = 5, teacher_feat_dim: int = 1024, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.top_k = min(top_k, in_channels)
        if ffn_dim is None:
            ffn_dim = hidden_dim * 4

        # 1. Multi-Scale Causal Temporal Backbone (L_RF = 253 steps)
        self.temporal_encoder = ChannelIndependentTemporalBackbone(
            in_features=4, hidden_dim=hidden_dim, kernel_size=kernel_size,
            dilations=dilations, dropout=dropout
        )

        # 2. Learnable Channel Identity Embeddings (Breaks Permutation Invariance)
        self.channel_identity_embed = nn.Parameter(torch.randn(1, in_channels, hidden_dim) * 0.02)

        # 3. Stacked Pre-LN Cross-Channel Transformer
        self.channel_transformer = nn.ModuleList([
            CrossChannelTransformerLayer(hidden_dim=hidden_dim, num_heads=num_heads, ffn_dim=ffn_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.transformer_norm = nn.LayerNorm(hidden_dim)

        # 4. Dense Point-Wise Channel Anomaly Head
        self.channel_head = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim // 2, 1, kernel_size=1)
        )
        # Calibrated output initialization
        nn.init.constant_(self.channel_head[3].bias, -2.0)

        # 5. DINOv2-Large 1024-dim Latent Distillation Head
        self.vlm_proj_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, teacher_feat_dim)
        )

    def extract_quadruplet_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts [raw x, velocity dx, diff_30, rolling_std_25] per channel."""
        # x: [B, C, L]
        B, C, L = x.shape

        # 1. Causal 1-step velocity
        pad_1 = F.pad(x, (1, 0), mode="replicate")
        velocity = pad_1[:, :, 1:] - pad_1[:, :, :-1]

        # 2. Causal 30-step rolling deviation
        pad_30 = F.pad(x, (29, 0), mode="replicate")
        mean_30 = F.avg_pool1d(pad_30, kernel_size=30, stride=1)
        diff_30 = x - mean_30

        # 3. Causal 25-step rolling volatility (standard deviation)
        pad_25 = F.pad(x, (24, 0), mode="replicate")
        mean_25 = F.avg_pool1d(pad_25, kernel_size=25, stride=1)
        sq_mean_25 = F.avg_pool1d(pad_25 ** 2, kernel_size=25, stride=1)
        var_25 = F.relu(sq_mean_25 - mean_25 ** 2)
        std_25 = torch.sqrt(var_25 + 1e-8)

        # Stack into [B, C, 4, L] -> [B * C, 4, L]
        feats = torch.stack([x, velocity, diff_30, std_25], dim=2)
        return feats.view(B * C, 4, L)

    def forward(self, x: torch.Tensor, prefix_len: int = 0, **kwargs) -> dict[str, torch.Tensor]:
        """
        Args:
            x: [B, C, L] or [B, L, C]
            prefix_len: warm-up prefix length to exclude from global window score.
        Returns:
            dict containing:
                pred_dense: [B, L] continuous system anomaly probability
                pred_scalar: [B] scalar window anomaly score
                pred_feature: [B, teacher_feat_dim] normalized VLM feature projection
                p_channel: [B, C, L] fine-grained sensor anomaly attribution
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        elif x.dim() == 3 and x.shape[1] != self.in_channels and x.shape[2] == self.in_channels:
            x = x.transpose(1, 2)

        B, C, L = x.shape
        assert C == self.in_channels, f"Expected {self.in_channels} channels, got {C}"

        # 1. Temporal Feature Extraction (C mapped to batch dimension)
        x_feats = self.extract_quadruplet_features(x)         # [B * C, 4, L]
        h_temp = self.temporal_encoder(x_feats)               # [B * C, hidden_dim, L]

        # 2. Channel Token Pooling & Channel Identity Injection
        h_tokens = torch.mean(h_temp, dim=-1).view(B, C, self.hidden_dim)  # [B, C, hidden_dim]
        h_tokens = h_tokens + self.channel_identity_embed                   # Injects channel identity

        # 3. Stacked Cross-Channel Relational Transformer
        h_cross = h_tokens
        for layer in self.channel_transformer:
            h_cross = layer(h_cross)
        h_cross = self.transformer_norm(h_cross)                            # [B, C, hidden_dim]

        # 4. Broadcast Cross-Channel Context Back into Temporal State
        h_enriched = h_temp + h_cross.view(B * C, self.hidden_dim, 1)       # [B * C, hidden_dim, L]

        # 5. Point-Wise Channel Anomaly Risk
        channel_logits = self.channel_head(h_enriched).view(B, C, L)        # [B, C, L]
        p_channel = torch.sigmoid(channel_logits)                           # [B, C, L]

        # 6. Top-K Probabilistic OR Disjunction Fusion
        top_p, _ = torch.topk(p_channel, k=self.top_k, dim=1)               # [B, top_k, L]
        top_p_clamped = torch.clamp(top_p, 1e-6, 1.0 - 1e-6)
        pred_dense = 1.0 - torch.prod(1.0 - top_p_clamped, dim=1)          # [B, L]
        pred_dense = torch.clamp(pred_dense, 1e-6, 1.0 - 1e-6)

        # 7. Balanced Integral Window Score
        target_dense = pred_dense[:, prefix_len:] if prefix_len > 0 else pred_dense
        max_p = torch.max(target_dense, dim=1)[0]
        k_top_l = min(10, target_dense.shape[1])
        top_p_mean = torch.topk(target_dense, k=k_top_l, dim=1)[0].mean(dim=1)
        pred_scalar = 0.5 * max_p + 0.5 * top_p_mean

        # 8. VLM Student Feature Representation
        h_global = torch.mean(h_cross, dim=1)                               # [B, hidden_dim]
        pred_feature = F.normalize(self.vlm_proj_head(h_global), p=2, dim=-1)

        return {
            "pred_dense": pred_dense,
            "p_system": pred_dense,
            "pred_scalar": pred_scalar,
            "pred_feature": pred_feature,
            "p_channel": p_channel,
            "channel_logits": channel_logits
        }


# Alias for cross-ablation suites
UniversalTCN_ChannelAware = UniversalTCNChannelAwareStudent

