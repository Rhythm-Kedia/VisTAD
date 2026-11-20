import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConvLayer(nn.Module):
    """Standard conv layer with causal lookback slice to preserve length L."""
    def __init__(self, in_channels, out_channels, dilation=1, dropout=0.1, causal=True):
        super().__init__()
        self.causal = causal
        pad = 4 * dilation if causal else 2 * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=pad, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        L = x.shape[-1]
        h = self.conv(x)
        if self.causal:
            h = h[:, :, :L]
        return self.drop(self.act(self.bn(h)))


class ResidualDilatedBlock(nn.Module):
    """Residual dilated conv block with causal temporal receptive fields."""
    def __init__(self, in_channels, out_channels, dilation, dropout=0.1, causal=True):
        super().__init__()
        self.dilation = dilation
        self.causal = causal
        pad = 4 * dilation if causal else 2 * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        res = self.skip(x)
        L = x.shape[-1]
        h = self.conv1(x)
        if self.causal:
            h = h[:, :, :L]
        h = self.drop(self.act(self.bn1(h)))
        
        h = self.conv2(h)
        if self.causal:
            h = h[:, :, :L]
        h = self.drop(self.bn2(h))
        return self.act(h + res)


class CNN1DStudent(nn.Module):
    """Temporal 1D-CNN / Residual TCN Student Model for TSAD.
    
    Supports Channel-Independent (CI) temporal modeling: processes each channel
    through a shared 1D temporal backbone, enabling zero-shot cross-entity
    and cross-dataset generalization with permutation invariance.
    """
    
    def __init__(self, in_channels=6, hidden_dim=128, num_layers=3, dropout=0.1,
                 teacher_feat_dim=2048, channel_independent=True, use_residual=False, causal=True):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.channel_independent = channel_independent
        self.use_residual = use_residual
        self.causal = causal
        
        curr_in = 1 if channel_independent else in_channels
        
        if use_residual:
            self.res_blocks = nn.ModuleList()
            for i in range(num_layers):
                d = 2 ** i
                self.res_blocks.append(
                    ResidualDilatedBlock(curr_in, hidden_dim, dilation=d, dropout=dropout, causal=causal)
                )
                curr_in = hidden_dim
            self.backbone = None
        else:
            layers = []
            for i in range(num_layers):
                d = 2 ** i
                layers.append(
                    CausalConvLayer(curr_in, hidden_dim, dilation=d, dropout=dropout, causal=causal)
                )
                curr_in = hidden_dim
            self.backbone = nn.Sequential(*layers)
            self.res_blocks = None
        
        self.dense_head = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        
        self.feat_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, teacher_feat_dim)
        )
        
    def _forward_backbone(self, x):
        if self.use_residual:
            h = x
            for block in self.res_blocks:
                h = block(h)
            return h
        return self.backbone(x)
        
    def forward(self, x, **kwargs):
        if self.channel_independent:
            B, K, L = x.shape
            x_ci = x.reshape(B * K, 1, L)
            h_ci = self._forward_backbone(x_ci)
            dense_ci = torch.sigmoid(self.dense_head(h_ci)).squeeze(1)
            dense_per_chan = dense_ci.reshape(B, K, L)
            
            # Permutation-invariant aggregation across channels: max anomaly probability
            dense_probs = torch.max(dense_per_chan, dim=1)[0]
            
            # Multi-scale morphology-aware anomaly pooling (spikes + sustained shifts/plateaus)
            max_p = torch.max(dense_probs, dim=1)[0]
            k_pool = min(10, L)
            top_p = torch.topk(dense_probs, k=k_pool, dim=1)[0].mean(dim=1)
            mean_p = torch.mean(dense_probs, dim=1)
            score_scalar = 0.4 * max_p + 0.3 * top_p + 0.3 * mean_p
            
            # Feature representation pooled over time and channels
            h_pooled = torch.mean(h_ci.reshape(B, K, -1, L).mean(dim=1), dim=-1)
            proj_feat = F.normalize(self.feat_projector(h_pooled), p=2, dim=-1)
            
            return {
                "pred_dense": dense_probs,
                "p_system": dense_probs,
                "score_scalar": score_scalar,
                "logits_dense": dense_probs,
                "pred_feature": proj_feat,
                "z_student": proj_feat,
                "features": proj_feat
            }
        else:
            h = self._forward_backbone(x)
            dense_logits = self.dense_head(h).squeeze(1)
            dense_probs = torch.sigmoid(dense_logits)
            
            max_p = torch.max(dense_probs, dim=1)[0]
            k_pool = min(10, dense_probs.shape[1])
            top_p = torch.topk(dense_probs, k=k_pool, dim=1)[0].mean(dim=1)
            mean_p = torch.mean(dense_probs, dim=1)
            score_scalar = 0.4 * max_p + 0.3 * top_p + 0.3 * mean_p
            
            h_pooled = torch.mean(h, dim=-1)
            proj_feat = F.normalize(self.feat_projector(h_pooled), p=2, dim=-1)
            
            return {
                "pred_dense": dense_probs,
                "p_system": dense_probs,
                "score_scalar": score_scalar,
                "logits_dense": dense_probs,
                "pred_feature": proj_feat,
                "z_student": proj_feat,
                "features": proj_feat
            }
