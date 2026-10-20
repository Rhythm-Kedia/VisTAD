import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDilatedBlock(nn.Module):
    """Residual dilated conv block for long-range temporal receptive fields."""
    def __init__(self, in_channels, out_channels, dilation, dropout=0.1):
        super().__init__()
        padding = 2 * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        res = self.skip(x)
        h = self.drop(self.act(self.bn1(self.conv1(x))))
        h = self.drop(self.bn2(self.conv2(h)))
        return self.act(h + res)


class CNN1DStudent(nn.Module):
    """Temporal 1D-CNN / Residual TCN Student Model for TSAD.
    
    Supports Channel-Independent (CI) temporal modeling: processes each channel
    through a shared 1D temporal backbone, enabling zero-shot cross-entity
    and cross-dataset generalization with permutation invariance.
    """
    
    def __init__(self, in_channels=6, hidden_dim=128, num_layers=3, dropout=0.1,
                 teacher_feat_dim=2048, channel_independent=True, use_residual=False):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.channel_independent = channel_independent
        self.use_residual = use_residual
        
        curr_in = 1 if channel_independent else in_channels
        
        if use_residual:
            self.res_blocks = nn.ModuleList()
            for i in range(num_layers):
                d = 2 ** i
                self.res_blocks.append(
                    ResidualDilatedBlock(curr_in, hidden_dim, dilation=d, dropout=dropout)
                )
                curr_in = hidden_dim
            self.backbone = None
        else:
            layers = []
            for i in range(num_layers):
                d = 2 ** i
                layers.extend([
                    nn.Conv1d(curr_in, hidden_dim, kernel_size=5, padding=2 * d, dilation=d),
                    nn.BatchNorm1d(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout)
                ])
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
        
    def forward(self, x):
        if self.channel_independent:
            B, K, L = x.shape
            x_ci = x.reshape(B * K, 1, L)
            h_ci = self._forward_backbone(x_ci)
            dense_ci = torch.sigmoid(self.dense_head(h_ci)).squeeze(1)
            dense_per_chan = dense_ci.reshape(B, K, L)
            
            # Permutation-invariant aggregation across channels: max anomaly probability
            dense_probs = torch.max(dense_per_chan, dim=1)[0]
            
            # Multi-scale anomaly pooling (captures both transient spikes and sustained shifts)
            max_p = torch.max(dense_probs, dim=1)[0]
            k_pool = min(10, L)
            top_p = torch.topk(dense_probs, k=k_pool, dim=1)[0].mean(dim=1)
            score_scalar = 0.6 * max_p + 0.4 * top_p
            
            # Feature representation pooled over time and channels
            h_pooled = torch.mean(h_ci.reshape(B, K, -1, L).mean(dim=1), dim=-1)
            proj_feat = self.feat_projector(h_pooled)
            
            return {
                "score_scalar": score_scalar,
                "logits_dense": dense_probs,
                "features": proj_feat
            }
        else:
            h = self._forward_backbone(x)
            dense_logits = self.dense_head(h).squeeze(1)
            dense_probs = torch.sigmoid(dense_logits)
            
            max_p = torch.max(dense_probs, dim=1)[0]
            k_pool = min(10, dense_probs.shape[1])
            top_p = torch.topk(dense_probs, k=k_pool, dim=1)[0].mean(dim=1)
            score_scalar = 0.6 * max_p + 0.4 * top_p
            
            h_pooled = torch.mean(h, dim=-1)
            proj_feat = self.feat_projector(h_pooled)
            
            return {
                "score_scalar": score_scalar,
                "logits_dense": dense_probs,
                "features": proj_feat
            }
