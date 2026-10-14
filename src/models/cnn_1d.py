import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1DStudent(nn.Module):
    """Temporal 1D-CNN Student Model for TSAD."""
    
    def __init__(self, in_channels, hidden_dim=128, num_layers=3, dropout=0.1, teacher_feat_dim=2048):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        
        layers = []
        curr_in = in_channels
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
        
        self.dense_head = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        
        self.feat_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, teacher_feat_dim)
        )
        
    def forward(self, x):
        h = self.backbone(x)
        dense_logits = self.dense_head(h).squeeze(1)
        dense_probs = torch.sigmoid(dense_logits)
        
        k_pool = min(10, dense_probs.shape[1])
        score_scalar = torch.topk(dense_probs, k=k_pool, dim=1)[0].mean(dim=1)
        
        h_pooled = torch.mean(h, dim=-1)
        proj_feat = self.feat_projector(h_pooled)
        
        return {
            "score_scalar": score_scalar,
            "logits_dense": dense_probs,
            "features": proj_feat
        }
