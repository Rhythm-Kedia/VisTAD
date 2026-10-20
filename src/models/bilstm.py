import torch
import torch.nn as nn


class BiLSTMStudent(nn.Module):
    """Bidirectional LSTM Student Model with Channel-Independent (CI) support."""
    
    def __init__(self, in_channels, hidden_dim=128, num_layers=2, dropout=0.1,
                 teacher_feat_dim=2048, channel_independent=True, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.channel_independent = channel_independent
        
        curr_in = 1 if channel_independent else in_channels
        self.lstm = nn.LSTM(
            input_size=curr_in,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.dense_head = nn.Linear(hidden_dim, 1)
        
        self.feat_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, teacher_feat_dim)
        )
        
    def forward(self, x):
        if self.channel_independent:
            B, K, L = x.shape
            x_ci = x.reshape(B * K, L, 1)
            h_ci, _ = self.lstm(x_ci)
            dense_ci = torch.sigmoid(self.dense_head(h_ci)).squeeze(-1)
            dense_per_chan = dense_ci.reshape(B, K, L)
            
            # Permutation-invariant aggregation across channels
            dense_probs = torch.max(dense_per_chan, dim=1)[0]
            
            # Multi-scale anomaly pooling
            max_p = torch.max(dense_probs, dim=1)[0]
            k_pool = min(10, L)
            top_p = torch.topk(dense_probs, k=k_pool, dim=1)[0].mean(dim=1)
            score_scalar = 0.6 * max_p + 0.4 * top_p
            
            h_pooled = torch.mean(h_ci.reshape(B, K, L, -1).mean(dim=1), dim=1)
            proj_feat = self.feat_projector(h_pooled)
            
            return {
                "score_scalar": score_scalar,
                "logits_dense": dense_probs,
                "features": proj_feat
            }
        else:
            x_perm = x.permute(0, 2, 1)
            h, _ = self.lstm(x_perm)
            
            dense_logits = self.dense_head(h).squeeze(-1)
            dense_probs = torch.sigmoid(dense_logits)
            
            max_p = torch.max(dense_probs, dim=1)[0]
            k_pool = min(10, dense_probs.shape[1])
            top_p = torch.topk(dense_probs, k=k_pool, dim=1)[0].mean(dim=1)
            score_scalar = 0.6 * max_p + 0.4 * top_p
            
            h_pooled = torch.mean(h, dim=1)
            proj_feat = self.feat_projector(h_pooled)
            
            return {
                "score_scalar": score_scalar,
                "logits_dense": dense_probs,
                "features": proj_feat
            }
