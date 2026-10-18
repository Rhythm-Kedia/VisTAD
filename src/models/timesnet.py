import torch
import torch.nn as nn
import torch.fft


class InceptionBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=5, padding=2)
        self.pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)
        )
        self.act = nn.GELU()

    def forward(self, x):
        out1 = self.conv1(x)
        out3 = self.conv3(x)
        out5 = self.conv5(x)
        outp = self.pool(x)
        return self.act(torch.cat([out1, out3, out5, outp], dim=1))


class TimesNetStudent(nn.Module):
    def __init__(self, in_channels, hidden_dim=128, top_k_periods=3, teacher_feat_dim=2048, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.k_periods = top_k_periods
        
        self.input_embed = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)
        self.inception = InceptionBlock2D(hidden_dim, hidden_dim)
        self.dense_head = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        
        self.feat_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, teacher_feat_dim)
        )

    def forward(self, x):
        B, C, L = x.shape
        h1d = self.input_embed(x)
        
        fft_vals = torch.fft.rfft(x.mean(dim=1), dim=-1)
        amplitudes = torch.abs(fft_vals)
        amplitudes[:, 0] = 0
        top_freqs = torch.topk(amplitudes, self.k_periods, dim=-1).indices
        
        period = max(2, min(L // 2, int(L / max(1, top_freqs[0, 0].item()))))
        length = (L // period) * period
        h_trimmed = h1d[:, :, :length]
        
        h_2d = h_trimmed.reshape(B, self.hidden_dim, length // period, period)
        h_2d_conv = self.inception(h_2d)
        h_recovered = h_2d_conv.reshape(B, self.hidden_dim, length)
        
        if length < L:
            pad = torch.zeros(B, self.hidden_dim, L - length, device=x.device)
            h_recovered = torch.cat([h_recovered, pad], dim=-1)
            
        h_final = h1d + h_recovered
        
        dense_logits = self.dense_head(h_final).squeeze(1)
        dense_probs = torch.sigmoid(dense_logits)
        
        # Multi-scale anomaly pooling (captures both transient spikes and sustained shifts)
        max_p = torch.max(dense_probs, dim=1)[0]
        k_pool = min(10, dense_probs.shape[1])
        top_p = torch.topk(dense_probs, k=k_pool, dim=1)[0].mean(dim=1)
        score_scalar = 0.6 * max_p + 0.4 * top_p
        
        h_pooled = torch.mean(h_final, dim=-1)
        proj_feat = self.feat_projector(h_pooled)
        
        return {
            "score_scalar": score_scalar,
            "logits_dense": dense_probs,
            "features": proj_feat
        }
