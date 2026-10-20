import torch
import torch.nn as nn
import torch.nn.functional as F
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
    """TimesNet Student Model with replicate padding, multi-period weighting, and CI support."""
    
    def __init__(self, in_channels, hidden_dim=128, top_k_periods=2, teacher_feat_dim=2048,
                 channel_independent=True, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.k_periods = top_k_periods
        self.channel_independent = channel_independent
        
        curr_in = 1 if channel_independent else in_channels
        self.input_embed = nn.Conv1d(curr_in, hidden_dim, kernel_size=1)
        self.inception = InceptionBlock2D(hidden_dim, hidden_dim)
        self.dense_head = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        
        self.feat_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, teacher_feat_dim)
        )

    def _transform_1d_to_2d(self, h1d, x_signal):
        # h1d: [N, D, L], x_signal: [N, 1, L]
        N, D, L = h1d.shape
        fft_vals = torch.fft.rfft(x_signal.squeeze(1), dim=-1)
        amplitudes = torch.abs(fft_vals)
        amplitudes[:, 0] = 0
        mean_amp = torch.mean(amplitudes, dim=0)
        
        top_freqs = torch.topk(mean_amp, min(self.k_periods, mean_amp.shape[0])).indices
        res_list = []
        weight_list = []
        
        for freq_idx in top_freqs:
            freq = freq_idx.item()
            if freq == 0:
                continue
            p = max(4, min(L // 2, int(L / freq)))
            pad_len = (p - (L % p)) % p
            h_pad = F.pad(h1d, (0, pad_len), mode="replicate") if pad_len > 0 else h1d
            h_2d = h_pad.reshape(N, D, (L + pad_len) // p, p)
            h_conv = self.inception(h_2d).reshape(N, D, L + pad_len)
            res_list.append(h_conv[:, :, :L])
            weight_list.append(mean_amp[freq_idx].item())
            
        if not res_list:
            return h1d
            
        weights = F.softmax(torch.tensor(weight_list, device=h1d.device, dtype=torch.float32), dim=0)
        h_recovered = sum(w * r for w, r in zip(weights, res_list))
        return h1d + h_recovered

    def forward(self, x):
        B, K, L = x.shape
        if self.channel_independent:
            x_ci = x.reshape(B * K, 1, L)
            h1d = self.input_embed(x_ci)
            h_final = self._transform_1d_to_2d(h1d, x_ci)
            dense_ci = torch.sigmoid(self.dense_head(h_final)).squeeze(1)
            dense_per_chan = dense_ci.reshape(B, K, L)
            
            dense_probs = torch.max(dense_per_chan, dim=1)[0]
            max_p = torch.max(dense_probs, dim=1)[0]
            k_pool = min(10, L)
            top_p = torch.topk(dense_probs, k=k_pool, dim=1)[0].mean(dim=1)
            score_scalar = 0.6 * max_p + 0.4 * top_p
            
            h_pooled = torch.mean(h_final.reshape(B, K, -1, L).mean(dim=1), dim=-1)
            proj_feat = self.feat_projector(h_pooled)
            
            return {
                "score_scalar": score_scalar,
                "logits_dense": dense_probs,
                "features": proj_feat
            }
        else:
            h1d = self.input_embed(x)
            h_final = self._transform_1d_to_2d(h1d, x)
            dense_logits = self.dense_head(h_final).squeeze(1)
            dense_probs = torch.sigmoid(dense_logits)
            
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
