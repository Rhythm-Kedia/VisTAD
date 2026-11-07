"""Stage-Coupled Causal Dilated TCN Student Architecture for SWaT.

Physical Plant Invariants:
SWaT consists of 6 sequential, physically coupled treatment stages:
- Stage 1: Raw Water Intake & Storage (5 channels: FIT101, LIT101, MV101, P101, P102)
- Stage 2: Chemical Dosing (11 channels: AIT201..AIT203, FIT201, MV201, P201..P206)
- Stage 3: Ultrafiltration (9 channels: DPIT301, FIT301, LIT301, MV301..MV304, P301..P302)
- Stage 4: Dechlorination (9 channels: AIT401..AIT402, FIT401, LIT401, P401..P404, UV401)
- Stage 5: Reverse Osmosis (13 channels: AIT501..AIT504, DPIT501, FIT501..FIT504, P501..P502, PIT501..PIT503)
- Stage 6: Distribution (4 channels: FIT601, P601..P603)
Total: 51 physical telemetry channels.

Architecture:
1. 6 parallel dilated causal stage sub-networks (k=5, d=1..32 -> Receptive Field = 253 steps)
2. Cross-stage causal fusion (192 -> 128 channels)
3. Dense temporal prediction head ([B, 200] point-wise probabilities)
4. Distillation projection head ([B, 1024] normalized latent vector aligning with DINOv2-Large)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

SWAT_STAGE_INDICES = {
    1: [0, 1, 2, 3, 4],                                      # 5 channels
    2: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],              # 11 channels
    3: [16, 17, 18, 19, 20, 21, 22, 23, 24],                 # 9 channels
    4: [25, 26, 27, 28, 29, 30, 31, 32, 33],                 # 9 channels
    5: [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46], # 13 channels
    6: [47, 48, 49, 50]                                      # 4 channels
}


class CausalConv1dBlock(nn.Module):
    """Dilated causal residual block with zero future leakage."""
    def __init__(self, in_channels, out_channels, kernel_size=5, dilation=1, dropout=0.1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        self.act2 = nn.GELU()
        self.norm = nn.BatchNorm1d(out_channels)
        self.drop = nn.Dropout(dropout)
        self.res = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        # Left causal padding
        residual = self.res(x)
        out = F.pad(x, (self.pad, 0))
        out = self.act1(self.conv1(out))
        out = F.pad(out, (self.pad, 0))
        out = self.drop(self.act2(self.conv2(out)))
        return self.norm(out + residual)


class StageSubNetwork(nn.Module):
    """Stage-specific temporal block capturing cross-channel physical invariants."""
    def __init__(self, in_channels, stage_dim=32, kernel_size=5, dilations=(1, 2, 4, 8, 16, 32), dropout=0.1):
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels, stage_dim, kernel_size=1)
        layers = []
        for d in dilations:
            layers.append(CausalConv1dBlock(stage_dim, stage_dim, kernel_size=kernel_size, dilation=d, dropout=dropout))
        self.blocks = nn.Sequential(*layers)

    def forward(self, x_stage):
        # x_stage: [B, C_stage, L]
        h = self.input_proj(x_stage)
        return self.blocks(h)


class StageCoupledTCNStudent(nn.Module):
    """Full Stage-Coupled Causal Dilated TCN Student."""
    def __init__(self, stage_dim=32, fusion_dim=128, feat_dim=1024, dropout=0.1):
        super().__init__()
        self.stage_indices = SWAT_STAGE_INDICES
        
        # 1. Six dedicated stage encoders
        self.stage_encoders = nn.ModuleDict({
            f"stage_{sid}": StageSubNetwork(
                in_channels=len(indices),
                stage_dim=stage_dim,
                kernel_size=5,
                dilations=(1, 2, 4, 8, 16, 32),
                dropout=dropout
            )
            for sid, indices in self.stage_indices.items()
        })
        
        # Total concatenated channels = 6 stages * stage_dim = 192
        total_stage_channels = len(self.stage_indices) * stage_dim

        # 2. Cross-Stage Fusion Network
        self.fusion_proj = nn.Sequential(
            nn.Conv1d(total_stage_channels, fusion_dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(fusion_dim)
        )
        self.fusion_blocks = nn.Sequential(
            CausalConv1dBlock(fusion_dim, fusion_dim, kernel_size=5, dilation=1, dropout=dropout),
            CausalConv1dBlock(fusion_dim, fusion_dim, kernel_size=5, dilation=2, dropout=dropout)
        )

        # 3. Dense Temporal Prediction Head (Point-wise probability [B, L])
        self.dense_head = nn.Sequential(
            nn.Conv1d(fusion_dim, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, 1, kernel_size=1)
        )

        # 4. Feature Distillation Head (Aligning with DINOv2-Large 1024-dim CLS)
        self.feat_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(fusion_dim, 256),
            nn.GELU(),
            nn.Linear(256, feat_dim)
        )

    def forward(self, x):
        """
        x: [B, 51, L] (or [B, L, 51] if passed in batch format)
        Returns:
            dict containing:
            - pred_dense: [B, L] point-wise probabilities in [0, 1]
            - window_score: [B] scalar window anomaly score in [0, 1]
            - student_feat: [B, 1024] normalized latent embedding
        """
        # Ensure shape [B, 51, L]
        if x.shape[1] != 51 and x.shape[2] == 51:
            x = x.permute(0, 2, 1)

        # 1. Forward each physical stage independently
        stage_feats = []
        for sid in range(1, 7):
            idx = self.stage_indices[sid]
            x_s = x[:, idx, :]  # [B, C_s, L]
            h_s = self.stage_encoders[f"stage_{sid}"](x_s)  # [B, stage_dim, L]
            stage_feats.append(h_s)

        # 2. Concatenate all 6 stages along channel dimension
        cat_feats = torch.cat(stage_feats, dim=1)  # [B, 192, L]

        # 3. Cross-stage fusion
        fused = self.fusion_proj(cat_feats)  # [B, 128, L]
        fused = self.fusion_blocks(fused)    # [B, 128, L]

        # 4. Dense point-wise prediction
        dense_logits = self.dense_head(fused).squeeze(1)  # [B, L]
        pred_dense = torch.sigmoid(dense_logits)          # [B, L]

        # 5. Window-level anomaly score
        window_score = torch.max(pred_dense, dim=-1)[0]   # [B]

        # 6. Latent feature projection
        feat = self.feat_head(fused)                      # [B, 1024]
        student_feat = F.normalize(feat, p=2, dim=-1)     # [B, 1024]

        return {
            "pred_dense": pred_dense,
            "window_score": window_score,
            "student_feat": student_feat
        }
