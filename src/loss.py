import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalBCELoss(nn.Module):
    """Focal Binary Cross-Entropy to dynamically down-weight easy nominal timesteps."""
    def __init__(self, gamma=2.0, pos_weight=2.0, eps=1e-7):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = float(pos_weight)
        self.eps = eps

    def forward(self, pred, target):
        p = torch.clamp(pred, self.eps, 1.0 - self.eps)
        # Positive focal term: - pos_weight * (1 - p)^gamma * log(p)
        pos_loss = - self.pos_weight * ((1.0 - p) ** self.gamma) * target * torch.log(p)
        # Negative focal term: - (p)^gamma * log(1 - p)
        neg_loss = - (p ** self.gamma) * (1.0 - target) * torch.log(1.0 - p)
        return torch.mean(pos_loss + neg_loss)


class TemporalDiceLoss(nn.Module):
    """Continuous temporal Soft-Dice / IoU loss to optimize contiguous segment overlap."""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred_dense, gt_dense, y_flag=None):
        if y_flag is not None:
            pos_mask = (y_flag > 0.5)
            if not pos_mask.any():
                return torch.tensor(0.0, device=pred_dense.device, dtype=pred_dense.dtype)
            p = pred_dense[pos_mask]
            y = gt_dense[pos_mask]
        else:
            p = pred_dense
            y = gt_dense
        
        intersection = torch.sum(p * y, dim=-1)
        cardinality = torch.sum(p.pow(2) + y.pow(2), dim=-1)
        dice = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        return torch.mean(1.0 - dice)


class TemporalTotalVariationLoss(nn.Module):
    """Penalizes high-frequency prediction oscillations to enforce smooth, contiguous plateau predictions."""
    def forward(self, pred_dense):
        diff = torch.abs(pred_dense[:, 1:] - pred_dense[:, :-1])
        return torch.mean(diff)


class PairwiseRankingMarginLoss(nn.Module):
    """Directly maximizes ranking separation margin between anomalous and nominal windows."""
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, scores, y_flag):
        pos_mask = (y_flag > 0.5)
        neg_mask = (y_flag <= 0.5)
        if not pos_mask.any() or not neg_mask.any():
            return torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
        
        pos_scores = scores[pos_mask]
        neg_scores = scores[neg_mask]
        
        # Matrix of differences: pos[i] - neg[j]
        diffs = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
        losses = F.relu(self.margin - diffs)
        return torch.mean(losses)


class MultiObjectiveKDLoss(nn.Module):
    """Composite Multi-Objective Loss supporting BCE, Focal, Dice, TV, and Ranking."""
    
    def __init__(self, w_sup=1.0, w_feat=0.1, w_resp=1.0, temperature=2.0, decision_threshold=0.63,
                 beta=8.0, pos_weight=2.0, loss_type="hybrid", w_dice=0.4, w_tv=0.1, w_rank=0.3, gamma=2.0):
        super().__init__()
        self.w_sup = w_sup
        self.w_feat = w_feat
        self.w_resp = w_resp
        self.temp = temperature
        self.threshold = decision_threshold
        self.beta = beta
        self.pos_weight = float(pos_weight)
        self.loss_type = loss_type
        self.w_dice = w_dice
        self.w_tv = w_tv
        self.w_rank = w_rank
        
        self.focal_loss = FocalBCELoss(gamma=gamma, pos_weight=pos_weight)
        self.dice_loss = TemporalDiceLoss()
        self.tv_loss = TemporalTotalVariationLoss()
        self.rank_loss = PairwiseRankingMarginLoss(margin=0.2)
        
    def forward(self, student_outputs, batch):
        losses = {}
        total_loss = 0.0
        
        # 1. Supervised Ground-Truth Anchor Loss (L_sup)
        if self.w_sup > 0.0:
            pred_dense = student_outputs["logits_dense"]
            gt_dense = batch["y_dense"].to(pred_dense.device)
            pred_scalar = student_outputs["score_scalar"]
            gt_flag = batch["y_flag"].to(pred_scalar.device)
            eps = 1e-7

            if self.loss_type in ["focal", "hybrid"]:
                l_dense = self.focal_loss(pred_dense, gt_dense)
                l_scalar = self.focal_loss(pred_scalar, gt_flag)
            else:
                p_d = torch.clamp(pred_dense, eps, 1.0 - eps)
                l_dense = - torch.mean(self.pos_weight * gt_dense * torch.log(p_d) + (1.0 - gt_dense) * torch.log(1.0 - p_d))
                p_s = torch.clamp(pred_scalar, eps, 1.0 - eps)
                l_scalar = - torch.mean(self.pos_weight * gt_flag * torch.log(p_s) + (1.0 - gt_flag) * torch.log(1.0 - p_s))

            l_sup_base = 0.6 * l_dense + 0.4 * l_scalar
            
            # Additional structural objectives for non-spike anomalies
            l_dice = self.dice_loss(pred_dense, gt_dense, y_flag=gt_flag) if (self.loss_type in ["dice", "hybrid"]) else torch.tensor(0.0)
            l_tv = self.tv_loss(pred_dense) if (self.loss_type in ["tv", "hybrid"]) else torch.tensor(0.0)
            l_rank = self.rank_loss(pred_scalar, gt_flag) if (self.loss_type in ["rank", "hybrid"]) else torch.tensor(0.0)

            l_sup = l_sup_base
            if self.loss_type == "hybrid":
                l_sup = l_sup_base + self.w_dice * l_dice + self.w_tv * l_tv + self.w_rank * l_rank

            losses["loss_sup"] = l_sup_base.item()
            losses["loss_dice"] = l_dice.item()
            losses["loss_tv"] = l_tv.item()
            losses["loss_rank"] = l_rank.item()
            total_loss += self.w_sup * l_sup
        else:
            losses["loss_sup"] = 0.0
            losses["loss_dice"] = 0.0
            losses["loss_tv"] = 0.0
            losses["loss_rank"] = 0.0
            
        # 2. Teacher Latent Feature Distillation (L_feat): Cosine Distance
        device = student_outputs["score_scalar"].device
        has_t = batch.get("has_teacher")
        if has_t is not None:
            t_mask = has_t.to(device).bool()
        else:
            t_mask = torch.ones(len(student_outputs["score_scalar"]), dtype=torch.bool, device=device)
            
        if self.w_feat > 0.0 and t_mask.any():
            student_feat = student_outputs["features"][t_mask]
            teacher_feat = batch["z_teacher"].to(device=device, dtype=student_feat.dtype)[t_mask]
            s_norm = F.normalize(student_feat, p=2, dim=-1)
            t_norm = F.normalize(teacher_feat, p=2, dim=-1)
            l_feat = 1.0 - torch.mean(torch.sum(s_norm * t_norm, dim=-1))
            losses["loss_feat"] = l_feat.item()
            total_loss += self.w_feat * l_feat
        else:
            losses["loss_feat"] = 0.0
            
        # 3. Calibrated Soft Teacher Response Distillation (L_resp)
        if self.w_resp > 0.0 and t_mask.any():
            pred_scores = student_outputs["score_scalar"][t_mask]
            teacher_raw = batch["a_teacher"].to(device=device, dtype=pred_scores.dtype)[t_mask]
            cal_teacher = torch.sigmoid(self.beta * (teacher_raw - self.threshold))
            eps = 1e-7
            p_s = torch.clamp(pred_scores, eps, 1.0 - eps)
            p_t = torch.clamp(cal_teacher, eps, 1.0 - eps)
            l_resp = - torch.mean(p_t * torch.log(p_s) + (1.0 - p_t) * torch.log(1.0 - p_s))
            losses["loss_resp"] = l_resp.item()
            total_loss += self.w_resp * l_resp
        else:
            losses["loss_resp"] = 0.0
            
        losses["loss_total"] = total_loss
        return total_loss, losses


class SimplifiedDistillationLoss(nn.Module):
    """Clean 2-term distillation loss without response-matching distortion:
    L_total = (L_Focal + L_Dice) + alpha * L_Feat
    where L_Feat is cosine distance between student 1024-dim embedding and DINOv2-Large feature.
    """
    def __init__(self, alpha=0.1, gamma=2.0, pos_weight=2.0):
        super().__init__()
        self.alpha = float(alpha)
        self.focal = FocalBCELoss(gamma=gamma, pos_weight=pos_weight)
        self.dice = TemporalDiceLoss()

    def forward(self, pred_dense, target_dense, student_feat=None, teacher_feat=None, y_flag=None):
        l_focal = self.focal(pred_dense, target_dense)
        l_dice = self.dice(pred_dense, target_dense, y_flag=y_flag)
        l_seg = l_focal + l_dice

        if student_feat is not None and teacher_feat is not None:
            s_norm = F.normalize(student_feat, p=2, dim=-1)
            t_norm = F.normalize(teacher_feat, p=2, dim=-1)
            sim = torch.sum(s_norm * t_norm, dim=-1)
            l_feat = torch.mean(1.0 - sim)
        else:
            l_feat = torch.tensor(0.0, device=pred_dense.device)

        total_loss = l_seg + self.alpha * l_feat
        return total_loss, {
            "loss_total": float(total_loss.item()),
            "loss_focal": float(l_focal.item()),
            "loss_dice": float(l_dice.item()),
            "loss_seg": float(l_seg.item()),
            "loss_feat": float(l_feat.item())
        }


