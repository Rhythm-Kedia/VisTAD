import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiObjectiveKDLoss(nn.Module):
    """Composite Knowledge Distillation Loss for TSAD."""
    
    def __init__(self, w_sup=1.0, w_feat=0.1, w_resp=1.0, temperature=2.0, decision_threshold=0.63, beta=8.0):
        super().__init__()
        self.w_sup = w_sup
        self.w_feat = w_feat
        self.w_resp = w_resp
        self.temp = temperature
        self.threshold = decision_threshold
        self.beta = beta
        
        self.bce_loss = nn.BCELoss()
        
    def forward(self, student_outputs, batch):
        losses = {}
        total_loss = 0.0
        
        # 1. Supervised Ground-Truth Anchor Loss (L_sup)
        if self.w_sup > 0.0:
            pred_dense = student_outputs["logits_dense"]
            gt_dense = batch["y_dense"].to(pred_dense.device)
            eps = 1e-7
            p_d = torch.clamp(pred_dense, eps, 1.0 - eps)
            # Imbalance-weighted BCE for dense steps (positive anomalies are rare)
            l_dense = - torch.mean(2.0 * gt_dense * torch.log(p_d) + (1.0 - gt_dense) * torch.log(1.0 - p_d))
            
            pred_scalar = student_outputs["score_scalar"]
            gt_flag = batch["y_flag"].to(pred_scalar.device)
            p_s = torch.clamp(pred_scalar, eps, 1.0 - eps)
            l_scalar = - torch.mean(2.0 * gt_flag * torch.log(p_s) + (1.0 - gt_flag) * torch.log(1.0 - p_s))
            
            l_sup = 0.6 * l_dense + 0.4 * l_scalar
            losses["loss_sup"] = l_sup.item()
            total_loss += self.w_sup * l_sup
        else:
            losses["loss_sup"] = 0.0

            
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
            # Soft recalibration around decision threshold
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

