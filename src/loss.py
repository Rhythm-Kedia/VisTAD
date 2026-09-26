import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiObjectiveKDLoss(nn.Module):
    """Composite Knowledge Distillation Loss for TSAD."""
    
    def __init__(self, w_sup=1.0, w_feat=0.5, w_resp=1.0, temperature=2.0):
        super().__init__()
        self.w_sup = w_sup
        self.w_feat = w_feat
        self.w_resp = w_resp
        self.temp = temperature
        
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        
    def forward(self, student_outputs, batch):
        losses = {}
        total_loss = 0.0
        
        # 1. Supervised Ground-Truth Anchor Loss (L_sup)
        if self.w_sup > 0.0:
            pred_dense = student_outputs["logits_dense"]
            gt_dense = batch["y_dense"].to(pred_dense.device)
            l_dense = self.bce_loss(pred_dense, gt_dense)
            
            pred_scalar = student_outputs["score_scalar"]
            gt_flag = batch["y_flag"].to(pred_scalar.device)
            l_scalar = self.bce_loss(pred_scalar, gt_flag)
            
            l_sup = 0.7 * l_dense + 0.3 * l_scalar
            losses["loss_sup"] = l_sup.item()
            total_loss += self.w_sup * l_sup
        else:
            losses["loss_sup"] = 0.0
            
        # 2. Teacher Latent Feature Distillation (L_feat)
        device = student_outputs["score_scalar"].device
        has_t = batch.get("has_teacher")
        if has_t is not None:
            t_mask = has_t.to(device).bool()
        else:
            t_mask = torch.ones(len(student_outputs["score_scalar"]), dtype=torch.bool, device=device)
            
        if self.w_feat > 0.0 and t_mask.any():
            student_feat = student_outputs["features"][t_mask]
            teacher_feat = batch["z_teacher"].to(device=device, dtype=student_feat.dtype)[t_mask]
            l_feat = self.mse_loss(student_feat, teacher_feat)
            losses["loss_feat"] = l_feat.item()
            total_loss += self.w_feat * l_feat
        else:
            losses["loss_feat"] = 0.0
            
        # 3. Soft Teacher Response Distillation (L_resp)
        if self.w_resp > 0.0 and t_mask.any():
            pred_scores = student_outputs["score_scalar"][t_mask]
            teacher_scores = batch["a_teacher"].to(device=device, dtype=pred_scores.dtype)[t_mask]
            eps = 1e-7
            p_s = torch.clamp(pred_scores, eps, 1.0 - eps)
            p_t = torch.clamp(teacher_scores, eps, 1.0 - eps)
            l_resp = - torch.mean(p_t * torch.log(p_s) + (1.0 - p_t) * torch.log(1.0 - p_s))
            losses["loss_resp"] = l_resp.item()
            total_loss += self.w_resp * l_resp
        else:
            losses["loss_resp"] = 0.0
            
        losses["loss_total"] = total_loss
        return total_loss, losses
