import os
import torch


class DistillationTrainer:
    """Manages student model training and distillation epochs."""
    
    def __init__(self, model, loss_fn, optimizer, device="cuda", checkpoint_dir="checkpoints"):
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_epoch_loss = 0.0
        running_losses = {"loss_sup": 0.0, "loss_feat": 0.0, "loss_resp": 0.0}
        num_batches = len(dataloader)
        
        for batch in dataloader:
            x = batch["x"].to(self.device)  # [B, C, L]
            
            self.optimizer.zero_grad()
            outputs = self.model(x)
            
            loss, loss_dict = self.loss_fn(outputs, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            
            total_epoch_loss += loss.item()
            for k in running_losses:
                running_losses[k] += loss_dict.get(k, 0.0)
                
        avg_loss = total_epoch_loss / max(1, num_batches)
        avg_components = {k: v / max(1, num_batches) for k, v in running_losses.items()}
        return avg_loss, avg_components
        
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                x = batch["x"].to(self.device)
                outputs = self.model(x)
                loss, _ = self.loss_fn(outputs, batch)
                total_loss += loss.item()
                
                all_preds.extend(outputs["score_scalar"].cpu().numpy())
                all_targets.extend(batch["y_ratio"].cpu().numpy())
                
        avg_loss = total_loss / max(1, num_batches)
        return avg_loss, all_preds, all_targets
        
    def save_checkpoint(self, filename="best_student.pt"):
        save_path = os.path.join(self.checkpoint_dir, filename)
        torch.save(self.model.state_dict(), save_path)
        return save_path

