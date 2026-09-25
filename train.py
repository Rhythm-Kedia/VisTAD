import os
import sys
import yaml
import argparse
import torch
from torch.utils.data import DataLoader

from src.dataset import TimeSeriesDataset, load_dataset_arrays
from src.models import build_student
from src.loss import MultiObjectiveKDLoss
from src.trainer import DistillationTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Student Model with VLM Distillation")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    # Allow overriding key knobs directly from command line
    parser.add_argument("overrides", nargs="*", help="Key-value overrides (e.g. student.architecture=bilstm)")
    return parser.parse_args()


def apply_overrides(cfg, overrides):
    """Applies dot-notation overrides like student.architecture=bilstm."""
    for override in overrides:
        if "=" in override:
            key_path, val = override.split("=", 1)
            keys = key_path.split(".")
            d = cfg
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            # Auto-cast numeric/bool values
            if val.lower() == "true":
                val = True
            elif val.lower() == "false":
                val = False
            else:
                try:
                    val = int(val)
                except ValueError:
                    try:
                        val = float(val)
                    except ValueError:
                        pass
            d[keys[-1]] = val
    return cfg


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = apply_overrides(cfg, args.overrides)
    
    print("==================================================================")
    print("              IE643 TSAD STUDENT DISTILLATION RUNNER              ")
    print("==================================================================")
    print(f"Dataset:            {cfg['dataset']['name']} ({cfg['dataset'].get('entity', '')})")
    print(f"Channel Filtering:  {cfg['channel_filter']['strategy']} (K={cfg['channel_filter']['k']})")
    print(f"Student Backbone:   {cfg['student']['architecture']}")
    print(f"Loss Objectives:    sup={cfg['training']['loss_weights']['w_sup']}, feat={cfg['training']['loss_weights']['w_feat']}, resp={cfg['training']['loss_weights']['w_resp']}")
    print("------------------------------------------------------------------")
    
    # 1. Load Raw Data
    raw_data, labels = load_dataset_arrays(
        dataset_name=cfg["dataset"]["name"],
        entity=cfg["dataset"].get("entity", "machine-1-1"),
        data_dir=cfg["dataset"].get("data_dir", "data")
    )
    
    # Check if Top-K filtering is active
    num_channels = raw_data.shape[1]
    if cfg["channel_filter"]["strategy"] != "none" and num_channels > cfg["channel_filter"]["k"]:
        in_channels = cfg["channel_filter"]["k"]
    else:
        in_channels = num_channels
    print(f"Student Input Channels: {in_channels} (Native: {num_channels})")
    
    # 2. Check for offline teacher cache
    cache_path = os.path.join(cfg["teacher"]["cache_dir"], f"{cfg['dataset']['name']}_teacher_cache.pt")
    teacher_cache = None
    if os.path.exists(cache_path):
        print(f"Loaded Teacher Cache: {cache_path}")
        teacher_cache = torch.load(cache_path)
    else:
        print(f"Notice: No teacher cache found at {cache_path}. Running with available targets.")
        
    # 3. Create Dataset & DataLoader
    train_dataset = TimeSeriesDataset(
        data=raw_data,
        labels=labels,
        window_size=cfg["dataset"]["window_size"],
        stride=cfg["dataset"]["stride"],
        filter_strategy=cfg["channel_filter"]["strategy"],
        k=cfg["channel_filter"]["k"],
        teacher_cache=teacher_cache
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        drop_last=False
    )
    print(f"Dataset created: {len(train_dataset)} windows of length {cfg['dataset']['window_size']}.")
    
    # 4. Build Student Model
    device = cfg["training"]["device"] if torch.cuda.is_available() else "cpu"
    student_model = build_student(
        name=cfg["student"]["architecture"],
        in_channels=in_channels,
        hidden_dim=cfg["student"].get("hidden_dim", 128),
        num_layers=cfg["student"].get("num_layers", 3),
        dropout=cfg["student"].get("dropout", 0.1),
        teacher_feat_dim=cfg["student"].get("teacher_feat_dim", 2048)
    )
    
    # 5. Build Loss Function & Optimizer
    loss_weights = cfg["training"]["loss_weights"]
    loss_fn = MultiObjectiveKDLoss(
        w_sup=loss_weights["w_sup"],
        w_feat=loss_weights["w_feat"],
        w_resp=loss_weights["w_resp"],
        temperature=cfg["training"].get("temperature", 2.0)
    )
    
    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"].get("weight_decay", 1e-4)
    )
    
    # 6. Train Student
    trainer = DistillationTrainer(
        model=student_model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=cfg["training"].get("checkpoint_dir", "checkpoints")
    )
    
    epochs = cfg["training"]["epochs"]
    print(f"Starting distillation training for {epochs} epochs on {device}...")
    
    best_loss = float("inf")
    for ep in range(1, epochs + 1):
        avg_loss, comps = trainer.train_epoch(train_loader)
        if ep % 5 == 0 or ep == 1 or ep == epochs:
            print(f"Epoch {ep:02d}/{epochs:02d} | Total Loss: {avg_loss:.4f} | Sup: {comps['loss_sup']:.4f} | Feat: {comps['loss_feat']:.4f} | Resp: {comps['loss_resp']:.4f}")
            
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_name = f"{cfg['dataset']['name']}_{cfg['student']['architecture']}_best.pt"
            saved_path = trainer.save_checkpoint(ckpt_name)
            
    print("------------------------------------------------------------------")
    print(f"Training Complete! Best model saved to: {saved_path}")


if __name__ == "__main__":
    main()

