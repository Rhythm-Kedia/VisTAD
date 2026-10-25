import os
import re
import glob
import json
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

from src.dataset import TimeSeriesDataset, load_dataset_arrays
from src.models import build_student
from src.loss import MultiObjectiveKDLoss
from src.trainer import DistillationTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Student Model with VLM Distillation")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name (smd, swat, msl, gecco) (overrides config)")
    parser.add_argument("--entity", type=str, default=None, help="Target entity name or 'all' for pooled training")
    parser.add_argument("--cache_dir", type=str, default=None, help="Custom teacher cache directory")
    parser.add_argument("--arch", type=str, default=None, help="Student backbone (cnn_1d, tcn, timesnet, bilstm)")
    parser.add_argument("--num_layers", type=int, default=None, help="Backbone depth / number of layers")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Training batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--w_sup", type=float, default=None, help="Supervised anchor loss weight")
    parser.add_argument("--w_feat", type=float, default=None, help="Latent feature distillation weight")
    parser.add_argument("--w_resp", type=float, default=None, help="Soft response distillation weight")
    parser.add_argument("--pos_weight", type=float, default=None, help="BCE loss positive class multiplier")
    parser.add_argument("--decision_threshold", type=float, default=None, help="Calibrated threshold for response loss")
    parser.add_argument("--tag", type=str, default=None, help="Experiment variant tag for checkpoint naming")
    parser.add_argument("--train_frac", type=float, default=0.5, help="Chronological fraction used for training")
    parser.add_argument("--val_frac", type=float, default=0.1, help="Fraction after train used for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
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


def split_series(raw, labels, train_frac, val_frac):
    """Chronological split preserving temporal ordering."""
    n = len(raw)
    a, b = int(n * train_frac), int(n * (train_frac + val_frac))
    return (raw[:a], labels[:a]), (raw[a:b], labels[a:b]), (raw[b:], labels[b:])


def window_metrics(scores, y_ratio):
    """AUROC / AUPRC / best-F1 of window scores against (y_ratio > 0)."""
    y = (np.asarray(y_ratio) > 0).astype(int)
    s = np.asarray(scores, dtype=float)
    if y.min() == y.max():
        return {"auroc": float("nan"), "auprc": float("nan"), "best_f1": float("nan"), "n": len(y)}
    p, r, _ = precision_recall_curve(y, s)
    f1 = (2 * p * r / np.clip(p + r, 1e-12, None)).max()
    return {"auroc": float(roc_auc_score(y, s)), "auprc": float(average_precision_score(y, s)),
            "best_f1": float(f1), "n": len(y)}


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = apply_overrides(cfg, args.overrides)
    
    # CLI Overrides
    if args.dataset:
        cfg["dataset"]["name"] = args.dataset
    if args.cache_dir:
        cfg["teacher"]["cache_dir"] = args.cache_dir
    if args.arch:
        cfg["student"]["architecture"] = args.arch
    if args.num_layers is not None:
        cfg["student"]["num_layers"] = args.num_layers
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
        
    lw = cfg["training"].setdefault("loss_weights", {})
    if args.w_sup is not None:
        lw["w_sup"] = args.w_sup
    if args.w_feat is not None:
        lw["w_feat"] = args.w_feat
    if args.w_resp is not None:
        lw["w_resp"] = args.w_resp
    if args.decision_threshold is not None:
        cfg["teacher"]["decision_threshold"] = args.decision_threshold
    pos_weight = args.pos_weight if args.pos_weight is not None else cfg["training"].get("pos_weight", 2.0)

    name = cfg["dataset"]["name"]
    cache_dir = cfg["teacher"]["cache_dir"]
    pattern = re.compile(rf"^{re.escape(name)}_(.+)_teacher_cache\.pt$")
    all_cached_entities = sorted(m.group(1) for p in glob.glob(os.path.join(cache_dir, f"{name}_*_teacher_cache.pt"))
                                 if (m := pattern.match(os.path.basename(p))))
    if not all_cached_entities:
        raise SystemExit(f"No teacher caches found in '{cache_dir}' for dataset '{name}'. "
                         f"Check --cache_dir or run run_teacher_cache.py first.")

    if args.entity and args.entity.lower() != "all":
        entities = [args.entity]
    else:
        entities = all_cached_entities

    arch = cfg["student"]["architecture"]
    tag_str = f"_{args.tag}" if args.tag else ""
    print(f"Dataset:            {name} ({len(entities)} entity/series cached)")
    print(f"Teacher Cache:      {cache_dir}")
    print(f"Channel Filtering:  {cfg['channel_filter']['strategy']} (K={cfg['channel_filter']['k']})")
    print(f"Student Backbone:   {arch} (Layers={cfg['student'].get('num_layers', 3)}, Hidden={cfg['student'].get('hidden_dim', 128)})")
    print(f"Loss Objectives:    sup={lw['w_sup']}, feat={lw['w_feat']}, resp={lw['w_resp']}")
    print(f"Decision Threshold: {cfg['teacher'].get('decision_threshold', 0.63)}")
    print(f"Split:              train={args.train_frac:.0%}, val={args.val_frac:.0%}, test={1 - args.train_frac - args.val_frac:.0%}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    w, stride = cfg["dataset"]["window_size"], cfg["dataset"]["stride"]
    bs_steps = cfg["dataset"].get("baseline_steps", 3000)

    def make_ds(raw, lab, cache, b_mean=None, b_std=None):
        return TimeSeriesDataset(data=raw, labels=lab, window_size=w, stride=stride,
                                 base_mean=b_mean, base_std=b_std,
                                 filter_strategy=cfg["channel_filter"]["strategy"],
                                 k=cfg["channel_filter"]["k"], teacher_cache=cache)

    train_sets, val_sets, test_sets = [], [], {}
    num_channels = None
    for ent in entities:
        raw_data, labels = load_dataset_arrays(
            dataset_name=name, entity=ent,
            data_dir=cfg["dataset"].get("data_dir", "data"))
        num_channels = raw_data.shape[1]
        base_m = np.mean(raw_data[:bs_steps], axis=0)
        base_s = np.std(raw_data[:bs_steps], axis=0) + 1e-6

        (tr_x, tr_y), (va_x, va_y), (te_x, te_y) = split_series(raw_data, labels, args.train_frac, args.val_frac)

        cache_path = os.path.join(cache_dir, f"{name}_{ent}_teacher_cache.pt")
        teacher_cache = None
        if os.path.exists(cache_path):
            teacher_cache = torch.load(cache_path, weights_only=False)
            usable = sum(1 for st in teacher_cache if st + w <= len(tr_x))
            print(f"[{ent}] cache: {len(teacher_cache)} windows ({usable} in train slice)")
        else:
            print(f"[{ent}] WARNING: no teacher cache at {cache_path}")

        if len(tr_x) >= w:
            train_sets.append(make_ds(tr_x, tr_y, teacher_cache, base_m, base_s))
        if len(va_x) >= w:
            val_sets.append(make_ds(va_x, va_y, None, base_m, base_s))
        if len(te_x) >= w:
            test_sets[ent] = make_ds(te_x, te_y, None, base_m, base_s)

    if cfg["channel_filter"]["strategy"] != "none" and num_channels > cfg["channel_filter"]["k"]:
        in_channels = cfg["channel_filter"]["k"]
    else:
        in_channels = num_channels
    print(f"Student Input Channels: {in_channels} (Native: {num_channels})")

    train_dataset = ConcatDataset(train_sets)
    bs = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=False, generator=gen)
    val_loader = DataLoader(ConcatDataset(val_sets), batch_size=bs, shuffle=False) if val_sets else None
    print(f"Train windows: {len(train_dataset)} | Val windows: {sum(len(d) for d in val_sets)} | Test entities: {len(test_sets)}")

    # Device selection
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Selected Compute Device: {device}")

    student_model = build_student(
        name=arch, in_channels=in_channels,
        hidden_dim=cfg["student"].get("hidden_dim", 128),
        num_layers=cfg["student"].get("num_layers", 3),
        dropout=cfg["student"].get("dropout", 0.1),
        teacher_feat_dim=cfg["student"].get("teacher_feat_dim", 2048)
    )
    
    loss_fn = MultiObjectiveKDLoss(
        w_sup=lw["w_sup"], w_feat=lw["w_feat"], w_resp=lw["w_resp"],
        temperature=cfg["training"].get("temperature", 2.0),
        decision_threshold=cfg["teacher"].get("decision_threshold", 0.63),
        pos_weight=pos_weight
    )
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=cfg["training"]["lr"],
                                  weight_decay=cfg["training"].get("weight_decay", 1e-4))
    epochs = cfg["training"]["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    ckpt_dir = cfg["training"].get("checkpoint_dir", "checkpoints")
    trainer = DistillationTrainer(model=student_model, loss_fn=loss_fn, optimizer=optimizer, device=device,
                                  checkpoint_dir=ckpt_dir)

    ent_tag = entities[0] if len(entities) == 1 else f"pooled{len(entities)}"
    ckpt_name = f"{name}_{ent_tag}_{arch}{tag_str}_best.pt"
    print(f"Distillation training starting for {epochs} epochs | Target ckpt: {ckpt_name}")

    best_val, saved_path = -1.0, None
    for ep in range(1, epochs + 1):
        avg_loss, comps = trainer.train_epoch(train_loader)
        scheduler.step()
        val_auc = float("nan")
        if val_loader is not None:
            preds, targets = trainer.predict(val_loader)
            val_auc = window_metrics(preds, targets)["auroc"]
        score = 0.5 if np.isnan(val_auc) else val_auc
        improved = score > best_val
        if improved:
            best_val = score
            saved_path = trainer.save_checkpoint(ckpt_name)
        if ep % 5 == 0 or ep == 1 or ep == epochs or improved:
            print(f"Epoch {ep:02d}/{epochs:02d} | Loss: {avg_loss:.4f} | Sup: {comps['loss_sup']:.4f} | "
                  f"Feat: {comps['loss_feat']:.4f} | Resp: {comps['loss_resp']:.4f} | "
                  f"Val AUROC: {val_auc:.4f}{'  *' if improved else ''}")

    print(f"Training complete. Best Val AUROC: {best_val:.4f}. Checkpoint: {saved_path}")

    # Evaluate on held-out test split
    trainer.load_checkpoint(ckpt_name)
    rows = []
    print(f"\n{'entity':<16}{'AUROC':>8}{'AUPRC':>8}{'bestF1':>8}{'windows':>9}")
    for ent, ds in test_sets.items():
        preds, targets = trainer.predict(DataLoader(ds, batch_size=bs, shuffle=False))
        m = window_metrics(preds, targets)
        rows.append(m)
        print(f"{ent:<16}{m['auroc']:>8.3f}{m['auprc']:>8.3f}{m['best_f1']:>8.3f}{m['n']:>9d}")
    valid = [r for r in rows if not np.isnan(r["auroc"])]
    if valid:
        mean_auc = float(np.mean([r["auroc"] for r in valid]))
        mean_prc = float(np.mean([r["auprc"] for r in valid]))
        mean_f1 = float(np.mean([r["best_f1"] for r in valid]))
        print(f"{'MEAN':<16}{mean_auc:>8.3f}{mean_prc:>8.3f}{mean_f1:>8.3f}")
        
    # Save training metadata JSON
    meta_path = os.path.join(ckpt_dir, f"{name}_{ent_tag}_{arch}{tag_str}_meta.json")
    meta = {
        "dataset": name,
        "entity": ent_tag,
        "arch": arch,
        "tag": args.tag,
        "epochs": epochs,
        "batch_size": bs,
        "lr": cfg["training"]["lr"],
        "loss_weights": lw,
        "pos_weight": pos_weight,
        "decision_threshold": cfg["teacher"].get("decision_threshold", 0.63),
        "best_val_auroc": best_val,
        "test_mean_auroc": mean_auc if valid else None,
        "test_mean_auprc": mean_prc if valid else None,
        "test_mean_f1": mean_f1 if valid else None,
        "checkpoint": saved_path
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved run metadata: {meta_path}")


if __name__ == "__main__":
    main()
