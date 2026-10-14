import os
import re
import glob
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
    parser.add_argument("--train_frac", type=float, default=0.5, help="Chronological fraction used for training")
    parser.add_argument("--val_frac", type=float, default=0.1, help="Fraction after train used for validation (checkpoint selection)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (shuffling, init, dropout)")
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
    """Chronological split. The train slice starts at index 0, so teacher-cache window starts
    (computed on the full series) stay valid for it and no change to run_teacher_cache.py is needed."""
    n = len(raw)
    a, b = int(n * train_frac), int(n * (train_frac + val_frac))
    return (raw[:a], labels[:a]), (raw[a:b], labels[a:b]), (raw[b:], labels[b:])


def window_metrics(scores, y_ratio):
    """AUROC / AUPRC / best-F1 of window scores against (y_ratio > 0). NaN if only one class present."""
    y = (np.asarray(y_ratio) > 0).astype(int)
    s = np.asarray(scores, dtype=float)
    if y.min() == y.max():
        return {"auroc": float("nan"), "auprc": float("nan"), "best_f1": float("nan"), "n": len(y)}
    p, r, _ = precision_recall_curve(y, s)
    f1 = (2 * p * r / np.clip(p + r, 1e-12, None)).max()
    return {"auroc": roc_auc_score(y, s), "auprc": average_precision_score(y, s),
            "best_f1": float(f1), "n": len(y)}


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = apply_overrides(cfg, args.overrides)

    # No entity in config or CLI: train on every series that has a teacher cache in cache_dir.
    # (Series are still handled one at a time internally: windows must not span two machines.)
    name = cfg["dataset"]["name"]
    pattern = re.compile(rf"^{re.escape(name)}_(.+)_teacher_cache\.pt$")
    entities = sorted(m.group(1) for p in glob.glob(os.path.join(cfg["teacher"]["cache_dir"], f"{name}_*_teacher_cache.pt"))
                      if (m := pattern.match(os.path.basename(p))))
    if not entities:
        raise SystemExit(f"No teacher caches found in '{cfg['teacher']['cache_dir']}' for dataset '{name}'. "
                         f"Run run_teacher_cache.py first.")

    print("==================================================================")
    print("              IE643 TSAD STUDENT DISTILLATION RUNNER              ")
    print("==================================================================")
    print(f"Dataset:            {cfg['dataset']['name']} ({len(entities)} series with a teacher cache)")
    print(f"Channel Filtering:  {cfg['channel_filter']['strategy']} (K={cfg['channel_filter']['k']})")
    print(f"Student Backbone:   {cfg['student']['architecture']}")
    lw = cfg["training"]["loss_weights"]
    print(f"Loss Objectives:    sup={lw['w_sup']}, feat={lw['w_feat']}, resp={lw['w_resp']}")
    print(f"Split (chronological): train={args.train_frac:.0%} val={args.val_frac:.0%} "
          f"test={1 - args.train_frac - args.val_frac:.0%}")
    print("------------------------------------------------------------------")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    gen = torch.Generator().manual_seed(args.seed)        # makes the shuffle order reproducible

    w, stride = cfg["dataset"]["window_size"], cfg["dataset"]["stride"]

    def make_ds(raw, lab, cache):
        return TimeSeriesDataset(data=raw, labels=lab, window_size=w, stride=stride,
                                 filter_strategy=cfg["channel_filter"]["strategy"],
                                 k=cfg["channel_filter"]["k"], teacher_cache=cache)

    # 1. Load every entity, split chronologically, build datasets
    train_sets, val_sets, test_sets = [], [], {}
    num_channels = None
    for ent in entities:
        raw_data, labels = load_dataset_arrays(
            dataset_name=cfg["dataset"]["name"], entity=ent,
            data_dir=cfg["dataset"].get("data_dir", "data"))
        num_channels = raw_data.shape[1]
        (tr_x, tr_y), (va_x, va_y), (te_x, te_y) = split_series(raw_data, labels, args.train_frac, args.val_frac)

        # Teacher cache is keyed by window start in the full series; only valid for the train slice (starts at 0).
        cache_path = os.path.join(cfg["teacher"]["cache_dir"], f"{cfg['dataset']['name']}_{ent}_teacher_cache.pt")
        teacher_cache = None
        if os.path.exists(cache_path):
            teacher_cache = torch.load(cache_path, weights_only=False)
            usable = sum(1 for st in teacher_cache if st + w <= len(tr_x))
            print(f"[{ent}] teacher cache: {len(teacher_cache)} windows, {usable} fall inside the train slice")
        else:
            print(f"[{ent}] WARNING: no teacher cache at {cache_path} -> no distillation targets for this entity")

        if len(tr_x) >= w:
            train_sets.append(make_ds(tr_x, tr_y, teacher_cache))
        if len(va_x) >= w:
            val_sets.append(make_ds(va_x, va_y, None))
        if len(te_x) >= w:
            test_sets[ent] = make_ds(te_x, te_y, None)

    if cfg["channel_filter"]["strategy"] != "none" and num_channels > cfg["channel_filter"]["k"]:
        in_channels = cfg["channel_filter"]["k"]
    else:
        in_channels = num_channels
    print(f"Student Input Channels: {in_channels} (Native: {num_channels})")

    train_dataset = ConcatDataset(train_sets)
    bs = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=False, generator=gen)
    val_loader = DataLoader(ConcatDataset(val_sets), batch_size=bs, shuffle=False) if val_sets else None
    print(f"Train windows: {len(train_dataset)} | Val windows: {sum(len(d) for d in val_sets)} | "
          f"Test entities: {len(test_sets)}")

    # 2. Build student, loss, optimizer
    device = cfg["training"]["device"] if torch.cuda.is_available() else "cpu"
    student_model = build_student(
        name=cfg["student"]["architecture"], in_channels=in_channels,
        hidden_dim=cfg["student"].get("hidden_dim", 128),
        num_layers=cfg["student"].get("num_layers", 3),
        dropout=cfg["student"].get("dropout", 0.1),
        teacher_feat_dim=cfg["student"].get("teacher_feat_dim", 2048))
    loss_fn = MultiObjectiveKDLoss(w_sup=lw["w_sup"], w_feat=lw["w_feat"], w_resp=lw["w_resp"],
                                   temperature=cfg["training"].get("temperature", 2.0))
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=cfg["training"]["lr"],
                                  weight_decay=cfg["training"].get("weight_decay", 1e-4))
    trainer = DistillationTrainer(model=student_model, loss_fn=loss_fn, optimizer=optimizer, device=device,
                                  checkpoint_dir=cfg["training"].get("checkpoint_dir", "checkpoints"))

    # 3. Train; select the checkpoint on validation AUROC (not training loss)
    epochs = cfg["training"]["epochs"]
    tag = entities[0] if len(entities) == 1 else f"pooled{len(entities)}"
    ckpt_name = f"{cfg['dataset']['name']}_{tag}_{cfg['student']['architecture']}_best.pt"
    print(f"Starting distillation training for {epochs} epochs on {device}...")

    best_val, saved_path = -1.0, None
    for ep in range(1, epochs + 1):
        avg_loss, comps = trainer.train_epoch(train_loader)
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

    print("------------------------------------------------------------------")
    print(f"Training complete. Best val AUROC {best_val:.4f}. Checkpoint: {saved_path}")

    # 4. Test the best checkpoint on each entity's held-out (last) slice
    trainer.load_checkpoint(ckpt_name)
    rows = []
    print(f"{'entity':<14}{'AUROC':>8}{'AUPRC':>8}{'bestF1':>8}{'windows':>9}")
    for ent, ds in test_sets.items():
        preds, targets = trainer.predict(DataLoader(ds, batch_size=bs, shuffle=False))
        m = window_metrics(preds, targets)
        rows.append(m)
        print(f"{ent:<14}{m['auroc']:>8.3f}{m['auprc']:>8.3f}{m['best_f1']:>8.3f}{m['n']:>9d}")
    valid = [r for r in rows if not np.isnan(r["auroc"])]
    if valid:
        print(f"{'MEAN':<14}{np.mean([r['auroc'] for r in valid]):>8.3f}"
              f"{np.mean([r['auprc'] for r in valid]):>8.3f}{np.mean([r['best_f1'] for r in valid]):>8.3f}")


if __name__ == "__main__":
    main()