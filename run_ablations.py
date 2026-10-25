import os
import sys
import glob
import json
import argparse
import subprocess
import pandas as pd


def df_to_markdown(df):
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def run_cmd(cmd_list):
    print(f"\n>>> Running: {' '.join(cmd_list)}")
    res = subprocess.run([sys.executable] + cmd_list, check=True)
    return res.returncode


def parse_args():
    parser = argparse.ArgumentParser(description="Master TSAD Student Distillation & Symmetric Ablation Runner")
    parser.add_argument("--study", type=str, default="all",
                        choices=["all", "backbone", "distillation", "smd_pooled", "channel_filter", "full_benchmarks"],
                        help="Ablation study to execute")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all", "gecco", "msl", "swat", "smd"],
                        help="Target dataset(s) for ablations (default: 'all' for symmetric sweep)")
    parser.add_argument("--cache_dir", type=str, default="cache",
                        help="Path to directory containing teacher caches")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs per experiment")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--backbones", type=str, default="cnn_1d,tcn",
                        help="Comma-separated student backbones (default: 'cnn_1d,tcn')")
    parser.add_argument("--output_summary", type=str, default="results/ablation_summary.md",
                        help="Summary markdown output file")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    cfg_map = {
        "smd": {"config": "configs/smd.yaml", "entity": "machine-1-1", "thresh": 0.63, "name": "SMD"},
        "swat": {"config": "configs/swat.yaml", "entity": "swat_attack", "thresh": 0.63, "name": "SWaT"},
        "msl": {"config": "configs/msl.yaml", "entity": "msl_test", "thresh": 0.68, "name": "MSL"},
        "gecco": {"config": "configs/gecco.yaml", "entity": "gecco_water", "thresh": 0.50, "name": "GECCO"}
    }
    
    tgt_datasets = ["smd", "swat", "msl", "gecco"] if args.dataset == "all" else [args.dataset]
    experiments = []
    
    # 1. Study: Symmetric Student Backbone Exploration
    if args.study in ["all", "backbone"]:
        all_backbones = [
            {"arch": "cnn_1d", "layers": 3, "tag": "backbone_cnn3", "label": "1D-CNN (3L)"},
            {"arch": "tcn", "layers": 5, "tag": "backbone_tcn5_residual", "label": "Residual TCN (5L)"},
            {"arch": "timesnet", "layers": 3, "tag": "backbone_timesnet", "label": "TimesNet (3L)"},
            {"arch": "bilstm", "layers": 2, "tag": "backbone_bilstm", "label": "BiLSTM (2L)"}
        ]
        sel_archs = [x.strip().lower() for x in args.backbones.split(",")]
        backbones = [b for b in all_backbones if b["arch"].lower() in sel_archs]
        for tgt in tgt_datasets:
            c = cfg_map[tgt]
            train_entity = "all" if tgt == "smd" else c["entity"]
            for b in backbones:
                experiments.append({
                    "name": f"Ablation_Backbone_{b['tag']}_{tgt.upper()}",
                    "study_type": "backbone",
                    "variant": b["label"],
                    "dataset": tgt,
                    "entity": train_entity,
                    "eval_entity": c["entity"],
                    "config": c["config"],
                    "arch": b["arch"],
                    "num_layers": b["layers"],
                    "tag": f"{b['tag']}_{tgt}",
                    "epochs": args.epochs,
                    "w_sup": 1.5 if tgt == "msl" else 1.0,
                    "w_feat": 0.05 if tgt == "msl" else 0.1,
                    "w_resp": 0.2 if tgt == "msl" else (0.5 if tgt == "gecco" else 1.0),
                    "pos_weight": 4.0 if tgt == "msl" else 2.0,
                    "thresh": c["thresh"]
                })
            
    # 2. Study: Symmetric Distillation Objective Ablation
    if args.study in ["all", "distillation"]:
        loss_variants = [
            {"name": "Full_VisTAD", "label": "Full VisTAD (Sup + Feat + Resp)", "tag": "loss_full", "w_sup": 1.0, "w_feat": 0.1, "w_resp": 1.0},
            {"name": "Supervised_Only", "label": "Supervised Only (w_feat=0, w_resp=0)", "tag": "loss_sup_only", "w_sup": 1.0, "w_feat": 0.0, "w_resp": 0.0},
            {"name": "No_Feature_Align", "label": "w/o Feature Alignment (w_feat=0)", "tag": "loss_no_feat", "w_sup": 1.0, "w_feat": 0.0, "w_resp": 1.0},
            {"name": "No_Response_Match", "label": "w/o Response Matching (w_resp=0)", "tag": "loss_no_resp", "w_sup": 1.0, "w_feat": 0.1, "w_resp": 0.0}
        ]
        for tgt in tgt_datasets:
            c = cfg_map[tgt]
            train_entity = "all" if tgt == "smd" else c["entity"]
            for lv in loss_variants:
                # scale resp weight for msl/gecco if not 0.0
                resp_w = lv["w_resp"]
                if resp_w > 0:
                    resp_w = 0.2 if tgt == "msl" else (0.5 if tgt == "gecco" else 1.0)
                experiments.append({
                    "name": f"Ablation_Loss_{lv['name']}_{tgt.upper()}",
                    "study_type": "distillation",
                    "variant": lv["label"],
                    "dataset": tgt,
                    "entity": train_entity,
                    "eval_entity": c["entity"],
                    "config": c["config"],
                    "arch": "tcn",
                    "num_layers": 5,
                    "tag": f"{lv['tag']}_{tgt}",
                    "epochs": args.epochs,
                    "w_sup": lv["w_sup"],
                    "w_feat": lv["w_feat"],
                    "w_resp": resp_w,
                    "thresh": c["thresh"]
                })

    # 3. Study: SMD Multi-Entity Foundation Model (Pooled 28 Machines vs Single Machine)
    if args.study in ["all", "smd_pooled"]:
        experiments.append({
            "name": "SMD_Single_Machine_1_1",
            "study_type": "smd_scaling",
            "variant": "Single Entity (machine-1-1)",
            "dataset": "smd",
            "entity": "machine-1-1",
            "eval_entity": "machine-1-1",
            "config": "configs/smd.yaml",
            "arch": "tcn",
            "num_layers": 5,
            "tag": "smd_single_m1_1",
            "epochs": args.epochs,
            "w_sup": 1.0, "w_feat": 0.1, "w_resp": 1.0,
            "thresh": 0.63
        })
        experiments.append({
            "name": "SMD_Pooled_28_Foundation",
            "study_type": "smd_scaling",
            "variant": "Foundation Model (Pooled 28 Entities)",
            "dataset": "smd",
            "entity": "all",
            "eval_entity": "machine-1-1",
            "config": "configs/smd.yaml",
            "arch": "tcn",
            "num_layers": 5,
            "tag": "smd_pooled28_foundation",
            "epochs": max(10, args.epochs),
            "w_sup": 1.0, "w_feat": 0.1, "w_resp": 1.0,
            "thresh": 0.63
        })
        
    # 4. Study: Symmetric Channel Filtering Strategy Ablation
    if args.study in ["all", "channel_filter"]:
        strategies = [
            ("median_shift", "Median Shift (Proposed)"),
            ("top_k_variance", "Top-K Variance"),
            ("top_k_zscore", "Top-K Z-Score"),
            ("none", "None (All Native Channels)")
        ]
        for tgt in tgt_datasets:
            c = cfg_map[tgt]
            train_entity = "all" if tgt == "smd" else c["entity"]
            for strat_code, strat_label in strategies:
                experiments.append({
                    "name": f"Ablation_Filter_{strat_code}_{tgt.upper()}",
                    "study_type": "channel_filter",
                    "variant": strat_label,
                    "dataset": tgt,
                    "entity": train_entity,
                    "eval_entity": c["entity"],
                    "config": c["config"],
                    "arch": "tcn",
                    "num_layers": 5,
                    "tag": f"filter_{strat_code}_{tgt}",
                    "epochs": args.epochs,
                    "w_sup": 1.0,
                    "w_feat": 0.1,
                    "w_resp": 0.2 if tgt == "msl" else (0.5 if tgt == "gecco" else 1.0),
                    "thresh": c["thresh"],
                    "overrides": [f"channel_filter.strategy={strat_code}"]
                })

    # 5. Full Benchmark Sweep across all 4 datasets
    if args.study in ["full_benchmarks"]:
        for ds in tgt_datasets:
            c = cfg_map[ds]
            train_entity = "all" if ds == "smd" else c["entity"]
            experiments.append({
                "name": f"FullBenchmark_{ds.upper()}_TCN",
                "study_type": "benchmark",
                "variant": "VisTAD TCN (Default)",
                "dataset": ds,
                "entity": train_entity,
                "eval_entity": c["entity"],
                "config": c["config"],
                "arch": "tcn",
                "num_layers": 5,
                "tag": f"benchmark_{ds}_tcn",
                "epochs": args.epochs,
                "w_sup": 1.0,
                "w_feat": 0.1,
                "w_resp": 0.2 if ds == "msl" else (0.5 if ds == "gecco" else 1.0),
                "thresh": c["thresh"]
            })

    print(f"Total symmetric experiments queued: {len(experiments)}")
    summary_rows = []
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n==================================================================")
        print(f"[{i}/{len(experiments)}] RUNNING EXPERIMENT: {exp['name']}")
        print(f"Dataset: {exp['dataset']} | Entity: {exp['entity']} | Arch: {exp['arch']} (L={exp['num_layers']})")
        print(f"Loss weights: sup={exp['w_sup']}, feat={exp['w_feat']}, resp={exp['w_resp']}")
        print(f"==================================================================")
        
        # 1. Train student
        train_cmd = [
            "train.py",
            "--config", exp["config"],
            "--dataset", exp["dataset"],
            "--entity", exp["entity"],
            "--cache_dir", args.cache_dir,
            "--arch", exp["arch"],
            "--num_layers", str(exp["num_layers"]),
            "--epochs", str(exp["epochs"]),
            "--batch_size", str(args.batch_size),
            "--w_sup", str(exp["w_sup"]),
            "--w_feat", str(exp["w_feat"]),
            "--w_resp", str(exp["w_resp"]),
            "--decision_threshold", str(exp["thresh"]),
            "--tag", exp["tag"]
        ]
        if "pos_weight" in exp:
            train_cmd.extend(["--pos_weight", str(exp["pos_weight"])])
        train_cmd.extend(exp.get("overrides", []))
        run_cmd(train_cmd)
        
        # Target checkpoint path
        ent_tag = exp["entity"] if exp["entity"] != "all" else "pooled28"
        ckpt_path = f"checkpoints/{exp['dataset']}_{ent_tag}_{exp['arch']}_{exp['tag']}_best.pt"
        if not os.path.exists(ckpt_path):
            candidates = glob.glob(f"checkpoints/{exp['dataset']}_*_{exp['arch']}*_{exp['tag']}_best.pt")
            if candidates:
                ckpt_path = candidates[0]
            else:
                candidates_any = glob.glob(f"checkpoints/{exp['dataset']}_*_{exp['tag']}_best.pt")
                if candidates_any:
                    ckpt_path = candidates_any[0]
                
        # 2. Evaluate student on held-out slice with distinct tag
        eval_ent = exp.get("eval_entity", "machine-1-1" if exp["entity"] == "all" else exp["entity"])
        eval_cmd = [
            "evaluate.py",
            "--config", exp["config"],
            "--dataset", exp["dataset"],
            "--entity", eval_ent,
            "--checkpoint", ckpt_path,
            "--arch", exp["arch"],
            "--num_layers", str(exp["num_layers"]),
            "--stride", "10",
            "--eval_slice", "test",
            "--tag", exp["tag"]
        ]
        run_cmd(eval_cmd)
        
        # Read eval results with tag
        eval_json_path = f"results/{exp['dataset']}_{eval_ent}_{exp['arch']}_{exp['tag']}_eval.json"
        if not os.path.exists(eval_json_path):
            eval_json_path = f"results/{exp['dataset']}_{eval_ent}_{exp['arch']}_eval.json"
            
        if os.path.exists(eval_json_path):
            with open(eval_json_path, "r") as f:
                res = json.load(f)
            pm = res["point_metrics"]
            aff_pot = res["affiliation_metrics_pot"]
            summary_rows.append({
                "Experiment": exp["name"],
                "Study": exp.get("study_type", args.study),
                "Variant": exp.get("variant", exp["tag"]),
                "Dataset": exp["dataset"].upper(),
                "Entity": eval_ent,
                "Backbone": exp["arch"],
                "Layers": exp["num_layers"],
                "AUROC": pm["auroc"],
                "PR-AUC": pm["pr_auc"],
                "Best Point F1": pm["best_f1"],
                "POT Precision": pm["precision"],
                "POT Recall": pm["recall"],
                "Affiliation F1": aff_pot["aff_f1"],
                "Mean Temporal IoU": aff_pot["mean_tiou"],
                "Latency (ms/win)": res["latency_ms_per_window"],
                "Params": res["parameters"]
            })

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        csv_path = args.output_summary.replace(".md", ".csv")
        df.to_csv(csv_path, index=False)
        
        md_table = df_to_markdown(df)
        with open(args.output_summary, "w") as f:
            f.write("# TSAD Student Distillation & Symmetric Ablation Study Results\n\n")
            f.write("### Complete Evaluation Matrix\n\n")
            f.write(md_table + "\n\n")
            
            # If multiple datasets ran, build symmetric cross-benchmark pivot tables
            unique_datasets = df["Dataset"].unique()
            if len(unique_datasets) > 1 and "Variant" in df.columns:
                f.write("### Symmetric Cross-Benchmark Summary: Affiliation-F1 (Range-Aware)\n\n")
                piv_aff = df.pivot_table(index="Variant", columns="Dataset", values="Affiliation F1", aggfunc="mean")
                piv_aff["Mean"] = piv_aff.mean(axis=1)
                piv_aff = piv_aff.reset_index()
                piv_aff_csv = args.output_summary.replace(".md", "_pivot_aff_f1.csv")
                piv_aff.to_csv(piv_aff_csv, index=False)
                f.write(df_to_markdown(piv_aff.round(4)) + "\n\n")
                
                f.write("### Symmetric Cross-Benchmark Summary: Held-Out AUROC\n\n")
                piv_auroc = df.pivot_table(index="Variant", columns="Dataset", values="AUROC", aggfunc="mean")
                piv_auroc["Mean"] = piv_auroc.mean(axis=1)
                piv_auroc = piv_auroc.reset_index()
                piv_auroc_csv = args.output_summary.replace(".md", "_pivot_auroc.csv")
                piv_auroc.to_csv(piv_auroc_csv, index=False)
                f.write(df_to_markdown(piv_auroc.round(4)) + "\n\n")
            
        print("\n" + "=" * 70)
        print("                 ABLATION STUDY SUMMARY RESULTS                   ")
        print("=" * 70)
        print(md_table)
        print(f"\nSaved summary artifacts to:\n  - {args.output_summary}\n  - {csv_path}\n")


if __name__ == "__main__":
    main()
