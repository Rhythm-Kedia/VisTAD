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
    parser = argparse.ArgumentParser(description="Master TSAD Student Distillation & Ablation Runner")
    parser.add_argument("--study", type=str, default="all",
                        choices=["all", "backbone", "distillation", "smd_pooled", "channel_filter", "full_benchmarks"],
                        help="Ablation study to execute")
    parser.add_argument("--dataset", type=str, default="gecco",
                        choices=["gecco", "msl", "swat", "smd"],
                        help="Target dataset for backbone/distillation ablations")
    parser.add_argument("--cache_dir", type=str, default="cache",
                        help="Path to directory containing teacher caches")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs per experiment")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--output_summary", type=str, default="results/ablation_summary.md",
                        help="Summary markdown output file")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    cfg_map = {
        "gecco": {"config": "configs/gecco.yaml", "entity": "gecco_water", "thresh": 0.50},
        "msl": {"config": "configs/msl.yaml", "entity": "msl_test", "thresh": 0.68},
        "swat": {"config": "configs/swat.yaml", "entity": "swat_attack", "thresh": 0.63},
        "smd": {"config": "configs/smd.yaml", "entity": "machine-1-1", "thresh": 0.63}
    }
    
    experiments = []
    
    # 1. Study: Student Backbone Exploration
    if args.study in ["all", "backbone"]:
        tgt = args.dataset
        c = cfg_map[tgt]
        backbones = [
            {"arch": "cnn_1d", "layers": 3, "tag": "backbone_cnn3"},
            {"arch": "tcn", "layers": 5, "tag": "backbone_tcn5_residual"},
            {"arch": "timesnet", "layers": 3, "tag": "backbone_timesnet"},
            {"arch": "bilstm", "layers": 2, "tag": "backbone_bilstm"}
        ]
        for b in backbones:
            experiments.append({
                "name": f"Ablation_Backbone_{b['tag']}",
                "dataset": tgt,
                "entity": c["entity"],
                "config": c["config"],
                "arch": b["arch"],
                "num_layers": b["layers"],
                "tag": b["tag"],
                "epochs": args.epochs,
                "w_sup": 1.0, "w_feat": 0.1, "w_resp": 0.5 if tgt != "swat" else 1.0,
                "thresh": c["thresh"]
            })
            
    # 2. Study: Distillation Objective Ablation
    if args.study in ["all", "distillation"]:
        tgt = args.dataset
        c = cfg_map[tgt]
        loss_variants = [
            {"name": "Full_VisTAD", "tag": "loss_full", "w_sup": 1.0, "w_feat": 0.1, "w_resp": 1.0},
            {"name": "Supervised_Only", "tag": "loss_sup_only", "w_sup": 1.0, "w_feat": 0.0, "w_resp": 0.0},
            {"name": "No_Feature_Align", "tag": "loss_no_feat", "w_sup": 1.0, "w_feat": 0.0, "w_resp": 1.0},
            {"name": "No_Response_Match", "tag": "loss_no_resp", "w_sup": 1.0, "w_feat": 0.1, "w_resp": 0.0}
        ]
        for lv in loss_variants:
            experiments.append({
                "name": f"Ablation_Loss_{lv['name']}",
                "dataset": tgt,
                "entity": c["entity"],
                "config": c["config"],
                "arch": "tcn",
                "num_layers": 5,
                "tag": lv["tag"],
                "epochs": args.epochs,
                "w_sup": lv["w_sup"], "w_feat": lv["w_feat"], "w_resp": lv["w_resp"],
                "thresh": c["thresh"]
            })

    # 3. Study: SMD Multi-Entity Foundation Model (Pooled 28 Machines vs Single Machine)
    if args.study in ["all", "smd_pooled"]:
        experiments.append({
            "name": "SMD_Single_Machine_1_1",
            "dataset": "smd",
            "entity": "machine-1-1",
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
            "dataset": "smd",
            "entity": "all",
            "config": "configs/smd.yaml",
            "arch": "tcn",
            "num_layers": 5,
            "tag": "smd_pooled28_foundation",
            "epochs": max(10, args.epochs),
            "w_sup": 1.0, "w_feat": 0.1, "w_resp": 1.0,
            "thresh": 0.63
        })
        
    # 4. Study: Channel Filtering Strategy Ablation
    if args.study in ["all", "channel_filter"]:
        tgt = args.dataset
        c = cfg_map[tgt]
        strategies = ["median_shift", "top_k_variance", "top_k_zscore", "none"]
        for strat in strategies:
            experiments.append({
                "name": f"Ablation_Filter_{strat}",
                "dataset": tgt,
                "entity": c["entity"],
                "config": c["config"],
                "arch": "tcn",
                "num_layers": 5,
                "tag": f"filter_{strat}",
                "epochs": args.epochs,
                "w_sup": 1.0, "w_feat": 0.1, "w_resp": 0.5 if tgt != "swat" else 1.0,
                "thresh": c["thresh"],
                "overrides": [f"channel_filter.strategy={strat}"]
            })

    # 5. Full Benchmark Sweep across all 4 datasets
    if args.study in ["full_benchmarks"]:
        for ds, c in cfg_map.items():
            experiments.append({
                "name": f"FullBenchmark_{ds.upper()}_TCN",
                "dataset": ds,
                "entity": "all" if ds == "smd" else c["entity"],
                "config": c["config"],
                "arch": "tcn",
                "num_layers": 5,
                "tag": f"benchmark_{ds}_tcn",
                "epochs": args.epochs,
                "w_sup": 1.0, "w_feat": 0.1, "w_resp": 0.2 if ds == "msl" else (0.5 if ds == "gecco" else 1.0),
                "thresh": c["thresh"]
            })

    print(f"Total experiments queued: {len(experiments)}")
    
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
        ] + exp.get("overrides", [])
        run_cmd(train_cmd)
        
        # Target checkpoint path
        ent_tag = exp["entity"] if exp["entity"] != "all" else "pooled28"
        ckpt_path = f"checkpoints/{exp['dataset']}_{ent_tag}_{exp['arch']}_{exp['tag']}_best.pt"
        if not os.path.exists(ckpt_path):
            # Check fallback name pattern
            candidates = glob.glob(f"checkpoints/{exp['dataset']}_*_{exp['arch']}_{exp['tag']}_best.pt")
            if candidates:
                ckpt_path = candidates[0]
                
        # 2. Evaluate student on held-out slice
        eval_ent = "machine-1-1" if exp["entity"] == "all" else exp["entity"]
        eval_cmd = [
            "evaluate.py",
            "--config", exp["config"],
            "--dataset", exp["dataset"],
            "--entity", eval_ent,
            "--checkpoint", ckpt_path,
            "--arch", exp["arch"],
            "--num_layers", str(exp["num_layers"]),
            "--stride", "10",
            "--eval_slice", "test"
        ]
        run_cmd(eval_cmd)
        
        # Read eval results
        eval_json_path = f"results/{exp['dataset']}_{eval_ent}_{exp['arch']}_eval.json"
        if os.path.exists(eval_json_path):
            with open(eval_json_path, "r") as f:
                res = json.load(f)
            pm = res["point_metrics"]
            aff_pot = res["affiliation_metrics_pot"]
            summary_rows.append({
                "Experiment": exp["name"],
                "Dataset": exp["dataset"],
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
            f.write("# TSAD Student Distillation & Ablation Study Results\n\n")
            f.write(md_table + "\n")
            
        print("\n" + "=" * 70)
        print("                 ABLATION STUDY SUMMARY RESULTS                   ")
        print("=" * 70)
        print(md_table)
        print(f"\nSaved summary artifacts to:\n  - {args.output_summary}\n  - {csv_path}\n")


if __name__ == "__main__":
    main()
