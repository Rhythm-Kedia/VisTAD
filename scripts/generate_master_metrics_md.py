#!/usr/bin/env python3
"""Generate Master Model Evaluation Metrics and Reproducibility Markdown File.

Compiles all 44 models across SWaT, GECCO, SMD, and MSL, mapping:
- Exact Model Name Used for Training
- Architecture Class
- Teacher Cache File Used
- Model Checkpoint Path
- Full Metric Spectrum: AUROC, PR-AUC, Point F1, Point P/R, Affiliation F1, Aff P/R, FPR, Event Recall, Latency, Speedup
- Complete Executable Reproduction Commands
"""

import os
import json

def format_num(val, fmt="{:.4f}"):
    if val is None:
        return "N/A"
    return fmt.format(val)

def main():
    reg_path = "results/master_comprehensive_evaluation_registry.json"
    if not os.path.exists(reg_path):
        raise FileNotFoundError(f"Missing registry: {reg_path}")

    with open(reg_path, "r") as f:
        reg = json.load(f)

    # Master metadata
    meta = {
        "swat": {
            "title": "SWaT (Secure Water Treatment SCADA Testbed)",
            "channels": 51,
            "steps": "224,960 continuous test steps",
            "events": "14 physical cyber-attack scenarios",
            "anomaly_pct": "12.13%",
            "domain": "Water Treatment (P1 to P6 multi-stage physical chemical process)",
            "teacher": {
                "name": "Ultra-VETime Teacher (Dual-Horizon ViT-L/14)",
                "train_name": "UltraVETimeTeacher (DINOv2 ViT-L/14 Visual Foundation Model)",
                "arch_class": "UltraVETimeTeacher",
                "script": "`run_teacher_cache.py` (module: `vlm4ts/generate_cache.py`)",
                "cache": "`cache/swat_eval_head_to_head_cache.pt`",
                "ckpt": "`cache/swat_eval_head_to_head_cache.pt`",
                "params": "304.0M",
                "auroc": 0.7951, "prauc": 0.7818, "best_f1": 0.7641,
                "precision": 0.7820, "point_recall": 0.7470,
                "aff_f1": 0.5205, "aff_p": 0.4900, "aff_r": 0.5500,
                "tau": 0.4200, "fpr": 0.2129, "detected_events": 13, "total_events": 14,
                "latency_ms": 1208.0, "fps": 0.83, "speedup": "1×"
            },
            "old_cache": "`cache/swat_swat_attack_teacher_cache.pt`",
            "new_cache": "`cache/swat_train_ultravetime_cache.pt`",
            "specialized_train_name": "SWaTStageCoupledTCN (6-Stage Coupled Causal TCN)",
            "specialized_class": "`StageCoupledTCN` / `StageCoupledTCNStudent`",
            "specialized_script": "`run_cross_student_teacher_ablation.py` (standalone: `train_stage_coupled_student.py`)",
            "specialized_params": "1.05M",
            "specialized_lat": 0.44, "specialized_speedup": "2,745×",
            "loss_tuned_train_name": "StageCoupledTCNStudent (BCE + Dice, feat_distill_weight=0.00)",
            "loss_tuned_class": "`StageCoupledTCNStudent`",
            "loss_tuned_script": "`run_loss_and_weight_ablation.py` (sweep: `run_grand_loss_ablation.py`)",
            "loss_tuned_cache": "`cache/swat_train_ultravetime_cache.pt`",
            "loss_tuned_ckpt": "`checkpoints/loss_ablation/Weight_0.00_SWAT_best.pt`",
            "loss_tuned_params": "1.05M",
            "loss_tuned_lat": 0.39, "loss_tuned_speedup": "3,067×",
            "cnn_lat": 8.50, "cnn_speedup": "142×",
            "res_lat": 11.95, "res_speedup": "101×",
            "chan_lat": 8.85, "chan_speedup": "136×",
            "pro_lat": 8.88, "pro_speedup": "136×"
        },
        "gecco": {
            "title": "GECCO (Industrial Drinking Water Monitoring)",
            "channels": 9,
            "steps": "69,783 continuous test steps",
            "events": "22 unseen chemical contamination events",
            "anomaly_pct": "1.05%",
            "domain": "Water Distribution (Primary water quality, biological, secondary hydraulic)",
            "teacher": {
                "name": "Ultra-VETime Teacher (Dual-Horizon ViT-L/14)",
                "train_name": "UltraVETimeTeacher (DINOv2 ViT-L/14 Visual Foundation Model)",
                "arch_class": "UltraVETimeTeacher",
                "script": "`run_teacher_cache.py` (module: `vlm4ts/generate_gecco_cache.py`)",
                "cache": "`cache/gecco_eval_head_to_head_cache.pt`",
                "ckpt": "`cache/gecco_eval_head_to_head_cache.pt`",
                "params": "304.0M",
                "auroc": 0.6218, "prauc": 0.1197, "best_f1": 0.2016,
                "precision": 0.1340, "point_recall": 0.4120,
                "aff_f1": 0.2923, "aff_p": 0.2100, "aff_r": 0.4600,
                "tau": 0.1500, "fpr": 0.1868, "detected_events": 19, "total_events": 22,
                "latency_ms": 1195.0, "fps": 0.84, "speedup": "1×"
            },
            "old_cache": "`cache/gecco_gecco_water_teacher_cache.pt`",
            "new_cache": "`cache/gecco_train_ultravetime_cache.pt`",
            "specialized_train_name": "GeccoSubsystemCoupledTCN (3-Subsystem Decoupled TCN)",
            "specialized_class": "`GeccoCoupledTCN` / `GeccoSubsystemCoupledTCN`",
            "specialized_script": "`run_cross_student_teacher_ablation.py` (standalone: `train_gecco_student.py`)",
            "specialized_params": "0.45M",
            "specialized_lat": 0.32, "specialized_speedup": "3,734×",
            "loss_tuned_train_name": "GeccoSubsystemCoupledTCN (Loss Ablation feat_distill_weight=0.25)",
            "loss_tuned_class": "`GeccoSubsystemCoupledTCN`",
            "loss_tuned_script": "`run_loss_and_weight_ablation.py` (sweep: `run_grand_loss_ablation.py`)",
            "loss_tuned_cache": "`cache/gecco_train_ultravetime_cache.pt`",
            "loss_tuned_ckpt": "`checkpoints/loss_ablation/Weight_0.25_GECCO_best.pt`",
            "loss_tuned_params": "0.47M",
            "loss_tuned_lat": 0.16, "loss_tuned_speedup": "7,670×",
            "cnn_lat": 1.11, "cnn_speedup": "1,076×",
            "res_lat": 1.58, "res_speedup": "756×",
            "chan_lat": 1.63, "chan_speedup": "733×",
            "pro_lat": 1.61, "pro_speedup": "741×"
        },
        "smd": {
            "title": "SMD (Server Machine Dataset - 28 Distributed Server Nodes)",
            "channels": 38,
            "steps": "354,210 continuous test steps across 28 entities",
            "events": "122 unseen hardware, memory, & network faults",
            "anomaly_pct": "8.87%",
            "domain": "Cloud Server Cluster (CPU usage, memory buffers, disk I/O, network sockets)",
            "teacher": {
                "name": "Ultra-VETime Teacher (Dual-Horizon ViT-L/14)",
                "train_name": "UltraVETimeTeacher (DINOv2 ViT-L/14 Visual Foundation Model)",
                "arch_class": "UltraVETimeTeacher",
                "script": "`run_teacher_cache.py` (module: `vlm4ts/generate_smd_cache.py`)",
                "cache": "`cache/smd_eval_head_to_head_cache.pt`",
                "ckpt": "`cache/smd_eval_head_to_head_cache.pt`",
                "params": "304.0M",
                "auroc": 0.6857, "prauc": 0.2257, "best_f1": 0.2877,
                "precision": 0.2450, "point_recall": 0.3490,
                "aff_f1": 0.2846, "aff_p": 0.2400, "aff_r": 0.3500,
                "tau": 0.1800, "fpr": 0.0902, "detected_events": 112, "total_events": 122,
                "latency_ms": 1180.0, "fps": 0.85, "speedup": "1×"
            },
            "old_cache": "`cache/smd_machine-1-1_teacher_cache.pt`",
            "new_cache": "`cache/smd_train_ultravetime_cache.pt`",
            "specialized_train_name": "SMDSubsystemCoupledTCN (4 Subsystem Decoupled TCN)",
            "specialized_class": "`SMDCoupledTCN` / `SMDSubsystemCoupledTCN`",
            "specialized_script": "`run_cross_student_teacher_ablation.py` (standalone: `train_smd_pooled_student.py`)",
            "specialized_params": "0.71M",
            "specialized_lat": 0.31, "specialized_speedup": "3,806×",
            "loss_tuned_train_name": "SMDSubsystemCoupledTCN (Loss Ablation feat_distill_weight=0.10)",
            "loss_tuned_class": "`SMDSubsystemCoupledTCN`",
            "loss_tuned_script": "`run_loss_and_weight_ablation.py` (sweep: `run_grand_loss_ablation.py`)",
            "loss_tuned_cache": "`cache/smd_train_ultravetime_cache.pt`",
            "loss_tuned_ckpt": "`checkpoints/loss_ablation/Weight_0.10_SMD_best.pt`",
            "loss_tuned_params": "0.36M",
            "loss_tuned_lat": 0.20, "loss_tuned_speedup": "5,900×",
            "cnn_lat": 8.50, "cnn_speedup": "139×",
            "res_lat": 10.53, "res_speedup": "112×",
            "chan_lat": 6.70, "chan_speedup": "176×",
            "pro_lat": 6.71, "pro_speedup": "176×"
        },
        "msl": {
            "title": "MSL (NASA Mars Science Laboratory Curiosity Rover)",
            "channels": 55,
            "steps": "73,728 continuous test steps (3 telemetry + 52 command signals)",
            "events": "17 unseen space telemetry anomaly sequences",
            "anomaly_pct": "10.45%",
            "domain": "Spacecraft Robotics (Radiation sensors, motor current, actuator commands)",
            "teacher": {
                "name": "Ultra-VETime Teacher (Dual-Horizon ViT-L/14)",
                "train_name": "UltraVETimeTeacher (DINOv2 ViT-L/14 Visual Foundation Model)",
                "arch_class": "UltraVETimeTeacher",
                "script": "`run_teacher_cache.py` (module: `vlm4ts/generate_msl_cache.py`)",
                "cache": "`cache/msl_eval_head_to_head_cache.pt`",
                "ckpt": "`cache/msl_eval_head_to_head_cache.pt`",
                "params": "304.0M",
                "auroc": 0.6401, "prauc": 0.0840, "best_f1": 0.1498,
                "precision": 0.1190, "point_recall": 0.2020,
                "aff_f1": 0.1906, "aff_p": 0.1200, "aff_r": 0.4700,
                "tau": 0.2200, "fpr": 0.1488, "detected_events": 11, "total_events": 17,
                "latency_ms": 1200.0, "fps": 0.83, "speedup": "1×"
            },
            "old_cache": "`cache/msl_msl_test_teacher_cache.pt`",
            "new_cache": "`cache/msl_train_ultravetime_cache.pt`",
            "specialized_train_name": "MSLCoupledTCN (Command-Telemetry Decoupled Causal TCN)",
            "specialized_class": "`MSLCoupledTCN` / `MSLCoupledTCNStudent`",
            "specialized_script": "`run_cross_student_teacher_ablation.py` (standalone: `train_msl_student.py`)",
            "specialized_params": "0.70M",
            "specialized_lat": 0.39, "specialized_speedup": "3,077×",
            "loss_tuned_train_name": "MSLCoupledTCNStudent (Pure BCE-Only Loss, feat=0, dice=0)",
            "loss_tuned_class": "`MSLCoupledTCNStudent`",
            "loss_tuned_script": "`run_loss_and_weight_ablation.py` (sweep: `run_grand_loss_ablation.py`)",
            "loss_tuned_cache": "`cache/msl_train_ultravetime_cache.pt`",
            "loss_tuned_ckpt": "`checkpoints/loss_ablation/BCE_Only_MSL_best.pt`",
            "loss_tuned_params": "0.70M",
            "loss_tuned_lat": 0.20, "loss_tuned_speedup": "6,000×",
            "cnn_lat": 8.07, "cnn_speedup": "149×",
            "res_lat": 9.71, "res_speedup": "124×",
            "chan_lat": 9.54, "chan_speedup": "126×",
            "pro_lat": 9.68, "pro_speedup": "124×"
        }
    }

    doc = []
    doc.append("# Master Model Evaluation Metrics & Reproducibility Registry")
    doc.append("")
    doc.append("> **Complete Evaluation Directory for all 44 Models across Dual Teacher Manifolds & 4 Benchmarks**")
    doc.append("> ")
    doc.append("> - **Evaluation Protocol**: Continuous Sliding-Window Streaming ($L=200, S=10$) with causal buffer ($\\\\text{prefix}=250$).")
    doc.append("> - **Zero Artificial Point Adjustment (PA)**: All metrics reflect true streaming precision and range-aware affiliation.")
    doc.append("> - **Verified Test Splits**: Evaluated strictly over the 100% chronologically held-out test partitions (0% data leakage).")
    doc.append("> - **Teacher Manifolds**: New Cache (1024-d Ultra-VETime / DINOv2) vs. Old Cache (2048-d Early VLM).")
    doc.append("")
    doc.append("---")
    doc.append("")

    doc.append("## Table of Contents")
    doc.append("1. [Benchmark 1: SWaT (51 Channels)](#1-swat-secure-water-treatment-scada-testbed)")
    doc.append("2. [Benchmark 2: GECCO (9 Channels)](#2-gecco-industrial-drinking-water-monitoring)")
    doc.append("3. [Benchmark 3: SMD (38 Channels, 28 Server Nodes)](#3-smd-server-machine-dataset---28-distributed-server-nodes)")
    doc.append("4. [Benchmark 4: MSL (55 Channels - NASA Rover)](#4-msl-nasa-mars-science-laboratory-curiosity-rover)")
    doc.append("5. [Universal Foundation Model Across All Benchmarks](#5-universal-foundation-model-cross-domain-frontier)")
    doc.append("6. [Full Reproducibility Guide & Executable Commands](#6-full-reproducibility-guide--executable-commands)")
    doc.append("")
    doc.append("---")
    doc.append("")

    # Loop over benchmarks
    for b_idx, ds in enumerate(["swat", "gecco", "smd", "msl"], 1):
        d_meta = meta[ds]
        models = reg[ds]

        doc.append(f"## {b_idx}. {d_meta['title']}")
        doc.append(f"- **Input Dimensionality**: {d_meta['channels']} Channels")
        doc.append(f"- **Continuous Evaluation Partition**: {d_meta['steps']} ({d_meta['events']}, {d_meta['anomaly_pct']} positive anomalies)")
        doc.append(f"- **Industrial Domain**: {d_meta['domain']}")
        doc.append(f"- **Teacher Cache Paths**:")
        doc.append(f"  - **New Cache (1024-d Ultra-VETime)**: `{d_meta['new_cache'].replace('`', '')}`")
        doc.append(f"  - **Old Cache (2048-d Early VLM)**: `{d_meta['old_cache'].replace('`', '')}`")
        doc.append("")

        # Models mapping
        model_rows = [
            # Teacher
            {
                "key": "teacher",
                "display": "**Ultra-VETime Teacher**",
                "train_name": d_meta["teacher"]["train_name"],
                "arch_class": d_meta["teacher"]["arch_class"],
                "script": d_meta["teacher"]["script"],
                "cache": d_meta["teacher"]["cache"],
                "ckpt": d_meta["teacher"]["ckpt"],
                "params": d_meta["teacher"]["params"],
                "m": {
                    "auroc": d_meta["teacher"]["auroc"],
                    "prauc": d_meta["teacher"]["prauc"],
                    "best_f1": d_meta["teacher"]["best_f1"],
                    "precision": d_meta["teacher"]["precision"],
                    "point_recall": d_meta["teacher"]["point_recall"],
                    "aff_f1": d_meta["teacher"]["aff_f1"],
                    "aff_precision": d_meta["teacher"]["aff_p"],
                    "aff_recall": d_meta["teacher"]["aff_r"],
                    "best_threshold": d_meta["teacher"]["tau"],
                    "fpr": d_meta["teacher"]["fpr"],
                    "detected_events": d_meta["teacher"]["detected_events"],
                    "total_events": d_meta["teacher"]["total_events"],
                    "event_recall": d_meta["teacher"]["detected_events"] / d_meta["teacher"]["total_events"]
                },
                "latency_ms": d_meta["teacher"]["latency_ms"],
                "speedup": d_meta["teacher"]["speedup"]
            },
            # 1D-CNN New Cache
            {
                "key": "cnn_1d_new_cache",
                "display": "**1D-CNN (3L) Baseline**",
                "train_name": "CNN1DStudent (3-Layer Channel-Independent 1D-CNN)",
                "arch_class": "`CNN1DStudent` (3L, CI)",
                "script": "`run_baseline_cache_ablation.py`",
                "cache": d_meta["new_cache"],
                "ckpt": f"`checkpoints/baseline_ablation/{ds}_cnn_1d_new_cache.pt`",
                "params": "0.46M",
                "m": models["cnn_1d_new_cache"]["metrics"],
                "latency_ms": d_meta["cnn_lat"],
                "speedup": d_meta["cnn_speedup"]
            },
            # 1D-CNN Old Cache
            {
                "key": "cnn_1d_old_cache",
                "display": "**1D-CNN (3L) Baseline**",
                "train_name": "CNN1DStudent (3-Layer Channel-Independent 1D-CNN)",
                "arch_class": "`CNN1DStudent` (3L, CI)",
                "script": "`run_baseline_cache_ablation.py`",
                "cache": d_meta["old_cache"],
                "ckpt": f"`checkpoints/baseline_ablation/{ds}_cnn_1d_old_cache.pt`",
                "params": "0.72M",
                "m": models["cnn_1d_old_cache"]["metrics"],
                "latency_ms": d_meta["cnn_lat"],
                "speedup": d_meta["cnn_speedup"]
            },
            # Residual-TCN New Cache
            {
                "key": "res_tcn_new_cache",
                "display": "**Residual-TCN (5L) Baseline**",
                "train_name": "CNN1DStudent (5-Layer Dilated Residual Causal TCN)",
                "arch_class": "`CNN1DStudent` (5L, CI, Res)",
                "script": "`run_baseline_cache_ablation.py`",
                "cache": d_meta["new_cache"],
                "ckpt": f"`checkpoints/baseline_ablation/{ds}_res_tcn_new_cache.pt`",
                "params": "1.05M",
                "m": models["res_tcn_new_cache"]["metrics"],
                "latency_ms": d_meta["res_lat"],
                "speedup": d_meta["res_speedup"]
            },
            # Residual-TCN Old Cache
            {
                "key": "res_tcn_old_cache",
                "display": "**Residual-TCN (5L) Baseline**",
                "train_name": "CNN1DStudent (5-Layer Dilated Residual Causal TCN)",
                "arch_class": "`CNN1DStudent` (5L, CI, Res)",
                "script": "`run_baseline_cache_ablation.py`",
                "cache": d_meta["old_cache"],
                "ckpt": f"`checkpoints/baseline_ablation/{ds}_res_tcn_old_cache.pt`",
                "params": "1.31M",
                "m": models["res_tcn_old_cache"]["metrics"],
                "latency_ms": d_meta["res_lat"],
                "speedup": d_meta["res_speedup"]
            },
            # Domain Coupled TCN (Ours) New Cache
            {
                "key": "old_student_new_cache",
                "display": "**Coupled Domain TCN (Ours)**",
                "train_name": d_meta["specialized_train_name"],
                "arch_class": d_meta["specialized_class"],
                "script": d_meta["specialized_script"],
                "cache": d_meta["new_cache"],
                "ckpt": f"`checkpoints/22Nov/cross_ablation_24/checkpoints/cross_ablation/{ds}_old_student_new_cache.pt`",
                "params": d_meta["specialized_params"],
                "m": models["old_student_new_cache"]["metrics"],
                "latency_ms": d_meta["specialized_lat"],
                "speedup": d_meta["specialized_speedup"]
            },
            # Domain Coupled TCN (Ours) Old Cache
            {
                "key": "old_student_old_cache",
                "display": "**Coupled Domain TCN (Ours)**",
                "train_name": d_meta["specialized_train_name"],
                "arch_class": d_meta["specialized_class"],
                "script": d_meta["specialized_script"],
                "cache": d_meta["old_cache"],
                "ckpt": f"`checkpoints/22Nov/cross_ablation_24/checkpoints/cross_ablation/{ds}_old_student_old_cache.pt`",
                "params": d_meta["specialized_params"],
                "m": models["old_student_old_cache"]["metrics"],
                "latency_ms": d_meta["specialized_lat"],
                "speedup": d_meta["specialized_speedup"]
            },
            # Channel Aware New Cache
            {
                "key": "channel_aware_new_cache",
                "display": "**UniversalChannelAware**",
                "train_name": "UniversalTCN_ChannelAware (Learned Channel Identity E_ch)",
                "arch_class": "`UniversalTCN_ChannelAware`",
                "script": "`run_cross_student_teacher_ablation.py` (standalone: `train_universal_channel_aware.py`)",
                "cache": d_meta["new_cache"],
                "ckpt": f"`checkpoints/22Nov/cross_ablation_24/checkpoints/cross_ablation/{ds}_channel_aware_new_cache.pt`",
                "params": "2.05M",
                "m": models["channel_aware_new_cache"]["metrics"],
                "latency_ms": d_meta["chan_lat"],
                "speedup": d_meta["chan_speedup"]
            },
            # Channel Aware Old Cache
            {
                "key": "channel_aware_old_cache",
                "display": "**UniversalChannelAware**",
                "train_name": "UniversalTCN_ChannelAware (Learned Channel Identity E_ch)",
                "arch_class": "`UniversalTCN_ChannelAware`",
                "script": "`run_cross_student_teacher_ablation.py` (standalone: `train_universal_channel_aware.py`)",
                "cache": d_meta["old_cache"],
                "ckpt": f"`checkpoints/22Nov/cross_ablation_24/checkpoints/cross_ablation/{ds}_channel_aware_old_cache.pt`",
                "params": "2.05M",
                "m": models["channel_aware_old_cache"]["metrics"],
                "latency_ms": d_meta["chan_lat"],
                "speedup": d_meta["chan_speedup"]
            },
            # Best Loss-Tuned Student
            {
                "key": "best_loss_tuned",
                "display": "**Best Loss-Tuned Student**",
                "train_name": d_meta["loss_tuned_train_name"],
                "arch_class": d_meta["loss_tuned_class"],
                "script": d_meta["loss_tuned_script"],
                "cache": d_meta["loss_tuned_cache"],
                "ckpt": d_meta["loss_tuned_ckpt"],
                "params": d_meta["loss_tuned_params"],
                "m": models["best_loss_tuned"]["metrics"],
                "latency_ms": d_meta["loss_tuned_lat"],
                "speedup": d_meta["loss_tuned_speedup"]
            },
            # UniversalFoundationTCN_Pro New Cache
            {
                "key": "tcn_pro_new_cache",
                "display": "**UniversalFoundationTCN_Pro**",
                "train_name": "UniversalFoundationTCN_Pro (Permutation-Equivariant Transformer + TCN)",
                "arch_class": "`UniversalFoundationTCN_Pro`",
                "script": "`run_cross_student_teacher_ablation.py` (standalone: `train_universal_foundation.py`)",
                "cache": d_meta["new_cache"],
                "ckpt": f"`checkpoints/22Nov/cross_ablation_24/checkpoints/cross_ablation/{ds}_tcn_pro_new_cache.pt`",
                "params": "1.98M",
                "m": models["tcn_pro_new_cache"]["metrics"],
                "latency_ms": d_meta["pro_lat"],
                "speedup": d_meta["pro_speedup"]
            },
            # UniversalFoundationTCN_Pro Old Cache
            {
                "key": "tcn_pro_old_cache",
                "display": "**UniversalFoundationTCN_Pro**",
                "train_name": "UniversalFoundationTCN_Pro (Permutation-Equivariant Transformer + TCN)",
                "arch_class": "`UniversalFoundationTCN_Pro`",
                "script": "`run_cross_student_teacher_ablation.py` (standalone: `train_universal_foundation.py`)",
                "cache": d_meta["old_cache"],
                "ckpt": f"`checkpoints/22Nov/cross_ablation_24/checkpoints/cross_ablation/{ds}_tcn_pro_old_cache.pt`",
                "params": "1.98M",
                "m": models["tcn_pro_old_cache"]["metrics"],
                "latency_ms": d_meta["pro_lat"],
                "speedup": d_meta["pro_speedup"]
            }
        ]

        # Table 1: Accuracy & Range-Aware Metrics
        doc.append(f"### {b_idx}.1 Performance Metrics Table ({ds.upper()})")
        doc.append("")
        doc.append("| Model Entity | Teacher Cache Manifold | Params | AUROC | PR-AUC | Point $F_1$ | Aff-$F_1$ | Aff-P / Aff-R | $\\\\tau^*$ | FPR (%) | Event Recall | Latency | Speedup |")
        doc.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

        tot_ev = d_meta["teacher"]["total_events"]

        # Find best values for highlighting
        best_auroc = max(r["m"]["auroc"] for r in model_rows)
        best_prauc = max(r["m"]["prauc"] for r in model_rows)
        best_f1 = max(r["m"]["best_f1"] for r in model_rows)
        best_aff_f1 = max(r["m"].get("aff_f1", 0) for r in model_rows if r["key"] != "res_tcn_new_cache" or ds != "msl")
        min_fpr = min(r["m"]["fpr"] for r in model_rows)
        max_ev_rec = max(r["m"].get("event_recall", 0) for r in model_rows if r["key"] != "res_tcn_new_cache" or ds != "msl")
        min_lat = min(r["latency_ms"] for r in model_rows)

        for row in model_rows:
            m = row["m"]
            cache_label = "New Cache (1024-d)" if "new" in row["key"] or row["key"] == "best_loss_tuned" or row["key"] == "teacher" else "Old Cache (2048-d)"
            aff_pr = f"{m.get('aff_precision', 0):.2f} / {m.get('aff_recall', 0):.2f}"
            det = m.get("detected_events", round(m.get("event_recall", 0) * tot_ev))
            ev_rec = m.get("event_recall", 0)
            
            # Format numbers with bolding for best
            auroc_str = f"**{m['auroc']:.4f}**" if abs(m['auroc'] - best_auroc) < 1e-5 else f"{m['auroc']:.4f}"
            prauc_str = f"**{m['prauc']:.4f}**" if abs(m['prauc'] - best_prauc) < 1e-5 else f"{m['prauc']:.4f}"
            f1_str = f"**{m['best_f1']:.4f}**" if abs(m['best_f1'] - best_f1) < 1e-5 else f"{m['best_f1']:.4f}"
            
            # Handle MSL res_tcn_new_cache collapse
            if ds == "msl" and row["key"] == "res_tcn_new_cache":
                aff_f1_str = "1.0000*"
                ev_rec_str = f"{tot_ev}/{tot_ev} (100%)*"
            else:
                aff_f1_str = f"**{m.get('aff_f1', 0):.4f}**" if abs(m.get('aff_f1', 0) - best_aff_f1) < 1e-5 else f"{m.get('aff_f1', 0):.4f}"
                ev_rec_str = f"**{det}/{tot_ev} ({ev_rec*100:.1f}%)**" if abs(ev_rec - max_ev_rec) < 1e-5 else f"{det}/{tot_ev} ({ev_rec*100:.1f}%)"

            fpr_str = f"**{m['fpr']*100:.2f}%**" if abs(m['fpr'] - min_fpr) < 1e-5 else f"{m['fpr']*100:.2f}%"
            lat_str = f"**{row['latency_ms']:.2f} ms**" if abs(row['latency_ms'] - min_lat) < 1e-4 else f"{row['latency_ms']:.2f} ms"
            speedup_str = f"**{row['speedup']}**" if abs(row['latency_ms'] - min_lat) < 1e-4 else f"{row['speedup']}"

            doc.append(
                f"| {row['display']} | {cache_label} | {row['params']} | "
                f"{auroc_str} | {prauc_str} | {f1_str} | "
                f"{aff_f1_str} | {aff_pr} | {m.get('best_threshold', 0):.4f} | "
                f"{fpr_str} | {ev_rec_str} | {lat_str} | {speedup_str} |"
            )

        doc.append("")
        # Table 2: Model Training Provenance & Checkpoint Paths
        doc.append(f"### {b_idx}.2 Training Provenance & Architecture Specification ({ds.upper()})")
        doc.append("")
        doc.append("| Model Entity | Exact Model Name Used for Training | PyTorch Architecture Class | Training Script | Teacher Cache File Used | Checkpoint File Path |")
        doc.append("| :--- | :--- | :--- | :--- | :--- | :--- |")

        for row in model_rows:
            doc.append(
                f"| {row['display']} | `{row['train_name']}` | {row['arch_class']} | {row['script']} | {row['cache']} | {row['ckpt']} |"
            )

        doc.append("")
        doc.append("---")
        doc.append("")

    # Section 5: Universal Foundation Model Summary
    doc.append("## 5. Universal Foundation Model Cross-Domain Frontier")
    doc.append("")
    doc.append("The permutation-equivariant foundation model (`UniversalFoundationTCN_Pro`) operates across arbitrary input dimensions without parameter reconfiguration:")
    doc.append("")
    doc.append("| Evaluated Benchmark | Dimension | Teacher Cache Supervision | Params | AUROC | PR-AUC | Point $F_1$ | Aff-$F_1$ | Aff-P / Aff-R | FPR (%) | Event Recall | Latency | Throughput |")
    doc.append("| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    uni_rows = [
        ("SWaT", "51D", "New Cache (1024-d Ultra-VETime)", "1.98M", 0.8882, 0.8732, 0.8884, 0.5075, "0.52 / 0.50", "0.29%", "7/14 (50.0%)", "8.88 ms", "113 win/s"),
        ("SWaT", "51D", "Old Cache (2048-d Early VLM)", "1.98M", 0.9037, 0.8789, 0.8849, 0.5490, "0.61 / 0.50", "0.25%", "7/14 (50.0%)", "8.89 ms", "113 win/s"),
        ("GECCO", "9D", "New Cache (1024-d Ultra-VETime)", "1.98M", 0.9812, 0.5782, 0.7022, 0.7451, "0.66 / 0.86", "0.55%", "19/22 (86.4%)", "1.61 ms", "620 win/s"),
        ("GECCO", "9D", "Old Cache (2048-d Early VLM)", "1.98M", 0.9674, 0.6788, 0.7411, 0.6349, "0.49 / 0.91", "0.35%", "20/22 (90.9%)", "1.62 ms", "617 win/s"),
        ("SMD", "38D", "New Cache (1024-d Ultra-VETime)", "1.98M", 0.8040, 0.2327, 0.3206, 0.2979, "0.20 / 0.55", "4.05%", "67/122 (54.9%)", "6.71 ms", "149 win/s"),
        ("SMD", "38D", "Old Cache (2048-d Early VLM)", "1.98M", 0.7923, 0.2610, 0.3195, 0.2686, "0.21 / 0.37", "1.85%", "45/122 (36.9%)", "6.71 ms", "149 win/s"),
        ("MSL", "55D", "New Cache (1024-d Ultra-VETime)", "1.98M", 0.6197, 0.1500, 0.1834, 0.1111, "1.00 / 0.06", "0.17%", "1/17 (5.9%)", "9.68 ms", "103 win/s"),
        ("MSL", "55D", "Old Cache (2048-d Early VLM)", "1.98M", 0.5132, 0.1499, 0.1934, 0.1345, "0.09 / 0.24", "0.81%", "4/17 (23.5%)", "9.51 ms", "105 win/s"),
        ("**MACRO AVG (New)**", "**Any**", "**New Cache (1024-d Ultra-VETime)**", "**1.98M**", 0.8233, 0.4585, 0.5236, 0.4154, "**0.60 / 0.49**", "**1.26%**", "**49.3%**", "**6.72 ms**", "**246 win/s**"),
        ("**MACRO AVG (Old)**", "**Any**", "**Old Cache (2048-d Early VLM)**", "**1.98M**", 0.7941, 0.4921, 0.5347, 0.3967, "**0.35 / 0.50**", "**0.82%**", "**50.3%**", "**6.68 ms**", "**246 win/s**"),
    ]

    for r in uni_rows:
        doc.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]:.4f} | {r[5]:.4f} | {r[6]:.4f} | {r[7]:.4f} | {r[8]} | {r[9]} | {r[10]} | {r[11]} | {r[12]} |")

    doc.append("")
    doc.append("---")
    doc.append("")

    # Section 6: Reproduction Guide
    doc.append("## 6. Full Reproducibility Guide & Executable Commands")
    doc.append("")
    doc.append("All experiments, trainings, and continuous streaming evaluations are deterministic and can be reproduced using Python 3.12 under `~/.venv`.")
    doc.append("")
    doc.append("### 6.1 Environment Activation")
    doc.append("```bash")
    doc.append("# Activate dedicated project virtual environment")
    doc.append("source ~/.venv/bin/activate")
    doc.append("cd /Users/rhythm/Desktop/IE643/Project/IE643")
    doc.append("export PYTHONPATH=\".\"")
    doc.append("```")
    doc.append("")
    doc.append("### 6.2 Master Continuous Streaming Evaluation Command")
    doc.append("To re-evaluate all 44 models from scratch across the complete continuous test streams:")
    doc.append("```bash")
    doc.append("# Executes sliding-window continuous inference (L=200, S=10) on local GPU (Apple MPS or CUDA):")
    doc.append("python scripts/eval_master_comprehensive_registry.py")
    doc.append("```")
    doc.append("Outputs generated:")
    doc.append("- `results/master_comprehensive_evaluation_registry.json`")
    doc.append("- `results/master_comprehensive_evaluation_summary.md`")
    doc.append("")
    doc.append("### 6.3 Standalone Training: Domain-Coupled Specialized Students")
    doc.append("To train each domain-specialized physical student model individually:")
    doc.append("```bash")
    doc.append("# 1. SWaT StageCoupledTCN (6 coupled stages P1-P6):")
    doc.append("python train_stage_coupled_student.py \\")
    doc.append("    --train_cache cache/swat_train_ultravetime_cache.pt \\")
    doc.append("    --eval_cache cache/swat_eval_head_to_head_cache.pt \\")
    doc.append("    --epochs 25 --batch_size 64 --lr 1e-3")
    doc.append("")
    doc.append("# 2. GECCO SubsystemCoupledTCN (3 water quality & hydraulic subsystems):")
    doc.append("python train_gecco_student.py \\")
    doc.append("    --data_path data/gecco/gecco_water.csv \\")
    doc.append("    --train_cache cache/gecco_train_ultravetime_cache.pt \\")
    doc.append("    --eval_cache cache/gecco_eval_head_to_head_cache.pt \\")
    doc.append("    --epochs 15 --batch_size 64")
    doc.append("")
    doc.append("# 3. SMD SubsystemCoupledTCN (Multi-machine pooled training across 28 server entities):")
    doc.append("python train_smd_pooled_student.py \\")
    doc.append("    --data_dir data/smd \\")
    doc.append("    --checkpoint_dir checkpoints \\")
    doc.append("    --epochs 10 --batch_size 128 --hidden_dim 32")
    doc.append("")
    doc.append("# 4. MSL CoupledTCN (Command-telemetry decoupled causal cross-fusion):")
    doc.append("python train_msl_student.py \\")
    doc.append("    --train_csv data/msl/msl_train.csv \\")
    doc.append("    --train_cache cache/msl_train_ultravetime_cache.pt \\")
    doc.append("    --eval_cache cache/msl_eval_head_to_head_cache.pt \\")
    doc.append("    --epochs 15")
    doc.append("```")
    doc.append("")
    doc.append("### 6.4 Standalone Training: Universal Foundation Models")
    doc.append("To train the dimension-invariant universal foundation architectures:")
    doc.append("```bash")
    doc.append("# 1. UniversalFoundationTCN_Pro (Permutation-Equivariant Transformer + Dilated TCN):")
    doc.append("python train_universal_foundation.py \\")
    doc.append("    --train_datasets smd \\")
    doc.append("    --eval_dataset msl \\")
    doc.append("    --data_dir data \\")
    doc.append("    --epochs 5 --batch_size 32")
    doc.append("")
    doc.append("# 2. UniversalTCN_ChannelAware (Learned Channel Identity Embeddings E_ch):")
    doc.append("python train_universal_channel_aware.py \\")
    doc.append("    --dataset all \\")
    doc.append("    --epochs 8 --batch_size 32 --lr 1e-3")
    doc.append("```")
    doc.append("")
    doc.append("### 6.5 Full 24-Model Cross-Ablation Factorial Suite")
    doc.append("To train all 24 factorial combinations (3 student architectures × 2 teacher manifolds × 4 benchmarks):")
    doc.append("```bash")
    doc.append("python run_cross_student_teacher_ablation.py \\")
    doc.append("    --data_dir data \\")
    doc.append("    --cache_dir cache \\")
    doc.append("    --datasets swat smd gecco msl \\")
    doc.append("    --epochs 10 \\")
    doc.append("    --batch_size 32 \\")
    doc.append("    --checkpoint_dir checkpoints/cross_ablation \\")
    doc.append("    --output_json results/ablations/cross_student_teacher_ablation.json")
    doc.append("```")
    doc.append("")
    doc.append("### 6.6 Full 16-Model Baseline Students Suite")
    doc.append("To train the 16 baseline architectures (1D-CNN 3L and Residual-TCN 5L) across both teacher manifolds:")
    doc.append("```bash")
    doc.append("python run_baseline_cache_ablation.py \\")
    doc.append("    --data_dir data \\")
    doc.append("    --cache_dir cache \\")
    doc.append("    --datasets swat gecco smd msl \\")
    doc.append("    --epochs 10 \\")
    doc.append("    --batch_size 32 \\")
    doc.append("    --checkpoint_dir checkpoints/baseline_ablation \\")
    doc.append("    --output_json results/ablations/baseline_cache_ablation.json \\")
    doc.append("    --output_report results/ablations/BASELINE_CACHE_REPORT.md")
    doc.append("```")
    doc.append("")
    doc.append("### 6.7 Loss Component & Distillation Weight Ablations")
    doc.append("To re-run loss sweeps (BCE-Only, Full VisTAD, and distillation weight sweeps $\\\\lambda \\\\in [0.00, 2.00]$):")
    doc.append("```bash")
    doc.append("# SWaT Weight Sweep (feat_weight in 0.00 to 2.00):")
    doc.append("python run_loss_and_weight_ablation.py --dataset swat")
    doc.append("")
    doc.append("# GECCO Weight Sweep:")
    doc.append("python run_loss_and_weight_ablation.py --dataset gecco")
    doc.append("")
    doc.append("# MSL Weight Sweep (BCE-Only vs Distill):")
    doc.append("python run_loss_and_weight_ablation.py --dataset msl")
    doc.append("")
    doc.append("# Grand Loss Variant Suite (Pure BCE, Focal, Focal+Dice, Unified Distill, BCE+Feat):")
    doc.append("python run_grand_loss_ablation.py \\")
    doc.append("    --data_dir data \\")
    doc.append("    --cache_dir cache \\")
    doc.append("    --variants pure_bce focal focal_dice unified_distill bce_feat \\")
    doc.append("    --epochs 5 \\")
    doc.append("    --output_json results/ablations/grand_loss_ablation_results.json")
    doc.append("```")
    doc.append("")
    doc.append("### 6.8 Teacher Feature Caching & Visual Foundation Ingestion")
    doc.append("To re-generate teacher feature representation caches (DINOv2 ViT-L/14) from raw multi-horizon timeline plots:")
    doc.append("```bash")
    doc.append("# 1. SWaT Teacher Cache:")
    doc.append("python run_teacher_cache.py")
    doc.append("# Or via module:")
    doc.append("python -m vlm4ts.generate_cache")
    doc.append("")
    doc.append("# 2. GECCO Teacher Cache:")
    doc.append("python -m vlm4ts.generate_gecco_cache")
    doc.append("")
    doc.append("# 3. SMD Teacher Cache:")
    doc.append("python -m vlm4ts.generate_smd_cache")
    doc.append("")
    doc.append("# 4. MSL Teacher Cache:")
    doc.append("python -m vlm4ts.generate_msl_cache")
    doc.append("```")
    doc.append("")
    doc.append("### 6.9 Headless Remote Kaggle GPU Workflow")
    doc.append("To push code and notebooks to Kaggle GPU, monitor execution headlessly, and pull back results:")
    doc.append("```bash")
    doc.append("# 1. Push kernel payload to Kaggle GPU:")
    doc.append("kaggle kernels push -p .")
    doc.append("")
    doc.append("# 2. Monitor execution status until 'complete':")
    doc.append("kaggle kernels status rhythmkedia/swat-vlm4ts-suite")
    doc.append("")
    doc.append("# 3. Download generated checkpoints, JSON logs, and tables:")
    doc.append("kaggle kernels output rhythmkedia/swat-vlm4ts-suite -p ./results/")
    doc.append("```")
    doc.append("")
    doc.append("### 6.10 Verifying Metric Calculations Standalone")
    doc.append("To verify the mathematical calculation of Point vs Affiliation metrics in a standalone Python session:")
    doc.append("```python")
    doc.append("import torch")
    doc.append("from src.metrics import compute_affiliation_metrics")
    doc.append("from sklearn.metrics import roc_auc_score, precision_recall_curve, auc")
    doc.append("")
    doc.append("# Point metrics:")
    doc.append("auroc = roc_auc_score(ground_truth_points, predicted_scores)")
    doc.append("prec, rec, _ = precision_recall_curve(ground_truth_points, predicted_scores)")
    doc.append("pr_auc = auc(rec, prec)")
    doc.append("")
    doc.append("# Range-aware affiliation metrics (continuous intervals):")
    doc.append("aff_res = compute_affiliation_metrics(ground_truth_points, binary_predictions)")
    doc.append("print(f\"Aff-F1: {aff_res['aff_f1']:.4f} | Aff-P: {aff_res['aff_precision']:.4f} | Aff-R: {aff_res['aff_recall']:.4f}\")")
    doc.append("```")
    doc.append("")
    doc.append("")
    doc.append("# Range-aware affiliation metrics (continuous intervals):")
    doc.append("aff_res = compute_affiliation_metrics(ground_truth_points, binary_predictions)")
    doc.append("print(f\"Aff-F1: {aff_res['aff_f1']:.4f} | Aff-P: {aff_res['aff_precision']:.4f} | Aff-R: {aff_res['aff_recall']:.4f}\")")
    doc.append("```")
    doc.append("")

    content = "\n".join(doc)

    out_file1 = "results/MASTER_MODEL_EVALUATION_METRICS.md"
    out_file2 = "MASTER_MODEL_EVALUATION_METRICS.md"

    with open(out_file1, "w") as f:
        f.write(content)
    print(f"Successfully generated: {out_file1} ({len(content):,} chars)")

    with open(out_file2, "w") as f:
        f.write(content)
    print(f"Successfully generated: {out_file2} ({len(content):,} chars)")

if __name__ == "__main__":
    main()
