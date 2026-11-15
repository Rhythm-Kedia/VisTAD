import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

for p in [CURRENT_DIR, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from serving import VisTADServingEngine

from src.dataset import load_dataset_arrays
from src.metrics import (
    find_events, 
    compute_affiliation_metrics, 
    compute_vus_metrics, 
    compute_point_metrics
)

# Page Configuration
st.set_page_config(
    page_title="VisTAD Telemetry Anomaly Detection",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, Minimal Styling
st.markdown("""
<style>
    .metric-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .metric-box .label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .metric-box .val {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-box .desc {
        font-size: 0.75rem;
        color: #64748b;
    }
    .badge-zeroshot {
        background-color: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1e40af;
        padding: 10px 14px;
        border-radius: 6px;
        font-size: 0.88rem;
        margin-bottom: 12px;
    }
    .badge-pure {
        background-color: #fffbeb;
        border: 1px solid #fde68a;
        color: #92400e;
        padding: 10px 14px;
        border-radius: 6px;
        font-size: 0.88rem;
        margin-bottom: 12px;
    }
    .model-spec-card {
        background-color: #f1f5f9;
        border-radius: 6px;
        padding: 10px 12px;
        margin-top: 8px;
        margin-bottom: 12px;
        font-size: 0.80rem;
        border-left: 3px solid #0284c7;
        line-height: 1.45;
    }
</style>
""", unsafe_allow_html=True)


def parse_ground_truth_column(col_series):
    """Robustly maps labels whether stored as integers, floats, booleans, or strings."""
    if pd.api.types.is_bool_dtype(col_series):
        return col_series.astype(int).values
        
    num_s = pd.to_numeric(col_series, errors="coerce")
    if num_s.notna().mean() >= 0.80:
        return (num_s.fillna(0).values >= 0.5).astype(int)
        
    str_s = col_series.astype(str).str.strip().str.lower()
    is_anomaly = str_s.apply(lambda x: 1 if any(t in x for t in ["attack", "anom", "fault", "error", "alarm", "true"]) else 0)
    return is_anomaly.values.astype(int)


def fmt_metric(val):
    """Formats float metric safely, replacing NaN or None with '-'."""
    if val is None or np.isnan(val):
        return "-"
    return f"{float(val):.3f}"


def get_feature_names_for_dataset(ds_name, n_channels):
    """Retrieves intuitive human-readable sensor tag names for known benchmark datasets."""
    ds_lower = ds_name.lower()
    data_dir = os.path.join(ROOT_DIR, "data")
    
    if "swat" in ds_lower:
        csv_p = os.path.join(data_dir, "swat/SWaT_Dataset_Attack_v0.csv")
        if not os.path.exists(csv_p):
            csv_p = os.path.join(data_dir, "samples/test_swat_water_scada.csv")
        if os.path.exists(csv_p):
            try:
                cols = list(pd.read_csv(csv_p, nrows=1).columns)
                feat_cols = [c for c in cols if c.lower() not in ["attack", "label", "timestamp", "time", "anomaly_label", "normal/attack"]]
                if len(feat_cols) == n_channels:
                    return feat_cols
            except Exception:
                pass
    elif "gecco" in ds_lower:
        csv_p = os.path.join(data_dir, "gecco/gecco_water.csv")
        if os.path.exists(csv_p):
            try:
                cols = list(pd.read_csv(csv_p, nrows=1).columns)
                feat_cols = [c for c in cols if c.lower() not in ["attack", "label", "timestamp", "time"]]
                if len(feat_cols) == n_channels:
                    return feat_cols
            except Exception:
                pass
    elif "msl" in ds_lower:
        csv_p = os.path.join(data_dir, "msl/msl_test.csv")
        if os.path.exists(csv_p):
            try:
                cols = list(pd.read_csv(csv_p, nrows=1).columns)
                feat_cols = [c for c in cols if c.lower() not in ["attack", "label", "timestamp", "time"]]
                if len(feat_cols) == n_channels:
                    return feat_cols
            except Exception:
                pass

    return [f"Sensor {i}" for i in range(n_channels)]


@st.cache_resource
def get_engine():
    return VisTADServingEngine()


def main():
    engine = get_engine()

    # Header
    st.title("VisTAD: Universal Cyber-Physical Telemetry Anomaly Detection")
    st.caption("Any-Dimension Foundation Models & Distilled Specialized Students with Continuous Probability Calibration.")

    # Sidebar: Input Configuration
    st.sidebar.header("1. Input Telemetry")
    data_source = st.sidebar.radio(
        "Source:",
        ["Benchmark Dataset Presets", "Upload Custom CSV"]
    )

    raw_data = None
    ground_truth = None
    clean_base_m = None
    clean_base_s = None
    time_index = None
    dataset_name = ""
    feature_names = []
    dataset_key = ""
    active_domain = "SWaT"

    if data_source == "Upload Custom CSV":
        samples_dir = os.path.join(ROOT_DIR, "data/samples")
        sample_files = [f for f in sorted(os.listdir(samples_dir)) if f.startswith("test_") and f.endswith(".csv")] if os.path.exists(samples_dir) else []
        sample_choice = st.sidebar.selectbox("Quick-Load Test Sample:", ["(Upload Custom File)"] + sample_files)
        uploaded_file = st.sidebar.file_uploader("Upload CSV File:", type=["csv"])

        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.sidebar.error(f"Failed to read CSV: {e}")
                return
            dataset_name = f"Uploaded CSV ({uploaded_file.name})"
            dataset_key = f"upload_{uploaded_file.name}"
        elif sample_choice != "(Upload Custom File)":
            sample_path = os.path.join(samples_dir, sample_choice)
            uploaded_df = pd.read_csv(sample_path)
            dataset_name = f"Sample CSV ({sample_choice})"
            dataset_key = f"sample_{sample_choice}"
        else:
            st.info("Upload a CSV file or select a test sample above to begin testing.")
            return

        all_cols = list(uploaded_df.columns)
        candidate_gt = [c for c in all_cols if any(k in c.lower() for k in ["label", "anomaly", "attack", "ground_truth", "target"])]
        gt_options = ["(None / Unlabeled)"] + all_cols
        default_gt_idx = gt_options.index(candidate_gt[0]) if candidate_gt else 0
        selected_gt_col = st.sidebar.selectbox("Ground Truth Label Column:", gt_options, index=default_gt_idx)

        # Detect timestamp
        ts_cols = [c for c in all_cols if any(k in c.lower() for k in ["time", "date", "timestamp"])]
        time_col = ts_cols[0] if ts_cols else None

        # Clean numeric sensor features
        feature_cols = [c for c in all_cols if c != selected_gt_col and c != time_col]
        converted_df = pd.DataFrame()
        for c in feature_cols:
            num_s = pd.to_numeric(uploaded_df[c], errors="coerce")
            if num_s.notna().sum() > len(uploaded_df) * 0.50:
                converted_df[c] = num_s

        converted_df = converted_df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

        if converted_df.shape[1] == 0:
            st.error("No numeric sensor columns found in the uploaded CSV.")
            return

        slice_len = st.sidebar.slider("Evaluation Window Length (steps):", 500, min(20000, len(converted_df)), min(3000, len(converted_df)), 500)
        raw_data = converted_df.values[:slice_len]
        feature_names = list(converted_df.columns)

        if selected_gt_col != "(None / Unlabeled)" and selected_gt_col in uploaded_df.columns:
            gt_series = parse_ground_truth_column(uploaded_df[selected_gt_col].iloc[:slice_len])
            ground_truth = gt_series

        if time_col is not None:
            time_index = uploaded_df[time_col].values[:slice_len]
        else:
            time_index = np.arange(len(raw_data))

        # Detect target domain
        name_lower = dataset_name.lower()
        if "swat" in name_lower:
            active_domain = "SWaT"
        elif "gecco" in name_lower:
            active_domain = "GECCO"
        elif "smd" in name_lower or "server" in name_lower:
            active_domain = "SMD"
        elif "msl" in name_lower or "space" in name_lower:
            active_domain = "MSL"
        else:
            active_domain = "SWaT"

        # Collapsible Data Preview
        with st.expander("Preview Uploaded Data Table", expanded=False):
            st.dataframe(uploaded_df.head(5), use_container_width=True)

    else:
        # Benchmark presets with offsets that contain genuine anomaly episodes
        preset_configs = {
            "SWaT Water Treatment SCADA (Attack Intervals) [51 Sensors]": {
                "ds": "swat", "entity": "swat_attack", "start": 1500, "len": 3000,
                "domain": "SWaT", "sample_file": "test_swat_water_scada.csv"
            },
            "GECCO Drinking Water (Chemical Drift) [9 Sensors]": {
                "ds": "gecco", "entity": "gecco_water", "start": 60000, "len": 2500,
                "domain": "GECCO", "sample_file": "test_gecco_water.csv"
            },
            "SMD Cloud Computing Server (Crash Incident) [38 Metrics]": {
                "ds": "smd", "entity": "machine-1-1", "start": 15000, "len": 3000,
                "domain": "SMD", "sample_file": "test_server_smd.csv"
            },
            "MSL Mars Rover (Telemetry Anomaly) [55 Channels]": {
                "ds": "msl", "entity": "msl_test", "start": 27000, "len": 3000,
                "domain": "MSL", "sample_file": "test_spacecraft_msl.csv"
            }
        }
        preset_choice = st.sidebar.selectbox("Choose Benchmark Dataset:", list(preset_configs.keys()))
        cfg = preset_configs[preset_choice]
        active_domain = cfg["domain"]
        dataset_name = preset_choice
        dataset_key = f"preset_{preset_choice}"

        slice_len = st.sidebar.slider("Evaluation Window Length (steps):", 1000, 5000, cfg["len"], 500)

        data_dir = os.path.join(ROOT_DIR, "data")
        raw_data = None
        ground_truth = None
        clean_base_m = None
        clean_base_s = None
        try:
            full_raw, full_labels = load_dataset_arrays(cfg["ds"], cfg["entity"], data_dir)
            st_idx = cfg["start"]
            ed_idx = min(st_idx + slice_len, len(full_raw))
            raw_data = full_raw[st_idx:ed_idx]
            ground_truth = full_labels[st_idx:ed_idx]
            time_index = np.arange(len(raw_data))
            feature_names = get_feature_names_for_dataset(cfg["ds"], raw_data.shape[1])
            
            # Load reference baseline from training split if available
            train_entity = {
                "swat": "swat_normal",
                "msl": "msl_train",
            }.get(cfg["ds"])
            if train_entity:
                try:
                    train_raw, _ = load_dataset_arrays(cfg["ds"], train_entity, data_dir)
                    clean_base_m = np.mean(train_raw, axis=0)
                    clean_base_s = np.std(train_raw, axis=0)
                    clean_base_s = np.where(clean_base_s == 0, 1.0, clean_base_s)
                    clean_base_s = np.maximum(clean_base_s, 0.05)
                except Exception:
                    clean_base_m = None
                    clean_base_s = None

            if clean_base_m is None:
                # Use standard first-50% training reference partition for single-file benchmarks (GECCO)
                ref_len = int(len(full_raw) * 0.50) if cfg["ds"] == "gecco" else min(3000, len(full_raw))
                clean_base_m = np.mean(full_raw[:ref_len], axis=0)
                clean_base_s = np.std(full_raw[:ref_len], axis=0)
                clean_base_s = np.where(clean_base_s == 0, 1.0, clean_base_s)
                clean_base_s = np.maximum(clean_base_s, 0.05)
        except Exception as e:
            st.error(f"Error loading benchmark dataset '{preset_choice}': {e}")
            return

    # Sidebar: Model Selection (Clean, Structured Hierarchical Selector)
    st.sidebar.header("2. Model Selection")

    # 1. Paradigm Selector
    paradigm_options = [
        "🌐 Zero-Shot Foundation (Cross-Domain)",
        "🔬 In-Domain Specialized Student",
        "🏛️ Grand Unified Foundation (All 4 Domains)"
    ]
    selected_paradigm_display = st.sidebar.selectbox(
        "Evaluation Paradigm:",
        paradigm_options,
        index=0,
        help="Zero-Shot Foundation evaluates models that have never seen this target plant during training. In-Domain students are specialized baselines."
    )

    selected_model_key = None
    if "Zero-Shot" in selected_paradigm_display:
        # Match Zero-Shot model for active domain
        zero_shot_map = {
            "SWaT": "swat_zero_shot",
            "GECCO": "gecco_zero_shot",
            "SMD": "smd_zero_shot",
            "MSL": "msl_zero_shot"
        }
        selected_model_key = zero_shot_map.get(active_domain, "swat_zero_shot")

    elif "Grand Unified" in selected_paradigm_display:
        selected_model_key = "grand_unified"

    else:
        # In-Domain Specialized Student: Choose Distillation Regimen and Architecture
        distill_choice = st.sidebar.radio(
            "Distillation Regimen:",
            ["✨ Distilled (VLM Teacher Features)", "🚫 Pure Supervised (Non-Distilled)"],
            index=0,
            help="Distilled students align their latent space with DINOv2 vision representations. Pure supervised students learn strictly from discrete 0/1 binary labels."
        )
        is_distilled = ("Distilled" in distill_choice and "Non-Distilled" not in distill_choice)

        if is_distilled:
            arch_choices = {
                f"Subsystem-Coupled TCN (Recommended)": f"{active_domain.lower()}_coupled_distilled",
                f"Standard Residual-TCN 5L (Top-6 Filtered)": f"{active_domain.lower()}_tcn_distilled"
            }
            chosen_arch_label = st.sidebar.selectbox(
                "Architecture Variant:",
                list(arch_choices.keys()),
                index=0
            )
            selected_model_key = arch_choices[chosen_arch_label]
        else:
            selected_model_key = f"{active_domain.lower()}_tcn_pure"

    catalog = getattr(engine, "MODEL_CATALOG", getattr(VisTADServingEngine, "MODEL_CATALOG", {}))
    model_cfg = catalog.get(selected_model_key, {})

    # Render Clean Model Specification Card
    st.sidebar.markdown(f"""
    <div class="model-spec-card">
        <b>Checkpoint:</b> <code>{model_cfg.get('file', '-')}</code><br>
        <b>Architecture:</b> {model_cfg.get('arch_name', '-')}<br>
        <b>Training Scope:</b> {model_cfg.get('trained_on', '-')}<br>
        <b>Distillation:</b> {model_cfg.get('distill_type', '-')}<br>
        <b>Calibrated Cutoff:</b> τ* = {model_cfg.get('default_threshold', 0.50):.4f}
    </div>
    """, unsafe_allow_html=True)

    # Sidebar: Operating Decision Threshold (Configurable by user, defaults to model-calibrated tau*)
    st.sidebar.header("3. Decision Threshold")
    default_tau = float(model_cfg.get("default_threshold", 0.50))

    if st.session_state.get("last_model_for_threshold") != selected_model_key:
        st.session_state["last_model_for_threshold"] = selected_model_key
        st.session_state["tau_slider_value"] = default_tau

    operating_tau = st.sidebar.slider(
        "Operating Threshold (tau):",
        min_value=0.01,
        max_value=1.00,
        value=st.session_state.get("tau_slider_value", default_tau),
        step=0.005,
        format="%.4f",
        key="tau_slider_value",
        help=f"Best calibrated default for {model_cfg.get('name')}: tau* = {default_tau:.4f}"
    )

    with st.sidebar.expander("Alarm Post-Processing (Hysteresis)", expanded=False):
        apply_smoothing = st.checkbox("Enable Hysteresis Smoothing", value=True, help="Merges fragmented threshold flickers into cohesive alarms")
        min_dur = st.slider("Minimum Duration (steps):", 1, 10, 3)
        merge_gap = st.slider("Merge Gap (steps):", 0, 30, 10)

    # Load Model Checkpoint
    try:
        engine.load_model(selected_model_key)
    except Exception as e:
        st.error(f"Failed to load checkpoint: {e}")
        return

    # Execute Inference
    results = engine.process_telemetry(
        raw_data=raw_data,
        window_size=200,
        stride=10,
        threshold_mode="fixed",
        fixed_threshold=operating_tau,
        apply_smoothing=apply_smoothing,
        min_duration=min_dur,
        merge_gap=merge_gap,
        base_mean=clean_base_m,
        base_std=clean_base_s
    )

    scores = results["continuous_scores"]
    channel_attrs = results.get("channel_attributions")
    event_details = results.get("event_details", [])
    operating_tau = results["threshold"]
    preds = results["binary_predictions"]
    detected_events = results["detected_events"]
    selected_ch = results["selected_channels"]

    # Paradigm & Regimen Banners
    is_zero_shot = (model_cfg.get("paradigm") == "Zero-Shot Foundation")
    is_pure_supervised = (model_cfg.get("distilled") is False)

    if is_zero_shot:
        st.markdown(f"""
        <div class="badge-zeroshot">
            🎯 <b>Zero-Shot Transfer Evaluation Active:</b> Evaluated on unseen <b>{active_domain}</b> telemetry. This model was trained <i>strictly on external datasets</i> ({model_cfg.get('trained_on')}) and has <b>never seen</b> any training data or anomaly labels from this plant. The continuous probability curve p_system(t) in [0, 1] demonstrates direct cross-domain zero-shot foundation transfer.
        </div>
        """, unsafe_allow_html=True)
    elif is_pure_supervised:
        st.markdown(f"""
        <div class="badge-pure">
            ⚠️ <b>Pure Supervised Ablation Active:</b> This model was trained strictly on discrete 0/1 binary labels without DINOv2 teacher distillation. Without geometric feature regularization, notice the elevated background noise floor and higher sensitivity to spurious sensor fluctuations.
        </div>
        """, unsafe_allow_html=True)

    # Pre-build incident records
    records = []
    if event_details:
        for idx_e, ev in enumerate(event_details):
            top_sensor_str = "-"
            if ev.get("top_sensors"):
                top_ch_idx, top_score = ev["top_sensors"][0]
                tag_name = feature_names[top_ch_idx] if top_ch_idx < len(feature_names) else f"Ch {top_ch_idx}"
                top_sensor_str = f"{tag_name} (p={top_score:.2f})"

            records.append({
                "Incident": f"Alarm #{idx_e + 1}",
                "Start Step": ev["start"],
                "End Step": ev["end"],
                "Duration": f"{ev['duration']} steps",
                "Peak Prob": f"{ev['peak_prob']:.3f}",
                "Top Root-Cause Sensor": top_sensor_str,
                "Severity": "CRITICAL" if ev["peak_prob"] > 0.85 else ("HIGH" if ev["peak_prob"] > 0.60 else "WARNING")
            })

    # Retrieve interactive table selections
    table_state = st.session_state.get("incident_table") or {}
    sel_rows = table_state.get("selection", {}).get("rows", []) if hasattr(table_state, "get") else getattr(table_state, "selection", {}).rows
    selected_indices = [r for r in sel_rows if isinstance(r, int) and 0 <= r < len(records)]

    # Quantitative Evaluation (if ground truth is available)
    eval_metrics = None
    has_ground_truth = (ground_truth is not None and np.any(ground_truth >= 0.5))
    if has_ground_truth:
        try:
            pt = compute_point_metrics(ground_truth, scores, threshold=operating_tau)
            aff = compute_affiliation_metrics(ground_truth, preds)
            vus = compute_vus_metrics(ground_truth, scores, max_buffer=50)
            eval_metrics = {
                "auroc": pt.get("auroc", np.nan),
                "prauc": pt.get("prauc", np.nan),
                "aff_f1": aff.get("aff_f1", np.nan),
                "aff_p": aff.get("aff_precision", np.nan),
                "aff_r": aff.get("aff_recall", np.nan)
            }
        except Exception:
            eval_metrics = None

    # Top KPI Metrics Row
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="label">Stream Length</div>
            <div class="val">{len(raw_data):,} steps</div>
            <div class="desc">{raw_data.shape[1]} sensor channels</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        model_type_label = "Zero-Shot Foundation" if is_zero_shot else ("Grand Unified" if "Grand" in model_cfg.get("name", "") else "Specialized Student")
        st.markdown(f"""
        <div class="metric-box">
            <div class="label">Active Model</div>
            <div class="val">{model_type_label}</div>
            <div class="desc">{model_cfg.get('arch_name', '')} ({len(selected_ch)} ch)</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="label">Inference Latency</div>
            <div class="val">{results['ms_per_window']:.3f} ms / win</div>
            <div class="desc">Total {results['inference_time_ms']:.1f} ms</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="label">Decision Threshold</div>
            <div class="val">{operating_tau:.4f}</div>
            <div class="desc">Operating Cutoff (tau)</div>
        </div>
        """, unsafe_allow_html=True)
    with k5:
        st.markdown(f"""
        <div class="metric-box">
            <div class="label">Detected Incidents</div>
            <div class="val">{len(detected_events)} alarms</div>
            <div class="desc">{'Nominal stream' if len(detected_events) == 0 else f'{sum(e[1]-e[0] for e in detected_events)} anomaly steps'}</div>
        </div>
        """, unsafe_allow_html=True)

    # Ground-Truth Evaluation Metrics Banner (if ground truth is present)
    if eval_metrics is not None:
        st.write("")
        st.markdown("**Ground-Truth Benchmark Evaluation Scores**")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("AUROC", fmt_metric(eval_metrics["auroc"]))
        m2.metric("PR-AUC", fmt_metric(eval_metrics["prauc"]))
        m3.metric("Affiliation F1", fmt_metric(eval_metrics["aff_f1"]))
        m4.metric("Affiliation Precision", fmt_metric(eval_metrics["aff_p"]))
        m5.metric("Affiliation Recall", fmt_metric(eval_metrics["aff_r"]))

    # Timeline Header and Optional Clear Selection
    st.write("")
    if selected_indices:
        col_chart_hdr, col_clear = st.columns([5, 1])
        with col_chart_hdr:
            st.subheader("Synchronized Interactive Timeline")
        with col_clear:
            if st.button("Clear Selection", use_container_width=True):
                if "incident_table" in st.session_state:
                    del st.session_state["incident_table"]
                st.rerun()
    else:
        st.subheader("Synchronized Interactive Timeline")

    # Synchronized Interactive Timeline
    num_rows = 3 if has_ground_truth else 2
    row_heights = [0.40, 0.42, 0.18] if has_ground_truth else [0.50, 0.50]
    sub_titles = [
        f"Normalized Sensor Telemetry ({dataset_name})", 
        f"VisTAD Continuous Anomaly Probability p_system(t) in [0, 1] vs Threshold (tau = {operating_tau:.3f})"
    ]
    if has_ground_truth:
        sub_titles.append("Ground Truth Anomaly State [0 = Nominal, 1 = Active Cyber-Physical Attack / Anomaly]")

    fig = make_subplots(
        rows=num_rows, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.06,
        row_heights=row_heights,
        subplot_titles=sub_titles
    )

    t_axis = time_index

    # 1. Shading for Detected Predicted Alarm Intervals (Score Plot)
    for s_e, e_e in detected_events:
        fig.add_vrect(
            x0=t_axis[s_e], x1=t_axis[min(e_e, len(t_axis)-1)], row=2, col=1,
            fillcolor="rgba(249, 115, 22, 0.20)", line_width=1, line_color="#f97316"
        )

    # 2. Sensor Traces (Top-3 Most Active Channels)
    ch_colors = ["#1e293b", "#0284c7", "#0d9488", "#7c3aed"]
    for idx_ch, ch in enumerate(selected_ch[:3]):
        s_mean = np.mean(raw_data[:, ch])
        s_std = np.std(raw_data[:, ch]) + 0.05
        norm_ch = (raw_data[:, ch] - s_mean) / s_std
        name_label = feature_names[ch] if ch < len(feature_names) else f"Ch {ch}"
        fig.add_trace(
            go.Scatter(
                x=t_axis, y=norm_ch, mode="lines", 
                name=f"{name_label} (Norm)", 
                line=dict(color=ch_colors[idx_ch % len(ch_colors)], width=1.3)
            ),
            row=1, col=1
        )

    # 3. Continuous Anomaly Likelihood Score s(t)
    fig.add_trace(
        go.Scatter(
            x=t_axis, y=scores, mode="lines", 
            name="Anomaly Probability p(t)", 
            line=dict(color="#dc2626", width=1.8),
            fill="tozeroy",
            fillcolor="rgba(220, 38, 38, 0.08)"
        ),
        row=2, col=1
    )

    # 4. Operating Decision Threshold Line
    fig.add_trace(
        go.Scatter(
            x=[t_axis[0], t_axis[-1]], y=[operating_tau, operating_tau], mode="lines",
            name=f"Threshold tau ({operating_tau:.3f})",
            line=dict(color="#2563eb", width=1.6, dash="dash")
        ),
        row=2, col=1
    )

    # 5. Explicit Ground Truth Step-Line and Selected Incident Overlay (Row 3, if available)
    if has_ground_truth:
        fig.add_trace(
            go.Scatter(
                x=t_axis, y=ground_truth, mode="lines",
                name="Ground Truth Attack y(t)",
                line=dict(shape="hv", color="#b91c1c", width=2),
                fill="tozeroy",
                fillcolor="rgba(185, 28, 28, 0.20)"
            ),
            row=3, col=1
        )
        fig.update_yaxes(title_text="Label (0/1)", row=3, col=1, range=[-0.1, 1.2], tickvals=[0, 1])

        # Mark any selected incident intervals in the Ground Truth graph
        if selected_indices:
            sel_mask = np.zeros(len(t_axis))
            for idx in selected_indices:
                s_rec = records[idx]
                s_st = s_rec["Start Step"]
                s_ed = s_rec["End Step"]
                sel_mask[s_st:min(s_ed, len(t_axis))] = 1.0

            fig.add_trace(
                go.Scatter(
                    x=t_axis, y=sel_mask, mode="lines",
                    name="Selected Incident",
                    line=dict(shape="hv", color="#1d4ed8", width=2.5),
                    fill="tozeroy",
                    fillcolor="rgba(37, 99, 235, 0.35)"
                ),
                row=3, col=1
            )

            for idx in selected_indices:
                s_rec = records[idx]
                s_st = s_rec["Start Step"]
                s_ed = s_rec["End Step"]
                fig.add_vrect(
                    x0=t_axis[s_st],
                    x1=t_axis[min(s_ed, len(t_axis) - 1)],
                    row=3,
                    col=1,
                    fillcolor="rgba(37, 99, 235, 0.30)",
                    line_width=2,
                    line_color="#1d4ed8",
                    layer="above"
                )
    elif selected_indices:
        for idx in selected_indices:
            s_rec = records[idx]
            s_st = s_rec["Start Step"]
            s_ed = s_rec["End Step"]
            fig.add_vrect(
                x0=t_axis[s_st],
                x1=t_axis[min(s_ed, len(t_axis) - 1)],
                row=2,
                col=1,
                fillcolor="rgba(37, 99, 235, 0.30)",
                line_width=2,
                line_color="#1d4ed8",
                layer="above"
            )

    fig.update_layout(
        height=680 if has_ground_truth else 550,
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=40, r=40, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="Sensor (std)", row=1, col=1)
    fig.update_yaxes(title_text="Prob p(t)", row=2, col=1, range=[-0.02, 1.05], tickvals=[0.0, 0.25, 0.5, 0.75, 1.0])
    fig.update_xaxes(title_text="Timestep", row=num_rows, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # Root Cause Sensor Attribution Section
    st.subheader("Root-Cause Sensor Attribution")
    if channel_attrs is not None:
        if selected_indices:
            if len(selected_indices) == 1:
                sel_idx = selected_indices[0]
                st_step = records[sel_idx]["Start Step"]
                ed_step = records[sel_idx]["End Step"]
                attr_title = f"Top Contributing Sensors for {records[sel_idx]['Incident']} (Steps {st_step} to {ed_step})"
                sensor_scores = np.mean(channel_attrs[:, st_step:ed_step], axis=1)
            else:
                inc_names = ", ".join(records[i]["Incident"] for i in selected_indices)
                attr_title = f"Top Contributing Sensors Across {len(selected_indices)} Selected Incidents ({inc_names})"
                sel_mask = np.zeros(channel_attrs.shape[1], dtype=bool)
                for idx in selected_indices:
                    st_step = records[idx]["Start Step"]
                    ed_step = records[idx]["End Step"]
                    sel_mask[st_step:min(ed_step, len(sel_mask))] = True
                sensor_scores = np.mean(channel_attrs[:, sel_mask], axis=1) if np.any(sel_mask) else np.mean(channel_attrs, axis=1)
        elif detected_events:
            attr_title = "Top Contributing Sensors Across All Detected Incidents"
            mask = np.zeros(len(scores), dtype=bool)
            for s_e, e_e in detected_events:
                mask[s_e:e_e] = True
            sensor_scores = np.mean(channel_attrs[:, mask], axis=1) if np.any(mask) else np.mean(channel_attrs, axis=1)
        else:
            attr_title = "Top Suspicious Sensors (Overall Stream Baseline)"
            sensor_scores = np.mean(channel_attrs, axis=1)

        top_k = min(8, len(sensor_scores))
        top_sensor_indices = np.argsort(sensor_scores)[::-1][:top_k]
        top_labels = [feature_names[i] if i < len(feature_names) else f"Ch {i}" for i in top_sensor_indices]
        top_values = [float(sensor_scores[i]) for i in top_sensor_indices]

        col_bar, col_detail = st.columns([3, 2])
        with col_bar:
            fig_bar = go.Figure(go.Bar(
                x=top_values[::-1],
                y=top_labels[::-1],
                orientation='h',
                marker=dict(color="#0284c7")
            ))
            fig_bar.update_layout(
                title=attr_title,
                xaxis_title="Attribution Anomaly Probability p_channel in [0, 1]",
                yaxis_title="Sensor Tag",
                xaxis=dict(range=[0, 1.05]),
                height=280,
                margin=dict(l=20, r=20, t=35, b=30)
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        with col_detail:
            st.markdown(f"**Root-Cause Diagnosis**")
            st.caption(f"Top sensor **{top_labels[0]}** reached peak probability **{top_values[0]:.3f}**.")
            st.markdown(f"- **Primary Suspect**: `{top_labels[0]}` (p = {top_values[0]:.3f})")
            if len(top_labels) > 1:
                st.markdown(f"- **Secondary Suspect**: `{top_labels[1]}` (p = {top_values[1]:.3f})")
            if len(top_labels) > 2:
                st.markdown(f"- **Tertiary Suspect**: `{top_labels[2]}` (p = {top_values[2]:.3f})")
            st.caption("Cross-channel transformer computes mutual physical correlation to isolate root cause sensors.")
    else:
        st.info("Root-cause per-channel attribution is active for Universal-TCN-Pro foundation models.")

    # Detected Incidents Ledger
    st.write("")
    st.subheader("Detected Incidents Ledger")
    st.caption("Select one or more rows below (click with Shift / Cmd / Ctrl to multi-select) to highlight intervals and compare root-cause sensors.")
    if detected_events:
        df_records = pd.DataFrame(records)
        st.dataframe(
            df_records,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="multi-row",
            key="incident_table"
        )
    else:
        st.success("Nominal State: No anomalous intervals detected exceeding the decision threshold.")


if __name__ == "__main__":
    main()
