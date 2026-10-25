import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure workspace root is in path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from streamlit_app.serving import VisTADServingEngine
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


@st.cache_resource
def get_engine():
    return VisTADServingEngine()


def main():
    engine = get_engine()

    # Header
    st.title("VisTAD: Time-Series Anomaly Detection")
    st.caption("Edge-deployable 1D-CNN distilled from VLM teacher for multivariate telemetry anomaly detection.")

    # Sidebar: Input Configuration
    st.sidebar.header("1. Input Telemetry")
    data_source = st.sidebar.radio(
        "Source:",
        ["Benchmark Dataset Presets", "Upload Custom CSV"]
    )

    raw_data = None
    ground_truth = None
    time_index = None
    dataset_name = ""
    feature_names = []
    default_model = "SMD Cloud Computing Expert"
    dataset_key = ""

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

        # Context-dependent default model
        domain_models = {
            "smd": "SMD Cloud Computing Expert",
            "swat": "SWaT Water Treatment SCADA Expert",
            "msl": "MSL Mars Rover Telemetry Expert",
            "gecco": "GECCO Water Quality Expert"
        }
        default_model = next((m for k, m in domain_models.items() if k in dataset_name.lower()), "Universal Foundation (Zero-Shot Cross-Domain)")

        # Collapsible Data Preview
        with st.expander("Preview Uploaded Data Table", expanded=False):
            st.dataframe(uploaded_df.head(5), use_container_width=True)

    else:
        # Benchmark presets with offsets that contain genuine anomaly episodes
        preset_configs = {
            "SMD Cloud Computing Server (Crash Incident)": {
                "ds": "smd", "entity": "machine-1-1", "start": 15000, "len": 3000,
                "model": "SMD Cloud Computing Expert"
            },
            "SWaT Water Treatment SCADA (Attack Intervals)": {
                "ds": "swat", "entity": "swat_attack", "start": 1000, "len": 3000,
                "model": "SWaT Water Treatment SCADA Expert"
            },
            "MSL Mars Rover (Telemetry Anomaly)": {
                "ds": "msl", "entity": "msl_test", "start": 45000, "len": 3000,
                "model": "MSL Mars Rover Telemetry Expert"
            },
            "GECCO Drinking Water (Chemical Drift)": {
                "ds": "gecco", "entity": "gecco_water", "start": 15500, "len": 2500,
                "model": "GECCO Water Quality Expert"
            }
        }
        preset_choice = st.sidebar.selectbox("Choose Benchmark Dataset:", list(preset_configs.keys()))
        cfg = preset_configs[preset_choice]
        default_model = cfg["model"]
        dataset_name = preset_choice
        dataset_key = f"preset_{preset_choice}"

        slice_len = st.sidebar.slider("Evaluation Window Length (steps):", 1000, 5000, cfg["len"], 500)

        data_dir = os.path.join(ROOT_DIR, "data")
        try:
            full_raw, full_labels = load_dataset_arrays(cfg["ds"], cfg["entity"], data_dir)
            st_idx = cfg["start"]
            ed_idx = min(st_idx + slice_len, len(full_raw))
            raw_data = full_raw[st_idx:ed_idx]
            ground_truth = full_labels[st_idx:ed_idx]
            time_index = np.arange(len(raw_data))
            feature_names = [f"Ch {i}" for i in range(raw_data.shape[1])]
        except Exception as e:
            st.error(f"Error loading benchmark dataset '{preset_choice}': {e}")
            return

    # Sidebar: Model Selection
    st.sidebar.header("2. Model Selection")
    model_choices = list(VisTADServingEngine.AVAILABLE_MODELS.keys())

    # Synchronize default model with dataset selection in session state
    if "selected_model_name" not in st.session_state:
        st.session_state["selected_model_name"] = default_model
        st.session_state["last_dataset_key"] = dataset_key

    if st.session_state.get("last_dataset_key") != dataset_key:
        st.session_state["last_dataset_key"] = dataset_key
        st.session_state["selected_model_name"] = default_model
        if "incident_table" in st.session_state:
            del st.session_state["incident_table"]

    selected_model_name = st.sidebar.selectbox(
        "Checkpoint:",
        model_choices,
        key="selected_model_name",
        help="Universal Foundation is trained across all 4 domains for zero-shot testing. Domain experts are specialized baselines."
    )

    # Sidebar: Operating Decision Threshold (Configurable by user, defaults to model-calibrated tau*)
    st.sidebar.header("3. Decision Threshold")
    model_cfg = VisTADServingEngine.AVAILABLE_MODELS.get(selected_model_name, {})
    default_tau = float(model_cfg.get("default_threshold", 0.20))

    if st.session_state.get("last_model_for_threshold") != selected_model_name:
        st.session_state["last_model_for_threshold"] = selected_model_name
        st.session_state["tau_slider_value"] = default_tau

    operating_tau = st.sidebar.slider(
        "Operating Threshold (tau):",
        min_value=0.01,
        max_value=1.00,
        value=st.session_state.get("tau_slider_value", default_tau),
        step=0.005,
        format="%.4f",
        key="tau_slider_value",
        help=f"Best calibrated default for {selected_model_name}: tau* = {default_tau:.4f}"
    )

    with st.sidebar.expander("Alarm Post-Processing (Hysteresis)", expanded=False):
        apply_smoothing = st.checkbox("Enable Hysteresis Smoothing", value=True, help="Merges fragmented threshold flickers into cohesive alarms")
        min_dur = st.slider("Minimum Duration (steps):", 1, 10, 3)
        merge_gap = st.slider("Merge Gap (steps):", 0, 30, 10)

    # Load Model Checkpoint
    try:
        engine.load_model(selected_model_name)
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
        merge_gap=merge_gap
    )

    scores = results["continuous_scores"]
    operating_tau = results["threshold"]
    preds = results["binary_predictions"]
    detected_events = results["detected_events"]
    selected_ch = results["selected_channels"]

    # Pre-build incident records
    records = []
    if detected_events:
        for idx_e, (st_idx, ed_idx) in enumerate(detected_events):
            dur = ed_idx - st_idx
            peak_score = float(np.max(scores[st_idx:ed_idx])) if dur > 0 else scores[st_idx]
            records.append({
                "Incident": f"Alarm #{idx_e + 1}",
                "Start Step": st_idx,
                "End Step": ed_idx,
                "Duration": f"{dur} steps",
                "Peak Score": f"{peak_score:.3f}",
                "Severity": "CRITICAL" if peak_score > 0.60 else "WARNING"
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
                "vus_pr": vus.get("vus_pr", np.nan),
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
        model_type_label = "Universal Foundation" if "Universal" in selected_model_name else "Domain Expert"
        arch_type = "Residual-TCN" if model_cfg.get("arch") == "tcn" else ("TimesNet" if model_cfg.get("arch") == "timesnet" else "CI 1D-CNN")
        st.markdown(f"""
        <div class="metric-box">
            <div class="label">Active Model</div>
            <div class="val">{model_type_label}</div>
            <div class="desc">{arch_type} ({len(selected_ch)} active ch)</div>
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
        m1.metric("Affiliation F1", fmt_metric(eval_metrics["aff_f1"]))
        m2.metric("Affiliation Precision", fmt_metric(eval_metrics["aff_p"]))
        m3.metric("Affiliation Recall", fmt_metric(eval_metrics["aff_r"]))
        m4.metric("AUROC", fmt_metric(eval_metrics["auroc"]))
        m5.metric("VUS-PR", fmt_metric(eval_metrics["vus_pr"]))

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
    row_heights = [0.42, 0.40, 0.18] if has_ground_truth else [0.52, 0.48]
    sub_titles = [
        f"Normalized Sensor Telemetry ({dataset_name})", 
        f"VisTAD Anomaly Likelihood Score vs Decision Threshold (tau = {operating_tau:.3f})"
    ]
    if has_ground_truth:
        sub_titles.append("Ground Truth Anomaly State [0 = Nominal, 1 = Active Event]")

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

    # 2. Sensor Traces (Top-2 Most Active Channels)
    ch_colors = ["#1e293b", "#0284c7", "#0d9488"]
    for idx_ch, ch in enumerate(selected_ch[:2]):
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
            name="Anomaly Score s(t)", 
            line=dict(color="#dc2626", width=1.6)
        ),
        row=2, col=1
    )

    # 4. Operating Decision Threshold Line
    fig.add_trace(
        go.Scatter(
            x=[t_axis[0], t_axis[-1]], y=[operating_tau, operating_tau], mode="lines",
            name=f"Threshold tau ({operating_tau:.3f})",
            line=dict(color="#2563eb", width=1.5, dash="dash")
        ),
        row=2, col=1
    )

    # 5. Explicit Ground Truth Step-Line and Selected Incident Overlay (Row 3, if available)
    if has_ground_truth:
        fig.add_trace(
            go.Scatter(
                x=t_axis, y=ground_truth, mode="lines",
                name="Ground Truth y(t)",
                line=dict(shape="hv", color="#dc2626", width=2),
                fill="tozeroy",
                fillcolor="rgba(220, 38, 38, 0.15)"
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
                    name="Selected Incident(s)",
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
        height=660 if has_ground_truth else 540,
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=40, r=40, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="Sensor (std)", row=1, col=1)
    fig.update_yaxes(title_text="Score", row=2, col=1, range=[0, max(1.0, float(np.max(scores) * 1.15))])
    fig.update_xaxes(title_text="Timestep", row=num_rows, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # Detected Incidents Ledger
    st.subheader("Detected Incidents Ledger")
    st.caption("Select rows to mark their intervals in the ground truth graph above.")
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
