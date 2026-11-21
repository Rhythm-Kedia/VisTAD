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

# Try importing heavy ML dependencies (requires torch/sklearn – not needed in Mock Preview mode)
try:
    from serving import VisTADServingEngine, load_model_registry
    from src.dataset import load_dataset_arrays
    from src.metrics import (
        find_events,
        compute_affiliation_metrics,
        compute_vus_metrics,
        compute_point_metrics
    )
    _HEAVY_IMPORTS_OK = True
except Exception as _import_err:
    _HEAVY_IMPORTS_OK = False
    def load_model_registry(path=None):
        return None

# Load master registry from external YAML config outside python folder
_CONFIG = load_model_registry() if _HEAVY_IMPORTS_OK else None
if _CONFIG is None:
    try:
        import yaml
        c_path = os.path.join(ROOT_DIR, "configs", "streamlit_models.yaml")
        if os.path.exists(c_path):
            with open(c_path, "r") as f:
                _CONFIG = yaml.safe_load(f)
    except Exception:
        _CONFIG = None

_MOCK_MODEL_CATALOG = dict(_CONFIG.get("models", {})) if _CONFIG else {}

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
    # ── Mock Preview Toggle ──────────────────────────────────────────────────
    mock_mode = st.sidebar.checkbox(
        "Mock Preview (no model weights needed)",
        value=not _HEAVY_IMPORTS_OK,
        disabled=not _HEAVY_IMPORTS_OK,
        help="Generates synthetic anomaly scores so you can explore the full UI without loading PyTorch models or checkpoints."
    )
    if not _HEAVY_IMPORTS_OK and not mock_mode:
        st.sidebar.warning("PyTorch / scikit-learn not found in this environment. Mock Preview is auto-enabled.")

    if mock_mode:
        engine = None
    else:
        engine = get_engine()

    # Header
    st.title("VisTAD: Vision based Time Series Anomaly Detection")
    #st.caption("Any-Dimension Foundation Models & Distilled Specialized Students with Continuous Probability Calibration.")

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
        # Explicit dataset origin selector: no automatic parsing or filename guessing
        dataset_origin = st.sidebar.selectbox(
            "Select Uploaded Dataset Domain:",
            [
                "(None / Custom Unseen Telemetry)",
                "SWaT (Water Treatment - 51 Channels)",
                "GECCO (Drinking Water Quality - 9 Channels)",
                "SMD (Server Machine Dataset - 38 Metrics)",
                "MSL (Mars Science Laboratory - 55 Channels)"
            ],
            index=0,
            help="Select the benchmark plant if this file matches its certified channel layout. If left as None/Custom, only the Universal Foundation Pro model will be presented."
        )

        # Quick-load samples from bundled sample_data/ (always available)
        samples_dir = os.path.join(CURRENT_DIR, "sample_data")
        sample_files = sorted([f for f in os.listdir(samples_dir) if f.endswith(".csv")]) if os.path.exists(samples_dir) else []
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
            sample_path = os.path.join(CURRENT_DIR, "sample_data", sample_choice)
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

        total_rows = len(converted_df)

        # ── Prediction Interval Selector (plot drag + slider, synced) ────────
        MIN_ROWS = 100
        state_key = f"interval_{dataset_key}_{total_rows}"
        ver_key = f"interval_ver_{dataset_key}_{total_rows}"
        slider_key = f"interval_slider_{dataset_key}_{total_rows}"
        default_interval = (0, min(2999, total_rows - 1))

        if state_key not in st.session_state:
            st.session_state[state_key] = default_interval
        if ver_key not in st.session_state:
            st.session_state[ver_key] = 0
        if slider_key not in st.session_state:
            st.session_state[slider_key] = st.session_state[state_key]

        def _on_slider_change():
            s, e = st.session_state[slider_key]
            if e - s < MIN_ROWS:
                e = min(s + MIN_ROWS, total_rows - 1)
                s = max(0, e - MIN_ROWS)
            st.session_state[state_key] = (s, e)
            st.session_state[slider_key] = (s, e)
            st.session_state[ver_key] += 1  # clears any old box on the plot

        st.sidebar.markdown("**Prediction Interval**")
        preview_col = st.sidebar.selectbox(
            "Preview channel:", list(converted_df.columns), key=f"prev_ch_{dataset_key}"
        )
        b1, b2 = st.sidebar.columns(2)
        reset_clicked = b1.button("Reset", use_container_width=True)
        full_clicked = b2.button("Full range", use_container_width=True)
        if reset_clicked or full_clicked:
            new_iv = default_interval if reset_clicked else (0, total_rows - 1)
            st.session_state[state_key] = new_iv
            st.session_state[slider_key] = new_iv
            st.session_state[ver_key] += 1

        interval_start, interval_end = st.session_state[state_key]

        # Selectable overview plot (downsampled for speed)
        MAX_PTS = 20000
        step = max(1, total_rows // MAX_PTS)
        x_full = np.arange(0, total_rows, step)
        y_full = converted_df[preview_col].values[::step]
        sel_mask = (x_full >= interval_start) & (x_full <= interval_end)

        fig_sel = go.Figure()
        fig_sel.add_trace(go.Scatter(
            x=x_full, y=y_full, mode="lines", name="Full signal",
            line=dict(color="#94a3b8", width=1)
        ))
        fig_sel.add_trace(go.Scatter(
            x=x_full[sel_mask], y=y_full[sel_mask], mode="lines", name="Selected interval",
            line=dict(color="#0284c7", width=2)
        ))
        fig_sel.add_vrect(
            x0=interval_start, x1=interval_end,
            fillcolor="rgba(2,132,199,0.12)", line_width=1, line_color="#0284c7", layer="below"
        )
        fig_sel.update_layout(
            height=230,
            margin=dict(l=30, r=20, t=30, b=30),
            template="plotly_white",
            dragmode="select",
            selectdirection="h",
            legend=dict(orientation="h", y=1.18),
            xaxis_title="Row index",
            yaxis_title=preview_col,
        )

        st.subheader("Select Prediction Interval")
        st.caption("Drag horizontally on the plot, or use the sidebar slider. Both stay in sync. "
                   f"Minimum {MIN_ROWS} rows.")
        event = st.plotly_chart(
            fig_sel,
            use_container_width=True,
            key=f"interval_chart_{dataset_key}_{st.session_state[ver_key]}",
            on_select="rerun",
            selection_mode=("box",),
        )

        # Plot drag -> update stored interval AND the slider (slider not created yet this run)
        boxes = []
        try:
            boxes = event.get("selection", {}).get("box", []) if event else []
        except Exception:
            boxes = []
        if boxes and boxes[0].get("x") and len(boxes[0]["x"]) == 2:
            x0, x1 = sorted(boxes[0]["x"])
            new_start = int(np.clip(round(x0), 0, total_rows - 1))
            new_end = int(np.clip(round(x1), 0, total_rows - 1))
            if new_end - new_start < MIN_ROWS:
                new_end = min(new_start + MIN_ROWS, total_rows - 1)
                new_start = max(0, new_end - MIN_ROWS)
            if (new_start, new_end) != tuple(st.session_state[state_key]):
                st.session_state[state_key] = (new_start, new_end)
                st.session_state[slider_key] = (new_start, new_end)
                st.rerun()

        # The two-handle range slider, synced with the plot
        st.sidebar.slider(
            "Row range to predict:",
            min_value=0,
            max_value=total_rows - 1,
            step=1,
            key=slider_key,          # value comes from session state only
            on_change=_on_slider_change,
            help="Drag the handles, or drag on the plot. Both update each other."
        )

        interval_start, interval_end = st.session_state[state_key]
        slice_df = converted_df.iloc[interval_start : interval_end + 1]
        raw_data = slice_df.values
        feature_names = list(converted_df.columns)

        st.sidebar.caption(
            f"**Selected:** rows {interval_start:,} → {interval_end:,} "
            f"({len(raw_data):,} steps, {raw_data.shape[1]} channels)"
        )

        if selected_gt_col != "(None / Unlabeled)" and selected_gt_col in uploaded_df.columns:
            gt_series = parse_ground_truth_column(
                uploaded_df[selected_gt_col].iloc[interval_start : interval_end + 1]
            )
            ground_truth = gt_series

        if time_col is not None:
            time_index = uploaded_df[time_col].values[interval_start : interval_end + 1]
        else:
            time_index = np.arange(len(raw_data))

        # Determine target domain based strictly on user selection (no guessing from filename)
        domain_expected_ch = {
            "SWaT": 51,
            "GECCO": 9,
            "SMD": 38,
            "MSL": 55
        }
        origin_domain_map = {
            "SWaT (Water Treatment - 51 Channels)": "SWaT",
            "GECCO (Drinking Water Quality - 9 Channels)": "GECCO",
            "SMD (Server Machine Dataset - 38 Metrics)": "SMD",
            "MSL (Mars Science Laboratory - 55 Channels)": "MSL"
        }

        selected_domain = origin_domain_map.get(dataset_origin)
        num_channels = raw_data.shape[1]

        if selected_domain is not None:
            expected_ch = domain_expected_ch.get(selected_domain, 0)
            if num_channels == expected_ch:
                active_domain = selected_domain
            else:
                st.sidebar.warning(
                    f"**Channel Count Mismatch**: '{selected_domain}' models require {expected_ch} channels, "
                    f"but this CSV has {num_channels} channels. "
                    "Routing to **Universal Foundation TCN Pro** (supports arbitrary sensor counts)."
                )
                active_domain = "CUSTOM"
        else:
            # If the user doesn't choose anything, present ONLY the Universal Pro model
            active_domain = "CUSTOM"

        # Collapsible Data Preview
        with st.expander("Preview Uploaded Data Table", expanded=False):
            st.dataframe(uploaded_df.head(5), use_container_width=True)

    else:
        # Benchmark presets with offsets that contain genuine anomaly episodes
        preset_configs = {
            "SWaT Water Treatment (Attack Intervals) [51 Sensors]": {
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
        # In mock mode: load from bundled sample_data CSVs (no data/ directory or torch needed)
        sample_data_dir = os.path.join(CURRENT_DIR, "sample_data")
        if mock_mode:
            sample_file_path = os.path.join(sample_data_dir, cfg.get("sample_file", ""))
            _feat_cols = None
            if os.path.exists(sample_file_path):
                _df = pd.read_csv(sample_file_path)
                _label_col = next((c for c in _df.columns if any(k in c.lower() for k in ["label", "attack", "anomaly"])), None)
                _feat_cols = [c for c in _df.columns if c not in ([_label_col] if _label_col else []) and c.lower() not in ["timestamp", "time", "date"]]
                _df_feats = _df[_feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
                full_raw = _df_feats.values
                full_labels = _df[_label_col].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(int) if _label_col else np.zeros(len(full_raw), dtype=int)
            else:
                # Fully synthetic fallback
                n_channels = {"SWaT": 51, "GECCO": 9, "SMD": 38, "MSL": 55}.get(cfg["domain"], 20)
                full_raw = np.random.randn(5000, n_channels).astype(np.float32)
                full_labels = np.zeros(5000, dtype=int)
                full_labels[1500:1700] = 1
            st_idx = min(cfg["start"], max(0, len(full_raw) - slice_len))
            ed_idx = min(st_idx + slice_len, len(full_raw))
            raw_data = full_raw[st_idx:ed_idx]
            ground_truth = full_labels[st_idx:ed_idx]
            time_index = np.arange(len(raw_data))
            feature_names = _feat_cols if _feat_cols is not None else [f"Sensor {i}" for i in range(raw_data.shape[1])]
            ref_len = min(3000, len(full_raw))
            clean_base_m = np.mean(full_raw[:ref_len], axis=0)
            clean_base_s = np.maximum(np.std(full_raw[:ref_len], axis=0), 0.05)
        else:
            try:
                full_raw, full_labels = load_dataset_arrays(cfg["ds"], cfg["entity"], data_dir)
                st_idx = cfg["start"]
                ed_idx = min(st_idx + slice_len, len(full_raw))
                raw_data = full_raw[st_idx:ed_idx]
                ground_truth = full_labels[st_idx:ed_idx]
                time_index = np.arange(len(raw_data))
                feature_names = get_feature_names_for_dataset(cfg["ds"], raw_data.shape[1])

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
                    ref_len = int(len(full_raw) * 0.50) if cfg["ds"] == "gecco" else min(3000, len(full_raw))
                    clean_base_m = np.mean(full_raw[:ref_len], axis=0)
                    clean_base_s = np.std(full_raw[:ref_len], axis=0)
                    clean_base_s = np.where(clean_base_s == 0, 1.0, clean_base_s)
                    clean_base_s = np.maximum(clean_base_s, 0.05)
            except Exception as e:
                st.error(f"Error loading benchmark dataset '{preset_choice}': {e}")
                return

    # -- Channel Permutation Invariance Stress Test -----------------------
    st.sidebar.markdown("---")
    permute_channels = st.sidebar.checkbox(
        "Permute Channel Order (Stress Test)",
        value=False,
        key="permute_channels_checkbox",
        help="Randomly shuffles sensor channel ordering. Tests architectural permutation equivariance: Universal Foundation TCN Pro remains invariant, whereas domain-specific coupled models experience performance degradation."
    )

    perm_seed_key = f"perm_seed_{dataset_key}"
    if perm_seed_key not in st.session_state:
        st.session_state[perm_seed_key] = 42

    if permute_channels:
        col_p1, col_p2 = st.sidebar.columns([3, 2])
        with col_p1:
            st.caption(f"Active Seed: `{st.session_state[perm_seed_key]}`")
        with col_p2:
            if st.button("Reshuffle", use_container_width=True):
                st.session_state[perm_seed_key] = int(np.random.randint(1, 100000))
                st.rerun()

        rng_p = np.random.default_rng(st.session_state[perm_seed_key])
        n_c = raw_data.shape[1]
        perm_idx = rng_p.permutation(n_c)

        # Apply permutation to telemetry array, sensor names, and baseline statistics
        raw_data = raw_data[:, perm_idx]
        feature_names = [feature_names[i] for i in perm_idx]
        if clean_base_m is not None and len(clean_base_m) == n_c:
            clean_base_m = clean_base_m[perm_idx]
        if clean_base_s is not None and len(clean_base_s) == n_c:
            clean_base_s = clean_base_s[perm_idx]

        perm_str = f"perm_{st.session_state[perm_seed_key]}"
    else:
        perm_str = "unperm"

    # Sidebar: Model Selection (Configured Dynamically from configs/streamlit_models.yaml)
    st.sidebar.header("2. Model Selection")

    if engine is not None:
        catalog = getattr(engine, "catalog", getattr(engine, "MODEL_CATALOG", _MOCK_MODEL_CATALOG))
    elif _HEAVY_IMPORTS_OK:
        catalog = getattr(VisTADServingEngine, "MODEL_CATALOG", _MOCK_MODEL_CATALOG)
    else:
        catalog = _MOCK_MODEL_CATALOG

    cfg_families = _CONFIG.get("families", {}) if _CONFIG else {}

    # Filter catalog by active domain
    domain_clean = active_domain.lower()
    matching_models = {}
    for mk, minfo in catalog.items():
        m_ds = str(minfo.get("dataset", minfo.get("domain", ""))).lower()
        if domain_clean in m_ds or m_ds in domain_clean or (domain_clean == "custom" and m_ds in ["all", "custom"]):
            matching_models[mk] = minfo

    if not matching_models:
        matching_models = catalog

    # Determine available families for this domain
    present_fam_keys = []
    for mk, minfo in matching_models.items():
        fam = minfo.get("family", "coupled_domain")
        if fam not in present_fam_keys:
            present_fam_keys.append(fam)

    # Sort families by order defined in config
    present_fam_keys.sort(key=lambda f: cfg_families.get(f, {}).get("order", 99))

    fam_display_map = {}
    for f_key in present_fam_keys:
        f_info = cfg_families.get(f_key, {})
        badge = f_info.get("badge", "")
        f_name = f_info.get("name", f_key.replace("_", " ").title())
        display_label = f"{f_name} [{badge}]" if badge else f_name
        fam_display_map[display_label] = f_key

    selected_fam_label = st.sidebar.selectbox(
        "Model Architecture Family:",
        list(fam_display_map.keys()),
        index=0,
        help="Select neural student or foundation architecture. Coupled Domain TCN models physical plant stages; Universal Foundation models arbitrary sensor counts."
    )
    selected_fam_key = fam_display_map[selected_fam_label]

    # Models belonging to chosen family
    fam_models = {
        mk: minfo for mk, minfo in matching_models.items()
        if minfo.get("family") == selected_fam_key
    }
    if not fam_models:
        fam_models = matching_models

    # Supervision Cache Manifold options for this family
    cache_variants = {}
    for mk, minfo in fam_models.items():
        c_type = minfo.get("cache_type", "New Cache (1024-d Ultra-VETime)")
        cache_variants[c_type] = mk

    if len(cache_variants) > 1:
        def_idx = 0
        for idx_c, c_name in enumerate(cache_variants.keys()):
            if "New Cache" in c_name or "1024" in c_name:
                def_idx = idx_c
                break
        selected_cache_label = st.sidebar.radio(
            "Teacher Supervision Cache:",
            list(cache_variants.keys()),
            index=def_idx,
            help="Teacher manifold used for supervision: New Cache (1024-d Ultra-VETime/DINOv2) vs Old Cache (2048-d Early VLM)."
        )
        selected_model_key = cache_variants[selected_cache_label]
    else:
        selected_model_key = next(iter(cache_variants.values()))

    model_cfg = catalog.get(selected_model_key, {})

    # Render Clean Model Specification Card
    bench_m = model_cfg.get("metrics", {})
    auroc_str = f"{bench_m['auroc']:.4f}" if "auroc" in bench_m else "-"
    aff_f1_str = f"{bench_m['aff_f1']:.4f}" if "aff_f1" in bench_m else "-"
    ev_rec_str = str(bench_m.get("event_recall", "-"))
    lat_str = f"{bench_m['latency_ms']:.2f} ms" if "latency_ms" in bench_m else "-"

    ckpt_display = os.path.basename(model_cfg.get("checkpoint", model_cfg.get("file", "-")))
    arch_display = model_cfg.get("arch_class", model_cfg.get("arch_name", "-"))
    params_display = model_cfg.get("params", "-")
    cache_display = model_cfg.get("cache_type", "-")

    st.sidebar.markdown(f"""
    <div class="model-spec-card">
        <b>Model:</b> {model_cfg.get('name', selected_model_key)}<br>
        <b>Architecture:</b> <code>{arch_display}</code> ({params_display})<br>
        <b>Supervision:</b> {cache_display}<br>
        <b>Checkpoint:</b> <code>{ckpt_display}</code><br>
        <b>Master Verified Benchmark:</b><br>
        &nbsp;• AUROC: <b>{auroc_str}</b> | Aff-F1: <b>{aff_f1_str}</b><br>
        &nbsp;• Event Recall: <b>{ev_rec_str}</b><br>
        &nbsp;• Target Latency: <b>{lat_str}</b><br>
        <b>Calibrated Cutoff:</b> τ* = 0.5
    </div>
    """, unsafe_allow_html=True)
    # Sidebar: Operating Decision Threshold (Always defaults to 0.50 for all models on opening/selection)
    st.sidebar.header("3. Decision Threshold")
    calibrated_tau = float(model_cfg.get("default_threshold", 0.50))

    if "tau_slider_value" not in st.session_state or st.session_state.get("last_model_for_threshold") != selected_model_key:
        st.session_state["last_model_for_threshold"] = selected_model_key
        st.session_state["tau_slider_value"] = 0.50

    operating_tau = st.sidebar.slider(
        "Operating Threshold (tau):",
        min_value=0.01,
        max_value=1.00,
        step=0.005,
        format="%.4f",
        key="tau_slider_value",
        help=f"Standard threshold: 0.5000 | Master calibrated cutoff for {model_cfg.get('name')}: tau* = {calibrated_tau:.4f}"
    )

    with st.sidebar.expander("Alarm Post-Processing (Hysteresis)", expanded=False):
        apply_smoothing = st.checkbox("Enable Hysteresis Smoothing", value=True, help="Merges fragmented threshold flickers into cohesive alarms")
        min_dur = st.slider("Minimum Duration (steps):", 1, 10, 3)
        merge_gap = st.slider("Merge Gap (steps):", 0, 30, 10)

    # Load Model Checkpoint / Mock Inference
    if mock_mode:
        st.sidebar.info("Mock Preview active — synthetic scores generated, no weights loaded.")
        rng = np.random.default_rng(seed=42)
        n_steps = len(raw_data) if raw_data is not None else 3000
        n_ch = raw_data.shape[1] if raw_data is not None else 6
        # Build a smooth base score with two synthetic anomaly bursts
        base_scores = np.clip(rng.normal(0.12, 0.06, n_steps), 0.0, 1.0)
        burst1_s, burst1_e = max(0, n_steps // 4), min(n_steps, n_steps // 4 + 180)
        burst2_s, burst2_e = max(0, n_steps * 2 // 3), min(n_steps, n_steps * 2 // 3 + 120)
        base_scores[burst1_s:burst1_e] = np.clip(rng.normal(0.82, 0.07, burst1_e - burst1_s), 0.0, 1.0)
        base_scores[burst2_s:burst2_e] = np.clip(rng.normal(0.74, 0.08, burst2_e - burst2_s), 0.0, 1.0)
        # Smooth with a moving average
        kernel = np.ones(15) / 15
        scores = np.convolve(base_scores, kernel, mode="same")
        preds = (scores >= operating_tau).astype(int)
        # Build channel attributions
        ch_attr = rng.uniform(0.0, 0.3, (n_ch, n_steps))
        ch_attr[:, burst1_s:burst1_e] = rng.uniform(0.5, 1.0, (n_ch, burst1_e - burst1_s))
        ch_attr[:, burst2_s:burst2_e] = rng.uniform(0.4, 0.9, (n_ch, burst2_e - burst2_s))
        channel_attrs = ch_attr
        detected_events = [(burst1_s, burst1_e), (burst2_s, burst2_e)]
        event_details = [
            {"start": burst1_s, "end": burst1_e, "duration": burst1_e - burst1_s,
             "peak_prob": float(np.max(scores[burst1_s:burst1_e])),
             "top_sensors": [(i, float(ch_attr[i, burst1_s:burst1_e].mean())) for i in np.argsort(ch_attr[:, burst1_s:burst1_e].mean(axis=1))[::-1][:5]]},
            {"start": burst2_s, "end": burst2_e, "duration": burst2_e - burst2_s,
             "peak_prob": float(np.max(scores[burst2_s:burst2_e])),
             "top_sensors": [(i, float(ch_attr[i, burst2_s:burst2_e].mean())) for i in np.argsort(ch_attr[:, burst2_s:burst2_e].mean(axis=1))[::-1][:5]]},
        ]
        selected_ch = list(range(min(6, n_ch)))
        results = {
            "continuous_scores": scores, "channel_attributions": ch_attr,
            "threshold": operating_tau, "binary_predictions": preds,
            "detected_events": detected_events, "event_details": event_details,
            "selected_channels": selected_ch,
            "inference_time_ms": 0.0, "ms_per_window": 0.0, "n_windows": 0,
        }
    else:
        # Load Model Checkpoint
        try:
            engine.load_model(selected_model_key)
        except Exception as e:
            st.error(f"Failed to load checkpoint: {e}")
            st.info("Tip: Enable **'Mock Preview'** in the sidebar to view the full UI without model weights.")
            return

        # Neural Inference Caching in Session State:
        # Avoids 2-5 second GPU/CPU re-computation on table clicks or slider adjustments
        inf_cache_key = f"inf_{selected_model_key}_{dataset_key}_{len(raw_data)}_{perm_str}"
        if inf_cache_key not in st.session_state:
            with st.spinner(f"Computing continuous inference for {model_cfg.get('name', selected_model_key)}..."):
                inf_res = engine.infer_continuous_scores(
                    raw_data=raw_data,
                    window_size=200,
                    stride=10,
                    prefix_len=250,
                    base_mean=clean_base_m,
                    base_std=clean_base_s
                )
                st.session_state[inf_cache_key] = inf_res
        else:
            inf_res = st.session_state[inf_cache_key]

        # Ultra-Fast Range Postprocessing (< 1 ms pure NumPy)
        post_res = engine.postprocess_alarms(
            continuous_scores=inf_res["continuous_scores"],
            channel_continuous=inf_res["channel_attributions"],
            operating_threshold=operating_tau,
            threshold_mode="fixed",
            apply_smoothing=apply_smoothing,
            min_duration=min_dur,
            merge_gap=merge_gap,
            raw_arr=inf_res["raw_arr"],
            base_m=inf_res["base_m"],
            base_s=inf_res["base_s"]
        )
        results = {**inf_res, **post_res}

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

    # Quantitative Evaluation (if ground truth is available and not in mock mode)
    eval_metrics = None
    has_ground_truth = (ground_truth is not None and np.any(ground_truth >= 0.5))
    if has_ground_truth and not mock_mode:
        try:
            pt = compute_point_metrics(ground_truth, scores, threshold=operating_tau)
            aff = compute_affiliation_metrics(ground_truth, preds)
            vus = compute_vus_metrics(ground_truth, scores, max_buffer=50)
            eval_metrics = {
                "auroc": pt.get("auroc", np.nan),
                "prauc": pt.get("pr_auc", pt.get("prauc", np.nan)),
                "aff_f1": aff.get("aff_f1", np.nan),
                "aff_p": aff.get("aff_precision", np.nan),
                "aff_r": aff.get("aff_recall", np.nan)
            }
        except Exception:
            eval_metrics = None

    # Permutation Stress Test Notice Banner
    if permute_channels:
        st.warning(
            f"**Permutation Equivariance Stress Test Active (Seed: {st.session_state[perm_seed_key]})**: All {raw_data.shape[1]} sensor channels "
            "have been randomly permuted."
        )

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
        fam_name = model_cfg.get("family_name", model_cfg.get("name", "Active Student"))
        arch_lbl = model_cfg.get("arch_class", model_cfg.get("arch_name", "TCN"))
        st.markdown(f"""
        <div class="metric-box">
            <div class="label">Active Model</div>
            <div class="val">{fam_name}</div>
            <div class="desc">{arch_lbl} ({len(selected_ch)} ch, {model_cfg.get('params', '-')})</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        latency_val = "Mock" if mock_mode else f"{results['ms_per_window']:.3f} ms / win"
        latency_desc = "Preview mode (no inference)" if mock_mode else f"Total {results['inference_time_ms']:.1f} ms"
        st.markdown(f"""
        <div class="metric-box">
            <div class="label">Inference Latency</div>
            <div class="val">{latency_val}</div>
            <div class="desc">{latency_desc}</div>
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
        st.markdown(f"**Active Slice Evaluation Scores (Current {len(raw_data):,} Steps)**")
        m1, m2, m3, m4, m5 = st.columns(5)
        m_auroc_ref = f"Master Certified 100% Split: {bench_m['auroc']:.4f}" if "auroc" in bench_m else None
        m_prauc_ref = f"Master Certified 100% Split: {bench_m['prauc']:.4f}" if "prauc" in bench_m else None
        m_aff_f1_ref = f"Master Certified 100% Split: {bench_m['aff_f1']:.4f}" if "aff_f1" in bench_m else None
        m1.metric("Slice AUROC", fmt_metric(eval_metrics["auroc"]), help=m_auroc_ref)
        m2.metric("Slice PR-AUC", fmt_metric(eval_metrics["prauc"]), help=m_prauc_ref)
        m3.metric("Slice Affiliation F1", fmt_metric(eval_metrics["aff_f1"]), help=m_aff_f1_ref)
        m4.metric("Slice Affiliation Precision", fmt_metric(eval_metrics["aff_p"]))
        m5.metric("Slice Affiliation Recall", fmt_metric(eval_metrics["aff_r"]))

        # # Explanatory architectural notes for baseline vs coupled models
        # if "baseline" in model_cfg.get("family", "").lower() or "cnn" in model_cfg.get("family", "").lower():
        #     st.caption(
        #         f"**Baseline Architecture Notice**: This uncoupled baseline model processes all {raw_data.shape[1]} sensor channels "
        #         "independently using max-pooling. In complex multi-stage industrial telemetry, independent channels are vulnerable to "
        #         "transient plant noise, which accounts for lower affiliation precision and the master-verified 0.1319 Aff-F1 in Table 1.1 "
        #         "compared to Coupled Domain TCN (Aff-F1 = 0.7165)."
        #     )

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
            sel_mask_inc = np.zeros(len(t_axis))
            for idx in selected_indices:
                s_rec = records[idx]
                s_st = s_rec["Start Step"]
                s_ed = s_rec["End Step"]
                sel_mask_inc[s_st:min(s_ed, len(t_axis))] = 1.0

            fig.add_trace(
                go.Scatter(
                    x=t_axis, y=sel_mask_inc, mode="lines",
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
            st.markdown("**Root-Cause Diagnosis**")
            st.caption(f"Top sensor **{top_labels[0]}** reached peak probability **{top_values[0]:.3f}**.")
            st.markdown(f"- **Primary Suspect**: `{top_labels[0]}` (p = {top_values[0]:.3f})")
            if len(top_labels) > 1:
                st.markdown(f"- **Secondary Suspect**: `{top_labels[1]}` (p = {top_values[1]:.3f})")
            if len(top_labels) > 2:
                st.markdown(f"- **Tertiary Suspect**: `{top_labels[2]}` (p = {top_values[2]:.3f})")
            st.caption("Cross-channel transformer computes mutual physical correlation to isolate root cause sensors.")
    else:
        st.info("Root-cause per-channel attribution is active for Universal-TCN-Pro foundation models.")

    # Detected Anomalous Incidents
    st.write("")
    st.subheader("Detected Anomalous Incidents")
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