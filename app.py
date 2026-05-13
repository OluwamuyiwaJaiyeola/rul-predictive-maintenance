# =============================================================================
# app.py - Kataoka Inc. Predictive Maintenance System
# Final Model: Random Forest Config C
# =============================================================================

import warnings
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

MODEL_PATH = Path("models/tuned/rul_regressor_final.pkl")
SCALER_PATH = Path("models/tuned/scaler_final.pkl")
MERGED_DATA_PATH = Path("data/processed/merged_dataset.csv")
FEATURED_DATA_PATH = Path("data/processed/featured_dataset.csv")

REPORT_R2 = 0.862
REPORT_OVERALL_MAE_HOURS = 630.3
REPORT_CRITICAL_MAE_HOURS = 129.7
REPORT_MIN_LEAD_HOURS = 234

CRITICAL_THRESHOLD = 168
WARNING_THRESHOLD = 720

TRAINING_BOUNDS = {
    "cumulative_hours": {"lower": 162.0, "upper": 8136.0},
    "total_maintenance_count": {"lower": 3.0, "upper": 5.0},
    "vibration_level": {"lower": 0.12, "upper": 0.68},
}

BASELINE_B_FEATURES = [
    "vibration_level", "motor_temperature", "torque_load", "power_consumption",
    "cumulative_hours", "total_maintenance_count", "total_downtime_hours",
    "avg_downtime_hours", "repair_count", "replacement_count", "lubrication_count",
    "calibration_count", "inspection_count", "model_type_encoded",
    "factory_location_encoded", "operating_environment_encoded",
]

CORE_FEATURES_V2 = [
    "load_stress_index",
    "vibration_rolling_std_7d", "temperature_rolling_std_7d", "torque_rolling_std_7d",
    "vibration_rolling_std_24h", "temperature_rolling_std_24h", "torque_rolling_std_24h",
    "vibration_rolling_mean_7d", "temperature_rolling_mean_7d", "torque_rolling_mean_7d",
    "vibration_short_long_diff", "temperature_short_long_diff",
]

CONFIG_C = BASELINE_B_FEATURES + CORE_FEATURES_V2


def load_assets():
    missing = [
        str(path) for path in [MODEL_PATH, SCALER_PATH, MERGED_DATA_PATH, FEATURED_DATA_PATH]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required project files:\n"
            + "\n".join(missing)
            + "\n\nRun the previous notebooks and confirm the files exist."
        )

    loaded_model = joblib.load(MODEL_PATH)
    loaded_scaler = joblib.load(SCALER_PATH)
    merged = pd.read_csv(MERGED_DATA_PATH, parse_dates=["timestamp", "failure_time"])
    featured = pd.read_csv(FEATURED_DATA_PATH, parse_dates=["timestamp", "failure_time"])
    return loaded_model, loaded_scaler, merged, featured


model, scaler, merged_df, featured_df = load_assets()


def add_label_encoding(df):
    df = df.copy()
    for col in ["model_type", "factory_location", "operating_environment"]:
        enc_col = f"{col}_encoded"
        if enc_col not in df.columns:
            encoder = LabelEncoder()
            df[enc_col] = encoder.fit_transform(df[col].astype(str))
    return df


merged_df = add_label_encoding(merged_df)
featured_df = add_label_encoding(featured_df)


def get_zone_label(rul):
    if rul < CRITICAL_THRESHOLD:
        return "CRITICAL"
    if rul < WARNING_THRESHOLD:
        return "WARNING"
    return "HEALTHY"


def get_zone_display(rul):
    return {
        "CRITICAL": "🔴 CRITICAL",
        "WARNING": "🟡 WARNING",
        "HEALTHY": "🟢 HEALTHY",
    }[get_zone_label(rul)]


def get_zone_color(rul):
    return {
        "CRITICAL": "#E8453C",
        "WARNING": "#F5A623",
        "HEALTHY": "#0BC49E",
    }[get_zone_label(rul)]


def get_recommendation(rul, model_type=""):
    is_welding = "WELDING" in str(model_type).upper()
    welding_note = (
        "\n⚠️ WELDING-W5 has no confirmed failure history in the training data. Treat this prediction as extrapolation."
        if is_welding else ""
    )
    if rul < CRITICAL_THRESHOLD:
        return (
            "CRITICAL: Immediate inspection required.\n"
            f"Predicted RUL: {rul:.0f} hours ({rul / 24:.1f} days).\n"
            f"Known uncertainty in Critical zone: approximately ±{REPORT_CRITICAL_MAE_HOURS:.0f} hours.\n"
            "Use this as an inspection trigger, not as a precise countdown."
            f"{welding_note}"
        )
    if rul < WARNING_THRESHOLD:
        return (
            "WARNING: Schedule maintenance within the next maintenance window.\n"
            f"Predicted RUL: {rul:.0f} hours ({rul / 24:.1f} days).\n"
            "Monitor rolling sensor trends and prepare downtime planning."
            f"{welding_note}"
        )
    return (
        "HEALTHY: Continue routine monitoring.\n"
        f"Predicted RUL: {rul:.0f} hours ({rul / 24:.1f} days).\n"
        "No immediate maintenance action required."
        f"{welding_note}"
    )


def clip_to_training_bounds(df):
    df = df.copy()
    df["cumulative_hours"] = df["cumulative_hours"].clip(
        lower=TRAINING_BOUNDS["cumulative_hours"]["lower"],
        upper=TRAINING_BOUNDS["cumulative_hours"]["upper"],
    )
    df["total_maintenance_count"] = df["total_maintenance_count"].clip(
        lower=TRAINING_BOUNDS["total_maintenance_count"]["lower"],
        upper=TRAINING_BOUNDS["total_maintenance_count"]["upper"],
    )
    df["vibration_level"] = df["vibration_level"].clip(
        lower=TRAINING_BOUNDS["vibration_level"]["lower"],
        upper=TRAINING_BOUNDS["vibration_level"]["upper"],
    )
    df["load_stress_index"] = df["torque_load"] * df["cumulative_hours"]
    return df


def compute_rolling_features(df, robot_id):
    robot_data = df[df["robot_id"] == robot_id].copy()
    robot_data = robot_data.sort_values("timestamp").reset_index(drop=True)
    if robot_data.empty:
        return None

    window_7d = 28
    window_24h = 4
    sensor_map = {
        "vibration_level": "vibration",
        "motor_temperature": "temperature",
        "torque_load": "torque",
        "power_consumption": "power",
    }

    for sensor_col, prefix in sensor_map.items():
        robot_data[f"{prefix}_rolling_mean_7d"] = robot_data[sensor_col].rolling(
            window_7d, min_periods=window_7d
        ).mean()
        robot_data[f"{prefix}_rolling_std_7d"] = robot_data[sensor_col].rolling(
            window_7d, min_periods=window_7d
        ).std()

    for sensor_col, prefix in [
        ("vibration_level", "vibration"),
        ("motor_temperature", "temperature"),
        ("torque_load", "torque"),
    ]:
        robot_data[f"{prefix}_rolling_std_24h"] = robot_data[sensor_col].rolling(
            window_24h, min_periods=window_24h
        ).std()

    robot_data["load_stress_index"] = robot_data["torque_load"] * robot_data["cumulative_hours"]

    vibration_short = robot_data["vibration_level"].rolling(window_24h, min_periods=window_24h).mean()
    temperature_short = robot_data["motor_temperature"].rolling(window_24h, min_periods=window_24h).mean()

    robot_data["vibration_short_long_diff"] = vibration_short - robot_data["vibration_rolling_mean_7d"]
    robot_data["temperature_short_long_diff"] = temperature_short - robot_data["temperature_rolling_mean_7d"]

    clean = robot_data.dropna(subset=CONFIG_C).copy()
    if clean.empty:
        return None
    return clip_to_training_bounds(clean)


def predict_history_for_robot(robot_id):
    history = compute_rolling_features(merged_df, robot_id)
    if history is None:
        return None
    x_scaled = scaler.transform(history[CONFIG_C].values)
    history = history.copy()
    history["predicted_rul"] = np.clip(model.predict(x_scaled), 0, None)
    history["predicted_zone"] = history["predicted_rul"].apply(get_zone_label)
    return history


def predict_latest_for_robot(robot_id):
    history = predict_history_for_robot(robot_id)
    if history is None or history.empty:
        return None, None
    return float(history["predicted_rul"].iloc[-1]), history


def build_fleet_predictions():
    rows = []
    for robot_id in sorted(merged_df["robot_id"].unique()):
        meta = merged_df[merged_df["robot_id"] == robot_id].iloc[0]
        latest_raw = merged_df[merged_df["robot_id"] == robot_id].sort_values("timestamp").iloc[-1]
        latest_rul, _ = predict_latest_for_robot(robot_id)

        status = "INSUFFICIENT DATA" if latest_rul is None else get_zone_label(latest_rul)
        model_type = str(meta["model_type"])
        is_welding = "WELDING" in model_type.upper()
        is_failure_robot = bool(meta.get("is_failure_robot", False))

        if is_welding:
            note = "Extrapolation: no WELDING-W5 failure history"
        elif not is_failure_robot:
            note = "Censored robot: higher uncertainty"
        else:
            note = ""

        rows.append({
            "Robot ID": robot_id,
            "Model Type": model_type,
            "Factory": meta["factory_location"],
            "Status": status,
            "Predicted RUL (hrs)": None if latest_rul is None else round(latest_rul),
            "Days Remaining": None if latest_rul is None else round(latest_rul / 24, 1),
            "Temperature": round(float(latest_raw["motor_temperature"]), 1),
            "Vibration": round(float(latest_raw["vibration_level"]), 3),
            "Cum. Hours": round(float(latest_raw["cumulative_hours"])),
            "Note": note,
        })

    fleet = pd.DataFrame(rows)
    fleet["sort_rul"] = fleet["Predicted RUL (hrs)"].fillna(10**9)
    return fleet.sort_values("sort_rul").drop(columns=["sort_rul"])


DARK_BG = "#0B1929"
CARD_BG = "#132030"
GRID_COLOR = "#1E3A52"
TEXT_COLOR = "#B0C4D8"


def make_fleet_zone_chart(fleet_df):
    chart_df = fleet_df[fleet_df["Status"].isin(["HEALTHY", "WARNING", "CRITICAL"])]
    if chart_df.empty:
        return go.Figure()
    counts = chart_df["Status"].value_counts()
    color_map = {"HEALTHY": "#0BC49E", "WARNING": "#F5A623", "CRITICAL": "#E8453C"}
    colors = [color_map.get(label, "#94A3B8") for label in counts.index]

    fig = go.Figure(go.Pie(
        labels=counts.index.tolist(),
        values=counts.values.tolist(),
        hole=0.55,
        marker=dict(colors=colors, line=dict(color=DARK_BG, width=2)),
        textinfo="percent+label",
        textfont=dict(color="white", size=12),
    ))
    fig.add_annotation(
        text=f"<b>{len(chart_df)}</b><br>Robots",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color="white"),
    )
    fig.update_layout(
        title=dict(text="Fleet Health Distribution", font=dict(color="white", size=14)),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font=dict(color=TEXT_COLOR),
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(font=dict(color=TEXT_COLOR)),
        height=320,
    )
    return fig


def make_rul_trend_chart(history, robot_id):
    fig = go.Figure()
    max_rul = max(float(history["predicted_rul"].max()), WARNING_THRESHOLD * 1.2)
    fig.add_hrect(y0=0, y1=CRITICAL_THRESHOLD, fillcolor="rgba(232,69,60,0.10)", line_width=0)
    fig.add_hrect(y0=CRITICAL_THRESHOLD, y1=WARNING_THRESHOLD, fillcolor="rgba(245,166,35,0.10)", line_width=0)
    fig.add_hrect(y0=WARNING_THRESHOLD, y1=max_rul, fillcolor="rgba(11,196,158,0.06)", line_width=0)

    colors = history["predicted_zone"].map(
        {"CRITICAL": "#E8453C", "WARNING": "#F5A623", "HEALTHY": "#0BC49E"}
    ).fillna("#0BC49E")

    fig.add_trace(go.Scatter(
        x=history["timestamp"],
        y=history["predicted_rul"],
        mode="lines+markers",
        line=dict(color="#0BC49E", width=2),
        marker=dict(color=colors, size=4),
        name="Predicted RUL",
    ))
    fig.add_hline(y=CRITICAL_THRESHOLD, line_dash="dot", line_color="#E8453C", annotation_text="Critical threshold")
    fig.add_hline(y=WARNING_THRESHOLD, line_dash="dot", line_color="#F5A623", annotation_text="Warning threshold")
    fig.update_layout(
        title=dict(text=f"Predicted RUL Over Time: {robot_id}", font=dict(color="white", size=14)),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_COLOR),
        xaxis=dict(gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, title="Predicted RUL (hours)"),
        margin=dict(l=60, r=40, t=50, b=40),
        showlegend=False,
        height=340,
    )
    return fig


def make_sensor_trend_chart(history):
    sensors = [
        ("vibration_level", "Vibration", "#0BC49E"),
        ("motor_temperature", "Temperature", "#F5A623"),
        ("torque_load", "Torque", "#7B5EA7"),
        ("power_consumption", "Power", "#94A3B8"),
    ]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[label for _, label, _ in sensors])
    for index, (col, label, color) in enumerate(sensors):
        fig.add_trace(
            go.Scatter(x=history["timestamp"], y=history[col], mode="lines", line=dict(color=color, width=1.5), showlegend=False),
            row=(index // 2) + 1,
            col=(index % 2) + 1,
        )
    fig.update_layout(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_COLOR),
        title=dict(text="Sensor Readings Over Time", font=dict(color="white", size=14)),
        margin=dict(l=40, r=20, t=60, b=40),
        height=360,
    )
    for index in range(1, 5):
        fig.update_xaxes(gridcolor=GRID_COLOR, row=(index - 1) // 2 + 1, col=(index - 1) % 2 + 1)
        fig.update_yaxes(gridcolor=GRID_COLOR, row=(index - 1) // 2 + 1, col=(index - 1) % 2 + 1)
    return fig


def make_feature_importance_chart():
    if not hasattr(model, "feature_importances_"):
        return go.Figure()
    importance_df = (
        pd.DataFrame({"feature": CONFIG_C, "importance": model.feature_importances_})
        .sort_values("importance", ascending=True)
        .tail(12)
    )
    fig = go.Figure(go.Bar(
        x=importance_df["importance"],
        y=importance_df["feature"],
        orientation="h",
        marker=dict(color="#0BC49E"),
    ))
    fig.update_layout(
        title=dict(text="Feature Importance (Top 12)", font=dict(color="white", size=14)),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_COLOR),
        xaxis=dict(gridcolor=GRID_COLOR, title="Importance"),
        yaxis=dict(gridcolor=GRID_COLOR),
        margin=dict(l=190, r=20, t=50, b=40),
        height=380,
    )
    return fig


def load_fleet_dashboard():
    fleet_df = build_fleet_predictions()
    n_total = len(fleet_df)
    n_healthy = int((fleet_df["Status"] == "HEALTHY").sum())
    n_warning = int((fleet_df["Status"] == "WARNING").sum())
    n_critical = int((fleet_df["Status"] == "CRITICAL").sum())

    summary_html = f"""
    <div class='metric-row'>
        <div class='metric-card teal'><div class='metric-label'>TOTAL ROBOTS</div><div class='metric-value'>{n_total}</div><div class='metric-note'>5 factories monitored</div></div>
        <div class='metric-card green'><div class='metric-label'>HEALTHY</div><div class='metric-value'>{n_healthy}</div><div class='metric-note'>RUL greater than 30 days</div></div>
        <div class='metric-card amber'><div class='metric-label'>WARNING</div><div class='metric-value'>{n_warning}</div><div class='metric-note'>RUL between 7 and 30 days</div></div>
        <div class='metric-card red'><div class='metric-label'>CRITICAL</div><div class='metric-value'>{n_critical}</div><div class='metric-note'>RUL less than 7 days</div></div>
        <div class='metric-card neutral'><div class='metric-label'>MODEL R²</div><div class='metric-value'>{REPORT_R2:.3f}</div><div class='metric-note'>Model fit, not accuracy</div></div>
    </div>
    <p class='advisory'>Advisory system only. Predictions support maintenance decisions but do not replace engineering judgement. Censored robots and WELDING-W5 robots carry higher uncertainty.</p>
    """

    display_cols = ["Robot ID", "Model Type", "Factory", "Status", "Predicted RUL (hrs)", "Days Remaining", "Temperature", "Vibration", "Note"]
    return summary_html, make_fleet_zone_chart(fleet_df), fleet_df[display_cols]


def load_robot_detail(robot_id):
    if not robot_id:
        return "Select a robot.", "", "", None, None, None

    latest_rul, history = predict_latest_for_robot(robot_id)
    if latest_rul is None or history is None:
        return ("<div class='status-card'>Insufficient history to compute rolling features for this robot.</div>", "No prediction available.", "", None, None, None)

    latest = history.iloc[-1]
    zone = get_zone_display(latest_rul)
    zone_color = get_zone_color(latest_rul)

    is_censored = not bool(latest.get("is_failure_robot", False))
    is_welding = "WELDING" in str(latest.get("model_type", "")).upper()

    warnings_list = []
    if latest_rul < CRITICAL_THRESHOLD:
        warnings_list.append(f"Critical zone uncertainty is approximately ±{REPORT_CRITICAL_MAE_HOURS:.0f} hours.")
    if is_censored:
        warnings_list.append("This robot has no confirmed failure event, so prediction uncertainty is higher.")
    if is_welding:
        warnings_list.append("WELDING-W5 has no confirmed failure examples in training.")

    warning_html = ""
    if warnings_list:
        warning_html = "<ul>" + "".join(f"<li>{item}</li>" for item in warnings_list) + "</ul>"

    status_html = f"""
    <div class='status-card' style='border-left: 4px solid {zone_color};'>
        <div style='color:{zone_color};font-size:22px;font-weight:800;'>{zone}</div>
        <div style='font-size:38px;font-weight:900;color:white;margin-top:6px;'>{latest_rul:.0f} <span style='font-size:15px;color:#94A3B8;'>hours remaining</span></div>
        <div style='color:#94A3B8;margin-top:6px;'>{latest_rul / 24:.1f} days · {latest['model_type']} · {latest['factory_location']}</div>
        <div class='advisory'>{warning_html}</div>
    </div>
    """

    sensor_markdown = f"""
### Latest Sensor Snapshot

| Signal | Current | 7-Day Mean |
|---|---:|---:|
| Vibration | {latest['vibration_level']:.3f} | {latest['vibration_rolling_mean_7d']:.3f} |
| Temperature | {latest['motor_temperature']:.1f} | {latest['temperature_rolling_mean_7d']:.1f} |
| Torque | {latest['torque_load']:.1f} | {latest['torque_rolling_mean_7d']:.1f} |
| Power | {latest['power_consumption']:.1f} | Not used in 7-day Config C mean |

### Operating Context

- Cumulative hours: {latest['cumulative_hours']:.0f}
- Maintenance count: {latest['total_maintenance_count']:.0f}
- Failure-history robot: {"Yes" if latest['is_failure_robot'] else "No, censored"}
"""

    return (
        status_html,
        get_recommendation(latest_rul, latest["model_type"]),
        sensor_markdown,
        make_rul_trend_chart(history, robot_id),
        make_sensor_trend_chart(history),
        make_feature_importance_chart(),
    )


def run_manual_prediction(
    vibration, temperature, torque, power, cumulative_hours,
    model_type, factory, environment,
    maint_count, total_downtime, avg_downtime,
    repair_count, replacement_count, lubrication_count, calibration_count, inspection_count,
):
    model_type_map = {v: i for i, v in enumerate(sorted(merged_df["model_type"].astype(str).unique()))}
    factory_map = {v: i for i, v in enumerate(sorted(merged_df["factory_location"].astype(str).unique()))}
    environment_map = {v: i for i, v in enumerate(sorted(merged_df["operating_environment"].astype(str).unique()))}

    cumulative_hours = float(np.clip(cumulative_hours, TRAINING_BOUNDS["cumulative_hours"]["lower"], TRAINING_BOUNDS["cumulative_hours"]["upper"]))
    maint_count = float(np.clip(maint_count, TRAINING_BOUNDS["total_maintenance_count"]["lower"], TRAINING_BOUNDS["total_maintenance_count"]["upper"]))
    vibration = float(np.clip(vibration, TRAINING_BOUNDS["vibration_level"]["lower"], TRAINING_BOUNDS["vibration_level"]["upper"]))

    rolling_means = featured_df[CORE_FEATURES_V2].mean(numeric_only=True)

    feature_vector = {
        "vibration_level": vibration,
        "motor_temperature": temperature,
        "torque_load": torque,
        "power_consumption": power,
        "cumulative_hours": cumulative_hours,
        "total_maintenance_count": maint_count,
        "total_downtime_hours": total_downtime,
        "avg_downtime_hours": avg_downtime,
        "repair_count": repair_count,
        "replacement_count": replacement_count,
        "lubrication_count": lubrication_count,
        "calibration_count": calibration_count,
        "inspection_count": inspection_count,
        "model_type_encoded": model_type_map.get(model_type, 0),
        "factory_location_encoded": factory_map.get(factory, 0),
        "operating_environment_encoded": environment_map.get(environment, 0),
        "load_stress_index": torque * cumulative_hours,
    }

    for feature in CORE_FEATURES_V2:
        if feature not in feature_vector:
            feature_vector[feature] = float(rolling_means.get(feature, 0))

    x = pd.DataFrame([feature_vector])[CONFIG_C].values
    rul = float(np.clip(model.predict(scaler.transform(x))[0], 0, None))
    zone_color = get_zone_color(rul)

    result_html = f"""
    <div class='status-card' style='border-left: 4px solid {zone_color};'>
        <div style='color:{zone_color};font-size:22px;font-weight:800;'>{get_zone_display(rul)}</div>
        <div style='font-size:38px;font-weight:900;color:white;margin-top:6px;'>{rul:.0f} <span style='font-size:15px;color:#94A3B8;'>hours remaining</span></div>
        <div style='color:#94A3B8;margin-top:6px;'>{rul / 24:.1f} days · {model_type} · {factory}</div>
        <p class='advisory'>Manual mode is approximate because rolling features require historical sensor data. Fleet-average rolling features were used.</p>
    </div>
    """
    return result_html, get_recommendation(rul, model_type)


custom_css = """
body, .gradio-container {
    background-color: #0B1929 !important;
    color: #E2E8F0 !important;
    font-family: Inter, Segoe UI, sans-serif !important;
}
.block, .panel, .form {
    background: #132030 !important;
    border: 1px solid #1E3A52 !important;
    border-radius: 10px !important;
}
input, select, textarea {
    background: #0B1929 !important;
    color: white !important;
    border-color: #1E3A52 !important;
}
h1, h2, h3, h4 { color: white !important; }
.metric-row { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 10px; }
.metric-card { background: #132030; border: 1px solid #1E3A52; border-radius: 10px; padding: 16px 22px; flex: 1; min-width: 140px; }
.metric-card.teal { border-top: 3px solid #0BC49E; }
.metric-card.green { border-top: 3px solid #4ADE80; }
.metric-card.amber { border-top: 3px solid #F5A623; }
.metric-card.red { border-top: 3px solid #E8453C; }
.metric-card.neutral { border-top: 3px solid #94A3B8; }
.metric-label { color: #94A3B8; font-size: 11px; letter-spacing: 2px; font-weight: 700; }
.metric-value { color: white; font-size: 34px; font-weight: 900; margin-top: 4px; }
.metric-note { color: #64748B; font-size: 12px; }
.status-card { background: #132030; border: 1px solid #1E3A52; border-radius: 10px; padding: 20px; }
.advisory { color: #94A3B8; font-size: 12px; margin-top: 8px; }
"""

all_robot_ids = sorted(merged_df["robot_id"].unique().tolist())

with gr.Blocks(
    title="Kataoka RUL Predictive Maintenance",
    theme=gr.themes.Base(primary_hue="teal", neutral_hue="slate"),
    css=custom_css,
) as app:
    gr.HTML("""
    <div style='background:#112233;border-bottom:1px solid #1E3A52;padding:18px 24px;margin-bottom:10px;'>
        <h1 style='margin:0;color:white;font-size:24px;font-weight:800;'>Kataoka Inc. Predictive Maintenance System</h1>
        <p style='margin:4px 0 0;color:#94A3B8;font-size:13px;'>Remaining Useful Life estimation · Random Forest Config C · 50 robots · Advisory decision support</p>
    </div>
    """)

    with gr.Tabs():
        with gr.TabItem("Fleet Dashboard"):
            fleet_summary_html = gr.HTML()
            fleet_zone_chart = gr.Plot()
            gr.Markdown("### Robot Registry")
            fleet_table = gr.Dataframe(interactive=False, wrap=True)
            refresh_btn = gr.Button("Refresh Fleet Data", variant="primary")
            refresh_btn.click(load_fleet_dashboard, inputs=[], outputs=[fleet_summary_html, fleet_zone_chart, fleet_table])
            app.load(load_fleet_dashboard, inputs=[], outputs=[fleet_summary_html, fleet_zone_chart, fleet_table])

        with gr.TabItem("Robot Detail"):
            robot_dropdown = gr.Dropdown(choices=all_robot_ids, value=all_robot_ids[0] if all_robot_ids else None, label="Select Robot")
            load_robot_btn = gr.Button("Load Robot Data", variant="primary")
            status_html = gr.HTML()
            recommendation_box = gr.Textbox(label="Maintenance Recommendation", lines=5, interactive=False)
            sensor_stats = gr.Markdown()
            with gr.Row():
                rul_chart = gr.Plot(label="Predicted RUL")
                sensor_chart = gr.Plot(label="Sensor Trends")
            importance_chart = gr.Plot(label="Feature Importance")
            robot_outputs = [status_html, recommendation_box, sensor_stats, rul_chart, sensor_chart, importance_chart]
            load_robot_btn.click(load_robot_detail, inputs=[robot_dropdown], outputs=robot_outputs)
            robot_dropdown.change(load_robot_detail, inputs=[robot_dropdown], outputs=robot_outputs)

        with gr.TabItem("Manual Prediction"):
            gr.Markdown("Manual mode is approximate because rolling features require robot history. The app fills those rolling features using fleet averages from the engineered dataset. Use Robot Detail mode for history-based predictions.")
            with gr.Row():
                with gr.Column():
                    vibration_in = gr.Slider(0.0, 1.0, value=0.35, step=0.001, label="Vibration Level")
                    temperature_in = gr.Slider(35.0, 90.0, value=58.0, step=0.1, label="Motor Temperature")
                    torque_in = gr.Slider(40.0, 185.0, value=105.0, step=0.5, label="Torque Load")
                    power_in = gr.Slider(400.0, 2400.0, value=1300.0, step=10.0, label="Power Consumption")
                    cumulative_hours_in = gr.Slider(0, 8136, value=3000, step=100, label="Cumulative Hours")
                    model_choices = sorted(merged_df["model_type"].astype(str).unique().tolist())
                    factory_choices = sorted(merged_df["factory_location"].astype(str).unique().tolist())
                    env_choices = sorted(merged_df["operating_environment"].astype(str).unique().tolist())
                    model_type_in = gr.Dropdown(model_choices, value=model_choices[0], label="Model Type")
                    factory_in = gr.Dropdown(factory_choices, value=factory_choices[0], label="Factory Location")
                    environment_in = gr.Dropdown(env_choices, value=env_choices[0], label="Operating Environment")

                with gr.Column():
                    maint_count_in = gr.Slider(0, 7, value=4, step=1, label="Total Maintenance Count")
                    total_downtime_in = gr.Slider(0, 500, value=50, step=5, label="Total Downtime Hours")
                    avg_downtime_in = gr.Slider(0, 100, value=10, step=1, label="Average Downtime Hours")
                    repair_count_in = gr.Slider(0, 20, value=2, step=1, label="Repair Count")
                    replacement_count_in = gr.Slider(0, 10, value=1, step=1, label="Replacement Count")
                    lubrication_count_in = gr.Slider(0, 20, value=3, step=1, label="Lubrication Count")
                    calibration_count_in = gr.Slider(0, 20, value=2, step=1, label="Calibration Count")
                    inspection_count_in = gr.Slider(0, 20, value=3, step=1, label="Inspection Count")
                    manual_btn = gr.Button("Run Manual Prediction", variant="primary")
                    manual_result = gr.HTML()
                    manual_recommendation = gr.Textbox(label="Recommendation", lines=5, interactive=False)

            manual_btn.click(
                run_manual_prediction,
                inputs=[
                    vibration_in, temperature_in, torque_in, power_in, cumulative_hours_in,
                    model_type_in, factory_in, environment_in,
                    maint_count_in, total_downtime_in, avg_downtime_in,
                    repair_count_in, replacement_count_in, lubrication_count_in,
                    calibration_count_in, inspection_count_in,
                ],
                outputs=[manual_result, manual_recommendation],
            )

        with gr.TabItem("About"):
            gr.Markdown(f"""
## About the System

This application demonstrates the final Random Forest Config C model selected after baseline, engineered modelling, and tuning experiments.

| Metric | Value |
|---|---:|
| Overall MAE | {REPORT_OVERALL_MAE_HOURS:.1f} hours ({REPORT_OVERALL_MAE_HOURS / 24:.1f} days) |
| Model R² | {REPORT_R2:.3f} |
| Critical Zone MAE | {REPORT_CRITICAL_MAE_HOURS:.1f} hours |
| Minimum Lead Time | {REPORT_MIN_LEAD_HOURS:.0f} hours ({REPORT_MIN_LEAD_HOURS / 24:.1f} days) |
| Missed Detections | 0 of 4 held-out failure robots |

### Important Limitations

- R² is model fit, not accuracy.
- Predictions for censored robots are risk estimates, not confirmed truth.
- Manual mode is approximate because rolling features require historical data.
- WELDING-W5 predictions are extrapolations because that model type has no confirmed failure history in training.
- Critical predictions should trigger inspection, not automatic maintenance.
""")

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
