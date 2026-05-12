# =============================================================================
# app.py — Kataoka Inc. Predictive Maintenance System
# Phase 7: Gradio Application
# Model: Random Forest Config C (28 features)
# =============================================================================

import gradio as gr
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder

# =============================================================================
# SECTION 1: LOAD MODEL AND DATA
# =============================================================================

MODEL_PATH  = "models/tuned/rul_regressor_final.pkl"
SCALER_PATH = "models/tuned/scaler_final.pkl"

model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

merged_df = pd.read_csv(
    "data/processed/merged_dataset.csv",
    parse_dates=["timestamp", "failure_time"]
)

featured_df = pd.read_csv(
    "data/processed/featured_dataset.csv",
    parse_dates=["timestamp", "failure_time"]
)

# =============================================================================
# SECTION 2: FEATURE DEFINITIONS
# =============================================================================

BASELINE_B_FEATURES = [
    "vibration_level", "motor_temperature", "torque_load", "power_consumption",
    "cumulative_hours", "total_maintenance_count", "total_downtime_hours",
    "avg_downtime_hours", "repair_count", "replacement_count",
    "lubrication_count", "calibration_count", "inspection_count",
    "model_type_encoded", "factory_location_encoded",
    "operating_environment_encoded",
]

CORE_FEATURES_V2 = [
    "load_stress_index",
    "vibration_rolling_std_7d", "temperature_rolling_std_7d",
    "torque_rolling_std_7d",
    "vibration_rolling_std_24h", "temperature_rolling_std_24h",
    "torque_rolling_std_24h",
    "vibration_rolling_mean_7d", "temperature_rolling_mean_7d",
    "torque_rolling_mean_7d",
    "vibration_short_long_diff", "temperature_short_long_diff",
]

CONFIG_C = BASELINE_B_FEATURES + CORE_FEATURES_V2  # 28 features

for col in ["model_type", "factory_location", "operating_environment"]:
    enc_col = col + "_encoded"
    if enc_col not in merged_df.columns:
        le = LabelEncoder()
        merged_df[enc_col] = le.fit_transform(merged_df[col])

for col in ["model_type", "factory_location", "operating_environment"]:
    enc_col = col + "_encoded"
    if enc_col not in featured_df.columns:
        le = LabelEncoder()
        featured_df[enc_col] = le.fit_transform(featured_df[col])

# =============================================================================
# SECTION 3: TRAINING BOUNDS
# Sourced directly from scaler diagnostics (verified against featured_dataset.csv).
# Three features are out-of-range for censored robots:
#   cumulative_hours:         training max 8,136  censored max 12,960
#   total_maintenance_count:  training max 5      censored max 7
#   vibration_level:          training min 0.12   censored min 0.08
# Clipping to these bounds prevents the StandardScaler from producing
# extreme normalised values that corrupt model predictions.
# load_stress_index is recomputed after clipping cumulative_hours
# because it is derived as torque_load * cumulative_hours.
# =============================================================================

TRAINING_BOUNDS = {
    "cumulative_hours"        : {"lower": 162.0, "upper": 8136.0},
    "total_maintenance_count" : {"lower": 3.0,   "upper": 5.0},
    "vibration_level"         : {"lower": 0.12,  "upper": 0.68},
}

# =============================================================================
# SECTION 4: HELPER FUNCTIONS
# =============================================================================

def get_zone(rul):
    if rul < 168:   return "🔴 CRITICAL"
    elif rul < 720: return "🟡 WARNING"
    else:           return "🟢 HEALTHY"

def get_zone_color(rul):
    if rul < 168:   return "#E8453C"
    elif rul < 720: return "#F5A623"
    else:           return "#0BC49E"

def get_recommendation(rul, robot_id):
    is_w5    = "W5" in str(robot_id) or "WELDING" in str(robot_id).upper()
    w5_note  = "\n⚠️ WELDING-W5: No failure history in training data. Treat as guidance only." if is_w5 else ""
    if rul < 168:
        return (f"🔴 CRITICAL — Immediate inspection required.\n"
                f"Predicted RUL: {rul:.0f} hours ({rul/24:.1f} days).\n"
                f"⚠️ Uncertainty: ±130 hours in Critical zone. "
                f"Treat this as a trigger to inspect, not a precise countdown.\n"
                f"Schedule maintenance as soon as operationally possible.{w5_note}")
    elif rul < 720:
        return (f"🟡 WARNING — Schedule maintenance within the next maintenance window.\n"
                f"Predicted RUL: {rul:.0f} hours ({rul/24:.1f} days).\n"
                f"Monitor sensor trends closely. "
                f"Plan downtime before RUL reaches Critical zone.{w5_note}")
    else:
        return (f"🟢 HEALTHY — Continue routine monitoring.\n"
                f"Predicted RUL: {rul:.0f} hours ({rul/24:.1f} days).\n"
                f"No immediate action required. "
                f"Review at next scheduled maintenance interval.{w5_note}")


def compute_rolling_features(df, robot_id):
    """
    Compute all 12 CORE_FEATURES_V2 engineered features for a given robot.

    After computing features, applies training-bounds clipping to three
    features that fall outside the training distribution for censored robots:
      - cumulative_hours (censored robots: up to 12,960 vs training max 8,136)
      - total_maintenance_count (censored: up to 7 vs training max 5)
      - vibration_level (censored: as low as 0.08 vs training min 0.12)

    load_stress_index is recomputed after clipping cumulative_hours because
    it is derived as torque_load * cumulative_hours.

    Returns the full history dataframe with all features computed and clipped.
    """
    robot_data = df[df["robot_id"] == robot_id].copy()
    robot_data = robot_data.sort_values("timestamp").reset_index(drop=True)

    WINDOW = 28
    SHORT  = 4

    sensors_map = {
        "vibration_level"   : "vibration",
        "motor_temperature" : "temperature",
        "torque_load"       : "torque",
        "power_consumption" : "power",
    }

    for sensor, short in sensors_map.items():
        robot_data[f"{short}_rolling_mean_7d"] = (
            robot_data[sensor].rolling(WINDOW, min_periods=WINDOW).mean()
        )
        robot_data[f"{short}_rolling_std_7d"] = (
            robot_data[sensor].rolling(WINDOW, min_periods=WINDOW).std()
        )

    for sensor, short in [
        ("vibration_level","vibration"),
        ("motor_temperature","temperature"),
        ("torque_load","torque"),
    ]:
        robot_data[f"{short}_rolling_std_24h"] = (
            robot_data[sensor].rolling(SHORT, min_periods=SHORT).std()
        )

    robot_data["load_stress_index"] = (
        robot_data["torque_load"] * robot_data["cumulative_hours"]
    )

    v_short = robot_data["vibration_level"].rolling(SHORT, min_periods=SHORT).mean()
    t_short = robot_data["motor_temperature"].rolling(SHORT, min_periods=SHORT).mean()
    robot_data["vibration_short_long_diff"]   = v_short - robot_data["vibration_rolling_mean_7d"]
    robot_data["temperature_short_long_diff"] = t_short - robot_data["temperature_rolling_mean_7d"]

    clean = robot_data.dropna(subset=CORE_FEATURES_V2).copy()
    if len(clean) == 0:
        return None

    # ── Clip to training bounds ───────────────────────────────────────────────
    clean["cumulative_hours"] = clean["cumulative_hours"].clip(
        lower=TRAINING_BOUNDS["cumulative_hours"]["lower"],
        upper=TRAINING_BOUNDS["cumulative_hours"]["upper"],
    )
    clean["total_maintenance_count"] = clean["total_maintenance_count"].clip(
        lower=TRAINING_BOUNDS["total_maintenance_count"]["lower"],
        upper=TRAINING_BOUNDS["total_maintenance_count"]["upper"],
    )
    clean["vibration_level"] = clean["vibration_level"].clip(
        lower=TRAINING_BOUNDS["vibration_level"]["lower"],
        upper=TRAINING_BOUNDS["vibration_level"]["upper"],
    )
    # Recompute load_stress_index after clipping cumulative_hours
    clean["load_stress_index"] = clean["torque_load"] * clean["cumulative_hours"]

    return clean


def predict_for_robot(robot_id):
    robot_data = compute_rolling_features(merged_df, robot_id)
    if robot_data is None or len(robot_data) == 0:
        return None, None, None

    X            = robot_data[CONFIG_C].values
    X_scaled     = scaler.transform(X)
    predictions  = np.clip(model.predict(X_scaled), 0, None)

    robot_data              = robot_data.copy()
    robot_data["predicted_rul"]  = predictions
    robot_data["predicted_zone"] = pd.cut(
        robot_data["predicted_rul"],
        bins=[-1, 168, 720, float("inf")],
        labels=["Critical", "Warning", "Healthy"]
    )

    return predictions[-1], get_zone(predictions[-1]), robot_data


def build_fleet_predictions():
    results = []
    for robot_id in sorted(merged_df["robot_id"].unique()):
        robot_meta = merged_df[merged_df["robot_id"] == robot_id].iloc[0]
        rul, zone, _ = predict_for_robot(robot_id)

        if rul is None:
            rul  = 5000
            zone = "🟢 HEALTHY"

        latest     = merged_df[merged_df["robot_id"] == robot_id].sort_values("timestamp").iloc[-1]
        zone_clean = zone.replace("🔴 ", "").replace("🟡 ", "").replace("🟢 ", "")
        is_w5      = "WELDING" in str(robot_meta["model_type"]).upper()
        is_failure = bool(robot_meta["is_failure_robot"])

        results.append({
            "Robot ID"           : robot_id,
            "Model Type"         : robot_meta["model_type"],
            "Factory"            : robot_meta["factory_location"],
            "Status"             : zone_clean,
            "Predicted RUL (hrs)": round(rul),
            "Days Remaining"     : round(rul / 24, 1),
            "Vibration"          : round(latest["vibration_level"], 3),
            "Temperature"        : round(latest["motor_temperature"], 1),
            "Cum. Hours"         : round(latest["cumulative_hours"]),
            "Note"               : (
                "⚠️ Extrapolation" if is_w5
                else ("" if is_failure else "No failure baseline")
            ),
        })

    return pd.DataFrame(results).sort_values("Predicted RUL (hrs)")


# =============================================================================
# SECTION 5: CHART BUILDERS
# =============================================================================

DARK_BG    = "#0B1929"
CARD_BG    = "#132030"
GRID_COLOR = "#1E3A52"
TEXT_COLOR = "#94A3B8"

def make_rul_trend_chart(robot_data, robot_id):
    fig = go.Figure()
    fig.add_hrect(y0=0,   y1=168, fillcolor="rgba(232,69,60,0.08)",  line_width=0, annotation_text="Critical", annotation_position="left")
    fig.add_hrect(y0=168, y1=720, fillcolor="rgba(245,166,35,0.08)", line_width=0, annotation_text="Warning",  annotation_position="left")
    fig.add_hrect(y0=720, y1=robot_data["predicted_rul"].max()*1.05, fillcolor="rgba(11,196,158,0.05)", line_width=0, annotation_text="Healthy", annotation_position="left")

    colors = robot_data["predicted_zone"].map({"Critical":"#E8453C","Warning":"#F5A623","Healthy":"#0BC49E"}).fillna("#0BC49E")
    fig.add_trace(go.Scatter(
        x=robot_data["timestamp"], y=robot_data["predicted_rul"],
        mode="lines+markers", line=dict(color="#0BC49E", width=2),
        marker=dict(color=colors, size=4), name="Predicted RUL",
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Predicted RUL: %{y:.0f} hrs<extra></extra>",
    ))
    fig.add_hline(y=168, line_dash="dot", line_color="#E8453C", opacity=0.6, annotation_text="Critical (168h)", annotation_font_color="#E8453C")
    fig.add_hline(y=720, line_dash="dot", line_color="#F5A623", opacity=0.6, annotation_text="Warning (720h)",  annotation_font_color="#F5A623")
    fig.update_layout(
        title=dict(text=f"Predicted RUL Over Time — {robot_id}", font=dict(color="white", size=14)),
        paper_bgcolor=DARK_BG, plot_bgcolor=CARD_BG, font=dict(color=TEXT_COLOR),
        xaxis=dict(gridcolor=GRID_COLOR), yaxis=dict(gridcolor=GRID_COLOR, title="Predicted RUL (hours)"),
        margin=dict(l=60,r=40,t=50,b=40), showlegend=False, height=320,
    )
    return fig


def make_fleet_zone_chart(fleet_df):
    counts    = fleet_df["Status"].value_counts()
    color_map = {"HEALTHY":"#0BC49E","WARNING":"#F5A623","CRITICAL":"#E8453C"}
    colors    = [color_map.get(l.upper(), "#94A3B8") for l in counts.index]
    fig = go.Figure(go.Pie(
        labels=counts.index.tolist(), values=counts.values.tolist(),
        hole=0.55, marker=dict(colors=colors, line=dict(color=DARK_BG, width=2)),
        textinfo="percent+label", textfont=dict(color="white", size=12),
        hovertemplate="<b>%{label}</b><br>%{value} robots (%{percent})<extra></extra>",
    ))
    fig.add_annotation(text=f"<b>{len(fleet_df)}</b><br>Robots", x=0.5, y=0.5,
                       showarrow=False, font=dict(size=16, color="white"))
    fig.update_layout(
        title=dict(text="Fleet Health Distribution", font=dict(color="white", size=14)),
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG, font=dict(color=TEXT_COLOR),
        margin=dict(l=20,r=20,t=50,b=20), showlegend=True,
        legend=dict(font=dict(color=TEXT_COLOR)), height=300,
    )
    return fig


def make_sensor_trend_chart(robot_data):
    sensors = [
        ("vibration_level",   "Vibration (mm/s)", "#0BC49E"),
        ("motor_temperature", "Temperature (°C)",  "#F5A623"),
        ("torque_load",       "Torque (Nm)",        "#7B5EA7"),
        ("power_consumption", "Power (W)",          "#94A3B8"),
    ]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[s[1] for s in sensors],
                        vertical_spacing=0.15, horizontal_spacing=0.1)
    for i, (col, label, color) in enumerate(sensors):
        fig.add_trace(go.Scatter(
            x=robot_data["timestamp"], y=robot_data[col],
            mode="lines", line=dict(color=color, width=1.5),
            showlegend=False,
            hovertemplate=f"<b>%{{x|%Y-%m-%d}}</b><br>{label}: %{{y:.3f}}<extra></extra>",
        ), row=(i//2)+1, col=(i%2)+1)
    fig.update_layout(paper_bgcolor=DARK_BG, plot_bgcolor=CARD_BG, font=dict(color=TEXT_COLOR),
                      title=dict(text="Sensor Readings Over Time", font=dict(color="white", size=14)),
                      margin=dict(l=40,r=20,t=60,b=40), height=360)
    for i in range(1,5):
        fig.update_xaxes(gridcolor=GRID_COLOR, row=(i-1)//2+1, col=(i-1)%2+1)
        fig.update_yaxes(gridcolor=GRID_COLOR, row=(i-1)//2+1, col=(i-1)%2+1)
    return fig


def make_feature_importance_chart():
    imp_df = pd.DataFrame({"feature": CONFIG_C, "importance": model.feature_importances_}
                          ).sort_values("importance", ascending=True).tail(12)
    colors = ["#E8453C" if "24h" in f
              else "#0BC49E" if any(x in f for x in ["std","stress","diff","slope"])
              else "#94A3B8" for f in imp_df["feature"]]
    fig = go.Figure(go.Bar(x=imp_df["importance"], y=imp_df["feature"],
                           orientation="h", marker=dict(color=colors),
                           hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"))
    fig.update_layout(
        title=dict(text="Feature Importance (Top 12)", font=dict(color="white", size=14)),
        paper_bgcolor=DARK_BG, plot_bgcolor=CARD_BG, font=dict(color=TEXT_COLOR),
        xaxis=dict(gridcolor=GRID_COLOR, title="Importance"),
        yaxis=dict(gridcolor=GRID_COLOR),
        margin=dict(l=180,r=20,t=50,b=40), height=380, showlegend=False,
    )
    return fig


# =============================================================================
# SECTION 6: GRADIO CALLBACK FUNCTIONS
# =============================================================================

def load_robot_detail(robot_id):
    if not robot_id:
        return [None] * 6

    rul, zone, history = predict_for_robot(robot_id)
    if rul is None:
        return ("⚠️ Insufficient data to compute features for this robot.",
                None, None, None, None, None)

    latest      = merged_df[merged_df["robot_id"] == robot_id].sort_values("timestamp").iloc[-1]
    is_w5       = "WELDING" in str(latest["model_type"]).upper()
    is_censored = not bool(latest["is_failure_robot"])
    zone_color  = get_zone_color(rul)
    days        = rul / 24

    uncertainty_notice = (
        "<p style='color:#F5A623;font-size:13px;margin-top:8px;'>"
        "⚠️ Critical zone uncertainty: ±130 hours. "
        "Treat as inspection trigger, not a precise countdown.</p>"
    ) if rul < 168 else ""

    w5_notice = (
        "<p style='color:#F5A623;font-size:13px;margin-top:8px;'>"
        "⚠️ WELDING-W5: No failure history in training data. "
        "Predictions are extrapolations — weight inspection findings more heavily.</p>"
    ) if is_w5 else ""

    censored_notice = (
        "<p style='color:#94A3B8;font-size:12px;margin-top:8px;'>"
        "ℹ️ This robot has no confirmed failure history. Features were clipped "
        "to training bounds before prediction. Treat with higher uncertainty.</p>"
    ) if is_censored else ""

    status_html = f"""
    <div style='background:#132030;border:1px solid {zone_color};border-radius:10px;
                padding:20px;border-left:4px solid {zone_color};'>
        <p style='color:{zone_color};font-size:22px;font-weight:bold;margin:0;'>{zone}</p>
        <p style='color:white;font-size:36px;font-weight:bold;margin:6px 0;'>
            {rul:.0f} <span style='font-size:16px;color:#94A3B8;'>hours remaining</span>
        </p>
        <p style='color:#94A3B8;font-size:14px;margin:0;'>
            {days:.1f} days · {latest['model_type']} · {latest['factory_location']}
        </p>
        {uncertainty_notice}{w5_notice}{censored_notice}
        <p style='color:#64748B;font-size:11px;margin-top:10px;'>
            ⓘ Advisory — Human decision required. No automated action will be taken.
        </p>
    </div>
    """

    sensor_stats = f"""
**Latest Sensor Readings**

| Sensor | Current Value | 7-Day Mean |
|---|---|---|
| Vibration | {latest['vibration_level']:.3f} mm/s | {history['vibration_rolling_mean_7d'].iloc[-1]:.3f} |
| Temperature | {latest['motor_temperature']:.1f} °C | {history['temperature_rolling_mean_7d'].iloc[-1]:.1f} |
| Torque | {latest['torque_load']:.1f} Nm | {history['torque_rolling_mean_7d'].iloc[-1]:.1f} |
| Power | {latest['power_consumption']:.1f} W | — |

**Operating Info**
- Cumulative hours: {latest['cumulative_hours']:.0f}
- Total maintenance events: {latest['total_maintenance_count']:.0f}
- Failure robot: {'Yes' if latest['is_failure_robot'] else 'No (no failure baseline)'}
    """

    return (
        status_html,
        get_recommendation(rul, robot_id),
        sensor_stats,
        make_rul_trend_chart(history, robot_id),
        make_sensor_trend_chart(merged_df[merged_df["robot_id"]==robot_id].sort_values("timestamp")),
        make_feature_importance_chart(),
    )


def load_fleet_dashboard():
    fleet_df   = build_fleet_predictions()
    zone_chart = make_fleet_zone_chart(fleet_df)

    n_critical = (fleet_df["Status"] == "CRITICAL").sum()
    n_warning  = (fleet_df["Status"] == "WARNING").sum()
    n_healthy  = (fleet_df["Status"] == "HEALTHY").sum()

    summary_html = f"""
    <div style='display:flex;gap:16px;flex-wrap:wrap;margin-bottom:8px;'>
        <div style='background:#132030;border:1px solid #0BC49E;border-radius:8px;padding:16px 24px;border-top:3px solid #0BC49E;flex:1;min-width:120px;'>
            <p style='color:#94A3B8;font-size:11px;letter-spacing:2px;margin:0;'>TOTAL ROBOTS</p>
            <p style='color:#0BC49E;font-size:36px;font-weight:bold;margin:4px 0;'>50</p>
            <p style='color:#64748B;font-size:12px;margin:0;'>5 factories monitored</p>
        </div>
        <div style='background:#132030;border:1px solid #4ADE80;border-radius:8px;padding:16px 24px;border-top:3px solid #4ADE80;flex:1;min-width:120px;'>
            <p style='color:#94A3B8;font-size:11px;letter-spacing:2px;margin:0;'>HEALTHY</p>
            <p style='color:#4ADE80;font-size:36px;font-weight:bold;margin:4px 0;'>{n_healthy}</p>
            <p style='color:#64748B;font-size:12px;margin:0;'>No action required</p>
        </div>
        <div style='background:#132030;border:1px solid #F5A623;border-radius:8px;padding:16px 24px;border-top:3px solid #F5A623;flex:1;min-width:120px;'>
            <p style='color:#94A3B8;font-size:11px;letter-spacing:2px;margin:0;'>WARNING</p>
            <p style='color:#F5A623;font-size:36px;font-weight:bold;margin:4px 0;'>{n_warning}</p>
            <p style='color:#64748B;font-size:12px;margin:0;'>Schedule maintenance</p>
        </div>
        <div style='background:#132030;border:1px solid #E8453C;border-radius:8px;padding:16px 24px;border-top:3px solid #E8453C;flex:1;min-width:120px;'>
            <p style='color:#94A3B8;font-size:11px;letter-spacing:2px;margin:0;'>CRITICAL</p>
            <p style='color:#E8453C;font-size:36px;font-weight:bold;margin:4px 0;'>{n_critical}</p>
            <p style='color:#64748B;font-size:12px;margin:0;'>Immediate inspection</p>
        </div>
        <div style='background:#132030;border:1px solid #94A3B8;border-radius:8px;padding:16px 24px;border-top:3px solid #94A3B8;flex:1;min-width:120px;'>
            <p style='color:#94A3B8;font-size:11px;letter-spacing:2px;margin:0;'>MODEL ACCURACY</p>
            <p style='color:white;font-size:36px;font-weight:bold;margin:4px 0;'>86.2%</p>
            <p style='color:#64748B;font-size:12px;margin:0;'>R² = 0.862</p>
        </div>
    </div>
    <p style='color:#64748B;font-size:11px;margin-top:4px;'>
        ⓘ Advisory system — All predictions are for informational purposes only.
        Human decision required before any maintenance action.
        Robots without failure history carry higher prediction uncertainty.
    </p>
    """

    display_df = fleet_df[[
        "Robot ID","Model Type","Factory","Status",
        "Predicted RUL (hrs)","Days Remaining","Temperature","Vibration","Note"
    ]].copy()

    return summary_html, zone_chart, display_df


def run_manual_prediction(
    vibration, temperature, torque, power, cumulative_hours,
    model_type, factory, environment,
    maint_count, downtime_hours, avg_downtime,
    repair_count, replacement_count, lubrication_count,
    calibration_count, inspection_count,
):
    model_type_map  = {"ARM-X7":0,"ARM-X9":1,"ASSEMBLY-A3":2,"CONVEYOR-C2":3,"WELDING-W5":4}
    factory_map     = {"Fukuoka Center":0,"Hokkaido Lab":1,"Nagoya Site":2,"Osaka Plant":3,"Tokyo Factory":4}
    environment_map = {"Heavy Industrial":0,"Standard Industrial":1}

    # Apply training-bounds clipping to manual inputs
    cumulative_hours = float(np.clip(cumulative_hours,
                             TRAINING_BOUNDS["cumulative_hours"]["lower"],
                             TRAINING_BOUNDS["cumulative_hours"]["upper"]))
    maint_count      = float(np.clip(maint_count,
                             TRAINING_BOUNDS["total_maintenance_count"]["lower"],
                             TRAINING_BOUNDS["total_maintenance_count"]["upper"]))
    vibration        = float(np.clip(vibration,
                             TRAINING_BOUNDS["vibration_level"]["lower"],
                             TRAINING_BOUNDS["vibration_level"]["upper"]))

    rolling_means = featured_df[CORE_FEATURES_V2].mean()
    load_stress   = torque * cumulative_hours

    feature_vector = [
        vibration, temperature, torque, power,
        cumulative_hours, maint_count, downtime_hours, avg_downtime,
        repair_count, replacement_count, lubrication_count,
        calibration_count, inspection_count,
        model_type_map.get(model_type, 0),
        factory_map.get(factory, 0),
        environment_map.get(environment, 0),
        load_stress,
        rolling_means["vibration_rolling_std_7d"],
        rolling_means["temperature_rolling_std_7d"],
        rolling_means["torque_rolling_std_7d"],
        rolling_means["vibration_rolling_std_24h"],
        rolling_means["temperature_rolling_std_24h"],
        rolling_means["torque_rolling_std_24h"],
        rolling_means["vibration_rolling_mean_7d"],
        rolling_means["temperature_rolling_mean_7d"],
        rolling_means["torque_rolling_mean_7d"],
        0.0, 0.0,
    ]

    X        = np.array(feature_vector).reshape(1, -1)
    X_scaled = scaler.transform(X)
    rul      = float(np.clip(model.predict(X_scaled)[0], 0, None))

    zone       = get_zone(rul)
    zone_color = get_zone_color(rul)
    is_w5      = model_type == "WELDING-W5"

    result_html = f"""
    <div style='background:#132030;border:1px solid {zone_color};border-radius:10px;
                padding:20px;border-left:4px solid {zone_color};'>
        <p style='color:{zone_color};font-size:22px;font-weight:bold;margin:0;'>{zone}</p>
        <p style='color:white;font-size:40px;font-weight:bold;margin:6px 0;'>
            {rul:.0f} <span style='font-size:16px;color:#94A3B8;'>hours remaining</span>
        </p>
        <p style='color:#94A3B8;font-size:14px;margin:0;'>{rul/24:.1f} days · {model_type} · {factory}</p>
        {"<p style='color:#F5A623;font-size:13px;margin-top:8px;'>⚠️ Critical zone uncertainty: ±130 hours.</p>" if rul < 168 else ""}
        {"<p style='color:#F5A623;font-size:13px;margin-top:8px;'>⚠️ WELDING-W5: No failure history. Extrapolation only.</p>" if is_w5 else ""}
        <p style='color:#64748B;font-size:12px;margin-top:8px;'>ℹ️ Manual input mode: Rolling features estimated from fleet averages.</p>
        <p style='color:#64748B;font-size:11px;margin-top:6px;'>ⓘ Advisory — Human decision required.</p>
    </div>
    """
    return result_html, get_recommendation(rul, model_type)


# =============================================================================
# SECTION 7: GRADIO UI
# =============================================================================

custom_css = """
body, .gradio-container {
    background-color: #0B1929 !important;
    color: #E2E8F0 !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}
.tab-nav button {
    background: #132030 !important;
    color: #94A3B8 !important;
    border: 1px solid #1E3A52 !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
}
.tab-nav button.selected {
    background: #0D9B7F !important;
    color: white !important;
    border-color: #0D9B7F !important;
}
.block, .panel, .form {
    background: #132030 !important;
    border: 1px solid #1E3A52 !important;
    border-radius: 8px !important;
}
label, .label-wrap span {
    color: #94A3B8 !important;
    font-size: 12px !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}
input, select, textarea {
    background: #0B1929 !important;
    color: white !important;
    border-color: #1E3A52 !important;
}
.markdown-body, p, span { color: #B0C4D8 !important; }
h1, h2, h3 { color: white !important; }
"""

all_robot_ids = sorted(merged_df["robot_id"].unique().tolist())

with gr.Blocks(title="Kataoka RUL Predictive Maintenance") as app:

    gr.HTML("""
    <div style='background:#112233;border-bottom:1px solid #1E3A52;padding:16px 24px;margin-bottom:8px;'>
        <div style='display:flex;align-items:center;gap:16px;'>
            <div style='background:linear-gradient(135deg,#0D9B7F,#0BC49E);border-radius:8px;padding:8px;'>
                <span style='font-size:22px;'>🤖</span>
            </div>
            <div>
                <h1 style='margin:0;color:white;font-size:22px;font-weight:700;'>
                    Kataoka Inc. — Predictive Maintenance System
                </h1>
                <p style='margin:0;color:#94A3B8;font-size:13px;'>
                    ML-powered Remaining Useful Life estimation · Random Forest v1 · 50 robots · 5 factories
                </p>
            </div>
            <div style='margin-left:auto;'>
                <span style='background:#041A14;border:1px solid #0BC49E;color:#0BC49E;
                             padding:4px 12px;border-radius:20px;font-size:12px;font-weight:600;'>● MODEL ACTIVE</span>
            </div>
        </div>
    </div>
    """)

    with gr.Tabs():

        with gr.TabItem("Fleet Dashboard"):
            fleet_summary_html = gr.HTML()
            fleet_zone_chart   = gr.Plot()
            gr.HTML("<h3 style='color:white;margin:16px 0 4px;'>Robot Registry</h3>"
                    "<p style='color:#64748B;font-size:12px;margin:0 0 8px;'>"
                    "All 50 robots · sorted by predicted RUL · robots without failure history carry higher uncertainty</p>")
            fleet_table = gr.Dataframe(
                headers=["Robot ID","Model Type","Factory","Status",
                         "Predicted RUL (hrs)","Days Remaining","Temperature","Vibration","Note"],
                interactive=False, wrap=True,
            )
            refresh_btn = gr.Button("🔄  Refresh Fleet Data", variant="primary")
            refresh_btn.click(fn=load_fleet_dashboard, inputs=[], outputs=[fleet_summary_html, fleet_zone_chart, fleet_table])
            app.load(fn=load_fleet_dashboard, inputs=[], outputs=[fleet_summary_html, fleet_zone_chart, fleet_table])

        with gr.TabItem("Robot Detail"):
            gr.HTML("<p style='color:#94A3B8;font-size:13px;margin:8px 0;'>Select any robot to load its full sensor history and RUL prediction.</p>")
            robot_dropdown = gr.Dropdown(choices=all_robot_ids, label="Select Robot", value=all_robot_ids[0])
            inspect_btn    = gr.Button("📊  Load Robot Data", variant="primary")
            status_html        = gr.HTML()
            recommendation_out = gr.Textbox(label="Recommendation", lines=4, interactive=False)
            sensor_stats_out   = gr.Markdown()
            with gr.Row():
                rul_chart_out    = gr.Plot(label="RUL Forecast")
                sensor_chart_out = gr.Plot(label="Sensor Trends")
            imp_chart_out = gr.Plot(label="Feature Importance")
            detail_outputs = [status_html, recommendation_out, sensor_stats_out, rul_chart_out, sensor_chart_out, imp_chart_out]
            inspect_btn.click(fn=load_robot_detail, inputs=[robot_dropdown], outputs=detail_outputs)
            robot_dropdown.change(fn=load_robot_detail, inputs=[robot_dropdown], outputs=detail_outputs)

        with gr.TabItem("Manual Prediction"):
            gr.HTML("""<div style='background:#132030;border:1px solid #1E3A52;border-radius:8px;padding:12px 16px;margin-bottom:12px;'>
                <p style='color:#94A3B8;font-size:13px;margin:0;'>Enter sensor readings for a single robot snapshot.
                Rolling features estimated from fleet averages. Inputs clipped to training bounds automatically.</p></div>""")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("<h4 style='color:#0BC49E;margin:0 0 8px;'>Sensor Readings</h4>")
                    vib_in    = gr.Slider(0.0,   1.0,    value=0.35,   step=0.001, label="Vibration Level (mm/s)")
                    temp_in   = gr.Slider(40.0,  90.0,   value=58.0,   step=0.1,   label="Motor Temperature (°C)")
                    torque_in = gr.Slider(60.0,  160.0,  value=108.0,  step=0.5,   label="Torque Load (Nm)")
                    power_in  = gr.Slider(800.0, 1800.0, value=1310.0, step=10.0,  label="Power Consumption (W)")
                    gr.HTML("<h4 style='color:#0BC49E;margin:12px 0 8px;'>Operating Info</h4>")
                    hours_in  = gr.Slider(0, 8136, value=3000, step=100, label="Cumulative Hours (max: 8,136)")
                    gr.HTML("<h4 style='color:#0BC49E;margin:12px 0 8px;'>Robot Configuration</h4>")
                    model_type_in = gr.Dropdown(choices=["ARM-X7","ARM-X9","ASSEMBLY-A3","CONVEYOR-C2","WELDING-W5"], value="ARM-X7", label="Robot Model Type")
                    factory_in    = gr.Dropdown(choices=["Fukuoka Center","Hokkaido Lab","Nagoya Site","Osaka Plant","Tokyo Factory"], value="Tokyo Factory", label="Factory Location")
                    env_in        = gr.Dropdown(choices=["Heavy Industrial","Standard Industrial"], value="Standard Industrial", label="Operating Environment")
                with gr.Column(scale=1):
                    gr.HTML("<h4 style='color:#0BC49E;margin:0 0 8px;'>Maintenance History</h4>")
                    maint_count_in  = gr.Slider(0, 5,   value=4,  step=1, label="Total Maintenance Count (max: 5)")
                    downtime_in     = gr.Slider(0, 500, value=50, step=5, label="Total Downtime Hours")
                    avg_downtime_in = gr.Slider(0, 100, value=10, step=1, label="Avg Downtime Hours")
                    repair_in       = gr.Slider(0, 20,  value=2,  step=1, label="Repair Count")
                    replacement_in  = gr.Slider(0, 10,  value=1,  step=1, label="Replacement Count")
                    lubrication_in  = gr.Slider(0, 20,  value=3,  step=1, label="Lubrication Count")
                    calibration_in  = gr.Slider(0, 20,  value=2,  step=1, label="Calibration Count")
                    inspection_in   = gr.Slider(0, 20,  value=3,  step=1, label="Inspection Count")
                    predict_btn        = gr.Button("⚡  Run Prediction", variant="primary", size="lg")
                    manual_result_html = gr.HTML()
                    manual_rec_out     = gr.Textbox(label="Recommendation", lines=4, interactive=False)
            predict_btn.click(
                fn=run_manual_prediction,
                inputs=[vib_in, temp_in, torque_in, power_in, hours_in,
                        model_type_in, factory_in, env_in,
                        maint_count_in, downtime_in, avg_downtime_in,
                        repair_in, replacement_in, lubrication_in,
                        calibration_in, inspection_in],
                outputs=[manual_result_html, manual_rec_out],
            )

        with gr.TabItem("About"):
            gr.Markdown("""
## About This System

**Kataoka Inc. Predictive Maintenance System**
Built by Oluwamuyiwa Jaiyeola · Data Scientist

---

### What This System Does

Analyses sensor data from 50 Kataoka industrial robots across 5 factories and predicts
how many operating hours remain before each robot needs maintenance, with a minimum of
**9.8 days advance warning**.

---

### Model Performance

| Metric | Value |
|---|---|
| Overall MAE | 630.3 hours (26.3 days) |
| Overall R² | 0.862 |
| Critical Zone MAE | 129.7 hours (±5.4 days) |
| Minimum Lead Time | 234 hours (9.8 days) |
| Missed Detections | 0 of 4 test robots |

---

### Known Limitations

- **Critical zone uncertainty**: ±130 hours when RUL < 168 hours. Treat as inspection trigger, not a precise countdown.
- **WELDING-W5 robots**: No failure history in training data. Predictions are extrapolations.
- **Robots without failure history (33 of 50)**: Features clipped to training bounds. Higher uncertainty than failure-history robots.
- **Data scope**: Training data covers the last 18 months of robot operation, not from commissioning. The 50-hour MAE target for components with 2,000+ hours remaining requires data from earlier in robot lifespans.

---

### Advisory Notice

All outputs are **advisory only**. No automated maintenance actions are triggered.
A qualified engineer must review and approve all decisions.

---

### Technical Stack
- Model: Random Forest Regressor · 28 features · sklearn
- Tracking: MLflow · PostgreSQL backend
- Training data: 14,786 readings · 17 failure robots
- GitHub: github.com/OluwamuyiwaJaiyeola/rul-predictive-maintenance
            """)

# =============================================================================
# SECTION 8: LAUNCH
# =============================================================================

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Base(primary_hue="teal", neutral_hue="slate"),
        css=custom_css,
    )
