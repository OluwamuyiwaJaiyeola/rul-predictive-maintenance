---

title: Kataoka RUL Predictive Maintenance
emoji: 🤖
colorFrom: green
colorTo: indigo
sdk: gradio
sdk_version: "6.13.0"
python_version: "3.11"
app_file: app.py
pinned: false

---

# Kataoka Inc. — Predictive Maintenance System
ML-powered Remaining Useful Life (RUL) estimation for industrial robots.

Built by Oluwamuyiwa Jaiyeola

## What This System Does
Predicts remaining useful life for 50 Kataoka industrial robots across 5 factories
using 18 months of sensor data (vibration, temperature, torque, power consumption).

## Model Performance
- Overall MAE: 630.3 hours
- Overall R²: 0.862
- Minimum lead time: 234 hours (9.8 days)
- Missed detections: 0 of 4 test robots

## Advisory Notice
All outputs are advisory only. Human decision required before any maintenance action.
