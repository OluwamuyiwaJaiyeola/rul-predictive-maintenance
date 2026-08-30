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

# Kataoka Inc. Predictive Maintenance

An end-to-end machine learning system that estimates the Remaining Useful Life (RUL) of industrial robots and converts predictions into maintenance risk alerts.

**Built by [Oluwamuyiwa Jaiyeola](https://github.com/OluwamuyiwaJaiyeola)**

[![Live Demo](https://img.shields.io/badge/Live_Demo-Hugging_Face-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/spaces/ThatBlvck/kataoka-rul-predictive-maintenance)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Gradio-6.13-FF7C00?logo=gradio&logoColor=white)](https://www.gradio.app/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Random_Forest-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)

## Live Application

Explore the deployed system: **[Launch the Kataoka Predictive Maintenance App](https://huggingface.co/spaces/ThatBlvck/kataoka-rul-predictive-maintenance)**

The application provides:

- A fleet dashboard covering all 50 robots
- Robot-level RUL and sensor trend analysis
- Health classifications for Healthy, Warning, and Critical states
- Manual prediction for scenario testing
- Maintenance recommendations with uncertainty notices

> **Advisory notice:** Predictions support maintenance decisions but do not replace engineering judgement. A human must approve every maintenance action.

## Project Overview

Kataoka Inc. operates industrial robots across five factories. Reactive maintenance creates unplanned downtime, repair costs, and operational risk. This project uses sensor and maintenance data to estimate how many operating hours remain before component failure, enabling engineers to plan inspections and maintenance earlier.

| Project scope | Value |
|---|---:|
| Industrial robots | 50 |
| Factory locations | 5 |
| Sensor readings | 76,730 |
| Observation period | Approximately 18 months |
| Sensor frequency | Every 6 hours |
| Confirmed failure robots | 17 |
| Censored robots with no recorded failure | 33 |

The sensor data includes vibration, motor temperature, torque load, and power consumption.

## Business Objective

The system is designed to help maintenance teams:

- Identify robots approaching failure
- Move from reactive to condition-based maintenance
- Prioritise inspection and maintenance resources
- Reduce unplanned downtime and failure-related costs
- Make fleet-level risk visible through an interactive dashboard

## Model Performance

The final model is a Random Forest regressor trained with 28 operational, maintenance, and engineered time-series features.

| Metric | Result |
|---|---:|
| Overall MAE | 630.3 hours, or 26.3 days |
| Overall R² | 0.862 |
| Critical-zone MAE | 129.7 hours |
| Warning-zone MAE | 219.0 hours |
| Healthy-zone MAE | 676.0 hours |
| Minimum warning lead time | 234 hours, or 9.8 days |
| Missed detections | 0 of 4 held-out failure robots |

R² describes model fit, not prediction accuracy. MAE and warning lead time should be used alongside R² when evaluating operational usefulness.

### Held-out failure detection evidence

Early detection was defined as the first timestamp at which the model predicted RUL below the 168-hour Critical threshold. Lead time was calculated from that timestamp to the confirmed failure time.

| Held-out robot | Lead time | Lead time in days | Result |
|---|---:|---:|---|
| ROB-0013 | 234 hours | 9.8 days | Detected |
| ROB-0018 | 570 hours | 23.8 days | Detected |
| ROB-0023 | 822 hours | 34.2 days | Detected |
| ROB-0027 | 924 hours | 38.5 days | Detected |
| **Minimum** | **234 hours** | **9.8 days** | **0 missed detections** |
| Mean across four robots | 637.5 hours | 26.6 days | Descriptive only |

The defensible operational claim is therefore: **all four held-out failures were detected, with at least 9.8 days of warning before each failure**. The 9.8-day figure is the minimum lead time, not the average.

## Data Understanding and EDA

Four source datasets were assessed across the full robot fleet.

### Key findings

- 17 of 50 robots, 34%, had a confirmed failure event.
- 33 robots, 66%, had no recorded failure and were treated as censored.
- Failed robots showed slightly higher vibration and temperature.
- Failure was associated more strongly with sensor instability than with smooth increases in raw readings.
- No missing values were found across the source datasets.
- Sensor readings occurred at consistent six-hour intervals.
- Maintenance records had no timestamps, preventing calculation of time since last maintenance.

### Failure rate by factory

| Factory location | Failure rate |
|---|---:|
| Fukuoka Center | 57% |
| Tokyo Factory | 50% |
| Hokkaido Lab | 33% |
| Osaka Plant | 25% |
| Nagoya Site | 10% |

### Failure coverage by robot model

| Model type | Failure robots | Censored robots | Failure rate |
|---|---:|---:|---:|
| ARM-X7 | 7 | 8 | 47% |
| ARM-X9 | 4 | 8 | 33% |
| CONVEYOR-C2 | 3 | 6 | 33% |
| ASSEMBLY-A3 | 3 | 7 | 30% |
| WELDING-W5 | 0 | 4 | 0% |

WELDING-W5 had no confirmed failures. Its predictions are extrapolations and carry additional uncertainty.

## Data Engineering and Label Strategy

RUL labels were calculated only for robots with confirmed failure timestamps. Robots without a confirmed failure were excluded from model training and used for inference only.

| Output | Description |
|---|---|
| `merged_dataset.csv` | 76,730 rows, 50 robots, 25 columns |
| `rul_dataset.csv` | 15,245 rows from 17 failure robots |
| `rul_hours` | Remaining life calculated from confirmed failure timestamps |
| `health_risk_label` | Healthy, Warning, or Critical based on RUL |
| `label_status` | Distinguishes labelled failure timelines from censored records |
| `cumulative_hours` | Operational age calculated from timestamps |

### Label integrity

- No failure labels were fabricated for censored robots.
- RUL was capped at 8,760 hours, equivalent to one operating year.
- Data was split by robot ID to prevent observations from one robot appearing in both training and test sets.
- Rolling features were calculated within each robot timeline to prevent cross-robot leakage.

### Risk-zone distribution

| Risk zone | Readings | Share |
|---|---:|---:|
| Healthy | 13,205 | 86.6% |
| Warning | 1,564 | 10.3% |
| Critical | 476 | 3.1% |

The imbalance was handled with sample weighting. Overall classification accuracy was not used because it would conceal poor performance in the minority Critical zone.

## Feature Engineering

Fifteen time-series and compound features were created to capture degradation patterns that raw sensor values missed.

| Feature group | Examples | Purpose |
|---|---|---|
| Seven-day rolling mean | Vibration, temperature, torque, power | Capture medium-term operating trends |
| Seven-day rolling standard deviation | Vibration, temperature, torque, power | Measure weekly instability |
| Twenty-four-hour rolling standard deviation | Vibration, temperature, torque | Detect short-term instability |
| Load stress | Torque multiplied by cumulative hours | Represent accumulated mechanical stress |
| Short-versus-long differences | Vibration and temperature | Detect recent deviation from historical behaviour |

After the seven-day rolling window was applied, 459 warm-up rows were removed. The resulting engineered dataset contained 14,786 rows and 40 columns.

## Modelling and Validation

Random Forest, Gradient Boosting, SVR, and XGBoost-based approaches were evaluated. The training and test split was performed at robot level, using 13 failure robots for training and four unseen failure robots for testing.

### Selected model comparison

| Configuration | Features | Overall MAE | R² | Critical MAE | Missed detections |
|---|---:|---:|---:|---:|---:|
| Sensor-only Random Forest | Raw sensors | 2,026.9 hrs | -0.201 | 2,310.5 hrs | Not selected |
| Baseline B Random Forest | 16 | 659.7 hrs | 0.855 | 59.1 hrs | 0 |
| Config C Random Forest | 28 | **630.3 hrs** | **0.862** | 129.7 hrs | **0** |
| Config D Random Forest | 31 | 631.8 hrs | 0.861 | 156.2 hrs | 0 |

The sensor-only models produced negative R² scores, showing that raw readings alone did not generalise. Operational context and engineered time-series features were necessary.

Config C was selected because it produced the best overall MAE and R² while maintaining zero missed detections. Its weaker Critical-zone MAE is reported explicitly rather than hidden.

### Tuning experiments

Eight tuning and modelling approaches were tested to improve Critical-zone precision, including elevated sample weights, grid search, custom scoring, XGBoost, Huber loss, and two-stage routing.

The experiments exposed a structural trade-off. Approaches that improved Critical-zone MAE substantially damaged overall performance, while approaches that protected overall performance did not improve Critical-zone precision. Only 364 Critical-zone training readings were available across 13 training robots, limiting the patterns the model could learn near failure.

## Final Model Decision

**Random Forest Config C** was registered as the production model.

| Detail | Value |
|---|---|
| Features | 28 |
| Training data | 14,786 readings from 13 robots |
| Test data | 5,008 readings from 4 held-out robots |
| MLflow registry name | `kataoka_rul_final` |
| Registry version | 1 |
| Model artifact | `models/tuned/rul_regressor_final.pkl` |
| Scaler artifact | `models/tuned/scaler_final.pkl` |

### Censored-robot validation

The 33 robots without recorded failures were passed through the final model as an inference check.

- Mean fleet-wide predicted RUL was 2,228 hours.
- Predicted RUL declined as cumulative operating hours increased.
- Robots generally moved from Healthy to Warning and Critical states over time.
- WELDING-W5 produced higher Critical percentages, reinforcing its documented extrapolation risk.

These checks test whether predictions behave plausibly. They do not provide ground-truth accuracy because the censored robots have no confirmed failure timestamp.

## Gradio Decision-Support Application

The deployed application contains four sections:

| Section | Function |
|---|---|
| Fleet Dashboard | Ranks 50 robots by predicted RUL and displays fleet health distribution |
| Robot Detail | Shows RUL history, sensor trends, feature importance, current readings, and recommendations |
| Manual Prediction | Accepts a single sensor and maintenance snapshot for approximate scenario testing |
| About | Documents model performance, limitations, and advisory conditions |

### Deployment safeguards

- Inference inputs are clipped to training bounds before scaling.
- Predictions for WELDING-W5 are flagged as extrapolations.
- Censored robots are marked as higher uncertainty.
- Critical predictions include an approximately ±130-hour uncertainty notice.
- Manual mode is labelled approximate because rolling features require historical readings.
- All outputs are presented as advisory and require human review.

## Deployment

The Gradio application is hosted on Hugging Face Spaces and automatically redeployed through GitHub Actions after a push to the `main` branch.

**Live app:** [huggingface.co/spaces/ThatBlvck/kataoka-rul-predictive-maintenance](https://huggingface.co/spaces/ThatBlvck/kataoka-rul-predictive-maintenance)

## KPI Assessment

| KPI | Target | Current status | Assessment |
|---|---:|---|---|
| Technical MAE | Less than 50 hours | 630.3 hours overall | Not achievable with the current observation window |
| Annual penalty-fee reduction | $2.5M | Technical capability delivered | Requires Kataoka breakdown-cost data for financial validation |
| Predictive-alert adoption | 90% | Application deployed | Requires live workflow integration and post-deployment measurement |

The 18-month dataset begins after the robots were commissioned. The model therefore does not observe complete operating lifecycles. A 50-hour MAE target should be treated as a future milestone after full-lifecycle data is collected from newly commissioned robots.

## Known Limitations

| Limitation | Operational impact | Improvement path |
|---|---|---|
| Critical-zone MAE is 129.7 hours | Critical alerts should trigger inspection, not act as exact failure countdowns | Collect more near-failure trajectories |
| Data covers approximately 18 months | Early-life and mid-life behaviour cannot be learned fully | Collect data from commissioning onward |
| WELDING-W5 has no failure examples | Predictions are extrapolated from other robot types | Retrain after confirmed W5 failures |
| Maintenance records have no timestamps | Time since last maintenance cannot be used as a feature | Add timestamps to future maintenance logs |
| 33 robots are censored | Their true RUL is unknown | Update labels as confirmed failures occur |

## Project Structure

```text
rul-predictive-maintenance/
├── data/
│   ├── raw/
│   └── processed/
│       ├── merged_dataset.csv
│       ├── rul_dataset.csv
│       └── featured_dataset.csv
├── models/
│   ├── baseline/
│   ├── engineered/
│   └── tuned/
│       ├── rul_regressor_final.pkl
│       └── scaler_final.pkl
├── notebooks/
│   ├── 01_eda_analysis.ipynb
│   ├── 02_data_engineering_label_strategy.ipynb
│   ├── 03_baseline_modeling.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_engineered_model_training.ipynb
│   └── 06_hyperparameter_tuning.ipynb
├── reports/
│   ├── figures/
│   └── stakeholder_summary.md
├── .github/workflows/deploy.yml
├── app.py
├── requirements.txt
└── README.md
```

## Technology Stack

- Python
- Pandas and NumPy
- scikit-learn
- XGBoost
- MLflow with PostgreSQL
- Plotly
- Gradio
- Hugging Face Spaces
- GitHub Actions

## Run Locally

```bash
git clone https://github.com/OluwamuyiwaJaiyeola/rul-predictive-maintenance.git
cd rul-predictive-maintenance
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Further Documentation

- Review the [`notebooks/`](notebooks/) directory for the complete analytical workflow.
- Read the [`stakeholder_summary.md`](reports/stakeholder_summary.md) report for phase-by-phase findings.
- Explore the **[live Hugging Face application](https://huggingface.co/spaces/ThatBlvck/kataoka-rul-predictive-maintenance)**.

