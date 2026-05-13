# Kataoka Inc. Predictive Maintenance
## Project Progress Summary
**Prepared by:** Oluwamuyiwa Jaiyeola  
**Last Updated:** Phase 8 Complete — Deployed  
**GitHub:** https://github.com/OluwamuyiwaJaiyeola/rul-predictive-maintenance  
**Live App:** https://huggingface.co/spaces/ThatBlvck/kataoka-rul-predictive-maintenance

---

## Project Overview

This project builds an end-to-end predictive maintenance system for Kataoka Inc.'s industrial robot fleet. The goal is to predict Remaining Useful Life (RUL) of robot components before failure occurs, enabling engineers to shift from reactive to condition-based maintenance.

---

## Phase 1: Data Understanding & EDA
**Status: Complete**

### What We Did
We explored four data sources covering 50 industrial robots operated by Kataoka Inc. across five factory locations in Japan.

### Key Findings

**Fleet Coverage**
- 50 robots actively monitored with sensor readings every 6 hours
- Data spans approximately 18 months of operations
- All robots have sensor and maintenance records

**Failure Coverage**
- 17 of 50 robots (34%) have a confirmed failure record
- 33 robots (66%) have no recorded failure, treated as censored
- Each failed robot has a single recorded failure event

**High-Risk Locations**

| Factory Location | Failure Rate |
|---|---|
| Fukuoka Center | 57% |
| Tokyo Factory | 50% |
| Hokkaido Lab | 33% |
| Osaka Plant | 25% |
| Nagoya Site | 10% |

**Failure Coverage by Robot Model Type**

| Model Type | Failure Robots | Censored Robots | Failure Rate |
|---|---|---|---|
| ARM-X7 | 7 | 8 | 47% |
| ARM-X9 | 4 | 8 | 33% |
| CONVEYOR-C2 | 3 | 6 | 33% |
| ASSEMBLY-A3 | 3 | 7 | 30% |
| WELDING-W5 | 0 | 4 | 0% |

WELDING-W5 has no confirmed failures. Predictions for this type are extrapolations and should be treated with additional caution.

**Sensor Behaviour**
- Failed robots show marginally higher vibration and temperature
- Failure is preceded by instability, not smooth upward trends
- Predictive performance depends on detecting behavioural changes over time, not absolute sensor values

**Data Quality**
- No missing values across all four datasets
- Sensor readings confirmed at consistent 6-hour intervals
- No imputation required

**Known Limitation**
- Maintenance log has no timestamps
- Time since last maintenance cannot be computed
- Maintenance features limited to aggregate counts and totals

---

## Phase 2: Data Engineering & Label Strategy
**Status: Complete**

### What Was Built

| Output | Description |
|---|---|
| merged_dataset.csv | Full dataset, 76,730 rows, 50 robots, 25 columns |
| rul_dataset.csv | Failure robots only, 15,245 rows, 17 robots |
| rul_hours | RUL labels from confirmed failure timestamps only |
| health_risk_label | Critical / Warning / Healthy from RUL thresholds |
| label_status | labelled_failure_timeline or censored_unlabelled |
| cumulative_hours | Operational age per robot from actual timestamps |
| 8 maintenance features | Aggregate counts and downtime per robot |

### Label Integrity
- RUL computed only from confirmed failure timestamps
- No fabricated labels assigned to censored robots
- Censored robots flagged explicitly for inference only
- RUL capped at 8,760 hours (one operating year)

### Class Imbalance Documented

| Risk Zone | Readings | Share |
|---|---|---|
| Healthy | 13,205 | 86.6% |
| Warning | 1,564 | 10.3% |
| Critical | 476 | 3.1% |

This imbalance is expected and was addressed through sample weighting in Phase 3. Overall accuracy is not a valid evaluation metric here.

### Modelling Strategy Confirmed
- RUL regression trained on 17 failure robots only
- Health risk classification derived from predicted RUL at inference
- Censored robots used for inference only, never as training labels

---

## Phase 3: Baseline Modelling
**Status: Complete**

### What Was Built
- Two baseline feature sets evaluated: sensors only (A) and full context (B)
- Three models trained: Random Forest, Gradient Boosting, SVR
- Robot-level train/test split: 13 train robots, 4 test robots
- Sample weighting applied to address class imbalance
- All runs logged to MLflow with PostgreSQL backend

### Baseline Results

| Model | Baseline | MAE (hrs) | MAE (days) | R² | Critical MAE |
|---|---|---|---|---|---|
| RandomForest | B | 659.7 | 27.5 | 0.855 | 59.1 hrs |
| GradientBoosting | B | 1042.5 | 43.4 | 0.699 | 81.0 hrs |
| RandomForest | A | 2026.9 | 84.5 | -0.201 | 2310.5 hrs |
| GradientBoosting | A | 2629.2 | 109.6 | -1.074 | 1055.6 hrs |
| SVR | A | 2215.4 | 92.3 | -0.446 | 2208.5 hrs |
| SVR | B | 2102.2 | 87.6 | -0.309 | 2072.3 hrs |

### Early Detection Results

| Robot | Lead Time (hours) | Lead Time (days) |
|---|---|---|
| ROB-0013 | 390 | 16.2 |
| ROB-0018 | 384 | 16.0 |
| ROB-0023 | 1,482 | 61.8 |
| ROB-0027 | 1,362 | 56.8 |

Every test robot received a Critical warning before failure. Minimum lead time: 384 hours (16 days). Zero missed detections.

### Key Findings
- Raw sensors alone have no predictive value (all Baseline A models R² negative)
- Cumulative hours is the dominant signal at baseline
- Random Forest Baseline B is the best model
- SVR dropped from all future phases

---

## Phase 4: Feature Engineering
**Status: Complete**

### What Was Built

15 engineered features added to capture degradation patterns that raw sensors cannot detect:

| Feature Group | Features | Purpose |
|---|---|---|
| 7-day rolling mean | vibration, temperature, torque, power | Smooth trend per sensor |
| 7-day rolling std | vibration, temperature, torque, power | Instability over one week |
| 24-hour rolling std | vibration, temperature, torque | Short-window instability |
| Compound | load_stress_index | torque × cumulative_hours |
| Compound | vibration_short_long_diff | Recent vs historical vibration |
| Compound | temperature_short_long_diff | Recent vs historical temperature |

### Output Dataset
- **featured_dataset.csv**: 14,786 rows × 40 columns
- CORE_FEATURES_V2 defined: 12 engineered features for Phase 5+
- CONFIG_C defined: 28 total features (16 Baseline B + 12 CORE_FEATURES_V2)

### Feature Validation
- All features computed before NaN drop to prevent leakage
- 459 warm-up rows dropped (28 readings × 17 robots for 7-day window)
- Feature behaviour validated: rolling std and load_stress_index rise as RUL approaches zero
- Groupby(robot_id) applied throughout to prevent cross-robot leakage

---

## Phase 5: Engineered Model Training
**Status: Complete**

### What Was Tested

| Configuration | Features | MAE (hrs) | R² | Critical MAE | Missed |
|---|---|---|---|---|---|
| Baseline B RF (Phase 3) | 16 | 659.7 | 0.855 | 59.1 | 0 |
| Config C RF | 28 | 630.3 | 0.862 | 129.7 | 0 |
| Config D RF | 31 | 631.8 | 0.861 | 156.2 | 0 |

### Key Finding: Critical MAE Regression
Config C improved overall MAE and R² but Critical MAE regressed from 59.1 to 129.7 hours. This was investigated thoroughly in Phase 6.

Root cause identified: the model learned to predict RUL primarily from cumulative_hours (36% feature importance) and total_maintenance_count (19%), both of which are age proxies. With 96.9% of readings in the Healthy zone, the model had no incentive to learn precise near-failure patterns from the 3.1% Critical readings.

### Phase 5 Best Model
- **Config C Random Forest, default parameters**
- Overall MAE: 630.3 hours
- Overall R²: 0.862
- Min lead time: 234 hours (9.8 days)
- Missed detections: 0

---

## Phase 6: Hyperparameter Tuning and Model Selection
**Status: Complete**

### Approaches Tested

8 distinct approaches were tested to recover Critical MAE below the Phase 3 baseline of 59.1 hours:

| Experiment | Approach | Overall MAE | Critical MAE | Missed | Verdict |
|---|---|---|---|---|---|
| Exp 1 | Sample weight elevation (8.95 → 25x) | 669.8 | 211.3 | 0 | Critical MAE worsened |
| Exp 2 | GridSearchCV MAE scoring | 905.1 | 153.9 | 1 | Missed detection — rejected |
| Exp 3 | GridSearchCV custom composite scorer | 1345.0 | 837.9 | 4 | Complete collapse — rejected |
| Exp 4 | XGBoost inverse RUL weights | 1256.3 | 70.8 | 0 | Overall MAE too high |
| Exp 5 | XGBoost weight sensitivity | ~1254 | ~78 | 0 | Same trade-off |
| Exp 6 | XGBoost Huber loss | 1377.6 | 64.8 | 0 | Overall MAE too high |
| Exp 7 | Two-stage hard routing | 590.8 | 256.8 | 0 | Critical MAE worst of all |
| Exp 7 | Two-stage soft routing | 910.1 | 303.3 | 4 | Missed all — rejected |

### Why No Approach Worked
The root cause is data volume. With only 364 Critical zone training readings from 13 robots, no tuning strategy can force the model to learn precise near-failure behaviour. The trade-off is structural: any approach that improved Critical MAE collapsed overall MAE, and vice versa.

### Censored Robot Inference Validation
All 33 never-failed robots were run through the trained model as an inference validation:
- Fleet-wide mean predicted RUL: 2,228 hours (physically plausible)
- Predicted RUL declines as cumulative hours increase (correct behaviour)
- Robots spend majority of their history in Healthy zone before transitioning to Warning and Critical
- WELDING-W5 robots show highest Critical percentages (11–30%), confirming documented extrapolation risk

### Final Model Decision
**Random Forest Config C, Phase 5 defaults** is confirmed as the final model.

| Metric | Phase 3 Baseline | Final Model | Status |
|---|---|---|---|
| Overall MAE | 659.7 hours | 630.3 hours | Improved |
| Overall R² | 0.855 | 0.862 | Improved |
| Critical MAE | 59.1 hours | 129.7 hours | Regressed — data limitation |
| Min lead time | 384 hours | 234 hours | Reduced but operationally viable |
| Missed detections | 0 | 0 | Maintained |

The Critical MAE limitation is a data constraint, not a model failure. Every new confirmed failure robot that accumulates in production will improve Critical zone precision. With 25 to 30 failure trajectories, significant improvement is expected.

**Model registered in MLflow Registry:** `kataoka_rul_final` version 1, stage=production  
**Artifacts saved:** `models/tuned/rul_regressor_final.pkl`, `models/tuned/scaler_final.pkl`

---

## Phase 7: Gradio Application
**Status: Complete**

### What Was Built
A four-tab advisory decision support application built with Gradio and deployed to Hugging Face Spaces.

| Tab | Description |
|---|---|
| Fleet Dashboard | All 50 robots sorted by predicted RUL. KPI cards for health zone distribution. Doughnut chart of fleet status. Robot registry table with predicted RUL, days remaining, temperature, vibration. |
| Robot Detail | Select any of 50 robots. Loads full 18-month sensor history. Displays RUL trend chart coloured by zone, 4-panel sensor chart, feature importance chart, latest sensor readings, and recommendation. |
| Manual Prediction | Slider inputs for all 28 features. Computes RUL from a single reading snapshot. Rolling features estimated from fleet averages. |
| About | System documentation, model performance, known limitations, advisory notice. |

### Technical Design Decisions
- **Training-bounds clipping** applied to all inference inputs before scaling. Three features fall outside the training distribution for censored robots: cumulative_hours (training max 8,136 vs censored max 12,960), total_maintenance_count (training max 5 vs censored max 7), vibration_level (training min 0.12 vs censored min 0.08). Clipping to training bounds prevents the scaler from producing extreme values.
- All outputs labelled **Advisory — Human decision required**
- WELDING-W5 predictions flagged as extrapolations throughout
- Critical zone predictions show ±130 hour uncertainty notice

---

## Phase 8: Deployment
**Status: Complete**

### Deployment Details
- **Platform:** Hugging Face Spaces
- **URL:** https://huggingface.co/spaces/ThatBlvck/kataoka-rul-predictive-maintenance
- **Method:** GitHub Actions workflow using HF Python API upload_folder
- **Trigger:** Automatic deployment on every push to main branch

### What Is Deployed
- `app.py` — full Gradio application
- `models/tuned/rul_regressor_final.pkl` — final Random Forest model
- `models/tuned/scaler_final.pkl` — fitted StandardScaler
- `data/processed/merged_dataset.csv` — all 50 robots for fleet dashboard
- `data/processed/featured_dataset.csv` — 17 failure robots with engineered features
- `requirements.txt` — pinned dependencies

---

## KPI Assessment
**Updated: May 2026**

| KPI | Target | Current Status | Assessment |
|---|---|---|---|
| Technical: MAE < 50 hours (2,000+ hr lifespan) | 50 hours | 630.3 hours overall | Not achievable with current dataset. No robot in training data had 2,000+ hours remaining when first observed. Requires full-lifespan data from commissioning. See KPI note below. |
| Business: $2.5M annual penalty fee reduction | $2.5M | Technical capability delivered | Model provides 9.8-day advance warning. Financial validation requires Kataoka's actual breakdown cost data. BA to confirm with operations. |
| User Adoption: 90% tasks via predictive alert | 90% | Tool delivered, adoption not yet measurable | Advisory tool deployed. 90% adoption requires live integration. Realistically a 12-month post-integration target. |

### KPI 1 Note: Why the 50-hour Target Cannot Be Met With This Data
The 18-month observation window captured robots that were already mid-to-late life when data collection began. All 17 failure robots accumulated between 2,328 and 8,136 hours before failing. Not a single robot in the training set had 2,000+ hours of life remaining at the start of observation. The model cannot predict what it has never seen.

**Recommended revision:** Set MAE < 630 hours as the Phase 1 baseline met. Set 50-hour MAE as a v2 milestone once Kataoka begins collecting sensor data from newly commissioned robots from day one.

---

## Known Limitations

| Limitation | Impact | Resolution Path |
|---|---|---|
| Critical MAE: 129.7 hours | Predictions in the final 7 days carry ±130 hour uncertainty. Treat Critical alerts as inspection triggers, not precise countdowns. | Accumulate more failure trajectories. Each new failure robot improves Critical zone precision. |
| Dataset scope: 18 months of late-stage data | Model cannot distinguish early-life from mid-life robots. All predictions assume robots are in their later operational phase. | Activate sensors on newly commissioned robots from day one. |
| WELDING-W5: no failure baseline | All W5 predictions are extrapolations from other robot types. Higher uncertainty than other model types. | Log the first W5 failure event carefully. This single data point will significantly improve W5 predictions. |
| Maintenance table: no timestamps | Time since last maintenance cannot be computed. Could be a strong degradation signal if available. | Request Kataoka IT to add timestamps to future maintenance log entries. |
| 33 censored robots | Their true RUL is unknown. Predictions clipped to training bounds but carry higher uncertainty than failure-history robots. | These robots will eventually fail. Each failure adds valuable training data. |

---

## Project Architecture

```
rul-predictive-maintenance/
├── data/
│   ├── raw/                    (4 original CSVs)
│   └── processed/
│       ├── merged_dataset.csv  (76,730 rows, 50 robots)
│       ├── rul_dataset.csv     (15,245 rows, 17 failure robots)
│       └── featured_dataset.csv (14,786 rows, 40 columns)
├── models/
│   ├── baseline/               (Phase 3 model artifacts)
│   ├── engineered/             (Phase 5 model artifacts)
│   └── tuned/
│       ├── rul_regressor_final.pkl   ← deployed model
│       └── scaler_final.pkl          ← deployed scaler
├── notebooks/
│   ├── 01_eda_analysis.ipynb
│   ├── 02_data_engineering_label_strategy.ipynb
│   ├── 03_baseline_modeling.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_engineered_model_training.ipynb
│   └── 06_hyperparameter_tuning.ipynb
├── reports/
│   ├── figures/                (20 charts from Phases 1-6)
│   └── stakeholder_summary.md  (this document)
├── .github/workflows/
│   └── deploy.yml              (GitHub Actions → HF Spaces)
├── app.py                      (Gradio application)
├── requirements.txt
└── README.md
```

---

## Final Model Summary

**Model:** Random Forest Regressor  
**Features:** 28 (Baseline B 16 + CORE_FEATURES_V2 12)  
**Training data:** 14,786 readings from 13 training robots  
**Test data:** 5,008 readings from 4 held-out robots  
**MLflow Registry:** `kataoka_rul_final` v1, stage=production

| Metric | Value |
|---|---|
| Overall MAE | 630.3 hours (26.3 days) |
| Overall R² | 0.862 |
| Critical Zone MAE | 129.7 hours (documented limitation) |
| Warning Zone MAE | 219.0 hours |
| Healthy Zone MAE | 676.0 hours |
| Minimum lead time | 234 hours (9.8 days) |
| Missed detections | 0 of 4 test robots |

---

*For full technical detail see the notebooks/ folder. All experiment runs tracked in MLflow with PostgreSQL backend.*  
*GitHub: https://github.com/OluwamuyiwaJaiyeola/rul-predictive-maintenance*  
*Live app: https://huggingface.co/spaces/ThatBlvck/kataoka-rul-predictive-maintenance*
