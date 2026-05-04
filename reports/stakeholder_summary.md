# Kataoka Inc. Predictive Maintenance
## Project Progress Summary
### Prepared by: Oluwamuyiwa Jaiyeola
### Last Updated: Phase 3 Complete
### GitHub: https://github.com/OluwamuyiwaJaiyeola/rul-predictive-maintenance

---

## Project Overview

This project builds an end-to-end predictive maintenance system for
Kataoka Inc.'s industrial robot fleet. The goal is to predict Remaining
Useful Life (RUL) of robot components before failure occurs, enabling
engineers to shift from reactive to condition-based maintenance.

---

## Phase 1: Data Understanding & EDA
### Status: Complete

### What We Did
We explored four data sources covering 50 industrial robots operated
by Kataoka Inc. across five factory locations in Japan.

### Key Findings

#### Fleet Coverage
- 50 robots actively monitored with sensor readings every 6 hours
- Data spans approximately 18 months of operations
- All robots have sensor and maintenance records

#### Failure Coverage
- 17 of 50 robots (34%) have a confirmed failure record
- 33 robots (66%) have no recorded failure, treated as censored
- Each failed robot has a single recorded failure event

#### High-Risk Locations
| Factory Location | Failure Rate |
|---|---|
| Fukuoka Center | 57% |
| Tokyo Factory | 50% |
| Hokkaido Lab | 33% |
| Osaka Plant | 25% |
| Nagoya Site | 10% |

#### Failure Coverage by Robot Model Type
| Model Type | Failure Robots | Censored Robots | Failure Rate |
|---|---|---|---|
| ARM-X7 | 7 | 8 | 47% |
| ARM-X9 | 4 | 8 | 33% |
| CONVEYOR-C2 | 3 | 6 | 33% |
| ASSEMBLY-A3 | 3 | 7 | 30% |
| WELDING-W5 | 0 | 4 | 0% |

WELDING-W5 has no confirmed failures. Predictions for this type
are extrapolations and should be treated with additional caution.

#### Sensor Behaviour
- Failed robots show marginally higher vibration and temperature
- Failure is preceded by instability, not smooth upward trends
- Predictive performance depends on detecting behavioural changes
  over time, not absolute sensor values

#### Data Quality
- No missing values across all four datasets
- Sensor readings confirmed at consistent 6-hour intervals
- No imputation required

#### Known Limitation
- Maintenance log has no timestamps
- Time since last maintenance cannot be computed
- Maintenance features limited to aggregate counts and totals

---

## Phase 2: Data Engineering & Label Strategy
### Status: Complete

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

This imbalance is expected and was addressed through sample weighting
in Phase 3. Overall accuracy is not a valid evaluation metric here.

### Modelling Strategy Confirmed
- RUL regression trained on 17 failure robots only
- Health risk classification derived from predicted RUL at inference
- Censored robots used for inference only, never as training labels

---

## Phase 3: Baseline Modelling
### Status: Complete

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

**Every test robot received a Critical warning before failure.**
**Minimum lead time: 384 hours (16 days). Zero missed detections.**

### Key Findings
- Raw sensors alone have no predictive value (all Baseline A models R² negative)
- Cumulative hours is the dominant signal at baseline
- Random Forest Baseline B is the best model
- SVR dropped from all future phases
- The model correctly prioritises predictions near failure where it matters most

### Benchmarks for Phase 5
| Metric | Target |
|---|---|
| Overall MAE | < 659.7 hours |
| Overall R² | > 0.855 |
| Critical MAE | < 59.1 hours |
| Min lead time | >= 384 hours |
| Missed detections | 0 (non-negotiable) |

---

## Decisions Required

| Decision | Owner | Status |
|---|---|---|
| Confirm Critical / Warning thresholds | Business / Operations | Proposed: <7 days Critical, 7-30 days Warning |
| Investigate Fukuoka and Tokyo high failure rates | Operations Manager | Pending |
| Confirm RUL cap at 8,760 hours | Project Manager | Pending |
| Review WELDING-W5 zero failure coverage | Engineering | Pending |

---

## What Happens Next

### Phase 4: Feature Engineering
Build temporal features that capture degradation patterns:
- Rolling mean and standard deviation for all sensors
- Wear trend slope per robot
- Load stress index
- Short-term vs long-term sensor difference

### Phase 5: Engineered Model Training
Retrain Random Forest and Gradient Boosting with engineered features.
Compare against Phase 3 baseline. Quantify improvement.

### Phase 6: Hyperparameter Tuning
GridSearchCV with GroupKFold by robot_id.
Select final model. Log to MLflow.

### Phase 7: Gradio Application
Engineer-facing interface with RUL prediction,
health status, and maintenance recommendation.

### Phase 8: Deployment
Hugging Face Spaces via GitHub.

---

## Supporting Figures
All charts saved in reports/figures/

| Figure | Description |
|---|---|
| failure_analysis.png | Failure type and root cause |
| failure_rate_by_factory_location.png | Risk by location |
| failure_coverage_by_model_type.png | Coverage by robot type |
| robot_id_model_type_scatter.png | Robot ID to model type mapping |
| failed_robots_reference_table.png | Failed robot details |
| sensor_distributions.png | Sensor reading ranges |
| failed_vs_nonfailed_sensors.png | Sensor comparison |
| sensor_trends_before_failure.png | Sensor behaviour before failure |
| rul_distribution.png | RUL label distribution |
| train_test_rul_distribution.png | Train vs test RUL coverage |
| baseline_comparison.png | Baseline A vs B performance |
| baseline_predicted_vs_actual.png | Predicted vs actual RUL scatter |

---

*For full technical detail see the notebooks/ folder.*
*All experiment runs tracked in MLflow with PostgreSQL backend.*
*GitHub: https://github.com/OluwamuyiwaJaiyeola/rul-predictive-maintenance*