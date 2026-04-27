# Kataoka Inc. Predictive Maintenance  
## EDA Findings Summary  
### Prepared by: Oluwamuyiwa Jaiyeola  
### Phase: 1 - Data Understanding & Exploratory Data Analysis  
### Status: Complete  

---

## What We Did

We explored four data sources covering 50 industrial robots  
operated by Kataoka Inc. across five factory locations in Japan.

The goal was to assess data quality, understand failure patterns,  
and analyse sensor behaviour before building predictive models.

---

## Key Findings

### 1. Fleet Coverage
- 50 robots are actively monitored with sensor readings at approximately 6-hour intervals  
- Data spans ~18 months of operations  
- All robots have both sensor and maintenance records  

---

### 2. Failure Coverage
- 17 of 50 robots (~34%) have a confirmed failure record  
- 33 robots (~66%) have no recorded failure and are treated as **censored observations**  
- Each failed robot has a single recorded failure event  
- This limits the amount of labelled data available for RUL modelling  

---

### 3. High-Risk Locations

| Factory Location | Failure Rate |
|---|---|
| Fukuoka Center | 57% |
| Tokyo Factory | 50% |
| Hokkaido Lab | 33% |
| Osaka Plant | 25% |
| Nagoya Site | 10% |

Fukuoka Center and Tokyo Factory show significantly higher failure rates.  
This may indicate environmental or operational differences that should be  
investigated alongside modelling efforts.

---

### 4. Most Common Failures
- Bearing failure is the most frequent failure type (6 cases)  
- Motor failure is second (5 cases)  
- Thermal cycling fatigue is the leading root cause (6 cases)  

This suggests mechanical wear and temperature-related stress are the primary risks.

---

### 5. Sensor Behaviour
- Failed robots show **marginally higher vibration and temperature**  
- Failure is **not preceded by smooth upward trends**  
- Instead, sensor readings show **instability and irregular fluctuations** before failure  

This indicates that predictive performance will depend on detecting  
**changes in behaviour over time**, rather than relying on absolute sensor values.

---

### 6. Data Quality
- No missing values detected across the four datasets  
- Sensor readings appear to follow a consistent 6-hour interval  
- No imputation is required at this stage  

---

### 7. Known Data Limitation
- The maintenance log does not contain timestamps  
- We cannot calculate “time since last maintenance”  
- Maintenance features will be limited to **aggregate metrics** such as:
  - total maintenance events  
  - total downtime  
  - maintenance type counts  

---

## Modelling Strategy Implications

Two modelling approaches will be used:

1. **RUL Regression**
   - Trained only on robots with confirmed failure events  
   - Predicts remaining time before failure  

2. **Health Classification**
   - Trained on the full fleet  
   - Categorises robots into Healthy / Warning / Critical  

Feature engineering will prioritise:
- rolling statistics  
- volatility and instability  
- time-based behavioural patterns  

Non-linear models (e.g. tree-based models) are expected to perform better than linear models.

---

## Decisions Required Before Phase 2

| Decision | Owner | Notes |
|---|---|---|
| Confirm health risk thresholds (Critical / Warning / Healthy) | Business / Operations | Default: Critical < 7 days, Warning 7–30 days |
| Confirm whether Fukuoka and Tokyo require separate investigation | Operations Manager | High failure rates may indicate environmental factors |
| Confirm RUL cap (proposed: 8760 hours / 1 year) | Project Manager | Defines prediction horizon and prevents unrealistic forecasts |

---

## What Happens Next

Phase 2 will build the analysis-ready dataset:

- Join robot metadata to sensor readings  
- Compute Remaining Useful Life (RUL) for the 17 failed robots  
- Correctly flag censored robots  
- Create health status labels for the full fleet  

These outputs will feed directly into model training in Phase 3.

---

## Supporting Figures

All charts referenced are saved in `reports/figures/`.

| Figure | Description |
|---|---|
| failure_analysis.png | Failure type and root cause distribution |
| failure_rate_by_factory_location.png | Risk by location |
| robot_distribution_by_factory_location.png | Fleet composition by site |
| sensor_distributions.png | Sensor reading ranges across fleet |
| failed_vs_nonfailed_sensors.png | Sensor comparison between failed and non-failed robots |
| sensor_trends_before_failure.png | Sensor behaviour before failure |
| general_sensor_correlation.png | Relationship between sensors |

---

*For full technical detail, see `notebooks/01_eda.ipynb`*  
*GitHub: https://github.com/OluwamuyiwaJaiyeola/rul-predictive-maintenance*