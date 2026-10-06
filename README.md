# Freight Rate Prediction Challenge — Machine Learning Assessment

Production-grade machine learning solution for predicting truckload freight spot rates (`posted_rate`) across heterogeneous lanes, equipment classes, and dynamic macroeconomic conditions.

---

## 1. Quick Start & Execution

### Setup Environment
```bash
python -m pip install -r requirements.txt
```

### End-to-End Pipeline
```bash
# 1. Train model with out-of-time chronological validation and fit on full dev data
python train.py

# 2. Generate final predictions (validation_predictions.csv and december-chart-inputs.csv)
python predict.py

# 3. Validate outputs with official scorer
python score.py --predictions validation_predictions.csv --december-predictions december-chart-inputs.csv
```

---

## 2. Key Results & Benchmark Summary

Models evaluated under strict **Out-of-Time Chronological Validation** (Train: Jan–Aug 2025 [38,477 loads], Val: Sep–Oct 2025 [9,523 loads]):

| Model Architecture | MAE ($) | RMSE ($) | MAPE (%) | $R^2$ | Fit Time | Selection Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **CatBoost Regressor** | **$120.19** | **$635.72** | **5.76%** | **0.8265** | **1.9s** | **SELECTED BEST** |
| LightGBM Regressor | $149.92 | $642.97 | 6.40% | 0.8225 | 2.4s | Strong Baseline |
| XGBoost Regressor | $181.46 | $691.53 | 8.32% | 0.7947 | 2.3s | Higher Variance |
| Random Forest Regressor | $186.67 | $678.89 | 8.22% | 0.8021 | 7.5s | Linear Boundary Limit |

---

## 3. Core Methodology Highlights

1. **Defensible Validation Strategy**:
   - Out-of-time chronological split (8 months dev train, 2 months dev validation).
   - Prevents temporal target leakage and accurately reflects future market shifts (Nov–Dec test period).

2. **Data-Quality Remediation**:
   - **Weight Sign-Flip Artifact**: 292 train and 145 validation rows had negative weights down to -47,500 lbs. Distributional analysis confirmed `abs(negative_weight)` matches positive weights exactly. Applying `abs(weight)` cleanly restored the ground truth.
   - **Minimal Missing Values**: Imputed with group/overall medians, preventing data loss.
   - **Unseen Cities**: 8 new cities in validation are seamlessly handled via latitude/longitude centroids and Haversine geospatial distance features.

3. **Domain Feature Engineering**:
   - Geospatial: Haversine distance, centroid offsets ($\Delta\text{lat}, \Delta\text{lon}$), distance-to-Haversine ratios.
   - Economic Interactions: `distance * quote_signal` (baseline quote anchor), `distance * market_index`, weight-to-distance ratios.
   - Equipment & Temporal: Day-of-week, weekend flags, day-of-year, and equipment type one-hot encodings.

---

## 4. Project Structure

```text
├── train-test.csv                       # Labeled development dataset (48,000 loads)
├── validation.csv                       # Test dataset (12,000 loads)
├── validation-predictions-template.csv  # Required output structure template
├── december-chart-inputs.csv            # 31-day December sensitivity analysis inputs
├── train.py                             # Model training and artifact serialization
├── predict.py                           # Batch prediction & template filling script
├── score.py                             # Official validation & December chart generator
├── ML_Assessment.ipynb                  # Interactive walkthrough notebook (12 sections)
├── report.pdf                           # Executive summary and technical assessment report
├── requirements.txt                     # Project dependencies
└── scorer_results/
    └── candidate_december.png           # Generated December 2025 sensitivity chart
```

---

## 5. Official Scorer Verification

Running `score.py` confirms 100% compliance:
- Validated **12,000** final predictions in `validation_predictions.csv`
- Validated **31** fixed December predictions in `december-chart-inputs.csv`
- Verified `scorer_results/candidate_december.png` chart generated without errors
