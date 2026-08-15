# Freight Rate Prediction Challenge (Spotter ML Assessment)

**Candidate:** Santosh Adapa  
**Model:** LightGBM Ensemble  
**Validation Metric:** 6.39% MAPE (Holdout: Sep-Oct 2025)

---

## 📌 Project Overview
This repository contains the end-to-end machine learning pipeline for predicting freight rates. The model is built to handle the high cardinality and geographic complexities of the freight market while remaining robust to seasonality and macroeconomic concept drift.

## 📂 Repository Structure
```text
├── train.py                          # Main ML pipeline (Feature Engineering, Training, Inference)
├── analysis.py                       # Exploratory Data Analysis (EDA) scripts
├── Spotter_ML_Assessment_Report.md   # Detailed PDF Draft (Validation strategy & Insights)
├── requirements.txt                  # Python dependencies
├── candidate_december.png            # Generated output chart
└── readme.md                         # Project documentation
```

---

## ⚙️ Environment Setup

Ensure you have Python 3.9+ installed. To install the required dependencies, run the following command in your terminal or VS Code:

```bash
python -m pip install -r requirements.txt
```

---

## 🚀 How to Run the Pipeline

### 1. Execute the Training Pipeline
The entire workflow (target encoding, geographic feature synthesis, and LightGBM ensemble training) is modularized into a single script. Run the following command:

```bash
python train.py
```
**Outputs Generated:**
- `validation_predictions.csv`
- `december-chart-inputs.csv`

### 2. Evaluate the Output
Once the pipeline finishes, use the provided scoring script to validate the CSVs and generate the final visualization:

```bash
python score.py --predictions validation_predictions.csv --december-predictions december-chart-inputs.csv
```

---

## 🔬 Key Engineering Highlights
* **Robust Validation:** Avoided random train/test splits to prevent data leakage. Used a strict time-based holdout (Train: Jan-Aug, Validate: Sep-Oct).
* **Geographic Features:** Engineered Haversine distances and circuity ratios to proxy routing complexity.
* **Target Encoding:** Implemented Bayesian smoothed target encoding to prevent the model from overfitting on rare city-pair routes.
* **Missing Data Imputation:** Synthesized missing December inputs (coordinates, market index) using training-set medians to ensure seamless pipeline execution.

*(Please refer to the enclosed PDF Report for deep-dive technical methodology and the path to production).*
