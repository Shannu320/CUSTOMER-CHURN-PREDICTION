# Customer Churn Prediction & Retention Intelligence

**Domain:** Telecom / Subscription Business Analytics  
**Status:** Production-ready ML pipeline with Flask web interface

---

## Project Overview

End-to-end predictive analytics solution to identify customers likely to discontinue service, understand churn drivers, and enable data-driven retention actions.

**Dataset:** Telco Customer Churn (7,043 customers, 20 features)  
**Target:** Binary classification (Churn: Yes/No)  
**Best Model:** XGBoost (tuned via stratified 5-fold CV + GridSearchCV)  
**Decision Threshold:** Optimized for retention team capacity (20%)

---

## Core Functionality

### 1. Data Ingestion & Quality (`notebook: Cells 3-9`)
- Loads raw CSV, handles whitespace in `TotalCharges`, converts to numeric
- Drops non-predictive `customerID`
- Validates missing values, distributions, target balance (26.5% churn rate)

### 2. Exploratory Data Analysis (`notebook: Cells 11-18`)
- Statistical summaries (mean, median, std, IQR) for numerical features
- Distribution plots + boxplots for tenure, monthly charges, total charges
- Correlation matrix for numerical features
- Categorical feature distributions split by churn (18 variables)

### 3. Statistical Analysis & Feature Engineering (`notebook: Cells 19-22`)
- **IQR outlier detection** on numerical features
- **8 engineered features:**
  - `tenure_band` — 4 ordinal bins (0-1yr, 1-2yr, 2-4yr, 4+yr)
  - `avg_monthly_charges` — TotalCharges / (tenure + 1)
  - `high_value_customer` — Top 25% by TotalCharges (binary)
  - `service_count` — Active services (0-9)
  - `contract_encoded` — Ordinal: Month-to-month=1, One year=2, Two year=3
  - `payment_risk_score` — Electronic check=3, Mailed check=2, Auto-pay=1
  - `senior_with_dependents` — Interaction flag
  - `paperless_electronic` — High-risk combo flag

### 4. Predictive Modeling (`notebook: Cells 29-32`)
**5 classifiers trained with hyperparameter tuning:**
| Model | Key Config |
|-------|------------|
| Logistic Regression | `class_weight='balanced', max_iter=1000` |
| Decision Tree | `class_weight='balanced'` |
| Random Forest | `class_weight='balanced', n_jobs=-1` |
| Neural Network (MLP) | `early_stopping=True, max_iter=500` |
| **XGBoost** | `scale_pos_weight` tuned, `eval_metric='logloss'` |

**Validation:** Stratified 5-fold CV (`StratifiedKFold`) on original (non-SMOTE) training data  
**Class Imbalance:** SMOTE applied to training fold only  
**Optimization Target:** Recall (catch churners)

### 5. Threshold Tuning (`notebook: Cells 35-39`)
- Precision-Recall curves for all models
- **Capacity-aligned threshold:** Selects threshold yielding ~20% predicted positives (retention team capacity)
- **Best-F1 threshold:** Alternative operating point
- Default 0.5 threshold compared for reference

### 6. Model Evaluation (`notebook: Cells 40-45`)
**Metrics per model at capacity threshold:**
- Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Confusion matrices
- Classification reports

**Feature Importance:**
- Permutation importance (model-agnostic, 10 repeats) for all 5 models
- Consensus ranking via average rank across models
- SHAP analysis for best model (XGBoost)

### 7. Churn Driver Analysis & Retention Actions (`notebook: Cells 46-48`)
**Top 10 drivers with business context:**
1. Contract Type (Month-to-month: 42% vs 3% for 2-year)
2. Tenure (0-1yr: 47% churn)
3. Payment Method (Electronic check: 45%)
4. Internet Service (Fiber optic: 42%)
5. Online Security (No: 2-3× higher)
6. Tech Support (No: 2-3× higher)
7. Monthly Charges (Above avg: higher churn)
8. Device Protection (No: elevated)

**6 actionable retention strategies** with specific tactics per segment

### 8. Model Persistence (`notebook: Cell 51`)
Saved artifacts for production use:
| File | Purpose |
|------|---------|
| `best_model.joblib` | Trained XGBoost classifier |
| `encoders.joblib` | Dict of LabelEncoders for categorical features |
| `scaler.joblib` | StandardScaler for numerical features |
| `threshold.joblib` | Optimal decision threshold (float) |
| `feature_names.joblib` | Feature column order (list) |
| `num_cols.joblib` / `cat_cols.joblib` | Column type lists |
| `high_value_threshold.joblib` | 75th percentile TotalCharges |

### 9. Churn-Risk Output (`notebook: Cell 54`)
Generates `churn_risk_output.csv` with **all 7,043 customers** scored:
- `churn_probability` — Model probability (0-1)
- `churn_risk_score` — Binary prediction at optimal threshold
- `risk_category` — Low / Medium / High (probability tertiles)
- `revenue_at_risk_monthly` — MonthlyCharges × probability
- `revenue_at_risk_annual` — × 12

**Summary metrics:**
- Total Monthly Revenue at Risk: ~$123K
- Total Annual Revenue at Risk: ~$1.48M
- High-Risk Customers: ~1,200 (18%)

### 10. Flask Web Application (`app.py` + `templates/index.html`)
**Production-ready inference API:**
- `/` — GET: form UI, POST: prediction with validation
- Input validation (ranges, required fields, type coercion)
- Feature engineering replicates training pipeline exactly
- Robust unseen-category handling in encoders
- Returns: Prediction (Churn/No Churn) + Probability + Threshold

**Run locally:**
```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

---

## Repository Structure

```
churn-app/
├── Customer_Churn_Prediction_Enhanced.ipynb   # Complete ML pipeline
├── app.py                                      # Flask inference server
├── templates/
│   └── index.html                              # Web prediction form
├── requirements.txt                            # Dependencies
├── churn_risk_output.csv                       # Scored customer data (key deliverable)
├── best_model.joblib                           # Trained model
├── encoders.joblib                             # Categorical encoders
├── scaler.joblib                               # Numerical scaler
├── threshold.joblib                            # Decision threshold
├── feature_names.joblib                        # Feature order
├── num_cols.joblib / cat_cols.joblib           # Column lists
├── high_value_threshold.joblib                 # HV threshold
└── README.md                                   # This file
```

---

## Key Results

| Metric | Value |
|--------|-------|
| **Best Model** | XGBoost |
| **Optimal Threshold** | ~0.32 |
| **Recall @ Capacity** | ~0.82 |
| **Precision @ Capacity** | ~0.48 |
| **F1-Score** | ~0.61 |
| **ROC-AUC** | ~0.86 |
| **Overall Churn Rate** | 26.5% |
| **Monthly Revenue at Risk** | ~$123K |
| **Annual Revenue at Risk** | ~$1.48M |
| **High-Risk Customers** | ~1,200 (18%) |

---

## Programmatic Usage

```python
import joblib
import pandas as pd

# Load artifacts
model = joblib.load('best_model.joblib')
encoders = joblib.load('encoders.joblib')
scaler = joblib.load('scaler.joblib')
threshold = joblib.load('threshold.joblib')
feature_names = joblib.load('feature_names.joblib')
num_cols = joblib.load('num_cols.joblib')
cat_cols = joblib.load('cat_cols.joblib')
high_value_threshold = joblib.load('high_value_threshold.joblib')

# Prepare input (raw features matching training schema)
input_data = {
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 1,
    'PhoneService': 'No',
    'MultipleLines': 'No phone service',
    'InternetService': 'DSL',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 70.35,
    'TotalCharges': 29.85
}

# Feature engineering (must match training)
def engineer_features(df):
    df = df.copy()
    df['tenure_band'] = pd.cut(df['tenure'], bins=[0,12,24,48,72], labels=['0-1yr','1-2yr','2-4yr','4+yr'], right=True)
    df['avg_monthly_charges'] = df['TotalCharges'] / (df['tenure'].clip(lower=0) + 1)
    df['high_value_customer'] = (df['TotalCharges'] > high_value_threshold).astype(int)
    service_cols = ['PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
    active = ~df[service_cols].isin(['No','No internet service','No phone service'])
    df['service_count'] = active.sum(axis=1).astype(int)
    df['contract_encoded'] = df['Contract'].map({'Month-to-month':1,'One year':2,'Two year':3}).fillna(1).astype(int)
    df['payment_risk_score'] = df['PaymentMethod'].map({'Electronic check':3,'Mailed check':2,'Bank transfer (automatic)':1,'Credit card (automatic)':1}).fillna(1).astype(int)
    df['senior_with_dependents'] = ((df['SeniorCitizen']==1) & (df['Dependents']=='Yes')).astype(int)
    df['paperless_electronic'] = ((df['PaperlessBilling']=='Yes') & (df['PaymentMethod']=='Electronic check')).astype(int)
    return df

# Predict
input_df = pd.DataFrame([input_data])
input_df = engineer_features(input_df)

# Encode categoricals
for col, encoder in encoders.items():
    if col in input_df.columns:
        input_df[col] = input_df[col].astype(str)
        known = set(encoder.classes_)
        default = encoder.classes_[0]
        input_df[col] = input_df[col].apply(lambda x: x if x in known else default)
        input_df[col] = encoder.transform(input_df[col])

# Scale numericals
input_df[num_cols] = scaler.transform(input_df[num_cols])

# Align columns
for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[feature_names]

# Inference
prob = model.predict_proba(input_df)[0, 1]
pred = "Churn" if prob >= threshold else "No Churn"
print(f"Prediction: {pred}, Probability: {prob:.4f}, Threshold: {threshold:.4f}")
```

---

## Dependencies

```
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.2.0
xgboost>=1.7.0
imbalanced-learn>=0.10.0
joblib>=1.2.0
flask>=2.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
shap>=0.41.0
```

Install: `pip install -r requirements.txt`

---

## BI Integration (Power BI / Tableau)

The `churn_risk_output.csv` is BI-ready. Key measures to create:

**DAX (Power BI) / Calculated Fields (Tableau):**
```dax
Churn Rate = DIVIDE(COUNTROWS(FILTER(ChurnData, ChurnData[Churn]="Yes")), COUNTROWS(ChurnData))

Monthly Revenue at Risk = SUMX(ChurnData, ChurnData[MonthlyCharges] * ChurnData[churn_probability])

Annual Revenue at Risk = [Monthly Revenue at Risk] * 12

High Risk Customers = CALCULATE(COUNTROWS(ChurnData), ChurnData[risk_category]="High")

High Risk Revenue = CALCULATE(SUMX(ChurnData, ChurnData[MonthlyCharges] * ChurnData[churn_probability]), ChurnData[risk_category]="High")

Retention ROI = DIVIDE(([High Risk Revenue] * 0.30 * 12) - ([High Risk Customers] * 50), [High Risk Customers] * 50)
```

**Recommended dashboard pages:**
1. **Executive KPIs** — Churn rate, revenue at risk, high-risk count, ROI
2. **Churn Drivers** — Bar charts by Contract, Payment, Internet, Tenure
3. **Risk Segmentation** — Treemap (risk category) + Scatter (charges vs probability)
4. **High-Risk Table** — Top 50 customers sortable by probability
5. **Revenue Trends** — Dual-axis by Tenure Band / Payment Method
6. **Retention Priority** — Revenue at risk by actionable segments

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Recall-optimized CV** | Business cost of missing churner > false alarm |
| **SMOTE on train only** | Prevents data leakage; test reflects real distribution |
| **Capacity-aligned threshold** | Model output actionable by retention team size |
| **Permutation importance** | Model-agnostic, works for all 5 classifiers |
| **Feature engineering in app.py** | Ensures training/serving parity |
| **Unseen category → most frequent** | Robust production handling |

---

## Deliverables Checklist

- [x] Clean dataset (`churn_risk_output.csv`)
- [x] EDA notebook (`Customer_Churn_Prediction_Enhanced.ipynb`)
- [x] Trained model + artifacts (7 `.joblib` files)
- [x] Evaluation report (notebook Cells 40-48)
- [x] Churn-risk output (scored CSV)
- [x] Flask web app (`app.py` + `templates/index.html`)
- [x] BI measure specifications (this README)

---

## License

Educational / portfolio project. Dataset: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM Sample Data).