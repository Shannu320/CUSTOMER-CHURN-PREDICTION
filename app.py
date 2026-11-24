from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load all model artifacts with error handling
try:
    loaded_model = joblib.load('best_model.joblib')
    threshold = joblib.load('threshold.joblib')
    encoders = joblib.load('encoders.joblib')
    scaler = joblib.load('scaler.joblib')
    feature_names = joblib.load('feature_names.joblib')
    num_cols = joblib.load('num_cols.joblib')
    cat_cols = joblib.load('cat_cols.joblib')
    high_value_threshold = joblib.load('high_value_threshold.joblib')
    logger.info("All model artifacts loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model artifacts: {e}")
    raise

app = Flask(__name__)

# Constants for feature engineering
service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
contract_map = {'Month-to-month': 1, 'One year': 2, 'Two year': 3}
payment_risk = {'Electronic check': 3, 'Mailed check': 2, 'Bank transfer (automatic)': 1, 'Credit card (automatic)': 1}


def engineer_features(input_df):
    """Apply the same feature engineering as training. Vectorized for efficiency."""
    df = input_df.copy()

    # 1. Tenure bands
    df['tenure_band'] = pd.cut(df['tenure'],
                                bins=[0, 12, 24, 48, 72],
                                labels=['0-1yr', '1-2yr', '2-4yr', '4+yr'],
                                right=True)

    # 2. Average monthly charges (guard against negative tenure)
    df['avg_monthly_charges'] = df['TotalCharges'] / (df['tenure'].clip(lower=0) + 1)

    # 3. High value customer
    df['high_value_customer'] = (df['TotalCharges'] > high_value_threshold).astype(int)

    # 4. Service count - vectorized
    active_mask = ~df[service_cols].isin(['No', 'No internet service', 'No phone service'])
    df['service_count'] = active_mask.sum(axis=1).astype(int)

    # 5. Contract encoded - handle unknown values
    df['contract_encoded'] = df['Contract'].map(contract_map).fillna(1).astype(int)

    # 6. Payment risk score - handle unknown values
    df['payment_risk_score'] = df['PaymentMethod'].map(payment_risk).fillna(1).astype(int)

    # 7. Senior with dependents
    df['senior_with_dependents'] = ((df['SeniorCitizen'] == 1) & (df['Dependents'] == 'Yes')).astype(int)

    # 8. Paperless + electronic check
    df['paperless_electronic'] = ((df['PaperlessBilling'] == 'Yes') & (df['PaymentMethod'] == 'Electronic check')).astype(int)

    return df


def make_prediction(input_data):
    """Make churn prediction with optimal threshold."""
    # Validate input
    required_fields = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                       'MonthlyCharges', 'TotalCharges']
    missing = [f for f in required_fields if f not in input_data]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    # Validate ranges
    if not 0 <= input_data['tenure'] <= 72:
        raise ValueError("Tenure must be between 0 and 72")
    if input_data['MonthlyCharges'] < 0 or input_data['TotalCharges'] < 0:
        raise ValueError("Charges must be non-negative")

    input_df = pd.DataFrame([input_data])

    # Apply feature engineering
    input_df = engineer_features(input_df)

    # Encode categorical features with robust unseen handling
    for col, encoder in encoders.items():
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(str)
            # Map unseen categories to most frequent class (index 0 after fit)
            known_classes = set(encoder.classes_)
            default_class = encoder.classes_[0]
            input_df[col] = input_df[col].map(lambda x: x if x in known_classes else default_class)
            input_df[col] = encoder.transform(input_df[col])

    # Validate all numerical columns exist
    missing_num = [c for c in num_cols if c not in input_df.columns]
    if missing_num:
        raise ValueError(f"Missing numerical columns after feature engineering: {missing_num}")

    # Scale numerical features
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # Ensure column order matches training - add missing as NaN then fill
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0  # or appropriate default
    input_df = input_df[feature_names]

    # Predict
    probability = loaded_model.predict_proba(input_df)[0, 1]
    prediction = 1 if probability >= threshold else 0

    return "Churn" if prediction == 1 else "No Churn", probability


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probability = None
    error = None

    if request.method == 'POST':
        try:
            input_data = {
                'gender': request.form['gender'],
                'SeniorCitizen': int(request.form['SeniorCitizen']),
                'Partner': request.form['Partner'],
                'Dependents': request.form['Dependents'],
                'tenure': int(request.form['tenure']),
                'PhoneService': request.form['PhoneService'],
                'MultipleLines': request.form['MultipleLines'],
                'InternetService': request.form['InternetService'],
                'OnlineSecurity': request.form['OnlineSecurity'],
                'OnlineBackup': request.form['OnlineBackup'],
                'DeviceProtection': request.form['DeviceProtection'],
                'TechSupport': request.form['TechSupport'],
                'StreamingTV': request.form['StreamingTV'],
                'StreamingMovies': request.form['StreamingMovies'],
                'Contract': request.form['Contract'],
                'PaperlessBilling': request.form['PaperlessBilling'],
                'PaymentMethod': request.form['PaymentMethod'],
                'MonthlyCharges': float(request.form['MonthlyCharges']),
                'TotalCharges': float(request.form['TotalCharges']),
            }

            prediction, probability = make_prediction(input_data)
            logger.info(f"Prediction: {prediction}, Probability: {probability:.4f}")

        except ValueError as e:
            error = f"Invalid input: {str(e)}"
            logger.warning(f"Validation error: {e}")
        except Exception as e:
            error = f"Prediction error: {str(e)}"
            logger.exception("Unexpected error during prediction")

    return render_template('index.html',
                           prediction=prediction,
                           probability=probability,
                           error=error,
                           threshold=threshold)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)