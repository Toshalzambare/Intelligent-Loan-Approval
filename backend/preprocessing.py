import joblib
import pandas as pd
import numpy as np
import os
from utils.validation import LoanApplication

# Paths to models
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
ENCODER_PATH = os.path.join(MODEL_DIR, 'encoder.pkl')
FEATURES_PATH = os.path.join(MODEL_DIR, 'feature_columns.pkl')

def load_preprocessors():
    if not (os.path.exists(SCALER_PATH) and os.path.exists(ENCODER_PATH) and os.path.exists(FEATURES_PATH)):
        return None, None, None
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    return scaler, encoder, feature_columns

scaler, encoder, feature_columns = load_preprocessors()

def preprocess_input(app: LoanApplication) -> np.ndarray:
    if scaler is None or encoder is None or feature_columns is None:
        raise RuntimeError("Preprocessing models are not loaded. Run the Colab notebook first and place .pkl files in backend/model/")

    # Convert to dataframe
    df = pd.DataFrame([app.model_dump()])

    numerical_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
    categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

    # Scale
    num_scaled = scaler.transform(df[numerical_cols])

    # Encode
    cat_encoded = encoder.transform(df[categorical_cols])

    # Combine
    processed_features = np.hstack((num_scaled, cat_encoded))
    
    return processed_features
