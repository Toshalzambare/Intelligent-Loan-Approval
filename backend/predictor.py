import joblib
import os
import numpy as np
import shap
from utils.validation import PredictionResponse

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'loan_model.pkl')
FEATURES_PATH = os.path.join(os.path.dirname(__file__), 'model', 'feature_columns.pkl')

def load_model_and_features():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
        return None, None
    return joblib.load(MODEL_PATH), joblib.load(FEATURES_PATH)

model, feature_columns = load_model_and_features()
explainer = shap.TreeExplainer(model) if model is not None else None

def get_prediction(features: np.ndarray) -> PredictionResponse:
    if model is None:
        raise RuntimeError("ML model not found. Run Colab notebook and save to backend/model/")
    
    # Predict (Assume 1 is Default/Rejected and 0 is Approved based on typical credit risk dataset)
    # Wait, in standard Kaggle credit risk: 0 = non-default (Approved), 1 = default (Rejected)
    pred_val = model.predict(features)[0]
    prob_default = model.predict_proba(features)[0][1]

    # Calculate Probability of Approval (which is 1 - Probability of Default)
    approval_prob = (1 - prob_default) * 100

    if pred_val == 1:
        prediction = "Rejected"
    else:
        if approval_prob > 80:
            prediction = "Approved"
        else:
            prediction = "Flagged"

    # Risk level formulation
    if approval_prob >= 80:
        risk_level = "Low Risk"
    elif approval_prob >= 50:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"

    # SHAP Explainability
    explainability_dict = {}
    if explainer is not None and feature_columns is not None:
        shap_vals = explainer.shap_values(features)
        
        # Random Forest returns a list for each class. We use class 1 (default risk).
        # We negate it so positive impact means "Approval impact"
        if isinstance(shap_vals, list):
            instance_shap = shap_vals[1][0] * -1
        else:
            if len(shap_vals.shape) == 3:
                instance_shap = shap_vals[0, :, 1] * -1
            else:
                instance_shap = shap_vals[0] * -1

        # Sort features by absolute impact
        feature_impact = zip(feature_columns, instance_shap)
        sorted_impact = sorted(feature_impact, key=lambda x: abs(x[1]), reverse=True)
        
        # Take top 6
        for f_name, imp in sorted_impact[:6]:
            explainability_dict[f_name] = float(imp)

    return PredictionResponse(
        prediction=prediction,
        approval_probability=round(approval_prob, 2),
        risk_level=risk_level,
        explainability=explainability_dict
    )
