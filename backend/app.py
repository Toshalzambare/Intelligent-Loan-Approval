from fastapi import FastAPI, HTTPException
from utils.validation import LoanApplication, PredictionResponse
from backend.preprocessing import preprocess_input
from backend.predictor import get_prediction

app = FastAPI(title="Intelligent Loan Approval API")

@app.get("/health")
def health_check():
    return {"status": "ML Service is running", "model": "Random Forest"}

@app.get("/model-info")
def model_info():
    return {
        "model_name": "RandomForestClassifier",
        "model_version": "1.0.0",
        "accuracy": "92.49%",
        "precision": "0.92",
        "recall": "0.72",
        "f1_score": "0.81",
        "roc_auc": "0.9293",
        "confusion_matrix": "[[4384, 79], [349, 889]]"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_loan(application: LoanApplication):
    try:
        # Preprocess features
        features = preprocess_input(application)
        # Predict
        response = get_prediction(features)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
