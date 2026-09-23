# Intelligent Loan Approval System

This project is an end-to-end Machine Learning web application designed to assess credit risk and predict whether a loan application should be approved or rejected. It features a robust predictive model, a FastAPI backend for real-time inference, and an interactive Streamlit frontend that provides AI explainability for each decision.

## 🧠 Machine Learning Architecture

### 1. Data Cleaning & Preprocessing
The model was trained on a comprehensive Credit Risk Dataset (from Kaggle). The preprocessing pipeline involves several crucial steps to ensure high data quality:
- **Missing Values & Outliers:** Null values and duplicates were removed. Extreme outliers in employment length and age were filtered out.
- **Numerical Scaling:** Used `StandardScaler` to normalize continuous features (e.g., `person_income`, `loan_amnt`, `person_emp_length`) to have a mean of 0 and standard deviation of 1.
- **Categorical Encoding:** Used `OneHotEncoder` to transform categorical variables (e.g., `loan_intent`, `person_home_ownership`, `loan_grade`) into a machine-readable format.
- **Class Imbalance Handling:** Since loan defaults (rejections) are traditionally underrepresented, we used **SMOTE (Synthetic Minority Over-sampling Technique)** to synthetically balance the training dataset.

### 2. Model Specifications
We trained a **Random Forest Classifier**. Random Forests were chosen because they naturally handle non-linear relationships, are robust to outliers, and provide excellent feature importance metrics (Gini Index).
* **Algorithm:** `RandomForestClassifier`
* **Hyperparameters:** `max_depth=15`, `random_state=42`
* **Explainability Engine:** `shap.TreeExplainer`

### 3. Libraries Used
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn`, `imbalanced-learn` (SMOTE)
* **Explainable AI (XAI):** `shap`
* **Backend Framework:** `FastAPI`, `uvicorn`, `pydantic`
* **Frontend Framework:** `streamlit`, `plotly`

### 4. Model Performance Metrics
The model was rigorously evaluated on a holdout test set to ensure generalizability.
* **Overall Accuracy:** `92.49%`
* **Precision (Class 1 - Default):** `0.92` (When the model predicts a default, it is correct 92% of the time)
* **Recall (Class 1 - Default):** `0.72` (The model successfully identifies 72% of all actual defaults)
* **F1-Score (Class 1 - Default):** `0.81`
* **ROC-AUC Score:** `0.9293` (Excellent capability to distinguish between approvals and rejections)

---

## ⚙️ How the Prediction Pipeline Works

1. **User Input:** The loan officer or applicant enters demographic and financial details into the Streamlit UI.
2. **API Request:** Streamlit sends a JSON payload to the FastAPI backend via a `POST /predict` request.
3. **Pydantic Validation:** The backend uses Pydantic models to strictly type-check and validate the incoming data (e.g., ensuring age > 18).
4. **Data Transformation:** The raw JSON is converted into a Pandas DataFrame and passed through the pre-fitted `StandardScaler` and `OneHotEncoder`.
5. **Inference:** The processed array is passed to `model.predict_proba()` to generate a probability score. If the default probability > 50%, the loan is rejected.
6. **SHAP Explainability:** The array is passed to `shap.TreeExplainer`, generating SHAP values that indicate exactly *which* features contributed positively or negatively to the final decision.
7. **Response:** FastAPI returns the prediction, confidence score, and SHAP feature impacts to Streamlit, which dynamically renders Plotly charts to explain the decision to the user.

---

## 🚀 Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/Toshalzambare/Intelligent-Loan-Approval.git
cd Intelligent-Loan-Approval
```

**2. Create a virtual environment**
```bash
python -m venv venv
# Windows:
.\\venv\\Scripts\\activate
# Mac/Linux:
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the API Backend**
Open a terminal and start the FastAPI server:
```bash
uvicorn backend.app:app --reload
```

**5. Run the Streamlit Frontend**
Open a second terminal, activate the environment, and start Streamlit:
```bash
streamlit run frontend/streamlit_app.py
```
