import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import json
import sys
import os
from datetime import datetime

# Add project root to sys.path so 'utils' can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.visualization import generate_portfolio_health_chart, generate_decision_activity_chart

API_URL = "http://localhost:8000"

# -----------------------------------------------------------------------------
# 1. TYPES & STATE INITIALIZATION
# -----------------------------------------------------------------------------
INITIAL_APPLICANTS = [
    {"id": "LN-24891", "name": "Jordan Matthews", "amount": 32000, "status": "Approved", "income": 96000, "credit_score": 742, "submitted": "Today"},
    {"id": "LN-24890", "name": "Maya Chen", "amount": 8600, "status": "Approved", "income": 110000, "credit_score": 780, "submitted": "Today"},
    {"id": "LN-24887", "name": "Elias Brooks", "amount": 41000, "status": "Flagged", "income": 85000, "credit_score": 640, "submitted": "Yesterday"},
]

CURRENT_USER = {
    "name": "Alex Kim",
    "role": "Senior Analyst",
    "email": "alex.kim@northstar.com",
    "initials": "AK",
    "color": "#059669"
}

def initialize_session_state() -> None:
    if "applicants" not in st.session_state:
        st.session_state.applicants = INITIAL_APPLICANTS
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = True
    if "test_data" not in st.session_state:
        st.session_state.test_data = {}

# -----------------------------------------------------------------------------
# 2. STYLES & LAYOUT SETUP
# -----------------------------------------------------------------------------
def inject_custom_styles() -> None:
    st.markdown("""
    <style>
        .stApp { background-color: #0b101d; color: #e2e8f0; }
        [data-testid="stSidebar"] { background-color: #0d1527; border-right: 1px solid #1e293b; }
        .metric-card { background-color: #131d33; border: 1px solid #1e2d4a; border-radius: 12px; padding: 18px; margin-bottom: 12px; }
        .card-title { color: #94a3b8; font-size: 0.85rem; font-weight: 500; margin-bottom: 4px; }
        .card-value { color: #ffffff; font-size: 1.8rem; font-weight: 700; margin: 4px 0; }
        .card-delta-pos { color: #10b981; font-size: 0.8rem; font-weight: 600; }
        .card-subtext { color: #64748b; font-size: 0.75rem; }
        .stButton>button { background-color: #e5a138; color: #0b101d; font-weight: 600; border-radius: 8px; border: none; padding: 0.5rem 1rem; }
        .stButton>button:hover { background-color: #d99126; color: #0b101d; }
        .decision-box-approve { background-color: #062b22; border: 1px solid #059669; border-radius: 12px; padding: 20px; color: #ffffff; margin-top: 15px; }
        .decision-box-reject { background-color: #450a0a; border: 1px solid #dc2626; border-radius: 12px; padding: 20px; color: #ffffff; margin-top: 15px; }
        .decision-box-flagged { background-color: #422006; border: 1px solid #d97706; border-radius: 12px; padding: 20px; color: #ffffff; margin-top: 15px; }
    </style>
    """, unsafe_allow_html=True)

def render_metric_card(title: str, delta: str, value: str, subtext: str) -> str:
    return f"""
    <div class="metric-card">
        <div class="card-title">{title}</div>
        <div class="card-delta-pos">{delta}</div>
        <div class="card-value">{value}</div>
        <div class="card-subtext">{subtext}</div>
    </div>
    """

# -----------------------------------------------------------------------------
# 3. API INTEGRATION
# -----------------------------------------------------------------------------
def predict_loan(data: dict):
    try:
        response = requests.post(f"{API_URL}/predict", json=data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.text}")
            return None
    except Exception as e:
        st.error(f"Failed to connect to API: {e}. Is FastAPI running?")
        return None

# -----------------------------------------------------------------------------
# 4. VIEW CONTROLLERS
# -----------------------------------------------------------------------------
def render_home() -> None:
    st.markdown("## Intelligent Loan Approval & Credit Risk Prediction")
    st.write("**Problem Statement**: Financial institutions receive thousands of loan applications and need to evaluate the applicant's creditworthiness efficiently.")
    st.write("**Project Objective**: Predict whether a loan application should be Approved, Rejected, or flagged as High Risk based on financial and demographic info.")
    
    st.markdown("### ML Workflow")
    st.code("""
    STREAMLIT -> Applicant Information -> "Predict Risk"
    ↓
    PYTHON API (FastAPI) -> Load .pkl models
    ↓
    Prediction Engine
    ↓
    Decision | Probability | Risk Level
    """)
    st.markdown("### Technology Stack")
    st.write("- **Backend**: FastAPI, Python, scikit-learn, joblib")
    st.write("- **Frontend**: Streamlit, Plotly")
    st.write("- **Model**: Random Forest Classifier (~90% Accuracy)")

def render_new_assessment() -> None:
    st.caption("CREDIT DECISIONING / INTAKE")
    st.markdown("## New Assessment (Loan Prediction)")
    st.write("Review an applicant's profile and generate an explainable decision.")
    st.markdown("---")

    col_test1, col_test2 = st.columns(2)
    with col_test1:
        if st.button("Load Positive Example (Approved)", use_container_width=True):
            st.session_state.test_data = {
                "name": "Sarah Jenkins", "age": 35, "income": 95000, "emp": 8.0,
                "amount": 15000, "rate": 7.5, "hist": 10, "home": "MORTGAGE",
                "intent": "PERSONAL", "grade": "A", "default": "N"
            }
            st.rerun()
    with col_test2:
        if st.button("Load Negative Example (Rejected)", use_container_width=True):
            st.session_state.test_data = {
                "name": "Michael Vance", "age": 22, "income": 30000, "emp": 1.0,
                "amount": 25000, "rate": 16.5, "hist": 2, "home": "RENT",
                "intent": "VENTURE", "grade": "E", "default": "Y"
            }
            st.rerun()

    t = st.session_state.test_data

    with st.form("assessment_form"):
        full_name = st.text_input("Full name", value=t.get("name", "Jordan Matthews"))
        c1, c2, c3 = st.columns(3)
        with c1:
            person_age = st.number_input("Age", value=t.get("age", 28), min_value=18)
            person_income = st.number_input("Annual income ($)", value=t.get("income", 96000))
            person_emp_length = st.number_input("Employment length (years)", value=float(t.get("emp", 6.0)))
        with c2:
            loan_amnt = st.number_input("Requested amount ($)", value=t.get("amount", 32000))
            loan_int_rate = st.number_input("Interest Rate (%)", value=float(t.get("rate", 10.5)))
            cb_person_cred_hist_length = st.number_input("Credit History Length (years)", value=t.get("hist", 4))
        with c3:
            person_home_ownership = st.selectbox("Home ownership", ["MORTGAGE", "OWN", "RENT", "OTHER"], index=["MORTGAGE", "OWN", "RENT", "OTHER"].index(t.get("home", "MORTGAGE")))
            loan_intent = st.selectbox("Loan intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"], index=["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"].index(t.get("intent", "PERSONAL")))
            loan_grade = st.selectbox("Loan grade", ["A", "B", "C", "D", "E", "F", "G"], index=["A", "B", "C", "D", "E", "F", "G"].index(t.get("grade", "A")))
            cb_person_default_on_file = st.selectbox("Historical Default?", ["N", "Y"], index=["N", "Y"].index(t.get("default", "N")))
            
        submit_btn = st.form_submit_button("Run Credit Assessment")

    if submit_btn:
        loan_percent_income = loan_amnt / person_income if person_income > 0 else 0
        
        payload = {
            "person_age": person_age,
            "person_income": person_income,
            "person_home_ownership": person_home_ownership,
            "person_emp_length": person_emp_length,
            "loan_intent": loan_intent,
            "loan_grade": loan_grade,
            "loan_amnt": loan_amnt,
            "loan_int_rate": loan_int_rate,
            "loan_percent_income": loan_percent_income,
            "cb_person_default_on_file": cb_person_default_on_file,
            "cb_person_cred_hist_length": cb_person_cred_hist_length
        }
        
        with st.spinner("Analyzing risk..."):
            result = predict_loan(payload)
            
        if result:
            status = result["prediction"]
            prob = result["approval_probability"]
            risk = result["risk_level"]
            explainability = result.get("explainability", {})
            
            box_class = "decision-box-approve" if status == "Approved" else "decision-box-reject" if status == "Rejected" else "decision-box-flagged"
            
            st.markdown(f"""
            <div class="{box_class}">
                <h3>RECOMMENDED DECISION: {status}</h3>
                <p>Applicant: <strong>{full_name}</strong></p>
                <hr>
                <div style="display: flex; justify-content: space-between; text-align: center;">
                    <div><strong>Approval Probability</strong><br>{prob}%</div>
                    <div><strong>Risk Level</strong><br>{risk}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show dynamic SHAP explainability
            if explainability:
                st.markdown("#### Key Factors (Explainability)")
                features = list(explainability.keys())
                importance = list(explainability.values())
                colors = ['#10b981' if i > 0 else '#ef4444' for i in importance]
                
                fig = go.Figure(go.Bar(x=importance, y=features, orientation='h', marker_color=colors))
                fig.update_layout(
                    title="Impact on Approval Score", 
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)', 
                    font=dict(color='white'),
                    margin=dict(l=0, r=0, t=30, b=0),
                    height=250
                )
                st.plotly_chart(fig, use_container_width=True)
                
            # Save to queue
            st.session_state.applicants.insert(0, {
                "id": f"LN-{24900 + len(st.session_state.applicants)}",
                "name": full_name,
                "amount": loan_amnt,
                "status": status,
                "income": person_income,
                "credit_score": 700,
                "submitted": "Just now"
            })

def render_risk_analysis() -> None:
    st.markdown("## Applicant Risk Analysis")
    st.write("Visualizations analyzing the broader portfolio and risk factors based on current sessions predictions.")
    
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.markdown("### Portfolio health")
        fig_donut = generate_portfolio_health_chart(st.session_state.applicants)
        st.plotly_chart(fig_donut, use_container_width=True)
    
    with col_right:
        st.markdown("### Decision Activity (Live Session)")
        fig_activity = generate_decision_activity_chart(st.session_state.applicants)
        st.plotly_chart(fig_activity, use_container_width=True)

def render_model_performance() -> None:
    st.markdown("## Model Performance & Explainability")
    try:
        health_res = requests.get(f"{API_URL}/model-info").json()
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(render_metric_card("Accuracy", "", health_res['accuracy'], "Test Set"), unsafe_allow_html=True)
        with c2:
            st.markdown(render_metric_card("Precision", "", health_res['precision'], "Test Set"), unsafe_allow_html=True)
        with c3:
            st.markdown(render_metric_card("F1 Score", "", health_res['f1_score'], "Test Set"), unsafe_allow_html=True)
        with c4:
            st.markdown(render_metric_card("ROC-AUC", "", health_res['roc_auc'], "Test Set"), unsafe_allow_html=True)

        st.markdown("### Confusion Matrix")
        st.code(health_res['confusion_matrix'])
        
    except:
        st.warning("Could not fetch live model info from API. Ensure FastAPI is running.")
    


# -----------------------------------------------------------------------------
# 5. MAIN ROUTER
# -----------------------------------------------------------------------------
NAV_ROUTES = {
    "1. Home": render_home,
    "2. Loan Prediction": render_new_assessment,
    "3. Risk Analysis": render_risk_analysis,
    "4. Model Performance": render_model_performance,
}

def main() -> None:
    st.set_page_config(page_title="CrediMetrics - Risk Operations", layout="wide")
    inject_custom_styles()
    initialize_session_state()

    with st.sidebar:
        st.markdown("### **CrediMetrics**")
        st.caption("WORKSPACE")
        selected_route = st.radio("Navigation", options=list(NAV_ROUTES.keys()), label_visibility="collapsed")
        
    route_fn = NAV_ROUTES.get(selected_route, render_home)
    route_fn()

if __name__ == "__main__":
    main()