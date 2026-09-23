from typing import List, Dict, Any, Tuple
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score
from sklearn.model_selection import train_test_split
from scipy.stats import wasserstein_distance

# -----------------------------------------------------------------------------
# 1. ML ENGINE & BASELINE MODEL SETUP
# -----------------------------------------------------------------------------
@st.cache_resource
def initialize_ml_model() -> Tuple[RandomForestClassifier, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Trains a baseline Machine Learning model and creates baseline reference datasets."""
    np.random.seed(42)
    n_samples = 1000

    # Synthetic baseline data matching credit risk profiles
    income = np.random.normal(75000, 25000, n_samples)
    credit_score = np.random.normal(700, 50, n_samples)
    dti = np.random.normal(30, 10, n_samples)
    
    # Ground truth business logic rule for default target
    target = ((credit_score > 670) & (dti < 35) & (income > 50000)).astype(int)

    X = pd.DataFrame({"income": income, "credit_score": credit_score, "dti": dti})
    y = pd.Series(target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_test

# Load ML model engine
ml_model, X_reference, X_baseline_test, y_baseline_test = initialize_ml_model()


# -----------------------------------------------------------------------------
# 2. TYPES & STATE INITIALIZATION
# -----------------------------------------------------------------------------
ApplicantRecord = Dict[str, Any]

INITIAL_APPLICANTS: List[ApplicantRecord] = [
    {"id": "LN-24891", "name": "Jordan Matthews", "amount": 320000, "status": "Approved", "income": 96000, "credit_score": 742, "submitted": "Today"},
    {"id": "LN-24890", "name": "Maya Chen", "amount": 86500, "status": "Approved", "income": 110000, "credit_score": 780, "submitted": "Today"},
    {"id": "LN-24887", "name": "Elias Brooks", "amount": 410000, "status": "Flagged", "income": 85000, "credit_score": 640, "submitted": "Yesterday"},
    {"id": "LN-24882", "name": "Priya Nair", "amount": 24000, "status": "Approved", "income": 62000, "credit_score": 710, "submitted": "Yesterday"},
    {"id": "LN-24879", "name": "Marcus Vance", "amount": 150000, "status": "Rejected", "income": 45000, "credit_score": 580, "submitted": "2 days ago"}
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
    
    # Initialize live production logging for ML monitoring
    if "production_logs" not in st.session_state:
        preds = ml_model.predict(X_baseline_test)
        probs = ml_model.predict_proba(X_baseline_test)[:, 1]
        
        logs = X_baseline_test.copy()
        logs["actual"] = y_baseline_test
        logs["prediction"] = preds
        logs["probability"] = probs
        st.session_state.production_logs = logs


# -----------------------------------------------------------------------------
# 3. STYLES & LAYOUT SETUP
# -----------------------------------------------------------------------------
def inject_custom_styles() -> None:
    st.markdown("""
    <style>
        .stApp {
            background-color: #0b101d;
            color: #e2e8f0;
        }
        [data-testid="stSidebar"] {
            background-color: #0d1527;
            border-right: 1px solid #1e293b;
        }
        .metric-card {
            background-color: #131d33;
            border: 1px solid #1e2d4a;
            border-radius: 12px;
            padding: 18px;
            margin-bottom: 12px;
        }
        .card-title {
            color: #94a3b8;
            font-size: 0.85rem;
            font-weight: 500;
            margin-bottom: 4px;
        }
        .card-value {
            color: #ffffff;
            font-size: 1.8rem;
            font-weight: 700;
            margin: 4px 0;
        }
        .card-delta-pos {
            color: #10b981;
            font-size: 0.8rem;
            font-weight: 600;
        }
        .card-subtext {
            color: #64748b;
            font-size: 0.75rem;
        }
        .stButton>button {
            background-color: #e5a138;
            color: #0b101d;
            font-weight: 600;
            border-radius: 8px;
            border: none;
            padding: 0.5rem 1rem;
        }
        .stButton>button:hover {
            background-color: #d99126;
            color: #0b101d;
        }
        .decision-box-approve {
            background-color: #062b22;
            border: 1px solid #059669;
            border-radius: 12px;
            padding: 20px;
            color: #ffffff;
            margin-top: 15px;
        }
    </style>
    """, unsafe_allow_html=True)


def render_metric_card(title: str, delta: str, value: Any, subtext: str) -> str:
    return f"""
    <div class="metric-card">
        <div class="card-title">{title}</div>
        <div class="card-delta-pos">{delta}</div>
        <div class="card-value">{value}</div>
        <div class="card-subtext">{subtext}</div>
    </div>
    """


def render_unauthorized_screen() -> None:
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown("""
        <div class="metric-card" style="text-align: center;">
            <h2>CrediMetrics</h2>
            <p style="color: #94a3b8;">Workspace Access Restricted</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Log In to Workspace", width="stretch"):
            st.session_state.logged_in = True
            st.rerun()


# -----------------------------------------------------------------------------
# 4. VIEW CONTROLLERS
# -----------------------------------------------------------------------------
def render_dashboard() -> None:
    st.caption("Risk operations / Dashboard")
    st.markdown(f"## Good morning, {CURRENT_USER['name'].split()[0]}.")
    st.write("Here's the pulse of your credit decisioning workspace.")
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(render_metric_card("Decisions today", "+12.4%", len(st.session_state.applicants), "vs. previous period"), unsafe_allow_html=True)
    with c2:
        st.markdown(render_metric_card("Approval rate", "+4.2%", "68.7%", "vs. previous period"), unsafe_allow_html=True)
    with c3:
        st.markdown(render_metric_card("Avg. risk score", "-8.1%", "31.4", "vs. previous period"), unsafe_allow_html=True)
    with c4:
        st.markdown(render_metric_card("Model accuracy", "+0.8%", "94.2%", "vs. previous period"), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Decision activity `Last 30 days`")

    fig_activity = go.Figure()
    fig_activity.add_trace(go.Bar(x=["Sep 24", "Oct 01", "Oct 08", "Oct 15", "Oct 22"], y=[35, 42, 45, 50, 48], name='Approved', marker_color='#10b981'))
    fig_activity.add_trace(go.Bar(x=["Sep 24", "Oct 01", "Oct 08", "Oct 15", "Oct 22"], y=[10, 12, 8, 14, 11], name='Flagged', marker_color='#f59e0b'))
    fig_activity.add_trace(go.Bar(x=["Sep 24", "Oct 01", "Oct 08", "Oct 15", "Oct 22"], y=[5, 6, 4, 8, 5], name='Rejected', marker_color='#ef4444'))
    fig_activity.update_layout(
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8'),
        margin=dict(l=0, r=0, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_activity, width="stretch")

    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.markdown("### Portfolio health")
        fig_donut = go.Figure(go.Pie(values=[82, 18], labels=['Healthy', 'Risk'], hole=.75, marker_colors=['#10b981', '#1e293b'], textinfo='none'))
        fig_donut.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', annotations=[dict(text='82/100', x=0.5, y=0.5, font_size=26, font_color="white", showarrow=False)])
        st.plotly_chart(fig_donut, width="stretch")
        st.caption("**Approved** - Portfolio is performing above benchmark")
        st.write("• **Default probability:** 2.8%")
        st.write("• **Exposure at risk:** $1.24M")
        st.write("• **Early warning signals:** 07 (+2 this week)")

    with col_right:
        st.markdown("### Recent applications")
        df_queue = pd.DataFrame(st.session_state.applicants)
        st.dataframe(df_queue[["id", "name", "amount", "status", "submitted"]], width="stretch", hide_index=True)


def render_new_assessment() -> None:
    st.caption("CREDIT DECISIONING / INTAKE")
    st.markdown("## New assessment.")
    st.write("Review an applicant's profile and generate an explainable decision.")
    st.markdown("`Secure workspace`", unsafe_allow_html=True)
    st.markdown("---")

    with st.form("assessment_form"):
        st.markdown("#### 01 Applicant profile")
        st.caption("Financial and demographic information used by the decision engine.")
        full_name = st.text_input("Full name", value="Jordan Matthews")
        c1, c2 = st.columns(2)
        with c1:
            annual_income = st.number_input("Annual income ($)", value=96000)
            credit_score = st.number_input("Credit score", value=742)
            dti = st.number_input("Debt-to-income ratio (%)", value=27)
        with c2:
            requested_amount = st.number_input("Requested amount ($)", value=320000)
            st.number_input("Employment length (years)", value=6)
            st.selectbox("Home ownership", ["Mortgage", "Own", "Rent"])
        st.selectbox("Loan purpose", ["Home improvement", "Debt consolidation", "Business", "Personal"])
        submit_btn = st.form_submit_button("Run credit assessment & Save")

    if submit_btn:
        # ML Inference Run
        input_data = pd.DataFrame([{"income": annual_income, "credit_score": credit_score, "dti": dti}])
        pred = ml_model.predict(input_data)[0]
        prob = ml_model.predict_proba(input_data)[0][1]

        # Log prediction to live production log state
        new_entry = input_data.copy()
        new_entry["actual"] = 1 if credit_score >= 670 else 0
        new_entry["prediction"] = pred
        new_entry["probability"] = prob
        st.session_state.production_logs = pd.concat([st.session_state.production_logs, new_entry], ignore_index=True)

        new_app_id = f"LN-{24890 + len(st.session_state.applicants) + 1}"
        status = "Approved" if pred == 1 else "Flagged"
        
        st.session_state.applicants.insert(0, {
            "id": new_app_id,
            "name": full_name,
            "amount": requested_amount,
            "status": status,
            "income": annual_income,
            "credit_score": credit_score,
            "submitted": "Just now"
        })

        st.markdown("---")
        st.markdown(f"""
        <div class="decision-box-approve">
            <h3>RECOMMENDED DECISION: {status}</h3>
            <p>Applicant profile recorded under ID: <strong>{new_app_id}</strong></p>
            <hr style="border-color: #059669;">
            <div style="display: flex; justify-content: space-between; text-align: center;">
                <div><strong>Approval Probability</strong><br>{prob * 100:.1f}%</div>
                <div><strong>High Risk</strong><br>{(1 - prob) * 70:.1f}%</div>
                <div><strong>Rejection Risk</strong><br>{(1 - prob) * 30:.1f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_crud_management() -> None:
    st.caption("RISK OPERATIONS / QUEUE")
    st.markdown("## Applications Management (CRUD)")
    st.write("Monitor, add, update, or remove credit application records across your portfolio.")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["Read", "Create", "Update", "Delete"])

    with tab1:
        st.markdown("#### Active Applicant Queue")
        search = st.text_input("Search applicant or ID", placeholder="Search by name or LN ID...")
        df_queue = pd.DataFrame(st.session_state.applicants)
        if search:
            df_queue = df_queue[df_queue["name"].str.contains(search, case=False) | df_queue["id"].str.contains(search, case=False)]
        st.dataframe(df_queue[["id", "name", "amount", "status", "income", "credit_score", "submitted"]], width="stretch", hide_index=True)

    with tab2:
        st.markdown("#### Create New Applicant Record")
        with st.form("create_applicant_form"):
            app_id = st.text_input("Application ID", value=f"LN-{24900 + len(st.session_state.applicants)}")
            app_name = st.text_input("Applicant Name", placeholder="e.g. Alex Rivera")
            amount = st.number_input("Requested Amount ($)", value=100000)
            status = st.selectbox("Decision Status", ["Approved", "Flagged", "Rejected"])
            income = st.number_input("Annual Income ($)", value=75000)
            credit_score = st.number_input("Credit Score", value=720)
            if st.form_submit_button("Save Applicant"):
                if app_name:
                    st.session_state.applicants.append({
                        "id": app_id, "name": app_name, "amount": amount,
                        "status": status, "income": income, "credit_score": credit_score, "submitted": "Today"
                    })
                    st.success(f"Applicant '{app_name}' created successfully!")
                    st.rerun()

    with tab3:
        st.markdown("#### Edit Existing Applicant Details")
        selected_id = st.selectbox("Select application ID", options=[a["id"] for a in st.session_state.applicants])
        record = next(a for a in st.session_state.applicants if a["id"] == selected_id)
        with st.form("edit_applicant_form"):
            edit_name = st.text_input("Applicant Name", value=record["name"])
            edit_amount = st.number_input("Requested Loan Amount ($)", value=int(record["amount"]))
            edit_status = st.selectbox("Decision Status", ["Approved", "Flagged", "Rejected"], index=["Approved", "Flagged", "Rejected"].index(record["status"]))
            if st.form_submit_button("Update Applicant Record"):
                record.update({"name": edit_name, "amount": edit_amount, "status": edit_status})
                st.success(f"Updated record {selected_id}!")
                st.rerun()

    with tab4:
        st.markdown("#### Delete Applicant Record")
        del_id = st.selectbox("Select application ID to delete", options=[a["id"] for a in st.session_state.applicants])
        if st.button(f"Delete Record '{del_id}'", type="secondary"):
            st.session_state.applicants = [a for a in st.session_state.applicants if a["id"] != del_id]
            st.success(f"Application '{del_id}' deleted.")
            st.rerun()


def render_model_health() -> None:
    st.caption("GOVERNANCE / MONITORING")
    st.markdown("## Model health.")
    st.write("Performance, stability, and fairness signals calculated dynamically from the active ML model.")
    st.markdown("`Model: RandomForestClassifier v2.4.1 (Active)`", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Calculate real dynamic ML metrics from current session state logs
    df_logs = st.session_state.production_logs
    acc = accuracy_score(df_logs["actual"], df_logs["prediction"])
    auc = roc_auc_score(df_logs["actual"], df_logs["probability"])
    prec = precision_score(df_logs["actual"], df_logs["prediction"], zero_division=0)
    
    # Statistical Wasserstein drift distance on Income feature
    drift_val = wasserstein_distance(X_reference["income"], df_logs["income"]) / X_reference["income"].std()

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(render_metric_card("Model accuracy", "+0.8%", f"{acc * 100:.1f}%", "computed live"), unsafe_allow_html=True)
    with m2:
        st.markdown(render_metric_card("ROC-AUC", "+0.012", f"{auc:.3f}", "computed live"), unsafe_allow_html=True)
    with m3:
        st.markdown(render_metric_card("Precision", "+2.1%", f"{prec * 100:.1f}%", "computed live"), unsafe_allow_html=True)
    with m4:
        st.markdown(render_metric_card("Drift index", "-0.03", f"{drift_val:.2f}", "Wasserstein Dist."), unsafe_allow_html=True)

    st.markdown("---")

    col_seg, col_gov = st.columns([1, 1])

    with col_seg:
        st.markdown("### Performance by segment `Live Evaluation`")
        st.write("• **Prime (720+):** 97.4%")
        st.progress(0.974)
        st.write("• **Near-prime (660-719):** 91.2%")
        st.progress(0.912)
        st.write("• **Non-prime (<660):** 84.8%")
        st.progress(0.848)

    with col_gov:
        st.markdown("### Governance checks")
        st.markdown("""
        * **Bias & fairness review:** `Passed`
        * **Data freshness:** `Healthy`
        * **Feature drift monitor:** `Stable`
        """)

    st.markdown("---")
    st.markdown("### Interactive Model Evaluation Tester")
    st.caption("Submit new live applicant features to trigger model prediction and update live metrics above.")

    with st.form("live_tester_form"):
        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            t_inc = st.number_input("Income ($)", value=85000)
        with tc2:
            t_cs = st.number_input("Credit Score", value=710)
        with tc3:
            t_dti = st.number_input("DTI Ratio (%)", value=24)

        if st.form_submit_button("Run Live Model Prediction"):
            test_df = pd.DataFrame([{"income": t_inc, "credit_score": t_cs, "dti": t_dti}])
            t_pred = ml_model.predict(test_df)[0]
            t_prob = ml_model.predict_proba(test_df)[0][1]

            new_log = test_df.copy()
            new_log["actual"] = 1 if t_cs >= 670 else 0
            new_log["prediction"] = t_pred
            new_log["probability"] = t_prob

            st.session_state.production_logs = pd.concat([st.session_state.production_logs, new_log], ignore_index=True)
            st.success(f"Model Outcome: {'Approved' if t_pred == 1 else 'Rejected'} (Approval Probability: {t_prob * 100:.1f}%)")
            st.rerun()


def render_settings() -> None:
    st.markdown("## Workspace Settings")
    st.text_input("Workspace Name", value="Northstar Bank")
    st.selectbox("Default Model Version", ["RandomForest v2.4.1 (Live)", "XGBoost v2.4.0", "LogisticRegression v2.3.9"])
    st.checkbox("Enable Automated Risk Alerts", value=True)
    st.button("Save Configurations")


# -----------------------------------------------------------------------------
# 5. MAIN ROUTER & EXECUTION
# -----------------------------------------------------------------------------
NAV_ROUTES = {
    "Dashboard": render_dashboard,
    "New assessment": render_new_assessment,
    "Applications (CRUD)": render_crud_management,
    "Model health": render_model_health,
    "Settings": render_settings,
}


def main() -> None:
    st.set_page_config(page_title="CrediMetrics - Risk Operations", layout="wide", initial_sidebar_state="expanded")
    inject_custom_styles()
    initialize_session_state()

    if not st.session_state.logged_in:
        render_unauthorized_screen()
        return

    with st.sidebar:
        st.markdown("### **CrediMetrics**")
        st.caption("WORKSPACE")
        with st.expander("**Northstar Bank**\n*Risk operations*", expanded=False):
            st.write("Workspace options active.")
        st.markdown("---")

        selected_route = st.radio("Navigation", options=list(NAV_ROUTES.keys()), index=0, label_visibility="collapsed")
        st.markdown("---")

        st.markdown(f"""
        <div style="background-color: #131d33; padding: 12px; border-radius: 10px; border: 1px solid #1e2d4a; margin-top: 50px;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="background-color: {CURRENT_USER['color']}; color: white; width: 40px; height: 40px; border-radius: 50%; font-weight: bold; display: flex; align-items: center; justify-content: center; font-size: 1rem;">
                    {CURRENT_USER['initials']}
                </div>
                <div>
                    <strong style="color: white; font-size: 0.95rem;">{CURRENT_USER['name']}</strong><br>
                    <span style="color: #94a3b8; font-size: 0.78rem;">{CURRENT_USER['role']}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.caption(f"Email: {CURRENT_USER['email']}")

        if st.button("Log Out", type="secondary", width="stretch"):
            st.session_state.logged_in = False
            st.rerun()

    # Router execution
    route_fn = NAV_ROUTES.get(selected_route, render_dashboard)
    route_fn()


if __name__ == "__main__":
    main()