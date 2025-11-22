from pathlib import Path
from datetime import datetime
from typing import Dict

import pandas as pd
import streamlit as st
import joblib

from utils_explainability import get_feature_contributions, summarize_input_for_log


BASE_PATH = Path(__file__).resolve().parent
DATA_PATH = BASE_PATH / "data" / "credit_risk_data.csv"
MODEL_PATH = BASE_PATH / "models" / "credit_model.joblib"
LOGS_PATH = BASE_PATH / "logs"
DECISION_LOG_PATH = LOGS_PATH / "decision_logs.csv"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_data():
    return pd.read_csv(DATA_PATH)


def ensure_log_dir():
    LOGS_PATH.mkdir(parents=True, exist_ok=True)


def log_decision(
    user_id: str,
    input_features: Dict,
    prediction_label: str,
    probability: float,
    consent_to_use_data: bool,
) -> None:
    ensure_log_dir()

    log_row = {
        "timestamp": datetime.utcnow().isoformat(),
        "user_id": user_id,
        "prediction": prediction_label,
        "probability": probability,
        "consent_to_use_data": consent_to_use_data,
        "features": summarize_input_for_log(input_features),
    }

    log_df = pd.DataFrame([log_row])

    if DECISION_LOG_PATH.exists():
        log_df.to_csv(DECISION_LOG_PATH, mode="a", header=False, index=False)
    else:
        log_df.to_csv(DECISION_LOG_PATH, index=False)


def load_logs() -> pd.DataFrame:
    if DECISION_LOG_PATH.exists():
        return pd.read_csv(DECISION_LOG_PATH)
    return pd.DataFrame(
        columns=[
            "timestamp",
            "user_id",
            "prediction",
            "probability",
            "consent_to_use_data",
            "features",
        ]
    )


st.set_page_config(
    page_title="Ethical AI in Banking",
    layout="wide",
)

st.sidebar.title("Ethical AI Credit Demo")
view = st.sidebar.radio(
    "View",
    ["Customer view", "Fairness dashboard", "Audit log"],
)

model = load_model()
data = load_data()


if view == "Customer view":
    st.title("Customer view")

    st.write("Fill in the details to see the credit decision and an explanation.")

    with st.form("customer_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.slider("Age", 18, 75, 30)
            employment_years = st.slider("Years of employment", 0, 35, 3)
        with col2:
            income = st.number_input(
                "Annual income (₹)", min_value=10000, max_value=3000000, value=600000, step=10000
            )
            loan_amount = st.number_input(
                "Requested loan amount (₹)",
                min_value=50000,
                max_value=5000000,
                value=500000,
                step=50000,
            )
        with col3:
            credit_score = st.slider("Credit score", 300, 900, 700)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital status", ["Single", "Married", "Divorced"])

        consent_to_use_data = st.checkbox(
            "Allow the bank to use this data (anonymised) to improve the model.",
            value=True,
        )
        user_id = st.text_input("Customer ID", value="CUST_DEMO_001")

        submitted = st.form_submit_button("Get decision")

    if submitted:
        input_dict = {
            "age": age,
            "income": income,
            "loan_amount": loan_amount,
            "credit_score": credit_score,
            "employment_years": employment_years,
            "gender": gender,
            "marital_status": marital_status,
        }
        input_df = pd.DataFrame([input_dict])

        prob = float(model.predict_proba(input_df)[0, 1])
        label = "APPROVED" if prob >= 0.5 else "REJECTED"

        log_decision(
            user_id=user_id,
            input_features=input_dict,
            prediction_label=label,
            probability=prob,
            consent_to_use_data=bool(consent_to_use_data),
        )

        st.subheader("Decision")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(
                label="Result",
                value=label,
                delta=f"{prob * 100:.1f}% approval probability",
            )
        with col_b:
            st.write("The probability comes from patterns in the training data.")

        st.subheader("Main factors")
        contrib_df = get_feature_contributions(model, input_df)
        top_k = contrib_df.head(5)
        st.write("Positive values support approval, negative values support rejection.")
        st.dataframe(top_k.rename(columns={"feature": "Feature", "contribution": "Impact"}))
        st.bar_chart(top_k.set_index("feature")["contribution"])


elif view == "Fairness dashboard":
    st.title("Fairness dashboard")

    if "approved" not in data.columns:
        st.error("Training data does not contain the 'approved' column.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Approval rate by gender")
            gender_rates = data.groupby("gender")["approved"].mean().rename("approval_rate")
            st.dataframe(gender_rates.reset_index())
            st.bar_chart(gender_rates)

        with col2:
            st.subheader("Approval rate by marital status")
            ms_rates = data.groupby("marital_status")["approved"].mean().rename("approval_rate")
            st.dataframe(ms_rates.reset_index())
            st.bar_chart(ms_rates)

        st.subheader("Notes")
        gender_gap = abs(
            data.groupby("gender")["approved"].mean().max()
            - data.groupby("gender")["approved"].mean().min()
        )
        st.write(f"Difference between best and worst gender approval rate: {gender_gap * 100:.1f}%.")


elif view == "Audit log":
    st.title("Audit log")

    logs_df = load_logs()

    if logs_df.empty:
        st.warning("No decisions logged yet. Use the customer view first.")
    else:
        st.subheader("Recent decisions")
        st.dataframe(logs_df.tail(50))

        st.subheader("Stats")
        total_decisions = len(logs_df)
        consent_rate = logs_df["consent_to_use_data"].mean() * 100.0
        st.write(f"Total decisions: {total_decisions}")
        st.write(f"Data consent rate: {consent_rate:.1f}%")

        st.subheader("Filter by customer ID")
        filter_id = st.text_input("Customer ID filter")

        if filter_id:
            filtered = logs_df[logs_df["user_id"] == filter_id]
            if filtered.empty:
                st.info("No entries for this ID.")
            else:
                st.dataframe(filtered)
