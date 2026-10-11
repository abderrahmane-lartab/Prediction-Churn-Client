"""Customer churn prediction UI built with Streamlit.

Run from the project root:
    streamlit run app/app.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
CLEANED_CSV = ROOT / "data/processed/cleaned_data.csv"
MODEL_PATH = ROOT / "data/processed/models/final_model.pkl"
PREPROCESSOR_PATH = ROOT / "data/processed/models/preprocessor.pkl"

NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
TENURE_BINS = [0, 12, 24, 48, 60, 72]
SERVICE_TO_NO = {"No phone service": "No", "No internet service": "No"}

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

st.title("Customer Churn Predictor")
st.caption("Enter the customer's data below to predict the risk of subscription cancellation.")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same cleaning used in the EDA notebook."""
    df = df.copy()
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.replace(SERVICE_TO_NO, "No")
    df = df.drop_duplicates()
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=TENURE_BINS, labels=False, include_lowest=True
    )
    return df


@st.cache_resource
def load():
    """Load the model and its exact preprocessor (build + cache it if needed)."""
    if not PREPROCESSOR_PATH.exists():
        df = clean_dataframe(pd.read_csv(CLEANED_CSV))
        X = df.drop(columns=["Churn"])
        y = df["Churn"]
        X_train, _, _, _ = train_test_split(
            X, y, test_size=0.20, random_state=40, stratify=y
        )
        cat_cols = [c for c in X.columns if c not in NUM_COLS]
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), NUM_COLS),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ]
        )
        preprocessor.fit(X_train)
        PREPROCESSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, PREPROCESSOR_PATH)
        return preprocessor, joblib.load(MODEL_PATH)

    return joblib.load(PREPROCESSOR_PATH), joblib.load(MODEL_PATH)


preprocessor, model = load()

with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Profile")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        monthly = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0, step=5.0)
        total = st.number_input("Total Charges ($)", min_value=0.0, value=500.0, step=50.0)

    with col2:
        st.subheader("Services")
        phone = st.selectbox("Phone Service", ["No", "Yes"])
        multiline = st.selectbox("Multiple Lines", ["No", "Yes"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_sec = st.selectbox("Online Security", ["No", "Yes"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes"])
        device_prot = st.selectbox("Device Protection", ["No", "Yes"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes"])
        stream_tv = st.selectbox("Streaming TV", ["No", "Yes"])
        stream_movies = st.selectbox("Streaming Movies", ["No", "Yes"])

    st.subheader("Contract & Billing")
    col3, col4, col5 = st.columns(3)
    with col3:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    with col4:
        paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
    with col5:
        payment = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )

    submitted = st.form_submit_button("Predict churn", type="primary")

if submitted:
    row = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": float(tenure),
        "PhoneService": phone,
        "MultipleLines": multiline,
        "InternetService": internet,
        "OnlineSecurity": online_sec,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_prot,
        "TechSupport": tech_support,
        "StreamingTV": stream_tv,
        "StreamingMovies": stream_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": float(monthly),
        "TotalCharges": float(total),
    }])

    cleaned = clean_dataframe(row)
    X = preprocessor.transform(cleaned)
    proba = float(model.predict_proba(X)[0, 1])
    danger = proba >= 0.5
    pct = round(proba * 100, 1)

    st.divider()
    if danger:
        st.error(f"High churn risk — **{pct}%** probability of cancelling.")
    else:
        st.success(f"Likely to stay — **{pct}%** churn probability.")
    st.progress(proba)
    st.caption(
        "⚠ This customer is at risk of churn."
        if danger
        else "✓ This customer is predicted to keep the subscription."
    )