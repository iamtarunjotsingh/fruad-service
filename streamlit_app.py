"""Interactive fraud scoring demo and session-level monitoring dashboard."""
import os

import pandas as pd
import requests
import streamlit as st

from fraud.serving import PCA_COLUMNS


API_URL = os.getenv("FRAUD_API_URL", "http://127.0.0.1:8000").rstrip("/")
if "://" not in API_URL:
    API_URL = f"https://{API_URL}"

st.set_page_config(page_title="Transaction Fraud Detection", layout="wide")
st.title("Transaction Fraud Detection")
st.caption("Score a transaction using the deployed API and inspect its model contributions.")

if "score_history" not in st.session_state:
    st.session_state.score_history = []

with st.sidebar:
    st.subheader("API status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=3)
        if health.ok:
            health_data = health.json()
            st.success(f"Model ready · {health_data.get('model_version', 'unknown')}")
        else:
            st.error(f"API unavailable ({health.status_code})")
    except requests.RequestException as exc:
        st.error(f"Cannot reach API: {exc}")

    st.subheader("Session monitoring")
    history = st.session_state.score_history
    if history:
        flagged = sum(item["is_fraud"] for item in history)
        st.metric("Transactions scored", len(history))
        st.metric("Flag rate", f"{flagged / len(history):.1%}")
        mean_probability = sum(item["fraud_probability"] for item in history) / len(history)
        st.metric("Mean fraud probability", f"{mean_probability:.3f}")
    else:
        st.caption("Metrics appear here after scoring transactions; no transaction inputs are retained.")

with st.form("transaction_form"):
    st.subheader("Transaction details")
    time_col, amount_col = st.columns(2)
    payload = {
        "Time": time_col.number_input("Time (seconds since dataset start)", min_value=0.0, max_value=1e9,
                                       value=0.0, step=60.0),
        "Amount": amount_col.number_input("Amount", min_value=0.0, max_value=1e7, value=1.0, step=1.0),
    }
    st.caption("V1–V28 are anonymized PCA components. Leave them at zero for a schema-only example.")
    for start in range(0, len(PCA_COLUMNS), 4):
        columns = st.columns(4)
        for column, feature in zip(columns, PCA_COLUMNS[start:start + 4]):
            payload[feature] = column.number_input(feature, value=0.0, step=0.1, format="%.5f")
    submitted = st.form_submit_button("Score transaction", type="primary")

if submitted:
    try:
        prediction_response = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
        prediction_response.raise_for_status()
        prediction = prediction_response.json()
    except requests.RequestException as exc:
        st.error(f"Scoring failed: {exc}")
    else:
        st.session_state.score_history.append({
            "fraud_probability": prediction["fraud_probability"],
            "is_fraud": prediction["is_fraud"],
        })
        decision = "Flag for review" if prediction["is_fraud"] else "Not flagged"
        score_col, threshold_col, decision_col = st.columns(3)
        score_col.metric("Calibrated fraud probability", f"{prediction['fraud_probability']:.2%}")
        threshold_col.metric("Raw-score threshold", f"{prediction['threshold']:.5f}")
        decision_col.metric("Decision", decision)
        st.caption(f"Model version: {prediction['model_version']} · Raw score: {prediction['raw_score']:.6f}")

        try:
            explanation_response = requests.post(f"{API_URL}/explain", json=payload, timeout=15)
            explanation_response.raise_for_status()
            explanation = explanation_response.json()
        except requests.RequestException as exc:
            st.error(f"Explanation unavailable: {exc}")
        else:
            contributions = pd.DataFrame(explanation["contributions"])
            st.subheader("Top feature contributions")
            st.caption("Tree SHAP contributions explain the raw LightGBM margin, not the calibrated probability. They are not causal explanations.")
            st.bar_chart(contributions.head(8).set_index("feature")["contribution"])
            st.dataframe(contributions.head(8), hide_index=True, use_container_width=True)

if st.session_state.score_history:
    st.subheader("Fraud probability by scored transaction (this session)")
    st.line_chart(pd.DataFrame({
        "Fraud probability": [item["fraud_probability"] for item in st.session_state.score_history],
    }))