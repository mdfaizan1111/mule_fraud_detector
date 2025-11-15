# fraud_app.py

import streamlit as st
import pandas as pd

from fraud_utils import (
    FEATURES,
    FEATURE_DESCRIPTIONS,
    predict_fraud,
)

# -------------------------------------------------------------------
# Page config
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Mule / Fraud Transaction Detector",
    layout="centered",
)


st.title("🕵️‍♂️ Mule / Fraud Transaction Detector")
st.write(
    "Enter details of a transaction and customer profile below. "
    "The model will estimate the probability that this is a mule / fraud transaction."
)


# -------------------------------------------------------------------
# Helper: options for categorical fields
# -------------------------------------------------------------------
CITY_OPTIONS = [
    "Delhi",
    "Mumbai",
    "Bengaluru",
    "Hyderabad",
    "Kolkata",
    "Chennai",
    "Pune",
    "Ahmedabad",
    "Jaipur",
    "Surat",
    "Other",
]

KYC_OPTIONS = ["eKYC", "Minimum", "Full"]

MERCHANT_CATEGORIES = [
    "Groceries",
    "Electronics",
    "Food",
    "Travel",
    "Utility",
    "Entertainment",
    "Gaming",
    "Crypto",
    "Wallet",
    "Other",
]


# -------------------------------------------------------------------
# UI: Collect inputs
# -------------------------------------------------------------------
with st.form("fraud_form"):
    st.subheader("Customer Profile")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age (years)",
            min_value=18,
            max_value=90,
            value=30,
            help=FEATURE_DESCRIPTIONS["age"],
        )

    with col2:
        city = st.selectbox(
            "City",
            options=CITY_OPTIONS,
            index=2,
            help=FEATURE_DESCRIPTIONS["city"],
        )

    col3, col4 = st.columns(2)
    with col3:
        account_tenure_months = st.number_input(
            "Account Tenure (months)",
            min_value=0,
            max_value=600,
            value=12,
            help=FEATURE_DESCRIPTIONS["account_tenure_months"],
        )
    with col4:
        avg_monthly_balance = st.number_input(
            "Average Monthly Balance (₹)",
            min_value=0.0,
            value=25000.0,
            step=1000.0,
            help=FEATURE_DESCRIPTIONS["avg_monthly_balance"],
        )

    kyc_type = st.selectbox(
        "KYC Type",
        options=KYC_OPTIONS,
        index=2,
        help=FEATURE_DESCRIPTIONS["kyc_type"],
    )

    st.markdown("---")
    st.subheader("Recent Account Activity (24 hours)")

    c1, c2 = st.columns(2)
    with c1:
        total_inflow_24hr = st.number_input(
            "Total Inflow (₹, 24h)",
            min_value=0.0,
            value=50000.0,
            step=1000.0,
            help=FEATURE_DESCRIPTIONS["total_inflow_24hr"],
        )
    with c2:
        count_inflow_24hr = st.number_input(
            "Number of Credit Transactions (24h)",
            min_value=0,
            value=3,
            help=FEATURE_DESCRIPTIONS["count_inflow_24hr"],
        )

    c3, c4 = st.columns(2)
    with c3:
        count_unique_creditors_24hr = st.number_input(
            "Unique Creditors (24h)",
            min_value=0,
            value=2,
            help=FEATURE_DESCRIPTIONS["count_unique_creditors_24hr"],
        )
    with c4:
        total_outflow_24hr = st.number_input(
            "Total Outflow (₹, 24h)",
            min_value=0.0,
            value=45000.0,
            step=1000.0,
            help=FEATURE_DESCRIPTIONS["total_outflow_24hr"],
        )

    c5, c6 = st.columns(2)
    with c5:
        count_outflow_24hr = st.number_input(
            "Number of Debit Transactions (24h)",
            min_value=0,
            value=4,
            help=FEATURE_DESCRIPTIONS["count_outflow_24hr"],
        )
    with c6:
        time_diff_first_inflow_to_outflow = st.number_input(
            "Minutes from First Inflow to First Outflow (24h)",
            min_value=0.0,
            value=10.0,
            step=1.0,
            help=FEATURE_DESCRIPTIONS["time_diff_first_inflow_to_outflow"],
        )

    percent_inflow_cashed_out_1hr = st.number_input(
        "Percent of Inflow Cashed Out within 1 hour (%)",
        min_value=0.0,
        max_value=100.0,
        value=80.0,
        step=1.0,
        help=FEATURE_DESCRIPTIONS["percent_inflow_cashed_out_1hr"],
    )

    st.markdown("---")
    st.subheader("Velocity & Device Behaviour")

    v1, v2 = st.columns(2)
    with v1:
        velocity_inflow_1hr = st.number_input(
            "Credit Transactions in Last 1 hour",
            min_value=0,
            value=1,
            help=FEATURE_DESCRIPTIONS["velocity_inflow_1hr"],
        )
    with v2:
        velocity_outflow_1hr = st.number_input(
            "Debit Transactions in Last 1 hour",
            min_value=0,
            value=2,
            help=FEATURE_DESCRIPTIONS["velocity_outflow_1hr"],
        )

    d1, d2, d3 = st.columns(3)
    with d1:
        device_change_last_48hr = st.selectbox(
            "Device Changed in Last 48h?",
            options=["No", "Yes"],
            index=0,
            help=FEATURE_DESCRIPTIONS["device_change_last_48hr"],
        )
    with d2:
        new_payee_added_last_7d = st.selectbox(
            "New Payee Added in Last 7 days?",
            options=["No", "Yes"],
            index=0,
            help=FEATURE_DESCRIPTIONS["new_payee_added_last_7d"],
        )
    with d3:
        international_ip_flag = st.selectbox(
            "International IP Detected?",
            options=["No", "Yes"],
            index=0,
            help=FEATURE_DESCRIPTIONS["international_ip_flag"],
        )

    st.markdown("---")
    st.subheader("Current Transaction Details")

    t1, t2 = st.columns(2)
    with t1:
        txn_amount = st.number_input(
            "Transaction Amount (₹)",
            min_value=0.0,
            value=20000.0,
            step=1000.0,
            help=FEATURE_DESCRIPTIONS["txn_amount"],
        )
    with t2:
        txn_hour = st.slider(
            "Transaction Hour of Day (0–23)",
            min_value=0,
            max_value=23,
            value=14,
            help=FEATURE_DESCRIPTIONS["txn_hour"],
        )

    merchant_category = st.selectbox(
        "Merchant Category",
        options=MERCHANT_CATEGORIES,
        help=FEATURE_DESCRIPTIONS["merchant_category"],
    )

    st.markdown("---")

    submitted = st.form_submit_button("Predict Fraud")


# -------------------------------------------------------------------
# Build input dict & run prediction
# -------------------------------------------------------------------
if submitted:
    # Map YES/NO to strings; conversion to 0/1 happens in fraud_utils
    user_input = {
        "age": age,
        "city": city,
        "account_tenure_months": account_tenure_months,
        "avg_monthly_balance": avg_monthly_balance,
        "kyc_type": kyc_type,
        "total_inflow_24hr": total_inflow_24hr,
        "count_inflow_24hr": count_inflow_24hr,
        "count_unique_creditors_24hr": count_unique_creditors_24hr,
        "total_outflow_24hr": total_outflow_24hr,
        "count_outflow_24hr": count_outflow_24hr,
        "time_diff_first_inflow_to_outflow": time_diff_first_inflow_to_outflow,
        "percent_inflow_cashed_out_1hr": percent_inflow_cashed_out_1hr,
        "velocity_inflow_1hr": velocity_inflow_1hr,
        "velocity_outflow_1hr": velocity_outflow_1hr,
        "device_change_last_48hr": device_change_last_48hr,
        "new_payee_added_last_7d": new_payee_added_last_7d,
        "international_ip_flag": international_ip_flag,
        "txn_amount": txn_amount,
        "txn_hour": txn_hour,
        "merchant_category": merchant_category,
    }

    try:
        label, prob = predict_fraud(user_input, threshold=0.5)

        st.subheader("Prediction Result")
        if label == 1:
            st.error(f"⚠️ Transaction is LIKELY FRAUDULENT (probability = {prob:.3f})")
        else:
            st.success(f"✅ Transaction is likely GENUINE (fraud probability = {prob:.3f})")

        # Optional debug view
        with st.expander("Show input data used for prediction"):
            st.write(pd.DataFrame([user_input]))

    except Exception as e:
        st.error(f"Error while running prediction: {e}")
