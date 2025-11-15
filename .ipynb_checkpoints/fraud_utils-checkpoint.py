
# fraud_utils.py

import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
MODEL_FILENAME = "mule_fraud_model.pkl"  # change here if your file name differs


# -------------------------------------------------------------------
# Feature definitions (must match training data)
# -------------------------------------------------------------------
FEATURES = [
    "age",
    "city",
    "account_tenure_months",
    "avg_monthly_balance",
    "kyc_type",
    "total_inflow_24hr",
    "count_inflow_24hr",
    "count_unique_creditors_24hr",
    "total_outflow_24hr",
    "count_outflow_24hr",
    "time_diff_first_inflow_to_outflow",
    "percent_inflow_cashed_out_1hr",
    "velocity_inflow_1hr",
    "velocity_outflow_1hr",
    "device_change_last_48hr",
    "new_payee_added_last_7d",
    "international_ip_flag",
    "txn_amount",
    "txn_hour",
    "merchant_category",
]

FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "age": "Customer age in years.",
    "city": "Customer’s primary city/location.",
    "account_tenure_months": "How long the account has been open (in months).",
    "avg_monthly_balance": "Average monthly balance in the account (₹).",
    "kyc_type": "Level of KYC completed (eKYC / Minimum / Full).",
    "total_inflow_24hr": "Total amount credited in the last 24 hours (₹).",
    "count_inflow_24hr": "Number of credit transactions in the last 24 hours.",
    "count_unique_creditors_24hr": "Number of distinct senders in the last 24 hours.",
    "total_outflow_24hr": "Total amount debited in the last 24 hours (₹).",
    "count_outflow_24hr": "Number of debit transactions in the last 24 hours.",
    "time_diff_first_inflow_to_outflow": (
        "Minutes between first credit and first debit in the last 24 hours."
    ),
    "percent_inflow_cashed_out_1hr": (
        "Percentage of credited amount withdrawn within 1 hour."
    ),
    "velocity_inflow_1hr": "Number of credit transactions in the last 1 hour.",
    "velocity_outflow_1hr": "Number of debit transactions in the last 1 hour.",
    "device_change_last_48hr": "Whether device changed in the last 48 hours (0=No, 1=Yes).",
    "new_payee_added_last_7d": "Whether a new payee was added in last 7 days (0=No, 1=Yes).",
    "international_ip_flag": "Whether login/transaction used an international IP (0=No, 1=Yes).",
    "txn_amount": "Amount of the current transaction (₹).",
    "txn_hour": "Hour of day of transaction (0–23).",
    "merchant_category": "High-level category of where money is going (e.g., Groceries, Travel).",
}


# -------------------------------------------------------------------
# Model loader (cached at module level)
# -------------------------------------------------------------------
_model = None


def load_model():
    """
    Load the trained pipeline from disk (only once).
    """
    global _model
    if _model is None:
        here = Path(__file__).resolve().parent
        model_path = here / MODEL_FILENAME
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        _model = joblib.load(model_path)
    return _model


# -------------------------------------------------------------------
# Input preparation
# -------------------------------------------------------------------
BINARY_COLUMNS = [
    "device_change_last_48hr",
    "new_payee_added_last_7d",
    "international_ip_flag",
]


def _to_binary(value: Any) -> int:
    """
    Convert various YES/NO / bool representations to 0/1.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"yes", "y", "true", "1"}:
            return 1
        if v in {"no", "n", "false", "0"}:
            return 0
    # fallback: treat anything else as 0
    return 0


def prepare_input_df(user_input: Dict[str, Any]) -> pd.DataFrame:
    """
    Take raw user input dict from the UI, ensure:
    - all expected feature columns are present
    - binary columns are converted to 0/1 ints
    - columns are in the correct order for the model
    """
    data = dict(user_input)  # shallow copy

    # Normalise binary columns
    for col in BINARY_COLUMNS:
        if col in data:
            data[col] = _to_binary(data[col])
        else:
            data[col] = 0  # default

    # Ensure all expected feature columns exist
    missing = [f for f in FEATURES if f not in data]
    if missing:
        raise ValueError(f"Missing required feature(s): {missing}")

    # Create single-row DataFrame with correct column order
    df = pd.DataFrame([[data[f] for f in FEATURES]], columns=FEATURES)
    return df


# -------------------------------------------------------------------
# Prediction API used by Streamlit app
# -------------------------------------------------------------------
def predict_fraud(user_input: Dict[str, Any], threshold: float = 0.5) -> Tuple[int, float]:
    """
    Given a raw input dict, return:
    - predicted label (0 = genuine, 1 = fraud)
    - fraud probability (float between 0 and 1)
    """
    model = load_model()
    X = prepare_input_df(user_input)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0, 1]
    else:
        # Some models don't expose predict_proba; fall back to decision_function
        if hasattr(model, "decision_function"):
            score = model.decision_function(X)[0]
            # simple logistic transform
            import math

            proba = 1 / (1 + math.exp(-score))
        else:
            # Very last resort: 0 or 1 directly from predict
            proba = float(model.predict(X)[0])

    label = int(proba >= threshold)
    return label, float(proba)
