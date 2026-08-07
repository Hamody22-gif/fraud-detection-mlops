"""Streamlit UI that calls the deployed fraud-detection API."""

import requests
import streamlit as st

API_URL = "https://fraud-detection-api-ip6h.onrender.com/predict"

st.set_page_config(page_title="Fraud Detection", page_icon="🕵️")
st.title("🕵️ Credit-Card Fraud Detection")
st.write("Enter a transaction — the model estimates the probability it's fraudulent.")

# --- Input widgets ---
amt = st.number_input("Amount ($)", min_value=0.0, value=120.0, step=10.0)
category = st.selectbox(
    "Category",
    [
        "grocery_pos",
        "grocery_net",
        "shopping_pos",
        "shopping_net",
        "gas_transport",
        "misc_pos",
        "misc_net",
        "entertainment",
        "food_dining",
        "health_fitness",
        "home",
        "kids_pets",
        "personal_care",
        "travel",
    ],
)
gender = st.selectbox("Gender", ["F", "M"])
trans_time = st.text_input("Transaction date & time", "2020-06-21 03:14:25")
dob = st.text_input("Cardholder date of birth", "1990-03-19")
time_since_last = st.number_input("Seconds since card's last transaction", min_value=0, value=3600)

# --- Call the API when the button is clicked ---
if st.button("Check for fraud", type="primary"):
    payload = {
        "amt": amt,
        "category": category,
        "gender": gender,
        "trans_date_trans_time": trans_time,
        "dob": dob,
        "time_since_last": time_since_last,
    }
    with st.spinner("Scoring… (first call may take ~30–60s if the API was asleep)"):
        try:
            response = requests.post(API_URL, json=payload, timeout=90)
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Couldn't reach the API: {e}")
        else:
            proba = result["fraud_probability"]
            st.metric("Fraud probability", f"{proba:.1%}")
            if result["is_fraud"]:
                st.error("⚠️ Predicted **FRAUD**")
            else:
                st.success("✅ Looks legitimate")
