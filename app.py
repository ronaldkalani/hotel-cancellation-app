import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('random_forest_model.pkl')

# Page config
st.set_page_config(page_title="Hotel Cancellation Predictor", layout="centered")

st.title("🏨 Hotel Booking Cancellation Predictor")
st.markdown("Predict whether a hotel booking will be **cancelled** based on customer details.")

# Sidebar inputs
st.sidebar.header("Enter Booking Information")

lead_time = st.sidebar.slider("Lead Time (days)", 0, 500, 100)
total_guests = st.sidebar.slider("Total Guests", 1, 6, 2)
deposit_type = st.sidebar.selectbox("Deposit Type", ['No Deposit', 'Non Refund', 'Refundable'])
market_segment = st.sidebar.selectbox("Market Segment", ['Online TA', 'Offline TA/TO', 'Direct'])

# Convert inputs to match training data structure
input_dict = {
    'lead_time': lead_time,
    'total_guests': total_guests,
    'deposit_type_Non Refund': 1 if deposit_type == 'Non Refund' else 0,
    'deposit_type_Refundable': 1 if deposit_type == 'Refundable' else 0,
    'market_segment_Offline TA/TO': 1 if market_segment == 'Offline TA/TO' else 0,
    'market_segment_Online TA': 1 if market_segment == 'Online TA' else 0,
    # Add any other required dummy variables here
}

# Build DataFrame
input_df = pd.DataFrame([input_dict])

# Predict
if st.button("Predict Cancellation"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.error("❌ This booking is **likely to be canceled**.")
    else:
        st.success("✅ This booking is **likely to be fulfilled**.")

