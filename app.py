import streamlit as st
import pandas as pd
import joblib

model = joblib.load('random_forest_model.pkl')

st.title("Hotel Booking Cancellation Predictor")

lead_time = st.slider("Lead Time (days)", 0, 500, 100)
total_guests = st.slider("Total Guests", 1, 6, 2)
deposit_type = st.selectbox("Deposit Type", ['No Deposit', 'Non Refund', 'Refundable'])
market_segment = st.selectbox("Market Segment", ['Online TA', 'Offline TA/TO', 'Direct'])

# Sample dummy input — should match training schema
input_dict = {
    'lead_time': lead_time,
    'total_guests': total_guests,
    'deposit_type_Non Refund': int(deposit_type == 'Non Refund'),
    'deposit_type_Refundable': int(deposit_type == 'Refundable'),
    'market_segment_Offline TA/TO': int(market_segment == 'Offline TA/TO'),
    'market_segment_Online TA': int(market_segment == 'Online TA'),
    # add other required features and dummy values
}

X_input = pd.DataFrame([input_dict])
prediction = model.predict(X_input)[0]

st.write(f"Prediction: {'❌ Will Cancel' if prediction else '✅ Will Not Cancel'}")
