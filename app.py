import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Industrial Machine Failure Prediction", layout="centered")
st.title("Industrial Machine Failure Prediction")
st.write("Enter real-time machine telemetry to predict failure risk.")

model = joblib.load('machine_failure_model.joblib')
feature_cols = joblib.load('model_features.joblib')

col1, col2 = st.columns(2)
with col1:
    m_type = st.selectbox("Machine Quality Type", ["L", "M", "H"])
    air_temp = st.number_input("Air Temperature [K]", value=300.0, step=0.1)
    proc_temp = st.number_input("Process Temperature [K]", value=310.0, step=0.1)
with col2:
    rpm = st.number_input("Rotational Speed [rpm]", value=1500, step=10)
    torque = st.number_input("Torque [Nm]", value=40.0, step=0.5)
    tool_wear = st.number_input("Tool Wear [min]", value=100, step=5)

if st.button("Predict Failure Risk", type="primary"):
    temp_diff = proc_temp - air_temp
    power_w = torque * rpm * (2 * np.pi / 60)

    input_dict = {
        'Air temperature [K]': air_temp,
        'Process temperature [K]': proc_temp,
        'Rotational speed [rpm]': rpm,
        'Torque [Nm]': torque,
        'Tool wear [min]': tool_wear,
        'Temp_diff': temp_diff,
        'Power_W': power_w,
        'Type_L': 1 if m_type == 'L' else 0,
        'Type_M': 1 if m_type == 'M' else 0
    }
    input_df = pd.DataFrame([input_dict])[feature_cols]
    prob = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]

    if pred == 1 or prob > 0.4:
        st.error(f"High Failure Risk Detected! (Failure Probability: {prob*100:.2f}%)")
    else:
        st.success(f"Machine Operating Normally (Failure Probability: {prob*100:.2f}%)")
