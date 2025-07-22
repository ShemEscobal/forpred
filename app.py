import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

st.set_page_config(page_title="Forest Fire Burned Area Predictor", page_icon="🔥", layout="centered")

st.markdown("""
<style>
.maroon-header { color: #800000; font-size: 2.2em; font-weight: bold; text-align: center; margin-bottom: 0.5em; }
.maroon-note { background: #fff3f3; color: #800000; border: 1.5px solid #800000; border-radius: 0.5em; padding: 0.7em 1em; margin-bottom: 1.5em; font-size: 1.1em; }
.result-box { border: 3px solid #000; border-left: 10px solid #800000; background: #fff; border-radius: 1.5rem; padding: 2em 1em; margin-top: 2.5em; text-align: center; box-shadow: 0 2px 16px #0001; }
.result-box h2 { color: #800000; font-weight: bold; letter-spacing: 1px; margin-bottom: 1em; }
.result-box .fs-4 { color: #000; font-size: 1.3em; }
.result-box .risk-label { font-weight: bold; text-transform: uppercase; color: #800000; font-size: 1.5em; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="maroon-header">Forest Fire Burned Area Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="maroon-note"><b>Note:</b> For demo purposes, all other features are fixed and hidden. Only temperature, wind, relative humidity, and rain can be changed.</div>', unsafe_allow_html=True)

# Input fields
col1, col2 = st.columns(2)
with col1:
    temp = st.number_input('Temp (°C)', min_value=2.2, max_value=33.3, value=25.0, step=0.01, format="%.2f")
    wind = st.number_input('Wind (km/h)', min_value=0.4, max_value=9.4, value=4.0, step=0.01, format="%.2f")
with col2:
    RH = st.number_input('RH (%)', min_value=15, max_value=100, value=45, step=1)
    rain = st.number_input('Rain (mm)', min_value=0.0, max_value=6.4, value=0.0, step=0.01, format="%.2f")

# Fixed demo values for other features
fixed_values = {
    'X': 4,
    'Y': 4,
    'month': 'aug',
    'day': 'fri',
    'FFMC': 85.0,
    'DMC': 26.2,
    'DC': 94.3,
    'ISI': 5.1
}

FEATURES = ['X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']

# Helper for risk classification
def classify_risk(area):
    if area < 0.5:
        return 'Low', '#228B22'
    elif area < 1.0:
        return 'Moderate', '#FF8C00'
    elif area < 1.5:
        return 'High', '#B22222'
    else:
        return 'Extreme', '#800000'

if st.button('Predict', use_container_width=True):
    # Prepare input
    values = [fixed_values['X'], fixed_values['Y'], fixed_values['month'], fixed_values['day'],
              fixed_values['FFMC'], fixed_values['DMC'], fixed_values['DC'], fixed_values['ISI'],
              temp, RH, wind, rain]
    X_input = pd.DataFrame([values], columns=FEATURES)
    # Load model and preprocessors
    model = joblib.load('fire_model_export/best_fire_prediction_model.pkl')
    encoder_path = 'fire_model_export/feature_encoder.pkl'
    scaler_path = 'fire_model_export/feature_scaler.pkl'
    encoder = joblib.load(encoder_path) if os.path.exists(encoder_path) else None
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    # Preprocess
    if encoder is not None and scaler is not None:
        X_encoded = encoder.transform(X_input.values)
        X_encoded = X_encoded.astype(float)
        X_final = X_encoded[:, np.r_[1:12, 13:29]]
        X_scaled = X_final.copy()
        X_scaled[:, -10:] = scaler.transform(X_scaled[:, -10:])
        X_for_pred = X_scaled
    else:
        # Label encode month/day if needed
        if X_input['month'].dtype == object:
            from sklearn.preprocessing import LabelEncoder
            le_month = LabelEncoder()
            X_input['month'] = le_month.fit_transform(X_input['month'].astype(str))
        if X_input['day'].dtype == object:
            from sklearn.preprocessing import LabelEncoder
            le_day = LabelEncoder()
            X_input['day'] = le_day.fit_transform(X_input['day'].astype(str))
        X_for_pred = X_input.values
    # Predict
    area = float(model.predict(X_for_pred)[0])
    area = max(area, 0)
    risk, color = classify_risk(area)
    st.markdown(f'''
    <div class="result-box">
      <h2>Prediction Result</h2>
      <div class="fs-4 mb-2"><b>Predicted Burned Area:</b> <span class="maroon-text">{area:.2f} hectares</span></div>
      <div class="fs-4"><b>Risk Level:</b> <span class="risk-label" style="color:{color}">{risk}</span></div>
    </div>
    ''', unsafe_allow_html=True) 
