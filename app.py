from flask import Flask, render_template, request
import joblib
import numpy as np
import os
import pandas as pd

app = Flask(__name__)

# Feature columns
FEATURES = ['X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
MONTHS = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
DAYS = ['mon','tue','wed','thu','fri','sat','sun']

# Helper for risk classification
def classify_risk(area):
    if area < 0.5:
        return 'Low', 'green'
    elif area < 1.0:
        return 'Moderate', 'orange'
    elif area < 1.5:
        return 'High', 'red'
    else:
        return 'Extreme', 'darkred'

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        # Get input values
        values = []
        for f in FEATURES:
            v = request.form[f]
            if f in ['month', 'day']:
                values.append(v)
            else:
                values.append(float(v))
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
        result = {'area': f"{area:.2f}", 'risk': risk, 'color': color}
    return render_template('index.html', result=result, months=MONTHS, days=DAYS)

if __name__ == '__main__':
    app.run(debug=True) 
