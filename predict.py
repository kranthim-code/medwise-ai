import joblib
import pandas as pd

model = joblib.load("model.pkl")

def predict_risk(input_data):
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    if prediction == 1:
        risk = "High Risk"
    else:
        risk = "Low Risk"

    return {
        "risk": risk,
        "probability": round(probability * 100, 2)
    }
    