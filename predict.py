import joblib
import pandas as pd
import shap

# Feature name mapping: raw model keys -> human-readable labels
FEATURE_LABELS = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest Pain Type",
    "trestbps": "Blood Pressure",
    "chol": "Cholesterol",
    "fbs": "Fasting Blood Sugar",
    "restecg": "Resting ECG",
    "thalach": "Max Heart Rate",
    "exang": "Exercise Angina",
    "oldpeak": "ST Depression",
    "slope": "ST Slope",
    "ca": "Blocked Vessels",
    "thal": "Thalassemia Type",
}

EXPECTED_FEATURES = list(FEATURE_LABELS.keys())

# Load model once at startup with a clear error if file is missing
try:
    model = joblib.load("model.pkl")
    explainer = shap.TreeExplainer(model)
except FileNotFoundError:
    raise FileNotFoundError(
        "model.pkl not found. Run model.py first to train and save the model."
    )


def validate_input(input_data: dict) -> None:
    """Raise ValueError if input is missing or has unexpected keys."""
    missing = [f for f in EXPECTED_FEATURES if f not in input_data]
    if missing:
        raise ValueError(f"Missing features: {missing}")


def predict_risk(input_data: dict) -> dict:
    """
    Predict heart disease risk for a single patient.

    Args:
        input_data: dict with keys matching EXPECTED_FEATURES

    Returns:
        dict with keys: risk, probability
    """
    validate_input(input_data)

    df = pd.DataFrame([input_data])[EXPECTED_FEATURES]  # enforce column order
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "risk": "High Risk" if prediction == 1 else "Low Risk",
        "probability": round(probability * 100, 2),
    }


def explain_risk(input_data: dict) -> list[dict]:
    """
    Return SHAP feature contributions for a single patient, sorted by impact.

    Args:
        input_data: dict with keys matching EXPECTED_FEATURES

    Returns:
        List of dicts: [{"feature": "Blood Pressure", "value": 0.42, "direction": "increases risk"}, ...]
        Positive SHAP value = pushes toward High Risk
        Negative SHAP value = pushes toward Low Risk
    """
    validate_input(input_data)

    df = pd.DataFrame([input_data])[EXPECTED_FEATURES]
    shap_values = explainer.shap_values(df)

    # shap_values shape varies by shap version:
    # Older: list of 2 arrays, shape (1, n_features) each → shap_values[1][0]
    # Newer: single 3D array, shape (1, n_features, 2) → shap_values[0, :, 1]
    import numpy as np
    sv = np.array(shap_values)
    if sv.ndim == 3:
        # shape (1, n_features, 2) — newer shap
        contributions = sv[0, :, 1]
    elif isinstance(shap_values, list):
        # shape list[(1, n_features), (1, n_features)] — older shap
        contributions = shap_values[1][0]
    else:
        contributions = sv[0]

    results = []
    for feature, shap_val in zip(EXPECTED_FEATURES, contributions):
        results.append({
            "feature": FEATURE_LABELS[feature],
            "value": round(float(shap_val), 4),
            "direction": "increases risk" if shap_val > 0 else "reduces risk",
        })

    # Sort by absolute impact, highest first
    results.sort(key=lambda x: abs(x["value"]), reverse=True)

    return results
    