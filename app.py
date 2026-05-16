import streamlit as st
from predict import predict_risk

st.title("MedWise AI 🩺")

st.write("Enter your health data:")

age = st.slider("Age", 20, 80)
sex_label = st.selectbox("Sex", ["Female", "Male"])
sex = 1 if sex_label == "Male" else 0
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.slider("Blood Pressure", 80, 200)
chol = st.slider("Cholesterol", 100, 400)
fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
restecg = st.selectbox("Rest ECG", [0, 1, 2])
thalach = st.slider("Max Heart Rate", 60, 200)
exang = st.selectbox("Exercise Angina", [0, 1])
oldpeak = st.slider("Oldpeak", 0.0, 5.0)
slope = st.selectbox("Slope", [0, 1, 2])
ca = st.selectbox("CA", [0, 1, 2, 3])
thal = st.selectbox("Thal", [0, 1, 2, 3])

if st.button("Predict"):
    user_data = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }

    result = predict_risk(user_data)

    st.write(f"Risk: {result['risk']}")
    st.write(f"Probability: {result['probability']}%")
