import streamlit as st
import joblib
import numpy as np

model = joblib.load("model.pkl")

# Page Setup
st.set_page_config(page_title="MedWise AI", layout="centered")
st.subheader("Personalized Health Risk Dashboard")

st.write("Enter your health information below")

# User Inputs
age = st.slider("Age", 0, 120, 30)

cholesterol = st.slider("Cholesterol Level (mg/dL)", 100, 400, 200)
blood_pressure = st.slider("Blood Pressure (mm Hg)", 80, 200, 120)
max_heart_rate = st.slider("Max Heart Rate (bpm)", 60, 220, 150)
exercise = st.selectbox("Exersise Level?", ["Low", "Medium", "High"])

# Prediction
if st.button("Predict Risk"):

    # Create input array
    input_data = np.array([[
        age,
        cholesterol,
        blood_pressure,
        max_heart_rate
    ]])

    # Make prediction
    prediction = model.predict(input_data)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("Higher Heart Disease Risk Detected")
        st.progress(75)
        st.write("Estimated Risk Score: 75%")

    else:
        st.success("Lower Heart Disease Risk")
        st.progress(25)
        st.write("Estimated Risk Score: 25%")

    st.write("Age:", age)
    st.write("Cholesterol:", cholesterol)
    st.write("Blood Pressure:", blood_pressure)
    st.write("Max Heart Rate:", max_heart_rate)
    st.write("Exercise Level:", exercise)




st.subheader("Top Risk Factors")

st.write("• High cholesterol increased risk")
st.write("• Low exercise level increased risk")
st.write("• Normal blood pressure lowered risk")
