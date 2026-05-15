import streamlit as st


# Page Setup
st.set_page_config(page_title="MedWise AI", layout="centered")
st.subheader("Personalized Health Risk Dashboard")

st.write("Enter your health information below")

# User Inputs
age = st.slider("Age", 0, 120, 30)

cholesterol = st.slider("Cholesterol Level (mg/dL)", 100, 400, 200)
blood_pressure = st.slider("Blood Pressure (mm Hg)", 80, 200, 120)
max_heart_rate = st.slider("Max Heart Rate (bpm)", 60, 220, 150)
excersize = st.selectbox("Exersize Level?", ["Low", "Medium", "High"])

# Prediction
if st.button("Predict Risk"):
    st.success
    