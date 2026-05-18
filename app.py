import streamlit as st
import pandas as pd
from datetime import datetime
from predict import predict_risk

st.title("MedWise AI 🩺")
st.write("Enter your health data:")

age = st.slider("Age", 20, 80)

sex_label = st.selectbox("Sex", ["Female", "Male"])
sex = 1 if sex_label == "Male" else 0

cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.slider("Blood Pressure", 80, 200)
chol = st.slider("Cholesterol", 100, 400)

fbs_label = st.selectbox("Fasting Blood Sugar > 120", ["No", "Yes"])
fbs = 1 if fbs_label == "Yes" else 0

restecg = st.selectbox("Rest ECG", [0, 1, 2])
thalach = st.slider("Max Heart Rate", 60, 200)

exang_label = st.selectbox("Exercise Angina", ["No", "Yes"])
exang = 1 if exang_label == "Yes" else 0

oldpeak = st.slider("Oldpeak", 0.0, 5.0)
slope = st.selectbox("Slope", [0, 1, 2])
ca = st.selectbox("CA", [0, 1, 2, 3])
thal = st.selectbox("Thal", [0, 1, 2, 3])

st.subheader("Upload Health Data")

uploaded_file = st.file_uploader(
    "Upload a lab report or health CSV",
    type=["csv", "txt", "pdf"]
)

if uploaded_file is not None:
    st.success("File uploaded successfully!")

    if uploaded_file.name.endswith(".csv"):
        lab_df = pd.read_csv(uploaded_file)

        st.write("Lab Data Preview:")
        st.dataframe(lab_df)

        st.subheader("AI Lab Insights")
        st.write("This section will identify abnormal values and explain possible risks.")

        if "cholesterol" in lab_df.columns:
            avg_chol = lab_df["cholesterol"].mean()

            if avg_chol > 240:
                st.warning("Cholesterol appears high.")
            else:
                st.success("Cholesterol looks normal.")

    else:
        st.write("File uploaded. CSV analysis is supported right now.")

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

    if result["risk"] == "High Risk":
        st.error(f"⚠️ Risk: {result['risk']}")
    else:
        st.success(f"✅ Risk: {result['risk']}")

    st.metric("Probability", f"{result['probability']}%")
    st.caption("This is not medical advice. Please consult a doctor.")

    st.subheader("Personalized Recommendations")

    if chol > 240:
        st.write("- Cholesterol is high")
    if trestbps > 130:
        st.write("- Blood pressure is elevated")
    if age > 50:
        st.write("- Age may increase risk")

    st.write("Suggested next steps:")
    st.write("- Talk to a doctor")
    st.write("- Track blood pressure weekly")
    st.write("- Improve diet/exercise habits")

    history = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "age": age,
        "sex": sex_label,
        "risk": result["risk"],
        "probability": result["probability"]
    }

    df_history = pd.DataFrame([history])
    df_history.to_csv("prediction_history.csv", mode="a", header=False, index=False)

    st.success("Prediction saved to history!")