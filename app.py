import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from predict import predict_risk, explain_risk
from pypdf import PdfReader
from openai import OpenAI

st.set_page_config(page_title="MedWise AI", page_icon="🩺", layout="wide")
st.title("MedWise AI 🩺")

HISTORY_FILE = "prediction_history.csv"
HISTORY_COLS = ["time", "age", "sex", "risk", "probability"]

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_predict, tab_history = st.tabs(["🔍 Predict", "📈 My History"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:

    # ── Core health inputs ────────────────────────────────────────────────────
    st.subheader("Your Health Data")
    col1, col2, col3 = st.columns(3)

    with col1:
        age        = st.slider("Age", 20, 80, 45)
        sex_label  = st.selectbox("Sex", ["Female", "Male"])
        sex        = 1 if sex_label == "Male" else 0
        cp         = st.selectbox("Chest Pain Type",
                                  [0, 1, 2, 3],
                                  format_func=lambda x: {
                                      0: "0 – Typical Angina",
                                      1: "1 – Atypical Angina",
                                      2: "2 – Non-Anginal Pain",
                                      3: "3 – Asymptomatic"}[x])
        trestbps   = st.slider("Resting Blood Pressure (mmHg)", 80, 200, 120)
        chol       = st.slider("Cholesterol (mg/dl)", 100, 400, 200)

    with col2:
        fbs_label  = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
        fbs        = 1 if fbs_label == "Yes" else 0
        restecg    = st.selectbox("Resting ECG",
                                  [0, 1, 2],
                                  format_func=lambda x: {
                                      0: "0 – Normal",
                                      1: "1 – ST-T Abnormality",
                                      2: "2 – Left Ventricular Hypertrophy"}[x])
        thalach    = st.slider("Max Heart Rate Achieved", 60, 200, 150)
        exang_label = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
        exang      = 1 if exang_label == "Yes" else 0

    with col3:
        oldpeak    = st.slider("ST Depression (Oldpeak)", 0.0, 5.0, 1.0)
        slope      = st.selectbox("ST Slope",
                                  [0, 1, 2],
                                  format_func=lambda x: {
                                      0: "0 – Upsloping",
                                      1: "1 – Flat",
                                      2: "2 – Downsloping"}[x])
        ca         = st.selectbox("Blocked Vessels (CA)",
                                  [0, 1, 2, 3],
                                  format_func=lambda x: f"{x} vessel{'s' if x != 1 else ''}")
        thal       = st.selectbox("Thalassemia (Thal)",
                                  [0, 1, 2, 3],
                                  format_func=lambda x: {
                                      0: "0 – Normal",
                                      1: "1 – Fixed Defect",
                                      2: "2 – Reversible Defect",
                                      3: "3 – Unknown"}[x])

    # ── Wearable data (optional) ──────────────────────────────────────────────
    with st.expander("⌚ Wearable Data (Optional)"):
        w_col1, w_col2, w_col3 = st.columns(3)
        with w_col1:
            resting_hr = st.slider("Resting Heart Rate (bpm)", 40, 120, 70)
        with w_col2:
            sleep_hrs  = st.slider("Avg Sleep (hours/night)", 3, 12, 7)
        with w_col3:
            daily_steps = st.slider("Daily Steps", 0, 20000, 7000)

        wearable_summary = f"Resting HR: {resting_hr} bpm, Sleep: {sleep_hrs} hrs/night, Daily steps: {daily_steps}"

        # Quick wearable health indicators
        w_flags = []
        if resting_hr > 100: w_flags.append("⚠️ Resting HR is elevated (>100 bpm)")
        if sleep_hrs < 6:    w_flags.append("⚠️ Sleep is below recommended (< 6 hrs)")
        if daily_steps < 5000: w_flags.append("⚠️ Daily steps are low (< 5,000)")

        if w_flags:
            for f in w_flags:
                st.warning(f)
        else:
            st.success("✅ Wearable metrics look healthy")

    # ── Lab report upload ─────────────────────────────────────────────────────
    st.subheader("Upload Health Data")
    uploaded_file = st.file_uploader(
        "Upload a lab report PDF or health CSV",
        type=["csv", "txt", "pdf"]
    )

    if uploaded_file is not None:
        st.success("File uploaded successfully!")

        if uploaded_file.name.endswith(".csv"):
            lab_df = pd.read_csv(uploaded_file)
            st.write("Lab Data Preview:")
            st.dataframe(lab_df)

            if "cholesterol" in lab_df.columns:
                avg_chol = lab_df["cholesterol"].mean()
                if avg_chol > 240:
                    st.warning(f"⚠️ Average cholesterol is high ({avg_chol:.0f} mg/dl)")
                else:
                    st.success(f"✅ Average cholesterol looks normal ({avg_chol:.0f} mg/dl)")

        elif uploaded_file.name.endswith(".pdf"):
            reader   = PdfReader(uploaded_file)
            pdf_text = "".join(page.extract_text() or "" for page in reader.pages)

            st.subheader("Extracted PDF Text")
            st.text_area("PDF Text", pdf_text, height=200)

            if st.button("Analyze PDF with AI"):
                client = OpenAI()
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{
                        "role": "user",
                        "content": (
                            "Analyze this lab report in simple language. "
                            "Identify abnormal values, possible health risks, and suggested next steps. "
                            "Do not diagnose. Tell the user to consult a doctor.\n\n"
                            f"Lab report:\n{pdf_text[:3000]}"  # limit to avoid token overflow
                        )
                    }]
                )
                st.subheader("AI Lab Analysis")
                st.write(response.choices[0].message.content)

    # ── Predict button ────────────────────────────────────────────────────────
    st.divider()
    if st.button("🔍 Predict My Risk", use_container_width=True):

        user_data = {
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
            "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
            "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
        }

        result      = predict_risk(user_data)
        shap_result = explain_risk(user_data)

        # Risk result
        if result["risk"] == "High Risk":
            st.error(f"⚠️ Risk Level: {result['risk']}")
        else:
            st.success(f"✅ Risk Level: {result['risk']}")

        st.metric("Probability of Heart Disease", f"{result['probability']}%")
        st.caption("This is not medical advice. Please consult a doctor.")

        # ── SHAP explanation chart ────────────────────────────────────────────
        st.subheader("Why this score? — Top Contributing Factors")

        features = [d["feature"] for d in shap_result]
        values   = [d["value"]   for d in shap_result]
        colors   = ["#e74c3c" if v > 0 else "#2ecc71" for v in values]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(features[::-1], values[::-1], color=colors[::-1])
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("SHAP Value (impact on risk score)")
        ax.set_title("Feature Contribution to Your Risk Score")

        # Add value labels on bars
        for bar, val in zip(bars, values[::-1]):
            ax.text(
                bar.get_width() + (0.002 if val >= 0 else -0.002),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}",
                va="center",
                ha="left" if val >= 0 else "right",
                fontsize=8
            )

        ax.legend(
            handles=[
                plt.Rectangle((0,0),1,1, color="#e74c3c", label="Increases risk"),
                plt.Rectangle((0,0),1,1, color="#2ecc71", label="Reduces risk"),
            ],
            loc="lower right", fontsize=8
        )
        fig.tight_layout()
        st.pyplot(fig)

        # ── Personalized recommendations ──────────────────────────────────────
        st.subheader("Personalized Recommendations")

        recs = []
        if chol > 240:      recs.append("🔴 Cholesterol is high — consider dietary changes and speak to your doctor")
        if trestbps > 130:  recs.append("🔴 Blood pressure is elevated — reduce sodium and monitor regularly")
        if age > 50:        recs.append("🟡 Age is a risk factor — regular cardiac check-ups are important")
        if resting_hr > 100: recs.append("🟡 Resting heart rate is high — limit caffeine and manage stress")
        if sleep_hrs < 6:   recs.append("🟡 Poor sleep affects heart health — aim for 7–9 hours")
        if daily_steps < 5000: recs.append("🟡 Low activity level — aim for 8,000–10,000 steps daily")

        if not recs:
            st.success("Your metrics look good! Keep up the healthy habits.")
        else:
            for rec in recs:
                st.write(rec)

        st.write("**Suggested next steps:**")
        st.write("- Talk to a cardiologist if risk is high")
        st.write("- Track blood pressure weekly")
        st.write("- Improve diet and exercise habits")

        # ── Save to history ───────────────────────────────────────────────────
        new_row = pd.DataFrame([{
            "time":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "age":         age,
            "sex":         sex_label,
            "risk":        result["risk"],
            "probability": result["probability"]
        }])

        if os.path.exists(HISTORY_FILE):
            new_row.to_csv(HISTORY_FILE, mode="a", header=False, index=False)
        else:
            new_row.to_csv(HISTORY_FILE, mode="w", header=True, index=False)

        st.success("✅ Prediction saved to history!")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — HISTORY
# ══════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.subheader("Your Prediction History")

    if not os.path.exists(HISTORY_FILE):
        st.info("No predictions yet. Run a prediction first!")
    else:
        df_hist = pd.read_csv(HISTORY_FILE, names=HISTORY_COLS, header=None)

        # Drop rows where 'time' literally says 'time' (old headerless files)
        df_hist = df_hist[df_hist["time"] != "time"].reset_index(drop=True)
        df_hist["probability"] = pd.to_numeric(df_hist["probability"], errors="coerce")
        df_hist["time"]        = pd.to_datetime(df_hist["time"], errors="coerce")

        # Summary stats
        s_col1, s_col2, s_col3 = st.columns(3)
        s_col1.metric("Total Predictions", len(df_hist))
        s_col2.metric("Avg Probability", f"{df_hist['probability'].mean():.1f}%")

        if len(df_hist) >= 2:
            trend = df_hist["probability"].iloc[-1] - df_hist["probability"].iloc[-2]
            s_col3.metric("Trend vs Last", f"{trend:+.1f}%",
                          delta_color="inverse")  # red = going up (bad), green = going down (good)

        # Line chart
        st.subheader("Risk Probability Over Time")
        chart_data = df_hist.set_index("time")[["probability"]]
        st.line_chart(chart_data)

        # Color-coded table
        st.subheader("All Predictions")

        def highlight_risk(row):
            color = "#fde8e8" if row["risk"] == "High Risk" else "#e8fde8"
            return [f"background-color: {color}"] * len(row)

        st.dataframe(
            df_hist.sort_values("time", ascending=False)
                   .style.apply(highlight_risk, axis=1),
            use_container_width=True
        )

        # Clear history
        st.divider()
        if st.button("🗑️ Clear History"):
            confirm = st.checkbox("Yes, I want to permanently delete my history")
            if confirm:
                os.remove(HISTORY_FILE)
                st.success("History cleared.")
                st.rerun()