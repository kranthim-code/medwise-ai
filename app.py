import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from predict import predict_risk, explain_risk
from pypdf import PdfReader
from openai import OpenAI
from lab_parser import extract_lab_values, summarize_for_ai
from report_generator import build_doctor_report

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

    def slider_input(label, min_val, max_val, default, key, step=1, fmt="%d"):
        """Render a slider and a number input that stay in sync."""
        s_col, n_col = st.columns([3, 1])
        with s_col:
            slider_val = st.slider(label, min_val, max_val, default,
                                   step=step, key=f"{key}_slider",
                                   label_visibility="visible")
        with n_col:
            st.write("")  # vertical align
            num_val = st.number_input(
                "value", min_value=float(min_val), max_value=float(max_val),
                value=float(slider_val), step=float(step),
                key=f"{key}_num", label_visibility="collapsed"
            )
        # Number input wins if different from slider (user typed)
        return int(num_val) if step == 1 else round(float(num_val), 1)

    with col1:
        age        = slider_input("Age", 20, 80, 45, "age")
        sex_label  = st.selectbox("Sex", ["Female", "Male"])
        sex        = 1 if sex_label == "Male" else 0
        cp         = st.selectbox("Chest Pain Type",
                                  [0, 1, 2, 3],
                                  format_func=lambda x: {
                                      0: "0 – Typical Angina",
                                      1: "1 – Atypical Angina",
                                      2: "2 – Non-Anginal Pain",
                                      3: "3 – Asymptomatic"}[x])
        trestbps   = slider_input("Resting Blood Pressure (mmHg)", 80, 200, 120, "trestbps")
        chol       = slider_input("Cholesterol (mg/dl)", 100, 400, 200, "chol")

    with col2:
        fbs_label  = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
        fbs        = 1 if fbs_label == "Yes" else 0
        restecg    = st.selectbox("Resting ECG",
                                  [0, 1, 2],
                                  format_func=lambda x: {
                                      0: "0 – Normal",
                                      1: "1 – ST-T Abnormality",
                                      2: "2 – Left Ventricular Hypertrophy"}[x])
        thalach    = slider_input("Max Heart Rate Achieved", 60, 200, 150, "thalach")
        exang_label = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
        exang      = 1 if exang_label == "Yes" else 0

    with col3:
        oldpeak    = slider_input("ST Depression (Oldpeak)", 0.0, 5.0, 1.0, "oldpeak", step=0.1)
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
            resting_hr  = slider_input("Resting Heart Rate (bpm)", 40, 120, 70, "rhr")
        with w_col2:
            sleep_hrs   = slider_input("Avg Sleep (hours/night)", 3, 12, 7, "sleep")
        with w_col3:
            daily_steps = slider_input("Daily Steps", 0, 20000, 7000, "steps", step=100)

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

            # ── Structured lab value extraction ───────────────────────────────
            st.subheader("📊 Extracted Lab Values")
            lab_results = extract_lab_values(pdf_text)

            if lab_results:
                lab_df = pd.DataFrame(lab_results)
                st.session_state["lab_results"] = lab_results

                def color_status(val):
                    if "🔴" in str(val): return "background-color: #fde8e8"
                    if "🟡" in str(val): return "background-color: #fff8e1"
                    if "✅" in str(val): return "background-color: #e8fde8"
                    return ""

                st.dataframe(
                    lab_df.style.applymap(color_status, subset=["Status"]),
                    use_container_width=True
                )

                abnormal = [r for r in lab_results if "Normal" not in r["Status"]]
                if abnormal:
                    st.warning(f"⚠️ {len(abnormal)} abnormal value(s) detected — see highlighted rows above")
                else:
                    st.success("✅ All detected lab values are within normal range")
            else:
                st.info("No standard lab values detected. Showing raw text below.")
                st.text_area("PDF Text", pdf_text, height=200)

            # ── AI analysis using structured data ─────────────────────────────
            if st.button("🤖 Analyze with AI"):
                client = OpenAI()
                structured_summary = summarize_for_ai(lab_results)

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{
                        "role": "user",
                        "content": (
                            "You are a helpful health assistant. Analyze these lab results in simple language. "
                            "Explain what each abnormal value means, possible health risks, and suggested next steps. "
                            "Do NOT diagnose. Always tell the user to consult a doctor.\n\n"
                            f"{structured_summary}\n\n"
                            f"Additional raw report context:\n{pdf_text[:1500]}"
                        )
                    }]
                )
                st.subheader("🤖 AI Lab Analysis")
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



        # ── Personalized Recommendations (rule-based) ─────────────────────────
        st.subheader("📋 Personalized Recommendations")

        recs = []
        if chol > 240:
            recs.append("🔴 Your cholesterol is high — reduce saturated fats, increase fiber, and ask your doctor about medication options.")
        elif chol > 200:
            recs.append("🟡 Your cholesterol is borderline — consider dietary changes like cutting fried foods and increasing vegetables.")
        if trestbps > 140:
            recs.append("🔴 Your blood pressure is high — reduce sodium intake, limit alcohol, and monitor it weekly.")
        elif trestbps > 120:
            recs.append("🟡 Your blood pressure is elevated — try reducing stress and cutting back on salty foods.")
        if resting_hr > 100:
            recs.append("🔴 Your resting heart rate is high — limit caffeine, manage stress, and discuss with your doctor.")
        if sleep_hrs < 6:
            recs.append("🟡 You're getting less than 6 hours of sleep — aim for 7–9 hours as poor sleep increases heart risk.")
        if daily_steps < 5000:
            recs.append("🟡 Your daily activity is low — try to reach 8,000–10,000 steps per day with short walks.")
        if age > 50:
            recs.append("🟡 Age is a risk factor — schedule regular cardiac check-ups at least once a year.")
        if result["risk"] == "High Risk":
            recs.append("🔴 Your overall risk is high — please consult a cardiologist soon.")

        # Always add top SHAP factor insight
        top = shap_result[0]
        recs.append(f"🔵 The biggest factor in your score is **{top['feature']}** — speak to your doctor about this specifically.")

        if not recs:
            recs.append("✅ Your metrics look good overall — keep up the healthy habits and get regular check-ups.")

        recs.append("👨‍⚕️ This is not medical advice. Always consult a qualified healthcare professional before making any health decisions.")

        ai_recommendations = "\n".join(recs)
        for rec in recs:
            st.write(rec)

        # ── Doctor Handoff PDF ─────────────────────────────────────────────────
        st.subheader("📄 Doctor Handoff Report")
        st.write("Download a PDF summary to share with your doctor.")

        lab_results_for_report = st.session_state.get("lab_results", None)

        pdf_bytes = build_doctor_report(
            patient_data=user_data,
            risk_result=result,
            shap_result=shap_result,
            ai_recommendations=ai_recommendations,
            lab_results=lab_results_for_report,
        )

        st.download_button(
            label="⬇️ Download Doctor Report (PDF)",
            data=pdf_bytes,
            file_name=f"medwise_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

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
                