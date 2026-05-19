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

HISTORY_FILE = "prediction_history.csv"
HISTORY_COLS = ["time", "age", "sex", "risk", "probability"]

# Population average risk by age/sex group
POPULATION_RISK = {
    ("Female", "20-40"): 15, ("Female", "41-55"): 25, ("Female", "56+"): 38,
    ("Male",   "20-40"): 28, ("Male",   "41-55"): 42, ("Male",   "56+"): 55,
}

def get_age_group(age):
    if age <= 40: return "20-40"
    if age <= 55: return "41-55"
    return "56+"

def calc_bmi(weight_kg, height_cm):
    h = height_cm / 100
    return round(weight_kg / (h * h), 1)

def bmi_category(bmi):
    if bmi < 18.5: return "Underweight", "#3498db"
    if bmi < 25:   return "Normal", "#2ecc71"
    if bmi < 30:   return "Overweight", "#f39c12"
    return "Obese", "#e74c3c"

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/color/96/heart-with-pulse.png", width=80)
    st.title("MedWise AI")
    st.caption("AI-powered heart health insights")
    st.divider()

    st.markdown("### How It Works")
    st.markdown("""
1. Enter your health data
2. Hit **Predict My Risk**
3. See your risk score + what's driving it
4. Download your doctor report
""")
    st.divider()

    st.markdown("### Healthy Ranges")
    st.markdown("""
| Metric | Healthy Range |
|--------|--------------|
| Blood Pressure | < 120 mmHg |
| Cholesterol | < 200 mg/dL |
| BMI | 18.5 - 24.9 |
| Resting HR | 60-100 bpm |
| Sleep | 7-9 hrs/night |
| Daily Steps | 8,000-10,000 |
""")
    st.divider()
    st.caption("This app is for informational purposes only. Always consult a doctor.")
    st.caption("Built by Kranthi Muthavarapu & Akash Inumella")

# ══════════════════════════════════════════════════════════════════════════════
# HERO SECTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='background: linear-gradient(135deg, #1a6e8a 0%, #27ae60 100%);
     padding: 2.5rem 2rem 2rem 2rem; border-radius: 16px; margin-bottom: 1.5rem;
     box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
    <h1 style='margin:0; color:white; font-size:2.4rem; font-weight:800;'>MedWise AI 🩺</h1>
    <p style='margin:0.5rem 0 0 0; color:rgba(255,255,255,0.85); font-size:1.1rem;'>
        Understand your heart health risk in minutes - powered by machine learning
    </p>
    <div style='display:flex; gap:2.5rem; margin-top:1.5rem; flex-wrap:wrap;'>
        <div style='text-align:center;'>
            <div style='font-size:1.8rem; font-weight:bold; color:white;'>303</div>
            <div style='font-size:0.8rem; color:rgba(255,255,255,0.75);'>Patients trained on</div>
        </div>
        <div style='text-align:center;'>
            <div style='font-size:1.8rem; font-weight:bold; color:white;'>13</div>
            <div style='font-size:0.8rem; color:rgba(255,255,255,0.75);'>Health features</div>
        </div>
        <div style='text-align:center;'>
            <div style='font-size:1.8rem; font-weight:bold; color:white;'>17</div>
            <div style='font-size:0.8rem; color:rgba(255,255,255,0.75);'>Lab tests parsed</div>
        </div>
        <div style='text-align:center;'>
            <div style='font-size:1.8rem; font-weight:bold; color:white;'>Free</div>
            <div style='font-size:0.8rem; color:rgba(255,255,255,0.75);'>No account needed</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_predict, tab_history = st.tabs(["🔍 Predict My Risk", "📈 My History"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:

    input_col, result_col = st.columns([1, 1], gap="large")

    with input_col:

        # ── GROUP 1: Basic Info ───────────────────────────────────────────────
        st.markdown("### 👤 Basic Information")
        with st.container():
            age       = st.slider("Age", 20, 80, 45)
            sex_label = st.selectbox("Sex", ["Female", "Male"])
            sex       = 1 if sex_label == "Male" else 0

            # BMI Calculator
            st.markdown("**BMI Calculator**")
            bmi_col1, bmi_col2 = st.columns(2)
            with bmi_col1:
                weight_kg = st.number_input("Weight (kg)", 30, 200, 70)
            with bmi_col2:
                height_cm = st.number_input("Height (cm)", 100, 220, 170)

            bmi = calc_bmi(weight_kg, height_cm)
            bmi_cat, bmi_color = bmi_category(bmi)
            st.markdown(f"""
<div style='background:#f8fbff; border-left:4px solid {bmi_color};
     border-radius:8px; padding:0.6rem 1rem; margin-top:0.3rem;'>
    <span style='font-weight:bold; color:{bmi_color};'>BMI: {bmi}</span>
    <span style='color:#666; font-size:0.9rem;'> - {bmi_cat}</span>
</div>
""", unsafe_allow_html=True)

        # ── GROUP 2: Heart Metrics ────────────────────────────────────────────
        st.markdown("### ❤️ Heart Metrics")
        with st.expander("Expand Heart Metrics", expanded=True):
            cp = st.selectbox("Chest Pain Type",
                              [0, 1, 2, 3],
                              format_func=lambda x: {
                                  0: "0 - Typical Angina",
                                  1: "1 - Atypical Angina",
                                  2: "2 - Non-Anginal Pain",
                                  3: "3 - Asymptomatic"}[x])
            trestbps = st.slider("Resting Blood Pressure (mmHg)", 80, 200, 120)
            thalach  = st.slider("Max Heart Rate Achieved", 60, 200, 150)
            oldpeak  = st.slider("ST Depression (Oldpeak)", 0.0, 5.0, 1.0, step=0.1)
            slope    = st.selectbox("ST Slope",
                                    [0, 1, 2],
                                    format_func=lambda x: {
                                        0: "0 - Upsloping",
                                        1: "1 - Flat",
                                        2: "2 - Downsloping"}[x])
            restecg  = st.selectbox("Resting ECG",
                                    [0, 1, 2],
                                    format_func=lambda x: {
                                        0: "0 - Normal",
                                        1: "1 - ST-T Abnormality",
                                        2: "2 - Left Ventricular Hypertrophy"}[x])
            exang_label = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
            exang    = 1 if exang_label == "Yes" else 0

        # ── GROUP 3: Blood Work ───────────────────────────────────────────────
        st.markdown("### 🧪 Blood Work")
        with st.expander("Expand Blood Work", expanded=True):
            chol      = st.slider("Cholesterol (mg/dl)", 100, 400, 200)
            fbs_label = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
            fbs       = 1 if fbs_label == "Yes" else 0
            ca        = st.selectbox("Blocked Vessels (CA)",
                                     [0, 1, 2, 3],
                                     format_func=lambda x: f"{x} vessel{'s' if x != 1 else ''}")
            thal      = st.selectbox("Thalassemia (Thal)",
                                     [0, 1, 2, 3],
                                     format_func=lambda x: {
                                         0: "0 - Normal",
                                         1: "1 - Fixed Defect",
                                         2: "2 - Reversible Defect",
                                         3: "3 - Unknown"}[x])

        # ── GROUP 4: Lifestyle ────────────────────────────────────────────────
        st.markdown("### 🏃 Lifestyle & Wearables")
        with st.expander("Expand Lifestyle Data"):
            resting_hr  = st.slider("Resting Heart Rate (bpm)", 40, 120, 70)
            sleep_hrs   = st.slider("Avg Sleep (hours/night)", 3, 12, 7)
            daily_steps = st.slider("Daily Steps", 0, 20000, 7000, step=100)

            w_flags = []
            if resting_hr > 100: w_flags.append("Resting HR is elevated (>100 bpm)")
            if sleep_hrs < 6:    w_flags.append("Sleep is below recommended (< 6 hrs)")
            if daily_steps < 5000: w_flags.append("Daily steps are low (< 5,000)")
            if bmi >= 30:        w_flags.append(f"BMI of {bmi} is in the obese range")
            elif bmi >= 25:      w_flags.append(f"BMI of {bmi} is in the overweight range")

            if w_flags:
                for f in w_flags:
                    st.warning(f"⚠️ {f}")
            else:
                st.success("✅ Lifestyle metrics look healthy")

        # ── GROUP 5: Manual Entry ─────────────────────────────────────────────
        with st.expander("✏️ Prefer to type values manually?"):
            st.caption("These override the sliders above if changed.")
            m1, m2 = st.columns(2)
            with m1:
                age      = st.number_input("Age", 20, 80, age, key="m_age")
                trestbps = st.number_input("Blood Pressure", 80, 200, trestbps, key="m_bp")
                chol     = st.number_input("Cholesterol", 100, 400, chol, key="m_chol")
                thalach  = st.number_input("Max Heart Rate", 60, 200, thalach, key="m_hr")
            with m2:
                oldpeak  = st.number_input("ST Depression", 0.0, 5.0, float(oldpeak), step=0.1, key="m_op")

        # ── Lab Upload ────────────────────────────────────────────────────────
        with st.expander("🧪 Upload Lab Report (Optional)"):
            uploaded_file = st.file_uploader(
                "Upload a lab report PDF or health CSV",
                type=["csv", "txt", "pdf"]
            )

            if uploaded_file is not None:
                st.success("File uploaded successfully!")

                if uploaded_file.name.endswith(".csv"):
                    lab_df = pd.read_csv(uploaded_file)
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
                            st.warning(f"⚠️ {len(abnormal)} abnormal value(s) detected")
                        else:
                            st.success("✅ All detected lab values are within normal range")
                    else:
                        st.info("No standard lab values detected.")
                        st.text_area("PDF Text", pdf_text, height=150)

        # ── Predict button ────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        predict_clicked = st.button("🔍 Predict My Risk", use_container_width=True, type="primary")

    # ── Results column ────────────────────────────────────────────────────────
    with result_col:
        st.markdown("### Results")

        if not predict_clicked:
            st.markdown("""
<div style='background:#f8fbff; border:2px dashed #b3d9f2; border-radius:12px;
     padding:3rem; text-align:center; color:#888; margin-top:2rem;'>
    <div style='font-size:3rem;'>🩺</div>
    <p style='margin-top:1rem; font-size:1rem;'>
        Fill in your health data and click<br><b>Predict My Risk</b> to see your results
    </p>
</div>
""", unsafe_allow_html=True)

        else:
            user_data = {
                "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
                "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
                "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
            }

            result      = predict_risk(user_data)
            shap_result = explain_risk(user_data)

            is_high     = result["risk"] == "High Risk"
            card_bg     = "#fff0f0" if is_high else "#f0fff4"
            card_border = "#e74c3c" if is_high else "#2ecc71"
            card_icon   = "⚠️" if is_high else "✅"
            card_color  = "#c0392b" if is_high else "#27ae60"

            # ── Risk Card ─────────────────────────────────────────────────────
            st.markdown(f"""
<div style='background:{card_bg}; border-left:6px solid {card_border};
     border-radius:12px; padding:1.5rem 2rem; margin-bottom:1rem;
     box-shadow: 0 2px 8px rgba(0,0,0,0.07);'>
    <div style='font-size:2rem; font-weight:bold; color:{card_color};'>
        {card_icon} {result['risk']}
    </div>
    <div style='font-size:0.9rem; color:#555; margin-top:0.3rem;'>
        Heart disease probability
    </div>
    <div style='font-size:3.5rem; font-weight:bold; color:{card_color}; margin-top:0.3rem; line-height:1;'>
        {result['probability']}%
    </div>
    <div style='font-size:0.8rem; color:#888; margin-top:0.5rem;'>
        Not medical advice. Please consult a doctor.
    </div>
</div>
""", unsafe_allow_html=True)

            # ── Risk Comparison ───────────────────────────────────────────────
            age_group  = get_age_group(age)
            pop_avg    = POPULATION_RISK.get((sex_label, age_group), 35)
            user_prob  = result["probability"]
            diff       = round(user_prob - pop_avg, 1)
            diff_color = "#e74c3c" if diff > 0 else "#27ae60"
            diff_label = f"+{diff}% higher" if diff > 0 else f"{abs(diff)}% lower"

            st.markdown(f"""
<div style='background:#f8fbff; border-radius:10px; padding:1rem 1.5rem;
     margin-bottom:1rem; border:1px solid #e1f0fb;'>
    <div style='font-size:0.85rem; color:#555; font-weight:600; margin-bottom:0.5rem;'>
        vs. Average for {sex_label}s aged {age_group}
    </div>
    <div style='display:flex; align-items:center; gap:1rem;'>
        <div style='text-align:center;'>
            <div style='font-size:1.4rem; font-weight:bold; color:{card_color};'>{user_prob}%</div>
            <div style='font-size:0.75rem; color:#888;'>Your risk</div>
        </div>
        <div style='font-size:1.5rem; color:#ccc;'>vs</div>
        <div style='text-align:center;'>
            <div style='font-size:1.4rem; font-weight:bold; color:#666;'>{pop_avg}%</div>
            <div style='font-size:0.75rem; color:#888;'>Population avg</div>
        </div>
        <div style='margin-left:auto; font-size:1rem; font-weight:bold; color:{diff_color};'>
            {diff_label} than average
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

            # ── BMI Summary in results ────────────────────────────────────────
            st.markdown(f"""
<div style='background:#f8fbff; border-left:4px solid {bmi_color};
     border-radius:8px; padding:0.6rem 1rem; margin-bottom:1rem;'>
    <span style='font-weight:bold; color:{bmi_color};'>BMI: {bmi}</span>
    <span style='color:#666; font-size:0.9rem;'> - {bmi_cat}</span>
    {"<span style='color:#e74c3c; font-size:0.85rem;'> - May increase heart risk</span>" if bmi >= 25 else "<span style='color:#27ae60; font-size:0.85rem;'> - Healthy weight</span>"}
</div>
""", unsafe_allow_html=True)

            # ── SHAP Chart ────────────────────────────────────────────────────
            st.markdown("#### What influenced your score?")

            top_shap = shap_result[:7]
            features = [d["feature"] for d in top_shap]
            values   = [d["value"]   for d in top_shap]
            colors   = ["#e74c3c" if v > 0 else "#2ecc71" for v in values]

            fig, ax = plt.subplots(figsize=(6, 3.5))
            fig.patch.set_facecolor("#f0f8ff")
            ax.set_facecolor("#f0f8ff")

            for i, (val, col) in enumerate(zip(values, colors)):
                ax.plot([0, val], [i, i], color=col, linewidth=2, alpha=0.8)
                ax.scatter(val, i, color=col, s=80, zorder=5)
                ax.text(val, i + 0.3, f"{val:+.3f}",
                        va="bottom", ha="center", fontsize=7.5, color="#1a2e3b")

            ax.axvline(0, color="#aaaaaa", linewidth=0.8, linestyle="--")
            ax.set_yticks(list(range(len(features))))
            ax.set_yticklabels(features, fontsize=9, color="#1a2e3b")
            ax.set_xlabel("SHAP Value", fontsize=9, color="#1a2e3b")
            ax.tick_params(axis="x", labelsize=8, colors="#1a2e3b")
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.spines["bottom"].set_color("#cccccc")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # ── Recommendations ───────────────────────────────────────────────
            st.markdown("#### Personalized Recommendations")

            recs = []
            if chol > 240:
                recs.append("Your cholesterol is high - reduce saturated fats, increase fiber, and ask your doctor about medication options.")
            elif chol > 200:
                recs.append("Your cholesterol is borderline - consider dietary changes like cutting fried foods and increasing vegetables.")
            if trestbps > 140:
                recs.append("Your blood pressure is high - reduce sodium intake, limit alcohol, and monitor it weekly.")
            elif trestbps > 120:
                recs.append("Your blood pressure is elevated - try reducing stress and cutting back on salty foods.")
            if bmi >= 30:
                recs.append(f"Your BMI of {bmi} is in the obese range - weight loss can significantly reduce heart disease risk.")
            elif bmi >= 25:
                recs.append(f"Your BMI of {bmi} is overweight - even a 5-10% weight reduction improves heart health.")
            if resting_hr > 100:
                recs.append("Your resting heart rate is high - limit caffeine, manage stress, and discuss with your doctor.")
            if sleep_hrs < 6:
                recs.append("You're getting less than 6 hours of sleep - aim for 7-9 hours as poor sleep increases heart risk.")
            if daily_steps < 5000:
                recs.append("Your daily activity is low - try to reach 8,000-10,000 steps per day with short walks.")
            if age > 50:
                recs.append("Age is a risk factor - schedule regular cardiac check-ups at least once a year.")
            if result["risk"] == "High Risk":
                recs.append("Your overall risk is high - please consult a cardiologist soon.")

            top = shap_result[0]
            recs.append(f"The biggest factor in your score is {top['feature']} - speak to your doctor about this specifically.")

            if not recs:
                recs.append("Your metrics look good overall - keep up the healthy habits and get regular check-ups.")

            recs.append("This is not medical advice. Always consult a qualified healthcare professional before making any health decisions.")

            ai_recommendations = "\n".join(f"• {r}" for r in recs)

            st.markdown("""
<div style='background:#f8fbff; border-radius:10px; padding:1rem 1.5rem; border:1px solid #e1f0fb;'>
""", unsafe_allow_html=True)
            for rec in recs:
                st.markdown(f"• {rec}")
            st.markdown("</div>", unsafe_allow_html=True)

            # ── Doctor PDF ────────────────────────────────────────────────────
            st.markdown("#### Doctor Handoff Report")
            st.caption("Download and bring this to your next appointment.")

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

            # ── Save to history ───────────────────────────────────────────────
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
    st.markdown("### Your Prediction History")

    if not os.path.exists(HISTORY_FILE):
        st.info("No predictions yet. Run a prediction first!")
    else:
        df_hist = pd.read_csv(HISTORY_FILE, names=HISTORY_COLS, header=None)
        df_hist = df_hist[df_hist["time"] != "time"].reset_index(drop=True)
        df_hist["probability"] = pd.to_numeric(df_hist["probability"], errors="coerce")
        df_hist["time"]        = pd.to_datetime(df_hist["time"], errors="coerce")

        s_col1, s_col2, s_col3 = st.columns(3)
        s_col1.metric("Total Predictions", len(df_hist))
        s_col2.metric("Avg Probability", f"{df_hist['probability'].mean():.1f}%")

        if len(df_hist) >= 2:
            trend = df_hist["probability"].iloc[-1] - df_hist["probability"].iloc[-2]
            s_col3.metric("Trend vs Last", f"{trend:+.1f}%", delta_color="inverse")

        st.markdown("#### Risk Probability Over Time")
        chart_data = df_hist.set_index("time")[["probability"]]
        st.line_chart(chart_data)

        st.markdown("#### All Predictions")

        def highlight_risk(row):
            color = "#fde8e8" if row["risk"] == "High Risk" else "#e8fde8"
            return [f"background-color: {color}"] * len(row)

        st.dataframe(
            df_hist.sort_values("time", ascending=False)
                   .style.apply(highlight_risk, axis=1),
            use_container_width=True
        )

        st.divider()
        if "confirm_clear" not in st.session_state:
            st.session_state.confirm_clear = False

        if st.button("🗑️ Clear History"):
            st.session_state.confirm_clear = True

        if st.session_state.confirm_clear:
            st.warning("Are you sure? This cannot be undone.")
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("✅ Yes, delete", key="confirm_yes"):
                    os.remove(HISTORY_FILE)
                    st.session_state.confirm_clear = False
                    st.success("History cleared.")
                    st.rerun()
            with col_no:
                if st.button("❌ Cancel", key="confirm_no"):
                    st.session_state.confirm_clear = False
                    st.rerun()