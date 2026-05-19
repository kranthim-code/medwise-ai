import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)


# ── Color palette ─────────────────────────────────────────────────────────────
RED    = colors.HexColor("#e74c3c")
GREEN  = colors.HexColor("#27ae60")
YELLOW = colors.HexColor("#f39c12")
BLUE   = colors.HexColor("#2980b9")
LIGHT_GRAY = colors.HexColor("#f5f5f5")
DARK_GRAY  = colors.HexColor("#2c3e50")


def build_doctor_report(
    patient_data: dict,
    risk_result: dict,
    shap_result: list[dict],
    ai_recommendations: str,
    lab_results: list[dict] = None,
) -> bytes:
    """
    Generate a doctor handoff PDF report and return it as bytes.

    Args:
        patient_data:       dict of raw health inputs (age, sex, chol, etc.)
        risk_result:        dict with keys 'risk' and 'probability'
        shap_result:        list of SHAP dicts from explain_risk()
        ai_recommendations: string from GPT/Claude personalized recs
        lab_results:        optional list of dicts from extract_lab_values()

    Returns:
        PDF as bytes (ready for st.download_button)
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()
    story  = []

    # ── Helper styles ─────────────────────────────────────────────────────────
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=22,
        textColor=DARK_GRAY,
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.gray,
        spaceAfter=12,
    )
    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=BLUE,
        spaceBefore=14,
        spaceAfter=6,
        borderPad=2,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=10,
        leading=15,
        textColor=DARK_GRAY,
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.gray,
        leading=12,
    )

    # ── Header ────────────────────────────────────────────────────────────────
    story.append(Paragraph("MedWise AI", title_style))
    story.append(Paragraph("Personalized Heart Risk Report", subtitle_style))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
        subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=BLUE))
    story.append(Spacer(1, 10))

    # ── Risk Summary ──────────────────────────────────────────────────────────
    story.append(Paragraph("Risk Assessment", section_style))

    risk_color = RED if risk_result["risk"] == "High Risk" else GREEN
    risk_label = risk_result["risk"]
    probability = risk_result["probability"]

    risk_table = Table(
        [[
            Paragraph(f"<b>{risk_label}</b>", ParagraphStyle(
                "RiskLabel", parent=styles["Normal"],
                fontSize=18, textColor=risk_color
            )),
            Paragraph(
                f"<b>{probability}%</b> probability of heart disease",
                ParagraphStyle("Prob", parent=styles["Normal"],
                               fontSize=13, textColor=DARK_GRAY)
            ),
        ]],
        colWidths=[2.5 * inch, 4.5 * inch],
    )
    risk_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GRAY),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [LIGHT_GRAY]),
        ("BOX", (0, 0), (-1, -1), 1, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(risk_table)
    story.append(Spacer(1, 10))

    # ── Patient Profile ───────────────────────────────────────────────────────
    story.append(Paragraph("Patient Profile", section_style))

    label_map = {
        "age": "Age", "sex": "Sex", "cp": "Chest Pain Type",
        "trestbps": "Blood Pressure (mmHg)", "chol": "Cholesterol (mg/dL)",
        "fbs": "Fasting Blood Sugar >120", "restecg": "Resting ECG",
        "thalach": "Max Heart Rate", "exang": "Exercise Angina",
        "oldpeak": "ST Depression", "slope": "ST Slope",
        "ca": "Blocked Vessels", "thal": "Thalassemia Type",
    }
    sex_map = {0: "Female", 1: "Male"}

    profile_data = [["Parameter", "Value"]]
    for key, label in label_map.items():
        val = patient_data.get(key, "—")
        if key == "sex":
            val = sex_map.get(val, val)
        profile_data.append([label, str(val)])

    profile_table = Table(profile_data, colWidths=[3.5 * inch, 3.5 * inch])
    profile_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(profile_table)

    # ── Top Risk Factors (SHAP) ───────────────────────────────────────────────
    story.append(Paragraph("Top Contributing Risk Factors", section_style))
    story.append(Paragraph(
        "These are the health factors that most influenced your risk score:",
        body_style
    ))
    story.append(Spacer(1, 6))

    shap_data = [["Factor", "Impact", "Direction"]]
    for item in shap_result[:7]:
        direction_color = RED if item["direction"] == "increases risk" else GREEN
        shap_data.append([
            item["feature"],
            f"{item['value']:+.4f}",
            item["direction"].title(),
        ])

    shap_table = Table(shap_data, colWidths=[2.8 * inch, 1.5 * inch, 2.7 * inch])
    shap_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(shap_table)

    # ── Lab Results (optional) ────────────────────────────────────────────────
    if lab_results:
        story.append(Paragraph("Lab Results", section_style))
        lab_data = [["Lab Test", "Your Value", "Normal Range", "Status"]]
        for r in lab_results:
            lab_data.append([
                r["Lab Test"],
                f"{r['Your Value']} {r['Unit']}",
                r["Normal Range"],
                r["Status"],
            ])

        lab_table = Table(lab_data, colWidths=[2 * inch, 1.5 * inch, 1.8 * inch, 1.7 * inch])
        lab_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), BLUE),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(lab_table)

    # ── AI Recommendations ────────────────────────────────────────────────────
    story.append(Paragraph("Personalized Recommendations", section_style))
    for line in ai_recommendations.split("\n"):
        line = line.strip()
        if line:
            story.append(Paragraph(line, body_style))
            story.append(Spacer(1, 4))

    # ── Disclaimer ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "DISCLAIMER: This report is generated by an AI system and is intended for "
        "informational purposes only. It does not constitute medical advice, diagnosis, "
        "or treatment. Always consult a qualified healthcare professional before making "
        "any medical decisions. MedWise AI is not a substitute for professional medical care.",
        disclaimer_style
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
    