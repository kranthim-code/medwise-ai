import re

# Clinical reference ranges: {lab_name: (min, max, unit)}
REFERENCE_RANGES = {
    "LDL":         (0,   100,  "mg/dL"),
    "HDL":         (40,  999,  "mg/dL"),
    "Cholesterol": (0,   200,  "mg/dL"),
    "Triglycerides":(0,  150,  "mg/dL"),
    "Glucose":     (70,  100,  "mg/dL"),
    "HbA1c":       (0,   5.7,  "%"),
    "Hemoglobin":  (12,  17.5, "g/dL"),
    "Sodium":      (136, 145,  "mEq/L"),
    "Potassium":   (3.5, 5.0,  "mEq/L"),
    "Creatinine":  (0.6, 1.2,  "mg/dL"),
    "BUN":         (7,   20,   "mg/dL"),
    "ALT":         (0,   56,   "U/L"),
    "AST":         (0,   40,   "U/L"),
    "TSH":         (0.4, 4.0,  "mIU/L"),
    "WBC":         (4.5, 11.0, "K/uL"),
    "RBC":         (4.2, 5.9,  "M/uL"),
    "Platelets":   (150, 400,  "K/uL"),
}

# Regex patterns to extract values from PDF text
# Matches patterns like "LDL: 130 mg/dL" or "LDL 130" or "LDL......130"
PATTERNS = {
    name: re.compile(
        rf"{re.escape(name)}\s*[:\.\-]?\s*(\d+\.?\d*)",
        re.IGNORECASE
    )
    for name in REFERENCE_RANGES
}


def extract_lab_values(text: str) -> list[dict]:
    """
    Extract lab values from raw PDF text and compare to reference ranges.

    Args:
        text: Raw text extracted from a lab report PDF

    Returns:
        List of dicts with keys: name, value, unit, min, max, status, flag
        status: "Normal" | "High" | "Low"
        flag: "✅" | "🔴" | "🟡"
    """
    results = []

    for name, pattern in PATTERNS.items():
        match = pattern.search(text)
        if not match:
            continue

        try:
            value = float(match.group(1))
        except ValueError:
            continue

        low, high, unit = REFERENCE_RANGES[name]

        if value < low:
            status = "Low"
            flag   = "🟡"
        elif value > high:
            status = "High"
            flag   = "🔴"
        else:
            status = "Normal"
            flag   = "✅"

        results.append({
            "Lab Test":       name,
            "Your Value":     value,
            "Unit":           unit,
            "Normal Range":   f"{low} – {high}",
            "Status":         f"{flag} {status}",
        })

    return results


def summarize_for_ai(lab_results: list[dict]) -> str:
    """
    Format extracted lab results as a clean summary string for the AI prompt.
    """
    if not lab_results:
        return "No structured lab values could be extracted."

    lines = ["Extracted Lab Results:"]
    for r in lab_results:
        lines.append(
            f"- {r['Lab Test']}: {r['Your Value']} {r['Unit']} "
            f"(Normal: {r['Normal Range']}) → {r['Status']}"
        )
    return "\n".join(lines)
    