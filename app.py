"""
Fair Loan Approval Predictor — Flask Backend
SDG 10: Reduced Inequalities
"""

import json
import os
import pickle

import numpy as np
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────
# Load Model Artifacts
# ─────────────────────────────────────────────────────────────

MODEL, SCALER, ENCODERS, METADATA = None, None, None, {}

def load_artifacts():
    global MODEL, SCALER, ENCODERS, METADATA
    with open("model/model.pkl", "rb") as f:
        MODEL = pickle.load(f)
    with open("model/scaler.pkl", "rb") as f:
        SCALER = pickle.load(f)
    with open("model/encoders.pkl", "rb") as f:
        ENCODERS = pickle.load(f)
    with open("model/metadata.json") as f:
        METADATA = json.load(f)


# ─────────────────────────────────────────────────────────────
# Validation Rules
# ─────────────────────────────────────────────────────────────

VALID_GENDER     = {"Male", "Female"}
VALID_MARRIED    = {"Yes", "No"}
VALID_DEPENDENTS = {"0", "1", "2", "3+"}
VALID_EDUCATION  = {"Graduate", "Not Graduate"}
VALID_SELF_EMP   = {"Yes", "No"}
VALID_AREA       = {"Urban", "Semiurban", "Rural"}
VALID_TERMS      = {120, 180, 240, 300, 360}

INCOME_MIN, INCOME_MAX       = 10_000, 500_000     # Rs.10K – Rs.5L/month
COINCOME_MIN, COINCOME_MAX   = 0, 300_000          # Rs.0   – Rs.3L/month
LOAN_MIN, LOAN_MAX           = 500, 20_000         # Rs.5L  – Rs.2Cr (in thousands)
TERM_VALUES                  = [120, 180, 240, 300, 360]


def validate_input(data: dict) -> list[str]:
    errors = []

    if data.get("gender") not in VALID_GENDER:
        errors.append(f"Gender must be one of: {', '.join(VALID_GENDER)}")
    if data.get("married") not in VALID_MARRIED:
        errors.append(f"Married must be Yes or No")
    if data.get("dependents") not in VALID_DEPENDENTS:
        errors.append(f"Dependents must be 0, 1, 2, or 3+")
    if data.get("education") not in VALID_EDUCATION:
        errors.append(f"Education must be Graduate or Not Graduate")
    if data.get("self_employed") not in VALID_SELF_EMP:
        errors.append(f"Self-Employed must be Yes or No")
    if data.get("property_area") not in VALID_AREA:
        errors.append(f"Property Area must be Urban, Semiurban, or Rural")

    # Numeric validations
    try:
        ai = float(data["applicant_income"])
        if not (INCOME_MIN <= ai <= INCOME_MAX):
            errors.append(f"Applicant Income must be between Rs.{INCOME_MIN:,} and Rs.{INCOME_MAX:,}")
    except (ValueError, KeyError):
        errors.append("Applicant Income must be a valid number")

    try:
        ca = float(data["coapplicant_income"])
        if not (COINCOME_MIN <= ca <= COINCOME_MAX):
            errors.append(f"Co-applicant Income must be between Rs.0 and Rs.{COINCOME_MAX:,}")
    except (ValueError, KeyError):
        errors.append("Co-applicant Income must be a valid number")

    try:
        la = float(data["loan_amount"])
        if not (LOAN_MIN <= la <= LOAN_MAX):
            errors.append(f"Loan Amount must be between Rs.{LOAN_MIN}K and Rs.{LOAN_MAX}K")
    except (ValueError, KeyError):
        errors.append("Loan Amount must be a valid number")

    try:
        lt = int(float(data["loan_term"]))
        if lt not in VALID_TERMS:
            errors.append(f"Loan Term must be one of: {', '.join(str(t) for t in TERM_VALUES)} months")
    except (ValueError, KeyError):
        errors.append("Loan Term must be a valid number")

    try:
        ch = float(data["credit_history"])
        if ch not in (0.0, 1.0):
            errors.append("Credit History must be 0 or 1")
    except (ValueError, KeyError):
        errors.append("Credit History must be 0 or 1")

    # Business logic checks
    if not errors:
        ai  = float(data["applicant_income"])
        la  = float(data["loan_amount"])
        lt  = int(float(data["loan_term"]))
        emi = (la * 1000) / lt
        if emi > ai * 0.9:
            errors.append(
                f"EMI (Rs.{emi:,.0f}/month) exceeds 90% of applicant income (Rs.{ai:,.0f}). "
                "Please reduce loan amount or choose a longer term."
            )

    return errors


# ─────────────────────────────────────────────────────────────
# Feature Engineering (mirrors train_model.py)
# ─────────────────────────────────────────────────────────────

def build_features(data: dict) -> np.ndarray:
    ai  = float(data["applicant_income"])
    ca  = float(data["coapplicant_income"])
    la  = float(data["loan_amount"])
    lt  = float(data["loan_term"])
    ch  = float(data["credit_history"])

    total_income   = ai + ca
    loan_to_income = la / (total_income / 12) if total_income > 0 else 0
    emi            = la / lt if lt > 0 else 0

    cat_vals = [
        data["gender"],
        data["married"],
        data["dependents"],
        data["education"],
        data["self_employed"],
        data["property_area"],
    ]
    cat_col_names = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]

    cat_enc = []
    for col, val in zip(cat_col_names, cat_vals):
        le = ENCODERS[col]
        try:
            encoded = int(le.transform([val])[0])
        except ValueError:
            encoded = 0
        cat_enc.append(encoded)

    num_enc = [ai, ca, la, lt, ch, total_income, loan_to_income, emi]
    features = np.array(cat_enc + num_enc, dtype=float).reshape(1, -1)
    return SCALER.transform(features)


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", metadata=METADATA)


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    errors = validate_input(data)
    if errors:
        return jsonify({"success": False, "errors": errors}), 400

    features = build_features(data)
    prediction = int(MODEL.predict(features)[0])
    proba      = MODEL.predict_proba(features)[0].tolist()

    confidence    = round(proba[prediction] * 100, 1)
    approval_prob = round(proba[1] * 100, 1)

    # Generate explainability hints
    ai  = float(data["applicant_income"])
    ca  = float(data["coapplicant_income"])
    la  = float(data["loan_amount"])
    lt  = float(data["loan_term"])
    ch  = float(data["credit_history"])
    total_income   = ai + ca
    loan_to_income = la / (total_income / 12) if total_income > 0 else 0
    emi            = (la * 1000) / lt if lt > 0 else 0

    factors = []
    if ch == 1.0:
        factors.append({"factor": "Credit History", "impact": "positive", "note": "Good credit history strongly supports approval"})
    else:
        factors.append({"factor": "Credit History", "impact": "negative", "note": "No/bad credit history is the strongest rejection signal"})

    if loan_to_income < 3:
        factors.append({"factor": "Loan-to-Income Ratio", "impact": "positive", "note": f"Ratio of {loan_to_income:.1f} is healthy (< 3)"})
    elif loan_to_income < 5:
        factors.append({"factor": "Loan-to-Income Ratio", "impact": "neutral", "note": f"Ratio of {loan_to_income:.1f} is moderate"})
    else:
        factors.append({"factor": "Loan-to-Income Ratio", "impact": "negative", "note": f"Ratio of {loan_to_income:.1f} is high (> 5)"})

    if total_income > 80000:
        factors.append({"factor": "Combined Income", "impact": "positive", "note": f"Strong combined income of Rs.{total_income:,.0f}"})
    elif total_income < 20000:
        factors.append({"factor": "Combined Income", "impact": "negative", "note": f"Low combined income of Rs.{total_income:,.0f}"})

    if data["education"] == "Graduate":
        factors.append({"factor": "Education", "impact": "positive", "note": "Graduate status improves approval likelihood"})

    if data["property_area"] == "Semiurban":
        factors.append({"factor": "Property Area", "impact": "positive", "note": "Semiurban properties have highest approval rates"})

    emi_ratio = (emi / ai * 100) if ai > 0 else 0
    if emi_ratio > 50:
        factors.append({"factor": "EMI Burden", "impact": "negative", "note": f"EMI is {emi_ratio:.0f}% of monthly income — very high"})
    elif emi_ratio < 30:
        factors.append({"factor": "EMI Burden", "impact": "positive", "note": f"EMI is {emi_ratio:.0f}% of monthly income — manageable"})

    return jsonify({
        "success":      True,
        "approved":     prediction == 1,
        "prediction":   "Approved" if prediction == 1 else "Rejected",
        "confidence":   confidence,
        "approval_prob": approval_prob,
        "factors":      factors,
        "derived": {
            "total_income":    round(total_income, 0),
            "loan_to_income":  round(loan_to_income, 2),
            "emi_monthly":     round(emi, 0),
            "emi_ratio":       round(emi_ratio, 1),
        }
    })


@app.route("/api/fairness")
def fairness():
    return jsonify(METADATA.get("fairness_report", {}))


@app.route("/api/metadata")
def metadata():
    return jsonify(METADATA)


@app.route("/api/stats")
def stats():
    """Dataset-level statistics for dashboard cards."""
    return jsonify({
        "dataset_size":      METADATA.get("dataset_size", 0),
        "approval_rate":     METADATA.get("approval_rate", 0),
        "best_model":        METADATA.get("best_model", ""),
        "model_results":     METADATA.get("model_results", {}),
        "feature_importances": METADATA.get("feature_importances", {}),
        "fairness_report":   METADATA.get("fairness_report", {}),
    })


# ─────────────────────────────────────────────────────────────
# Entry Point — load artifacts at module level for gunicorn
# ─────────────────────────────────────────────────────────────

if os.path.exists("model/model.pkl"):
    load_artifacts()

if __name__ == "__main__":
    if not os.path.exists("model/model.pkl"):
        print("[ERROR] Model not found. Run: python train_model.py")
        exit(1)
    print("\n  Fair Loan Predictor running at http://127.0.0.1:5000\n")
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
