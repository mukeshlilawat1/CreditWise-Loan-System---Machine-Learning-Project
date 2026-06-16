
import os, json, logging
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")

model         = joblib.load(os.path.join(MODEL_DIR, "loan_model.pkl"))
scaler        = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
label_encoders= joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))
num_imputer   = joblib.load(os.path.join(MODEL_DIR, "num_imputer.pkl"))
cat_imputer   = joblib.load(os.path.join(MODEL_DIR, "cat_imputer.pkl"))

with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
    metadata = json.load(f)

FEATURE_NAMES   = metadata["feature_names"]
NUMERICAL_COLS  = metadata["numerical_cols"]
CATEGORICAL_COLS= metadata["categorical_cols"]

logger.info(f"✅ Model loaded | Accuracy={metadata['accuracy']}")


def _build_reasoning(probability: float, input_data: dict, risk: str) -> list[str]:
    reasons = []
    credit_score = float(input_data.get("Credit_Score", 600))
    dti          = float(input_data.get("DTI_Ratio", 0.5))
    income       = float(input_data.get("Applicant_Income", 0))
    savings      = float(input_data.get("Savings", 0))
    existing     = float(input_data.get("Existing_Loans", 0))
    loan_amount  = float(input_data.get("Loan_Amount", 0))
    emp_status   = input_data.get("Employment_Status", "")

    if credit_score >= 750:
        reasons.append("Excellent credit score strongly supports approval.")
    elif credit_score >= 650:
        reasons.append("Good credit score supports approval.")
    elif credit_score < 550:
        reasons.append("Low credit score is a significant risk factor.")

    if dti < 0.3:
        reasons.append("Low debt-to-income ratio indicates healthy finances.")
    elif dti > 0.6:
        reasons.append("High debt-to-income ratio raises repayment concerns.")

    if savings > loan_amount * 0.2:
        reasons.append("Adequate savings buffer mitigates repayment risk.")
    elif savings < loan_amount * 0.05:
        reasons.append("Insufficient savings relative to loan amount.")

    if emp_status in ("Salaried", "Government"):
        reasons.append("Stable employment status reduces default risk.")
    elif emp_status == "Unemployed":
        reasons.append("Unemployed status is a major risk concern.")

    if existing > 3:
        reasons.append("Multiple existing loans increase default probability.")

    return reasons or [f"Model confidence {probability:.1%} based on overall financial profile."]


def _risk_level(prob_approved: float) -> str:
    if prob_approved >= 0.80:  return "LOW"
    if prob_approved >= 0.55:  return "MEDIUM"
    if prob_approved >= 0.35:  return "HIGH"
    return "VERY_HIGH"


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "UP",
        "service": "CreditWise ML Service",
        "model": metadata["model_type"],
        "accuracy": metadata["accuracy"]
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        body = request.get_json(force=True)
        if not body:
            return jsonify({"error": "Request body is required"}), 400

        logger.info(f"Prediction request: {body}")

        # ── Build DataFrame with ONLY training features ──────────────
        row = {col: body.get(col, np.nan) for col in NUMERICAL_COLS + CATEGORICAL_COLS}
        df  = pd.DataFrame([row])

        # ── Impute ───────────────────────────────────────────────────
        df[NUMERICAL_COLS]  = num_imputer.transform(df[NUMERICAL_COLS])
        df[CATEGORICAL_COLS]= cat_imputer.transform(df[CATEGORICAL_COLS])

        # ── Encode ───────────────────────────────────────────────────
        for col in CATEGORICAL_COLS:
            le = label_encoders[col]
            val = df[col].values[0]
            if val in le.classes_:
                df[col] = le.transform([val])
            else:
                df[col] = le.transform([le.classes_[0]])  # fallback

        # ── Feature engineering ──────────────────────────────────────
        df["DTI_Ratio_sq"]          = df["DTI_Ratio"] ** 2
        df["Credit_Score_sq"]       = df["Credit_Score"] ** 2
        df["Applicant_Income_log"]  = np.log1p(df["Applicant_Income"])

        # ── Align to exact training feature order ────────────────────
        df = df[FEATURE_NAMES]

        # ── Scale & predict ──────────────────────────────────────────
        X_scaled = scaler.transform(df)
        proba     = model.predict_proba(X_scaled)[0]    # [prob_no, prob_yes]
        prob_approved = float(proba[1])
        prediction    = "APPROVED" if prob_approved >= 0.5 else "REJECTED"
        confidence    = round(max(proba) * 100, 2)
        risk          = _risk_level(prob_approved)
        reasons       = _build_reasoning(prob_approved, body, risk)

        result = {
            "prediction":    prediction,
            "confidence":    confidence,
            "riskLevel":     risk,
            "probApproved":  round(prob_approved * 100, 2),
            "probRejected":  round(float(proba[0]) * 100, 2),
            "reasoning":     reasons,
            "modelVersion":  metadata["model_type"]
        }
        logger.info(f"Prediction result: {prediction} ({confidence}% confidence)")
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "modelType":    metadata["model_type"],
        "accuracy":     metadata["accuracy"],
        "precision":    metadata["precision"],
        "recall":       metadata["recall"],
        "f1Score":      metadata["f1"],
        "features":     FEATURE_NAMES,
        "totalFeatures": len(FEATURE_NAMES)
    })

# ──────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)