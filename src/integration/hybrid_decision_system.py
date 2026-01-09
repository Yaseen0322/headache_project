import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path

# ---------------- PATHS ----------------
MODEL_PATH = Path("models/fmri_classifier.pkl")
FEATURES_PATH = Path("data/processed/fmri/fmri_features.csv")
OCR_KB_PATH = Path("data/processed/ocr/headache_knowledge.json")
SHAP_PATH = Path("outputs/explainability/shap_feature_importance.csv")

# ---------------- LOAD COMPONENTS ----------------
model = joblib.load(MODEL_PATH)
features_df = pd.read_csv(FEATURES_PATH)

if "subject_id" in features_df.columns:
    subject_ids = features_df["subject_id"]
    X = features_df.drop(columns=["subject_id"])
else:
    subject_ids = [f"sub_{i}" for i in range(len(features_df))]
    X = features_df

with open(OCR_KB_PATH, "r") as f:
    ocr_kb = json.load(f)

shap_importance = pd.read_csv(SHAP_PATH)

# ---------------- FUNCTIONS ----------------
def get_recommendations(label):
    if label == 1:
        return ocr_kb["migraine"]
    else:
        return {
            "precautions": ["Maintain healthy lifestyle"],
            "treatment": ["Consult physician if symptoms persist"]
        }

def explain_prediction():
    top_features = shap_importance.head(3)["feature"].tolist()
    return top_features

# ---------------- MAIN PIPELINE ----------------
def hybrid_decision(index=0):
    sample = X.iloc[[index]]
    subject = subject_ids[index]

    proba = model.predict_proba(sample)[0]
    pred_label = np.argmax(proba)
    confidence = round(proba[pred_label], 3)

    diagnosis = "Migraine" if pred_label == 1 else "Non-migraine"

    rec = get_recommendations(pred_label)
    explanation = explain_prediction()

    result = {
        "subject_id": subject,
        "diagnosis": diagnosis,
        "confidence": confidence,
        "key_brain_features": explanation,
        "precautions": rec.get("precautions", []),
        "treatment_guidance": rec.get("treatment", [])
    }

    return result

# ---------------- TEST RUN ----------------
if __name__ == "__main__":
    output = hybrid_decision(index=0)
    print("\n=== HYBRID DECISION OUTPUT ===")
    for k, v in output.items():
        print(f"{k}: {v}")
