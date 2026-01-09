import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
FEATURES_PATH = Path("data/processed/fmri/fmri_features.csv")
MODEL_PATH = Path("models/fmri_classifier.pkl")
OUT_DIR = Path("outputs/explainability")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
X = pd.read_csv(FEATURES_PATH)

# Drop subject_id if present
if "subject_id" in X.columns:
    X = X.drop(columns=["subject_id"])

# Load model
model = joblib.load(MODEL_PATH)

print("[INFO] Loaded model and features")

# SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# ---- GLOBAL EXPLANATION ----
plt.figure()
shap.summary_plot(
    shap_values[1],  # migraine class
    X,
    show=False
)
plt.savefig(OUT_DIR / "shap_summary.png", bbox_inches="tight")
plt.close()

print("[OK] SHAP summary plot saved")

# ---- FEATURE IMPORTANCE ----
importance = pd.DataFrame({
    "feature": X.columns,
    "mean_abs_shap": abs(shap_values[1]).mean(axis=0)
}).sort_values(by="mean_abs_shap", ascending=False)

importance.to_csv(OUT_DIR / "shap_feature_importance.csv", index=False)

print("[OK] Feature importance saved")
print(importance.head())
