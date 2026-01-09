import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Paths
FEATURES_PATH = Path("data/processed/fmri/fmri_features.csv")
LABELS_PATH = Path("data/processed/fmri/fmri_labels.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Load data
X = pd.read_csv(FEATURES_PATH)
y = pd.read_csv(LABELS_PATH)["label"]
# Drop non-numeric identifier
if "subject_id" in X.columns:
    X = X.drop(columns=["subject_id"])

print("[INFO] Features shape:", X.shape)
print("[INFO] Labels count:", y.value_counts())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

# Train
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n[RESULT] Accuracy:", round(accuracy, 3))
print("\n[REPORT]\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(model, MODEL_DIR / "fmri_classifier.pkl")
print("\n[OK] Model saved at models/fmri_classifier.pkl")
