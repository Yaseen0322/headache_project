import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

# Paths
KB_PATH = Path("data/processed/ocr/headache_knowledge.json")
OUT_DIR = Path("data/processed/ocr")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load knowledge base
with open(KB_PATH, "r") as f:
    kb = json.load(f)

texts = []
labels = []

for headache_type, content in kb.items():
    combined_text = " ".join(
        content["symptoms"] +
        content["precautions"] +
        content["treatment"]
    )
    texts.append(combined_text)
    labels.append(headache_type)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(texts)

# Save features
features_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
features_df.to_csv(OUT_DIR / "ocr_features.csv", index=False)

# Save labels
labels_df = pd.DataFrame({"label": labels})
labels_df.to_csv(OUT_DIR / "ocr_labels.csv", index=False)

print("[OK] OCR features and labels generated")
print("Features shape:", features_df.shape)
print("Labels:", labels)
