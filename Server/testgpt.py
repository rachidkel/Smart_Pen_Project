
import sys
import joblib
import numpy as np
import pandas as pd

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "model.pkl"
META_PATH = "metadata.pkl"

# ===============================
# LOAD MODEL & METADATA
# ===============================
print("📦 Loading model...")
model = joblib.load(MODEL_PATH)
metadata = joblib.load(META_PATH)

feature_names = metadata["feature_names"]

# ===============================
# FEATURE EXTRACTION (same as training)
# ===============================
def extract_features(df):
    features = {}

    for col in ["ax", "ay", "az", "gx", "gy", "gz"]:
        values = df[col].values

        features[f"{col}_mean"] = np.mean(values)
        features[f"{col}_std"] = np.std(values)
        features[f"{col}_min"] = np.min(values)
        features[f"{col}_max"] = np.max(values)
        features[f"{col}_p25"] = np.percentile(values, 25)
        features[f"{col}_p75"] = np.percentile(values, 75)

    return pd.DataFrame([features])

# ===============================
# MAIN
# ===============================
if len(sys.argv) != 2:
    print("❌ Usage: python predict_new_file.py <file.csv>")
    sys.exit(1)

file_path = sys.argv[1]

print(f"📂 Loading file: {file_path}")
df = pd.read_csv(file_path)

# Safety check
required_cols = {"ax", "ay", "az", "gx", "gy", "gz"}
if not required_cols.issubset(df.columns):
    print("❌ CSV must contain: ax, ay, az, gx, gy, gz")
    sys.exit(1)

# Extract features
X = extract_features(df)

# Ensure correct feature order
X = X[feature_names]

# Predict
prediction = model.predict(X)[0]
proba = model.predict_proba(X)[0]

print("\n🎯 PREDICTION RESULT")
print("==============================")
print(f"🖊️ Writing quality: {prediction}")
print("\n📊 Confidence:")
for cls, p in zip(model.classes_, proba):
    print(f"  {cls}: {p:.2%}")
