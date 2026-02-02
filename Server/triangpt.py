"""
SMART PEN CLASSIFIER - FINAL REGULARIZED VERSION
Stable • Generalizable • Deployment-ready
"""

import numpy as np
import pandas as pd
import glob
import pickle
import json
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

warnings.filterwarnings("ignore")

print("=" * 60)
print("🎯 SMART PEN CLASSIFIER - FINAL VERSION")
print("=" * 60)

# ---------------- CONFIG ----------------
DATA_PATH = "data"
IMU_FEATURES = ["ax", "ay", "az", "gx", "gy", "gz"]

# We keep ONLY robust statistics (anti-overfitting)
STATS = ["mean", "std", "p25", "p75"]  # 4 stats × 6 sensors = 24 features

# ---------------- LOAD DATA ----------------
print("\n📂 Loading data...")
files = glob.glob(f"{DATA_PATH}/*/*/*.csv")
print(f"Found {len(files)} CSV files")

features, labels = [], []

for file in files:
    try:
        df = pd.read_csv(file)
        if len(df) < 20:
            continue

        sample = []

        for sensor in IMU_FEATURES:
            if sensor in df.columns:
                data = np.nan_to_num(df[sensor].values, nan=0.0, posinf=0.0, neginf=0.0)
                sample.extend([
                    np.mean(data),
                    np.std(data),
                    np.percentile(data, 25),
                    np.percentile(data, 75)
                ])
            else:
                sample.extend([0.0] * 4)

        if len(sample) == 24:
            features.append(sample)
            labels.append(df["quality"].iloc[0].lower())

    except Exception:
        continue

print(f"✅ Samples used: {len(features)}")
print(f"📊 Feature dimension: {len(features[0])}")

# ---------------- ENCODE LABELS ----------------
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

print("\n🏷️ Classes:", label_encoder.classes_)
for cls, count in zip(label_encoder.classes_, np.bincount(y)):
    print(f"  {cls}: {count}")

# ---------------- SPLIT ----------------
print("\n🎯 Train / Test split (80 / 20)")
X_train, X_test, y_train, y_test = train_test_split(
    features,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------- SCALE ----------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------- MODEL ----------------
print("\n🌳 Training regularized RandomForest...")

model = RandomForestClassifier(
    n_estimators=120,
    max_depth=10,                 # 🔒 CRITICAL
    min_samples_split=10,
    min_samples_leaf=4,
    max_features="sqrt",
    max_samples=0.8,
    bootstrap=True,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ---------------- EVALUATION ----------------
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
gap = train_acc - test_acc

print("\n📈 PERFORMANCE")
print(f"  Train Accuracy: {train_acc:.2%}")
print(f"  Test Accuracy:  {test_acc:.2%}")
print(f"  Gap:            {gap:.2%}")

# ---------------- CROSS VALIDATION ----------------
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
print("\n🎯 Cross-Validation")
print(f"  Scores: {cv_scores}")
print(f"  Mean: {cv_scores.mean():.3f} ± {cv_scores.std()*2:.3f}")

# ---------------- REPORT ----------------
print("\n📝 Classification Report")
print(classification_report(y_test, y_test_pred, target_names=label_encoder.classes_))

print("\n🎯 Confusion Matrix")
print(confusion_matrix(y_test, y_test_pred))

# ---------------- SAVE ----------------
print("\n💾 Saving model...")
pickle.dump(model, open("smart_pen_model.pkl", "wb"))
pickle.dump(label_encoder, open("label_encoder.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

metadata = {
    "accuracy": {
        "train": float(train_acc),
        "test": float(test_acc),
        "gap": float(gap),
        "cv_mean": float(cv_scores.mean())
    },
    "dataset": {
        "samples": len(features),
        "features": len(features[0])
    }
}

json.dump(metadata, open("model_metadata.json", "w"), indent=2)

print("✅ Model saved")
print("✅ Metadata saved")

# ---------------- FEATURE IMPORTANCE ----------------
print("\n🔍 Top Feature Importances")

feature_names = []
for s in IMU_FEATURES:
    for st in STATS:
        feature_names.append(f"{s}_{st}")

importances = model.feature_importances_
top = np.argsort(importances)[-10:][::-1]

for i, idx in enumerate(top, 1):
    print(f"{i:2d}. {feature_names[idx]:15s} {importances[idx]:.4f}")

# ---------------- FINAL VERDICT ----------------
print("\n" + "=" * 60)
if test_acc >= 0.80 and gap <= 0.12:
    print("✅ MODEL READY FOR DEPLOYMENT")
elif test_acc >= 0.75:
    print("⚠️  USABLE — collect more data")
else:
    print("❌ NOT READY")
print("=" * 60)
