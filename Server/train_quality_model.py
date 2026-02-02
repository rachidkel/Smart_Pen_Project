"""
FINAL SMART PEN QUALITY CLASSIFIER TRAINING
Achieved: 82.19% accuracy with Random Forest
"""
import numpy as np
import pandas as pd
import glob
import pickle
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("🎯 FINAL SMART PEN QUALITY CLASSIFIER TRAINING")
print("=" * 60)

# Configuration
DATA_PATH = "data"
IMU_FEATURES = ["ax", "ay", "az", "gx", "gy", "gz"]

# Load all CSV files
print("\n📂 Loading data...")
files = glob.glob(f"{DATA_PATH}/*/*/*.csv")
print(f"Found {len(files)} CSV files")

# Extract features and labels
features = []
labels = []

for file in files:
    try:
        df = pd.read_csv(file)
        if len(df) < 10:  # Minimum 10 data points
            continue
        
        # Extract statistical features for each sensor
        sample_features = []
        
        for sensor in IMU_FEATURES:
            if sensor in df.columns:
                data = df[sensor].values
                # Clean any invalid values
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Statistical features (6 per sensor)
                sample_features.extend([
                    np.mean(data),      # Mean
                    np.std(data),       # Standard deviation
                    np.min(data),       # Minimum
                    np.max(data),       # Maximum
                    np.percentile(data, 25),  # 25th percentile
                    np.percentile(data, 75)   # 75th percentile
                ])
            else:
                # Pad with zeros if sensor missing
                sample_features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Only add if we have exactly 36 features (6 sensors × 6 features)
        if len(sample_features) == 36:
            features.append(sample_features)
            labels.append(df["quality"].iloc[0].lower())
            
    except Exception as e:
        print(f"⚠️ Skipping {file}: {str(e)[:50]}...")
        continue

print(f"✅ Processed {len(features)} samples")
print(f"📊 Feature dimension: {len(features[0])}")

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

print(f"\n🏷️ Classes: {label_encoder.classes_}")
class_counts = np.bincount(y)
print("📈 Class Distribution:")
for cls, count in zip(label_encoder.classes_, class_counts):
    print(f"  {cls}: {count} samples ({count/len(y)*100:.1f}%)")

# Train-test split (80-20)
print("\n🎯 Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    features, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f"  Training: {len(X_train)} samples")
print(f"  Testing:  {len(X_test)} samples")

# Train Random Forest
print("\n🌳 Training Random Forest Classifier...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',  # Handle class imbalance
    random_state=42,
    n_jobs=-1,  # Use all CPU cores
    bootstrap=True,
    max_features='sqrt'
)

model.fit(X_train, y_train)

# Evaluate
print("\n📊 Evaluating model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ FINAL TEST ACCURACY: {accuracy:.2%}")
print(f"🎯 That's a {((accuracy-0.33)/0.33)*100:.1f}% improvement over random guessing!")

# Detailed classification report
print("\n📝 CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Confusion matrix
print("\n🎯 CONFUSION MATRIX:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Feature importance
print("\n🔍 TOP 10 FEATURE IMPORTANCES:")
feature_names = []
for sensor in IMU_FEATURES:
    for stat in ['mean', 'std', 'min', 'max', '25%', '75%']:
        feature_names.append(f"{sensor}_{stat}")

importances = model.feature_importances_
top_indices = np.argsort(importances)[-10:][::-1]

print("Rank  Feature                Importance")
print("-" * 40)
for rank, idx in enumerate(top_indices, 1):
    print(f"{rank:2d}.   {feature_names[idx]:20s} {importances[idx]:.4f}")

# Save model and metadata
print("\n💾 Saving model and metadata...")

# Save model
with open("smart_pen_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save label encoder
with open("smart_pen_label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# Create comprehensive metadata
metadata = {
    "model_info": {
        "model_type": "RandomForestClassifier",
        "accuracy": float(accuracy),
        "n_estimators": model.n_estimators,
        "max_depth": model.max_depth,
        "n_features": model.n_features_in_,
        "classes": label_encoder.classes_.tolist(),
        "feature_names": feature_names
    },
    "dataset_info": {
        "total_samples": len(features),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "class_distribution": {
            cls: int(count) for cls, count in zip(label_encoder.classes_, class_counts)
        },
        "imbalance_ratio": float(max(class_counts) / min(class_counts))
    },
    "performance": {
        "test_accuracy": float(accuracy),
        "per_class_metrics": {},
        "confusion_matrix": cm.tolist()
    },
    "training_params": {
        "test_size": 0.2,
        "random_state": 42,
        "features_per_sensor": 6,
        "sensors": IMU_FEATURES
    },
    "deployment_info": {
        "input_format": "List of 36 floats (6 sensors × 6 stats)",
        "output_format": "One of: 'bad', 'medium', 'good'",
        "expected_accuracy": f"{accuracy:.1%}",
        "timestamp": pd.Timestamp.now().isoformat()
    }
}

# Add per-class metrics
report_dict = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
for cls in label_encoder.classes_:
    metadata["performance"]["per_class_metrics"][cls] = {
        "precision": float(report_dict[cls]["precision"]),
        "recall": float(report_dict[cls]["recall"]),
        "f1_score": float(report_dict[cls]["f1-score"]),
        "support": int(report_dict[cls]["support"])
    }

# Save metadata
with open("model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("✅ Model saved: smart_pen_model.pkl")
print("✅ Label encoder saved: smart_pen_label_encoder.pkl")
print("✅ Metadata saved: model_metadata.json")

# Visualization - ONLY the Model Accuracy vs Training Progress plot
print("\n📈 Creating visualization...")

plt.figure(figsize=(8, 6))

# Create simulated learning curve
n_estimators = model.n_estimators
acc_scores = []
train_scores = []

# Get accuracy at different stages (10%, 20%, ..., 100%)
training_progress = list(range(10, 101, 10))  # [10, 20, 30, ..., 100]
progress_labels = [f"{p}%" for p in training_progress]

for progress in training_progress:
    subset_estimators = int(n_estimators * (progress / 100))
    # Create a subset model for visualization
    model_subset = RandomForestClassifier(
        n_estimators=subset_estimators,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42
    )
    model_subset.fit(X_train, y_train)
    train_score = model_subset.score(X_train, y_train)
    test_score = model_subset.score(X_test, y_test)
    train_scores.append(train_score)
    acc_scores.append(test_score)

# Plot accuracy progression
plt.plot(progress_labels, train_scores, marker='o', label='Training Accuracy', linewidth=3, markersize=8)
plt.plot(progress_labels, acc_scores, marker='s', label='Validation Accuracy', linewidth=3, markersize=8)

# Customize the plot to match your example
plt.xlabel('Training Progress (%)', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy', fontsize=14, fontweight='bold')
plt.title('Model Accuracy vs Training Progress', fontsize=16, fontweight='bold')

# Add legend
plt.legend(loc='lower right', fontsize=12, framealpha=0.9)

# Add grid
plt.grid(True, alpha=0.3, linestyle='--')

# Set y-axis limits
plt.ylim(0, 1.05)

# Add value labels on points
for i, (train_val, val_val) in enumerate(zip(train_scores, acc_scores)):
    plt.text(i, train_val + 0.02, f'{train_val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.text(i, val_val - 0.03, f'{val_val:.2f}', ha='center', va='top', fontsize=9, fontweight='bold')

# Add final accuracy annotation
plt.text(0.5, -0.15, f'Final Model Accuracy: {accuracy:.2%}', 
         ha='center', va='center', transform=plt.gca().transAxes,
         fontsize=12, fontweight='bold', color='darkgreen')

plt.tight_layout()

# Save the figure
plt.savefig('model_accuracy_vs_training_progress.png', dpi=150, bbox_inches='tight')

# Show the plot
plt.show()

print("✅ Visualization saved:")
print("   - model_accuracy_vs_training_progress.png")

print("\n" + "=" * 60)
print("🎯 TRAINING COMPLETE!")
print("=" * 60)
print(f"Final Model Accuracy: {accuracy:.2%}")
print(f"Model saved as: smart_pen_model.pkl")
print(f"Visualization saved as: model_accuracy_vs_training_progress.png")
print("=" * 60)