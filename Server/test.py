
"""
SMART PEN QUALITY CLASSIFIER - SINGLE FILE TESTING
Loads trained model and tests individual CSV files
"""
import numpy as np
import pandas as pd
import pickle
import json
import sys
import os
from sklearn.preprocessing import LabelEncoder

print("=" * 60)
print("🖋️ SMART PEN QUALITY CLASSIFIER - FILE TESTER")
print("=" * 60)

# Configuration - MUST match training configuration
IMU_FEATURES = ["ax", "ay", "az", "gx", "gy", "gz"]

def extract_features_from_csv(file_path):
    """
    Extract features from a single CSV file
    MUST match exactly how features were extracted during training
    """
    try:
        # Load CSV file
        df = pd.read_csv(file_path)
        
        if len(df) < 10:
            print(f"⚠️ Warning: File has only {len(df)} data points (minimum 10 required)")
            # Continue anyway, but results may be less accurate
        
        # Initialize feature vector
        sample_features = []
        
        # Extract features for each sensor (EXACTLY as done during training)
        for sensor in IMU_FEATURES:
            if sensor in df.columns:
                data = df[sensor].values
                # Clean any invalid values (EXACTLY as done during training)
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Extract the SAME 6 statistical features
                sample_features.extend([
                    np.mean(data),               # Mean
                    np.std(data),                # Standard deviation
                    np.min(data),                # Minimum
                    np.max(data),                # Maximum
                    np.percentile(data, 25),     # 25th percentile
                    np.percentile(data, 75)      # 75th percentile
                ])
            else:
                # Pad with zeros if sensor missing (EXACTLY as done during training)
                sample_features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Check if we have exactly 36 features
        if len(sample_features) != 36:
            print(f"❌ Error: Expected 36 features, got {len(sample_features)}")
            return None
        
        return sample_features
        
    except Exception as e:
        print(f"❌ Error processing file: {str(e)}")
        return None

def load_model_and_encoder():
    """
    Load the trained model and label encoder
    """
    try:
        # Load model
        with open("smart_pen_model.pkl", "rb") as f:
            model = pickle.load(f)
        
        # Load label encoder
        with open("smart_pen_label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
        
        # Load metadata for reference
        with open("model_metadata.json", "r") as f:
            metadata = json.load(f)
        
        print("✅ Model and encoder loaded successfully")
        print(f"📊 Model Accuracy: {metadata['model_info']['accuracy']:.2%}")
        print(f"🎯 Classes: {label_encoder.classes_.tolist()}")
        
        return model, label_encoder, metadata
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("   Please run the training script first to create the model files")
        return None, None, None
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return None, None, None

def test_single_file(file_path, model, label_encoder):
    """
    Test a single CSV file and return prediction
    """
    print(f"\n📂 Testing file: {os.path.basename(file_path)}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    
    # Extract features
    print("📊 Extracting features...")
    features = extract_features_from_csv(file_path)
    
    if features is None:
        return None
    
    # Reshape for sklearn (single sample)
    features_array = np.array(features).reshape(1, -1)
    
    # Make prediction
    print("🤖 Making prediction...")
    prediction_encoded = model.predict(features_array)[0]
    prediction_prob = model.predict_proba(features_array)[0]
    
    # Decode prediction
    quality_class = label_encoder.inverse_transform([prediction_encoded])[0]
    
    # Get confidence scores
    confidence_scores = {}
    for i, cls in enumerate(label_encoder.classes_):
        confidence_scores[cls] = prediction_prob[i]
    
    return quality_class, confidence_scores, features

def display_results(file_path, quality_class, confidence_scores, metadata):
    """
    Display prediction results in a user-friendly format
    """
    print("\n" + "=" * 60)
    print("📋 PREDICTION RESULTS")
    print("=" * 60)
    print(f"📁 File: {os.path.basename(file_path)}")
    print(f"🏷️ Predicted Quality: {quality_class.upper()}")
    
    # Display confidence scores
    print("\n🔍 Confidence Scores:")
    for cls in sorted(confidence_scores.keys(), 
                      key=lambda x: confidence_scores[x], 
                      reverse=True):
        conf = confidence_scores[cls]
        bar_length = int(conf * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        print(f"  {cls.upper():6s}: {bar} {conf:.2%}")
    
    # Display model info
    print(f"\n📊 Model Info:")
    print(f"  Accuracy: {metadata['model_info']['accuracy']:.2%}")
    print(f"  Classes: {', '.join(metadata['model_info']['classes'])}")
    
    # Add interpretation
    print("\n💡 Interpretation:")
    if quality_class == 'good':
        print("  ✅ This pen appears to be in GOOD condition")
        print("  ✓ Suitable for precise writing tasks")
    elif quality_class == 'medium':
        print("  ⚠️ This pen appears to be in MEDIUM condition")
        print("  ✓ Acceptable for general use")
        print("  ✗ May need monitoring or maintenance")
    elif quality_class == 'bad':
        print("  ❌ This pen appears to be in BAD condition")
        print("  ✗ May need repair or replacement")
        print("  ✗ Not suitable for precise work")
    
    # Check confidence threshold
    max_conf = max(confidence_scores.values())
    if max_conf < 0.5:
        print("\n⚠️ Low Confidence Warning: Prediction confidence is below 50%")
        print("   Consider manual inspection of this pen")
    elif max_conf > 0.9:
        print(f"\n✅ High Confidence: Prediction is {max_conf:.2%} certain")
    
    print("=" * 60)

def test_multiple_files(file_paths):
    """
    Test multiple files and display summary
    """
    # Load model
    model, label_encoder, metadata = load_model_and_encoder()
    if model is None:
        return
    
    results = []
    
    for file_path in file_paths:
        print(f"\n{'='*40}")
        print(f"Testing: {os.path.basename(file_path)}")
        
        result = test_single_file(file_path, model, label_encoder)
        if result:
            quality_class, confidence_scores, _ = result
            results.append((file_path, quality_class, max(confidence_scores.values())))
            
            # Display individual results
            display_results(file_path, quality_class, confidence_scores, metadata)
    
    # Display summary if multiple files
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("📊 SUMMARY OF ALL FILES")
        print("=" * 60)
        
        summary = {
            'good': 0,
            'medium': 0,
            'bad': 0
        }
        
        print("\nFile                Quality    Confidence")
        print("-" * 40)
        for file_path, quality, confidence in results:
            filename = os.path.basename(file_path)
            if len(filename) > 20:
                filename = filename[:17] + "..."
            print(f"{filename:20s} {quality.upper():8s}   {confidence:.2%}")
            summary[quality] += 1
        
        print("\n📈 Quality Distribution:")
        total = len(results)
        for quality, count in summary.items():
            percentage = (count / total) * 100
            bar = "█" * int(percentage / 5)  # Each █ = 5%
            print(f"  {quality.upper():6s}: {bar} {count}/{total} ({percentage:.1f}%)")
        
        print("=" * 60)

def main():
    """
    Main function to run the tester
    """
    # Check command line arguments
    if len(sys.argv) > 1:
        # Files provided as command line arguments
        file_paths = sys.argv[1:]
    else:
        # Interactive mode
        print("\n📁 Please enter the path(s) to CSV file(s) to test")
        print("   (Separate multiple files with spaces, or enter 'exit' to quit)")
        
        user_input = input("\nEnter file path(s): ").strip()
        
        if user_input.lower() in ['exit', 'quit']:
            return
        
        file_paths = user_input.split()
    
    # Validate files
    valid_files = []
    for file_path in file_paths:
        if os.path.exists(file_path):
            if file_path.lower().endswith('.csv'):
                valid_files.append(file_path)
            else:
                print(f"⚠️ Warning: {file_path} is not a CSV file")
        else:
            print(f"❌ File not found: {file_path}")
    
    if not valid_files:
        print("❌ No valid CSV files to test")
        return
    
    # Test the files
    test_multiple_files(valid_files)

if __name__ == "__main__":
    # First check if model files exist
    required_files = ["smart_pen_model.pkl", "smart_pen_label_encoder.pkl", "model_metadata.json"]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("❌ Missing required model files:")
        for f in missing_files:
            print(f"   - {f}")
        print("\n💡 Please run the training script first to create these files")
        print("   Then run this testing script again")
    else:
        main()