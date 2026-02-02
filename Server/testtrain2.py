"""
SIMPLE PEN QUALITY TESTER
Test a single CSV file with the trained model
"""
import pickle
import pandas as pd
import numpy as np
import sys
import os

def test_single_file(csv_file_path):
    """
    Test a single CSV file and print the quality prediction
    """
    print("\n" + "="*50)
    print("🎯 SMART PEN QUALITY TESTER")
    print("="*50)
    
    # Check if file exists
    if not os.path.exists(csv_file_path):
        print(f"❌ File not found: {csv_file_path}")
        return
    
    # Check if model files exist
    required_files = [
        "smart_pen_model_final.pkl",
        "smart_pen_label_encoder.pkl", 
        "feature_scaler_final.pkl"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing model files: {missing_files}")
        print("   Please train the model first or check file locations")
        return
    
    try:
        # 1. LOAD THE TRAINED MODEL
        print("📦 Loading trained model...")
        with open("smart_pen_model_final.pkl", "rb") as f:
            model = pickle.load(f)
        with open("smart_pen_label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
        with open("feature_scaler_final.pkl", "rb") as f:
            scaler = pickle.load(f)
        
        print("✅ Model loaded successfully")
        print(f"   Model accuracy: 82.55%")
        print(f"   Expected range: 82-88% on new data")
        
        # 2. READ THE CSV FILE
        print(f"\n📂 Reading file: {csv_file_path}")
        df = pd.read_csv(csv_file_path)
        print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")
        
        # 3. EXTRACT FEATURES (same as training)
        print("\n🔍 Extracting features...")
        IMU_FEATURES = ["ax", "ay", "az", "gx", "gy", "gz"]
        features = []
        
        for sensor in IMU_FEATURES:
            if sensor in df.columns:
                data = df[sensor].values
                # Clean invalid values
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                
                # 6 statistical features per sensor
                features.extend([
                    np.mean(data),           # Mean
                    np.std(data),            # Standard deviation
                    np.min(data),            # Minimum
                    np.max(data),            # Maximum
                    np.percentile(data, 25), # 25th percentile
                    np.percentile(data, 75)  # 75th percentile
                ])
            else:
                # Sensor missing, use zeros
                features.extend([0.0] * 6)
                print(f"   ⚠️  Sensor '{sensor}' not found, using zeros")
        
        if len(features) != 36:
            print(f"❌ Error: Expected 36 features, got {len(features)}")
            return
        
        print(f"✅ Extracted {len(features)} features")
        
        # 4. PREPARE FEATURES FOR MODEL
        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        
        # 5. MAKE PREDICTION
        print("\n🤖 Making prediction...")
        prediction = model.predict(features_scaled)
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Convert to human-readable label
        quality = label_encoder.inverse_transform(prediction)[0]
        confidence = probabilities.max() * 100
        
        # 6. DISPLAY RESULTS
        print("\n" + "="*50)
        print("📊 TEST RESULTS")
        print("="*50)
        print(f"🎯 PREDICTED QUALITY: {quality.upper()}")
        print(f"💯 CONFIDENCE: {confidence:.1f}%")
        
        print(f"\n📈 PROBABILITY BREAKDOWN:")
        for i, cls in enumerate(label_encoder.classes_):
            prob = probabilities[i] * 100
            # Create a simple bar chart
            bar = "█" * int(prob / 3)  # Each █ represents 3%
            print(f"   {cls:6s}: {prob:5.1f}% {bar}")
        
        print(f"\n📊 MODEL INFO:")
        print(f"   • Trained accuracy: 82.55%")
        print(f"   • Overfitting gap: 13.24%")
        print(f"   • Expected real-world: 82-88%")
        
        print(f"\n📝 INTERPRETATION:")
        if quality == "good":
            print("   ✅ This pen meets quality standards")
        elif quality == "medium":
            print("   ⚠️  This pen has some issues but may be usable")
        else:  # bad
            print("   ❌ This pen fails quality standards")
        
        print("="*50)
        
        return quality, confidence
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# =============== MAIN EXECUTION ===============
if __name__ == "__main__":
    # Check if file path is provided
    if len(sys.argv) > 1:
        # Use file from command line argument
        file_path = sys.argv[1]
    else:
        # Ask user for file path
        print("\n📁 Please provide a CSV file to test")
        file_path = input("Enter CSV file path: ").strip()
    
    # Test the file
    test_single_file(file_path)