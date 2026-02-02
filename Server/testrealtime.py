"""
SMART PEN QUALITY TESTER - REAL-TIME TESTING PAGE
Integrates data collection with model prediction
"""
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import csv
import datetime
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import json
from threading import Lock
import time
import atexit

app = Flask(__name__)
app.secret_key = 'smart-pen-secret-key-2024'

# ===== Global State =====
is_testing = False
test_data = []
current_test_file = None
test_start_time = None
prediction_lock = Lock()
latest_prediction = None

# Model components
model = None
label_encoder = None
scaler = None
feature_names = []
IMU_FEATURES = ["ax", "ay", "az", "gx", "gy", "gz"]

DATA_FOLDER = "data"
TEST_RESULTS_FOLDER = "test_results"
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(TEST_RESULTS_FOLDER, exist_ok=True)

# ===== Load Model =====
def load_model():
    """Load the trained model and preprocessing components"""
    global model, label_encoder, scaler, feature_names
    
    try:
        print("🔍 Loading trained model...")
        
        # Load model
        with open("smart_pen_model_final.pkl", "rb") as f:
            model = pickle.load(f)
        
        # Load label encoder
        with open("smart_pen_label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
        
        # Load scaler
        with open("feature_scaler_final.pkl", "rb") as f:
            scaler = pickle.load(f)
        
        # Load metadata for feature names
        with open("model_metadata_final.json", "r") as f:
            metadata = json.load(f)
        
        # Generate feature names (same as during training)
        feature_names = []
        for sensor in IMU_FEATURES:
            for stat in ['mean', 'std', 'min', 'max', '25%', '75%']:
                feature_names.append(f"{sensor}_{stat}")
        
        print("✅ Model loaded successfully")
        print(f"   Classes: {label_encoder.classes_.tolist()}")
        print(f"   Model accuracy: {metadata['model_info']['test_accuracy']:.2%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

# ===== Feature Extraction =====
def extract_features_from_data(data_points):
    """
    Extract features from collected data points
    Must match exactly how features were extracted during training
    """
    if len(data_points) < 10:
        return None
    
    # Convert to DataFrame for easier processing
    df = pd.DataFrame(data_points)
    
    # Initialize feature vector
    sample_features = []
    
    # Extract features for each sensor
    for sensor in IMU_FEATURES:
        if sensor in df.columns:
            data = df[sensor].values
            # Clean any invalid values
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Extract the SAME 6 statistical features as during training
            sample_features.extend([
                np.mean(data),               # Mean
                np.std(data),                # Standard deviation
                np.min(data),                # Minimum
                np.max(data),                # Maximum
                np.percentile(data, 25),     # 25th percentile
                np.percentile(data, 75)      # 75th percentile
            ])
        else:
            # Pad with zeros if sensor missing
            sample_features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Check if we have exactly 36 features
    if len(sample_features) != 36:
        print(f"⚠️ Expected 36 features, got {len(sample_features)}")
        return None
    
    return sample_features

def analyze_pen_quality():
    """Analyze collected data and predict pen quality"""
    global test_data, model, label_encoder, scaler, latest_prediction
    
    if not test_data or len(test_data) < 10:
        return None, None, "Not enough data. Please write for at least 1 second."
    
    with prediction_lock:
        try:
            # Extract features
            features = extract_features_from_data(test_data)
            if features is None:
                return None, None, "Failed to extract features"
            
            # Scale features
            features_array = np.array(features).reshape(1, -1)
            features_scaled = scaler.transform(features_array)
            
            # Make prediction
            prediction = model.predict(features_scaled)[0]
            probabilities = model.predict_proba(features_scaled)[0]
            
            # Get quality label
            quality = label_encoder.inverse_transform([prediction])[0]
            confidence = probabilities[prediction] * 100
            
            # Store latest prediction
            latest_prediction = {
                'quality': quality,
                'confidence': confidence,
                'probabilities': probabilities.tolist(),
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            # Get detailed probabilities
            prob_details = {}
            for i, cls in enumerate(label_encoder.classes_):
                prob_details[cls] = probabilities[i] * 100
            
            return quality, confidence, prob_details
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None, None, str(e)

def save_test_results(quality, confidence, prob_details):
    """Save test results to file"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_result_{timestamp}.json"
    filepath = os.path.join(TEST_RESULTS_FOLDER, filename)
    
    result = {
        'timestamp': timestamp,
        'quality': quality,
        'confidence': confidence,
        'probabilities': prob_details,
        'data_points': len(test_data),
        'duration_seconds': (datetime.datetime.now() - test_start_time).total_seconds() if test_start_time else 0
    }
    
    with open(filepath, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"✅ Test results saved: {filename}")
    return filepath

# ===== Routes =====

@app.route('/')
def index():
    return redirect('/test')

@app.route('/test')
def test_page():
    """Main testing page"""
    model_loaded = model is not None
    return render_template(
        "test.html",
        testing=is_testing,
        model_loaded=model_loaded,
        prediction=latest_prediction
    )

@app.route('/start_test', methods=['POST'])
def start_test():
    """Start collecting data for testing"""
    global is_testing, test_data, test_start_time, latest_prediction
    
    if not model:
        return jsonify({'error': 'Model not loaded. Please train model first.'}), 400
    
    # Reset test data
    test_data = []
    test_start_time = datetime.datetime.now()
    is_testing = True
    latest_prediction = None
    
    print(f"🟢 Test started at {test_start_time}")
    return jsonify({
        'status': 'started',
        'timestamp': test_start_time.isoformat(),
        'message': 'Start writing with the pen...'
    })

@app.route('/stop_test', methods=['POST'])
def stop_test():
    """Stop testing and analyze collected data"""
    global is_testing
    
    if not is_testing:
        return jsonify({'error': 'No test in progress'}), 400
    
    is_testing = False
    
    # Analyze the collected data
    quality, confidence, prob_details = analyze_pen_quality()
    
    if quality is None:
        return jsonify({
            'error': 'Analysis failed',
            'details': prob_details  # This contains the error message
        }), 400
    
    # Save results
    result_file = save_test_results(quality, confidence, prob_details)
    
    # Prepare response
    response = {
        'status': 'completed',
        'quality': quality,
        'confidence': confidence,
        'probabilities': prob_details,
        'data_points': len(test_data),
        'duration_seconds': (datetime.datetime.now() - test_start_time).total_seconds(),
        'result_file': os.path.basename(result_file)
    }
    
    # Add interpretation
    if quality == 'good':
        response['interpretation'] = '✅ This pen is in GOOD condition - suitable for precise writing'
    elif quality == 'medium':
        response['interpretation'] = '⚠️ This pen is in MEDIUM condition - acceptable for general use'
    else:
        response['interpretation'] = '❌ This pen is in BAD condition - may need repair or replacement'
    
    print(f"🛑 Test completed: {quality} with {confidence:.1f}% confidence")
    return jsonify(response)

@app.route('/test_data', methods=['POST'])
def receive_test_data():
    """Receive IMU data during testing"""
    global test_data, is_testing
    
    if not is_testing:
        return jsonify({'status': 'ignored', 'message': 'Test not active'}), 200
    
    try:
        data = request.get_json(force=True)
        
        # Extract IMU data
        imu_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'ax': float(data.get('ax', 0)),
            'ay': float(data.get('ay', 0)),
            'az': float(data.get('az', 0)),
            'gx': float(data.get('gx', 0)),
            'gy': float(data.get('gy', 0)),
            'gz': float(data.get('gz', 0))
        }
        
        # Add to test data
        test_data.append(imu_data)
        
        # Optional: Analyze periodically during testing
        if len(test_data) % 50 == 0:  # Analyze every 50 data points
            quality, confidence, _ = analyze_pen_quality()
            if quality:
                print(f"📊 Real-time analysis: {quality} ({confidence:.1f}%)")
        
        return jsonify({'status': 'ok', 'count': len(test_data)}), 200
        
    except Exception as e:
        print(f"❌ Error receiving test data: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/test_status')
def test_status():
    """Get current test status"""
    return jsonify({
        'testing': is_testing,
        'data_points': len(test_data),
        'latest_prediction': latest_prediction,
        'model_loaded': model is not None
    })

@app.route('/analyze_now')
def analyze_now():
    """Analyze current data without stopping test"""
    if len(test_data) < 10:
        return jsonify({'error': 'Not enough data'}), 400
    
    quality, confidence, prob_details = analyze_pen_quality()
    
    if quality is None:
        return jsonify({'error': 'Analysis failed'}), 400
    
    return jsonify({
        'quality': quality,
        'confidence': confidence,
        'probabilities': prob_details,
        'data_points': len(test_data),
        'timestamp': datetime.datetime.now().isoformat()
    })

@app.route('/clear_test')
def clear_test():
    """Clear current test data"""
    global test_data, latest_prediction
    test_data = []
    latest_prediction = None
    return jsonify({'status': 'cleared', 'message': 'Test data cleared'})

@app.route('/model_info')
def model_info():
    """Get model information"""
    if not model:
        return jsonify({'error': 'Model not loaded'}), 400
    
    try:
        with open("model_metadata_final.json", "r") as f:
            metadata = json.load(f)
        
        return jsonify({
            'model_type': 'RandomForestClassifier',
            'accuracy': metadata['model_info']['test_accuracy'],
            'classes': label_encoder.classes_.tolist(),
            'training_samples': metadata['dataset_info']['total_samples'],
            'overfitting_gap': metadata['model_info']['overfitting_gap'],
            'expected_range': metadata['performance_summary']['confidence_interval']
        })
    except:
        return jsonify({
            'model_type': 'RandomForestClassifier',
            'classes': label_encoder.classes_.tolist() if label_encoder else [],
            'status': 'Model loaded, metadata not available'
        })

# ===== Initialize =====
@atexit.register
def cleanup():
    print("🔄 Cleaning up...")
    if is_testing:
        print("⚠️ Test was in progress, saving partial data...")
        if test_data:
            quality, confidence, _ = analyze_pen_quality()
            if quality:
                save_test_results(quality, confidence, {})

if __name__ == "__main__":
    # Load model on startup
    if load_model():
        print("=" * 60)
        print("🚀 SMART PEN QUALITY TESTER STARTED")
        print("=" * 60)
        print(f"📊 Model: Random Forest Classifier")
        print(f"🎯 Classes: {label_encoder.classes_.tolist()}")
        print(f"🌐 Server running on http://0.0.0.0:5000")
        print("=" * 60)
    else:
        print("⚠️ Starting without model. Train model first for predictions.")
    
    app.run(host="0.0.0.0", port=5000, debug=False)