# check_models.py - Simple model verification script
import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path

def check_models():
    """Simple function to verify models are working"""
    
    print("=" * 50)
    print("🔍 STUDENT ACADEMIC PERFORMANCE MODEL CHECKER")
    print("=" * 50)
    
    # Model paths
    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / "models"
    
    # Check if models exist
    print("\n📁 Checking model files...")
    clf_path = model_dir / "student_risk_classifier.pkl"
    reg_path = model_dir / "gradient_boosting_model_full.pkl"
    
    if not clf_path.exists():
        print(f"❌ Classifier not found: {clf_path}")
        return
    if not reg_path.exists():
        print(f"❌ Regressor not found: {reg_path}")
        return
    print("✅ Model files found")
    
    # Load models
    print("\n📦 Loading models...")
    try:
        clf_data = joblib.load(clf_path)
        reg_data = joblib.load(reg_path)
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        return
    
    # Extract actual models
    print("\n🔧 Extracting models from containers...")
    
    # Handle classifier
    if isinstance(clf_data, dict):
        print(f"   Classifier is dictionary with keys: {list(clf_data.keys())}")
        # Try to find the model
        for key in ['model', 'classifier', 'pipeline']:
            if key in clf_data and hasattr(clf_data[key], 'predict'):
                clf_model = clf_data[key]
                print(f"   ✅ Using classifier from key: '{key}'")
                break
        else:
            # Check all values
            for key, value in clf_data.items():
                if hasattr(value, 'predict'):
                    clf_model = value
                    print(f"   ✅ Using classifier from key: '{key}'")
                    break
            else:
                print("   ❌ No classifier model found")
                return
    else:
        clf_model = clf_data
        print("   ✅ Classifier is direct model")
    
    # Handle regressor
    if isinstance(reg_data, dict):
        print(f"   Regressor is dictionary with keys: {list(reg_data.keys())}")
        for key in ['model', 'regressor', 'pipeline']:
            if key in reg_data and hasattr(reg_data[key], 'predict'):
                reg_model = reg_data[key]
                print(f"   ✅ Using regressor from key: '{key}'")
                break
        else:
            for key, value in reg_data.items():
                if hasattr(value, 'predict'):
                    reg_model = value
                    print(f"   ✅ Using regressor from key: '{key}'")
                    break
            else:
                print("   ❌ No regressor model found")
                return
    else:
        reg_model = reg_data
        print("   ✅ Regressor is direct model")
    
    # Load feature order from metadata
    print("\n📋 Loading feature order...")
    metadata_path = model_dir / "metadata_classification_v1.json"
    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
        feature_order = metadata.get('feature_order', metadata.get('features', []))
        print(f"✅ Loaded {len(feature_order)} features")
        print(f"   First 5 features: {feature_order[:5]}")
    except Exception as e:
        print(f"⚠️ Could not load metadata: {e}")
        print("   Creating dummy feature list...")
        feature_order = [f'feature_{i}' for i in range(25)]
    
    # Create test data
    print("\n🧪 Creating test data...")
    test_data = pd.DataFrame(
        np.random.rand(3, len(feature_order)),  # 3 test samples
        columns=feature_order
    )
    print(f"✅ Created test data with shape: {test_data.shape}")
    
    # Make predictions
    print("\n🎯 Making test predictions...")
    try:
        # Classifier predictions
        clf_pred = clf_model.predict(test_data)
        clf_proba = clf_model.predict_proba(test_data)
        print(f"\n✅ CLASSIFIER RESULTS:")
        print(f"   Predictions: {clf_pred}")
        print(f"   Risk probabilities: {clf_proba[:, 1]}")
        print(f"   At-risk students: {sum(clf_pred)} out of {len(clf_pred)}")
        
        # Regressor predictions
        reg_pred = reg_model.predict(test_data)
        print(f"\n✅ REGRESSOR RESULTS:")
        print(f"   Predictions: {[f'{x:.2f}' for x in reg_pred]}")
        print(f"   Average score: {reg_pred.mean():.2f}")
        print(f"   Score range: {reg_pred.min():.2f} - {reg_pred.max():.2f}")
        
        print("\n" + "=" * 50)
        print("✅ ALL TESTS PASSED! Models are working correctly.")
        print("=" * 50)
        
        # Quick model info
        print("\n📊 MODEL INFORMATION:")
        print(f"   Classifier type: {type(clf_model).__name__}")
        print(f"   Regressor type: {type(reg_model).__name__}")
        
        # Check if pipeline has steps
        if hasattr(reg_model, 'steps'):
            print(f"   Pipeline steps: {[step[0] for step in reg_model.steps]}")
        
    except Exception as e:
        print(f"\n❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_models()