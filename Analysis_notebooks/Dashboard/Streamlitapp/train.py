import pickle
import json
import os

# Fix: use correct full path
model_path = "Dashboard/Streamlitapp/models/"

try:
    with open(model_path + "classification_model.pkl", "rb") as f:
        clf = pickle.load(f)
    print("✅ classification_model.pkl loaded")
    
    with open(model_path + "gradient_boosting_regression.pkl", "rb") as f:
        reg = pickle.load(f)
    print("✅ gradient_boosting_regression.pkl loaded")
    
    with open(model_path + "metadata_v1.json", "r") as f:
        meta = json.load(f)
    print("✅ metadata_v1.json loaded")
    
    print("\nAll models loaded successfully!")
    
except Exception as e:
    print("❌ Error:", e)