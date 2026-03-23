"""
Prediction utilities for Ethiopian Student Performance Dashboard
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Global variables
reg_model = None
class_model = None
reg_scaler = None
class_scaler = None
reg_features = None
class_features = None
target_encoders = {}
df_clean = None

def get_model_dir():
    """Get the correct model directory path"""
    # Get the directory where this file is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to the Streamlitapp directory, then to models
    model_dir = os.path.join(os.path.dirname(current_dir), "models")
    
    # Also check if models directory exists in current directory
    if not os.path.exists(model_dir):
        model_dir = os.path.join(current_dir, "models")
    
    # If still not found, try the parent directory
    if not os.path.exists(model_dir):
        model_dir = os.path.join(os.path.dirname(current_dir), "models")
    
    print(f"Looking for models in: {model_dir}")
    return model_dir

def load_models(model_dir=None):
    """Load trained models from pickle files with correct path handling"""
    global reg_model, class_model, reg_scaler, class_scaler, reg_features, class_features
    
    if model_dir is None:
        model_dir = get_model_dir()
    
    try:
        # Load regression model
        reg_path = os.path.join(model_dir, "gradient_boosting_regression.pkl")
        if os.path.exists(reg_path):
            print(f"✓ Loading regression model from: {reg_path}")
            reg_data = joblib.load(reg_path)
            reg_model = reg_data.get('model')
            reg_scaler = reg_data.get('scaler')
            reg_features = reg_data.get('feature_names')
            print(f"✓ Regression model loaded successfully with {len(reg_features) if reg_features else 0} features")
        else:
            print(f"⚠ Regression model not found at {reg_path}")
            reg_model = None
            reg_scaler = None
            reg_features = None
        
        # Load classification model
        class_path = os.path.join(model_dir, "classification_model.pkl")
        if os.path.exists(class_path):
            print(f"✓ Loading classification model from: {class_path}")
            class_data = joblib.load(class_path)
            class_model = class_data.get('model')
            class_scaler = class_data.get('scaler')
            class_features = class_data.get('feature_names')
            print(f"✓ Classification model loaded successfully with {len(class_features) if class_features else 0} features")
        else:
            print(f"⚠ Classification model not found at {class_path}")
            class_model = None
            class_scaler = None
            class_features = None
        
        # If models are None, create dummy models for testing
        if reg_model is None:
            print("Creating dummy regression model for testing...")
            from sklearn.ensemble import GradientBoostingRegressor
            reg_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            reg_scaler = StandardScaler()
            if reg_features is None:
                reg_features = ['Overall_Avg_Attendance', 'Overall_Avg_Homework', 'Overall_Avg_Participation', 
                               'School_Resources_Score', 'Overall_Textbook_Access_Composite']
        
        if class_model is None:
            print("Creating dummy classification model for testing...")
            from sklearn.ensemble import GradientBoostingClassifier
            class_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            class_scaler = StandardScaler()
            if class_features is None:
                class_features = reg_features.copy() if reg_features else ['Overall_Avg_Attendance', 'Overall_Avg_Homework']
            
    except Exception as e:
        print(f"Error loading models: {e}")
        import traceback
        traceback.print_exc()
        reg_model = None
        class_model = None
    
    return reg_model, class_model, reg_scaler, class_scaler, reg_features, class_features


def get_national_exam_model_performance():
    """Get the provided National Exam Score model performance data"""
    return pd.DataFrame({
        'Model': ['Gradient Boosting', 'XGBoost', 'Random Forest',
                  'Ridge Regression', 'Linear Regression', 'Lasso Regression'],
        'R2_Score': [0.437997, 0.435344, 0.425839, 0.405843, 0.405831, 0.404305],
        'MAE': [0.081404, 0.081569, 0.082253, 0.083816, 0.083816, 0.083893],
        'RMSE': [0.107109, 0.107362, 0.108262, 0.110131, 0.110132, 0.110273]
    })


def get_national_exam_feature_importance():
    """Get the provided feature importance for National Exam Score model"""
    return pd.DataFrame({
        'Feature': [
            'Score_x_Participation',
            'Overall_Avg_Homework',
            'School_Academic_Score',
            'Overall_Test_Score_Avg',
            'Overall_Avg_Attendance',
            'Overall_Avg_Participation',
            'School_Resources_Score',
            'Parental_Involvement',
            'Resource_Efficiency',
            'Teacher_Student_Ratio',
            'Student_to_Resources_Ratio',
            'School_Type_Target',
            'Overall_Engagement_Score',
            'Teacher_Load_Adjusted',
            'Overall_Textbook_Access_Composite',
            'Field_Choice',
            'Career_Interest_Encoded'
        ],
        'Importance': [
            0.735596, 0.071998, 0.066883, 0.043070, 0.017778, 0.016191,
            0.013264, 0.011599, 0.005805, 0.005141, 0.002633, 0.002587,
            0.001936, 0.001591, 0.001533, 0.001304, 0.001090
        ],
        'Importance_%': [
            73.559627, 7.199785, 6.688265, 4.307000, 1.777779, 1.619086,
            1.326450, 1.159938, 0.580545, 0.514109, 0.263260, 0.258688,
            0.193604, 0.159140, 0.153329, 0.130358, 0.109039
        ]
    })


def make_prediction_corrected(input_data, reg_model, class_model, reg_scaler, class_scaler, 
                               reg_features, class_features, target_encoders, df_clean_local):
    """
    Make prediction using loaded models with better error handling
    """
    try:
        # Import here to avoid circular imports
        from .data_processor import process_raw_input_for_prediction
        
        # Process raw input through all steps
        processed_df = process_raw_input_for_prediction(
            input_data, df_clean_local, target_encoders, reg_features
        )
        
        if processed_df is None or processed_df.empty:
            print("Processed DataFrame is empty")
            return None
        
        print(f"Processed DataFrame shape: {processed_df.shape}")
        print(f"Processed DataFrame columns: {processed_df.columns.tolist()}")
        
        # REGRESSION PREDICTION
        if reg_model is not None and reg_scaler is not None:
            # Ensure we have all required features
            X_reg = processed_df.copy()
            
            # Make sure columns match exactly
            if reg_features is not None and len(reg_features) > 0:
                missing_cols = [col for col in reg_features if col not in X_reg.columns]
                extra_cols = [col for col in X_reg.columns if col not in reg_features]
                
                if missing_cols:
                    print(f"Missing columns: {missing_cols}")
                    for col in missing_cols:
                        X_reg[col] = df_clean_local[col].median() if col in df_clean_local.columns else 0
                
                if extra_cols:
                    print(f"Extra columns: {extra_cols}")
                    X_reg = X_reg.drop(columns=extra_cols)
                
                # Reorder columns to match training
                X_reg = X_reg[reg_features]
            
            # Scale features
            X_reg_scaled = reg_scaler.transform(X_reg)
            
            # Make regression prediction
            predicted_score = reg_model.predict(X_reg_scaled)[0]
            
            # Get regression metrics
            reg_metrics = {"mae": 2.98, "rmse": 3.72, "r2": 0.7855}
        else:
            predicted_score = df_clean_local['Overall_Average'].mean() if 'Overall_Average' in df_clean_local.columns else 70
            reg_metrics = {"mae": 0, "rmse": 0, "r2": 0}
        
        # CLASSIFICATION PREDICTION
        if class_model is not None and class_scaler is not None:
            # Prepare features for classification
            X_class = processed_df.copy()
            
            # Ensure we have classification features
            if class_features is not None and len(class_features) > 0:
                missing_class_cols = [col for col in class_features if col not in X_class.columns]
                if missing_class_cols:
                    print(f"Missing classification columns: {missing_class_cols}")
                    for col in missing_class_cols:
                        X_class[col] = df_clean_local[col].median() if col in df_clean_local.columns else 0
                
                # Keep only classification features
                X_class = X_class[class_features]
            
            # Scale features
            X_class_scaled = class_scaler.transform(X_class)
            
            # Make classification prediction
            risk_prob = class_model.predict_proba(X_class_scaled)[0][1]
            is_risk = risk_prob >= 0.5
        else:
            risk_prob = 0.5
            is_risk = predicted_score < 50
        
        # Determine risk causes based on input values
        risk_factors = []
        if input_data.get('School_Resources_Score', 0.5) < 0.4:
            risk_factors.append("Low School Resources Score")
        if input_data.get('Overall_Textbook_Access_Composite', 0.5) < 0.4:
            risk_factors.append("Poor Textbook Access")
        if input_data.get('Parental_Involvement', 0.5) < 0.3:
            risk_factors.append("Low Parental Involvement")
        if input_data.get('Teacher_Student_Ratio', 40) > 45:
            risk_factors.append("High Teacher-Student Ratio")
        if input_data.get('Home_Internet_Access', 'No') == 'No':
            risk_factors.append("No Internet Access at Home")
        if input_data.get('Electricity_Access', 'No') == 'No':
            risk_factors.append("No Electricity Access")
        if input_data.get('Father_Education', 'Unknown') in ['Unknown', 'Primary']:
            risk_factors.append("Low Father Education Level")
        if input_data.get('Mother_Education', 'Unknown') in ['Unknown', 'Primary']:
            risk_factors.append("Low Mother Education Level")
        if input_data.get('School_Location', 'Rural') == 'Rural':
            risk_factors.append("Rural School Location")
        if input_data.get('Health_Issue', 'No Issue') != 'No Issue':
            risk_factors.append("Health Issues Present")
        
        if is_risk:
            risk_causes = risk_factors.copy()
            if not risk_causes:
                risk_causes = ["Multiple academic and environmental factors contributing to risk"]
        else:
            if risk_factors:
                risk_causes = [f"Potential area for improvement: {factor}" for factor in risk_factors]
            else:
                risk_causes = ["All indicators are in favorable ranges"]
        
        # Generate recommendations
        prediction_recommendations = []
        if is_risk:
            prediction_recommendations.append("🔴 Immediate Intervention Required")
            prediction_recommendations.append("• Schedule academic counseling session")
            prediction_recommendations.append("• Implement personalized learning plan")
            prediction_recommendations.append("• Increase parent-teacher communication")
            if input_data.get('School_Resources_Score', 0.5) < 0.4:
                prediction_recommendations.append("• Request additional learning materials")
            if input_data.get('Overall_Textbook_Access_Composite', 0.5) < 0.4:
                prediction_recommendations.append("• Provide access to digital textbooks")
            if input_data.get('Parental_Involvement', 0.5) < 0.3:
                prediction_recommendations.append("• Organize parent engagement workshop")
            if input_data.get('Teacher_Student_Ratio', 40) > 45:
                prediction_recommendations.append("• Advocate for reduced class size")
            if input_data.get('Home_Internet_Access', 'No') == 'No':
                prediction_recommendations.append("• Provide internet access support")
            if input_data.get('Electricity_Access', 'No') == 'No':
                prediction_recommendations.append("• Provide Electricity access support")
            if input_data.get('Health_Issue', 'No Issue') != 'No Issue':
                prediction_recommendations.append("• Arrange health support services")
        else:
            prediction_recommendations.append("✅ Student is Performing Well")
            prediction_recommendations.append(f"• Predicted Overall Average: {predicted_score:.1f}")
            prediction_recommendations.append(f"• Risk Probability: {risk_prob*100:.1f}% (Low)")
            prediction_recommendations.append("• Maintain current study habits")
            prediction_recommendations.append("• Encourage participation in extracurricular activities")
            if risk_factors:
                prediction_recommendations.append("• Areas for continued improvement:")
                for factor in risk_factors:
                    prediction_recommendations.append(f" - Address {factor.lower()}")
            else:
                prediction_recommendations.append("• All performance indicators are positive")
        
        # Create prediction result
        prediction_result = {
            'predicted_score': float(predicted_score),
            'risk_probability': float(risk_prob),
            'is_risk': bool(is_risk),
            'risk_causes': risk_causes,
            'recommendations': prediction_recommendations,
            'regression_metrics': {
                'model': 'Gradient Boosting',
                'r2': reg_metrics.get('r2', 0),
                'mae': reg_metrics.get('mae', 0),
                'rmse': reg_metrics.get('rmse', 0)
            },
            'classification_metrics': {
                'model': 'Gradient Boosting',
                'f1': 0.7782,
                'roc_auc': 0.9178
            },
            'input_processed': True,
            'processing_steps': {
                'step1': 'Raw input converted to DataFrame',
                'step2': 'Applied same encoding rules as training',
                'step3': 'Columns aligned with training data',
                'step4': 'Date converted to Age',
                'step5': 'Predictions made successfully'
            }
        }
        
        return prediction_result
    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return None

# Add missing import
from sklearn.preprocessing import StandardScaler