"""
Prediction utilities for Ethiopian Student Performance Dashboard
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler


def get_model_dir():
    """Get the correct model directory path"""
    # Get the directory where this file is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to the Streamlitapp directory, then to models
    model_dir = os.path.join(os.path.dirname(current_dir), "models")
    
    # Also check if models directory exists in current directory
    if not os.path.exists(model_dir):
        model_dir = os.path.join(current_dir, "models")
    
    return model_dir


def load_models(model_dir=None):
    """Load trained models from pickle files"""
    if model_dir is None:
        model_dir = get_model_dir()
    
    reg_model = None
    class_model = None
    reg_scaler = None
    class_scaler = None
    reg_features = None
    class_features = None
    
    try:
        # Load regression model
        reg_path = os.path.join(model_dir, "gradient_boosting_regression.pkl")
        if os.path.exists(reg_path):
            print(f"✓ Loading regression model from: {reg_path}")
            reg_data = joblib.load(reg_path)
            if isinstance(reg_data, dict):
                reg_model = reg_data.get('model')
                reg_scaler = reg_data.get('scaler')
                reg_features = reg_data.get('feature_names')
            else:
                reg_model = reg_data
                reg_scaler = StandardScaler()
                reg_features = None
            print(f"✓ Regression model loaded")
        else:
            print(f"⚠ Regression model not found at {reg_path}")
        
        # Load classification model
        class_path = os.path.join(model_dir, "classification_model.pkl")
        if os.path.exists(class_path):
            print(f"✓ Loading classification model from: {class_path}")
            class_data = joblib.load(class_path)
            if isinstance(class_data, dict):
                class_model = class_data.get('model')
                class_scaler = class_data.get('scaler')
                class_features = class_data.get('feature_names')
            else:
                class_model = class_data
                class_scaler = StandardScaler()
                class_features = None
            print(f"✓ Classification model loaded")
        else:
            print(f"⚠ Classification model not found at {class_path}")
        
        # Create dummy models if needed (for testing)
        if reg_model is None:
            from sklearn.ensemble import GradientBoostingRegressor
            reg_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            reg_scaler = StandardScaler()
            reg_features = ['Overall_Avg_Attendance', 'Overall_Avg_Homework', 
                           'Overall_Avg_Participation', 'School_Resources_Score', 
                           'Overall_Textbook_Access_Composite']
        
        if class_model is None:
            from sklearn.ensemble import GradientBoostingClassifier
            class_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            class_scaler = StandardScaler()
            class_features = reg_features.copy() if reg_features else ['Overall_Avg_Attendance', 'Overall_Avg_Homework']
            
    except Exception as e:
        print(f"Error loading models: {e}")
    
    return reg_model, class_model, reg_scaler, class_scaler, reg_features, class_features


class PredictionEngine:
    """Prediction engine class"""
    
    def __init__(self, model_paths=None):
        self.reg_model = None
        self.clf_model = None
        self.reg_scaler = None
        self.clf_scaler = None
        self.reg_features = None
        self.clf_features = None
        self.load_models(model_paths)
    
    def load_models(self, model_paths=None):
        """Load models"""
        if model_paths:
            reg_path = model_paths.get('regression')
            class_path = model_paths.get('classification')
        else:
            reg_path = None
            class_path = None
        
        if reg_path and os.path.exists(reg_path):
            reg_data = joblib.load(reg_path)
            if isinstance(reg_data, dict):
                self.reg_model = reg_data.get('model')
                self.reg_scaler = reg_data.get('scaler')
                self.reg_features = reg_data.get('feature_names')
            else:
                self.reg_model = reg_data
                self.reg_scaler = StandardScaler()
        
        if class_path and os.path.exists(class_path):
            class_data = joblib.load(class_path)
            if isinstance(class_data, dict):
                self.clf_model = class_data.get('model')
                self.clf_scaler = class_data.get('scaler')
                self.clf_features = class_data.get('feature_names')
            else:
                self.clf_model = class_data
                self.clf_scaler = StandardScaler()
    
    def predict_score(self, features_df):
        """Predict overall average score"""
        if self.reg_model is None:
            # Fallback formula
            score = 50.0
            if 'School_Resources_Score' in features_df.columns:
                score = (60.4 * features_df['School_Resources_Score'].values[0] +
                         17.9 * features_df.get('Overall_Engagement_Score', pd.Series([70])).values[0] / 100 +
                         7.25 * features_df.get('School_Academic_Score', pd.Series([0.5])).values[0] +
                         7.14 * features_df.get('Overall_Textbook_Access_Composite', pd.Series([0.5])).values[0] +
                         2.87 * features_df.get('Overall_Avg_Attendance', pd.Series([75])).values[0] / 100 +
                         2.02 * features_df.get('Teacher_Student_Ratio', pd.Series([40])).values[0] / 100 +
                         1.72 * features_df.get('Overall_Avg_Homework', pd.Series([65])).values[0] / 100 +
                         0.85 * features_df.get('Overall_Avg_Participation', pd.Series([70])).values[0] / 100) * 0.8 + 20
                score = max(0, min(100, score))
            return score
        
        try:
            if self.reg_scaler and self.reg_features:
                X = features_df[self.reg_features]
                X_scaled = self.reg_scaler.transform(X)
                return self.reg_model.predict(X_scaled)[0]
            else:
                return self.reg_model.predict(features_df)[0]
        except Exception as e:
            print(f"Prediction error: {e}")
            return 50.0
    
    def predict_risk(self, features_df):
        """Predict risk probability"""
        if self.clf_model is None:
            score = self.predict_score(features_df)
            return 1 / (1 + np.exp(-0.15 * (50 - score)))
        
        try:
            if self.clf_scaler and self.clf_features:
                X = features_df[self.clf_features]
                X_scaled = self.clf_scaler.transform(X)
                return self.clf_model.predict_proba(X_scaled)[0][1]
            else:
                return self.clf_model.predict_proba(features_df)[0][1]
        except Exception as e:
            print(f"Risk prediction error: {e}")
            return 0.5
    
    def get_recommendations(self, input_data, predicted_score, risk_prob):
        """Generate recommendations"""
        recommendations = []
        
        if risk_prob > 0.5:
            recommendations.append("🔴 **IMMEDIATE INTERVENTION REQUIRED**")
            recommendations.append("• Schedule academic counseling session")
            recommendations.append("• Implement personalized learning plan")
            recommendations.append("• Increase parent-teacher communication")
        else:
            recommendations.append("✅ **STUDENT IS PERFORMING WELL**")
            recommendations.append(f"• Predicted Score: {predicted_score:.1f}")
            recommendations.append(f"• Risk Probability: {risk_prob*100:.1f}%")
            recommendations.append("• Maintain current study habits")
        
        if input_data.get('School_Resources_Score', 0.5) < 0.4:
            recommendations.append("• 📚 Request additional learning materials")
        if input_data.get('Overall_Textbook_Access_Composite', 0.5) < 0.4:
            recommendations.append("• 📖 Provide access to digital textbooks")
        if input_data.get('Parental_Involvement', 0.5) < 0.3:
            recommendations.append("• 👪 Organize parent engagement workshop")
        if input_data.get('Teacher_Student_Ratio', 40) > 45:
            recommendations.append("• 👥 Advocate for reduced class size")
        if input_data.get('Overall_Avg_Attendance', 75) < 80:
            recommendations.append("• 📅 Implement attendance improvement program")
        
        return recommendations


def make_prediction_corrected(input_data, reg_model, class_model, reg_scaler, class_scaler,
                               reg_features, class_features, target_encoders, df_clean_local):
    """Make prediction using loaded models"""
    try:
        from .data_processor import DataProcessor
        
        processor = DataProcessor()
        
        # Create dataframe from input
        df = pd.DataFrame([input_data])
        
        # Preprocess
        df = processor.load_and_preprocess_data(df)
        df = processor.encode_categorical_features(df)
        
        # Ensure all expected columns
        if reg_features:
            for col in reg_features:
                if col not in df.columns:
                    df[col] = 0
        
        # Regression prediction
        if reg_model is not None:
            if reg_scaler and reg_features:
                X_reg = df[reg_features]
                X_reg_scaled = reg_scaler.transform(X_reg)
                predicted_score = reg_model.predict(X_reg_scaled)[0]
            else:
                predicted_score = reg_model.predict(df)[0]
        else:
            predicted_score = 70.0
        
        # Classification prediction
        if class_model is not None:
            if class_scaler and class_features:
                X_class = df[class_features]
                X_class_scaled = class_scaler.transform(X_class)
                risk_prob = class_model.predict_proba(X_class_scaled)[0][1]
            else:
                risk_prob = class_model.predict_proba(df)[0][1]
        else:
            risk_prob = 0.5
        
        is_risk = risk_prob >= 0.5
        
        # Risk factors
        risk_factors = []
        if input_data.get('School_Resources_Score', 0.5) < 0.4:
            risk_factors.append("Low School Resources Score")
        if input_data.get('Overall_Textbook_Access_Composite', 0.5) < 0.4:
            risk_factors.append("Poor Textbook Access")
        if input_data.get('Parental_Involvement', 0.5) < 0.3:
            risk_factors.append("Low Parental Involvement")
        if input_data.get('Teacher_Student_Ratio', 40) > 45:
            risk_factors.append("High Teacher-Student Ratio")
        
        if is_risk:
            risk_causes = risk_factors.copy() if risk_factors else ["Multiple factors contributing to risk"]
        else:
            risk_causes = [f"Potential area for improvement: {factor}" for factor in risk_factors] if risk_factors else ["All indicators are favorable"]
        
        # Recommendations
        recommendations = []
        if is_risk:
            recommendations.append("🔴 Immediate Intervention Required")
            recommendations.append("• Schedule academic counseling session")
            recommendations.append("• Implement personalized learning plan")
            if input_data.get('School_Resources_Score', 0.5) < 0.4:
                recommendations.append("• Request additional learning materials")
            if input_data.get('Overall_Textbook_Access_Composite', 0.5) < 0.4:
                recommendations.append("• Provide access to digital textbooks")
        else:
            recommendations.append("✅ Student is Performing Well")
            recommendations.append(f"• Predicted Overall Average: {predicted_score:.1f}")
            recommendations.append(f"• Risk Probability: {risk_prob*100:.1f}%")
            recommendations.append("• Maintain current study habits")
        
        return {
            'predicted_score': float(predicted_score),
            'risk_probability': float(risk_prob),
            'is_risk': bool(is_risk),
            'risk_causes': risk_causes,
            'recommendations': recommendations,
            'regression_metrics': {'model': 'XGBoost', 'r2': 0.7855, 'mae': 2.98, 'rmse': 3.72},
            'classification_metrics': {'model': 'Gradient Boosting', 'f1': 0.7782, 'roc_auc': 0.9178},
            'processing_steps': {
                'step1': 'Raw input converted to DataFrame',
                'step2': 'Applied encoding rules',
                'step3': 'Columns aligned',
                'step4': 'Predictions made'
            }
        }
    except Exception as e:
        print(f"Prediction error: {e}")
        return None


def get_national_exam_model_performance():
    """Get National Exam Score model performance data"""
    return pd.DataFrame({
        'Model': ['Gradient Boosting', 'XGBoost', 'Random Forest',
                  'Ridge Regression', 'Linear Regression', 'Lasso Regression'],
        'R2_Score': [0.437997, 0.435344, 0.425839, 0.405843, 0.405831, 0.404305],
        'MAE': [0.081404, 0.081569, 0.082253, 0.083816, 0.083816, 0.083893],
        'RMSE': [0.107109, 0.107362, 0.108262, 0.110131, 0.110132, 0.110273]
    })


def get_national_exam_feature_importance():
    """Get National Exam Score feature importance"""
    return pd.DataFrame({
        'Feature': [
            'Score_x_Participation', 'Overall_Avg_Homework', 'School_Academic_Score',
            'Overall_Test_Score_Avg', 'Overall_Avg_Attendance', 'Overall_Avg_Participation',
            'School_Resources_Score', 'Parental_Involvement', 'Resource_Efficiency',
            'Teacher_Student_Ratio', 'Student_to_Resources_Ratio', 'School_Type_Target',
            'Overall_Engagement_Score', 'Teacher_Load_Adjusted',
            'Overall_Textbook_Access_Composite', 'Field_Choice', 'Career_Interest_Encoded'
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