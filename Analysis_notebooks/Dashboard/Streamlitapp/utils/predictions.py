# utils/predictions.py
"""
Prediction utilities for Ethiopian Student Performance Dashboard
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class PredictionEngine:
    """Prediction engine using saved trained models"""
    
    def __init__(self, model_dir=None):
        self.model_dir = model_dir or Path(__file__).resolve().parent.parent / "models"
        self.reg_model = None
        self.clf_model = None
        self.feature_order = []
        self._load_models()
    
    def _load_models(self):
        """Load saved models"""
        try:
            # Load regression model
            reg_path = self.model_dir / "gradient_boosting_model_full.pkl"
            if reg_path.exists():
                reg_data = joblib.load(reg_path)
                if isinstance(reg_data, dict):
                    self.reg_model = reg_data.get('model', reg_data.get('regressor'))
                else:
                    self.reg_model = reg_data
            
            # Load classification model
            clf_path = self.model_dir / "student_risk_classifier.pkl"
            if clf_path.exists():
                clf_data = joblib.load(clf_path)
                if isinstance(clf_data, dict):
                    self.clf_model = clf_data.get('model', clf_data.get('classifier'))
                else:
                    self.clf_model = clf_data
            
            # Load metadata
            metadata_path = self.model_dir / "metadata_classification_v1.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                    self.feature_order = self.metadata.get('feature_order', self.metadata.get('features', []))
                    
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def predict_score(self, input_data):
        """Predict overall average score"""
        if self.reg_model is None:
            # Fallback calculation
            engagement = (input_data.get('Overall_Avg_Attendance', 75) * 0.4 +
                         input_data.get('Overall_Avg_Homework', 65) * 0.3 +
                         input_data.get('Overall_Avg_Participation', 70) * 0.3) / 100
            score = (60.4 * input_data.get('School_Resources_Score', 0.5) +
                    17.9 * engagement +
                    7.14 * input_data.get('Overall_Textbook_Access_Composite', 0.5) +
                    2.87 * input_data.get('Overall_Avg_Attendance', 75)/100 +
                    2.02 * input_data.get('Teacher_Student_Ratio', 40)/100 +
                    1.72 * input_data.get('Overall_Avg_Homework', 65)/100 +
                    0.85 * input_data.get('Overall_Avg_Participation', 70)/100) * 0.8 + 20
            return max(0, min(100, score))
        
        try:
            # Simple preprocessing
            df = pd.DataFrame([input_data])
            
            # Basic encoding
            if 'Gender' in df.columns:
                df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
            if 'Home_Internet_Access' in df.columns:
                df['Home_Internet_Access'] = df['Home_Internet_Access'].map({'Yes': 1, 'No': 0})
            if 'Electricity_Access' in df.columns:
                df['Electricity_Access'] = df['Electricity_Access'].map({'Yes': 1, 'No': 0})
            if 'School_Location' in df.columns:
                df['School_Location'] = df['School_Location'].map({'Urban': 1, 'Rural': 0})
            if 'Textbook_Access' in df.columns:
                df['Textbook_Access'] = df['Textbook_Access'].map({'Yes': 1, 'No': 0})
            
            # Calculate engagement
            attendance = input_data.get('Overall_Avg_Attendance', 75)
            homework = input_data.get('Overall_Avg_Homework', 65)
            participation = input_data.get('Overall_Avg_Participation', 70)
            df['Overall_Engagement_Score'] = (attendance * 0.4 + homework * 0.3 + participation * 0.3)
            
            # Ensure all features are present
            if self.feature_order:
                for feature in self.feature_order:
                    if feature not in df.columns:
                        df[feature] = 0
                df = df[self.feature_order]
            
            prediction = self.reg_model.predict(df)[0]
            return max(0, min(100, prediction))
        except Exception as e:
            print(f"Prediction error: {e}")
            return 70.0
    
    def predict_risk(self, input_data):
        """Predict risk probability"""
        if self.clf_model is None:
            score = self.predict_score(input_data)
            return 1 / (1 + np.exp(-0.15 * (50 - score)))
        
        try:
            df = pd.DataFrame([input_data])
            
            # Same preprocessing as above
            if 'Gender' in df.columns:
                df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
            if 'Home_Internet_Access' in df.columns:
                df['Home_Internet_Access'] = df['Home_Internet_Access'].map({'Yes': 1, 'No': 0})
            if 'Electricity_Access' in df.columns:
                df['Electricity_Access'] = df['Electricity_Access'].map({'Yes': 1, 'No': 0})
            if 'School_Location' in df.columns:
                df['School_Location'] = df['School_Location'].map({'Urban': 1, 'Rural': 0})
            
            attendance = input_data.get('Overall_Avg_Attendance', 75)
            homework = input_data.get('Overall_Avg_Homework', 65)
            participation = input_data.get('Overall_Avg_Participation', 70)
            df['Overall_Engagement_Score'] = (attendance * 0.4 + homework * 0.3 + participation * 0.3)
            
            if self.feature_order:
                for feature in self.feature_order:
                    if feature not in df.columns:
                        df[feature] = 0
                df = df[self.feature_order]
            
            prediction = self.clf_model.predict_proba(df)[0][1]
            return prediction
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


# ============================================================================
# COMPATIBILITY FUNCTIONS
# ============================================================================

def load_models():
    """Compatibility function to load models"""
    engine = PredictionEngine()
    return (engine.reg_model, engine.clf_model, None, None, 
            engine.feature_order, engine.feature_order)


def make_prediction_corrected(input_data, reg_model, class_model, reg_scaler, class_scaler,
                               reg_features, class_features, target_encoders, df_clean_local):
    """Compatibility function for predictions"""
    engine = PredictionEngine()
    predicted_score = engine.predict_score(input_data)
    risk_prob = engine.predict_risk(input_data)
    
    return {
        'predicted_score': predicted_score,
        'risk_probability': risk_prob,
        'is_risk': risk_prob > 0.5,
        'risk_causes': [],
        'recommendations': [],
        'regression_metrics': {'model': 'XGBoost', 'r2': 0.7855, 'mae': 2.98, 'rmse': 3.72},
        'classification_metrics': {'model': 'Gradient Boosting', 'f1': 0.7782, 'roc_auc': 0.9178},
        'processing_steps': {}
    }


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
            'School_Resources_Score', 'Parental_Involvement'
        ],
        'Importance': [0.7356, 0.0720, 0.0669, 0.0431, 0.0178, 0.0162, 0.0133, 0.0116]
    })