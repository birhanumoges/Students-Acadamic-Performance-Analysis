# utils/predictions.py
import pandas as pd
import numpy as np
import joblib
import os

class PredictionEngine:
    """Prediction engine using saved models"""
    
    def __init__(self, model_paths):
        self.reg_model = None
        self.clf_model = None
        self.load_models(model_paths)
        
    def load_models(self, model_paths):
        """Load saved models"""
        try:
            if os.path.exists(model_paths.get('regression', '')):
                self.reg_model = joblib.load(model_paths['regression'])
            if os.path.exists(model_paths.get('classification', '')):
                self.clf_model = joblib.load(model_paths['classification'])
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def predict_score(self, features_df):
        """Predict overall average score"""
        if self.reg_model is None:
            return 50.0
        try:
            return self.reg_model.predict(features_df)[0]
        except:
            return 50.0
    
    def predict_risk(self, features_df):
        """Predict risk probability"""
        if self.clf_model is None:
            return 0.5
        try:
            return self.clf_model.predict_proba(features_df)[0][1]
        except:
            return 0.5
    
    def get_recommendations(self, input_data, predicted_score, risk_prob):
        """Generate recommendations based on input data and predictions"""
        recommendations = []
        
        if risk_prob > 0.5:
            recommendations.append("🔴 **Immediate Intervention Required**")
            recommendations.append("• Schedule academic counseling session")
            recommendations.append("• Implement personalized learning plan")
            recommendations.append("• Increase parent-teacher communication")
        else:
            recommendations.append("✅ **Student is Performing Well**")
            recommendations.append(f"• Predicted Score: {predicted_score:.1f}")
            recommendations.append(f"• Risk Probability: {risk_prob*100:.1f}%")
            recommendations.append("• Maintain current study habits")
        
        # Specific recommendations based on input
        if input_data.get('School_Resources_Score', 0.5) < 0.4:
            recommendations.append("• Request additional learning materials")
        if input_data.get('Overall_Textbook_Access_Composite', 0.5) < 0.4:
            recommendations.append("• Provide access to digital textbooks")
        if input_data.get('Parental_Involvement', 0.5) < 0.3:
            recommendations.append("• Organize parent engagement workshop")
        if input_data.get('Teacher_Student_Ratio', 40) > 45:
            recommendations.append("• Advocate for reduced class size")
        if input_data.get('Overall_Avg_Attendance', 75) < 80:
            recommendations.append("• Implement attendance improvement program")
        if input_data.get('Overall_Avg_Homework', 65) < 60:
            recommendations.append("• Provide homework support and tutoring")
        
        return recommendations
    
    def batch_predict(self, df, features):
        """Batch prediction for multiple students"""
        if self.reg_model is None or self.clf_model is None:
            return None, None
        
        try:
            X = df[features]
            predicted_scores = self.reg_model.predict(X)
            risk_probs = self.clf_model.predict_proba(X)[:, 1]
            return predicted_scores, risk_probs
        except Exception as e:
            print(f"Batch prediction error: {e}")
            return None, None