# utils/predictions.py
import pandas as pd
import numpy as np
import joblib
import os
import json

class PredictionEngine:
    """Prediction engine using saved models"""
    
    def __init__(self, model_paths, config_path=None):
        self.reg_model = None
        self.clf_model = None
        self.model_paths = model_paths
        self.config = self._load_config(config_path)
        self.load_models()
        
    def _load_config(self, config_path):
        """Load configuration"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}
    
    def load_models(self):
        """Load saved models"""
        try:
            if os.path.exists(self.model_paths.get('regression', '')):
                self.reg_model = joblib.load(self.model_paths['regression'])
                print("✅ Regression model loaded")
            else:
                print(f"⚠️ Regression model not found at {self.model_paths.get('regression', '')}")
                
            if os.path.exists(self.model_paths.get('classification', '')):
                self.clf_model = joblib.load(self.model_paths['classification'])
                print("✅ Classification model loaded")
            else:
                print(f"⚠️ Classification model not found at {self.model_paths.get('classification', '')}")
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def predict_score(self, features_df):
        """Predict overall average score"""
        if self.reg_model is None:
            # Fallback to weighted formula based on feature importance
            if 'School_Resources_Score' in features_df.columns:
                score = (60.4 * features_df['School_Resources_Score'].values[0] +
                         17.9 * features_df['Overall_Engagement_Score'].values[0] / 100 +
                         7.25 * features_df['School_Academic_Score'].values[0] +
                         7.14 * features_df['Overall_Textbook_Access_Composite'].values[0] +
                         2.87 * features_df['Overall_Avg_Attendance'].values[0] / 100 +
                         2.02 * features_df['Teacher_Student_Ratio'].values[0] / 100 +
                         1.72 * features_df['Overall_Avg_Homework'].values[0] / 100 +
                         0.85 * features_df['Overall_Avg_Participation'].values[0] / 100) * 0.8 + 20
                return max(0, min(100, score))
            return 50.0
        try:
            return self.reg_model.predict(features_df)[0]
        except:
            return 50.0
    
    def predict_risk(self, features_df):
        """Predict risk probability"""
        if self.clf_model is None:
            # Fallback to logistic function based on score
            score = self.predict_score(features_df)
            risk = 1 / (1 + np.exp(-0.15 * (50 - score)))
            return risk
        try:
            return self.clf_model.predict_proba(features_df)[0][1]
        except:
            return 0.5
    
    def get_recommendations(self, input_data, predicted_score, risk_prob):
        """Generate recommendations based on input data and predictions"""
        recommendations = []
        
        # Main recommendation based on risk
        if risk_prob > 0.5:
            recommendations.append("🔴 **IMMEDIATE INTERVENTION REQUIRED**")
            recommendations.append("• Schedule academic counseling session within 1 week")
            recommendations.append("• Implement personalized learning plan")
            recommendations.append("• Increase parent-teacher communication (weekly updates)")
            recommendations.append("• Assign academic mentor or peer tutor")
        else:
            recommendations.append("✅ **STUDENT IS PERFORMING WELL**")
            recommendations.append(f"• Predicted Score: {predicted_score:.1f}")
            recommendations.append(f"• Risk Probability: {risk_prob*100:.1f}% (Low)")
            recommendations.append("• Maintain current study habits and routines")
            recommendations.append("• Encourage participation in extracurricular activities")
        
        # Specific recommendations based on input values
        if input_data.get('School_Resources_Score', 0.5) < 0.4:
            recommendations.append("• 📚 **Request additional learning materials** - Current school resources are below optimal level")
        if input_data.get('Overall_Textbook_Access_Composite', 0.5) < 0.4:
            recommendations.append("• 📖 **Provide access to digital textbooks** - Textbook access is limited")
        if input_data.get('Parental_Involvement', 0.5) < 0.3:
            recommendations.append("• 👪 **Organize parent engagement workshop** - Increase parental involvement")
        if input_data.get('Teacher_Student_Ratio', 40) > 45:
            recommendations.append("• 👥 **Advocate for reduced class size** - Current ratio may limit individual attention")
        if input_data.get('Overall_Avg_Attendance', 75) < 80:
            recommendations.append("• 📅 **Implement attendance improvement program** - Target 90%+ attendance")
        if input_data.get('Overall_Avg_Homework', 65) < 60:
            recommendations.append("• 📝 **Provide homework support and tutoring** - Homework completion needs improvement")
        if input_data.get('Overall_Avg_Participation', 70) < 65:
            recommendations.append("• 💬 **Encourage class participation** - Active participation improves learning")
        
        # Add improvement suggestions
        if risk_prob < 0.3:
            recommendations.append("🏆 **Maintain Excellence** - Continue current successful strategies")
        elif risk_prob < 0.5:
            recommendations.append("📈 **Preventive Measures** - Monitor progress monthly to prevent decline")
        else:
            recommendations.append("🚨 **Urgent Action** - Implement intervention plan immediately")
        
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