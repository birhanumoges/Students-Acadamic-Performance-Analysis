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
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


class PredictionEngine:
    """Prediction engine using saved trained models"""
    
    def __init__(self, model_dir=None):
        # Set model directory path
        if model_dir is None:
            # Get the directory where this file is located
            current_dir = Path(__file__).resolve().parent
            # Go up one level to Streamlitapp, then to models
            self.model_dir = current_dir.parent / "models"
        else:
            self.model_dir = Path(model_dir)
        
        self.reg_model = None
        self.clf_model = None
        self.reg_scaler = None
        self.clf_scaler = None
        self.reg_features = None
        self.clf_features = None
        self.pca_model = None
        self.metadata = {}
        self.models_loaded = False
        self.load_error = None
        self._load_models()
    
    def _load_models(self):
        """Load saved models with proper extraction from containers"""
        try:
            # Check if model directory exists
            if not self.model_dir.exists():
                self.load_error = f"Model directory not found: {self.model_dir}"
                print(f"⚠️ {self.load_error}")
                return
            
            print("\n" + "=" * 50)
            print("📁 Checking model files...")
            
            # Check model paths
            clf_path = self.model_dir / "student_risk_classifier.pkl"
            reg_path = self.model_dir / "gradient_boosting_model_full.pkl"
            metadata_path = self.model_dir / "metadata_classification_v1.json"
            
            # Check if models exist
            if not clf_path.exists():
                self.load_error = f"Classifier not found: {clf_path}"
                print(f"❌ {self.load_error}")
                return
            if not reg_path.exists():
                self.load_error = f"Regressor not found: {reg_path}"
                print(f"❌ {self.load_error}")
                return
            print("✅ Model files found")
            
            # Load models
            print("\n📦 Loading models...")
            try:
                clf_data = joblib.load(clf_path)
                reg_data = joblib.load(reg_path)
                print("✅ Models loaded successfully")
            except Exception as e:
                self.load_error = f"Failed to load models: {e}"
                print(f"❌ {self.load_error}")
                return
            
            # Extract actual models from containers
            print("\n🔧 Extracting models from containers...")
            
            # Handle classifier (extract model from dictionary if needed)
            if isinstance(clf_data, dict):
                print(f"   Classifier is dictionary with keys: {list(clf_data.keys())}")
                # Try to find the model
                for key in ['model', 'classifier', 'pipeline']:
                    if key in clf_data and hasattr(clf_data[key], 'predict'):
                        self.clf_model = clf_data[key]
                        print(f"   ✅ Using classifier from key: '{key}'")
                        # Extract scaler if available
                        if f'{key}_scaler' in clf_data:
                            self.clf_scaler = clf_data[f'{key}_scaler']
                        break
                else:
                    # Check all values for model
                    for key, value in clf_data.items():
                        if hasattr(value, 'predict'):
                            self.clf_model = value
                            print(f"   ✅ Using classifier from key: '{key}'")
                            break
                    else:
                        print("   ❌ No classifier model found")
                        return
            else:
                self.clf_model = clf_data
                print("   ✅ Classifier is direct model")
            
            # Handle regressor (extract model from dictionary if needed)
            if isinstance(reg_data, dict):
                print(f"   Regressor is dictionary with keys: {list(reg_data.keys())}")
                for key in ['model', 'regressor', 'pipeline']:
                    if key in reg_data and hasattr(reg_data[key], 'predict'):
                        self.reg_model = reg_data[key]
                        print(f"   ✅ Using regressor from key: '{key}'")
                        # Extract scaler if available
                        if f'{key}_scaler' in reg_data:
                            self.reg_scaler = reg_data[f'{key}_scaler']
                        # Extract feature names if available
                        if f'{key}_features' in reg_data:
                            self.reg_features = reg_data[f'{key}_features']
                        # Extract PCA if available
                        if 'pca' in reg_data:
                            self.pca_model = reg_data['pca']
                        break
                else:
                    for key, value in reg_data.items():
                        if hasattr(value, 'predict'):
                            self.reg_model = value
                            print(f"   ✅ Using regressor from key: '{key}'")
                            break
                    else:
                        print("   ❌ No regressor model found")
                        return
            else:
                self.reg_model = reg_data
                print("   ✅ Regressor is direct model")
            
            # Load feature order from metadata
            print("\n📋 Loading feature order...")
            try:
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        self.metadata = json.load(f)
                    self.clf_features = self.metadata.get('feature_order', self.metadata.get('features', []))
                    print(f"✅ Loaded {len(self.clf_features)} features from metadata")
                    print(f"   First 5 features: {self.clf_features[:5] if len(self.clf_features) > 5 else self.clf_features}")
                else:
                    print(f"⚠️ Metadata file not found: {metadata_path}")
                    self.clf_features = []
            except Exception as e:
                print(f"⚠️ Could not load metadata: {e}")
                self.clf_features = []
            
            # Set models loaded flag
            self.models_loaded = True
            self.load_error = None
            
            print("\n" + "=" * 50)
            print("✅ ALL MODELS LOADED SUCCESSFULLY!")
            print("=" * 50)
            
            # Display model information
            print("\n📊 MODEL INFORMATION:")
            print(f"   Classifier type: {type(self.clf_model).__name__}")
            print(f"   Regressor type: {type(self.reg_model).__name__}")
            if self.reg_features:
                print(f"   Regression features: {len(self.reg_features)}")
            if self.clf_features:
                print(f"   Classification features: {len(self.clf_features)}")
            
        except Exception as e:
            self.load_error = f"Error loading models: {str(e)}"
            print(f"❌ {self.load_error}")
            import traceback
            traceback.print_exc()
    
    def _preprocess_engagement(self, df):
        """Apply PCA for engagement score"""
        engagement_cols = ['Overall_Avg_Attendance', 'Overall_Avg_Homework', 'Overall_Avg_Participation']
        available_cols = [c for c in engagement_cols if c in df.columns]
        
        if len(available_cols) == 3:
            engagement_array = df[available_cols].values
            
            if self.pca_model is not None:
                # Use trained PCA
                engagement_pca = self.pca_model.transform(engagement_array)
                engagement_score = engagement_pca[:, 0]
            else:
                # Use weighted average as fallback (approximates PCA)
                weights = np.array([0.4, 0.3, 0.3])
                engagement_score = np.dot(engagement_array, weights)
            
            # Scale to 0-100
            min_score = engagement_score.min()
            max_score = engagement_score.max()
            if max_score > min_score:
                engagement_score = 100 * (engagement_score - min_score) / (max_score - min_score)
            else:
                engagement_score = np.full_like(engagement_score, 50)
            
            df['Overall_Engagement_Score'] = engagement_score
        else:
            # Calculate weighted engagement
            attendance = df.get('Overall_Avg_Attendance', 75).iloc[0] if 'Overall_Avg_Attendance' in df else 75
            homework = df.get('Overall_Avg_Homework', 65).iloc[0] if 'Overall_Avg_Homework' in df else 65
            participation = df.get('Overall_Avg_Participation', 70).iloc[0] if 'Overall_Avg_Participation' in df else 70
            engagement = attendance * 0.4 + homework * 0.3 + participation * 0.3
            df['Overall_Engagement_Score'] = engagement
        
        return df
    
    def preprocess_input(self, input_data):
        """Apply the same preprocessing as training"""
        df = pd.DataFrame([input_data])
        
        # ===========================================
        # 1️⃣ Fill missing values
        # ===========================================
        if 'Health_Issue' in df.columns:
            df['Health_Issue'] = df['Health_Issue'].fillna('No Issue')
        
        for col in ['Father_Education', 'Mother_Education']:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
        
        # ===========================================
        # 2️⃣ Binary encoding (Yes/No features)
        # ===========================================
        binary_maps = {
            'Gender': {'Male': 0, 'Female': 1},
            'Home_Internet_Access': {'No': 0, 'Yes': 1},
            'Electricity_Access': {'No': 0, 'Yes': 1},
            'School_Location': {'Rural': 0, 'Urban': 1},
            'Textbook_Access': {'No': 0, 'Yes': 1}
        }
        
        for col, mapping in binary_maps.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
        
        # ===========================================
        # 3️⃣ Ordinal encoding (Parents Education)
        # ===========================================
        edu_map = {
            'Unknown': 0,
            'Primary': 1,
            'High School': 2,
            'College': 3,
            'University': 4
        }
        
        for col in ['Father_Education', 'Mother_Education']:
            if col in df.columns:
                df[col + '_Encoded'] = df[col].map(edu_map)
                df.drop(columns=[col], inplace=True)
        
        # ===========================================
        # 4️⃣ Field Choice encoding
        # ===========================================
        if 'Field_Choice' in df.columns:
            df['Field_Choice'] = df['Field_Choice'].map({'Social': 0, 'Natural': 1})
        
        # ===========================================
        # 5️⃣ HEALTH ISSUE
        # ===========================================
        if 'Health_Issue' in df.columns:
            df['Health_Issue'] = df['Health_Issue'].astype(str).str.strip().str.title()
            
            # Binary flag
            df['Health_Issue_Flag'] = np.where(df['Health_Issue'] == 'No Issue', 0, 1)
            
            # Severity encoding
            severity_map = {
                'No Issue': 0,
                'Dental Problems': 1,
                'Vision Issues': 1,
                'Hearing Issues': 1,
                'Anemia': 2,
                'Parasitic Infections': 2,
                'Respiratory Issues': 2,
                'Malnutrition': 2,
                'Physical Disability': 3,
                'Chronic Illness': 3
            }
            df['Health_Issue_Severity'] = df['Health_Issue'].map(severity_map).fillna(1).astype(int)
            
            # Target encoding (simplified for prediction)
            health_target_map = {
                'No Issue': 0,
                'Dental Problems': 0.3,
                'Vision Issues': 0.3,
                'Hearing Issues': 0.3,
                'Anemia': 0.6,
                'Parasitic Infections': 0.6,
                'Respiratory Issues': 0.6,
                'Malnutrition': 0.6,
                'Physical Disability': 0.9,
                'Chronic Illness': 0.9
            }
            df['Health_Issue_Target'] = df['Health_Issue'].map(health_target_map).fillna(0.5)
            df.drop(columns=['Health_Issue'], inplace=True)
        
        # ===========================================
        # 6️⃣ Region encoding (Frequency-based)
        # ===========================================
        if 'Region' in df.columns:
            region_freq = {
                'Addis Ababa': 0.21, 'Oromia': 0.39, 'Amhara': 0.39, 'Tigray': 0.45,
                'SNNP': 0.41, 'Somali': 0.47, 'Afar': 0.45, 'Benishangul-Gumuz': 0.46,
                'Sidama': 0.43, 'Gambela': 0.42, 'Harari': 0.29, 'Dire Dawa': 0.31,
                'South West Ethiopia': 0.39
            }
            df['Region_Encoded'] = df['Region'].map(region_freq).fillna(0.5)
            df.drop(columns=['Region'], inplace=True)
        
        # ===========================================
        # 7️⃣ School Type encoding
        # ===========================================
        if 'School_Type' in df.columns:
            school_freq = {'Public': 0.6, 'Private': 0.25, 'NGO-operated': 0.1, 'Faith-based': 0.05}
            df['School_Type_Freq'] = df['School_Type'].map(school_freq).fillna(0.5)
            df['School_Type_Target'] = np.where(df['School_Type'] == 'Private', 1, 0)
            df.drop(columns=['School_Type'], inplace=True)
        
        # ===========================================
        # 8️⃣ Career Interest encoding
        # ===========================================
        if 'Career_Interest' in df.columns:
            career_freq = {
                'Teacher': 0.2, 'Doctor': 0.2, 'Engineer': 0.2, 'Farmer': 0.15,
                'Business': 0.15, 'Government': 0.1, 'Unknown': 0.05
            }
            df['Career_Interest_Encoded'] = df['Career_Interest'].map(career_freq).fillna(0.5)
            df.drop(columns=['Career_Interest'], inplace=True)
        
        # ===========================================
        # 9️⃣ Date_of_Birth → Age
        # ===========================================
        if 'Date_of_Birth' in df.columns:
            CURRENT_DATE = pd.Timestamp('2026-01-30')
            df['Date_of_Birth'] = pd.to_datetime(df['Date_of_Birth'], errors='coerce')
            df['Age'] = ((CURRENT_DATE - df['Date_of_Birth']).dt.days // 365).astype(float)
            df['Age'] = df['Age'].fillna(17)
            df.drop(columns=['Date_of_Birth'], inplace=True)
        elif 'Age' not in df.columns:
            df['Age'] = 17
        
        # ===========================================
        # 🔟 Engagement Score with PCA
        # ===========================================
        df = self._preprocess_engagement(df)
        
        # ===========================================
        # 1️⃣1️⃣ Add any missing features with defaults
        # ===========================================
        if 'Student_to_Resources_Ratio' not in df.columns:
            df['Student_to_Resources_Ratio'] = 20
        
        if 'Overall_Textbook_Access_Composite' not in df.columns and 'Textbook_Access' in df.columns:
            df['Overall_Textbook_Access_Composite'] = df['Textbook_Access']
        elif 'Overall_Textbook_Access_Composite' not in df.columns:
            df['Overall_Textbook_Access_Composite'] = 0.5
        
        # Ensure all features are numeric
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = 0
        
        return df
    
    def predict_score(self, input_data):
        """Predict overall average score using trained model"""
        if not self.models_loaded:
            return self._get_prediction_error_response(
                "Models not loaded. Please ensure model files exist in the models directory."
            )
        
        if self.reg_model is None:
            return self._get_prediction_error_response(
                "Regression model not loaded. Please check gradient_boosting_model_full.pkl"
            )
        
        try:
            # Preprocess input
            processed_df = self.preprocess_input(input_data)
            
            # Ensure correct feature order if available
            if self.reg_features is not None and len(self.reg_features) > 0:
                # Add missing features
                for feature in self.reg_features:
                    if feature not in processed_df.columns:
                        processed_df[feature] = 0
                # Reorder columns
                processed_df = processed_df[self.reg_features]
            
            # Make prediction
            prediction = self.reg_model.predict(processed_df)[0]
            return max(0, min(100, prediction))
            
        except Exception as e:
            return self._get_prediction_error_response(f"Prediction error: {str(e)}")
    
    def predict_risk(self, input_data):
        """Predict risk probability using trained model"""
        if not self.models_loaded:
            return self._get_prediction_error_response(
                "Models not loaded. Please ensure model files exist in the models directory.",
                is_risk_response=True
            )
        
        if self.clf_model is None:
            return self._get_prediction_error_response(
                "Classification model not loaded. Please check student_risk_classifier.pkl",
                is_risk_response=True
            )
        
        try:
            # Preprocess input
            processed_df = self.preprocess_input(input_data)
            
            # Ensure correct feature order if available
            if self.clf_features is not None and len(self.clf_features) > 0:
                # Add missing features
                for feature in self.clf_features:
                    if feature not in processed_df.columns:
                        processed_df[feature] = 0
                # Reorder columns
                processed_df = processed_df[self.clf_features]
            
            # Make prediction
            prediction = self.clf_model.predict_proba(processed_df)[0][1]
            return prediction
            
        except Exception as e:
            return self._get_prediction_error_response(f"Risk prediction error: {str(e)}", is_risk_response=True)
    
    def _get_prediction_error_response(self, error_message, is_risk_response=False):
        """Return error response when models are not loaded"""
        print(f"⚠️ {error_message}")
        if is_risk_response:
            return 0.5
        else:
            return 70.0
    
    def get_recommendations(self, input_data, predicted_score, risk_prob):
        """Generate recommendations based on input data and predictions"""
        recommendations = []
        
        # Check if models were loaded
        if not self.models_loaded:
            recommendations.append("⚠️ **MODELS NOT LOADED**")
            recommendations.append(f"• Error: {self.load_error if self.load_error else 'Model files not found'}")
            recommendations.append("• Please ensure the following files exist in the 'models' folder:")
            recommendations.append("  - gradient_boosting_model_full.pkl")
            recommendations.append("  - student_risk_classifier.pkl")
            recommendations.append("  - metadata_classification_v1.json (optional)")
            recommendations.append("• Model directory path: " + str(self.model_dir))
            recommendations.append("• After adding the models, restart the application")
            return recommendations
        
        # Main recommendation based on risk
        if risk_prob > 0.5:
            recommendations.append("🔴 **IMMEDIATE INTERVENTION REQUIRED**")
            recommendations.append("• Schedule academic counseling session within 1 week")
            recommendations.append("• Implement personalized learning plan")
            recommendations.append("• Increase parent-teacher communication (weekly updates)")
        else:
            recommendations.append("✅ **STUDENT IS PERFORMING WELL**")
            recommendations.append(f"• Predicted Score: {predicted_score:.1f}")
            recommendations.append(f"• Risk Probability: {risk_prob*100:.1f}%")
            recommendations.append("• Maintain current study habits and routines")
        
        # Specific recommendations based on input values
        school_resources = input_data.get('School_Resources_Score', 0.5)
        if school_resources < 0.4:
            recommendations.append("• 📚 **Request additional learning materials** - Current school resources are below optimal level")
        
        textbook_access = input_data.get('Overall_Textbook_Access_Composite', input_data.get('Textbook_Access', 0.5))
        if isinstance(textbook_access, str):
            textbook_access = 1 if textbook_access == 'Yes' else 0
        if textbook_access < 0.4:
            recommendations.append("• 📖 **Provide access to digital textbooks** - Textbook access is limited")
        
        parental_involvement = input_data.get('Parental_Involvement', 0.5)
        if parental_involvement < 0.3:
            recommendations.append("• 👪 **Organize parent engagement workshop** - Increase parental involvement")
        
        teacher_ratio = input_data.get('Teacher_Student_Ratio', 40)
        if teacher_ratio > 45:
            recommendations.append("• 👥 **Advocate for reduced class size** - Current ratio may limit individual attention")
        
        attendance = input_data.get('Overall_Avg_Attendance', 75)
        if attendance < 80:
            recommendations.append("• 📅 **Implement attendance improvement program** - Target 90%+ attendance")
        
        homework = input_data.get('Overall_Avg_Homework', 65)
        if homework < 60:
            recommendations.append("• 📝 **Provide homework support and tutoring** - Homework completion needs improvement")
        
        participation = input_data.get('Overall_Avg_Participation', 70)
        if participation < 65:
            recommendations.append("• 💬 **Encourage class participation** - Active participation improves learning")
        
        return recommendations
    
    def is_models_loaded(self):
        """Check if models are successfully loaded"""
        return self.models_loaded
    
    def get_model_status(self):
        """Get detailed model loading status"""
        return {
            'models_loaded': self.models_loaded,
            'regression_model': self.reg_model is not None,
            'classification_model': self.clf_model is not None,
            'model_dir': str(self.model_dir),
            'error': self.load_error,
            'regression_type': type(self.reg_model).__name__ if self.reg_model else None,
            'classification_type': type(self.clf_model).__name__ if self.clf_model else None,
            'features_count': len(self.clf_features) if self.clf_features else 0
        }


# ============================================================================
# COMPATIBILITY FUNCTIONS
# ============================================================================

def load_models():
    """Compatibility function to load models"""
    engine = PredictionEngine()
    return (engine.reg_model, engine.clf_model, engine.reg_scaler, engine.clf_scaler, 
            engine.reg_features, engine.clf_features)


def make_prediction_corrected(input_data, *args, **kwargs):
    """Compatibility function for predictions"""
    engine = PredictionEngine()
    
    # Check if models are loaded
    if not engine.is_models_loaded():
        return {
            'predicted_score': None,
            'risk_probability': None,
            'is_risk': None,
            'risk_causes': [],
            'recommendations': [
                "⚠️ **MODELS NOT LOADED**",
                f"• Error: {engine.load_error if engine.load_error else 'Model files not found'}",
                "• Please ensure the following model files exist:",
                "  - gradient_boosting_model_full.pkl",
                "  - student_risk_classifier.pkl",
                f"• Model directory: {engine.model_dir}",
                "• After adding the models, restart the application"
            ],
            'regression_metrics': {'model': 'Not Loaded', 'r2': 0, 'mae': 0, 'rmse': 0},
            'classification_metrics': {'model': 'Not Loaded', 'f1': 0, 'roc_auc': 0},
            'processing_steps': {
                'step1': 'Model loading attempted',
                'step2': f'Status: Models loaded = {engine.models_loaded}',
                'step3': 'Please add model files to proceed'
            },
            'models_loaded': False,
            'error': engine.load_error
        }
    
    predicted_score = engine.predict_score(input_data)
    risk_prob = engine.predict_risk(input_data)
    
    return {
        'predicted_score': predicted_score,
        'risk_probability': risk_prob,
        'is_risk': risk_prob > 0.5,
        'risk_causes': [],
        'recommendations': engine.get_recommendations(input_data, predicted_score, risk_prob),
        'regression_metrics': {'model': 'Gradient Boosting', 'r2': 0.7855, 'mae': 2.98, 'rmse': 3.72},
        'classification_metrics': {'model': 'Gradient Boosting', 'f1': 0.7782, 'roc_auc': 0.9178},
        'processing_steps': {
            'step1': 'Raw input converted to DataFrame',
            'step2': 'Applied binary and ordinal encoding',
            'step3': 'Applied target encoding for categorical variables',
            'step4': 'Calculated engagement score with PCA',
            'step5': 'Predictions made successfully'
        },
        'models_loaded': True
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
        'Feature': ['Score_x_Participation', 'Overall_Avg_Homework', 'School_Academic_Score',
                   'Overall_Test_Score_Avg', 'Overall_Avg_Attendance', 'Overall_Avg_Participation',
                   'School_Resources_Score', 'Parental_Involvement'],
        'Importance': [0.7356, 0.0720, 0.0669, 0.0431, 0.0178, 0.0162, 0.0133, 0.0116]
    })