# utils/data_processor.py
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """Data preprocessing and encoding class"""
    
    def __init__(self, config_path=None):
        self.target_encoders = {}
        self.feature_names = None
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path):
        """Load configuration from JSON file"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}
    
    def load_and_preprocess_data(self, df_original):
        """Load and preprocess the Ethiopian student dataset"""
        df = df_original.copy()
        
        # Drop Student_ID
        df = df.drop(columns=['Student_ID'], errors='ignore')
        
        # Encode Field_Choice
        if 'Field_Choice' in df.columns:
            df['Field_Choice'] = df['Field_Choice'].map({'Social': 0, 'Natural': 1})
        
        # Fill missing Career_Interest
        if 'Career_Interest' in df.columns:
            df['Career_Interest'] = df['Career_Interest'].fillna('Unknown')
        
        # Define education stages
        lower_primary = ['Grade_1', 'Grade_2', 'Grade_3', 'Grade_4']
        upper_primary = ['Grade_5', 'Grade_6', 'Grade_7', 'Grade_8']
        secondary = ['Grade_9', 'Grade_10']
        preparatory = ['Grade_11', 'Grade_12']
        
        stages = {
            'Lower_Primary': lower_primary,
            'Upper_Primary': upper_primary,
            'Secondary': secondary,
            'Preparatory': preparatory
        }
        
        def stage_average(df, grades, metric_keywords):
            cols = []
            for g in grades:
                for keyword in metric_keywords:
                    cols += [c for c in df.columns if c.startswith(g) and keyword.lower() in c.lower()]
            cols = list(set(cols))
            return df[cols].mean(axis=1) if len(cols) > 0 else pd.Series(0, index=df.index), cols
        
        # Aggregate metrics
        metrics_dict = {
            'Test_Score': ['Test_Score'],
            'Attendance': ['Attendance'],
            'HW_Completion': ['Homework_Completion'],
            'Participation': ['Participation']
        }
        
        cols_to_drop = []
        
        for metric_name, keywords in metrics_dict.items():
            for stage_name, grades in stages.items():
                col_name = f'Avg_{metric_name}_{stage_name}'
                df[col_name], original_cols = stage_average(df, grades, keywords)
                cols_to_drop += original_cols
        
        df.drop(columns=list(set(cols_to_drop)), inplace=True, errors='ignore')
        
        # Textbook access
        textbook_cols = [c for c in df.columns if 'Textbook' in c]
        for col in textbook_cols:
            if col in df.columns:
                df[col] = df[col].replace({'Yes': 1, 'No': 0})
        
        def textbook_access(df, grade_prefixes):
            cols = []
            for g in grade_prefixes:
                cols.extend([c for c in df.columns if c.startswith(g) and 'Textbook' in c])
            return df[cols].mean(axis=1) if len(cols) > 0 else pd.Series(0, index=df.index)
        
        new_cols_df = pd.DataFrame({
            'Textbook_Access_1_4': textbook_access(df, lower_primary),
            'Textbook_Access_5_8': textbook_access(df, upper_primary),
            'Textbook_Access_9_10': textbook_access(df, secondary),
            'Textbook_Access_11_12': textbook_access(df, preparatory)
        })
        
        df = pd.concat([df, new_cols_df], axis=1)
        df = df.loc[:, ~df.columns.duplicated()]
        
        # National exams - track-based averages
        social_subjects = ['National_Exam_History', 'National_Exam_Geography',
                           'National_Exam_Economics', 'National_Exam_Math_Social']
        natural_subjects = ['National_Exam_Biology', 'National_Exam_Chemistry',
                            'National_Exam_Physics', 'National_Exam_Math_Natural']
        
        # Convert to numeric
        for col in social_subjects + natural_subjects:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'Field_Choice' in df.columns:
            df['Social_Track_Subject_Avg'] = df[social_subjects].mean(axis=1) if all(s in df.columns for s in social_subjects) else 0
            df['Natural_Track_Subject_Avg'] = df[natural_subjects].mean(axis=1) if all(n in df.columns for n in natural_subjects) else 0
            df['Track_Subject_Average'] = np.where(df['Field_Choice'] == 0, df['Social_Track_Subject_Avg'], df['Natural_Track_Subject_Avg'])
        
        common_subjects = ['National_Exam_Aptitude', 'National_Exam_English', 'National_Exam_Civics_and_Ethical_Education']
        df['Common_Exam_Average'] = df[common_subjects].mean(axis=1) if all(c in df.columns for c in common_subjects) else 0
        
        # Drop columns
        drop_cols = [c for c in df.columns if c.startswith('Grade_')]
        drop_cols += [c for c in df.columns if c.startswith('National_Exam_')]
        df = df.drop(columns=drop_cols, errors='ignore')
        
        # Fill nulls
        df['Health_Issue'] = df['Health_Issue'].fillna('No Issue') if 'Health_Issue' in df.columns else 'No Issue'
        df['Father_Education'] = df['Father_Education'].fillna('Unknown') if 'Father_Education' in df.columns else 'Unknown'
        df['Mother_Education'] = df['Mother_Education'].fillna('Unknown') if 'Mother_Education' in df.columns else 'Unknown'
        
        # Composite features
        textbook_cols = ['Textbook_Access_1_4', 'Textbook_Access_5_8', 'Textbook_Access_9_10', 'Textbook_Access_11_12']
        df['Overall_Textbook_Access_Composite'] = df[[c for c in textbook_cols if c in df.columns]].mean(axis=1) if any(c in df.columns for c in textbook_cols) else 0.5
        
        attendance_cols = ['Avg_Attendance_Lower_Primary', 'Avg_Attendance_Upper_Primary',
                           'Avg_Attendance_Secondary', 'Avg_Attendance_Preparatory']
        df['Overall_Avg_Attendance'] = df[[c for c in attendance_cols if c in df.columns]].mean(axis=1) if any(c in df.columns for c in attendance_cols) else 75
        
        homework_cols = ['Avg_HW_Completion_Lower_Primary', 'Avg_HW_Completion_Upper_Primary',
                         'Avg_HW_Completion_Secondary', 'Avg_HW_Completion_Preparatory']
        df['Overall_Avg_Homework'] = df[[c for c in homework_cols if c in df.columns]].mean(axis=1) if any(c in df.columns for c in homework_cols) else 65
        
        participation_cols = ['Avg_Participation_Lower_Primary', 'Avg_Participation_Upper_Primary',
                              'Avg_Participation_Secondary', 'Avg_Participation_Preparatory']
        df['Overall_Avg_Participation'] = df[[c for c in participation_cols if c in df.columns]].mean(axis=1) if any(c in df.columns for c in participation_cols) else 70
        
        df['Overall_Engagement_Score'] = (
            df['Overall_Avg_Attendance'] * 0.4 +
            df['Overall_Avg_Homework'] * 0.3 +
            df['Overall_Avg_Participation'] * 0.3
        )
        
        return df
    
    def encode_categorical_features(self, df):
        """Apply categorical encoding to the dataset"""
        df_encoded = df.copy()
        
        # Binary encoding
        binary_maps = {
            'Gender': {'Male': 0, 'Female': 1},
            'Home_Internet_Access': {'No': 0, 'Yes': 1},
            'Electricity_Access': {'No': 0, 'Yes': 1},
            'School_Location': {'Rural': 0, 'Urban': 1}
        }
        
        for col, mapping in binary_maps.items():
            if col in df_encoded.columns:
                df_encoded[col] = df_encoded[col].map(mapping)
        
        # Ordinal encoding for parent education
        edu_map = {'Unknown': 0, 'Primary': 1, 'High School': 2, 'College': 3, 'University': 4}
        
        for col in ['Father_Education', 'Mother_Education']:
            if col in df_encoded.columns:
                df_encoded[col + '_Encoded'] = df_encoded[col].map(edu_map)
                df_encoded.drop(columns=[col], inplace=True)
        
        # Health Issue encoding
        if 'Health_Issue' in df_encoded.columns:
            df_encoded['Health_Issue_Flag'] = np.where(df_encoded['Health_Issue'] == 'No Issue', 0, 1)
            
            severity_map = {
                'No Issue': 0, 'Dental Problems': 1, 'Vision Issues': 1, 'Hearing Issues': 1,
                'Anemia': 2, 'Parasitic Infections': 2, 'Respiratory Issues': 2,
                'Malnutrition': 2, 'Physical Disability': 3, 'Chronic Illness': 3
            }
            
            df_encoded['Health_Issue_Severity'] = df_encoded['Health_Issue'].map(severity_map).fillna(1).astype(int)
            df_encoded['Health_Issue_Target'] = np.where(df_encoded['Health_Issue'] == 'No Issue', 0, 
                                                          np.where(df_encoded['Health_Issue_Severity'] <= 2, 0.5, 1))
            df_encoded.drop(columns=['Health_Issue'], inplace=True)
        
        # Region encoding
        if 'Region' in df_encoded.columns:
            region_freq = df_encoded['Region'].value_counts(normalize=True).to_dict()
            df_encoded['Region_Encoded'] = df_encoded['Region'].map(region_freq)
            df_encoded.drop(columns=['Region'], inplace=True)
        
        # School Type encoding
        if 'School_Type' in df_encoded.columns:
            school_freq = df_encoded['School_Type'].value_counts(normalize=True).to_dict()
            df_encoded['School_Type_Freq'] = df_encoded['School_Type'].map(school_freq)
            df_encoded['School_Type_Target'] = np.where(df_encoded['School_Type'] == 'Private', 1, 0)
            df_encoded.drop(columns=['School_Type'], inplace=True)
        
        # Career Interest encoding
        if 'Career_Interest' in df_encoded.columns:
            career_freq = df_encoded['Career_Interest'].value_counts(normalize=True).to_dict()
            df_encoded['Career_Interest_Encoded'] = df_encoded['Career_Interest'].map(career_freq)
            df_encoded.drop(columns=['Career_Interest'], inplace=True)
        
        # Date of birth to age
        if 'Date_of_Birth' in df_encoded.columns:
            CURRENT_DATE = pd.Timestamp('2026-01-30')
            df_encoded['Date_of_Birth'] = pd.to_datetime(df_encoded['Date_of_Birth'], errors='coerce')
            df_encoded['Age'] = ((CURRENT_DATE - df_encoded['Date_of_Birth']).dt.days // 365).astype(float)
            df_encoded.drop(columns=['Date_of_Birth'], inplace=True)
        
        return df_encoded
    
    def prepare_features_for_prediction(self, input_data):
        """Prepare input data for prediction"""
        # Create dataframe from input
        df = pd.DataFrame([input_data])
        
        # Process through the same pipeline
        df = self.load_and_preprocess_data(df)
        df = self.encode_categorical_features(df)
        
        return df