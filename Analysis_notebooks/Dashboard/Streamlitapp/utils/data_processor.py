"""
Data processing utilities for Ethiopian Student Performance Dashboard
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Color scheme
COLOR_SCHEME = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#18A999',
    'warning': '#F18F01',
    'danger': '#C73E1D',
    'light': '#F0F3F5',
    'dark': '#2C3E50',
    'text': '#2C3E50',
    'background': '#FFFFFF',
    'low_perf': '#C73E1D',
    'medium_perf': '#F18F01',
    'high_perf': '#18A999'
}


class DataProcessor:
    """Data preprocessing class"""
    
    def __init__(self):
        self.target_encoders = {}
        self.reg_features = None
        self.class_features = None
    
    def load_and_preprocess_data(self, df_original):
        """Load and preprocess the Ethiopian student dataset"""
        df = df_original.copy()
        
        # Add numeric Student_ID if not present or if it's string
        if 'Student_ID' not in df.columns:
            df['Student_ID'] = range(1, len(df) + 1)
        else:
            try:
                # Try to convert to numeric
                df['Student_ID'] = pd.to_numeric(df['Student_ID'], errors='coerce')
                # Fill NaN with sequential numbers
                df['Student_ID'] = df['Student_ID'].fillna(range(1, len(df) + 1)).astype(int)
            except:
                # If conversion fails, create new IDs
                df['Student_ID'] = range(1, len(df) + 1)
        
        # Drop School_ID
        df = df.drop(columns=['School_ID'], errors='ignore')
        
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
            for stage_name, grades in [('Lower_Primary', lower_primary), ('Upper_Primary', upper_primary),
                                        ('Secondary', secondary), ('Preparatory', preparatory)]:
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
        
        # Add Age if not present
        if 'Age' not in df.columns and 'Date_of_Birth' in df.columns:
            CURRENT_DATE = pd.Timestamp('2026-01-30')
            df['Date_of_Birth'] = pd.to_datetime(df['Date_of_Birth'], errors='coerce')
            df['Age'] = ((CURRENT_DATE - df['Date_of_Birth']).dt.days // 365).astype(float)
            df['Age'] = df['Age'].fillna(17)
        elif 'Age' not in df.columns:
            df['Age'] = 17
        
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
        
        return df_encoded
    
    def prepare_features_for_prediction(self, input_data):
        """Prepare input data for prediction"""
        df = pd.DataFrame([input_data])
        df = self.load_and_preprocess_data(df)
        df = self.encode_categorical_features(df)
        return df


# Global functions for backward compatibility
def load_and_preprocess_data(df):
    processor = DataProcessor()
    return processor.load_and_preprocess_data(df)


def encode_categorical_features(df):
    processor = DataProcessor()
    return processor.encode_categorical_features(df)


def prepare_target_encoders(df_raw, df_clean):
    """Prepare target encoders - kept for compatibility"""
    pass


def process_raw_input_for_prediction(raw_input, df_clean, target_encoders, reg_features):
    """Process raw input for prediction"""
    processor = DataProcessor()
    df = pd.DataFrame([raw_input])
    df = processor.load_and_preprocess_data(df)
    df = processor.encode_categorical_features(df)
    
    # Ensure all expected columns are present
    if reg_features:
        for col in reg_features:
            if col not in df.columns:
                df[col] = 0
    
    return df[reg_features] if reg_features else df