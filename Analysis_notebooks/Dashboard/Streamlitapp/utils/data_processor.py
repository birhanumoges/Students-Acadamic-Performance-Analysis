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

# Global variables
df_original = None
df_clean = None
target_encoders = {}


def load_and_preprocess_data(df=None):
    """Load and preprocess the Ethiopian student dataset"""
    if df is None:
        return None
    
    df = df.copy()
    
    # Drop Student_ID if exists (not used in ML)
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
    
    # Helper function to aggregate grades
    def stage_average(df, grades, metric_keywords):
        cols = []
        for g in grades:
            for keyword in metric_keywords:
                cols += [c for c in df.columns if c.startswith(g) and keyword.lower() in c.lower()]
        cols = list(set(cols))
        return df[cols].mean(axis=1) if cols else pd.Series(0, index=df.index), cols
    
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
    
    # Drop original grade-level columns
    df.drop(columns=list(set(cols_to_drop)), inplace=True, errors='ignore')
    
    # Aggregate textbook access
    textbook_cols = [c for c in df.columns if 'Textbook' in c]
    for col in textbook_cols:
        df[col] = df[col].replace({'Yes': 1, 'No': 0}).infer_objects(copy=False)
    
    def textbook_access(df, grade_prefixes):
        cols = []
        for g in grade_prefixes:
            cols.extend([c for c in df.columns if c.startswith(g) and 'Textbook' in c])
        return df[cols].mean(axis=1) if cols else pd.Series(0, index=df.index)
    
    new_cols_df = pd.DataFrame({
        'Textbook_Access_1_4': textbook_access(df, lower_primary),
        'Textbook_Access_5_8': textbook_access(df, upper_primary),
        'Textbook_Access_9_10': textbook_access(df, secondary),
        'Textbook_Access_11_12': textbook_access(df, preparatory)
    })
    df = pd.concat([df, new_cols_df], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Create composite features
    if 'Overall_Textbook_Access_Composite' not in df.columns:
        df['Overall_Textbook_Access_Composite'] = df[['Textbook_Access_1_4', 'Textbook_Access_5_8',
                                                      'Textbook_Access_9_10', 'Textbook_Access_11_12']].mean(axis=1)
    
    # Attendance columns
    attendance_cols = [c for c in df.columns if 'Avg_Attendance_' in c]
    if attendance_cols:
        df['Overall_Avg_Attendance'] = df[attendance_cols].mean(axis=1)
    else:
        df['Overall_Avg_Attendance'] = 75
    
    # Homework columns
    homework_cols = [c for c in df.columns if 'Avg_HW_Completion_' in c]
    if homework_cols:
        df['Overall_Avg_Homework'] = df[homework_cols].mean(axis=1)
    else:
        df['Overall_Avg_Homework'] = 65
    
    # Participation columns
    participation_cols = [c for c in df.columns if 'Avg_Participation_' in c]
    if participation_cols:
        df['Overall_Avg_Participation'] = df[participation_cols].mean(axis=1)
    else:
        df['Overall_Avg_Participation'] = 70
    
    # Engagement score
    df['Overall_Engagement_Score'] = (
        df['Overall_Avg_Attendance'] * 0.4 +
        df['Overall_Avg_Homework'] * 0.3 +
        df['Overall_Avg_Participation'] * 0.3
    )
    
    # Drop aggregated columns
    drop_cols = [c for c in df.columns if c.startswith('Avg_') and c not in 
                ['Overall_Avg_Attendance', 'Overall_Avg_Homework', 'Overall_Avg_Participation']]
    df.drop(columns=drop_cols, inplace=True, errors='ignore')
    
    return df


def encode_categorical_features(df):
    """Apply categorical encoding to the dataset"""
    df_encoded = df.copy()
    
    CURRENT_DATE = pd.Timestamp('2026-01-30')
    MAX_UNIQUE_OHE = 8
    ALPHA = 10
    
    # Fill missing values
    if 'Health_Issue' in df_encoded.columns:
        df_encoded['Health_Issue'] = df_encoded['Health_Issue'].fillna('No Issue')
    
    for col in ['Father_Education', 'Mother_Education']:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].fillna('Unknown')
    
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
    edu_map = {
        'Unknown': 0,
        'Primary': 1,
        'High School': 2,
        'College': 3,
        'University': 4
    }
    
    for col in ['Father_Education', 'Mother_Education']:
        if col in df_encoded.columns:
            df_encoded[col + '_Encoded'] = df_encoded[col].map(edu_map)
            df_encoded.drop(columns=[col], inplace=True)
    
    # Health Issue encoding
    if 'Health_Issue' in df_encoded.columns:
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
        df_encoded['Health_Issue_Flag'] = np.where(df_encoded['Health_Issue'] == 'No Issue', 0, 1)
        df_encoded['Health_Issue_Severity'] = (
            df_encoded['Health_Issue']
            .map(severity_map)
            .fillna(1)
            .astype(int)
        )
        df_encoded.drop(columns=['Health_Issue'], inplace=True)
    
    # Date of Birth to Age
    if 'Date_of_Birth' in df_encoded.columns:
        df_encoded['Date_of_Birth'] = pd.to_datetime(df_encoded['Date_of_Birth'], errors='coerce')
        df_encoded['Age'] = ((CURRENT_DATE - df_encoded['Date_of_Birth']).dt.days // 365).astype(float)
        df_encoded.drop(columns=['Date_of_Birth'], inplace=True)
    
    return df_encoded


def prepare_target_encoders(df_raw, df_clean):
    """Prepare target encoders from training data"""
    global target_encoders
    TARGET = 'Overall_Average'
    ALPHA = 10
    
    def target_encode_smooth(df, col, target, alpha=ALPHA):
        global_mean = df[target].mean()
        stats = df.groupby(col)[target].agg(['mean', 'count'])
        smooth = (stats['count'] * stats['mean'] + alpha * global_mean) / (stats['count'] + alpha)
        return smooth.to_dict()
    
    if 'Region' in df_raw.columns and TARGET in df_clean.columns:
        target_encoders['Region'] = target_encode_smooth(df_raw, 'Region', TARGET)
    
    if 'School_Type' in df_raw.columns and TARGET in df_clean.columns:
        target_encoders['School_Type'] = target_encode_smooth(df_raw, 'School_Type', TARGET)
    
    if 'Career_Interest' in df_raw.columns and TARGET in df_clean.columns:
        target_encoders['Career_Interest'] = target_encode_smooth(df_raw, 'Career_Interest', TARGET)
    
    if 'Health_Issue' in df_raw.columns and TARGET in df_clean.columns:
        target_encoders['Health_Issue'] = target_encode_smooth(df_raw, 'Health_Issue', TARGET)


def process_raw_input_for_prediction(raw_input, df_clean, target_encoders, reg_features):
    """Process raw input for prediction"""
    try:
        input_df = pd.DataFrame([raw_input])
        encoded_df = input_df.copy()
        
        # Binary encoding
        binary_maps = {
            'Gender': {'Male': 0, 'Female': 1},
            'Home_Internet_Access': {'No': 0, 'Yes': 1},
            'Electricity_Access': {'No': 0, 'Yes': 1},
            'School_Location': {'Rural': 0, 'Urban': 1}
        }
        
        for col, mapping in binary_maps.items():
            if col in encoded_df.columns:
                encoded_df[col] = encoded_df[col].map(mapping)
        
        # Field Choice encoding
        if 'Field_Choice' in encoded_df.columns:
            encoded_df['Field_Choice'] = encoded_df['Field_Choice'].map({'Social': 0, 'Natural': 1})
        
        # Parent education encoding
        edu_map = {
            'Unknown': 0,
            'Primary': 1,
            'High School': 2,
            'College': 3,
            'University': 4
        }
        
        for col in ['Father_Education', 'Mother_Education']:
            if col in encoded_df.columns:
                encoded_df[col + '_Encoded'] = encoded_df[col].map(edu_map)
        
        # Health Issue encoding
        if 'Health_Issue' in encoded_df.columns:
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
            encoded_df['Health_Issue_Flag'] = np.where(encoded_df['Health_Issue'] == 'No Issue', 0, 1)
            encoded_df['Health_Issue_Severity'] = (
                encoded_df['Health_Issue']
                .map(severity_map)
                .fillna(1)
                .astype(int)
            )
            
            if 'Health_Issue' in target_encoders:
                health_encoder = target_encoders['Health_Issue']
                encoded_df['Health_Issue_Target'] = encoded_df['Health_Issue'].map(health_encoder).fillna(0.5)
        
        # Region encoding
        if 'Region' in encoded_df.columns and 'Region' in target_encoders:
            region_encoder = target_encoders['Region']
            encoded_df['Region_Encoded'] = encoded_df['Region'].map(region_encoder).fillna(0.5)
        
        # School Type encoding
        if 'School_Type' in encoded_df.columns:
            if 'School_Type' in target_encoders:
                school_encoder = target_encoders['School_Type']
                encoded_df['School_Type_Target'] = encoded_df['School_Type'].map(school_encoder).fillna(0.5)
            encoded_df['School_Type_Freq'] = 0.5
        
        # Career Interest encoding
        if 'Career_Interest' in encoded_df.columns and 'Career_Interest' in target_encoders:
            career_encoder = target_encoders['Career_Interest']
            encoded_df['Career_Interest_Encoded'] = encoded_df['Career_Interest'].map(career_encoder).fillna(0.5)
        
        # Date to Age
        if 'Date_of_Birth' in encoded_df.columns:
            CURRENT_DATE = pd.Timestamp('2026-01-30')
            encoded_df['Date_of_Birth'] = pd.to_datetime(encoded_df['Date_of_Birth'], errors='coerce')
            encoded_df['Age'] = ((CURRENT_DATE - encoded_df['Date_of_Birth']).dt.days // 365).astype(float)
        
        # Align columns
        expected_columns = reg_features.tolist() if hasattr(reg_features, 'tolist') else list(reg_features) if reg_features else []
        
        aligned_df = pd.DataFrame(index=[0])
        
        for col in expected_columns:
            if col in encoded_df.columns:
                aligned_df[col] = encoded_df[col].iloc[0]
            elif col == 'Overall_Engagement_Score':
                attendance = raw_input.get('Overall_Avg_Attendance', 75)
                homework = raw_input.get('Overall_Avg_Homework', 65)
                participation = raw_input.get('Overall_Avg_Participation', 70)
                aligned_df[col] = (attendance * 0.4 + homework * 0.3 + participation * 0.3)
            elif col in ['Overall_Avg_Attendance', 'Overall_Avg_Homework', 'Overall_Avg_Participation']:
                aligned_df[col] = raw_input.get(col, 50)
            elif col in ['Teacher_Student_Ratio']:
                aligned_df[col] = 40.0
            elif col in ['Student_to_Resources_Ratio']:
                aligned_df[col] = 20.0
            elif col in ['Age']:
                aligned_df[col] = 15.0
            else:
                aligned_df[col] = 0.5
        
        return aligned_df.astype(float)
    except Exception as e:
        print(f"Error in process_raw_input_for_prediction: {e}")
        return None