"""
Data processing utilities for Ethiopian Student Performance Dashboard
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Color scheme for color blindness accessibility
COLOR_SCHEME = {
    'primary': '#2E86AB',  # Blue
    'secondary': '#A23B72',  # Purple
    'success': '#18A999',  # Teal
    'warning': '#F18F01',  # Orange
    'danger': '#C73E1D',  # Red
    'light': '#F0F3F5',  # Light gray
    'dark': '#2C3E50',  # Dark blue-gray
    'text': '#2C3E50',
    'background': '#FFFFFF',
    'low_perf': '#C73E1D',  # Red for low performance
    'medium_perf': '#F18F01',  # Orange for medium performance
    'high_perf': '#18A999'  # Teal for high performance
}

# Global variables for data
df_original = None
df_clean = None
target_encoders = {}

def load_original_data(file_path="C:/Users/DELL/AIgravity/ethiopian_students_dataset.csv"):
    """Load the original dataset"""
    global df_original
    df_original = pd.read_csv(file_path)
    print(f"Initial dataset shape: {df_original.shape}")
    return df_original

def load_and_preprocess_data(df=None):
    """Load and preprocess the Ethiopian student dataset - ENHANCED VERSION"""
    global df_original, df_clean
    
    if df is None:
        if df_original is None:
            load_original_data()
        df = df_original.copy()
    else:
        df = df.copy()
    
    # ================================
    # 1️⃣ INITIAL CLEANING & ENCODING
    # ================================
    # Drop Student_ID (never used in ML)
    df = df.drop(columns=['Student_ID'], errors='ignore')
    
    # Encode Field_Choice (Social=0, Natural=1)
    df['Field_Choice'] = df['Field_Choice'].map({'Social': 0, 'Natural': 1})
    
    # Fill missing Career_Interest with "Unknown"
    df['Career_Interest'] = df['Career_Interest'].fillna('Unknown')
    
    # ================================
    # 2️⃣ DEFINE EDUCATION STAGES
    # ================================
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
    
    # ================================
    # 3️⃣ HELPER FUNCTION TO AGGREGATE GRADES
    # ================================
    def stage_average(df, grades, metric_keywords):
        """
        Compute average across all columns for a given stage and metric keywords.
        Returns the average series and list of original columns used.
        """
        cols = []
        for g in grades:
            for keyword in metric_keywords:
                cols += [c for c in df.columns if c.startswith(g) and keyword.lower() in c.lower()]
        cols = list(set(cols))
        return df[cols].mean(axis=1), cols
    
    # ================================
    # 4️⃣ AGGREGATE TEST SCORE, ATTENDANCE, HW, PARTICIPATION
    # ================================
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
    df.drop(columns=list(set(cols_to_drop)), inplace=True)
    
    # ================================
    # 5️⃣ AGGREGATE TEXTBOOK ACCESS
    # ================================
    # Convert Yes/No → 1/0 safely
    textbook_cols = [c for c in df.columns if 'Textbook' in c]
    for col in textbook_cols:
        df[col] = df[col].replace({'Yes': 1, 'No': 0}).infer_objects(copy=False)
    
    # Helper function for textbook access per stage
    def textbook_access(df, grade_prefixes):
        cols = []
        for g in grade_prefixes:
            cols.extend([c for c in df.columns if c.startswith(g) and 'Textbook' in c])
        return df[cols].mean(axis=1) if len(cols) > 0 else pd.Series(0, index=df.index)
    
    # Create aggregated textbook access per stage
    new_cols_df = pd.DataFrame({
        'Textbook_Access_1_4': textbook_access(df, lower_primary),
        'Textbook_Access_5_8': textbook_access(df, upper_primary),
        'Textbook_Access_9_10': textbook_access(df, secondary),
        'Textbook_Access_11_12': textbook_access(df, preparatory)
    })
    df = pd.concat([df, new_cols_df], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]  # remove duplicates
    
    # ================================
    # 6️⃣ TRACK-BASED NATIONAL EXAMS
    # ================================
    # Subjects per track
    social_subjects = ['National_Exam_History', 'National_Exam_Geography',
                       'National_Exam_Economics', 'National_Exam_Math_Social']
    natural_subjects = ['National_Exam_Biology', 'National_Exam_Chemistry',
                        'National_Exam_Physics', 'National_Exam_Math_Natural']
    
    # Track-specific averages
    df['Social_Track_Subject_Avg'] = df[social_subjects].mean(axis=1)
    df['Natural_Track_Subject_Avg'] = df[natural_subjects].mean(axis=1)
    
    # Track-based assignment
    df['Track_Subject_Average'] = np.where(
        df['Field_Choice'] == 0,
        df['Social_Track_Subject_Avg'],
        df['Natural_Track_Subject_Avg']
    )
    
    # Common subjects for all students
    common_subjects = ['National_Exam_Aptitude', 'National_Exam_English',
                       'National_Exam_Civics_and_Ethical_Education']
    df['Common_Exam_Average'] = df[common_subjects].mean(axis=1)
    
    # Overall Track Exam Average
    df['Track_Exam_Average'] = (df['Common_Exam_Average'] + df['Track_Subject_Average']) / 2
    
    # DROP ORIGINAL HIGH-DIMENSION COLUMNS
    drop_cols = [c for c in df.columns if c.startswith('Grade_')]
    drop_cols += [c for c in df.columns if c.startswith('National_Exam_')]
    df = df.drop(columns=drop_cols)
    
    # -------------------------------
    # 0️⃣ Drop leaking exam average columns
    # -------------------------------
    leak_cols = [
        'Total_National_Exam_Score',
        'Social_Track_Subject_Avg',
        'Natural_Track_Subject_Avg',
        'Track_Exam_Average',
        'Track_Subject_Average',
        'Common_Exam_Average',
        'Avg_Score_Secondary',
        'Avg_Score_Preparatory',
        'Avg_Score_Lower_Primary',
        'Avg_Score_Upper_Primary',
        'Avg_Test_Score_Secondary', 'Avg_Test_Score_Preparatory',
        'Avg_Test_Score_Lower_Primary', 'Avg_Test_Score_Upper_Primary',
        'School_ID', 'Total_Test_Score']
    df = df.drop(columns=[c for c in leak_cols if c in df.columns])
    
    # fix null value
    df['Health_Issue'] = df['Health_Issue'].fillna('No Issue')
    df['Father_Education'] = df['Father_Education'].fillna('Unknown')
    df['Mother_Education'] = df['Mother_Education'].fillna('Unknown')
    
    # ============================================================
    # NEW ENHANCED PREPROCESSING CODE
    # ============================================================
    # -----------------------------
    # Create composite features
    # -----------------------------
    df['Overall_Textbook_Access_Composite'] = df[['Textbook_Access_1_4', 'Textbook_Access_5_8',
                                          'Textbook_Access_9_10', 'Textbook_Access_11_12']].mean(axis=1)
    
    # Attendance columns
    attendance_cols = [
        'Avg_Attendance_Lower_Primary',
        'Avg_Attendance_Upper_Primary',
        'Avg_Attendance_Secondary',
        'Avg_Attendance_Preparatory'
    ]
    df['Overall_Avg_Attendance'] = df[attendance_cols].mean(axis=1)
    
    # Homework columns
    homework_cols = [
        'Avg_HW_Completion_Lower_Primary',
        'Avg_HW_Completion_Upper_Primary',
        'Avg_HW_Completion_Secondary',
        'Avg_HW_Completion_Preparatory'
    ]
    df['Overall_Avg_Homework'] = df[homework_cols].mean(axis=1)
    
    # Participation columns
    participation_cols = [
        'Avg_Participation_Lower_Primary',
        'Avg_Participation_Upper_Primary',
        'Avg_Participation_Secondary',
        'Avg_Participation_Preparatory'
    ]
    df['Overall_Avg_Participation'] = df[participation_cols].mean(axis=1)
    
    # -----------------------------
    # Composite engagement score (weighted) - FIXED: Values are 1-100
    # -----------------------------
    df['Overall_Engagement_Score'] = (
        df['Overall_Avg_Attendance'] * 0.4 +
        df['Overall_Avg_Homework'] * 0.3 +
        df['Overall_Avg_Participation'] * 0.3
    )
    
    #==================================
    # DROP ORIGINAL HIGH-DIMENSION COLUMNS
    #==================================
    drop_cols = []
    # Test Scores
    drop_cols += [c for c in df.columns if c.startswith('Avg_Test_Score_')]
    # Textbook Access
    drop_cols += [c for c in df.columns if c.startswith('Textbook_Access_')]
    # Attendance, Participation, Homework
    drop_cols += [c for c in df.columns if c.startswith('Avg_Attendance_')]
    drop_cols += [c for c in df.columns if c.startswith('Avg_Participation_')]
    drop_cols += [c for c in df.columns if c.startswith('Avg_HW_Completion_')]
    # Drop safely
    df = df.drop(columns=drop_cols, errors='ignore')
    
    return df


def encode_categorical_features(df):
    """Apply ENHANCED categorical encoding to the dataset"""
    df_encoded = df.copy()
    
    # ============================================================
    # NEW ENHANCED ENCODING CODE
    # ============================================================
    # -------------------------------
    # 0️⃣ Configuration
    # -------------------------------
    CURRENT_DATE = pd.Timestamp('2026-01-30')
    MAX_UNIQUE_OHE = 8
    ALPHA = 10
    
    # TARGET variable name (adjust if needed)
    TARGET = 'Overall_Average' if 'Overall_Average' in df_encoded.columns else 'Total_National_Exam_Score'
    
    # -------------------------------
    # 1️⃣ Fill missing values
    # -------------------------------
    if 'Health_Issue' in df_encoded.columns:
        df_encoded['Health_Issue'] = df_encoded['Health_Issue'].fillna('No Issue')
    
    for col in ['Father_Education', 'Mother_Education']:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].fillna('Unknown')
    
    # -------------------------------
    # 2️⃣ Binary encoding (Yes/No features)
    # -------------------------------
    binary_maps = {
        'Gender': {'Male': 0, 'Female': 1},
        'Home_Internet_Access': {'No': 0, 'Yes': 1},
        'Electricity_Access': {'No': 0, 'Yes': 1},
        'School_Location': {'Rural': 0, 'Urban': 1}
    }
    
    for col, mapping in binary_maps.items():
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map(mapping)
    
    # -------------------------------
    # 3️⃣ Ordinal encoding (Parents Education)
    # -------------------------------
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
    
    # -------------------------------
    # 4️⃣ Smoothed Target Encoding Function
    # -------------------------------
    def target_encode_smooth(df, col, target, alpha=ALPHA):
        global_mean = df[target].mean()
        stats = df.groupby(col)[target].agg(['mean', 'count'])
        smooth = (stats['count'] * stats['mean'] + alpha * global_mean) / (stats['count'] + alpha)
        return df[col].map(smooth).fillna(global_mean)
    
    # -------------------------------
    # 5️⃣ HEALTH ISSUE — FIXED & IMPROVED
    # -------------------------------
    if 'Health_Issue' in df_encoded.columns:
        # 5.1 Binary flag: has any health issue
        df_encoded['Health_Issue_Flag'] = np.where(df_encoded['Health_Issue'] == 'No Issue', 0, 1)
        
        # 5.2 Severity encoding (domain-informed)
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
        df_encoded['Health_Issue_Severity'] = (
            df_encoded['Health_Issue']
            .map(severity_map)
            .fillna(1)
            .astype(int)
        )
        
        # 5.3 Target encoding (impact on outcome)
        if TARGET in df_encoded.columns:
            df_encoded['Health_Issue_Target'] = target_encode_smooth(df_encoded, 'Health_Issue', TARGET)
        
        # Drop original column
        df_encoded.drop(columns=['Health_Issue'], inplace=True)
    
    # -------------------------------
    # 6️⃣ Region encoding (Target Encoding)
    # -------------------------------
    if 'Region' in df_encoded.columns and TARGET in df_encoded.columns:
        df_encoded['Region_Encoded'] = target_encode_smooth(df_encoded, 'Region', TARGET)
        df_encoded.drop(columns=['Region'], inplace=True)
    
    # -------------------------------
    # 7️⃣ School Type (Frequency + Target Encoding)
    # -------------------------------
    if 'School_Type' in df_encoded.columns:
        freq_map = df_encoded['School_Type'].value_counts(normalize=True).to_dict()
        df_encoded['School_Type_Freq'] = df_encoded['School_Type'].map(freq_map)
        
        if TARGET in df_encoded.columns:
            df_encoded['School_Type_Target'] = target_encode_smooth(df_encoded, 'School_Type', TARGET)
        
        df_encoded.drop(columns=['School_Type'], inplace=True)
    
    # -------------------------------
    # 8️⃣ Career Interest (Target Encoding)
    # -------------------------------
    if 'Career_Interest' in df_encoded.columns and TARGET in df_encoded.columns:
        df_encoded['Career_Interest_Encoded'] = target_encode_smooth(df_encoded, 'Career_Interest', TARGET)
        df_encoded.drop(columns=['Career_Interest'], inplace=True)
    
    # -------------------------------
    # 9️⃣ Safe One-Hot Encoding (low-cardinality)
    # -------------------------------
    remaining_cats = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
    safe_ohe_cols = [col for col in remaining_cats if df_encoded[col].nunique() <= MAX_UNIQUE_OHE]
    if safe_ohe_cols:
        df_encoded = pd.get_dummies(df_encoded, columns=safe_ohe_cols, drop_first=True)
    
    # -------------------------------
    # 🔟 Date_of_Birth → Age
    # -------------------------------
    if 'Date_of_Birth' in df_encoded.columns:
        df_encoded['Date_of_Birth'] = pd.to_datetime(df_encoded['Date_of_Birth'], errors='coerce')
        df_encoded['Age'] = ((CURRENT_DATE - df_encoded['Date_of_Birth']).dt.days // 365).astype(float)
        df_encoded.drop(columns=['Date_of_Birth'], inplace=True)
    
    # -------------------------------
    # 🔟 Drop Raw Categorical Columns
    # -------------------------------
    drop_cols = [
        'Father_Education', 'Mother_Education', 'Career_Interest',
        'Health_Issue', 'Region', 'Date_of_Birth',
        'School_ID', 'School_Type', 'Health_Issue_Binary'
    ]
    df_encoded.drop(columns=[c for c in drop_cols if c in df_encoded.columns], inplace=True)
    
    return df_encoded


def prepare_target_encoders(df_raw, df_clean):
    """Prepare target encoders from training data for prediction"""
    global target_encoders
    TARGET = 'Overall_Average'
    ALPHA = 10
    
    def target_encode_smooth(df, col, target, alpha=ALPHA):
        global_mean = df[target].mean()
        stats = df.groupby(col)[target].agg(['mean', 'count'])
        smooth = (stats['count'] * stats['mean'] + alpha * global_mean) / (stats['count'] + alpha)
        return smooth.to_dict()
    
    # Region encoding
    if 'Region' in df_raw.columns and TARGET in df_clean.columns:
        target_encoders['Region'] = target_encode_smooth(df_raw, 'Region', TARGET)
    
    # School Type encoding
    if 'School_Type' in df_raw.columns and TARGET in df_clean.columns:
        target_encoders['School_Type'] = target_encode_smooth(df_raw, 'School_Type', TARGET)
    
    # Career Interest encoding
    if 'Career_Interest' in df_raw.columns and TARGET in df_clean.columns:
        target_encoders['Career_Interest'] = target_encode_smooth(df_raw, 'Career_Interest', TARGET)
    
    # Health Issue encoding
    if 'Health_Issue' in df_raw.columns and TARGET in df_clean.columns:
        target_encoders['Health_Issue'] = target_encode_smooth(df_raw, 'Health_Issue', TARGET)


def process_raw_input_for_prediction(raw_input, df_clean, target_encoders, reg_features):
    """
    Process raw input according to the specified steps:
    Step1: Convert raw input to DataFrame
    Step2: Apply SAME encoding rules as training
    Step3: ALIGN COLUMNS (THIS IS CRITICAL) - ADDED Overall_Engagement_Score
    Step4: Date → Age
    Step5: Return processed data for prediction
    """
    try:
        # Step 1: Convert raw input to DataFrame
        input_df = pd.DataFrame([raw_input])
        
        # Step 2: Apply SAME encoding rules as training
        # Create a copy for encoding
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
        
        # Ordinal encoding for parent education
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
            # Binary flag
            encoded_df['Health_Issue_Flag'] = np.where(encoded_df['Health_Issue'] == 'No Issue', 0, 1)
            
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
            encoded_df['Health_Issue_Severity'] = (
                encoded_df['Health_Issue']
                .map(severity_map)
                .fillna(1)
                .astype(int)
            )
            
            # Target encoding
            if 'Health_Issue' in target_encoders:
                health_encoder = target_encoders['Health_Issue']
                encoded_df['Health_Issue_Target'] = encoded_df['Health_Issue'].map(health_encoder).fillna(df_clean['Overall_Average'].mean())
        
        # Region encoding (Target Encoding)
        if 'Region' in encoded_df.columns and 'Region' in target_encoders:
            region_encoder = target_encoders['Region']
            encoded_df['Region_Encoded'] = encoded_df['Region'].map(region_encoder).fillna(df_clean['Overall_Average'].mean())
        
        # School Type encoding (Frequency + Target Encoding)
        if 'School_Type' in encoded_df.columns:
            # Frequency encoding
            if 'School_Type' in df_clean.columns:
                freq_map = df_clean['School_Type_Freq'].value_counts(normalize=True).to_dict() if 'School_Type_Freq' in df_clean.columns else {}
                encoded_df['School_Type_Freq'] = encoded_df['School_Type'].map(freq_map).fillna(0)
            
            # Target encoding
            if 'School_Type' in target_encoders:
                school_type_encoder = target_encoders['School_Type']
                encoded_df['School_Type_Target'] = encoded_df['School_Type'].map(school_type_encoder).fillna(df_clean['Overall_Average'].mean())
        
        # Career Interest encoding
        if 'Career_Interest' in encoded_df.columns and 'Career_Interest' in target_encoders:
            career_encoder = target_encoders['Career_Interest']
            encoded_df['Career_Interest_Encoded'] = encoded_df['Career_Interest'].map(career_encoder).fillna(df_clean['Overall_Average'].mean())
        
        # Step 4: Date → Age
        if 'Date_of_Birth' in encoded_df.columns:
            CURRENT_DATE = pd.Timestamp('2026-01-30')
            encoded_df['Date_of_Birth'] = pd.to_datetime(encoded_df['Date_of_Birth'], errors='coerce')
            encoded_df['Age'] = ((CURRENT_DATE - encoded_df['Date_of_Birth']).dt.days // 365).astype(float)
        
        # Step 3: ALIGN COLUMNS (CRITICAL) - WITH Overall_Engagement_Score
        # Get all expected columns from training
        expected_columns = reg_features.tolist() if hasattr(reg_features, 'tolist') else list(reg_features)
        
        # Create final aligned dataframe
        aligned_df = pd.DataFrame(index=[0])
        
        # First, collect all encoded columns we have
        encoded_columns = {}
        for col in encoded_df.columns:
            if col in expected_columns:
                encoded_columns[col] = encoded_df[col].iloc[0]
        
        # Add all expected columns with appropriate values
        for col in expected_columns:
            if col in encoded_columns:
                aligned_df[col] = encoded_columns[col]
            else:
                # Set appropriate default values based on column type
                if col in ['Gender', 'Home_Internet_Access', 'Electricity_Access',
                          'School_Location', 'Field_Choice', 'Health_Issue_Flag']:
                    aligned_df[col] = 0
                elif col in ['Father_Education_Encoded', 'Mother_Education_Encoded',
                            'Health_Issue_Severity']:
                    aligned_df[col] = 0
                elif col in ['Parental_Involvement', 'Region_Encoded', 'School_Resources_Score',
                            'School_Academic_Score',
                            'School_Type_Freq', 'School_Type_Target', 'Overall_Textbook_Access_Composite',
                            'Career_Interest_Encoded', 'Health_Issue_Target']:
                    aligned_df[col] = 0.5
                elif col in ['Overall_Engagement_Score']:
                    # Calculate Overall_Engagement_Score using the same formula as training
                    attendance = raw_input.get('Overall_Avg_Attendance', 75)
                    homework = raw_input.get('Overall_Avg_Homework', 65)
                    participation = raw_input.get('Overall_Avg_Participation', 70)
                    aligned_df[col] = (attendance * 0.4 + homework * 0.3 + participation * 0.3)
                elif col in ['Overall_Avg_Attendance', 'Overall_Avg_Homework',
                            'Overall_Avg_Participation']:
                    aligned_df[col] = raw_input.get(col, 50)
                elif col == 'Teacher_Student_Ratio':
                    aligned_df[col] = 40.0
                elif col == 'Student_to_Resources_Ratio':
                    aligned_df[col] = 20.0
                elif col == 'Age':
                    aligned_df[col] = 15.0
                elif any(x in col for x in ['Region_', 'School_Type_', 'Health_Issue_']):
                    aligned_df[col] = 0
                else:
                    if col in df_clean.columns:
                        aligned_df[col] = df_clean[col].median()
                    else:
                        aligned_df[col] = 0
        
        # Ensure numeric types
        aligned_df = aligned_df.astype(float)
        return aligned_df
    except Exception as e:
        print(f"Error in process_raw_input_for_prediction: {e}")
        return None