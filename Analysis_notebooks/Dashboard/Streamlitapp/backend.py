import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import streamlit as st

# Machine Learning imports
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import socketserver
# Patch for PySpark on Windows when imported via SHAP
if not hasattr(socketserver, 'UnixStreamServer'):
    socketserver.UnixStreamServer = type('UnixStreamServer', (object,), {})

import shap
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')
import sys
import os

# Color scheme for color blindness accessibility
COLOR_SCHEME = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Purple
    'success': '#18A999',      # Teal
    'warning': '#F18F01',      # Orange
    'danger': '#C73E1D',       # Red
    'light': '#F0F3F5',        # Light gray
    'dark': '#2C3E50',         # Dark blue-gray
    'text': '#2C3E50',
    'background': '#FFFFFF',
    'low_perf': '#C73E1D',
    'medium_perf': '#F18F01',
    'high_perf': '#18A999'
}

# NATIONAL EXAM MODEL PERFORMANCE
NATIONAL_EXAM_MODEL_PERFORMANCE = pd.DataFrame({
    'Model': [
        'Gradient Boosting', 'XGBoost', 'Random Forest',
        'Ridge Regression', 'Linear Regression', 'Lasso Regression'
    ],
    'R2_Score': [0.437061, 0.432566, 0.420462, 0.405090, 0.405089, 0.404188],
    'MAE': [0.081473, 0.081776, 0.082588, 0.083994, 0.083995, 0.083939],
    'RMSE': [0.107198, 0.107625, 0.108767, 0.110200, 0.110200, 0.110284]
})

NATIONAL_EXAM_FEATURE_IMPORTANCE = pd.DataFrame({
    'Feature': [
        'Overall_Avg_Homework', 'School_Academic_Score', 'Overall_Test_Score_Avg',
        'Overall_Avg_Attendance', 'Overall_Avg_Participation', 'School_Resources_Score',
        'Parental_Involvement', 'Teacher_Student_Ratio', 'Student_to_Resources_Ratio',
        'School_Type_Target', 'Overall_Engagement_Score', 'Overall_Textbook_Access_Composite',
        'Field_Choice', 'Career_Interest_Encoded'
    ],
    'Importance': [
        0.003189,0.090328,0.418729,0.016825,0.232912,0.015366,0.014162,
        0.007387,0.004941,0.002499,0.190178,0.001734,0.001365,0.000385
    ],
    'Importance_%': [
        0.318879,9.032789,41.872894,1.682508,23.291219,1.536566,1.416215,
        0.738663,0.494074,0.249928,19.017832,0.173439,0.136537,0.038457
    ]
})

@st.cache_data
def get_original_data():
    try:
        df_original = pd.read_csv(r"C:/Users/DELL/Documents/project_data/ethiopian_students_dataset.csv")
    except Exception as e:
        # Fallback to empty mock data if file not found, to allow app to run
        df_original = pd.DataFrame(columns=['Student_ID', 'Field_Choice', 'Career_Interest', 'Health_Issue', 
                                            'Father_Education', 'Mother_Education', 'Overall_Average'])
    return df_original

@st.cache_data
def load_and_preprocess_data(df_original):
    df = df_original.copy()
    if df.empty: return df
    
    df = df.drop(columns=['Student_ID'], errors='ignore')
    if 'Field_Choice' in df.columns:
        df['Field_Choice'] = df['Field_Choice'].map({'Social': 0, 'Natural': 1})
    if 'Career_Interest' in df.columns:
        df['Career_Interest'] = df['Career_Interest'].fillna('Unknown')
    
    lower_primary = ['Grade_1', 'Grade_2', 'Grade_3', 'Grade_4']
    upper_primary = ['Grade_5', 'Grade_6', 'Grade_7', 'Grade_8']
    secondary     = ['Grade_9', 'Grade_10']
    preparatory   = ['Grade_11', 'Grade_12']
    
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
        return df[cols].mean(axis=1) if len(cols)>0 else pd.Series(0, index=df.index), cols
    
    metrics_dict = {
        'Test_Score': ['Test_Score'], 'Attendance': ['Attendance'],
        'HW_Completion': ['Homework_Completion'], 'Participation': ['Participation']
    }
    
    cols_to_drop = []
    for metric_name, keywords in metrics_dict.items():
        for stage_name, grades in stages.items():
            col_name = f'Avg_{metric_name}_{stage_name}'
            df[col_name], original_cols = stage_average(df, grades, keywords)
            cols_to_drop += original_cols
    
    df.drop(columns=list(set(cols_to_drop)), inplace=True, errors='ignore')
    
    textbook_cols = [c for c in df.columns if 'Textbook' in c]
    for col in textbook_cols:
        df[col] = df[col].replace({'Yes': 1, 'No': 0}).infer_objects(copy=False)
    
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
    
    social_subjects = ['National_Exam_History', 'National_Exam_Geography', 'National_Exam_Economics', 'National_Exam_Math_Social']
    natural_subjects = ['National_Exam_Biology', 'National_Exam_Chemistry', 'National_Exam_Physics', 'National_Exam_Math_Natural']
    
    if all(c in df.columns for c in social_subjects + natural_subjects):
        df['Social_Track_Subject_Avg']  = df[social_subjects].mean(axis=1)
        df['Natural_Track_Subject_Avg'] = df[natural_subjects].mean(axis=1)
        df['Track_Subject_Average'] = np.where(df.get('Field_Choice', 0) == 0, df['Social_Track_Subject_Avg'], df['Natural_Track_Subject_Avg'])
        common_subjects = ['National_Exam_Aptitude', 'National_Exam_English', 'National_Exam_Civics_and_Ethical_Education']
        df['Common_Exam_Average'] = df[common_subjects].mean(axis=1)
        df['Track_Exam_Average'] = (df['Common_Exam_Average'] + df['Track_Subject_Average']) / 2

    drop_cols = [c for c in df.columns if c.startswith('Grade_') or c.startswith('National_Exam_')]
    df = df.drop(columns=drop_cols, errors='ignore')
    
    leak_cols = ['Total_National_Exam_Score', 'Social_Track_Subject_Avg', 'Natural_Track_Subject_Avg',
                 'Track_Exam_Average', 'Track_Subject_Average', 'Common_Exam_Average', 'Avg_Score_Secondary',
                 'Avg_Score_Preparatory', 'Avg_Score_Lower_Primary', 'Avg_Score_Upper_Primary',
                 'Avg_Test_Score_Secondary', 'Avg_Test_Score_Preparatory', 'Avg_Test_Score_Lower_Primary',
                 'Avg_Test_Score_Upper_Primary', 'School_ID', 'Total_Test_Score']
    df = df.drop(columns=[c for c in leak_cols if c in df.columns])
    
    if 'Health_Issue' in df.columns: df['Health_Issue'] = df['Health_Issue'].fillna('No Issue')
    if 'Father_Education' in df.columns: df['Father_Education'] = df['Father_Education'].fillna('Unknown')
    if 'Mother_Education' in df.columns: df['Mother_Education'] = df['Mother_Education'].fillna('Unknown')
    
    df['Overall_Textbook_Access_Composite'] = df[['Textbook_Access_1_4', 'Textbook_Access_5_8', 'Textbook_Access_9_10', 'Textbook_Access_11_12']].mean(axis=1)
    df['Overall_Avg_Attendance'] = df[[c for c in df.columns if c.startswith('Avg_Attendance_')]].mean(axis=1)
    df['Overall_Avg_Homework'] = df[[c for c in df.columns if c.startswith('Avg_HW_Completion_')]].mean(axis=1)
    df['Overall_Avg_Participation'] = df[[c for c in df.columns if c.startswith('Avg_Participation_')]].mean(axis=1)
    df['Overall_Engagement_Score'] = (df['Overall_Avg_Attendance'] * 0.4 + df['Overall_Avg_Homework'] * 0.3 + df['Overall_Avg_Participation'] * 0.3)
    
    drop_cols = [c for c in df.columns if c.startswith('Avg_Test_Score_')] + \
                [c for c in df.columns if c.startswith('Textbook_Access_')] + \
                [c for c in df.columns if c.startswith('Avg_Attendance_')] + \
                [c for c in df.columns if c.startswith('Avg_Participation_')] + \
                [c for c in df.columns if c.startswith('Avg_HW_Completion_')]
    df = df.drop(columns=drop_cols, errors='ignore')
    return df

@st.cache_data
def encode_categorical_features(df):
    df_encoded = df.copy()
    if df_encoded.empty: return df_encoded
    
    CURRENT_DATE = pd.Timestamp('2026-01-30')
    MAX_UNIQUE_OHE = 8
    ALPHA = 10
    TARGET = 'Overall_Average' if 'Overall_Average' in df_encoded.columns else 'Total_National_Exam_Score'
    
    binary_maps = {'Gender': {'Male': 0, 'Female': 1}, 'Home_Internet_Access': {'No': 0, 'Yes': 1},
                   'Electricity_Access': {'No': 0, 'Yes': 1}, 'School_Location': {'Rural': 0, 'Urban': 1}}
    for col, mapping in binary_maps.items():
        if col in df_encoded.columns: df_encoded[col] = df_encoded[col].map(mapping)
    
    edu_map = {'Unknown': 0, 'Primary': 1, 'High School': 2, 'College': 3, 'University': 4}
    for col in ['Father_Education', 'Mother_Education']:
        if col in df_encoded.columns:
            df_encoded[col + '_Encoded'] = df_encoded[col].map(edu_map)
            df_encoded.drop(columns=[col], inplace=True)
            
    def target_encode_smooth(df, col, target, alpha=ALPHA):
        if target not in df.columns: return pd.Series(0, index=df.index)
        global_mean = df[target].mean()
        stats = df.groupby(col)[target].agg(['mean', 'count'])
        smooth = (stats['count'] * stats['mean'] + alpha * global_mean) / (stats['count'] + alpha)
        return df[col].map(smooth).fillna(global_mean)

    if 'Health_Issue' in df_encoded.columns:
        df_encoded['Health_Issue_Flag'] = np.where(df_encoded['Health_Issue'] == 'No Issue', 0, 1)
        severity_map = {'No Issue': 0, 'Dental Problems': 1, 'Vision Issues': 1, 'Hearing Issues': 1,
                        'Anemia': 2, 'Parasitic Infections': 2, 'Respiratory Issues': 2, 'Malnutrition': 2,
                        'Physical Disability': 3, 'Chronic Illness': 3}
        df_encoded['Health_Issue_Severity'] = df_encoded['Health_Issue'].map(severity_map).fillna(1).astype(int)
        if TARGET in df_encoded.columns: df_encoded['Health_Issue_Target'] = target_encode_smooth(df_encoded, 'Health_Issue', TARGET)
        df_encoded.drop(columns=['Health_Issue'], inplace=True)
        
    if 'Region' in df_encoded.columns:
        df_encoded['Region_Encoded'] = target_encode_smooth(df_encoded, 'Region', TARGET)
        df_encoded.drop(columns=['Region'], inplace=True)
        
    if 'School_Type' in df_encoded.columns:
        freq_map = df_encoded['School_Type'].value_counts(normalize=True).to_dict()
        df_encoded['School_Type_Freq'] = df_encoded['School_Type'].map(freq_map)
        if TARGET in df_encoded.columns: df_encoded['School_Type_Target'] = target_encode_smooth(df_encoded, 'School_Type', TARGET)
        df_encoded.drop(columns=['School_Type'], inplace=True)
        
    if 'Career_Interest' in df_encoded.columns:
        df_encoded['Career_Interest_Encoded'] = target_encode_smooth(df_encoded, 'Career_Interest', TARGET)
        df_encoded.drop(columns=['Career_Interest'], inplace=True)
        
    remaining_cats = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
    safe_ohe_cols = [col for col in remaining_cats if df_encoded[col].nunique() <= MAX_UNIQUE_OHE]
    if safe_ohe_cols: df_encoded = pd.get_dummies(df_encoded, columns=safe_ohe_cols, drop_first=True)
    
    if 'Date_of_Birth' in df_encoded.columns:
        df_encoded['Date_of_Birth'] = pd.to_datetime(df_encoded['Date_of_Birth'], errors='coerce')
        df_encoded['Age'] = ((CURRENT_DATE - df_encoded['Date_of_Birth']).dt.days // 365).astype(float)
        df_encoded.drop(columns=['Date_of_Birth'], inplace=True)
        
    drop_cols = ['Father_Education', 'Mother_Education','Career_Interest', 'Health_Issue', 'Region','Date_of_Birth', 'School_ID', 'School_Type','Health_Issue_Binary']
    df_encoded.drop(columns=[c for c in drop_cols if c in df_encoded.columns], inplace=True)
    
    return df_encoded

@st.cache_resource
def train_regression_models(df):
    TARGET = 'Overall_Average'
    if TARGET not in df.columns: return {}, {}, "No Model", None, None, [], None, None
    X = df.drop(columns=[TARGET], errors='ignore').select_dtypes(include=[np.number])
    if X.shape[1] == 0: return {}, {}, "No Model", None, None, [], None, None
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # We load only XGBoost and GradientBoosting to make it faster
    models = {
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1)
    }
    
    results, feature_importances, trained_models = {}, {}, {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        results[name] = {"mae": mean_absolute_error(y_test, y_pred), "rmse": np.sqrt(mean_squared_error(y_test, y_pred)), 
                         "r2": r2_score(y_test, y_pred), "y_test": y_test, "y_pred": y_pred}
        trained_models[name] = model
        if hasattr(model, 'feature_importances_'):
            feature_importances[name] = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            
    best_model_name = max(results, key=lambda x: results[x]['r2']) if results else "No Model"
    best_model = trained_models.get(best_model_name)
    return results, feature_importances, best_model_name, best_model, scaler, X.columns, X_train, X_test, trained_models

@st.cache_resource
def train_risk_classification(df):
    if 'Overall_Average' not in df.columns: return None
    df_copy = df.copy()
    df_copy['Risk_NotRisk'] = (df_copy['Overall_Average'] < 50).astype(int)
    X = df_copy.drop(['Risk_NotRisk', 'Overall_Average'], axis=1, errors='ignore').select_dtypes(include=[np.number])
    y = df_copy['Risk_NotRisk']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    
    gb_base = GradientBoostingClassifier(n_estimators=50, random_state=42)
    gb_base.fit(X_train_res, y_train_res)
    
    y_probs = gb_base.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_probs >= 0.50).astype(int)
    
    return {
        'f1': f1_score(y_test, y_pred, pos_label=1), 'roc_auc': roc_auc_score(y_test, y_probs),
        'cm': confusion_matrix(y_test, y_pred), 'y_test': y_test, 'y_probs': y_probs,
        'model': gb_base, 'feature_importance': pd.Series(gb_base.feature_importances_, index=X.columns).sort_values(ascending=False),
        'scaler': scaler, 'feature_names': X.columns.tolist(), 'X_train': X_train, 'X_test': X_test
    }

@st.cache_data
def perform_clustering():
    cluster_sizes_data = {'Low': 39380, 'Medium': 38933, 'High': 21687}
    regional_risk_data = {'Somali': 47.398699, 'Benishangul-Gumuz': 45.542895, 'Afar': 45.271891, 'Tigray': 44.758569,
                          'Sidama': 43.237808, 'Gambela': 42.241869, 'SNNP': 40.569923, 'Oromia': 39.208222,
                          'Amhara': 39.180777, 'South West Ethiopia': 39.175258, 'Dire Dawa': 31.365403,
                          'Harari': 28.723770, 'Addis Ababa': 21.323982}
    regional_cluster_distribution = {
        'Addis Ababa': {'Low': 0.21, 'Medium': 0.60, 'High': 0.19},
        'Afar': {'Low': 0.45, 'Medium': 0.33, 'High': 0.22},
        'Amhara': {'Low': 0.39, 'Medium': 0.38, 'High': 0.23},
        'Benishangul-Gumuz': {'Low': 0.46, 'Medium': 0.32, 'High': 0.23},
        'Dire Dawa': {'Low': 0.31, 'Medium': 0.48, 'High': 0.21},
        'Gambela': {'Low': 0.42, 'Medium': 0.36, 'High': 0.22},
        'Harari': {'Low': 0.29, 'Medium': 0.50, 'High': 0.21},
        'Oromia': {'Low': 0.39, 'Medium': 0.39, 'High': 0.22},
        'SNNP': {'Low': 0.41, 'Medium': 0.37, 'High': 0.22},
        'Sidama': {'Low': 0.43, 'Medium': 0.34, 'High': 0.23},
        'Somali': {'Low': 0.47, 'Medium': 0.30, 'High': 0.23},
        'South West Ethiopia': {'Low': 0.39, 'Medium': 0.39, 'High': 0.22},
        'Tigray': {'Low': 0.45, 'Medium': 0.34, 'High': 0.22}
    }
    return pd.Series(cluster_sizes_data), pd.Series(regional_risk_data), pd.DataFrame(regional_cluster_distribution).T

# PLOTTING FUNCTIONS
def create_datatype_bar_plot(df_original):
    dtypes = df_original.dtypes.value_counts()
    fig = go.Figure(data=[go.Bar(x=dtypes.index.astype(str), y=dtypes.values, marker_color=COLOR_SCHEME['primary'], text=dtypes.values, textposition='auto')])
    fig.update_layout(title="Data Type Distribution", plot_bgcolor=COLOR_SCHEME['background'], paper_bgcolor=COLOR_SCHEME['background'], font=dict(color=COLOR_SCHEME['text']))
    return fig

def create_score_distribution_plot(df_clean):
    fig = go.Figure()
    if 'Overall_Average' in df_clean.columns:
        fig.add_trace(go.Histogram(x=df_clean['Overall_Average'], nbinsx=30, marker_color=COLOR_SCHEME['primary'], opacity=0.7))
    fig.update_layout(title="Distribution of Overall Average Scores", plot_bgcolor=COLOR_SCHEME['background'], paper_bgcolor=COLOR_SCHEME['background'])
    return fig

def create_students_by_region_plot(df_raw):
    fig = go.Figure()
    if 'Region' in df_raw.columns:
        counts = df_raw['Region'].value_counts().sort_values(ascending=True)
        fig.add_trace(go.Bar(y=counts.index, x=counts.values, orientation='h', marker_color=COLOR_SCHEME['primary'], text=counts.values, textposition='auto'))
    fig.update_layout(title="Number of Students by Region", height=500, plot_bgcolor=COLOR_SCHEME['background'], paper_bgcolor=COLOR_SCHEME['background'])
    return fig

def create_correlation_heatmap(df_clean):
    numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if 'Risk_NotRisk' in numeric_cols: numeric_cols.remove('Risk_NotRisk')
    if len(numeric_cols) > 15 and 'Overall_Average' in df_clean.columns:
        correlations = df_clean[numeric_cols].corr()['Overall_Average'].abs().sort_values(ascending=False)
        top_features = correlations.head(15).index.tolist()
        corr_data = df_clean[top_features].corr()
    else:
        corr_data = df_clean[numeric_cols].corr()
    fig = go.Figure(data=go.Heatmap(z=corr_data.values, x=corr_data.columns, y=corr_data.index, colorscale='RdBu', zmid=0))
    fig.update_layout(title="Feature Correlation Heatmap", height=600, plot_bgcolor=COLOR_SCHEME['background'], paper_bgcolor=COLOR_SCHEME['background'])
    return fig

def create_regression_comparison_plot(regression_models, best_reg_model):
    if not regression_models: return go.Figure()
    models = list(regression_models.keys())
    r2_scores = [regression_models[m]['r2'] for m in models]
    colors = [COLOR_SCHEME['secondary'] if m == best_reg_model else COLOR_SCHEME['primary'] for m in models]
    fig = go.Figure(data=[go.Bar(x=models, y=r2_scores, marker_color=colors, text=[f'{score:.3f}' for score in r2_scores], textposition='auto')])
    fig.update_layout(title="Regression Model R² Score Comparison", plot_bgcolor=COLOR_SCHEME['background'], paper_bgcolor=COLOR_SCHEME['background'])
    return fig

def create_actual_vs_predicted_plot(regression_models, best_reg_model):
    if not regression_models or best_reg_model not in regression_models: return go.Figure()
    best_reg = regression_models[best_reg_model]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=best_reg['y_test'], y=best_reg['y_pred'], mode='markers', name='Predictions', marker=dict(color=COLOR_SCHEME['primary'], size=6, opacity=0.6)))
    fig.add_trace(go.Scatter(x=[best_reg['y_test'].min(), best_reg['y_test'].max()], y=[best_reg['y_test'].min(), best_reg['y_test'].max()], mode='lines', line=dict(dash='dash', color=COLOR_SCHEME['secondary'])))
    fig.update_layout(title=f"Actual vs Predicted - {best_reg_model}", plot_bgcolor=COLOR_SCHEME['background'], paper_bgcolor=COLOR_SCHEME['background'])
    return fig

def create_confusion_matrix_plot(cm):
    fig = go.Figure(data=go.Heatmap(z=cm, x=['Not Risk', 'Risk'], y=['Not Risk', 'Risk'], colorscale='Blues', text=cm, texttemplate='%{text}'))
    fig.update_layout(title="Risk Classification Confusion Matrix", plot_bgcolor=COLOR_SCHEME['background'], paper_bgcolor=COLOR_SCHEME['background'])
    return fig
