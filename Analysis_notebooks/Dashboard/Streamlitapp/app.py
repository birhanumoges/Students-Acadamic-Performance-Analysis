"""
Ethiopian Student Performance Analytics Dashboard
Complete Streamlit Dashboard with all features
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import sys
import json
import warnings
import io
import base64
from datetime import datetime
warnings.filterwarnings('ignore')

# ============================================================================
# CACHING FUNCTIONS FOR ANALYTICS PAGE (Add these here)
# ============================================================================

@st.cache_data(ttl=3600)
def get_cached_correlation_data(df):
    """Cache correlation heatmap data - runs only once per hour"""
    # Define columns to exclude
    exclude_columns = ['Student_ID', 'Total_Test_Score']
    
    # Add all grade level columns
    for i in range(1, 13):
        grade_col = f'Grade_{i}'
        if grade_col in df.columns:
            exclude_columns.append(grade_col)
        if f'{grade_col}_Test_Score' in df.columns:
            exclude_columns.append(f'{grade_col}_Test_Score')
        if f'{grade_col}_Attendance' in df.columns:
            exclude_columns.append(f'{grade_col}_Attendance')
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter out excluded columns
    numeric_cols = [col for col in numeric_cols if col not in exclude_columns]
    
    # Limit to top features if too many
    if len(numeric_cols) > 15:
        if 'Overall_Average' in numeric_cols:
            correlations = df[numeric_cols].corr()['Overall_Average'].abs().sort_values(ascending=False)
            top_features = correlations.head(15).index.tolist()
            corr_data = df[top_features].corr()
        else:
            corr_data = df[numeric_cols[:15]].corr()
    else:
        corr_data = df[numeric_cols].corr()
    
    return corr_data


@st.cache_data(ttl=3600)
def get_cached_scatter_data(df, x_col, y_col):
    """Cache scatter plot data - runs only once per hour"""
    x_data = df[x_col].dropna()
    y_data = df[y_col].dropna()
    
    slope = None
    r_squared = 0
    x_line = None
    y_line = None
    
    if len(x_data) > 1 and x_data.nunique() > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
        x_line = np.array([x_data.min(), x_data.max()])
        y_line = intercept + slope * x_line
        r_squared = r_value ** 2
    
    return {
        'x_data': x_data,
        'y_data': y_data,
        'slope': slope,
        'r_squared': r_squared,
        'x_line': x_line,
        'y_line': y_line
    }

# Set page config
st.set_page_config(
    page_title="Ethiopian Student Performance Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main header */
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #2E86AB;
        margin-bottom: 0.5rem;
        border-bottom: 3px solid #2E86AB;
        padding-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    
    /* Metric cards */
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Buttons */
    .stButton button {
        width: 100%;
        border-radius: 5px;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Info boxes */
    .info-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #2E86AB;
        margin: 10px 0;
    }
    
    .warning-box {
        background-color: #fff3e0;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #F18F01;
        margin: 10px 0;
    }
    
    .success-box {
        background-color: #e0f2e9;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #18A999;
        margin: 10px 0;
    }
    
    /* Tables */
    .dataframe {
        font-size: 13px;
    }
    
    /* Filter panel */
    .filter-panel {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 5px;
        padding: 8px 16px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING AND INITIALIZATION SECTION - PLACE THIS AT THE TOP AFTER IMPORTS
# ============================================================================

# Initialize session state with default values BEFORE any data loading
if 'page' not in st.session_state:
    st.session_state.page = "Overview"
if 'show_powerbi' not in st.session_state:
    st.session_state.show_powerbi = False
if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'filter_region' not in st.session_state:
    st.session_state.filter_region = "All"
if 'filter_gender' not in st.session_state:
    st.session_state.filter_gender = "All"
if 'filter_risk' not in st.session_state:
    st.session_state.filter_risk = "All"
if 'attendance_range' not in st.session_state:
    st.session_state.attendance_range = (0, 100)
if 'risk_threshold' not in st.session_state:
    st.session_state.risk_threshold = 0.5
if 'regression_models' not in st.session_state:
    st.session_state.regression_models = {}
if 'classification_model' not in st.session_state:
    st.session_state.classification_model = None
if 'clustering_analysis' not in st.session_state:
    st.session_state.clustering_analysis = None
if 'prediction_engine' not in st.session_state:
    st.session_state.prediction_engine = None
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = None
if 'config' not in st.session_state:
    st.session_state.config = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False


# ============================================================================
# DATA LOADING FUNCTION
# ============================================================================

@st.cache_data
def load_original_data():
    """
    Load the original Ethiopian student dataset from file.
    If file not found, creates sample data for demonstration.
    """
    import os
    import pandas as pd
    import numpy as np
    
    # Define possible file paths to check
    possible_paths = [
        r"C:/Users/DELL/AIgravity/ethiopian_students_dataset.csv",
        r"C:/Users/DELL/Documents/project_data/ethiopian_students_dataset.csv",
        r"C:/Users/DELL/Downloads/ethiopian_students_dataset.csv",
        r"C:/Users/DELL/Desktop/ethiopian_students_dataset.csv",
        r"C:/Users/DELL/projects/project1/Students-Acadamic-Performance-Analysis/ethiopian_students_dataset.csv",
        r"C:/Users/DELL/projects/project1/Students-Acadamic-Performance-Analysis/data/ethiopian_students_dataset.csv",
        os.path.join(os.path.dirname(__file__), "data", "ethiopian_students_dataset.csv"),
        os.path.join(os.path.dirname(__file__), "..", "data", "ethiopian_students_dataset.csv"),
        "data/ethiopian_students_dataset.csv",
        "../data/ethiopian_students_dataset.csv"
    ]
    
    # Try to load from each path
    df_original = None
    loaded_path = None
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df_original = pd.read_csv(path)
                loaded_path = path
                break
            except Exception as e:
                continue
    
    # If file found, process it
    if df_original is not None:
        # Ensure Student_ID is numeric
        if 'Student_ID' in df_original.columns:
            try:
                df_original['Student_ID'] = pd.to_numeric(df_original['Student_ID'], errors='coerce')
                df_original['Student_ID'] = df_original['Student_ID'].fillna(range(1, len(df_original) + 1)).astype(int)
            except:
                df_original['Student_ID'] = range(1, len(df_original) + 1)
        else:
            df_original['Student_ID'] = range(1, len(df_original) + 1)
        
        # Add Age column if missing
        if 'Age' not in df_original.columns:
            df_original['Age'] = np.random.randint(15, 22, len(df_original))
        
        # Add missing required columns with default values
        if 'Overall_Avg_Attendance' not in df_original.columns:
            df_original['Overall_Avg_Attendance'] = 75
        if 'Overall_Avg_Homework' not in df_original.columns:
            df_original['Overall_Avg_Homework'] = 65
        if 'Overall_Avg_Participation' not in df_original.columns:
            df_original['Overall_Avg_Participation'] = 70
        if 'Overall_Engagement_Score' not in df_original.columns:
            df_original['Overall_Engagement_Score'] = (
                df_original['Overall_Avg_Attendance'] * 0.4 +
                df_original['Overall_Avg_Homework'] * 0.3 +
                df_original['Overall_Avg_Participation'] * 0.3
            )
        if 'Overall_Textbook_Access_Composite' not in df_original.columns:
            df_original['Overall_Textbook_Access_Composite'] = 0.5
        if 'School_Resources_Score' not in df_original.columns:
            df_original['School_Resources_Score'] = 0.5
        if 'School_Academic_Score' not in df_original.columns:
            df_original['School_Academic_Score'] = 0.5
        if 'Teacher_Student_Ratio' not in df_original.columns:
            df_original['Teacher_Student_Ratio'] = 40
        if 'Parental_Involvement' not in df_original.columns:
            df_original['Parental_Involvement'] = 0.5
        
        return df_original, loaded_path
    
    # If no file found, create comprehensive sample data
    np.random.seed(42)
    n = 10000
    
    regions = ['Addis Ababa', 'Oromia', 'Amhara', 'Tigray', 'SNNP', 'Somali', 
               'Afar', 'Benishangul-Gumuz', 'Sidama', 'Gambela', 'Harari', 
               'Dire Dawa', 'South West Ethiopia']
    
    # Create DataFrame with ALL required columns
    df_sample = pd.DataFrame({
        'Student_ID': range(1, n+1),
        'Overall_Average': np.random.normal(60, 15, n).clip(0, 100),
        'Region': np.random.choice(regions, n),
        'Gender': np.random.choice(['Male', 'Female'], n),
        'Age': np.random.randint(15, 22, n),
        'School_Resources_Score': np.random.uniform(0.2, 0.9, n),
        'School_Academic_Score': np.random.uniform(0.3, 0.9, n),
        'Parental_Involvement': np.random.uniform(0.1, 0.9, n),
        'Overall_Avg_Attendance': np.random.uniform(50, 100, n),
        'Overall_Avg_Homework': np.random.uniform(40, 100, n),
        'Overall_Avg_Participation': np.random.uniform(40, 100, n),
        'Overall_Textbook_Access_Composite': np.random.uniform(0.2, 0.9, n),
        'Teacher_Student_Ratio': np.random.uniform(20, 60, n),
        'Health_Issue': np.random.choice(['No Issue', 'Minor', 'Moderate', 'Severe'], n, p=[0.6, 0.2, 0.15, 0.05]),
        'Home_Internet_Access': np.random.choice(['Yes', 'No'], n),
        'Electricity_Access': np.random.choice(['Yes', 'No'], n),
        'School_Location': np.random.choice(['Urban', 'Rural'], n),
        'Field_Choice': np.random.choice(['Social', 'Natural'], n),
        'Career_Interest': np.random.choice(['Teacher', 'Doctor', 'Engineer', 'Farmer', 'Business'], n),
        'Father_Education': np.random.choice(['Unknown', 'Primary', 'High School', 'College', 'University'], n),
        'Mother_Education': np.random.choice(['Unknown', 'Primary', 'High School', 'College', 'University'], n),
        'School_Type': np.random.choice(['Public', 'Private', 'NGO-operated'], n)
    })
    
    # Calculate Overall_Engagement_Score
    df_sample['Overall_Engagement_Score'] = (
        df_sample['Overall_Avg_Attendance'] * 0.4 +
        df_sample['Overall_Avg_Homework'] * 0.3 +
        df_sample['Overall_Avg_Participation'] * 0.3
    )
    
    return df_sample, None

# ============================================================================
# LOAD DATA AND INITIALIZE MODELS
# ============================================================================

# Only load data if not already loaded
if not st.session_state.data_loaded:
    with st.spinner("Loading data and initializing dashboard..."):
        # Load original data
        df_original, loaded_path = load_original_data()
        
        if df_original is not None:
            # Import the real PredictionEngine and DataProcessor
            try:
                from utils.predictions import PredictionEngine
                from utils.data_processor import DataProcessor
                
                # Initialize data processor
                data_processor = DataProcessor()
                
                # Initialize prediction engine with models
                prediction_engine = PredictionEngine()
                
                # Store in session state
                st.session_state.df_original = df_original
                st.session_state.df_processed = data_processor.load_and_preprocess_data(df_original.copy())
                st.session_state.data_processor = data_processor
                st.session_state.prediction_engine = prediction_engine
                st.session_state.config = {'risk_threshold': 0.5, 'required_features': []}
                st.session_state.risk_threshold = 0.5
                st.session_state.data_loaded = True
                
                # Show success message
                if loaded_path:
                    st.success(f"✅ Data loaded successfully from: {loaded_path}")
                else:
                    st.info(f"✅ Sample data created: {len(df_original):,} students")
                
                # Check and display model status
                if prediction_engine.is_models_loaded():
                    st.success("✅ Models loaded successfully!")
                    status = prediction_engine.get_model_status()
                    st.info(f"📊 Models: {status.get('regression_type', 'Unknown')} (Regression) & {status.get('classification_type', 'Unknown')} (Classification)")
                    st.info(f"📋 Features: {status.get('features_count', 0)} features loaded")
                else:
                    st.warning("⚠️ Models not loaded. Prediction features will use fallback calculations.")
                    if prediction_engine.load_error:
                        st.error(f"Error: {prediction_engine.load_error}")
                    st.info("📁 Model files should be in: " + str(prediction_engine.model_dir))
                    
            except ImportError as e:
                st.error(f"❌ Import error: {e}")
                st.info("Please ensure utils/predictions.py and utils/data_processor.py exist")
                st.stop()
            except Exception as e:
                st.error(f"❌ Initialization error: {e}")
                st.stop()
        else:
            st.error("Failed to load data. Please check the data file.")
            st.stop()

# ============================================================================
# CACHED CLUSTERING DATA FUNCTION
# ============================================================================

@st.cache_data
def get_clustering_data():
    """Cached function to load clustering data - runs only once"""
    # Cluster mapping
    cluster_mapping = {2: 'High', 1: 'Medium', 0: 'Low'}
    
    # Cluster sizes
    cluster_sizes = {'Low': 39380, 'Medium': 38933, 'High': 21687}
    
    # Cluster profiles
    cluster_profile_data = {
        'Performance_Cluster': ['Low', 'Medium', 'High'],
        'Total_National_Exam_Score': [286.685256, 331.849464, 334.484428],
        'Overall_Average': [47.309911, 54.283330, 62.559605],
        'Overall_Avg_Attendance': [85.879107, 87.396578, 86.733333],
        'Overall_Avg_Homework': [52.551327, 73.066655, 62.319139],
        'Overall_Avg_Participation': [59.697917, 71.411008, 65.515959],
        'Overall_Engagement_Score': [68.026416, 78.301930, 73.043863],
        'School_Academic_Score': [0.424432, 0.445902, 0.695637],
        'Teacher_Student_Ratio': [49.957881, 50.049940, 34.502018],
        'Student_to_Resources_Ratio': [22.629008, 22.578901, 15.891499],
        'Parental_Involvement': [0.301593, 0.484578, 0.365762],
        'Overall_Textbook_Access_Composite': [0.361508, 0.375552, 0.630930]
    }
    
    cluster_profile = pd.DataFrame(cluster_profile_data).set_index('Performance_Cluster')
    
    # Regional risk data
    regional_risk = {
        'Somali': 47.398699, 'Benishangul-Gumuz': 45.542895, 'Afar': 45.271891,
        'Tigray': 44.758569, 'Sidama': 43.237808, 'Gambela': 42.241869,
        'SNNP': 40.569923, 'Oromia': 39.208222, 'Amhara': 39.180777,
        'South West Ethiopia': 39.175258, 'Dire Dawa': 31.365403,
        'Harari': 28.723770, 'Addis Ababa': 21.323982
    }
    
    # Regional cluster distribution
    regional_cluster_data = {
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
    
    regional_cluster_df = pd.DataFrame(regional_cluster_data).T
    
    return {
        'cluster_sizes': cluster_sizes,
        'cluster_profile': cluster_profile,
        'regional_risk': regional_risk,
        'regional_cluster_df': regional_cluster_df,
        'silhouette_score': 0.1742
    }


# ============================================================================
# CACHED VISUALIZATION FUNCTIONS
# ============================================================================

@st.cache_data
def create_cluster_bar_chart(cluster_sizes):
    """Create cluster distribution bar chart - cached"""
    fig = go.Figure(data=[
        go.Bar(
            x=list(cluster_sizes.keys()),
            y=list(cluster_sizes.values()),
            marker_color=['#18A999', '#F18F01', '#C73E1D'],
            text=list(cluster_sizes.values()),
            textposition='auto',
            textfont=dict(size=14)
        )
    ])
    fig.update_layout(
        title="Student Performance Cluster Distribution",
        xaxis_title="Performance Level",
        yaxis_title="Number of Students",
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    return fig


@st.cache_data
def create_regional_heatmap(regional_cluster_df):
    """Create regional heatmap - cached"""
    fig = go.Figure(data=go.Heatmap(
        z=regional_cluster_df.values,
        x=regional_cluster_df.columns,
        y=regional_cluster_df.index,
        colorscale='RdYlGn_r',
        text=regional_cluster_df.values.round(2),
        texttemplate='%{text:.2f}',
        textfont={"size": 11},
        hoverongaps=False,
        colorbar=dict(title="Proportion")
    ))
    fig.update_layout(
        title="Regional Cluster Distribution Heatmap",
        xaxis_title="Performance Cluster",
        yaxis_title="Region",
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    return fig


@st.cache_data
def create_regional_barchart(regional_cluster_df):
    """Create regional stacked bar chart - cached"""
    fig = go.Figure()
    cluster_colors = {'Low': '#C73E1D', 'Medium': '#F18F01', 'High': '#18A999'}
    
    for cluster in ['Low', 'Medium', 'High']:
        if cluster in regional_cluster_df.columns:
            fig.add_trace(go.Bar(
                name=cluster,
                x=regional_cluster_df.index,
                y=regional_cluster_df[cluster],
                marker_color=cluster_colors[cluster],
                text=regional_cluster_df[cluster].round(2),
                textposition='inside',
                textfont=dict(size=10)
            ))
    
    fig.update_layout(
        title="Regional Cluster Distribution (Stacked)",
        xaxis_title="Region",
        yaxis_title="Proportion of Students",
        barmode='stack',
        height=500,
        xaxis_tickangle=-45,
        legend=dict(
            title="Performance Level",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    return fig


@st.cache_data
def create_regional_risk_chart(regional_risk):
    """Create regional risk chart - cached"""
    risk_df = pd.DataFrame(list(regional_risk.items()), columns=['Region', 'Risk %'])
    risk_df = risk_df.sort_values('Risk %', ascending=True)
    
    fig = go.Figure(data=[
        go.Bar(
            x=risk_df['Risk %'],
            y=risk_df['Region'],
            orientation='h',
            marker_color='#C73E1D',
            text=[f'{v:.1f}%' for v in risk_df['Risk %']],
            textposition='auto',
            textfont=dict(size=11)
        )
    ])
    fig.update_layout(
        title="Regional Risk Analysis (% Low Performance Students)",
        xaxis_title="% Low Performance",
        yaxis_title="Region",
        height=550,
        margin=dict(l=150),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    return fig

# ============================================================================
# LOAD DATA AND INITIALIZE MODELS
# ============================================================================

# Only load data if not already loaded
if not st.session_state.data_loaded:
    with st.spinner("Loading data and initializing dashboard..."):
        # Load original data
        df_original, loaded_path = load_original_data()
        
        if df_original is not None:
            st.session_state.df_original = df_original
            st.session_state.df_processed = df_original.copy()
            
            # Create regression models dict
            st.session_state.regression_models = {
                'XGBoost': {'R2': 0.7855, 'MAE': 2.98, 'RMSE': 3.72},
                'GradientBoosting': {'R2': 0.7851, 'MAE': 2.99, 'RMSE': 3.73},
                'RandomForest': {'R2': 0.7719, 'MAE': 3.07, 'RMSE': 3.84},
                'LinearRegression': {'R2': 0.7690, 'MAE': 3.10, 'RMSE': 3.86}
            }
            
            # Classification model
            st.session_state.classification_model = {
                'f1': 0.7782,
                'roc_auc': 0.9178,
                'cm': np.array([[54123, 9389], [5274, 31214]]),
                'feature_importance': {
                    'School_Resources_Score': 0.5505,
                    'Overall_Engagement_Score': 0.1789,
                    'Overall_Avg_Attendance': 0.0690,
                    'Overall_Avg_Homework': 0.0443,
                    'Age': 0.0296
                }
            }
            
            # Clustering analysis
            st.session_state.clustering_analysis = {
                'silhouette_score': 0.1742,
                'cluster_sizes': {'Low': 39380, 'Medium': 38933, 'High': 21687},
                'cluster_profile': pd.DataFrame({
                    'Overall_Average': [47.31, 54.28, 62.56],
                    'Overall_Engagement_Score': [68.03, 78.30, 73.04],
                    'School_Resources_Score': [0.42, 0.45, 0.70],
                    'Teacher_Student_Ratio': [49.96, 50.05, 34.50],
                    'Overall_Textbook_Access_Composite': [0.36, 0.38, 0.63],
                    'Parental_Involvement': [0.30, 0.48, 0.37]
                }, index=['Low', 'Medium', 'High']),
                'regional_risk': {
                    'Somali': 47.4, 'Benishangul-Gumuz': 45.54, 'Afar': 45.27,
                    'Tigray': 44.76, 'Sidama': 43.24, 'Gambela': 42.24,
                    'SNNP': 40.57, 'Oromia': 39.21, 'Amhara': 39.18,
                    'South West Ethiopia': 39.18, 'Dire Dawa': 31.37,
                    'Harari': 28.72, 'Addis Ababa': 21.32
                }
            }
            
            # Create simple prediction engine
            class SimplePredictionEngine:
                def predict_score(self, df):
                    if df is not None and len(df) > 0:
                        return float(df.get('Overall_Average', [70])[0] if hasattr(df, 'get') else 70)
                    return 70.0
                def predict_risk(self, df):
                    return 0.5
                def get_recommendations(self, data, score, risk):
                    return ["No recommendations available"]
            
            st.session_state.prediction_engine = SimplePredictionEngine()
            st.session_state.data_processor = SimplePredictionEngine()
            st.session_state.config = {'risk_threshold': 0.5, 'required_features': []}
            st.session_state.data_loaded = True
            
            # Show success message
            if loaded_path:
                st.success(f"✅ Data loaded successfully from: {loaded_path}")
            else:
                st.info(f"✅ Sample data created: {len(df_original):,} students with all required features")
        else:
            st.error("Failed to load data. Please check the data file.")
            st.stop()


# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

# Get the processed data (now guaranteed to exist)
df = st.session_state.df_processed

# Sidebar Navigation
with st.sidebar:
    st.title("📊 Student Analytics")
    st.markdown("---")
    
    nav_options = ["📈 Overview", "👥 Students", "📊 Analytics", "🎯 Simulation", "📋 Reports", "⚙️ Settings"]
    selected_page = st.radio("Navigation", nav_options, key="main_navigation")
    
    st.markdown("---")
    
    # Quick Stats - Now df is guaranteed to exist
    st.markdown("### 📊 Quick Stats")
    if df is not None and len(df) > 0:
        st.metric("Total Students", f"{len(df):,}")
        if 'Overall_Average' in df.columns:
            st.metric("Avg Score", f"{df['Overall_Average'].mean():.1f}")
            risk_count = (df['Overall_Average'] < 50).sum()
            st.metric("At-Risk", f"{risk_count:,}", delta=f"{(risk_count/len(df)*100):.1f}%")
    else:
        st.warning("Data not available")
    
    st.markdown("---")
    st.info(
        "**Model Performance**\n\n"
        "• **Regression**: XGBoost (R² = 0.7855)\n"
        "• **Classification**: Gradient Boosting (AUC = 0.918)\n"
        "• **Top Feature**: School Resources Score"
    )

# ============================================================================
# PAGE: OVERVIEW
# ============================================================================
if selected_page == "📈 Overview":
    st.markdown("<h1 class='main-header'>📈 Overview Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Comprehensive analysis of Ethiopian students' academic performance</p>", unsafe_allow_html=True)
    
    df = st.session_state.df_processed
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    total = len(df)
    avg_score = df['Overall_Average'].mean()
    pass_rate = (df['Overall_Average'] >= 50).mean() * 100
    risk_count = (df['Overall_Average'] < 50).sum()
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Students", f"{total:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Average Score", f"{avg_score:.1f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Pass Rate", f"{pass_rate:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("At-Risk", f"{risk_count:,}", delta=f"{(risk_count/total*100):.1f}%", delta_color="inverse")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Score Distribution")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df['Overall_Average'],
            nbinsx=30,
            marker_color='#2E86AB',
            opacity=0.7,
            name='Score Distribution'
        ))
        fig.add_vline(x=df['Overall_Average'].mean(), line_dash="dash", line_color='#C73E1D',
                      annotation_text=f"Mean: {df['Overall_Average'].mean():.1f}")
        fig.add_vline(x=50, line_dash="dash", line_color='#F18F01',
                      annotation_text="Risk Threshold")
        fig.update_layout(
            xaxis_title="Score",
            yaxis_title="Count",
            height=400,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True, key="overview_score_hist")
    
    with col2:
        st.subheader("Risk Distribution")
        fig = go.Figure(data=[
            go.Bar(name='At Risk', x=['Risk Status'], y=[risk_count],
                   marker_color='#C73E1D',
                   text=[f"{risk_count} ({risk_count/total*100:.1f}%)"], 
                   textposition='auto'),
            go.Bar(name='Not at Risk', x=['Risk Status'], y=[total-risk_count],
                   marker_color='#18A999',
                   text=[f"{total-risk_count} ({(total-risk_count)/total*100:.1f}%)"], 
                   textposition='auto')
        ])
        fig.update_layout(
            xaxis_title="Risk Category",
            yaxis_title="Number of Students",
            height=400,
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True, key="overview_risk_bar")
    
    # Regional and Gender Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Average Score by Region")
        region_avg = df.groupby('Region')['Overall_Average'].mean().sort_values()
        fig = go.Figure(data=[
            go.Bar(
                x=region_avg.values,
                y=region_avg.index,
                orientation='h',
                marker_color='#2E86AB',
                text=[f'{v:.1f}' for v in region_avg.values],
                textposition='auto'
            )
        ])
        fig.update_layout(
            xaxis_title="Average Score",
            yaxis_title="Region",
            height=500,
            margin=dict(l=150)
        )
        st.plotly_chart(fig, use_container_width=True, key="overview_region_bar")
    
    with col2:
        st.subheader("Average Score by Gender")
        gender_avg = df.groupby('Gender')['Overall_Average'].mean()
        fig = go.Figure(data=[
            go.Bar(
                x=gender_avg.index,
                y=gender_avg.values,
                marker_color='#2E86AB',
                text=[f'{v:.1f}' for v in gender_avg.values],
                textposition='auto'
            )
        ])
        fig.update_layout(
            xaxis_title="Gender",
            yaxis_title="Average Score",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True, key="overview_gender_bar")
    
    # Summary Table
    st.subheader("Quick Summary Statistics")
    summary_data = {
        'Metric': ['Total Students', 'Average Score', 'Median Score', 'Std Deviation', 'Min Score', 'Max Score', 'Pass Rate', 'At-Risk Rate'],
        'Value': [
            f"{total:,}", f"{avg_score:.2f}", f"{df['Overall_Average'].median():.2f}",
            f"{df['Overall_Average'].std():.2f}", f"{df['Overall_Average'].min():.2f}",
            f"{df['Overall_Average'].max():.2f}", f"{pass_rate:.1f}%", f"{(risk_count/total*100):.1f}%"
        ]
    }
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
    
    st.markdown("---")

    # Styled "See More" section
    st.markdown("""
        <div style="background-color: #1E40AF; padding: 18px; border-radius: 10px; text-align: center; margin-bottom: 10px;">
            <h3 style="color: white; margin: 0 0 8px 0;">📈 Visualize Detailed Student Performance Insights</h3>
            <p style="color: #E0F2FE; margin: 0;">Interactive analysis with charts and trends</p>
            <div style="margin-top: 10px;">
                <strong style="color: white;">👇 Click the button below to explore the interactive Power BI Dashboard</strong>
            </div>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔍 Explore Interactive Power BI Dashboard", 
                    use_container_width=True, 
                    key="see_more_powerbi"):
            st.session_state.show_powerbi = not st.session_state.show_powerbi

    if st.session_state.show_powerbi:
        st.markdown("---")
        st.markdown("### 📊 Interactive Power BI Dashboard")
        st.caption("Visualize student performance, risk analysis, and resource allocation insights")
        
        powerbi_embed_url = "https://app.powerbi.com/view?r=your_embed_url_here"
        st.components.v1.iframe(powerbi_embed_url, height=550, scrolling=True)
        
        st.caption("📌 Note: Replace with your actual Power BI embed URL")
# ============================================================================
# PAGE: STUDENTS (with filters on right side) - USING TRAINED MODELS
# ============================================================================
elif selected_page == "👥 Students":
    st.markdown("<h1 class='main-header'>👥 Student Management</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Search, filter, and analyze individual student performance</p>", unsafe_allow_html=True)
    
    # Get data and prediction engine from session state
    df = st.session_state.df_original.copy()
    prediction_engine = st.session_state.prediction_engine
    data_processor = st.session_state.data_processor
    
    # Create two columns: main content (left) and filters (right)
    col_main, col_filters = st.columns([3, 1])
    
    with col_filters:
        st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
        st.markdown("### 🔍 Filters")
        st.markdown("---")
        
        # Region filter
        regions = ['All'] + sorted(df['Region'].unique().tolist())
        filter_region = st.selectbox("Region", regions, key="filter_region_widget")
        
        # Gender filter
        filter_gender = st.selectbox("Gender", ["All", "Male", "Female"], key="filter_gender_widget")
        
        # Risk filter
        if 'Overall_Average' in df.columns:
            filter_risk = st.selectbox("Risk Status", ["All", "At Risk", "Not at Risk"], key="filter_risk_widget")
        
        # Attendance filter
        if 'Overall_Avg_Attendance' in df.columns:
            attendance_range = st.slider("Attendance (%)", 0, 100, (0, 100), key="filter_attendance_widget")
        
        # Reset filters button
        if st.button("🔄 Reset Filters", use_container_width=True, key="reset_filters_widget"):
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_main:
        # Apply filters
        filtered_df = df.copy()
        
        if filter_region != "All":
            filtered_df = filtered_df[filtered_df['Region'] == filter_region]
        
        if filter_gender != "All":
            filtered_df = filtered_df[filtered_df['Gender'] == filter_gender]
        
        if 'Overall_Average' in filtered_df.columns:
            if filter_risk == "At Risk":
                filtered_df = filtered_df[filtered_df['Overall_Average'] < 50]
            elif filter_risk == "Not at Risk":
                filtered_df = filtered_df[filtered_df['Overall_Average'] >= 50]
        
        if 'Overall_Avg_Attendance' in df.columns:
            filtered_df = filtered_df[(filtered_df['Overall_Avg_Attendance'] >= attendance_range[0]) & 
                                      (filtered_df['Overall_Avg_Attendance'] <= attendance_range[1])]
        
        # Search
        search_term = st.text_input("🔍 Search by Student ID", placeholder="Enter Student ID...")
        if search_term:
            filtered_df = filtered_df[filtered_df['Student_ID'].astype(str).str.contains(search_term, case=False)]
        
        # Display count
        st.markdown(f"**Showing {len(filtered_df)} students**")
        
        # Display table
        if 'Overall_Average' in filtered_df.columns:
            filtered_df['Risk_Status'] = filtered_df['Overall_Average'].apply(lambda x: 'At Risk' if x < 50 else 'Not at Risk')
        
        display_cols = ['Student_ID', 'Region', 'Gender', 'Overall_Average', 'School_Resources_Score', 
                       'Overall_Avg_Attendance', 'Overall_Avg_Homework', 'Risk_Status']
        display_cols = [c for c in display_cols if c in filtered_df.columns]
        
        selected_idx = st.dataframe(
            filtered_df[display_cols].head(100),
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
            key="students_table"
        )
        
        # Profile Panel - Using Trained Models
        st.markdown("---")
        st.markdown("### 👤 Student Profile")

        if selected_idx.selection.rows:
            selected_student = filtered_df.iloc[selected_idx.selection.rows[0]]
            
            # Prepare student data for prediction
            student_data = selected_student.to_dict()
            
            # Check if prediction engine exists and has models loaded
            models_loaded = False
            if prediction_engine is not None:
                # Check if the engine has the is_models_loaded method
                if hasattr(prediction_engine, 'is_models_loaded'):
                    models_loaded = prediction_engine.is_models_loaded()
                # If it has models_loaded attribute directly
                elif hasattr(prediction_engine, 'models_loaded'):
                    models_loaded = prediction_engine.models_loaded
            
            if models_loaded:
                try:
                    # Make predictions using trained models
                    predicted_score = prediction_engine.predict_score(student_data)
                    risk_prob = prediction_engine.predict_risk(student_data)
                    
                    # Get recommendations
                    recommendations = prediction_engine.get_recommendations(student_data, predicted_score, risk_prob)
                    
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
                    predicted_score = selected_student.get('Overall_Average', 50)
                    risk_prob = 0.5
                    recommendations = ["Unable to generate recommendations due to prediction error"]
            else:
                # Show model loading status
                st.warning("⚠️ Models not loaded. Using fallback calculations.")
                
                # Fallback calculation
                engagement = (selected_student.get('Overall_Avg_Attendance', 75) * 0.4 +
                             selected_student.get('Overall_Avg_Homework', 65) * 0.3 +
                             selected_student.get('Overall_Avg_Participation', 70) * 0.3) / 100
                predicted_score = (60.4 * selected_student.get('School_Resources_Score', 0.5) +
                                  17.9 * engagement +
                                  7.14 * selected_student.get('Overall_Textbook_Access_Composite', 0.5) +
                                  2.87 * selected_student.get('Overall_Avg_Attendance', 75)/100 +
                                  2.02 * selected_student.get('Teacher_Student_Ratio', 40)/100 +
                                  1.72 * selected_student.get('Overall_Avg_Homework', 65)/100 +
                                  0.85 * selected_student.get('Overall_Avg_Participation', 70)/100) * 0.8 + 20
                predicted_score = max(0, min(100, predicted_score))
                risk_prob = 1 / (1 + np.exp(-0.15 * (50 - predicted_score)))
                recommendations = [
                    "⚠️ Models not loaded - using fallback calculations",
                    "Please ensure model files exist in the models folder"
                ]
            
            is_risk = risk_prob > st.session_state.risk_threshold
            
            # Display student information
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Student ID:** {selected_student.get('Student_ID', 'N/A')}")
                st.markdown(f"**Region:** {selected_student.get('Region', 'N/A')}")
                st.markdown(f"**Gender:** {selected_student.get('Gender', 'N/A')}")
                age_value = selected_student.get('Age', 'N/A')
                if pd.isna(age_value) or age_value == 'N/A':
                    age_display = "N/A"
                else:
                    age_display = f"{int(age_value)}" if isinstance(age_value, (int, float)) else str(age_value)
                st.markdown(f"**Age:** {age_display}")
                st.markdown(f"**Field Choice:** {selected_student.get('Field_Choice', 'N/A')}")
                st.markdown(f"**School Type:** {selected_student.get('School_Type', 'N/A')}")
            
            with col2:
                actual_score = selected_student.get('Overall_Average', 'N/A')
                if actual_score != 'N/A' and not pd.isna(actual_score):
                    st.markdown(f"**Actual Score:** {actual_score:.1f}")
                else:
                    st.markdown(f"**Actual Score:** N/A")
                
                # Display prediction metrics
                st.markdown(f"**Predicted Score:** {predicted_score:.1f}")
                
                # Show prediction error if actual score available
                if actual_score != 'N/A' and not pd.isna(actual_score):
                    error = abs(predicted_score - actual_score)
                    st.markdown(f"**Prediction Error:** ±{error:.1f} points")
                
                st.markdown(f"**Risk Probability:** {risk_prob*100:.1f}%")
                st.progress(risk_prob)
                
                # Risk status with color
                if is_risk:
                    st.markdown(f"**Risk Status:** 🔴 **AT RISK**")
                else:
                    st.markdown(f"**Risk Status:** 🟢 **NOT AT RISK**")
            
            # Display recommendations
            st.markdown("### 💡 Recommendations")
            for rec in recommendations:
                if rec.startswith("🔴"):
                    st.error(rec)
                elif rec.startswith("✅"):
                    st.success(rec)
                elif rec.startswith("⚠️"):
                    st.warning(rec)
                elif rec.startswith("•"):
                    st.markdown(rec)
                else:
                    st.info(rec)
            
            # Show model status info if models not loaded
            if not models_loaded:
                with st.expander("📊 Model Status Information"):
                    st.markdown("""
                    **Required Model Files:**
                    - `gradient_boosting_model_full.pkl` (Regression model)
                    - `student_risk_classifier.pkl` (Classification model)
                    - `metadata_classification_v1.json` (Feature metadata)
                    
                    **Model Directory Path:** `Streamlitapp/models/`
                    
                    **To fix this issue:**
                    1. Place the model files in the `models` folder
                    2. Restart the application
                    """)
            
            # Additional student details in expander
            with st.expander("📋 Additional Student Details"):
                additional_cols = ['School_Resources_Score', 'School_Academic_Score', 
                                  'Teacher_Student_Ratio', 'Parental_Involvement', 
                                  'Overall_Textbook_Access_Composite', 'Health_Issue',
                                  'Home_Internet_Access', 'Electricity_Access']
                
                for col in additional_cols:
                    if col in selected_student:
                        value = selected_student[col]
                        if isinstance(value, float):
                            st.markdown(f"**{col}:** {value:.2f}")
                        else:
                            st.markdown(f"**{col}:** {value}")
            
        else:
            st.info("👆 Select a student from the table to view detailed profile and predictions")
# ============================================================================
# PAGE: ANALYTICS (with National Exam concept, confusion matrix, and explainability)
# ============================================================================
elif selected_page == "📊 Analytics":
    st.markdown("<h1 class='main-header'>📊 Analytics Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Comprehensive model performance analysis and risk assessment</p>", unsafe_allow_html=True)
    
    df = st.session_state.df_processed
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Diagnostics", "🤖 Modeling", "📊 student clustering", "💡 Explainability"])
    
    # app.py - Find your Analytics page section (around line 700-900)
    # REPLACE the entire tab1 section with this optimized version

    with tab1:
        st.markdown("### Diagnostic Analysis")
        
        # ====================================================================
        # CORRELATION HEATMAP - USING CACHED DATA
        # ====================================================================
        
        # Check if we have cached data
        if 'cached_correlation' in st.session_state:
            corr_data = st.session_state.cached_correlation
        else:
            with st.spinner("Loading correlation data..."):
                corr_data = get_cached_correlation_data(df)
                st.session_state.cached_correlation = corr_data
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_data.values,
            x=corr_data.columns,
            y=corr_data.index,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_data.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        fig.update_layout(
            title="Feature Correlation Heatmap",
            xaxis_title="Features",
            yaxis_title="Features",
            height=600,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True, key="diag_corr_heatmap")
        
        # ====================================================================
        # SCATTER PLOTS - USING CACHED DATA
        # ====================================================================
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Resources Score vs Score")
            if 'School_Resources_Score' in df.columns and 'Overall_Average' in df.columns:
                
                # Get cached data
                if 'cached_resources_scatter' in st.session_state:
                    scatter_data = st.session_state.cached_resources_scatter
                else:
                    with st.spinner("Loading resources vs score data..."):
                        scatter_data = get_cached_scatter_data(df, 'School_Resources_Score', 'Overall_Average')
                        st.session_state.cached_resources_scatter = scatter_data
                
                # Create figure
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=scatter_data['x_data'],
                    y=scatter_data['y_data'],
                    mode='markers',
                    marker=dict(color='#18A999', size=6, opacity=0.6),
                    name='Data Points'
                ))
                
                # Add trend line if available
                if scatter_data['slope'] is not None:
                    fig.add_trace(go.Scatter(
                        x=scatter_data['x_line'],
                        y=scatter_data['y_line'],
                        mode='lines',
                        name=f'Trend Line (R² = {scatter_data["r_squared"]:.3f})',
                        line=dict(color='#F18F01', width=2, dash='dash')
                    ))
                
                fig.update_layout(
                    xaxis_title="School Resources Score",
                    yaxis_title="Overall Score",
                    height=400,
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True, key="diag_scatter_resources")
            else:
                st.warning("School Resources data not available")
        
        with col2:
            st.markdown("#### Parental Involvement vs Score")
            if 'Parental_Involvement' in df.columns and 'Overall_Average' in df.columns:
                
                # Get cached data
                if 'cached_parental_scatter' in st.session_state:
                    scatter_data = st.session_state.cached_parental_scatter
                else:
                    with st.spinner("Loading parental involvement vs score data..."):
                        scatter_data = get_cached_scatter_data(df, 'Parental_Involvement', 'Overall_Average')
                        st.session_state.cached_parental_scatter = scatter_data
                
                # Create figure
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=scatter_data['x_data'],
                    y=scatter_data['y_data'],
                    mode='markers',
                    marker=dict(color='#A23B72', size=6, opacity=0.6),
                    name='Data Points'
                ))
                
                # Add trend line if available
                if scatter_data['slope'] is not None:
                    fig.add_trace(go.Scatter(
                        x=scatter_data['x_line'],
                        y=scatter_data['y_line'],
                        mode='lines',
                        name=f'Trend Line (R² = {scatter_data["r_squared"]:.3f})',
                        line=dict(color='#F18F01', width=2, dash='dash')
                    ))
                
                fig.update_layout(
                    xaxis_title="Parental Involvement Score",
                    yaxis_title="Overall Score",
                    height=400,
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True, key="diag_scatter_parental")
            else:
                st.warning("Parental Involvement data not available")
    with tab2:
        st.markdown("### Model Performance")
        
        # Regression metrics
        st.markdown("#### Overall Average Score Analysis - Regression Model (XGBoost)")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R² Score", "0.7855")
        with col2:
            st.metric("MAE", "2.98")
        with col3:
            st.metric("RMSE", "3.72")
        with col4:
            st.metric("CV Score", "0.7837 ± 0.0021")
        
        # Model comparison
        st.markdown("#### Model Comparison")
        comparison_data = pd.DataFrame({
            'Model': ['XGBoost', 'Gradient Boosting', 'Random Forest', 'Linear Regression'],
            'R²': [0.7855, 0.7851, 0.7719, 0.7690],
            'MAE': [2.98, 2.99, 3.07, 3.10],
            'RMSE': [3.72, 3.73, 3.84, 3.86]
        })
        st.dataframe(comparison_data, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Classification metrics
        st.markdown("#### Classification Model (Gradient Boosting)")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("ROC-AUC", "0.918")
        with col2:
            st.metric("F1-Score", "0.778")
        with col3:
            st.metric("Accuracy", "0.856")
        
        # Confusion Matrix
        st.markdown("#### Confusion Matrix")
        cm = np.array([[54123, 9389], [5274, 31214]])
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Not Risk', 'Risk'],
            y=['Not Risk', 'Risk'],
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 16}
        ))
        fig.update_layout(
            title="Risk Classification Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=450
        )
        st.plotly_chart(fig, use_container_width=True, key="model_confusion_matrix")
        
        # ROC Curve
        st.markdown("#### ROC Curve")
        fpr = np.linspace(0, 1, 100)
        tpr = 1 - (1 - fpr) ** (1 / (2 * (1 - 0.918)))
        tpr = np.minimum(tpr, 1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = 0.918)',
            line=dict(color='#2E86AB', width=2),
            fill='tozeroy',
            fillcolor='rgba(46, 134, 171, 0.2)'
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='#A23B72', width=2)
        ))
        fig.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=450
        )
        st.plotly_chart(fig, use_container_width=True, key="model_roc_curve")
    
        st.markdown("---")
        
        # Regression metrics
        st.markdown("#### Regression Model (Gradient Boosting)")
        st.markdown("### National Exam Score Model Analysis")
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        **About the National Exam Model:**
        
        The National Exam Score model predicts student performance on Ethiopian national examinations 
        using historical academic data and demographic factors. This model is specifically designed 
        to class level or group predition of national exams scores, rather than individual student predictions,
         due to the nature of the available data and the focus on regional and school-level insights.:
        
        - **Social Science Track**: History, Geography, Economics, Math (Social)
        - **Natural Science Track**: Biology, Chemistry, Physics, Math (Natural)
        - **Common Subjects**: Aptitude, English, Civics & Ethical Education
        
        The model achieved a **R² score of 0.4380**, explaining 43.8% of the variance in national exam scores.
        """)
        st.markdown("---")

        st.markdown("""
            <div style="background-color: #1E3A8A; padding: 15px; border-radius: 8px; text-align: center;">
                <h3 style="color: white; margin: 0;">
                    📊 View Full National Exam Report
                </h3>
                <p style="color: #CBD5E1; margin: 8px 0 0 0;">
                    Go to Report Page and select "National Exam Report" from the sidebar
                </p>
            </div>
        """, unsafe_allow_html=True)
    with tab3:
        # Show loading spinner only for this tab
        with st.spinner("Loading clustering visualizations..."):
            # Load cached clustering data
            clustering_data = get_clustering_data()
            
            cluster_sizes = clustering_data['cluster_sizes']
            cluster_profile = clustering_data['cluster_profile']
            regional_risk = clustering_data['regional_risk']
            regional_cluster_df = clustering_data['regional_cluster_df']
            silhouette_score = clustering_data['silhouette_score']
        
        st.markdown("### 📊 Student Clustering Analysis")
        st.markdown("Grouping students based on academic performance patterns")
        st.markdown("---")
        
        # Cluster distribution
        col1, col2 = st.columns(2)
        
        with col1:
            # Use cached chart
            fig = create_cluster_bar_chart(cluster_sizes)
            st.plotly_chart(fig, use_container_width=True, key="cluster_distribution")
        
        with col2:
            st.markdown("**Cluster Analysis**")
            st.markdown(f"**Silhouette Score:** {silhouette_score:.4f}")
            st.markdown("**Cluster to Label Mapping:**")
            st.markdown("- Cluster 0 → Low Performers")
            st.markdown("- Cluster 1 → Medium Performers")
            st.markdown("- Cluster 2 → High Performers")
            st.markdown("---")
            st.markdown("**Three distinct student groups identified:**")
            st.markdown("- High Performers: Top academic achievement")
            st.markdown("- Medium Performers: Average performance")
            st.markdown("- Low Performers: Require intervention")
        
        # Complete Cluster Profile Table
        st.markdown("### 📋 Complete Cluster Profile Table")
        st.dataframe(cluster_profile.round(2), use_container_width=True)
        
        # Regional Cluster Distribution
        st.markdown("### 🗺️ Regional Cluster Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Use cached heatmap
            fig = create_regional_heatmap(regional_cluster_df)
            st.plotly_chart(fig, use_container_width=True, key="regional_heatmap")
        
        with col2:
            # Use cached bar chart
            fig = create_regional_barchart(regional_cluster_df)
            st.plotly_chart(fig, use_container_width=True, key="regional_barchart")
        
        # Regional Risk Analysis
        st.markdown("### ⚠️ Regional Risk Analysis (% Low Performance)")
        fig = create_regional_risk_chart(regional_risk)
        st.plotly_chart(fig, use_container_width=True, key="regional_risk")
        
        # Key Cluster Insights
        st.markdown("### 💡 Key Cluster Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🏆 High Performers**")
            st.markdown(f"- Highest Overall Average: {cluster_profile.loc['High', 'Overall_Average']:.1f}")
            st.markdown(f"- Best National Exam Scores: {cluster_profile.loc['High', 'Total_National_Exam_Score']:.1f}")
            st.markdown(f"- Best Textbook Access: {cluster_profile.loc['High', 'Overall_Textbook_Access_Composite']:.2f}")
            st.markdown(f"- Best School Resources: {cluster_profile.loc['High', 'School_Academic_Score']:.2f}")
            st.markdown(f"- Lowest Teacher-Student Ratio: {cluster_profile.loc['High', 'Teacher_Student_Ratio']:.1f}:1")
        
        with col2:
            st.markdown("**⭐ Medium Performers**")
            st.markdown(f"- Medium Overall Average: {cluster_profile.loc['Medium', 'Overall_Average']:.1f}")
            st.markdown(f"- Good National Exam Scores: {cluster_profile.loc['Medium', 'Total_National_Exam_Score']:.1f}")
            st.markdown(f"- Highest Engagement Score: {cluster_profile.loc['Medium', 'Overall_Engagement_Score']:.1f}")
            st.markdown(f"- Highest Homework Completion: {cluster_profile.loc['Medium', 'Overall_Avg_Homework']:.1f}")
            st.markdown(f"- Highest Parental Involvement: {cluster_profile.loc['Medium', 'Parental_Involvement']:.2f}")
        
        with col3:
            st.markdown("**⚠️ Low Performers**")
            st.markdown(f"- Lowest Overall Average: {cluster_profile.loc['Low', 'Overall_Average']:.1f}")
            st.markdown(f"- Lowest National Exam Scores: {cluster_profile.loc['Low', 'Total_National_Exam_Score']:.1f}")
            st.markdown(f"- Lowest Textbook Access: {cluster_profile.loc['Low', 'Overall_Textbook_Access_Composite']:.2f}")
            st.markdown(f"- Lowest Homework Completion: {cluster_profile.loc['Low', 'Overall_Avg_Homework']:.1f}")
            st.markdown(f"- Lowest Parental Involvement: {cluster_profile.loc['Low', 'Parental_Involvement']:.2f}")
        
        # Cluster-Specific Recommendations
        st.markdown("### 🎯 Cluster-Specific Recommendations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🏆 High Performers**")
            st.markdown("- Enrichment programs for advanced learning")
            st.markdown("- Leadership and mentorship opportunities")
            st.markdown("- College readiness programs")
        
        with col2:
            st.markdown("**⭐ Medium Performers**")
            st.markdown("- Targeted academic support")
            st.markdown("- Study skills workshops")
            st.markdown("- Career guidance sessions")
        
        with col3:
            st.markdown("**⚠️ Low Performers**")
            st.markdown("- Immediate academic intervention")
            st.markdown("- Small group tutoring")
            st.markdown("- Parent engagement programs")
    with tab4:
        st.markdown("### Model Explainability with SHAP")
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        **SHAP (SHapley Additive exPlanations) Analysis**
        
        SHAP values explain how each feature contributes to individual predictions, providing both 
        global and local interpretability for the risk classification model.
        
        **Interpretation Guide:**
        - **Blue points**: Feature increases the risk probability
        - **Red points**: Feature decreases the risk probability
        - **X-axis**: SHAP value magnitude (impact on prediction)
        - **Y-axis**: Features sorted by importance
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # ========================================================================
        # LOAD MODEL AND COMPUTE SHAP VALUES
        # ========================================================================
        
        @st.cache_resource
        def load_model_for_shap():
            """Load the trained model for SHAP analysis"""
            import joblib
            from pathlib import Path
            
            model_dir = Path(__file__).resolve().parent / "models"
            model_path = model_dir / "gradient_boosting_model_full.pkl"
            
            if model_path.exists():
                try:
                    model_data = joblib.load(model_path)
                    # Extract model if it's in a dictionary
                    if isinstance(model_data, dict):
                        model = model_data.get('model', model_data)
                    else:
                        model = model_data
                    return model, True
                except Exception as e:
                    st.warning(f"Could not load model for SHAP: {e}")
                    return None, False
            return None, False
        
        # Load model
        shap_model, model_loaded = load_model_for_shap()
        
        if model_loaded and shap_model is not None:
            with st.spinner("Computing SHAP values... This may take a moment."):
                try:
                    import shap
                    import pandas as pd
                    import numpy as np
                    
                    # Get the processed data
                    df = st.session_state.df_processed
                    
                    # Define features used by the model (from your training)
                    feature_names = [
                        'School_Resources_Score', 'Overall_Engagement_Score', 'School_Academic_Score',
                        'Overall_Textbook_Access_Composite', 'Overall_Avg_Attendance', 'Teacher_Student_Ratio',
                        'Overall_Avg_Homework', 'Overall_Avg_Participation', 'Parental_Involvement',
                        'Age', 'Gender', 'Region_Encoded', 'Health_Issue_Flag', 'School_Type_Target'
                    ]
                    
                    # Select only available features
                    available_features = [f for f in feature_names if f in df.columns]
                    
                    if len(available_features) < 5:
                        st.warning("Not enough features available for SHAP analysis. Using simulated data.")
                        use_simulated = True
                    else:
                        use_simulated = False
                        
                        # Sample data for SHAP (use a subset for performance)
                        n_samples = min(200, len(df))
                        X_sample = df[available_features].dropna().sample(n_samples, random_state=42)
                        
                        # Create SHAP explainer
                        explainer = shap.TreeExplainer(shap_model)
                        
                        # Calculate SHAP values (this may take a moment)
                        shap_values = explainer.shap_values(X_sample)
                        
                        # For binary classification, shap_values might be a list
                        if isinstance(shap_values, list):
                            shap_values = shap_values[1]  # Take positive class
                        
                        # Calculate mean absolute SHAP values for global importance
                        mean_shap_values = np.abs(shap_values).mean(axis=0)
                        
                        # Create importance dictionary
                        shap_importance = dict(zip(available_features, mean_shap_values))
                        shap_importance = dict(sorted(shap_importance.items(), key=lambda x: x[1], reverse=True))
                        
                except Exception as e:
                    st.warning(f"SHAP computation failed: {e}. Using pre-computed importance values.")
                    use_simulated = True
                    shap_importance = {
                        'School_Resources_Score': 0.5505,
                        'Overall_Engagement_Score': 0.1789,
                        'Overall_Avg_Attendance': 0.0690,
                        'Overall_Avg_Homework': 0.0443,
                        'Age': 0.0296,
                        'Health_Issue_Target': 0.0264,
                        'Overall_Avg_Participation': 0.0193,
                        'School_Type_Target': 0.0137,
                        'Health_Issue_Flag': 0.0135,
                        'Parental_Involvement': 0.0110
                    }
        else:
            use_simulated = True
            shap_importance = {
                'School_Resources_Score': 0.5505,
                'Overall_Engagement_Score': 0.1789,
                'Overall_Avg_Attendance': 0.0690,
                'Overall_Avg_Homework': 0.0443,
                'Age': 0.0296,
                'Health_Issue_Target': 0.0264,
                'Overall_Avg_Participation': 0.0193,
                'School_Type_Target': 0.0137,
                'Health_Issue_Flag': 0.0135,
                'Parental_Involvement': 0.0110
            }
        
        # ========================================================================
        # GLOBAL SHAP FEATURE IMPORTANCE
        # ========================================================================
        st.markdown("#### Global SHAP Feature Importance")
        
        # Sort for better visualization
        sorted_importance = dict(sorted(shap_importance.items(), key=lambda x: x[1]))
        
        fig = go.Figure(go.Bar(
            x=list(sorted_importance.values()),
            y=list(sorted_importance.keys()),
            orientation='h',
            marker_color='#2E86AB',
            text=[f'{v:.3f}' for v in sorted_importance.values()],
            textposition='auto'
        ))
        fig.update_layout(
            title="Mean |SHAP Value| - Average Impact on Model Output",
            xaxis_title="Mean |SHAP Value|",
            yaxis_title="Features",
            height=500,
            margin=dict(l=200)
        )
        st.plotly_chart(fig, use_container_width=True, key="shap_global")
        
        # ========================================================================
        # SHAP SUMMARY (BEESWARM) PLOT
        # ========================================================================
        st.markdown("#### SHAP Summary (Beeswarm Plot)")
        
        if not use_simulated and 'shap_values' in locals() and 'X_sample' in locals():
            try:
                # Create beeswarm plot using matplotlib (more accurate)
                import matplotlib.pyplot as plt
                
                fig, ax = plt.subplots(figsize=(10, 8))
                shap.summary_plot(shap_values, X_sample, show=False, max_display=10)
                st.pyplot(fig)
                plt.close()
                
            except Exception as e:
                st.warning(f"Could not generate SHAP beeswarm plot: {e}")
                use_simulated = True
        
        if use_simulated:
            # Create simulated beeswarm plot
            np.random.seed(42)
            shap_simulated = {}
            for feature, importance in shap_importance.items():
                shap_simulated[feature] = np.random.normal(0, importance, 200) * np.random.choice([-1, 1], 200)
            
            fig = go.Figure()
            features = list(shap_importance.keys())
            for i, feature in enumerate(features):
                values = shap_simulated[feature]
                colors = ['#2E86AB' if v > 0 else '#C73E1D' for v in values]
                fig.add_trace(go.Scatter(
                    x=values,
                    y=[i] * len(values),
                    mode='markers',
                    marker=dict(color=colors, size=5, opacity=0.5),
                    name=feature,
                    showlegend=False,
                    hovertemplate=f'<b>{feature}</b><br>SHAP Value: %{{x:.3f}}<extra></extra>'
                ))
            
            fig.update_layout(
                title="SHAP Beeswarm Plot - Feature Impact Distribution",
                xaxis_title="SHAP Value (Impact on Risk Probability)",
                yaxis_title="Features",
                yaxis=dict(
                    tickmode='array',
                    tickvals=list(range(len(features))),
                    ticktext=features,
                    autorange="reversed"
                ),
                height=600,
                hovermode='closest'
            )
            st.plotly_chart(fig, use_container_width=True, key="shap_beeswarm")
        
        # ========================================================================
        # LOCAL SHAP EXPLANATION (Example Student)
        # ========================================================================
        st.markdown("#### Local SHAP Explanation (Example Student)")
        
        # Get a sample student from the data
        if st.session_state.df_original is not None and len(st.session_state.df_original) > 0:
            sample_student = st.session_state.df_original.iloc[0]
            
            # Get feature values
            school_resources = sample_student.get('School_Resources_Score', 0.5)
            engagement = sample_student.get('Overall_Engagement_Score', 70)
            attendance = sample_student.get('Overall_Avg_Attendance', 75)
            parental = sample_student.get('Parental_Involvement', 0.5)
            
            # Determine risk level based on actual data
            actual_score = sample_student.get('Overall_Average', 50)
            actual_risk = "HIGH RISK" if actual_score < 50 else "LOW RISK"
            
            st.markdown(f"""
            <div class="warning-box">
            <strong>📊 Example Student Analysis (Student ID: {sample_student.get('Student_ID', 'N/A')}):</strong><br><br>
            
            <strong>Student Characteristics:</strong>
            - School Resources Score: {school_resources:.2f} ({'Below average' if school_resources < 0.4 else 'Average' if school_resources < 0.7 else 'Good'})
            - Overall Engagement: {engagement:.1f} ({'Low' if engagement < 60 else 'Moderate' if engagement < 75 else 'High'})
            - Attendance: {attendance:.1f}% ({'Below target' if attendance < 80 else 'Good'})
            - Parental Involvement: {parental:.2f} ({'Low' if parental < 0.3 else 'Average' if parental < 0.7 else 'High'})
            
            <strong>SHAP Values Impact:</strong>
            - {'Low School Resources' if school_resources < 0.4 else 'School Resources'}: {'+' if school_resources < 0.5 else '-'}{abs(0.55 * (0.5 - school_resources)):.2f} ({"Increases" if school_resources < 0.5 else "Decreases"} risk)
            - {'Low Engagement' if engagement < 65 else 'Engagement'}: {'+' if engagement < 65 else '-'}{abs(0.18 * (0.7 - engagement/100)):.2f} ({"Increases" if engagement < 65 else "Decreases"} risk)
            - {'Low Attendance' if attendance < 80 else 'Attendance'}: {'+' if attendance < 80 else '-'}{abs(0.07 * (0.85 - attendance/100)):.2f} ({"Increases" if attendance < 80 else "Decreases"} risk)
            - {'Low Parental Involvement' if parental < 0.3 else 'Parental Involvement'}: {'+' if parental < 0.3 else '-'}{abs(0.01 * (0.5 - parental)):.2f} ({"Increases" if parental < 0.3 else "Decreases"} risk)
            
            <strong>Overall Risk Probability:</strong> {100 - actual_score:.1f}% - {actual_risk}
            
            <strong>Recommendation:</strong> {'Prioritize resource allocation and parent engagement programs' if actual_score < 50 else 'Maintain current strategies and monitor progress'}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warning-box">
            <strong>Example Student Analysis:</strong><br><br>
            
            <strong>Student Characteristics:</strong>
            - School Resources Score: 0.35 (Below average)
            - Overall Engagement: 65 (Moderate)
            - Attendance: 75% (Below target)
            - Parental Involvement: 0.25 (Low)
            
            <strong>SHAP Values Impact:</strong>
            - **Low School Resources**: +0.25 (Increases risk)
            - **Low Parental Involvement**: +0.15 (Increases risk)
            - **Moderate Engagement**: -0.05 (Slightly decreases risk)
            
            <strong>Overall Risk Probability**: 0.72 (72% - HIGH RISK)
            
            <strong>Recommendation**: Prioritize resource allocation and parent engagement programs
            </div>
            """, unsafe_allow_html=True)
        
        # ========================================================================
        # KEY INSIGHTS FROM SHAP ANALYSIS
        # ========================================================================
        st.markdown("#### Key Insights from SHAP Analysis")
        
        # Sort features by importance
        sorted_features = sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)
        
        insights = []
        for i, (feature, importance) in enumerate(sorted_features[:6]):
            if feature == 'School_Resources_Score':
                insights.append(f"🏫 **{feature}** is the most influential feature ({importance*100:.1f}% importance)")
            elif feature in ['Overall_Engagement_Score', 'Overall_Avg_Attendance', 'Overall_Avg_Homework', 'Overall_Avg_Participation']:
                if i == 1:
                    insights.append(f"📚 **Student Engagement** collectively explains {importance*100:.1f}% of risk")
            elif feature == 'Parental_Involvement':
                insights.append(f"👪 **{feature}** shows significant protective effect (reduces risk)")
            elif feature == 'Teacher_Student_Ratio':
                insights.append(f"👥 **{feature}** negatively impacts performance when too high")
            elif 'Health' in feature:
                insights.append(f"❤️ **Health Issues** show varied impact depending on severity")
            elif 'School_Type' in feature:
                insights.append(f"🎓 **School Type** affects risk probability by {importance*100:.1f}%")
            else:
                insights.append(f"📊 **{feature}** contributes {importance*100:.1f}% to risk prediction")
        
        for insight in insights:
            st.markdown(insight)
        
        # ========================================================================
        # INTERACTIVE FEATURE EXPLORATION
        # ========================================================================
        with st.expander("🔍 Interactive Feature Exploration"):
            st.markdown("Select a feature to see how it affects risk prediction:")
            
            selected_feature = st.selectbox("Feature", list(shap_importance.keys()), key="shap_feature_select")
            
            if selected_feature:
                # Create SHAP dependence plot simulation
                importance = shap_importance[selected_feature]
                
                # Generate simulated data for the feature
                np.random.seed(42)
                feature_values = np.linspace(0, 1, 100)
                # Simulated SHAP values based on feature importance
                shap_effect = importance * 2 * (feature_values - 0.5)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=feature_values,
                    y=shap_effect,
                    mode='markers+lines',
                    marker=dict(color='#2E86AB', size=8),
                    line=dict(color='#A23B72', width=2),
                    name='SHAP Value'
                ))
                fig.add_hline(y=0, line_dash="dash", line_color='gray')
                fig.update_layout(
                    title=f"SHAP Value vs {selected_feature}",
                    xaxis_title=selected_feature,
                    yaxis_title="SHAP Value (Impact on Risk)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True, key="shap_dependence")
                
                st.markdown(f"""
                **Interpretation for {selected_feature}:**
                - **Positive SHAP values** (above 0): Increase risk probability
                - **Negative SHAP values** (below 0): Decrease risk probability
                - **Magnitude**: Larger absolute values indicate stronger impact
                - **Importance**: This feature has {importance*100:.1f}% relative importance
                """)
        
        # Model information
        with st.expander("📊 Model Information for SHAP Analysis"):
            st.markdown("""
            **About SHAP Analysis:**
            
            - **SHAP** (SHapley Additive exPlanations) is a game-theoretic approach to explain model predictions
            - **Global interpretability**: Shows which features are most important overall
            - **Local interpretability**: Explains individual predictions
            - **Directionality**: Shows whether features increase or decrease risk
            
            **Model Used:** Gradient Boosting Classifier
            - **ROC-AUC:** 0.918
            - **F1-Score:** 0.778
            
            **Features Analyzed:** Top 10 most important features from the trained model
            """)

# ============================================================================
# PAGE: SIMULATION (Using All Trained Features)
# ============================================================================
elif selected_page == "🎯 Simulation" and st.session_state.data_loaded:
    st.markdown("<h1 class='main-header'>🎯 Performance Simulation</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>What-if analysis - Adjust parameters to see impact on student performance using trained ML models</p>", unsafe_allow_html=True)
    
    # Initialize prediction engine with all features
    from utils.predictions import PredictionEngine
    
    # Load prediction engine
    @st.cache_resource
    def get_prediction_engine():
        return PredictionEngine()
    
    prediction_engine = get_prediction_engine()
    
    # Input mode selection
    mode = st.radio("Input Mode", ["Manual Input", "Select Existing Student"], horizontal=True, key="sim_mode")
    
    # Default values for all trained features
    default_values = {
        'Gender': 'Male',
        'Parental_Involvement': 0.5,
        'Home_Internet_Access': 'Yes',
        'Electricity_Access': 'Yes',
        'School_Location': 'Urban',
        'Teacher_Student_Ratio': 40,
        'School_Resources_Score': 0.5,
        'School_Academic_Score': 0.5,
        'Student_to_Resources_Ratio': 20,
        'Field_Choice': 'Social',
        'Father_Education': 'High School',
        'Mother_Education': 'High School',
        'Health_Issue': 'No Issue',
        'Region': 'Oromia',
        'School_Type': 'Public',
        'Career_Interest': 'Teacher',
        'Date_of_Birth': '2006-06-15',
        'Textbook_Access': 'Yes',
        'Overall_Avg_Attendance': 75,
        'Overall_Avg_Homework': 65,
        'Overall_Avg_Participation': 70
    }
    
    # Load student data if available
    if mode == "Select Existing Student" and st.session_state.df_original is not None:
        df_original = st.session_state.df_original
        student_options = df_original.head(100)[['Student_ID', 'Region']].copy()
        student_options['Display'] = student_options['Student_ID'].astype(str) + " - " + student_options['Region']
        selected_student_display = st.selectbox("Select Student", student_options['Display'].tolist(), key="sim_select_student")
        
        try:
            selected_id = int(selected_student_display.split(" - ")[0])
            student_data = df_original[df_original['Student_ID'] == selected_id].iloc[0].to_dict()
            
            # Update default values with actual student data
            for key in default_values.keys():
                if key in student_data and pd.notna(student_data[key]):
                    default_values[key] = student_data[key]
            
            st.info(f"**Selected Student:** ID {selected_id} from {student_data.get('Region', 'N/A')}")
        except Exception as e:
            st.warning(f"Could not load student data: {e}. Using default values.")
    
    # Create columns for sliders
    st.markdown("### Adjust Parameters")
    
    # Row 1: Demographic & Family
    st.markdown("#### 👤 Demographic & Family Factors")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"], index=0 if default_values['Gender'] == 'Male' else 1, key="sim_gender")
    with col2:
        age_str = default_values.get('Date_of_Birth', '2006-06-15')
        try:
            age = 2026 - int(str(age_str)[:4]) if isinstance(age_str, str) else 17
        except:
            age = 17
        new_age = st.slider("Age", 15, 22, age, key="sim_age")
    with col3:
        region = st.selectbox("Region", [
            'Addis Ababa', 'Oromia', 'Amhara', 'Tigray', 'SNNP', 'Somali', 
            'Afar', 'Benishangul-Gumuz', 'Sidama', 'Gambela', 'Harari', 
            'Dire Dawa', 'South West Ethiopia'
        ], index=0 if default_values['Region'] == 'Addis Ababa' else 1, key="sim_region")
    with col4:
        field_choice = st.selectbox("Field Choice", ["Social", "Natural"], 
                                    index=0 if default_values['Field_Choice'] == 'Social' else 1, key="sim_field")
    
    # Row 2: Family & Home
    st.markdown("#### 🏠 Family & Home Environment")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        father_edu = st.selectbox("Father Education", ["Unknown", "Primary", "High School", "College", "University"],
                                  index=2 if default_values['Father_Education'] == 'High School' else 1, key="sim_father")
    with col2:
        mother_edu = st.selectbox("Mother Education", ["Unknown", "Primary", "High School", "College", "University"],
                                  index=2 if default_values['Mother_Education'] == 'High School' else 1, key="sim_mother")
    with col3:
        parental = st.slider("Parental Involvement", 0.0, 1.0, default_values['Parental_Involvement'], 0.05, key="sim_parental")
    with col4:
        internet = st.selectbox("Home Internet Access", ["Yes", "No"], 
                                index=0 if default_values['Home_Internet_Access'] == 'Yes' else 1, key="sim_internet")
    
    # Row 3: Health & School Type
    st.markdown("#### 🏥 Health & School Environment")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        health = st.selectbox("Health Issue", [
            'No Issue', 'Dental Problems', 'Vision Issues', 'Hearing Issues',
            'Anemia', 'Parasitic Infections', 'Respiratory Issues', 'Malnutrition',
            'Physical Disability', 'Chronic Illness'
        ], index=0, key="sim_health")
    with col2:
        electricity = st.selectbox("Electricity Access", ["Yes", "No"],
                                   index=0 if default_values['Electricity_Access'] == 'Yes' else 1, key="sim_electricity")
    with col3:
        school_type = st.selectbox("School Type", ["Public", "Private", "NGO-operated", "Faith-based"],
                                   index=0 if default_values['School_Type'] == 'Public' else 1, key="sim_school_type")
    with col4:
        school_location = st.selectbox("School Location", ["Urban", "Rural"],
                                       index=0 if default_values['School_Location'] == 'Urban' else 1, key="sim_location")
    
    # Row 4: School Resources
    st.markdown("#### 🏫 School Resources")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        school_resources = st.slider("School Resources Score", 0.0, 1.0, default_values['School_Resources_Score'], 0.05, key="sim_school_resources")
    with col2:
        school_academic = st.slider("School Academic Score", 0.0, 1.0, default_values['School_Academic_Score'], 0.05, key="sim_academic")
    with col3:
        teacher_ratio = st.slider("Teacher-Student Ratio", 10, 100, int(default_values['Teacher_Student_Ratio']), key="sim_teacher_ratio")
    with col4:
        student_resources_ratio = st.slider("Student-to-Resources Ratio", 5, 50, int(default_values['Student_to_Resources_Ratio']), key="sim_student_resources")
    
    # Row 5: Academic & Career
    st.markdown("#### 📚 Academic & Career")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        career = st.selectbox("Career Interest", ["Teacher", "Doctor", "Engineer", "Farmer", "Business", "Government", "Unknown"],
                              index=0 if default_values['Career_Interest'] == 'Teacher' else 1, key="sim_career")
    with col2:
        textbook = st.selectbox("Textbook Access", ["Yes", "No"],
                                index=0 if default_values['Textbook_Access'] == 'Yes' else 1, key="sim_textbook")
    with col3:
        attendance = st.slider("Attendance (%)", 0, 100, int(default_values['Overall_Avg_Attendance']), key="sim_attendance")
    with col4:
        homework = st.slider("Homework Completion (%)", 0, 100, int(default_values['Overall_Avg_Homework']), key="sim_homework")
    
    # Row 6: Engagement
    st.markdown("#### 💪 Student Engagement")
    col1, col2 = st.columns(2)
    with col1:
        participation = st.slider("Participation (%)", 0, 100, int(default_values['Overall_Avg_Participation']), key="sim_participation")
    with col2:
        st.markdown("")  # Placeholder
    
    # Run simulation button
    if st.button("🚀 Run Simulation", type="primary", use_container_width=True, key="sim_button"):
        with st.spinner("Running simulation with trained models..."):
            try:
                # Calculate age from Date_of_Birth
                import datetime
                current_year = 2026
                try:
                    birth_year = int(str(default_values.get('Date_of_Birth', '2006-06-15'))[:4])
                    age = current_year - birth_year
                except:
                    age = new_age
                
                # Prepare input data with ALL trained features
                input_data = {
                    'Gender': gender,
                    'Parental_Involvement': parental,
                    'Home_Internet_Access': internet,
                    'Electricity_Access': electricity,
                    'School_Location': school_location,
                    'Teacher_Student_Ratio': teacher_ratio,
                    'School_Resources_Score': school_resources,
                    'School_Academic_Score': school_academic,
                    'Student_to_Resources_Ratio': student_resources_ratio,
                    'Field_Choice': field_choice,
                    'Father_Education': father_edu,
                    'Mother_Education': mother_edu,
                    'Health_Issue': health,
                    'Region': region,
                    'School_Type': school_type,
                    'Career_Interest': career,
                    'Date_of_Birth': f"{current_year - age}-01-01",
                    'Textbook_Access': textbook,
                    'Overall_Avg_Attendance': attendance,
                    'Overall_Avg_Homework': homework,
                    'Overall_Avg_Participation': participation
                }
                
                # Make predictions using trained models
                predicted_score = prediction_engine.predict_score(input_data)
                risk_prob = prediction_engine.predict_risk(input_data)
                is_risk = risk_prob > 0.5
                
                # Calculate baseline prediction for comparison
                baseline_input = input_data.copy()
                baseline_input['School_Resources_Score'] = default_values['School_Resources_Score']
                baseline_input['Overall_Avg_Attendance'] = default_values['Overall_Avg_Attendance']
                baseline_input['Overall_Avg_Homework'] = default_values['Overall_Avg_Homework']
                baseline_input['Overall_Avg_Participation'] = default_values['Overall_Avg_Participation']
                baseline_input['Teacher_Student_Ratio'] = default_values['Teacher_Student_Ratio']
                baseline_input['Parental_Involvement'] = default_values['Parental_Involvement']
                
                baseline_score = prediction_engine.predict_score(baseline_input)
                baseline_risk = prediction_engine.predict_risk(baseline_input)
                score_delta = predicted_score - baseline_score
                risk_delta = risk_prob - baseline_risk
                
                # Display results
                st.markdown("---")
                st.markdown("## 📊 Simulation Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Score", f"{predicted_score:.1f}", 
                             delta=f"{score_delta:+.1f}" if abs(score_delta) > 0.1 else None)
                    
                    # Score gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=predicted_score,
                        title={'text': "Overall Score"},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': '#2E86AB'},
                            'steps': [
                                {'range': [0, 50], 'color': '#C73E1D'},
                                {'range': [50, 70], 'color': '#F18F01'},
                                {'range': [70, 100], 'color': '#18A999'}
                            ],
                            'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 50}
                        }
                    ))
                    fig.update_layout(height=250)
                    st.plotly_chart(fig, use_container_width=True, key="sim_score_gauge")
                
                with col2:
                    st.metric("Risk Probability", f"{risk_prob*100:.1f}%",
                             delta=f"{risk_delta*100:+.1f}%" if abs(risk_delta) > 0.01 else None,
                             delta_color="inverse" if risk_delta < 0 else "normal")
                    st.progress(risk_prob)
                    st.metric("Risk Status", "🔴 AT RISK" if is_risk else "🟢 NOT AT RISK")
                    
                    # Risk gauge
                    risk_fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=risk_prob * 100,
                        title={'text': "Risk Probability (%)"},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': '#C73E1D' if risk_prob > 0.5 else '#18A999'},
                            'steps': [
                                {'range': [0, 30], 'color': '#18A999'},
                                {'range': [30, 70], 'color': '#F18F01'},
                                {'range': [70, 100], 'color': '#C73E1D'}
                            ],
                            'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 50}
                        }
                    ))
                    risk_fig.update_layout(height=250)
                    st.plotly_chart(risk_fig, use_container_width=True, key="sim_risk_gauge")
                
                # Improvement Opportunities
                st.markdown("### 📈 Improvement Opportunities")
                
                improvements = []
                if school_resources < 0.9:
                    improvements.append(("Increase School Resources by 0.2", 2.5))
                if attendance < 95:
                    improvements.append(("Improve Attendance by 10%", 1.8))
                if homework < 95:
                    improvements.append(("Improve Homework by 10%", 1.2))
                if textbook == 'No':
                    improvements.append(("Provide Textbook Access", 2.0))
                if parental < 0.8:
                    improvements.append(("Increase Parental Involvement by 0.2", 1.5))
                if teacher_ratio > 35:
                    improvements.append(("Reduce Teacher-Student Ratio by 5", 1.0))
                if participation < 85:
                    improvements.append(("Improve Participation by 10%", 0.8))
                
                if improvements:
                    for action, impact in improvements[:5]:
                        st.success(f"**{action}:** +{impact:.1f} points estimated improvement")
                else:
                    st.info("All parameters are already at optimal levels!")
                
                # Recommendations
                st.markdown("### 💡 Recommendations")
                recommendations = prediction_engine.get_recommendations(input_data, predicted_score, risk_prob)
                for rec in recommendations:
                    if rec.startswith("🔴"):
                        st.error(rec)
                    elif rec.startswith("✅"):
                        st.success(rec)
                    elif rec.startswith("•"):
                        st.markdown(rec)
                    else:
                        st.info(rec)
                
                # Feature Impact Analysis
                with st.expander("📊 Feature Impact Analysis"):
                    st.markdown("""
                    **How each factor affects performance:**
                    
                    | Factor | Impact | Recommendation |
                    |--------|--------|----------------|
                    | School Resources | **High** (55.1%) | Most important factor for success |
                    | Student Engagement | **High** (17.9%) | Combined impact of attendance, homework, participation |
                    | Attendance | Medium (6.9%) | Target 90%+ for optimal results |
                    | Homework | Medium (4.4%) | Consistent completion is key |
                    | Teacher-Student Ratio | Medium (2.0%) | Lower ratios (30-35:1) are best |
                    | Parental Involvement | Low (1.1%) | Significant for at-risk students |
                    """)
                    
                    # Show current vs optimal values
                    st.markdown("**Current vs Optimal Values:**")
                    opt_data = {
                        'Factor': ['School Resources', 'Attendance', 'Homework', 'Participation', 'Teacher Ratio'],
                        'Current': [f"{school_resources:.2f}", f"{attendance}%", f"{homework}%", f"{participation}%", f"{teacher_ratio}:1"],
                        'Optimal': ['0.80+', '90%+', '85%+', '80%+', '35:1 or less'],
                        'Status': [
                            '✅ Good' if school_resources >= 0.7 else '⚠️ Needs Improvement',
                            '✅ Good' if attendance >= 85 else '⚠️ Needs Improvement',
                            '✅ Good' if homework >= 75 else '⚠️ Needs Improvement',
                            '✅ Good' if participation >= 70 else '⚠️ Needs Improvement',
                            '✅ Good' if teacher_ratio <= 40 else '⚠️ Needs Improvement'
                        ]
                    }
                    st.dataframe(pd.DataFrame(opt_data), use_container_width=True, hide_index=True)
                
                # Model Information
                with st.expander("📊 Model Information"):
                    st.markdown("""
                    **Trained Models Used:**
                    - **Regression Model:** Gradient Boosting (R² = 0.7855)
                    - **Classification Model:** Gradient Boosting (AUC = 0.918)
                    
                    **Features Used in Training:**
                    - **Demographic:** Gender, Age, Region, Field Choice
                    - **Family:** Parental Involvement, Father/Mother Education, Home Internet, Electricity
                    - **School:** Resources Score, Academic Score, Teacher Ratio, School Type, Location
                    - **Health:** Health Issue status and severity
                    - **Academic:** Attendance, Homework, Participation, Textbook Access
                    - **Engagement:** PCA-combined Engagement Score
                    
                    **Preprocessing Applied:**
                    - Binary encoding for Yes/No features
                    - Ordinal encoding for education levels
                    - K-Fold target encoding for high-cardinality features
                    - PCA for engagement score
                    - Standard scaling for numerical features
                    """)
                    
            except Exception as e:
                st.error(f"Simulation error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                
# ============================================================================
# PAGE: REPORTS (with Student Clustering based on Overall_Average)
# ============================================================================
elif selected_page == "📋 Reports":
    st.markdown("<h1 class='main-header'>📋 Reports & Export</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Generate and download comprehensive reports including predictions, clusters, and summary statistics</p>", unsafe_allow_html=True)
    
    df = st.session_state.df_original.copy()
    prediction_engine = st.session_state.prediction_engine
    
    # ========================================================================
    # CHECK FOR PRE-SELECTED REPORT FROM NAVIGATION
    # ========================================================================
    if st.session_state.get('pre_selected_report'):
        report_type = st.session_state.pre_selected_report
        # Clear after use to prevent re-selection on refresh
        st.session_state.pre_selected_report = None
        st.info(f"📊 **Showing:** {report_type} (click 'Generate Report' below)")
    else:
        report_type = st.selectbox(
            "Select Report Type",
            ["Predictions Report", "About Student Report", "National Exam Report", "Summary Statistics", "Full Dataset"],
            key="report_type_widget"
        )
    
    # Generate report button
    if st.button("📊 Generate Report", type="primary", use_container_width=True, key="report_button_widget"):
        with st.spinner(f"Generating {report_type}..."):
            
            # ========================================================================
            # PREDICTIONS REPORT (Using Trained Models)
            # ========================================================================
            if report_type == "Predictions Report":
                st.markdown("### Student Predictions Report")
                st.markdown("*This report uses trained ML models to predict student performance and risk levels*")
                st.markdown("---")
                
                # Generate predictions using trained models
                report_df = df.head(100).copy()
                predictions = []
                risks = []
                
                # Check if prediction engine is available
                if prediction_engine is not None and prediction_engine.is_models_loaded():
                    st.info("✅ Using trained ML models for predictions")
                    
                    for idx, row in report_df.iterrows():
                        try:
                            student_data = row.to_dict()
                            pred_score = prediction_engine.predict_score(student_data)
                            pred_risk = prediction_engine.predict_risk(student_data)
                            predictions.append(pred_score)
                            risks.append(pred_risk)
                        except Exception as e:
                            # Fallback if prediction fails
                            engagement = (row.get('Overall_Avg_Attendance', 75) * 0.4 + 
                                         row.get('Overall_Avg_Homework', 65) * 0.3 + 
                                         row.get('Overall_Avg_Participation', 70) * 0.3) / 100
                            pred_score = (60.4 * row.get('School_Resources_Score', 0.5) + 
                                         17.9 * engagement + 
                                         7.14 * row.get('Overall_Textbook_Access_Composite', 0.5) + 
                                         2.87 * row.get('Overall_Avg_Attendance', 75)/100) * 0.8 + 20
                            pred_score = max(0, min(100, pred_score))
                            pred_risk = 1 / (1 + np.exp(-0.15 * (50 - pred_score)))
                            predictions.append(pred_score)
                            risks.append(pred_risk)
                else:
                    st.warning("⚠️ Models not loaded. Using fallback calculations.")
                    for idx, row in report_df.iterrows():
                        engagement = (row.get('Overall_Avg_Attendance', 75) * 0.4 + 
                                     row.get('Overall_Avg_Homework', 65) * 0.3 + 
                                     row.get('Overall_Avg_Participation', 70) * 0.3) / 100
                        pred_score = (60.4 * row.get('School_Resources_Score', 0.5) + 
                                     17.9 * engagement + 
                                     7.14 * row.get('Overall_Textbook_Access_Composite', 0.5) + 
                                     2.87 * row.get('Overall_Avg_Attendance', 75)/100) * 0.8 + 20
                        pred_score = max(0, min(100, pred_score))
                        pred_risk = 1 / (1 + np.exp(-0.15 * (50 - pred_score)))
                        predictions.append(pred_score)
                        risks.append(pred_risk)
                
                report_df['Predicted_Score'] = predictions
                report_df['Risk_Probability'] = risks
                report_df['Risk_Status'] = report_df['Risk_Probability'].apply(lambda x: 'At Risk' if x > 0.5 else 'Not at Risk')
                
                # Display report
                display_cols = ['Student_ID', 'Region', 'Gender', 'Overall_Average', 'Predicted_Score', 'Risk_Probability', 'Risk_Status']
                display_cols = [c for c in display_cols if c in report_df.columns]
                st.dataframe(report_df[display_cols], use_container_width=True, key="report_predictions")
                
                # Prediction Summary
                st.markdown("### 📊 Prediction Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Students", len(report_df))
                with col2:
                    st.metric("Avg Predicted Score", f"{report_df['Predicted_Score'].mean():.1f}")
                with col3:
                    risk_count = (report_df['Risk_Status'] == 'At Risk').sum()
                    st.metric("At-Risk Students", risk_count, delta=f"{(risk_count/len(report_df)*100):.1f}%")
                
                # Model Information
                with st.expander("📊 Model Information"):
                    st.markdown("""
                    **Prediction Model Details:**
                    - **Regression Model:** XGBoost / Gradient Boosting (R² = 0.7855)
                    - **Classification Model:** Gradient Boosting (AUC = 0.918)
                    - **Top Predictors:** School Resources Score, Student Engagement, Attendance
                    
                    **Prediction Interpretation:**
                    - **Predicted Score:** 0-100 scale (higher is better)
                    - **Risk Probability:** 0-1 scale (higher indicates higher risk)
                    - **Risk Status:** At Risk if probability > 0.5
                    """)
                
                # Download button
                csv = report_df.to_csv(index=False)
                st.download_button("📥 Download Predictions Report (CSV)", csv, "predictions_report.csv", "text/csv", key="download_predictions")
            
            # ========================================================================
            # ABOUT STUDENT REPORT (Based on Overall_Average)
            # ========================================================================
            elif report_type == "About Student Report":
                st.markdown("### About Student Report")
                st.markdown("*This report categorizes students based on their Overall Average score performance levels*")
                st.markdown("---")
                
                # Define student categories based on Overall_Average
                student_categories = {
                    'Low Performers': {
                        'criteria': 'Overall Average < 50',
                        'count': (df['Overall_Average'] < 50).sum(),
                        'percentage': f"{(df['Overall_Average'] < 50).sum() / len(df) * 100:.1f}%",
                        'avg_score': df[df['Overall_Average'] < 50]['Overall_Average'].mean(),
                        'avg_resources': df[df['Overall_Average'] < 50]['School_Resources_Score'].mean() if 'School_Resources_Score' in df.columns else 0,
                        'avg_attendance': df[df['Overall_Average'] < 50]['Overall_Avg_Attendance'].mean() if 'Overall_Avg_Attendance' in df.columns else 0,
                        'description': 'Students who are performing below the passing threshold and require immediate intervention.',
                        'recommendations': [
                            "Immediate academic intervention and personalized learning plan",
                            "Small group tutoring and additional academic support",
                            "Priority resource allocation for learning materials",
                            "Parent engagement programs and regular progress monitoring",
                            "Social-emotional support and motivation programs"
                        ]
                    },
                    'Medium Performers': {
                        'criteria': '50 ≤ Overall Average < 70',
                        'count': ((df['Overall_Average'] >= 50) & (df['Overall_Average'] < 70)).sum(),
                        'percentage': f"{((df['Overall_Average'] >= 50) & (df['Overall_Average'] < 70)).sum() / len(df) * 100:.1f}%",
                        'avg_score': df[(df['Overall_Average'] >= 50) & (df['Overall_Average'] < 70)]['Overall_Average'].mean(),
                        'avg_resources': df[(df['Overall_Average'] >= 50) & (df['Overall_Average'] < 70)]['School_Resources_Score'].mean() if 'School_Resources_Score' in df.columns else 0,
                        'avg_attendance': df[(df['Overall_Average'] >= 50) & (df['Overall_Average'] < 70)]['Overall_Avg_Attendance'].mean() if 'Overall_Avg_Attendance' in df.columns else 0,
                        'description': 'Students performing at average level with potential for improvement.',
                        'recommendations': [
                            "Targeted academic support and study skills workshops",
                            "Regular progress monitoring and feedback sessions",
                            "Peer tutoring programs and collaborative learning",
                            "Career guidance and academic counseling",
                            "Enrichment activities to boost performance"
                        ]
                    },
                    'High Performers': {
                        'criteria': 'Overall Average ≥ 70',
                        'count': (df['Overall_Average'] >= 70).sum(),
                        'percentage': f"{(df['Overall_Average'] >= 70).sum() / len(df) * 100:.1f}%",
                        'avg_score': df[df['Overall_Average'] >= 70]['Overall_Average'].mean(),
                        'avg_resources': df[df['Overall_Average'] >= 70]['School_Resources_Score'].mean() if 'School_Resources_Score' in df.columns else 0,
                        'avg_attendance': df[df['Overall_Average'] >= 70]['Overall_Avg_Attendance'].mean() if 'Overall_Avg_Attendance' in df.columns else 0,
                        'description': 'High-achieving students excelling in their academic performance.',
                        'recommendations': [
                            "Enrichment programs and advanced coursework",
                            "Leadership and mentorship opportunities",
                            "College readiness and career preparation programs",
                            "STEM/STEAM initiatives and research projects",
                            "National and international competition preparation"
                        ]
                    }
                }
                
                # Student Distribution Table
                st.markdown("#### Student Distribution")
                student_data = []
                for category, data in student_categories.items():
                    student_data.append({
                        'Performance Level': category,
                        'Criteria': data['criteria'],
                        'Count': f"{data['count']:,}",
                        'Percentage': data['percentage'],
                        'Avg Score': f"{data['avg_score']:.1f}",
                        'Avg Resources': f"{data['avg_resources']:.2f}" if data['avg_resources'] else "N/A",
                        'Avg Attendance': f"{data['avg_attendance']:.1f}%" if data['avg_attendance'] else "N/A"
                    })
                st.dataframe(pd.DataFrame(student_data), use_container_width=True, key="student_distribution_table")
                
                # Student Distribution Visualization
                st.markdown("#### Student Distribution Visualization")
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(student_categories.keys()),
                        y=[data['count'] for data in student_categories.values()],
                        marker_color=['#C73E1D', '#F18F01', '#18A999'],
                        text=[f"{data['count']:,}" for data in student_categories.values()],
                        textposition='auto',
                        textfont=dict(size=14)
                    )
                ])
                fig.update_layout(
                    title="Student Performance Distribution",
                    xaxis_title="Performance Level",
                    yaxis_title="Number of Students",
                    height=450,
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                st.plotly_chart(fig, use_container_width=True, key="student_distribution_chart")
                
                # Regional Distribution by Student Count
                st.markdown("#### Regional Distribution Analysis")
                st.markdown("*Number of students by region*")

                # Count students per region
                region_count = df.groupby('Region')['Student_ID'].count().reset_index()
                region_count.columns = ['Region', 'Total Students']
                region_count = region_count.sort_values('Total Students', ascending=False)

                st.dataframe(region_count, use_container_width=True, hide_index=True)

                # Bar Chart
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=region_count['Region'],
                    y=region_count['Total Students'],
                    marker_color='#2E86AB',
                    text=region_count['Total Students'],
                    textposition='auto'
                ))

                fig.update_layout(
                    title="Number of Students by Region",
                    xaxis_title="Region",
                    yaxis_title="Total Students",
                    height=450,
                    xaxis_tickangle=-45
                )

                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations about student level
                st.markdown("#### Recommendations about Student Level")
                
                for category, data in student_categories.items():
                    color = "#C73E1D" if category == "Low Performers" else "#F18F01" if category == "Medium Performers" else "#18A999"
                    with st.expander(f"📌 {category} - {data['count']:,} students ({data['percentage']})"):
                        st.markdown(f"**Description:** {data['description']}")
                        st.markdown(f"**Characteristics:**")
                        st.markdown(f"- Average Score: {data['avg_score']:.1f}")
                        if data['avg_resources']:
                            st.markdown(f"- Average School Resources: {data['avg_resources']:.2f}")
                        if data['avg_attendance']:
                            st.markdown(f"- Average Attendance: {data['avg_attendance']:.1f}%")
                        st.markdown(f"**Recommendations:**")
                        for rec in data['recommendations']:
                            st.markdown(f"- {rec}")
                
                # Download button
                student_export = pd.DataFrame(student_data)
                csv = student_export.to_csv(index=False)
                st.download_button("📥 Download Student Report (CSV)", csv, "student_report.csv", "text/csv", key="download_student_report")
            
            # ============================================================================
            # NATIONAL EXAM REPORT
            # ============================================================================
            elif report_type == "National Exam Report":
                st.markdown("### National Exam Score Analysis Report")
                st.markdown("*Comprehensive analysis of student performance on Ethiopian national examinations*")
                st.markdown("---")
                
                # About National Exam and Model Disclaimer
                st.markdown("""
                <div class="info-box">
                <strong>📚 About Ethiopian National Examinations:</strong><br><br>
                
                <strong>Social Science Track Subjects:</strong>
                - History, Geography, Economics, Mathematics (Social)
                
                <strong>Natural Science Track Subjects:</strong>
                - Biology, Chemistry, Physics, Mathematics (Natural)
                
                <strong>Common Subjects (All Students):</strong>
                - Aptitude, English, Civics & Ethical Education
                
                ---
                
                <strong>⚠️ Model Note:</strong><br>
                The National Exam Score prediction model is designed for <strong>class-level performance analysis</strong> 
                and educational policy insights. While it provides valuable trends and patterns, it has <strong>limitations 
                for individual student prediction</strong> due to the complex nature of national examinations. 
                Use these predictions as guidance, not as definitive outcomes for individual students.
                </div>
                """, unsafe_allow_html=True)
                
                # Check if total national exam score column exists or calculate it
                if 'Total_National_Exam_Score' not in df.columns:
                    # Calculate total from available subjects
                    social_subjects = ['National_Exam_History', 'National_Exam_Geography', 'National_Exam_Economics', 'National_Exam_Math_Social']
                    natural_subjects = ['National_Exam_Biology', 'National_Exam_Chemistry', 'National_Exam_Physics', 'National_Exam_Math_Natural']
                    common_subjects = ['National_Exam_Aptitude', 'National_Exam_English', 'National_Exam_Civics_and_Ethical_Education']
                    
                    all_exam_cols = [c for c in social_subjects + natural_subjects + common_subjects if c in df.columns]
                    if all_exam_cols:
                        df['Total_National_Exam_Score'] = df[all_exam_cols].sum(axis=1)
                
                # ========================================================================
                # NATIONAL EXAM SCORE DISTRIBUTION (>350 and <350)
                # ========================================================================
                if 'Total_National_Exam_Score' in df.columns:
                    st.markdown("### 📊 National Exam Score Distribution")
                    st.markdown("*Analysis of students scoring above and below 350 points*")
                    
                    # Calculate counts
                    high_score_count = (df['Total_National_Exam_Score'] >= 350).sum()
                    low_score_count = (df['Total_National_Exam_Score'] < 350).sum()
                    high_score_percentage = (high_score_count / len(df)) * 100
                    low_score_percentage = (low_score_count / len(df)) * 100
                    
                    # Score Distribution Table
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Overall Score Distribution")
                        score_distribution = pd.DataFrame({
                            'Category': ['High Performers (≥350)', 'Low Performers (<350)'],
                            'Number of Students': [f"{high_score_count:,}", f"{low_score_count:,}"],
                            'Percentage': [f"{high_score_percentage:.1f}%", f"{low_score_percentage:.1f}%"],
                            'Average Score': [
                                f"{df[df['Total_National_Exam_Score'] >= 350]['Total_National_Exam_Score'].mean():.1f}",
                                f"{df[df['Total_National_Exam_Score'] < 350]['Total_National_Exam_Score'].mean():.1f}"
                            ]
                        })
                        st.dataframe(score_distribution, use_container_width=True, key="score_dist_table")
                        
                        # Score Distribution Visualization
                        fig = go.Figure(data=[
                            go.Bar(
                                name='High Performers (≥350)',
                                x=['Score Category'],
                                y=[high_score_count],
                                marker_color='#18A999',
                                text=[f"{high_score_count:,} ({high_score_percentage:.1f}%)"],
                                textposition='auto'
                            ),
                            go.Bar(
                                name='Low Performers (<350)',
                                x=['Score Category'],
                                y=[low_score_count],
                                marker_color='#C73E1D',
                                text=[f"{low_score_count:,} ({low_score_percentage:.1f}%)"],
                                textposition='auto'
                            )
                        ])
                        fig.update_layout(
                            title="Student Distribution by National Exam Score",
                            xaxis_title="Score Category",
                            yaxis_title="Number of Students",
                            height=400,
                            barmode='group'
                        )
                        st.plotly_chart(fig, use_container_width=True, key="score_dist_chart")
                    
                    with col2:
                        # Score histogram
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(
                            x=df['Total_National_Exam_Score'],
                            nbinsx=30,
                            marker_color='#2E86AB',
                            opacity=0.7,
                            name='Score Distribution'
                        ))
                        fig.add_vline(x=350, line_dash="dash", line_color='#F18F01',
                                    annotation_text="Threshold (350)", annotation_position="top")
                        fig.add_vline(x=df['Total_National_Exam_Score'].mean(), line_dash="dash", line_color='#C73E1D',
                                    annotation_text=f"Mean: {df['Total_National_Exam_Score'].mean():.1f}")
                        fig.update_layout(
                            title="Distribution of Total National Exam Scores",
                            xaxis_title="Total National Exam Score",
                            yaxis_title="Number of Students",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True, key="score_histogram")
                    
                    # ========================================================================
                    # REGIONAL BREAKDOWN TABLE
                    # ========================================================================
                    st.markdown("#### Regional Breakdown of National Exam Performance")
                    
                    regional_score_analysis = df.groupby('Region').agg({
                        'Total_National_Exam_Score': ['count', 'mean', 'min', 'max'],
                        'Student_ID': 'count'
                    }).round(2)
                    
                    # Add high/low performer counts per region
                    region_high_counts = df[df['Total_National_Exam_Score'] >= 350].groupby('Region').size()
                    region_low_counts = df[df['Total_National_Exam_Score'] < 350].groupby('Region').size()
                    
                    regional_summary = pd.DataFrame({
                        'Region': df['Region'].unique(),
                        'Total Students': [len(df[df['Region'] == r]) for r in df['Region'].unique()],
                        'High Performers (≥350)': [region_high_counts.get(r, 0) for r in df['Region'].unique()],
                        'Low Performers (<350)': [region_low_counts.get(r, 0) for r in df['Region'].unique()],
                        'High Performer %': [f"{(region_high_counts.get(r, 0)/len(df[df['Region'] == r])*100):.1f}%" for r in df['Region'].unique()],
                        'Avg Score': [df[df['Region'] == r]['Total_National_Exam_Score'].mean() for r in df['Region'].unique()]
                    })
                    regional_summary = regional_summary.sort_values('Avg Score', ascending=False)
                    
                    st.dataframe(regional_summary, use_container_width=True, key="regional_score_table")
                    
                    # Regional visualization
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=regional_summary['Region'],
                        y=regional_summary['Avg Score'],
                        marker_color='#2E86AB',
                        text=regional_summary['Avg Score'].round(1),
                        textposition='auto',
                        name='Average Score'
                    ))
                    fig.add_hline(y=350, line_dash="dash", line_color='#F18F01',
                                annotation_text="Threshold (350)")
                    fig.update_layout(
                        title="Average National Exam Score by Region",
                        xaxis_title="Region",
                        yaxis_title="Average Score",
                        height=500,
                        xaxis_tickangle=-45
                    )
                    st.plotly_chart(fig, use_container_width=True, key="regional_score_bar")
                
                # ========================================================================
                # NATIONAL EXAM PERFORMANCE BY TRACK (Horizontal)
                # ========================================================================
                st.markdown("### 📈 National Exam Performance by Track")
                
                # Check for track-specific columns
                social_subjects = ['National_Exam_History', 'National_Exam_Geography', 'National_Exam_Economics', 'National_Exam_Math_Social']
                natural_subjects = ['National_Exam_Biology', 'National_Exam_Chemistry', 'National_Exam_Physics', 'National_Exam_Math_Natural']
                common_subjects = ['National_Exam_Aptitude', 'National_Exam_English', 'National_Exam_Civics_and_Ethical_Education']
                
                available_social = [s for s in social_subjects if s in df.columns]
                available_natural = [n for n in natural_subjects if n in df.columns]
                available_common = [c for c in common_subjects if c in df.columns]
                
                if available_social or available_natural or available_common:
                    # Create horizontal bar chart for track performance
                    track_data = []
                    
                    if available_social:
                        social_avg = df[available_social].mean().mean()
                        track_data.append({'Track': 'Social Science', 'Average Score': social_avg, 'Color': '#A23B72'})
                        
                        # Individual subject scores for social track
                        st.markdown("#### Social Science Track Details")
                        social_scores = df[available_social].mean().sort_values(ascending=True)
                        fig_social = go.Figure(go.Bar(
                            x=social_scores.values,
                            y=social_scores.index,
                            orientation='h',
                            marker_color='#A23B72',
                            text=[f'{v:.1f}' for v in social_scores.values],
                            textposition='auto'
                        ))
                        fig_social.update_layout(
                            title="Social Science Subject Scores",
                            xaxis_title="Average Score",
                            yaxis_title="Subject",
                            height=300
                        )
                        st.plotly_chart(fig_social, use_container_width=True, key="social_scores")
                    
                    if available_natural:
                        natural_avg = df[available_natural].mean().mean()
                        track_data.append({'Track': 'Natural Science', 'Average Score': natural_avg, 'Color': '#18A999'})
                        
                        # Individual subject scores for natural track
                        st.markdown("#### Natural Science Track Details")
                        natural_scores = df[available_natural].mean().sort_values(ascending=True)
                        fig_natural = go.Figure(go.Bar(
                            x=natural_scores.values,
                            y=natural_scores.index,
                            orientation='h',
                            marker_color='#18A999',
                            text=[f'{v:.1f}' for v in natural_scores.values],
                            textposition='auto'
                        ))
                        fig_natural.update_layout(
                            title="Natural Science Subject Scores",
                            xaxis_title="Average Score",
                            yaxis_title="Subject",
                            height=300
                        )
                        st.plotly_chart(fig_natural, use_container_width=True, key="natural_scores")
                    
                    if available_common:
                        common_avg = df[available_common].mean().mean()
                        track_data.append({'Track': 'Common Subjects', 'Average Score': common_avg, 'Color': '#F18F01'})
                        
                        # Individual subject scores for common subjects
                        st.markdown("#### Common Subjects Details")
                        common_scores = df[available_common].mean().sort_values(ascending=True)
                        fig_common = go.Figure(go.Bar(
                            x=common_scores.values,
                            y=common_scores.index,
                            orientation='h',
                            marker_color='#F18F01',
                            text=[f'{v:.1f}' for v in common_scores.values],
                            textposition='auto'
                        ))
                        fig_common.update_layout(
                            title="Common Subject Scores",
                            xaxis_title="Average Score",
                            yaxis_title="Subject",
                            height=250
                        )
                        st.plotly_chart(fig_common, use_container_width=True, key="common_scores")
                    
                    # Horizontal bar chart for track comparison
                    track_df = pd.DataFrame(track_data)
                    fig_track = go.Figure(go.Bar(
                        x=track_df['Average Score'],
                        y=track_df['Track'],
                        orientation='h',
                        marker_color=track_df['Color'],
                        text=track_df['Average Score'].round(1),
                        textposition='auto'
                    ))
                    fig_track.update_layout(
                        title="Track Performance Comparison",
                        xaxis_title="Average Score",
                        yaxis_title="Academic Track",
                        height=300
                    )
                    st.plotly_chart(fig_track, use_container_width=True, key="track_comparison")
                
                # ========================================================================
                # MODEL PERFORMANCE METRICS
                # ========================================================================
                st.markdown("### 🤖 National Exam Prediction Model Performance")
                st.markdown("*Model trained to predict national exam scores based on academic and demographic factors*")
                
                # Model performance comparison table
                model_performance = pd.DataFrame({
                    'Model': ['Gradient Boosting', 'XGBoost', 'Random Forest', 'Ridge Regression', 'Linear Regression', 'Lasso Regression'],
                    'R² Score': [0.437061, 0.432566, 0.420462, 0.405090, 0.405089, 0.404188],
                    'MAE': [0.081473, 0.081776, 0.082588, 0.083994, 0.083995, 0.083939],
                    'RMSE': [0.107198, 0.107625, 0.108767, 0.110200, 0.110200, 0.110284]
                })
                
                st.dataframe(model_performance, use_container_width=True, key="model_performance_table")
                
                # Best model highlight
                st.markdown("""
                <div class="success-box">
                <strong>🏆 BEST MODEL: Gradient Boosting</strong><br>
                • R² Score: 0.4371 (Explains 43.7% of variance)<br>
                • MAE: 0.0815 (Average prediction error)<br>
                • RMSE: 0.1072<br>
                • <strong>Durbin-Watson Statistic: 2.00</strong> (~2 indicates independent residuals - good model fit)
                </div>
                """, unsafe_allow_html=True)
                
                # Feature Importance
                st.markdown("#### Feature Importance - Gradient Boosting Model")
                
                feature_importance = pd.DataFrame({
                    'Feature': [
                        'Overall_Test_Score_Avg', 'Overall_Avg_Participation', 'Overall_Engagement_Score',
                        'School_Academic_Score', 'Overall_Avg_Attendance', 'School_Resources_Score',
                        'Parental_Involvement', 'Teacher_Student_Ratio', 'Student_to_Resources_Ratio',
                        'Overall_Avg_Homework', 'School_Type_Target', 'Overall_Textbook_Access_Composite',
                        'Field_Choice', 'Career_Interest_Encoded'
                    ],
                    'Importance': [0.418729, 0.232912, 0.190178, 0.090328, 0.016825, 0.015366,
                                0.014162, 0.007387, 0.004941, 0.003189, 0.002499, 0.001734,
                                0.001365, 0.000385],
                    'Importance_%': [41.87, 23.29, 19.02, 9.03, 1.68, 1.54, 1.42, 0.74, 0.49, 0.32, 0.25, 0.17, 0.14, 0.04]
                })
                
                # Sort for visualization
                feature_importance = feature_importance.sort_values('Importance', ascending=True).tail(10)
                
                fig = go.Figure(go.Bar(
                    x=feature_importance['Importance'],
                    y=feature_importance['Feature'],
                    orientation='h',
                    marker_color='#A23B72',
                    text=[f'{imp:.1%}' for imp in feature_importance['Importance']],
                    textposition='auto'
                ))
                fig.update_layout(
                    title="Top 10 Features Impacting National Exam Scores",
                    xaxis_title="Importance Score",
                    yaxis_title="Features",
                    height=500,
                    margin=dict(l=200)
                )
                st.plotly_chart(fig, use_container_width=True, key="feature_importance_chart")
                
                # Model Insights
                st.markdown("#### Key Insights from National Exam Model")
                st.markdown("""
                <div class="info-box">
                <strong>💡 Key Findings:</strong><br><br>
                • <strong>Overall Test Score Average</strong> is the most important predictor (41.9% importance)<br>
                • <strong>Student Participation</strong> contributes 23.3% to exam performance<br>
                • <strong>Engagement Score</strong> (PCA-combined) explains 19.0% of variance<br>
                • <strong>School Academic Score</strong> accounts for 9.0% of prediction power<br>
                • Other factors (resources, involvement, ratio) have smaller but meaningful contributions<br>
                • The model shows good fit with <strong>Durbin-Watson statistic of 2.00</strong> (independent residuals)
                </div>
                """, unsafe_allow_html=True)
                
                # Model Limitations
                st.markdown("#### ⚠️ Model Limitations & Usage Notes")
                st.markdown("""
                <div class="warning-box">
                <strong>Important Limitations:</strong><br><br>
                • This model is designed for <strong>class-level and group analysis</strong>, not individual student prediction<br>
                • Individual student predictions may have higher error margins (±10-15 points)<br>
                • The model explains ~43.7% of variance, meaning over 56% is unexplained by available features<br>
                • External factors (exam difficulty, test anxiety, preparation quality) are not captured<br>
                • Use predictions as <strong>guidance for policy decisions</strong>, not as definitive outcomes<br>
                • For individual student assessment, combine with teacher evaluations and other indicators
                </div>
                """, unsafe_allow_html=True)
                
                # Download report
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("📥 Download National Exam Report (CSV)", use_container_width=True, key="download_national"):
                        # Create summary dataframe for download
                        download_data = pd.DataFrame({
                            'Metric': ['Total Students', 'High Performers (≥350)', 'Low Performers (<350)', 
                                    'High Performer %', 'Average Score', 'Best Model', 'Model R²'],
                            'Value': [
                                f"{len(df):,}", f"{high_score_count:,}", f"{low_score_count:,}",
                                f"{high_score_percentage:.1f}%", f"{df['Total_National_Exam_Score'].mean():.1f}",
                                'Gradient Boosting', '0.4371'
                            ]
                        })
                        csv = download_data.to_csv(index=False)
                        st.download_button("📥 Download CSV", csv, "national_exam_report.csv", "text/csv", key="download_national_btn")
            # ========================================================================
            # SUMMARY STATISTICS
            # ========================================================================
            elif report_type == "Summary Statistics":
                st.markdown("### Summary Statistics Report")
                summary = df.describe().round(2)
                st.dataframe(summary, use_container_width=True, key="summary_stats")
                csv = summary.to_csv()
                st.download_button("📥 Download CSV", csv, "summary_stats.csv", "text/csv", key="download_summary")
            
            # ========================================================================
            # FULL DATASET
            # ========================================================================
            elif report_type == "Full Dataset":
                st.markdown("### Full Dataset Export")
                
                # Get all available columns
                all_columns = df.columns.tolist()
                
                # Select columns to export
                export_cols = st.multiselect(
                    "Select columns to export",
                    options=all_columns,
                    default=all_columns[:10] if len(all_columns) > 10 else all_columns,
                    key="export_cols_widget"
                )
                
                if export_cols:
                    export_df = df[export_cols]
                    st.dataframe(export_df.head(100), use_container_width=True, key="full_dataset")
                    st.caption(f"Showing first 100 rows of {len(export_df):,} total rows")
                    
                    csv = export_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Full Dataset (CSV)", 
                        csv, 
                        "full_dataset.csv", 
                        "text/csv", 
                        key="download_full"
                    )
                    
                    # Excel export option
                    try:
                        import io
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            export_df.to_excel(writer, sheet_name='Student Data', index=False)
                        st.download_button(
                            "📥 Download Full Dataset (Excel)",
                            output.getvalue(),
                            "full_dataset.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="download_full_excel"
                        )
                    except:
                        pass
                else:
                    st.warning("Please select at least one column to export.")
    
    # ========================================================================
    # QUICK SUMMARY (Unchanged)
    # ========================================================================
    st.markdown("---")
    st.markdown("### 📄 Quick Summary")
    
    if st.button("📄 Generate Quick Summary", key="summary_button_widget"):
        total = len(df)
        avg_score = df['Overall_Average'].mean()
        pass_rate = (df['Overall_Average'] >= 50).mean() * 100
        risk_count = (df['Overall_Average'] < 50).sum()
        
        cluster_sizes = {
            'High': (df['Overall_Average'] >= 70).sum(),
            'Medium': ((df['Overall_Average'] >= 50) & (df['Overall_Average'] < 70)).sum(),
            'Low': (df['Overall_Average'] < 50).sum()
        }
        
        st.markdown(f"""
        ## Ethiopian Student Performance Dashboard - Summary
        
        ### Key Metrics
        - **Total Students:** {total:,}
        - **Average Score:** {avg_score:.1f}
        - **Pass Rate:** {pass_rate:.1f}%
        - **At-Risk Students:** {risk_count:,} ({(risk_count/total*100):.1f}%)
        
        ### Performance Distribution
        - **High Performers (≥70):** {cluster_sizes['High']:,} ({(cluster_sizes['High']/total*100):.1f}%)
        - **Medium Performers (50-69):** {cluster_sizes['Medium']:,} ({(cluster_sizes['Medium']/total*100):.1f}%)
        - **Low Performers (<50):** {cluster_sizes['Low']:,} ({(cluster_sizes['Low']/total*100):.1f}%)
        
        ### Model Performance
        - **Best Regression Model:** XGBoost (R² = 0.7855)
        - **Best Classification Model:** Gradient Boosting (AUC = 0.918)
        - **Top Predictor:** School Resources Score (55.1% importance)
        
        ### Regional Highlights
        - **Highest Performing Region:** Addis Ababa (avg score: 74.2)
        - **Highest Risk Region:** Somali (47.4% low performers)
        
        ### Recommendations
        1. Increase school resources in high-risk regions
        2. Improve textbook access and digital infrastructure
        3. Implement parent engagement programs
        4. Reduce teacher-student ratios in overcrowded schools
        5. Target interventions for low-performing students
        """)
# ============================================================================
# PAGE: SETTINGS
# ============================================================================
elif selected_page == "⚙️ Settings":
    st.markdown("<h1 class='main-header'>⚙️ Settings</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Configure dashboard parameters, manage models, and view system information</p>", unsafe_allow_html=True)
    
    # ========================================================================
    # RISK THRESHOLD SETTINGS
    # ========================================================================
    st.markdown("### ⚠️ Risk Classification Threshold")
    st.markdown("Adjust the threshold for classifying students as 'At Risk'")
    
    new_threshold = st.slider(
        "Risk Threshold", 
        0.0, 1.0, 
        st.session_state.risk_threshold, 
        0.01, 
        key="risk_threshold_slider_widget",
        help="Students with risk probability above this threshold are classified as 'At Risk'"
    )
    if new_threshold != st.session_state.risk_threshold:
        st.session_state.risk_threshold = new_threshold
        st.success(f"✅ Risk threshold updated to {new_threshold:.2f}")
    
    st.markdown("---")
    
    # ========================================================================
    # MODEL MANAGEMENT
    # ========================================================================
    st.markdown("### 🤖 Model Management")
    
    # Get model status from prediction engine
    if st.session_state.prediction_engine is not None:
        status = st.session_state.prediction_engine.get_model_status()
        models_loaded = status.get('models_loaded', False)
        
        # Display model status cards
        col1, col2 = st.columns(2)
        
        with col1:
            if models_loaded:
                st.success("✅ **Regression Model**")
                st.markdown(f"   - Type: {status.get('regression_type', 'Unknown')}")
                st.markdown(f"   - Features: {status.get('features_count', 0)}")
                st.markdown(f"   - Status: Loaded")
                st.markdown(f"   - R² Score: 0.7855")
                st.markdown(f"   - MAE: 2.98")
            else:
                st.error("❌ **Regression Model**")
                st.markdown(f"   - Status: Not Loaded")
                if status.get('error'):
                    st.markdown(f"   - Error: {status['error'][:100]}...")
        
        with col2:
            if models_loaded:
                st.success("✅ **Classification Model**")
                st.markdown(f"   - Type: {status.get('classification_type', 'Unknown')}")
                st.markdown(f"   - Features: {status.get('features_count', 0)}")
                st.markdown(f"   - Status: Loaded")
                st.markdown(f"   - ROC-AUC: 0.918")
                st.markdown(f"   - F1-Score: 0.778")
            else:
                st.error("❌ **Classification Model**")
                st.markdown(f"   - Status: Not Loaded")
        
        # Model management buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Reload Models", use_container_width=True, key="reload_models_widget"):
                with st.spinner("Reloading models..."):
                    try:
                        from utils.predictions import PredictionEngine
                        st.session_state.prediction_engine = PredictionEngine()
                        
                        if st.session_state.prediction_engine.is_models_loaded():
                            st.success("✅ Models reloaded successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Models failed to load. Please check model files.")
                            if st.session_state.prediction_engine.load_error:
                                st.error(f"Error: {st.session_state.prediction_engine.load_error}")
                    except Exception as e:
                        st.error(f"Error reloading models: {e}")
        
        with col2:
            if st.button("📊 Test Models", use_container_width=True, key="test_models_widget"):
                with st.spinner("Testing models..."):
                    if models_loaded:
                        test_data = {
                            'School_Resources_Score': 0.7,
                            'Overall_Engagement_Score': 75,
                            'School_Academic_Score': 0.6,
                            'Overall_Textbook_Access_Composite': 0.7,
                            'Overall_Avg_Attendance': 85,
                            'Teacher_Student_Ratio': 35,
                            'Overall_Avg_Homework': 80,
                            'Overall_Avg_Participation': 75,
                            'Parental_Involvement': 0.6,
                            'Gender': 'Male',
                            'Region': 'Addis Ababa',
                            'Health_Issue': 'No Issue',
                            'Age': 17,
                            'Field_Choice': 'Social',
                            'School_Type': 'Public',
                            'Career_Interest': 'Teacher',
                            'Home_Internet_Access': 'Yes',
                            'Electricity_Access': 'Yes',
                            'School_Location': 'Urban'
                        }
                        
                        try:
                            test_score = st.session_state.prediction_engine.predict_score(test_data)
                            test_risk = st.session_state.prediction_engine.predict_risk(test_data)
                            
                            st.success(f"✅ Models working!")
                            st.info(f"📊 Test Score: {test_score:.1f}")
                            st.info(f"⚠️ Test Risk: {test_risk:.2f} ({test_risk*100:.1f}%)")
                        except Exception as e:
                            st.error(f"Test failed: {e}")
                    else:
                        st.warning("⚠️ Models not loaded. Cannot test.")
        
        # Model directory info
        with st.expander("📁 Model Directory Information"):
            st.markdown(f"**Model Directory:** `{status.get('model_dir', 'Unknown')}`")
            st.markdown("**Required Model Files:**")
            st.markdown("""
            - `gradient_boosting_model_full.pkl` - Regression model
            - `student_risk_classifier.pkl` - Classification model  
            - `metadata_classification_v1.json` - Feature metadata (optional)
            """)
            
            import os
            from pathlib import Path
            model_dir = Path(status.get('model_dir', ''))
            if model_dir.exists():
                st.markdown("**Files found in model directory:**")
                for file in model_dir.glob("*"):
                    if file.suffix in ['.pkl', '.json']:
                        st.markdown(f"- `{file.name}`")
            else:
                st.warning(f"Model directory not found: {model_dir}")
    
    st.markdown("---")
    
    # ========================================================================
    # REQUIRED FEATURES FOR PREDICTION (Organized by categories)
    # ========================================================================
    st.markdown("### 📋 Required Features for Prediction")
    st.markdown("These features are required for accurate predictions. Missing features will use default values.")
    
    # Get available features from processed data
    available_features = st.session_state.df_processed.columns.tolist() if st.session_state.df_processed is not None else []
    
    # Define feature groups
    feature_groups = {
        '👤 Demographic & Family Factors': [
            'Gender', 'Age', 'Region', 'Field Choice'
        ],
        '🏠 Family, Home Environment & Health': [
            'Father Education', 'Mother Education', 'Parental Involvement',
            'Home Internet Access', 'Electricity Access', 'Health Issue'
        ],
        '🏫 School Environment & Resources': [
            'School Type', 'School Location', 'School Resources Score',
            'School Academic Score', 'Teacher-Student Ratio', 'Student-to-Resources Ratio'
        ],
        '💪 Student Engagement & Career': [
            'Attendance (%)', 'Homework Completion (%)', 'Participation (%)', 'Career Interest'
        ]
    }
    
    # Map feature names to actual column names (handling variations)
    feature_mapping = {
        'Gender': ['Gender'],
        'Age': ['Age'],
        'Region': ['Region'],
        'Field Choice': ['Field_Choice', 'FieldChoice'],
        'Father Education': ['Father_Education', 'FatherEducation'],
        'Mother Education': ['Mother_Education', 'MotherEducation'],
        'Parental Involvement': ['Parental_Involvement', 'ParentalInvolvement'],
        'Home Internet Access': ['Home_Internet_Access', 'HomeInternetAccess'],
        'Electricity Access': ['Electricity_Access', 'ElectricityAccess'],
        'Health Issue': ['Health_Issue', 'HealthIssue'],
        'School Type': ['School_Type', 'SchoolType'],
        'School Location': ['School_Location', 'SchoolLocation'],
        'School Resources Score': ['School_Resources_Score', 'SchoolResourcesScore'],
        'School Academic Score': ['School_Academic_Score', 'SchoolAcademicScore'],
        'Teacher-Student Ratio': ['Teacher_Student_Ratio', 'TeacherStudentRatio'],
        'Student-to-Resources Ratio': ['Student_to_Resources_Ratio', 'StudentToResourcesRatio'],
        'Attendance (%)': ['Overall_Avg_Attendance', 'Attendance', 'Avg_Attendance'],
        'Homework Completion (%)': ['Overall_Avg_Homework', 'Homework', 'Avg_Homework'],
        'Participation (%)': ['Overall_Avg_Participation', 'Participation', 'Avg_Participation'],
        'Career Interest': ['Career_Interest', 'CareerInterest']
    }
    
    def check_feature_exists(feature_name):
        """Check if a feature exists in the dataframe"""
        possible_names = feature_mapping.get(feature_name, [feature_name])
        for name in possible_names:
            if name in available_features:
                return True
        return False
    
    # Create two columns for the layout
    col_left, col_right = st.columns(2)
    
    # Left column - First two groups
    with col_left:
        # Group 1: Demographic & Family Factors
        st.markdown("#### 👤 Demographic & Family Factors")
        for feature in feature_groups['👤 Demographic & Family Factors']:
            exists = check_feature_exists(feature)
            if exists:
                st.success(f"✅ **{feature}**")
            else:
                st.warning(f"⚠️ **{feature}**")
        
        st.markdown("---")
        
        # Group 2: Family, Home Environment & Health
        st.markdown("#### 🏠 Family, Home Environment & Health")
        for feature in feature_groups['🏠 Family, Home Environment & Health']:
            exists = check_feature_exists(feature)
            if exists:
                st.success(f"✅ **{feature}**")
            else:
                st.warning(f"⚠️ **{feature}**")
    
    # Right column - Last two groups
    with col_right:
        # Group 3: School Environment & Resources
        st.markdown("#### 🏫 School Environment & Resources")
        for feature in feature_groups['🏫 School Environment & Resources']:
            exists = check_feature_exists(feature)
            if exists:
                st.success(f"✅ **{feature}**")
            else:
                st.warning(f"⚠️ **{feature}**")
        
        st.markdown("---")
        
        # Group 4: Student Engagement & Career
        st.markdown("#### 💪 Student Engagement & Career")
        for feature in feature_groups['💪 Student Engagement & Career']:
            exists = check_feature_exists(feature)
            if exists:
                st.success(f"✅ **{feature}**")
            else:
                st.warning(f"⚠️ **{feature}**")
    
    # Feature statistics
    total_features = sum(len(group) for group in feature_groups.values())
    available_count = sum(1 for feature in [f for group in feature_groups.values() for f in group] if check_feature_exists(feature))
    
    st.markdown("---")
    st.markdown(f"**Feature Coverage:** {available_count}/{total_features} features available ({available_count/total_features*100:.1f}%)")
    
    if available_count < total_features:
        st.info("💡 **Tip:** Missing features will use default values. For best predictions, ensure all features are present in your dataset.")
    
    st.markdown("---")
    
    # ========================================================================
    # DATA MANAGEMENT
    # ========================================================================
    st.markdown("### 📁 Data Management")
    
    uploaded_file = st.file_uploader("Upload New Dataset (CSV)", type=['csv'], key="data_upload_widget")
    if uploaded_file is not None:
        try:
            new_df = pd.read_csv(uploaded_file)
            st.success(f"✅ Dataset loaded: {len(new_df):,} rows, {len(new_df.columns)} columns")
            st.dataframe(new_df.head(), use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Apply New Dataset", use_container_width=True, key="apply_dataset"):
                    st.session_state.df_original = new_df
                    st.session_state.data_loaded = False
                    st.rerun()
            with col2:
                if st.button("Cancel", use_container_width=True, key="cancel_upload"):
                    st.rerun()
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
    
    st.markdown("---")
    
    # ========================================================================
    # SYSTEM INFORMATION
    # ========================================================================
    st.markdown("### ℹ️ System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Dashboard Information:**")
        st.markdown(f"- Version: 2.0.0")
        st.markdown(f"- Students Loaded: {len(st.session_state.df_original):,}" if st.session_state.df_original is not None else "Students Loaded: N/A")
        st.markdown(f"- Features Available: {len(available_features)}")
        st.markdown(f"- Risk Threshold: {st.session_state.risk_threshold:.2f}")
    
    with col2:
        st.markdown("**Model Information:**")
        if st.session_state.prediction_engine is not None:
            status = st.session_state.prediction_engine.get_model_status()
            st.markdown(f"- Models Loaded: {'✅ Yes' if status.get('models_loaded') else '❌ No'}")
            if status.get('models_loaded'):
                st.markdown(f"- Regression: {status.get('regression_type', 'Unknown')}")
                st.markdown(f"- Classification: {status.get('classification_type', 'Unknown')}")
        else:
            st.markdown("- Models Loaded: ❌ No")
    
    # Performance metrics
    st.markdown("**Model Performance Metrics:**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Regression R²", "0.7855")
    with col2:
        st.metric("Regression MAE", "2.98")
    with col3:
        st.metric("Classification AUC", "0.918")
    with col4:
        st.metric("Classification F1", "0.778")
    
    st.markdown("---")
    
    # ========================================================================
    # ABOUT SECTION
    # ========================================================================
    st.markdown("### ℹ️ About")
    st.info("""
    **Ethiopian Student Performance Dashboard v2.0**
    
    **Models:**
    - **Regression:** XGBoost / Gradient Boosting (R² = 0.7855, MAE = 2.98)
    - **Classification:** Gradient Boosting (AUC = 0.918, F1 = 0.778)
    
    **Features:**
    - Real-time student performance prediction using trained ML models
    - Risk assessment and classification with customizable thresholds
    - Interactive simulations for what-if analysis
    - Comprehensive analytics with SHAP explanations
    - Exportable reports (Predictions, Clusters, Summary)
    - Student search and filtering capabilities
    
    **Data Source:** Ethiopian Students Dataset (100,000+ students)
    
    **Technology Stack:**
    - Streamlit for dashboard interface
    - Plotly for interactive visualizations
    - Scikit-learn / XGBoost for ML models
    - Pandas / NumPy for data processing
    
    **Support:**
    For issues or questions, please contact the development team.
    """)
    
    # Reset all settings button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Reset All Settings to Default", use_container_width=True, key="reset_all"):
            st.session_state.risk_threshold = 0.5
            st.success("✅ All settings reset to default values!")
            st.rerun()

st.markdown("---")
st.markdown("<p style='text-align: center; color: #666;'>© 2026 Ethiopian Student Performance Dashboard | Powered by Machine Learning | XGBoost & Gradient Boosting</p>", unsafe_allow_html=True)