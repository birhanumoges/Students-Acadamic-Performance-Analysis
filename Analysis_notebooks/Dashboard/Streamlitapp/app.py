"""
Ethiopian Student Performance Analytics Dashboard
Streamlit Version - Complete Replication of Dash App
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

# Set page config FIRST
st.set_page_config(
    page_title="Ethiopian Student Performance Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import utilities
from utils.data_processor import DataProcessor, COLOR_SCHEME
from utils.predictions import PredictionEngine, load_models, make_prediction_corrected
from utils.visualizations import Visualizer, initialize_global_vars, set_global_data, set_prediction_result

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = "Overview"
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'prediction_engine' not in st.session_state:
    st.session_state.prediction_engine = None
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = None
if 'config' not in st.session_state:
    st.session_state.config = None
if 'risk_threshold' not in st.session_state:
    st.session_state.risk_threshold = 0.5
if 'regression_models' not in st.session_state:
    st.session_state.regression_models = {}
if 'classification_model' not in st.session_state:
    st.session_state.classification_model = None
if 'clustering_analysis' not in st.session_state:
    st.session_state.clustering_analysis = None

# Paths
CONFIG_PATH = 'config/settings.json'
DATA_PATH = 'data/ethiopian_students_dataset.csv'

# Load configuration
@st.cache_data
def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    return {}

# Load data
@st.cache_data
def load_data():
    """Load and preprocess data"""
    try:
        # Check for data file in multiple locations
        possible_paths = [
            DATA_PATH,
            r"C:/Users/DELL/Documents/project_data/ethiopian_students_dataset.csv",
            r"C:/Users/DELL/AIgravity/ethiopian_students_dataset.csv",
            "../data/ethiopian_students_dataset.csv"
        ]
        
        df_original = None
        for path in possible_paths:
            if os.path.exists(path):
                df_original = pd.read_csv(path)
                st.success(f"✅ Data loaded from: {path}")
                break
        
        if df_original is None:
            # Create sample data
            st.warning("Dataset not found. Using sample data for demonstration.")
            np.random.seed(42)
            n = 1000
            regions = ['Addis Ababa', 'Oromia', 'Amhara', 'Tigray', 'SNNP', 'Somali']
            df_original = pd.DataFrame({
                'Student_ID': range(1, n+1),
                'Overall_Average': np.random.normal(60, 15, n).clip(0, 100),
                'Region': np.random.choice(regions, n),
                'Gender': np.random.choice(['Male', 'Female'], n),
                'School_Resources_Score': np.random.uniform(0.2, 0.9, n),
                'Parental_Involvement': np.random.uniform(0.1, 0.9, n),
                'Overall_Avg_Attendance': np.random.uniform(50, 100, n),
                'Overall_Avg_Homework': np.random.uniform(40, 100, n),
                'Overall_Avg_Participation': np.random.uniform(40, 100, n),
                'Overall_Textbook_Access_Composite': np.random.uniform(0.2, 0.9, n),
                'Teacher_Student_Ratio': np.random.uniform(20, 60, n),
                'School_Academic_Score': np.random.uniform(0.3, 0.9, n),
                'Health_Issue': np.random.choice(['No Issue', 'Minor', 'Moderate'], n),
                'Age': np.random.randint(15, 22, n),
                'School_Type': np.random.choice(['Public', 'Private'], n),
                'School_Location': np.random.choice(['Urban', 'Rural'], n),
                'Field_Choice': np.random.choice(['Social', 'Natural'], n),
                'Career_Interest': np.random.choice(['Teacher', 'Doctor', 'Engineer'], n),
                'Father_Education': np.random.choice(['High School', 'College', 'Primary'], n),
                'Mother_Education': np.random.choice(['High School', 'College', 'Primary'], n),
                'Home_Internet_Access': np.random.choice(['Yes', 'No'], n),
                'Electricity_Access': np.random.choice(['Yes', 'No'], n)
            })
        
        # Preprocess
        processor = DataProcessor()
        df_processed = processor.load_and_preprocess_data(df_original)
        df_encoded = processor.encode_categorical_features(df_processed)
        
        return df_original, df_processed, df_encoded
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

# Load models
@st.cache_resource
def load_models_cached():
    """Load trained models"""
    reg_model, class_model, reg_scaler, class_scaler, reg_features, class_features = load_models()
    
    # Create regression models dict
    regression_models = {
        'XGBoost': {'R2': 0.7855, 'MAE': 2.98, 'RMSE': 3.72},
        'GradientBoosting': {'R2': 0.7851, 'MAE': 2.99, 'RMSE': 3.73},
        'RandomForest': {'R2': 0.7719, 'MAE': 3.07, 'RMSE': 3.84},
        'LinearRegression': {'R2': 0.7690, 'MAE': 3.10, 'RMSE': 3.86}
    }
    
    # Classification model
    classification_model = {
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
    clustering_analysis = {
        'silhouette_score': 0.1742,
        'cluster_sizes': {'Low': 39380, 'Medium': 38933, 'High': 21687},
        'regional_risk': {
            'Somali': 47.4, 'Benishangul-Gumuz': 45.54, 'Afar': 45.27,
            'Tigray': 44.76, 'Sidama': 43.24, 'Gambela': 42.24,
            'SNNP': 40.57, 'Oromia': 39.21, 'Amhara': 39.18,
            'Addis Ababa': 21.32
        }
    }
    
    return regression_models, classification_model, clustering_analysis, reg_model, class_model, reg_scaler, class_scaler, reg_features, class_features

# Initialize components
def initialize_components():
    config = load_config()
    processor = DataProcessor()
    engine = PredictionEngine()
    return processor, engine, config

# Load data and initialize
if st.session_state.df_original is None:
    with st.spinner("Loading data and models..."):
        df_original, df_processed, df_encoded = load_data()
        if df_original is not None:
            regression_models, classification_model, clustering_analysis, reg_model, class_model, reg_scaler, class_scaler, reg_features, class_features = load_models_cached()
            
            st.session_state.df_original = df_original
            st.session_state.df_processed = df_processed
            st.session_state.df_encoded = df_encoded
            st.session_state.regression_models = regression_models
            st.session_state.classification_model = classification_model
            st.session_state.clustering_analysis = clustering_analysis
            st.session_state.reg_model = reg_model
            st.session_state.class_model = class_model
            st.session_state.reg_scaler = reg_scaler
            st.session_state.class_scaler = class_scaler
            st.session_state.reg_features = reg_features
            st.session_state.class_features = class_features
            
            config = load_config()
            st.session_state.config = config
            st.session_state.risk_threshold = config.get('risk_threshold', 0.5)
            st.session_state.data_processor = DataProcessor()
            st.session_state.prediction_engine = PredictionEngine()
            
            # Initialize global vars for visualizations
            initialize_global_vars()
            set_global_data(regression_models, "XGBoost", {}, classification_model, 
                           clustering_analysis, None, df_original, df_encoded)
            
            st.success("✅ Dashboard ready!")
        else:
            st.error("Failed to load data")
            st.stop()

# Sidebar Navigation
st.sidebar.title("📊 Student Analytics")
st.sidebar.markdown("---")

nav_options = ["📈 Overview", "👥 Students", "📊 Analytics", "🎯 Simulation", "📋 Reports", "⚙️ Settings"]
selected_page = st.sidebar.radio("Navigation", nav_options)

st.sidebar.markdown("---")

# Quick stats in sidebar
if st.session_state.df_processed is not None:
    df = st.session_state.df_processed
    st.sidebar.markdown("### 📊 Quick Stats")
    st.sidebar.metric("Total Students", f"{len(df):,}")
    if 'Overall_Average' in df.columns:
        st.sidebar.metric("Avg Score", f"{df['Overall_Average'].mean():.1f}")
        risk_count = (df['Overall_Average'] < 50).sum()
        st.sidebar.metric("At-Risk", f"{risk_count:,}", delta=f"{(risk_count/len(df)*100):.1f}%")

st.sidebar.markdown("---")
st.sidebar.info(
    "**Model Performance**\n\n"
    "• **Regression**: XGBoost (R² = 0.7855)\n"
    "• **Classification**: Gradient Boosting (AUC = 0.918)\n"
    "• **Top Feature**: School Resources Score"
)

# ============================================================================
# PAGE: OVERVIEW
# ============================================================================
if selected_page == "📈 Overview":
    st.markdown("<h1 style='color: #2E86AB;'>📈 Overview Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("Comprehensive analysis of Ethiopian students' academic performance")
    
    df = st.session_state.df_processed
    df_original = st.session_state.df_original
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    total = len(df)
    avg_score = df['Overall_Average'].mean() if 'Overall_Average' in df.columns else 0
    pass_rate = (df['Overall_Average'] >= 50).mean() * 100 if 'Overall_Average' in df.columns else 0
    risk_count = (df['Overall_Average'] < 50).sum() if 'Overall_Average' in df.columns else 0
    
    with col1:
        st.metric("Total Students", f"{total:,}")
    with col2:
        st.metric("Average Score", f"{avg_score:.1f}")
    with col3:
        st.metric("Pass Rate", f"{pass_rate:.1f}%")
    with col4:
        st.metric("At-Risk", f"{risk_count:,}", delta=f"{(risk_count/total*100):.1f}%", delta_color="inverse")
    
    # Charts
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(Visualizer.create_score_histogram(df), use_container_width=True)
    with col2:
        st.plotly_chart(Visualizer.create_risk_bar(df), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(Visualizer.create_region_bar(df_original), use_container_width=True)
    with col2:
        st.plotly_chart(Visualizer.create_gender_bar(df), use_container_width=True)
    
    # Correlation Heatmap
    st.subheader("Feature Correlation Analysis")
    st.plotly_chart(Visualizer.create_correlation_heatmap(df), use_container_width=True)
    
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

# ============================================================================
# PAGE: STUDENTS
# ============================================================================
elif selected_page == "👥 Students":
    st.markdown("<h1 style='color: #2E86AB;'>👥 Student Management</h1>", unsafe_allow_html=True)
    st.markdown("Search, filter, and analyze individual student performance")
    
    df_original = st.session_state.df_original.copy()
    data_processor = st.session_state.data_processor
    prediction_engine = st.session_state.prediction_engine
    
    # Prepare display data
    display_df = df_original.copy()
    if 'Overall_Average' in display_df.columns:
        display_df['Risk_Status'] = display_df['Overall_Average'].apply(lambda x: 'At Risk' if x < 50 else 'Not at Risk')
    
    # Filters in sidebar
    st.sidebar.markdown("### 🔍 Filters")
    
    if 'Region' in display_df.columns:
        regions = ['All'] + sorted(display_df['Region'].unique().tolist())
        selected_region = st.sidebar.selectbox("Region", regions)
        if selected_region != 'All':
            display_df = display_df[display_df['Region'] == selected_region]
    
    if 'Gender' in display_df.columns:
        gender_filter = st.sidebar.selectbox("Gender", ["All", "Male", "Female"])
        if gender_filter != "All":
            display_df = display_df[display_df['Gender'] == gender_filter]
    
    st.sidebar.caption(f"Showing {len(display_df)} students")
    
    # Search
    search_term = st.text_input("🔍 Search by Student ID", placeholder="Enter Student ID...")
    if search_term:
        display_df = display_df[display_df['Student_ID'].astype(str).str.contains(search_term, case=False)]
    
    # Display table
    display_cols = ['Student_ID', 'Region', 'Gender', 'Overall_Average', 'Risk_Status']
    display_cols = [c for c in display_cols if c in display_df.columns]
    
    selected_idx = st.dataframe(
        display_df[display_cols].head(100),
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row"
    )
    
    # Profile Panel
    st.markdown("---")
    st.markdown("### 👤 Student Profile")
    
    if selected_idx.selection.rows:
        selected_student = display_df.iloc[selected_idx.selection.rows[0]]
        
        # Get predictions
        student_data = selected_student.to_dict()
        features_df = data_processor.prepare_features_for_prediction(student_data)
        predicted_score = prediction_engine.predict_score(features_df) if prediction_engine else 70.0
        risk_prob = prediction_engine.predict_risk(features_df) if prediction_engine else 0.5
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Student ID:** {selected_student.get('Student_ID', 'N/A')}")
            st.markdown(f"**Region:** {selected_student.get('Region', 'N/A')}")
            st.markdown(f"**Gender:** {selected_student.get('Gender', 'N/A')}")
            st.markdown(f"**Age:** {selected_student.get('Age', 'N/A')}")
        with col2:
            st.markdown(f"**Actual Score:** {selected_student.get('Overall_Average', 'N/A'):.1f}" if 'Overall_Average' in selected_student else "N/A")
            st.markdown(f"**Predicted Score:** {predicted_score:.1f}")
            st.markdown(f"**Risk Probability:** {risk_prob*100:.1f}%")
            st.progress(risk_prob)
            st.markdown(f"**Risk Status:** {'🔴 AT RISK' if risk_prob > 0.5 else '🟢 NOT AT RISK'}")
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        recommendations = prediction_engine.get_recommendations(student_data, predicted_score, risk_prob) if prediction_engine else ["No recommendations available"]
        for rec in recommendations:
            if rec.startswith("🔴"):
                st.error(rec)
            elif rec.startswith("✅"):
                st.success(rec)
            else:
                st.info(rec)
    else:
        st.info("👆 Select a student from the table to view detailed profile")

# ============================================================================
# PAGE: ANALYTICS
# ============================================================================
elif selected_page == "📊 Analytics":
    st.markdown("<h1 style='color: #2E86AB;'>📊 Analytics Dashboard</h1>", unsafe_allow_html=True)
    
    df = st.session_state.df_processed
    config = st.session_state.config
    regression_models = st.session_state.regression_models
    classification_model = st.session_state.classification_model
    
    tab1, tab2, tab3 = st.tabs(["🔍 Diagnostics", "🤖 Modeling", "💡 Explainability"])
    
    with tab1:
        st.plotly_chart(Visualizer.create_correlation_heatmap(df), use_container_width=True)
        st.plotly_chart(Visualizer.create_scatter_plot(df, 'Overall_Avg_Attendance', 'Overall_Average'), use_container_width=True)
    
    with tab2:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Best Regression", "XGBoost")
            st.metric("R² Score", "0.7855")
        with col2:
            st.metric("MAE", "2.98")
            st.metric("RMSE", "3.72")
        with col3:
            st.metric("Best Classification", "Gradient Boosting")
            st.metric("ROC-AUC", "0.918")
            st.metric("F1-Score", "0.778")
        
        # Feature importance
        reg_importance = config.get('regression_feature_importance', {})
        st.plotly_chart(Visualizer.create_feature_importance_plot(reg_importance, "Regression Feature Importance"), use_container_width=True)
        
        class_importance = config.get('classification_feature_importance', {})
        st.plotly_chart(Visualizer.create_feature_importance_plot(class_importance, "Classification Feature Importance"), use_container_width=True)
    
    with tab3:
        st.info("SHAP analysis would appear here with trained models")
        st.markdown("""
        **SHAP Value Interpretation:**
        - Blue: Feature increases risk probability
        - Red: Feature decreases risk probability
        - Magnitude: Strength of impact on prediction
        """)

# ============================================================================
# PAGE: SIMULATION
# ============================================================================
elif selected_page == "🎯 Simulation":
    st.markdown("<h1 style='color: #2E86AB;'>🎯 Performance Simulation</h1>", unsafe_allow_html=True)
    
    data_processor = st.session_state.data_processor
    prediction_engine = st.session_state.prediction_engine
    
    col1, col2 = st.columns(2)
    with col1:
        school_resources = st.slider("School Resources Score", 0.0, 1.0, 0.5, 0.05)
        attendance = st.slider("Attendance (%)", 0, 100, 75)
        homework = st.slider("Homework (%)", 0, 100, 65)
    with col2:
        participation = st.slider("Participation (%)", 0, 100, 70)
        textbook = st.slider("Textbook Access", 0.0, 1.0, 0.5, 0.05)
        teacher_ratio = st.slider("Teacher-Student Ratio", 10, 100, 40)
    
    if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
        input_data = {
            'School_Resources_Score': school_resources,
            'Overall_Textbook_Access_Composite': textbook,
            'Teacher_Student_Ratio': teacher_ratio,
            'Overall_Avg_Attendance': attendance,
            'Overall_Avg_Homework': homework,
            'Overall_Avg_Participation': participation,
            'School_Academic_Score': 0.5,
            'Parental_Involvement': 0.5,
            'Gender': 0,
            'Region': 'Oromia',
            'Health_Issue': 'No Issue',
            'Age': 17
        }
        
        engagement = (attendance * 0.4 + homework * 0.3 + participation * 0.3) / 100
        input_data['Overall_Engagement_Score'] = engagement * 100
        
        features_df = data_processor.prepare_features_for_prediction(input_data)
        predicted_score = prediction_engine.predict_score(features_df) if prediction_engine else 70.0
        risk_prob = prediction_engine.predict_risk(features_df) if prediction_engine else 0.5
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Score", f"{predicted_score:.1f}")
            st.plotly_chart(Visualizer.create_risk_gauge(risk_prob), use_container_width=True)
        with col2:
            st.metric("Risk Probability", f"{risk_prob*100:.1f}%")
            st.progress(risk_prob)
            st.markdown(f"**Status:** {'🔴 AT RISK' if risk_prob > 0.5 else '🟢 NOT AT RISK'}")

# ============================================================================
# PAGE: REPORTS
# ============================================================================
elif selected_page == "📋 Reports":
    st.markdown("<h1 style='color: #2E86AB;'>📋 Reports</h1>", unsafe_allow_html=True)
    
    df_original = st.session_state.df_original
    config = st.session_state.config
    
    report_type = st.selectbox("Report Type", ["Predictions Report", "Summary Statistics", "Full Dataset"])
    
    if st.button("📊 Generate Report", type="primary", use_container_width=True):
        if report_type == "Predictions Report":
            report_df = df_original.head(100).copy()
            if 'Overall_Average' in report_df.columns:
                report_df['Risk_Status'] = report_df['Overall_Average'].apply(lambda x: 'At Risk' if x < 50 else 'Not at Risk')
            st.dataframe(report_df, use_container_width=True)
            csv = report_df.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, "predictions_report.csv", "text/csv")
        
        elif report_type == "Summary Statistics":
            summary = df_original.describe().round(2)
            st.dataframe(summary, use_container_width=True)
            csv = summary.to_csv()
            st.download_button("📥 Download CSV", csv, "summary_stats.csv", "text/csv")
        
        elif report_type == "Full Dataset":
            st.dataframe(df_original.head(100), use_container_width=True)
            csv = df_original.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, "full_dataset.csv", "text/csv")

# ============================================================================
# PAGE: SETTINGS
# ============================================================================
elif selected_page == "⚙️ Settings":
    st.markdown("<h1 style='color: #2E86AB;'>⚙️ Settings</h1>", unsafe_allow_html=True)
    
    new_threshold = st.slider("Risk Threshold", 0.0, 1.0, st.session_state.risk_threshold, 0.01)
    if new_threshold != st.session_state.risk_threshold:
        st.session_state.risk_threshold = new_threshold
        st.success(f"Risk threshold updated to {new_threshold:.2f}")
    
    st.markdown("---")
    st.markdown("### 📋 Required Features")
    required = st.session_state.config.get('required_features', [
        "School_Resources_Score", "Overall_Engagement_Score", "Overall_Avg_Attendance",
        "Overall_Avg_Homework", "Overall_Avg_Participation", "Teacher_Student_Ratio"
    ])
    for f in required:
        if f in st.session_state.df_processed.columns:
            st.success(f"✅ {f}")
        else:
            st.warning(f"⚠️ {f}")

st.markdown("---")
st.markdown("<p style='text-align: center;'>© 2026 Ethiopian Student Performance Dashboard | Powered by Machine Learning</p>", unsafe_allow_html=True)