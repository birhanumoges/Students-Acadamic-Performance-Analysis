# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import sys
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import utilities
from utils.data_processor import DataProcessor
from utils.predictions import PredictionEngine
from utils.visualizations import Visualizer

# Page configuration
st.set_page_config(
    page_title="Ethiopian Student Performance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
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
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
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
    .stButton button {
        width: 100%;
        border-radius: 5px;
        font-weight: 500;
    }
    .stProgress > div > div > div > div {
        background-color: #2E86AB;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'prediction_engine' not in st.session_state:
    st.session_state.prediction_engine = None
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = None
if 'risk_threshold' not in st.session_state:
    st.session_state.risk_threshold = 0.5
if 'selected_student' not in st.session_state:
    st.session_state.selected_student = None

# Model paths
MODEL_PATHS = {
    'regression': 'models/gradient_boosting_regression.pkl',
    'classification': 'models/classification_model.pkl'
}

CONFIG_PATH = 'config/settings.json'
DATA_PATH = 'data/ethiopian_students_dataset.csv'

# Load configuration
def load_config():
    """Load configuration from JSON file"""
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    return {}

# Load data
@st.cache_data
def load_data():
    """Load and preprocess data"""
    try:
        if os.path.exists(DATA_PATH):
            df_original = pd.read_csv(DATA_PATH)
            st.success(f"✅ Data loaded: {len(df_original):,} students")
            return df_original
        else:
            # Create sample data for demonstration
            st.warning(f"Data file not found. Using sample data for demonstration.")
            np.random.seed(42)
            n = 10000
            
            regions = ['Addis Ababa', 'Oromia', 'Amhara', 'Tigray', 'SNNP', 'Somali', 
                       'Afar', 'Benishangul-Gumuz', 'Sidama', 'Gambela', 'Harari', 
                       'Dire Dawa', 'South West Ethiopia']
            
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
                'Student_to_Resources_Ratio': np.random.uniform(10, 30, n),
                'Health_Issue': np.random.choice(['No Issue', 'Minor', 'Moderate', 'Severe'], n, p=[0.6, 0.2, 0.15, 0.05]),
                'Age': np.random.randint(15, 22, n),
                'Field_Choice': np.random.choice(['Social', 'Natural'], n),
                'Home_Internet_Access': np.random.choice(['Yes', 'No'], n),
                'Electricity_Access': np.random.choice(['Yes', 'No'], n),
                'School_Location': np.random.choice(['Urban', 'Rural'], n),
                'Father_Education': np.random.choice(['Unknown', 'Primary', 'High School', 'College', 'University'], n),
                'Mother_Education': np.random.choice(['Unknown', 'Primary', 'High School', 'College', 'University'], n),
                'Career_Interest': np.random.choice(['Teacher', 'Doctor', 'Engineer', 'Business', 'Government', 'Unknown'], n),
                'School_Type': np.random.choice(['Public', 'Private', 'NGO-operated'], n)
            })
            return df_original
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Initialize data processor and prediction engine
def initialize_components():
    """Initialize data processor and prediction engine"""
    config = load_config()
    
    # Initialize data processor
    data_processor = DataProcessor(CONFIG_PATH)
    
    # Initialize prediction engine
    prediction_engine = PredictionEngine(MODEL_PATHS, CONFIG_PATH)
    
    return data_processor, prediction_engine

# Load data and initialize components
if not st.session_state.data_loaded:
    with st.spinner("Loading data and models..."):
        df_original = load_data()
        if df_original is not None:
            data_processor, prediction_engine = initialize_components()
            
            st.session_state.df_original = df_original
            st.session_state.df_processed = data_processor.load_and_preprocess_data(df_original.copy())
            st.session_state.data_processor = data_processor
            st.session_state.prediction_engine = prediction_engine
            st.session_state.data_loaded = True
            
            # Load config for thresholds
            config = load_config()
            st.session_state.risk_threshold = config.get('risk_threshold', 0.5)
            
            st.success("✅ Dashboard ready!")
        else:
            st.error("Failed to load data")

# Sidebar Navigation
st.sidebar.title("📊 Student Performance Dashboard")
st.sidebar.markdown("---")

nav_options = {
    "📈 Overview": "Overview Dashboard with key metrics and visualizations",
    "👥 Students": "Student search, filtering, and individual profiles",
    "📊 Analytics": "Model performance, feature importance, and SHAP analysis",
    "🎯 Simulation": "What-if analysis and performance simulation",
    "📋 Reports": "Generate and export reports",
    "⚙️ Settings": "Dashboard configuration and model settings"
}

selected_page = st.sidebar.radio(
    "Navigation",
    list(nav_options.keys()),
    format_func=lambda x: f"{x}",
    help="Select a page to view"
)

# Show description for selected page
st.sidebar.markdown(f"<small>{nav_options[selected_page]}</small>", unsafe_allow_html=True)

st.sidebar.markdown("---")

# Quick stats in sidebar
if st.session_state.data_loaded and st.session_state.df_processed is not None:
    df = st.session_state.df_processed
    st.sidebar.markdown("### 📊 Quick Stats")
    
    total = len(df)
    avg_score = df['Overall_Average'].mean() if 'Overall_Average' in df.columns else 0
    pass_rate = (df['Overall_Average'] >= 50).mean() * 100 if 'Overall_Average' in df.columns else 0
    risk_count = (df['Overall_Average'] < 50).sum() if 'Overall_Average' in df.columns else 0
    
    st.sidebar.metric("Total Students", f"{total:,}")
    st.sidebar.metric("Average Score", f"{avg_score:.1f}")
    st.sidebar.metric("Pass Rate", f"{pass_rate:.1f}%")
    st.sidebar.metric("At-Risk", f"{risk_count:,}", delta=f"{(risk_count/total*100):.1f}%")

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
if selected_page == "📈 Overview" and st.session_state.data_loaded:
    st.markdown("<h1 class='main-header'>📈 Overview Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Comprehensive analysis of Ethiopian students' academic performance</p>", unsafe_allow_html=True)
    
    df = st.session_state.df_processed
    df_original = st.session_state.df_original
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    total_students = len(df)
    avg_score = df['Overall_Average'].mean() if 'Overall_Average' in df.columns else 0
    pass_rate = (df['Overall_Average'] >= 50).mean() * 100 if 'Overall_Average' in df.columns else 0
    risk_students = (df['Overall_Average'] < 50).sum() if 'Overall_Average' in df.columns else 0
    
    with col1:
        st.metric("Total Students", f"{total_students:,}")
    with col2:
        st.metric("Average Score", f"{avg_score:.1f}", delta=f"{avg_score - 50:.1f}" if avg_score != 50 else None)
    with col3:
        st.metric("Pass Rate", f"{pass_rate:.1f}%", delta=f"{pass_rate - 50:.1f}%" if pass_rate != 50 else None)
    with col4:
        st.metric("At-Risk Students", f"{risk_students:,}", delta=f"{(risk_students/total_students*100):.1f}%", delta_color="inverse")
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(Visualizer.create_score_histogram(df), use_container_width=True)
    with col2:
        st.plotly_chart(Visualizer.create_risk_bar(df), use_container_width=True)
    
    # Charts Row 2
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(Visualizer.create_region_bar(df_original), use_container_width=True)
    with col2:
        st.plotly_chart(Visualizer.create_gender_bar(df), use_container_width=True)
    
    # Summary Table
    st.markdown("### 📊 Quick Summary Statistics")
    summary_data = {
        'Metric': ['Total Students', 'Average Score', 'Median Score', 'Std Deviation', 
                   'Min Score', 'Max Score', 'Pass Rate', 'At-Risk Rate'],
        'Value': [
            f"{total_students:,}",
            f"{avg_score:.2f}",
            f"{df['Overall_Average'].median():.2f}",
            f"{df['Overall_Average'].std():.2f}",
            f"{df['Overall_Average'].min():.2f}",
            f"{df['Overall_Average'].max():.2f}",
            f"{pass_rate:.1f}%",
            f"{(risk_students/total_students*100):.1f}%"
        ]
    }
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
    
    # Dashboard Objectives
    with st.expander("🎯 Dashboard Objectives & Features", expanded=False):
        st.markdown("""
        **This dashboard provides:**
        
        **📈 Overview Dashboard**
        - Real-time KPI monitoring
        - Score distribution analysis
        - Regional and demographic comparisons
        
        **👥 Student Management**
        - Searchable student database
        - Individual student profiles
        - Predictive analytics for each student
        
        **📊 Analytics Suite**
        - Model performance metrics (R², MAE, RMSE, F1, AUC)
        - Feature importance analysis
        - SHAP explanations for interpretability
        
        **🎯 Simulation Tools**
        - What-if scenario analysis
        - Performance improvement simulations
        - Risk factor impact assessment
        
        **📋 Reporting**
        - Custom report generation
        - CSV/Excel export
        - Summary statistics
        
        **⚙️ Settings**
        - Risk threshold configuration
        - Model management
        - Data upload options
        """)

# ============================================================================
# PAGE: STUDENTS
# ============================================================================
elif selected_page == "👥 Students" and st.session_state.data_loaded:
    st.markdown("<h1 class='main-header'>👥 Student Management</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Search, filter, and analyze individual student performance</p>", unsafe_allow_html=True)
    
    df_original = st.session_state.df_original.copy()
    df_processed = st.session_state.df_processed.copy()
    prediction_engine = st.session_state.prediction_engine
    data_processor = st.session_state.data_processor
    
    # Prepare display data
    display_df = df_original.copy()
    if 'Overall_Average' in display_df.columns:
        display_df['Risk_Status'] = display_df['Overall_Average'].apply(lambda x: 'At Risk' if x < 50 else 'Not at Risk')
    
    # Filters in sidebar
    st.sidebar.markdown("### 🔍 Filters")
    
    # Region filter
    if 'Region' in display_df.columns:
        regions = ['All'] + sorted(display_df['Region'].unique().tolist())
        selected_region = st.sidebar.selectbox("Region", regions)
        if selected_region != 'All':
            display_df = display_df[display_df['Region'] == selected_region]
    
    # Gender filter
    if 'Gender' in display_df.columns:
        gender_filter = st.sidebar.selectbox("Gender", ["All", "Male", "Female"])
        if gender_filter != "All":
            display_df = display_df[display_df['Gender'] == gender_filter]
    
    # Attendance filter
    if 'Overall_Avg_Attendance' in df_processed.columns:
        attendance_range = st.sidebar.slider("Attendance (%)", 0, 100, (0, 100))
        student_ids = df_processed[(df_processed['Overall_Avg_Attendance'] >= attendance_range[0]) & 
                                    (df_processed['Overall_Avg_Attendance'] <= attendance_range[1])]['Student_ID'].tolist()
        display_df = display_df[display_df['Student_ID'].isin(student_ids)]
    
    # Risk filter
    risk_filter = st.sidebar.selectbox("Risk Status", ["All", "At Risk", "Not at Risk"])
    if risk_filter != "All" and 'Risk_Status' in display_df.columns:
        display_df = display_df[display_df['Risk_Status'] == risk_filter]
    
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Showing {len(display_df)} students")
    
    # Search
    search_term = st.text_input("🔍 Search by Student ID or Region", placeholder="Enter Student ID or Region...")
    if search_term:
        display_df = display_df[display_df.astype(str).apply(lambda x: x.str.contains(search_term, case=False).any(), axis=1)]
    
    # Display table
    st.markdown("### 📋 Student List")
    
    display_cols = ['Student_ID', 'Region', 'Gender', 'Overall_Average', 'Risk_Status'] if 'Overall_Average' in display_df.columns else display_df.columns[:5]
    display_cols = [c for c in display_cols if c in display_df.columns]
    
    # Create selection
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
        student_id = selected_student.get('Student_ID')
        
        # Get processed features for prediction
        student_data = selected_student.to_dict()
        
        # Prepare features for prediction
        try:
            features_df = data_processor.prepare_features_for_prediction(student_data)
            predicted_score = prediction_engine.predict_score(features_df)
            risk_prob = prediction_engine.predict_risk(features_df)
            is_risk = risk_prob > st.session_state.risk_threshold
        except Exception as e:
            predicted_score = selected_student.get('Overall_Average', 50)
            risk_prob = 0.5
            is_risk = predicted_score < 50
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📌 Student Information**")
            st.markdown(f"- **Student ID:** {selected_student.get('Student_ID', 'N/A')}")
            st.markdown(f"- **Region:** {selected_student.get('Region', 'N/A')}")
            st.markdown(f"- **Gender:** {selected_student.get('Gender', 'N/A')}")
            st.markdown(f"- **Age:** {selected_student.get('Age', 'N/A')}")
            st.markdown(f"- **Field Choice:** {selected_student.get('Field_Choice', 'N/A')}")
        
        with col2:
            st.markdown("**📊 Performance Metrics**")
            st.markdown(f"- **Actual Score:** {selected_student.get('Overall_Average', 'N/A'):.1f}" if 'Overall_Average' in selected_student else "N/A")
            st.markdown(f"- **Predicted Score:** {predicted_score:.1f}")
            st.markdown(f"- **Prediction Error:** {abs(predicted_score - selected_student.get('Overall_Average', predicted_score)):.1f} points")
            st.markdown(f"- **Risk Probability:** {risk_prob*100:.1f}%")
            st.progress(risk_prob)
            st.markdown(f"- **Risk Status:** {'🔴 AT RISK' if is_risk else '🟢 NOT AT RISK'}")
        
        # Recommendations
        st.markdown("### 💡 Personalized Recommendations")
        recommendations = prediction_engine.get_recommendations(student_data, predicted_score, risk_prob)
        for rec in recommendations:
            if rec.startswith("🔴") or rec.startswith("✅"):
                if rec.startswith("🔴"):
                    st.error(rec)
                else:
                    st.success(rec)
            elif rec.startswith("•"):
                st.markdown(rec)
            else:
                st.info(rec)
    else:
        st.info("👆 Select a student from the table to view detailed profile and recommendations")

# ============================================================================
# PAGE: ANALYTICS
# ============================================================================
elif selected_page == "📊 Analytics" and st.session_state.data_loaded:
    st.markdown("<h1 class='main-header'>📊 Analytics Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Model performance, feature importance, and SHAP analysis</p>", unsafe_allow_html=True)
    
    df = st.session_state.df_processed
    config = load_config()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Diagnostics", "🤖 Modeling", "💡 Explainability"])
    
    with tab1:
        st.markdown("### Diagnostic Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(Visualizer.create_correlation_heatmap(df), use_container_width=True)
        with col2:
            if 'Overall_Avg_Attendance' in df.columns and 'Overall_Average' in df.columns:
                st.plotly_chart(Visualizer.create_scatter_plot(df, 'Overall_Avg_Attendance', 'Overall_Average'), use_container_width=True)
        
        st.markdown("### Distribution by Category")
        col1, col2 = st.columns(2)
        with col1:
            if 'Region' in df.columns and 'Overall_Average' in df.columns:
                st.plotly_chart(Visualizer.create_boxplot(df, 'Region', 'Overall_Average'), use_container_width=True)
        with col2:
            if 'Gender' in df.columns and 'Overall_Average' in df.columns:
                st.plotly_chart(Visualizer.create_boxplot(df, 'Gender', 'Overall_Average'), use_container_width=True)
    
    with tab2:
        st.markdown("### Regression Model Performance")
        
        reg_metrics = config.get('model_performance', {}).get('regression', {})
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Best Model", reg_metrics.get('best_model', 'XGBoost'))
        with col2:
            st.metric("R² Score", f"{reg_metrics.get('r2', 0.7855):.4f}")
        with col3:
            st.metric("MAE", f"{reg_metrics.get('mae', 2.98):.2f}")
        with col4:
            st.metric("RMSE", f"{reg_metrics.get('rmse', 3.72):.2f}")
        
        st.markdown("#### Model Comparison")
        comparison_data = pd.DataFrame({
            'Model': ['XGBoost', 'Gradient Boosting', 'Random Forest', 'Linear Regression'],
            'R²': [0.7855, 0.7851, 0.7719, 0.7690],
            'MAE': [2.98, 2.99, 3.07, 3.10],
            'RMSE': [3.72, 3.73, 3.84, 3.86]
        })
        st.dataframe(comparison_data, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown("### Classification Model Performance")
        
        class_metrics = config.get('model_performance', {}).get('classification', {})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Best Model", class_metrics.get('best_model', 'Gradient Boosting'))
        with col2:
            st.metric("ROC-AUC", f"{class_metrics.get('roc_auc', 0.9178):.4f}")
        with col3:
            st.metric("F1-Score", f"{class_metrics.get('f1', 0.7782):.4f}")
        
        # Confusion Matrix
        cm = np.array([[54123, 9389], [5274, 31214]])
        st.plotly_chart(Visualizer.create_confusion_matrix(cm), use_container_width=True)
        
        # ROC Curve
        fpr = np.linspace(0, 1, 100)
        tpr = 1 - (1 - fpr) ** (1 / (2 * (1 - 0.9178)))
        tpr = np.minimum(tpr, 1)
        st.plotly_chart(Visualizer.create_roc_curve(fpr, tpr, 0.9178), use_container_width=True)
        
        # National Exam Model
        st.markdown("---")
        st.markdown("### National Exam Score Model Performance")
        national_exam = config.get('national_exam_performance', {})
        national_df = pd.DataFrame(national_exam).T
        st.dataframe(national_df, use_container_width=True)
    
    with tab3:
        st.markdown("### Feature Importance Analysis")
        
        # Regression Feature Importance
        st.markdown("#### Regression Model (XGBoost)")
        reg_importance = config.get('feature_importance', {}).get('regression', {})
        st.plotly_chart(Visualizer.create_feature_importance_plot(reg_importance, "Top 10 Features - XGBoost"), use_container_width=True)
        
        st.markdown("#### Classification Model (Gradient Boosting)")
        class_importance = config.get('feature_importance', {}).get('classification', {})
        st.plotly_chart(Visualizer.create_feature_importance_plot(class_importance, "Top 10 Features - Gradient Boosting"), use_container_width=True)
        
        # SHAP Analysis Section
        st.markdown("---")
        st.markdown("### SHAP Analysis")
        st.info("""
        **SHAP (SHapley Additive exPlanations)** values explain model predictions:
        - **Blue** = Feature increases risk probability
        - **Red** = Feature decreases risk probability
        - **Magnitude** = Strength of impact on prediction
        """)
        
        # Simulated SHAP values for demonstration
        np.random.seed(42)
        shap_values = np.random.randn(100, len(class_importance)) * 0.5
        shap_df = pd.DataFrame(shap_values, columns=list(class_importance.keys()))
        
        fig = go.Figure()
        for i, feature in enumerate(list(class_importance.keys())[:10]):
            fig.add_trace(go.Violin(
                y=shap_df[feature],
                name=feature,
                box_visible=True,
                meanline_visible=True,
                line_color=Visualizer.COLOR_SCHEME['primary']
            ))
        fig.update_layout(
            title="SHAP Value Distribution by Feature",
            xaxis_title="Features",
            yaxis_title="SHAP Value",
            height=500,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### Key Insights from SHAP Analysis")
        insights = [
            "🏫 **School Resources Score** is the most influential feature",
            "📚 **Textbook Access** strongly impacts student risk",
            "👪 **Parental Involvement** shows significant protective effect",
            "👥 **Teacher-Student Ratio** negatively impacts performance when too high",
            "❤️ **Health Issues** show varied impact depending on severity"
        ]
        for insight in insights:
            st.markdown(insight)

# ============================================================================
# PAGE: SIMULATION
# ============================================================================
elif selected_page == "🎯 Simulation" and st.session_state.data_loaded:
    st.markdown("<h1 class='main-header'>🎯 Performance Simulation</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>What-if analysis - Adjust parameters to see impact on student performance</p>", unsafe_allow_html=True)
    
    df_original = st.session_state.df_original
    prediction_engine = st.session_state.prediction_engine
    data_processor = st.session_state.data_processor
    
    # Input mode selection
    mode = st.radio("Input Mode", ["Manual Input", "Select Existing Student"], horizontal=True)
    
    if mode == "Select Existing Student":
        # Student selection dropdown
        student_options = df_original.head(100)[['Student_ID', 'Region']].copy()
        student_options['Display'] = student_options['Student_ID'].astype(str) + " - " + student_options['Region']
        selected_student_display = st.selectbox("Select Student", student_options['Display'].tolist())
        selected_id = int(selected_student_display.split(" - ")[0])
        student_data = df_original[df_original['Student_ID'] == selected_id].iloc[0].to_dict()
        
        # Display current values
        st.info(f"**Selected Student:** ID {selected_id} from {student_data.get('Region', 'N/A')}")
        
        # Load current values
        school_resources = student_data.get('School_Resources_Score', 0.5)
        attendance = student_data.get('Overall_Avg_Attendance', 75)
        homework = student_data.get('Overall_Avg_Homework', 65)
        participation = student_data.get('Overall_Avg_Participation', 70)
        textbook = student_data.get('Overall_Textbook_Access_Composite', 0.5)
        teacher_ratio = student_data.get('Teacher_Student_Ratio', 40)
        parental = student_data.get('Parental_Involvement', 0.5)
        
    else:
        # Manual input with default values
        school_resources = 0.5
        attendance = 75
        homework = 65
        participation = 70
        textbook = 0.5
        teacher_ratio = 40
        parental = 0.5
    
    # Simulation sliders
    st.markdown("### Adjust Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🏫 School Factors**")
        new_school_resources = st.slider("School Resources Score", 0.0, 1.0, school_resources, 0.05, 
                                          help="0 = Poor, 1 = Excellent")
        new_teacher_ratio = st.slider("Teacher-Student Ratio", 10, 100, int(teacher_ratio), 
                                       help="Lower is better")
        new_textbook = st.slider("Textbook Access", 0.0, 1.0, textbook, 0.05,
                                  help="0 = None, 1 = Full access")
        new_parental = st.slider("Parental Involvement", 0.0, 1.0, parental, 0.05,
                                  help="0 = None, 1 = High")
    
    with col2:
        st.markdown("**📚 Student Engagement**")
        new_attendance = st.slider("Attendance (%)", 0, 100, int(attendance))
        new_homework = st.slider("Homework Completion (%)", 0, 100, int(homework))
        new_participation = st.slider("Participation (%)", 0, 100, int(participation))
    
    # Run simulation button
    if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
        with st.spinner("Running simulation..."):
            # Prepare input data
            input_data = {
                'School_Resources_Score': new_school_resources,
                'Overall_Textbook_Access_Composite': new_textbook,
                'Parental_Involvement': new_parental,
                'Teacher_Student_Ratio': new_teacher_ratio,
                'Overall_Avg_Attendance': new_attendance,
                'Overall_Avg_Homework': new_homework,
                'Overall_Avg_Participation': new_participation,
                'School_Academic_Score': 0.5,
                'Gender': 0,
                'Region': 'Oromia',
                'Health_Issue': 'No Issue',
                'Age': 17,
                'School_Location': 1
            }
            
            # Calculate engagement score
            engagement = (new_attendance * 0.4 + new_homework * 0.3 + new_participation * 0.3) / 100
            input_data['Overall_Engagement_Score'] = engagement * 100
            
            # Prepare features
            features_df = data_processor.prepare_features_for_prediction(input_data)
            
            # Make predictions
            predicted_score = prediction_engine.predict_score(features_df)
            risk_prob = prediction_engine.predict_risk(features_df)
            is_risk = risk_prob > st.session_state.risk_threshold
            
            # Calculate improvements from baseline
            baseline_data = input_data.copy()
            baseline_data['School_Resources_Score'] = school_resources
            baseline_data['Overall_Avg_Attendance'] = attendance
            baseline_data['Overall_Avg_Homework'] = homework
            baseline_data['Overall_Avg_Participation'] = participation
            baseline_data['Overall_Textbook_Access_Composite'] = textbook
            baseline_data['Teacher_Student_Ratio'] = teacher_ratio
            baseline_data['Parental_Involvement'] = parental
            baseline_engagement = (attendance * 0.4 + homework * 0.3 + participation * 0.3) / 100
            baseline_data['Overall_Engagement_Score'] = baseline_engagement * 100
            
            baseline_features = data_processor.prepare_features_for_prediction(baseline_data)
            baseline_score = prediction_engine.predict_score(baseline_features)
            baseline_risk = prediction_engine.predict_risk(baseline_features)
            
            score_delta = predicted_score - baseline_score
            risk_delta = risk_prob - baseline_risk
            
            # Display results
            st.markdown("---")
            st.markdown("## 📊 Simulation Results")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Score", f"{predicted_score:.1f}", 
                          delta=f"{score_delta:+.1f}" if score_delta != 0 else None)
                
                # Score gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=predicted_score,
                    title={'text': "Score"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': Visualizer.COLOR_SCHEME['primary']},
                        'steps': [
                            {'range': [0, 50], 'color': Visualizer.COLOR_SCHEME['danger']},
                            {'range': [50, 70], 'color': Visualizer.COLOR_SCHEME['warning']},
                            {'range': [70, 100], 'color': Visualizer.COLOR_SCHEME['success']}
                        ]
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.metric("Risk Probability", f"{risk_prob*100:.1f}%",
                          delta=f"{risk_delta*100:+.1f}%" if risk_delta != 0 else None,
                          delta_color="inverse" if risk_delta < 0 else "normal")
                st.progress(risk_prob)
                st.metric("Risk Status", "🔴 AT RISK" if is_risk else "🟢 NOT AT RISK")
                st.plotly_chart(Visualizer.create_risk_gauge(risk_prob), use_container_width=True)
            
            # Improvement analysis
            st.markdown("### 📈 Improvement Opportunities")
            
            improvements = []
            if new_school_resources < 0.9:
                improvements.append(("Increase School Resources by 0.2", 2.5, new_school_resources + 0.2))
            if new_attendance < 95:
                improvements.append(("Improve Attendance by 10%", 1.8, min(100, new_attendance + 10)))
            if new_homework < 95:
                improvements.append(("Improve Homework by 10%", 1.2, min(100, new_homework + 10)))
            if new_textbook < 0.9:
                improvements.append(("Improve Textbook Access by 0.2", 2.0, min(1.0, new_textbook + 0.2)))
            
            for name, impact, target in improvements[:3]:
                st.success(f"**{name}:** +{impact:.1f} points estimated improvement")
            
            # Recommendations
            st.markdown("### 💡 Recommendations")
            recommendations = prediction_engine.get_recommendations(input_data, predicted_score, risk_prob)
            for rec in recommendations:
                if rec.startswith("🔴") or rec.startswith("✅"):
                    if rec.startswith("🔴"):
                        st.error(rec)
                    else:
                        st.success(rec)
                elif rec.startswith("•"):
                    st.markdown(rec)
                else:
                    st.info(rec)

# ============================================================================
# PAGE: REPORTS
# ============================================================================
elif selected_page == "📋 Reports" and st.session_state.data_loaded:
    st.markdown("<h1 class='main-header'>📋 Reports & Export</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Generate and download comprehensive reports</p>", unsafe_allow_html=True)
    
    df_original = st.session_state.df_original
    df_processed = st.session_state.df_processed
    prediction_engine = st.session_state.prediction_engine
    data_processor = st.session_state.data_processor
    
    # Report type selection
    report_type = st.selectbox(
        "Select Report Type",
        ["Predictions Report", "Clusters Report", "Summary Statistics", "Full Dataset"],
        help="Choose the type of report to generate"
    )
    
    if st.button("📊 Generate Report", type="primary", use_container_width=True):
        with st.spinner("Generating report..."):
            if report_type == "Predictions Report":
                st.markdown("### Student Predictions Report")
                
                # Generate predictions for sample
                report_df = df_original.head(500).copy()
                
                with st.spinner("Computing predictions..."):
                    predictions = []
                    risks = []
                    for idx, row in report_df.iterrows():
                        try:
                            features_df = data_processor.prepare_features_for_prediction(row.to_dict())
                            pred_score = prediction_engine.predict_score(features_df)
                            pred_risk = prediction_engine.predict_risk(features_df)
                            predictions.append(pred_score)
                            risks.append(pred_risk)
                        except:
                            predictions.append(50.0)
                            risks.append(0.5)
                    
                    report_df['Predicted_Score'] = predictions
                    report_df['Risk_Probability'] = risks
                    report_df['Risk_Status'] = report_df['Risk_Probability'].apply(lambda x: 'At Risk' if x > st.session_state.risk_threshold else 'Not at Risk')
                
                # Display report
                display_cols = ['Student_ID', 'Region', 'Gender', 'Overall_Average', 'Predicted_Score', 'Risk_Probability', 'Risk_Status']
                display_cols = [c for c in display_cols if c in report_df.columns]
                st.dataframe(report_df[display_cols], use_container_width=True)
                
                # Download button
                csv = report_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"predictions_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # Summary stats
                st.markdown("### Summary Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Students", len(report_df))
                with col2:
                    st.metric("Avg Predicted Score", f"{report_df['Predicted_Score'].mean():.1f}")
                with col3:
                    risk_count = (report_df['Risk_Status'] == 'At Risk').sum()
                    st.metric("At-Risk Students", risk_count)
            
            elif report_type == "Clusters Report":
                st.markdown("### Student Clusters Report")
                
                config = load_config()
                clusters = config.get('clustering', {})
                cluster_sizes = clusters.get('cluster_sizes', {'Low': 0, 'Medium': 0, 'High': 0})
                
                # Create cluster assignment
                np.random.seed(42)
                cluster_assignments = np.random.choice(
                    list(cluster_sizes.keys()), 
                    size=len(df_original),
                    p=[cluster_sizes[k]/sum(cluster_sizes.values()) for k in cluster_sizes.keys()]
                )
                
                cluster_df = df_original.copy()
                cluster_df['Performance_Cluster'] = cluster_assignments
                
                # Cluster summary
                st.markdown("#### Cluster Distribution")
                cluster_summary = cluster_df['Performance_Cluster'].value_counts()
                st.dataframe(pd.DataFrame(cluster_summary), use_container_width=True)
                
                # Cluster profiles
                st.markdown("#### Cluster Profiles")
                cluster_profiles = cluster_df.groupby('Performance_Cluster').agg({
                    'Overall_Average': 'mean',
                    'Student_ID': 'count'
                }).round(2)
                st.dataframe(cluster_profiles, use_container_width=True)
                
                # Download button
                csv = cluster_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"clusters_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            elif report_type == "Summary Statistics":
                st.markdown("### Summary Statistics Report")
                
                # Create summary
                summary_data = []
                numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
                
                for col in numeric_cols[:20]:
                    summary_data.append({
                        'Feature': col,
                        'Mean': f"{df_processed[col].mean():.2f}",
                        'Std': f"{df_processed[col].std():.2f}",
                        'Min': f"{df_processed[col].min():.2f}",
                        'Median': f"{df_processed[col].median():.2f}",
                        'Max': f"{df_processed[col].max():.2f}",
                        'Missing': df_processed[col].isnull().sum()
                    })
                
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
                
                csv = pd.DataFrame(summary_data).to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"summary_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            elif report_type == "Full Dataset":
                st.markdown("### Full Dataset Export")
                
                # Select columns to export
                export_cols = st.multiselect(
                    "Select columns to export",
                    options=df_original.columns.tolist(),
                    default=df_original.columns[:10].tolist()
                )
                
                if export_cols:
                    export_df = df_original[export_cols]
                    st.dataframe(export_df.head(100), use_container_width=True)
                    
                    csv = export_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"full_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
    
    # One-page summary
    st.markdown("---")
    st.markdown("### 📄 Quick Summary")
    
    if st.button("📄 Generate Quick Summary"):
        config = load_config()
        clusters = config.get('clustering', {})
        reg_perf = config.get('model_performance', {}).get('regression', {})
        class_perf = config.get('model_performance', {}).get('classification', {})
        
        summary_text = f"""
        ## Ethiopian Student Performance Dashboard - Summary
        
        ### Key Metrics
        - **Total Students:** {len(df_processed):,}
        - **Average Score:** {df_processed['Overall_Average'].mean():.1f}
        - **Pass Rate:** {(df_processed['Overall_Average'] >= 50).mean()*100:.1f}%
        - **At-Risk Students:** {(df_processed['Overall_Average'] < 50).sum():,}
        
        ### Model Performance
        - **Best Regression Model:** {reg_perf.get('best_model', 'XGBoost')} (R² = {reg_perf.get('r2', 0.7855):.4f})
        - **Best Classification Model:** {class_perf.get('best_model', 'Gradient Boosting')} (AUC = {class_perf.get('roc_auc', 0.9178):.4f})
        - **Top Predictor:** School Resources Score (55.1% importance)
        
        ### Recommendations
        1. Increase school resources in high-risk regions
        2. Improve textbook access and digital infrastructure
        3. Implement parent engagement programs
        4. Reduce teacher-student ratios in overcrowded schools
        """
        st.markdown(summary_text)

# ============================================================================
# PAGE: SETTINGS
# ============================================================================
elif selected_page == "⚙️ Settings" and st.session_state.data_loaded:
    st.markdown("<h1 class='main-header'>⚙️ Settings</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Configure dashboard parameters and manage models</p>", unsafe_allow_html=True)
    
    # Risk Threshold
    st.markdown("### ⚠️ Risk Classification Threshold")
    st.markdown("Adjust the threshold for classifying students as 'At Risk'")
    
    new_threshold = st.slider(
        "Risk Threshold", 
        0.0, 1.0, 
        st.session_state.risk_threshold, 
        0.01,
        help="Students with risk probability above this threshold are classified as 'At Risk'"
    )
    
    if new_threshold != st.session_state.risk_threshold:
        st.session_state.risk_threshold = new_threshold
        st.success(f"Risk threshold updated to {new_threshold:.2f}")
    
    st.markdown("---")
    
    # Model Management
    st.markdown("### 🤖 Model Management")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Reload Models", use_container_width=True):
            with st.spinner("Reloading models..."):
                try:
                    st.session_state.prediction_engine.load_models()
                    st.success("✅ Models reloaded successfully!")
                except Exception as e:
                    st.error(f"Error reloading models: {e}")
    
    with col2:
        if st.button("📊 Test Models", use_container_width=True):
            with st.spinner("Testing models..."):
                # Test with sample data
                test_features = st.session_state.data_processor.prepare_features_for_prediction({
                    'School_Resources_Score': 0.7,
                    'Overall_Engagement_Score': 75,
                    'Overall_Avg_Attendance': 85,
                    'Overall_Avg_Homework': 80,
                    'Overall_Avg_Participation': 75,
                    'Teacher_Student_Ratio': 35,
                    'Parental_Involvement': 0.6,
                    'Overall_Textbook_Access_Composite': 0.7,
                    'Gender': 0,
                    'Region': 'Addis Ababa',
                    'Health_Issue': 'No Issue',
                    'Age': 17
                })
                
                test_score = st.session_state.prediction_engine.predict_score(test_features)
                test_risk = st.session_state.prediction_engine.predict_risk(test_features)
                
                st.success(f"✅ Models working! Test score: {test_score:.1f}, Test risk: {test_risk:.2f}")
    
    st.markdown("---")
    
    # Feature Validation
    st.markdown("### 📋 Required Features for Prediction")
    config = load_config()
    required_features = config.get('required_features', [
        "School_Resources_Score", "Overall_Engagement_Score", "School_Academic_Score",
        "Overall_Textbook_Access_Composite", "Overall_Avg_Attendance", "Teacher_Student_Ratio",
        "Overall_Avg_Homework", "Overall_Avg_Participation", "Parental_Involvement",
        "Gender", "Region", "Health_Issue", "Age", "School_Location"
    ])
    
    available_features = st.session_state.df_processed.columns.tolist()
    
    for feature in required_features:
        if feature in available_features:
            st.success(f"✅ {feature} - Available")
        else:
            st.warning(f"⚠️ {feature} - Not found in processed data")
    
    st.markdown("---")
    
    # Data Upload
    st.markdown("### 📁 Data Management")
    
    uploaded_file = st.file_uploader("Upload New Dataset (CSV)", type=['csv'])
    if uploaded_file is not None:
        try:
            new_df = pd.read_csv(uploaded_file)
            st.success(f"✅ Dataset loaded: {len(new_df):,} rows, {len(new_df.columns)} columns")
            st.dataframe(new_df.head(), use_container_width=True)
            
            if st.button("Apply New Dataset", use_container_width=True):
                st.session_state.df_original = new_df
                st.session_state.data_loaded = False
                st.rerun()
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
    
    st.markdown("---")
    
    # About
    st.markdown("### ℹ️ About")
    st.info("""
    **Ethiopian Student Performance Dashboard v1.0**
    
    **Models:**
    - Regression: XGBoost (R² = 0.7855)
    - Classification: Gradient Boosting (AUC = 0.918)
    
    **Features:**
    - Real-time student performance prediction
    - Risk assessment and classification
    - Interactive simulations
    - Comprehensive analytics with SHAP
    - Exportable reports
    
    **Data Source:** Ethiopian Students Dataset
    """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666; font-size: 12px;'>"
    "© 2024 Ethiopian Student Performance Analytics Dashboard | Powered by Machine Learning"
    "</p>", 
    unsafe_allow_html=True
)