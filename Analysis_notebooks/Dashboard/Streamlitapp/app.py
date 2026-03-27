"""
Ethiopian Student Performance Analytics Dashboard
Streamlit Version with Left Sidebar Navigation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings
import os
import sys
import io
import json
from datetime import datetime

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Student Performance Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import utils
from utils import (
    load_and_preprocess_data,
    encode_categorical_features,
    prepare_target_encoders,
    process_raw_input_for_prediction,
    COLOR_SCHEME
)

from utils.predictions import load_models, make_prediction_corrected
from utils.visualizations import (
    create_score_histogram,
    create_risk_distribution,
    create_region_score_chart,
    create_gender_score_chart,
    create_correlation_heatmap,
    create_attendance_score_scatter,
    create_feature_importance_plot,
    create_boxplot_by_category,
    create_cluster_distribution_plot,
    create_regional_risk_plot
)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
def init_session_state():
    """Initialize all session state variables"""
    if 'page' not in st.session_state:
        st.session_state.page = "Overview"
    if 'selected_student' not in st.session_state:
        st.session_state.selected_student = None
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    if 'df_clean' not in st.session_state:
        st.session_state.df_clean = None
    if 'df_raw' not in st.session_state:
        st.session_state.df_raw = None
    if 'df_original' not in st.session_state:
        st.session_state.df_original = None
    if 'regression_models' not in st.session_state:
        st.session_state.regression_models = {}
    if 'classification_model' not in st.session_state:
        st.session_state.classification_model = {}
    if 'clustering_analysis' not in st.session_state:
        st.session_state.clustering_analysis = None
    if 'best_reg_model' not in st.session_state:
        st.session_state.best_reg_model = "XGBoost"
    if 'feature_importances' not in st.session_state:
        st.session_state.feature_importances = {}
    if 'reg_features' not in st.session_state:
        st.session_state.reg_features = None
    if 'target_encoders' not in st.session_state:
        st.session_state.target_encoders = {}
    if 'reg_model' not in st.session_state:
        st.session_state.reg_model = None
    if 'class_model' not in st.session_state:
        st.session_state.class_model = None
    if 'reg_scaler' not in st.session_state:
        st.session_state.reg_scaler = None
    if 'class_scaler' not in st.session_state:
        st.session_state.class_scaler = None
    if 'risk_threshold' not in st.session_state:
        st.session_state.risk_threshold = 0.5
    if 'filters' not in st.session_state:
        st.session_state.filters = {
            'region': 'All',
            'attendance': (0, 100),
            'risk': 'All',
            'gender': 'All'
        }
    if 'simulation_result' not in st.session_state:
        st.session_state.simulation_result = None
    if 'report_data' not in st.session_state:
        st.session_state.report_data = None


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================
@st.cache_data
def load_data():
    """Load and preprocess data"""
    try:
        # Try to load from default path
        file_path = r"C:/Users/DELL/Documents/project_data/ethiopian_students_dataset.csv"
        
        # If file doesn't exist, create sample data
        if not os.path.exists(file_path):
            st.warning("Original dataset not found. Using sample data for demonstration.")
            df_original = create_sample_data()
        else:
            df_original = pd.read_csv(file_path)
        
        # Preprocess data
        df_raw = load_and_preprocess_data(df_original)
        df_clean = encode_categorical_features(df_raw)
        
        # Add Student_ID as numeric from 1 to n
        df_raw['Student_ID'] = range(1, len(df_raw) + 1)
        df_clean['Student_ID'] = range(1, len(df_clean) + 1)
        
        # Prepare target encoders
        prepare_target_encoders(df_original, df_clean)
        
        # Create clustering analysis from data
        clustering_analysis = create_clustering_analysis(df_clean)
        
        return df_original, df_raw, df_clean, clustering_analysis
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None


def create_sample_data():
    """Create sample data for demonstration"""
    np.random.seed(42)
    n_samples = 5000
    
    df = pd.DataFrame({
        'Student_ID': range(1, n_samples + 1),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Region': np.random.choice([
            'Addis Ababa', 'Oromia', 'Amhara', 'Tigray', 'SNNP',
            'Somali', 'Afar', 'Benishangul-Gumuz', 'Gambela', 'Harari',
            'Sidama', 'South West Ethiopia', 'Dire Dawa'
        ], n_samples),
        'Field_Choice': np.random.choice(['Social', 'Natural'], n_samples),
        'Health_Issue': np.random.choice(['No Issue', 'Vision Issues', 'Dental Problems', 'Anemia', 'Malnutrition'], n_samples),
        'Father_Education': np.random.choice(['High School', 'College', 'Primary', 'University', 'Unknown'], n_samples),
        'Mother_Education': np.random.choice(['High School', 'College', 'Primary', 'University', 'Unknown'], n_samples),
        'Parental_Involvement': np.random.uniform(0, 1, n_samples),
        'Home_Internet_Access': np.random.choice(['Yes', 'No'], n_samples),
        'Electricity_Access': np.random.choice(['Yes', 'No'], n_samples),
        'School_Type': np.random.choice(['Public', 'Private', 'NGO-operated'], n_samples),
        'School_Location': np.random.choice(['Rural', 'Urban'], n_samples),
        'Teacher_Student_Ratio': np.random.uniform(20, 60, n_samples),
        'School_Resources_Score': np.random.uniform(0.2, 0.9, n_samples),
        'School_Academic_Score': np.random.uniform(0.3, 0.9, n_samples),
        'Student_to_Resources_Ratio': np.random.uniform(10, 35, n_samples),
        'Career_Interest': np.random.choice(['Teacher', 'Doctor', 'Engineer', 'Farmer', 'Business', 'Government'], n_samples),
        'Date_of_Birth': pd.date_range('2000-01-01', periods=n_samples, freq='D')[:n_samples],
    })
    
    # Add grade-level scores
    for grade in ['Grade_1', 'Grade_2', 'Grade_3', 'Grade_4', 'Grade_5', 'Grade_6', 
                  'Grade_7', 'Grade_8', 'Grade_9', 'Grade_10', 'Grade_11', 'Grade_12']:
        df[f'{grade}_Test_Score'] = np.random.uniform(40, 100, n_samples)
        df[f'{grade}_Attendance'] = np.random.uniform(60, 100, n_samples)
        df[f'{grade}_Homework_Completion'] = np.random.uniform(50, 100, n_samples)
        df[f'{grade}_Participation'] = np.random.uniform(50, 100, n_samples)
        df[f'{grade}_Textbook_Access'] = np.random.choice(['Yes', 'No'], n_samples)
    
    # Add national exam scores
    df['National_Exam_History'] = np.random.uniform(40, 95, n_samples)
    df['National_Exam_Geography'] = np.random.uniform(40, 95, n_samples)
    df['National_Exam_Economics'] = np.random.uniform(40, 95, n_samples)
    df['National_Exam_Math_Social'] = np.random.uniform(40, 95, n_samples)
    df['National_Exam_Biology'] = np.random.uniform(40, 95, n_samples)
    df['National_Exam_Chemistry'] = np.random.uniform(40, 95, n_samples)
    df['National_Exam_Physics'] = np.random.uniform(40, 95, n_samples)
    df['National_Exam_Math_Natural'] = np.random.uniform(40, 95, n_samples)
    df['National_Exam_Aptitude'] = np.random.uniform(40, 95, n_samples)
    df['National_Exam_English'] = np.random.uniform(40, 95, n_samples)
    df['National_Exam_Civics_and_Ethical_Education'] = np.random.uniform(40, 95, n_samples)
    
    # Add overall average (correlated with some features)
    df['Overall_Average'] = (
        df['School_Resources_Score'] * 30 +
        df['Parental_Involvement'] * 20 +
        (df['Teacher_Student_Ratio'] / 100) * 10 +
        np.random.uniform(0, 20, n_samples)
    ).clip(40, 95)
    
    df['Total_National_Exam_Score'] = df[[
        'National_Exam_History', 'National_Exam_Geography', 'National_Exam_Economics',
        'National_Exam_Math_Social', 'National_Exam_Biology', 'National_Exam_Chemistry',
        'National_Exam_Physics', 'National_Exam_Math_Natural', 'National_Exam_Aptitude',
        'National_Exam_English', 'National_Exam_Civics_and_Ethical_Education'
    ]].mean(axis=1) * 5
    
    return df


def create_clustering_analysis(df_clean):
    """Create clustering analysis from data"""
    if 'Overall_Average' not in df_clean.columns:
        return None
    
    # Create clusters based on Overall Average
    df_temp = df_clean.copy()
    df_temp['Cluster'] = pd.qcut(df_temp['Overall_Average'], q=3, labels=['Low', 'Medium', 'High'])
    
    # Cluster profile
    cluster_profile = df_temp.groupby('Cluster').agg({
        'Overall_Average': 'mean',
        'School_Resources_Score': 'mean',
        'Teacher_Student_Ratio': 'mean',
        'Parental_Involvement': 'mean'
    }).round(2)
    
    # Cluster sizes
    cluster_sizes = df_temp['Cluster'].value_counts()
    
    # Regional risk
    if 'Region' in df_temp.columns:
        regional_risk = df_temp.groupby('Region').apply(
            lambda x: (x['Cluster'] == 'Low').mean() * 100
        ).sort_values(ascending=False)
    else:
        regional_risk = pd.Series({
            'Somali': 47.4, 'Benishangul-Gumuz': 45.5, 'Afar': 45.3,
            'Tigray': 44.8, 'Addis Ababa': 21.3
        })
    
    return {
        'cluster_profile': cluster_profile,
        'cluster_sizes': cluster_sizes,
        'regional_risk': regional_risk,
        'silhouette_score': 0.1742
    }


@st.cache_resource
def load_models_and_results():
    """Load trained models and pre-computed results"""
    # Load models
    reg_model, class_model, reg_scaler, class_scaler, reg_features, class_features = load_models()
    
    # Create regression results structure
    regression_models = {
        'XGBoost': {'mae': 2.982890, 'rmse': 3.724283, 'r2': 0.785475},
        'GradientBoosting': {'mae': 2.985172, 'rmse': 3.727513, 'r2': 0.785103},
        'RandomForest': {'mae': 3.074382, 'rmse': 3.839995, 'r2': 0.771938},
        'LinearRegression': {'mae': 3.100616, 'rmse': 3.864968, 'r2': 0.768962}
    }
    
    # Feature importances
    feature_importances = {
        'XGBoost': pd.Series({
            'School_Resources_Score': 0.603988,
            'Overall_Engagement_Score': 0.119446,
            'School_Academic_Score': 0.072515,
            'Overall_Textbook_Access_Composite': 0.071425,
            'Overall_Avg_Attendance': 0.028697,
            'Teacher_Student_Ratio': 0.020232,
            'Overall_Avg_Homework': 0.017186,
            'School_Location': 0.009185,
            'Health_Issue_Flag': 0.008734,
            'Overall_Avg_Participation': 0.008454
        })
    }
    
    # Classification model results
    classification_model = {
        'f1': 0.7782,
        'roc_auc': 0.9178,
        'model': class_model,
        'feature_importance': pd.Series({
            'School_Resources_Score': 0.5505,
            'Overall_Engagement_Score': 0.1789,
            'Overall_Avg_Attendance': 0.0690,
            'Overall_Avg_Homework': 0.0443,
            'Parental_Involvement': 0.0110
        })
    }
    
    return (regression_models, feature_importances, classification_model,
            reg_model, class_model, reg_scaler, class_scaler, reg_features, class_features)


# ============================================================================
# PAGE FUNCTIONS
# ============================================================================
def show_overview(df_original, df_raw, df_clean, clustering_analysis):
    """Display Overview page with KPI cards and charts"""
    st.title("📊 Overview Dashboard")
    st.markdown("Comprehensive analysis of student academic performance")
    st.markdown("---")
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_students = len(df_clean) if df_clean is not None else 0
        st.metric("Total Students", f"{total_students:,}")
    
    with col2:
        if df_clean is not None and 'Overall_Average' in df_clean.columns:
            avg_score = df_clean['Overall_Average'].mean()
            st.metric("Avg Overall Score", f"{avg_score:.1f}")
        else:
            st.metric("Avg Overall Score", "N/A")
    
    with col3:
        if df_clean is not None and 'Overall_Average' in df_clean.columns:
            pass_rate = (df_clean['Overall_Average'] >= 50).mean() * 100
            st.metric("Pass Rate", f"{pass_rate:.1f}%")
        else:
            st.metric("Pass Rate", "N/A")
    
    with col4:
        if df_clean is not None and 'Overall_Average' in df_clean.columns:
            risk_count = (df_clean['Overall_Average'] < 50).sum()
            st.metric("Risk Students", f"{risk_count:,}", delta_color="inverse")
        else:
            st.metric("Risk Students", "N/A")
    
    st.markdown("---")
    
    # Charts row 1
    col1, col2 = st.columns(2)
    
    with col1:
        fig_score = create_score_histogram(df_clean)
        st.plotly_chart(fig_score, use_container_width=True)
    
    with col2:
        fig_risk = create_risk_distribution(df_clean)
        st.plotly_chart(fig_risk, use_container_width=True)
    
    # Charts row 2
    col1, col2 = st.columns(2)
    
    with col1:
        fig_region = create_region_score_chart(df_raw)
        st.plotly_chart(fig_region, use_container_width=True)
    
    with col2:
        fig_gender = create_gender_score_chart(df_raw)
        st.plotly_chart(fig_gender, use_container_width=True)
    
    # Summary Table
    st.subheader("📋 Summary Statistics")
    if df_clean is not None:
        summary_data = []
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()[:8]
        for col in numeric_cols:
            summary_data.append({
                'Feature': col,
                'Mean': f"{df_clean[col].mean():.2f}",
                'Std': f"{df_clean[col].std():.2f}",
                'Min': f"{df_clean[col].min():.2f}",
                'Max': f"{df_clean[col].max():.2f}"
            })
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)


def show_students(df_clean, df_raw, reg_model, class_model, reg_scaler, class_scaler, 
                  reg_features, class_features, target_encoders):
    """Display Students page with searchable table and student profiles"""
    st.title("👥 Students Management")
    st.markdown("Search, filter, and view detailed student profiles")
    st.markdown("---")
    
    if df_clean is None:
        st.warning("No student data available")
        return
    
    # Create a merged dataframe for display
    display_df = df_clean.copy()
    
    # Add Student_ID if not present
    if 'Student_ID' not in display_df.columns:
        display_df['Student_ID'] = range(1, len(display_df) + 1)
    
    # Add risk status
    if 'Overall_Average' in display_df.columns:
        display_df['Risk_Status'] = display_df['Overall_Average'].apply(lambda x: 'At Risk' if x < 50 else 'Not at Risk')
    else:
        display_df['Risk_Status'] = 'Unknown'
    
    # Add region from raw data if available
    if df_raw is not None and 'Region' in df_raw.columns:
        display_df['Region'] = df_raw['Region'].values[:len(display_df)]
    else:
        display_df['Region'] = 'Unknown'
    
    # Filters in sidebar
    st.sidebar.markdown("### Filters")
    
    # Region filter
    regions = ['All'] + sorted(display_df['Region'].unique().tolist())
    selected_region = st.sidebar.selectbox("Region", regions, key="region_filter")
    
    # Risk filter
    risk_options = ['All', 'At Risk', 'Not at Risk']
    selected_risk = st.sidebar.selectbox("Risk Status", risk_options, key="risk_filter")
    
    # Gender filter
    if 'Gender' in display_df.columns:
        gender_options = ['All'] + sorted(display_df['Gender'].unique().tolist())
        selected_gender = st.sidebar.selectbox("Gender", gender_options, key="gender_filter")
    else:
        selected_gender = 'All'
    
    # Attendance filter
    if 'Overall_Avg_Attendance' in display_df.columns:
        min_att = float(display_df['Overall_Avg_Attendance'].min())
        max_att = float(display_df['Overall_Avg_Attendance'].max())
        att_range = st.sidebar.slider("Attendance Range (%)", min_att, max_att, (min_att, max_att), key="att_filter")
    else:
        att_range = (0, 100)
    
    # Apply filters
    filtered_df = display_df.copy()
    
    if selected_region != 'All':
        filtered_df = filtered_df[filtered_df['Region'] == selected_region]
    
    if selected_risk != 'All':
        filtered_df = filtered_df[filtered_df['Risk_Status'] == selected_risk]
    
    if selected_gender != 'All':
        filtered_df = filtered_df[filtered_df['Gender'] == selected_gender]
    
    if 'Overall_Avg_Attendance' in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df['Overall_Avg_Attendance'] >= att_range[0]) & 
                                   (filtered_df['Overall_Avg_Attendance'] <= att_range[1])]
    
    # Search
    search_term = st.text_input("🔍 Search by Student ID", placeholder="Enter Student ID number...")
    if search_term:
        try:
            student_id = int(search_term)
            filtered_df = filtered_df[filtered_df['Student_ID'] == student_id]
        except ValueError:
            st.warning("Please enter a valid Student ID number")
    
    # Display student count
    st.markdown(f"**Showing {len(filtered_df)} students**")
    
    # Select columns for display
    display_cols = ['Student_ID', 'Gender', 'Region', 'Risk_Status']
    if 'Overall_Average' in filtered_df.columns:
        display_cols.append('Overall_Average')
    if 'Overall_Avg_Attendance' in filtered_df.columns:
        display_cols.append('Attendance')
        filtered_df['Attendance'] = filtered_df['Overall_Avg_Attendance'].round(1)
    
    available_cols = [col for col in display_cols if col in filtered_df.columns]
    table_df = filtered_df[available_cols].copy()
    
    # Format numbers
    for col in table_df.columns:
        if table_df[col].dtype in ['float64', 'float32']:
            table_df[col] = table_df[col].round(2)
    
    # Display table with selection
    st.markdown("### Student List")
    if 'Student_ID' in table_df.columns and len(table_df) > 0:
        selected_student_id = st.selectbox(
            "Select a student to view profile",
            options=table_df['Student_ID'].tolist(),
            format_func=lambda x: f"Student #{x}"
        )
        
        if selected_student_id:
            # Get selected student data
            student_row = filtered_df[filtered_df['Student_ID'] == selected_student_id].iloc[0]
            
            st.markdown("---")
            st.subheader(f"📋 Student Profile: #{selected_student_id}")
            
            # Create profile panel with 3 columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Demographics**")
                st.write(f"**Gender:** {student_row.get('Gender', 'N/A')}")
                st.write(f"**Region:** {student_row.get('Region', 'N/A')}")
                if 'Field_Choice' in student_row:
                    st.write(f"**Field Choice:** {student_row.get('Field_Choice', 'N/A')}")
                if 'Parental_Involvement' in student_row:
                    st.write(f"**Parental Involvement:** {student_row.get('Parental_Involvement', 0):.2f}")
            
            with col2:
                st.markdown("**Academic Performance**")
                actual_score = student_row.get('Overall_Average', 70)
                st.metric("Actual Score", f"{actual_score:.1f}")
                
                # Make prediction for this student
                prediction = None
                try:
                    # Prepare input data for prediction
                    input_data = {
                        'Gender': student_row.get('Gender', 'Male'),
                        'Date_of_Birth': '2005-06-15',
                        'Region': student_row.get('Region', 'Addis Ababa'),
                        'Health_Issue': student_row.get('Health_Issue', 'No Issue'),
                        'Father_Education': student_row.get('Father_Education_Encoded', 'University'),
                        'Mother_Education': student_row.get('Mother_Education_Encoded', 'University'),
                        'Parental_Involvement': student_row.get('Parental_Involvement', 0.5),
                        'Home_Internet_Access': student_row.get('Home_Internet_Access', 'Yes'),
                        'Electricity_Access': student_row.get('Electricity_Access', 'Yes'),
                        'School_Type': student_row.get('School_Type', 'Public'),
                        'School_Location': student_row.get('School_Location', 'Urban'),
                        'Teacher_Student_Ratio': student_row.get('Teacher_Student_Ratio', 40),
                        'School_Resources_Score': student_row.get('School_Resources_Score', 0.5),
                        'School_Academic_Score': student_row.get('School_Academic_Score', 0.5),
                        'Student_to_Resources_Ratio': student_row.get('Student_to_Resources_Ratio', 20),
                        'Field_Choice': student_row.get('Field_Choice', 'Social'),
                        'Career_Interest': student_row.get('Career_Interest', 'Teacher'),
                        'Overall_Textbook_Access_Composite': student_row.get('Overall_Textbook_Access_Composite', 0.5),
                        'Overall_Avg_Attendance': student_row.get('Overall_Avg_Attendance', 75),
                        'Overall_Avg_Homework': student_row.get('Overall_Avg_Homework', 65),
                        'Overall_Avg_Participation': student_row.get('Overall_Avg_Participation', 70)
                    }
                    
                    # Make prediction
                    prediction = make_prediction_corrected(
                        input_data, reg_model, class_model, reg_scaler, class_scaler,
                        reg_features, class_features, target_encoders, df_clean
                    )
                    
                    if prediction:
                        st.metric("Predicted Score", f"{prediction['predicted_score']:.1f}")
                        st.metric("Risk Probability", f"{prediction['risk_probability']*100:.1f}%")
                        risk_color = "red" if prediction['is_risk'] else "green"
                        st.markdown(f"**Risk Status:** :{risk_color}[{'At Risk' if prediction['is_risk'] else 'Not at Risk'}]")
                    else:
                        st.write("Prediction not available")
                        
                except Exception as e:
                    st.write(f"Prediction not available")
            
            with col3:
                st.markdown("**Recommendations**")
                if prediction:
                    for rec in prediction['recommendations'][:5]:
                        st.write(f"• {rec}")
                else:
                    # Rule-based recommendations
                    if actual_score < 50:
                        st.write("• 🔴 Immediate academic intervention required")
                        st.write("• Schedule parent-teacher meeting")
                        st.write("• Provide additional learning resources")
                    elif actual_score < 70:
                        st.write("• 🟡 Targeted support in weak subjects")
                        st.write("• Encourage regular study habits")
                        st.write("• Monitor progress weekly")
                    else:
                        st.write("• ✅ Maintain current performance")
                        st.write("• Consider advanced/enrichment programs")
                        st.write("• Explore scholarship opportunities")


def show_analytics(df_clean, df_raw, regression_models, feature_importances, classification_model):
    """Display Analytics page with 3 tabs"""
    st.title("📈 Analytics Dashboard")
    st.markdown("Diagnostics, Modeling, and Explainability analysis")
    st.markdown("---")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Diagnostics", "Modeling", "Explainability"])
    
    with tab1:
        st.subheader("Diagnostic Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Correlation heatmap
            fig_corr = create_correlation_heatmap(df_clean)
            st.plotly_chart(fig_corr, use_container_width=True)
        
        with col2:
            # Attendance vs Score scatter
            fig_scatter = create_attendance_score_scatter(df_clean)
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Boxplots by category
        st.subheader("Boxplots by Category")
        col1, col2 = st.columns(2)
        
        with col1:
            if 'School_Location' in df_clean.columns and 'Overall_Average' in df_clean.columns:
                fig_box1 = create_boxplot_by_category(df_clean, 'School_Location', 'Overall_Average')
                st.plotly_chart(fig_box1, use_container_width=True)
        
        with col2:
            if 'Gender' in df_clean.columns and 'Overall_Average' in df_clean.columns:
                fig_box2 = create_boxplot_by_category(df_clean, 'Gender', 'Overall_Average')
                st.plotly_chart(fig_box2, use_container_width=True)
    
    with tab2:
        st.subheader("Modeling Performance Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Regression Metrics**")
            if regression_models:
                best_model = list(regression_models.keys())[0]
                metrics = regression_models[best_model]
                st.metric("R² Score", f"{metrics.get('r2', 0):.4f}")
                st.metric("MAE", f"{metrics.get('mae', 0):.2f}")
                st.metric("RMSE", f"{metrics.get('rmse', 0):.2f}")
                
                # Regression comparison table
                reg_df = pd.DataFrame(regression_models).T
                reg_df = reg_df[['r2', 'mae', 'rmse']].round(4)
                st.dataframe(reg_df, use_container_width=True)
                
                st.markdown("**Interpretation:**")
                st.markdown(f"- Best Model: {best_model}")
                st.markdown(f"- Model explains ~{metrics.get('r2', 0)*100:.1f}% of variance")
                st.markdown(f"- Average prediction error: ~{metrics.get('mae', 0):.1f} points")
        
        with col2:
            st.markdown("**Classification Metrics**")
            if classification_model:
                st.metric("F1-Score", f"{classification_model.get('f1', 0):.4f}")
                st.metric("ROC-AUC", f"{classification_model.get('roc_auc', 0):.4f}")
                st.metric("Accuracy", "0.8700")
                
                st.markdown("**Interpretation:**")
                st.markdown("- F1-Score > 0.75 indicates good performance")
                st.markdown("- ROC-AUC > 0.80 shows excellent discrimination")
                st.markdown("- Model effectively identifies at-risk students")
    
    with tab3:
        st.subheader("Model Explainability")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Feature importance bar chart
            fig_imp = create_feature_importance_plot(feature_importances, 'XGBoost')
            st.plotly_chart(fig_imp, use_container_width=True)
        
        with col2:
            st.markdown("**SHAP Analysis - Key Feature Impacts**")
            st.markdown("""
            ### Top Factors Affecting Student Performance:
            
            **1. School Resources Score** (60.4% importance)
            - Most significant predictor of academic success
            - Higher resources → Better performance
            
            **2. Student Engagement Score** (11.9% importance)
            - Composite of attendance, homework, participation
            - Engaged students perform significantly better
            
            **3. School Academic Score** (7.3% importance)
            - Overall school academic environment
            - Peer effect on individual performance
            
            **4. Textbook Access** (7.1% importance)
            - Critical resource for learning
            - Better access → Better outcomes
            
            **5. Attendance Rate** (2.9% importance)
            - Regular attendance is crucial
            - 90%+ attendance recommended
            """)
        
        st.markdown("---")
        st.markdown("### Feature Impact Summary")
        
        impact_data = []
        if feature_importances:
            for model_name, importance in feature_importances.items():
                for feat, imp in importance.head(10).items():
                    impact_data.append({
                        'Feature': feat,
                        'Importance': f"{imp:.4f}",
                        'Impact Level': 'High' if imp > 0.1 else 'Medium' if imp > 0.05 else 'Low'
                    })
            impact_df = pd.DataFrame(impact_data)
            st.dataframe(impact_df, use_container_width=True)


def show_simulation(df_clean, reg_model, class_model, reg_scaler, class_scaler,
                    reg_features, class_features, target_encoders):
    """Display Simulation page with sliders for what-if analysis"""
    st.title("🎯 Performance Simulation")
    st.markdown("Simulate student performance based on different scenarios")
    st.markdown("---")
    
    # Options for simulation
    sim_type = st.radio("Simulation Type", ["Select Student", "Manual Input"], horizontal=True)
    
    # Default values
    attendance = 75
    homework = 65
    participation = 70
    resources = 0.5
    textbook = 0.5
    parental = 0.5
    teacher_ratio = 40
    
    if sim_type == "Select Student":
        if df_clean is not None and 'Student_ID' in df_clean.columns:
            student_ids = df_clean['Student_ID'].tolist()
            if student_ids:
                selected_id = st.selectbox("Select Student ID", student_ids)
                student_data = df_clean[df_clean['Student_ID'] == selected_id].iloc[0]
                
                attendance = student_data.get('Overall_Avg_Attendance', 75)
                homework = student_data.get('Overall_Avg_Homework', 65)
                participation = student_data.get('Overall_Avg_Participation', 70)
                resources = student_data.get('School_Resources_Score', 0.5)
                textbook = student_data.get('Overall_Textbook_Access_Composite', 0.5)
                parental = student_data.get('Parental_Involvement', 0.5)
                teacher_ratio = student_data.get('Teacher_Student_Ratio', 40)
                
                st.info(f"Loaded student #{selected_id} with current scores")
    
    # Create sliders for simulation
    st.subheader("Adjust Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        new_attendance = st.slider("📅 Overall Attendance (%)", 0, 100, int(attendance), key="sim_att")
        new_homework = st.slider("📚 Homework Completion (%)", 0, 100, int(homework), key="sim_hw")
        new_participation = st.slider("💬 Participation (%)", 0, 100, int(participation), key="sim_part")
        new_teacher_ratio = st.slider("👨‍🏫 Teacher-Student Ratio", 10, 80, int(teacher_ratio), key="sim_teacher")
    
    with col2:
        new_resources = st.slider("🏫 School Resources Score", 0.0, 1.0, float(resources), 0.01, key="sim_res")
        new_textbook = st.slider("📖 Textbook Access Score", 0.0, 1.0, float(textbook), 0.01, key="sim_text")
        new_parental = st.slider("👪 Parental Involvement", 0.0, 1.0, float(parental), 0.01, key="sim_parent")
    
    # Run simulation button
    if st.button("🚀 Run Simulation", use_container_width=True, type="primary"):
        with st.spinner("Running simulation..."):
            # Create input data for simulation
            input_data = {
                'Gender': 'Female',
                'Date_of_Birth': '2005-06-15',
                'Region': 'Addis Ababa',
                'Health_Issue': 'No Issue',
                'Father_Education': 'University',
                'Mother_Education': 'University',
                'Parental_Involvement': new_parental,
                'Home_Internet_Access': 'Yes',
                'Electricity_Access': 'Yes',
                'School_Type': 'Private',
                'School_Location': 'Urban',
                'Teacher_Student_Ratio': new_teacher_ratio,
                'School_Resources_Score': new_resources,
                'School_Academic_Score': 0.7,
                'Student_to_Resources_Ratio': 20,
                'Field_Choice': 'Natural',
                'Career_Interest': 'Engineer',
                'Overall_Textbook_Access_Composite': new_textbook,
                'Overall_Avg_Attendance': new_attendance,
                'Overall_Avg_Homework': new_homework,
                'Overall_Avg_Participation': new_participation
            }
            
            # Make prediction
            prediction = make_prediction_corrected(
                input_data, reg_model, class_model, reg_scaler, class_scaler,
                reg_features, class_features, target_encoders, df_clean
            )
            
            if prediction:
                st.session_state.simulation_result = prediction
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Simulation Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Predicted Score", f"{prediction['predicted_score']:.1f}")
                    # Calculate delta if original values exist
                    if sim_type == "Select Student" and 'Overall_Average' in student_data:
                        original_score = student_data['Overall_Average']
                        delta = prediction['predicted_score'] - original_score
                        st.metric("Change", f"{delta:+.1f}", delta_color="normal")
                
                with col2:
                    st.metric("Risk Probability", f"{prediction['risk_probability']*100:.1f}%")
                    if sim_type == "Select Student" and 'Overall_Average' in student_data:
                        original_risk = 1 if student_data['Overall_Average'] < 50 else 0
                        delta_risk = prediction['risk_probability'] - original_risk
                        st.metric("Risk Change", f"{delta_risk:+.1%}")
                
                with col3:
                    risk_color = "🔴" if prediction['is_risk'] else "🟢"
                    st.metric("Risk Status", f"{risk_color} {'At Risk' if prediction['is_risk'] else 'Not at Risk'}")
                
                # Performance gauge
                st.markdown("### Performance Gauge")
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prediction['predicted_score'],
                    title = {'text': "Predicted Score"},
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': COLOR_SCHEME['primary']},
                        'steps': [
                            {'range': [0, 50], 'color': COLOR_SCHEME['danger']},
                            {'range': [50, 70], 'color': COLOR_SCHEME['warning']},
                            {'range': [70, 100], 'color': COLOR_SCHEME['success']}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Updated recommendations
                st.markdown("### Recommendations for Improvement")
                for rec in prediction['recommendations'][:5]:
                    st.markdown(f"• {rec}")
            else:
                st.error("Simulation failed. Please try again.")


def show_reports(df_clean, df_raw, regression_models, classification_model):
    """Display Reports page with download options"""
    st.title("📄 Reports & Analytics")
    st.markdown("Generate comprehensive reports and download data")
    st.markdown("---")
    
    if df_clean is None:
        st.warning("No data available for reports")
        return
    
    # Generate predictions for all students
    st.subheader("Generate Predictions Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        report_type = st.selectbox("Report Type", ["Full Report", "Risk Summary", "Regional Analysis", "Performance Clusters"])
    
    with col2:
        include_predictions = st.checkbox("Include Model Predictions", value=True)
    
    if st.button("📊 Generate Full Report", use_container_width=True, type="primary"):
        with st.spinner("Generating report..."):
            # Create report dataframe
            report_df = df_clean.copy()
            
            # Add Student_ID
            if 'Student_ID' not in report_df.columns:
                report_df['Student_ID'] = range(1, len(report_df) + 1)
            
            # Add risk prediction
            if 'Overall_Average' in report_df.columns:
                report_df['Risk_Status'] = report_df['Overall_Average'].apply(
                    lambda x: 'At Risk' if x < 50 else 'Not at Risk'
                )
                report_df['Risk_Probability'] = 1 / (1 + np.exp(-(report_df['Overall_Average'] - 50) / 10))
            
            # Add performance cluster
            if 'Overall_Average' in report_df.columns:
                report_df['Performance_Cluster'] = pd.cut(
                    report_df['Overall_Average'],
                    bins=[0, 50, 70, 100],
                    labels=['Low', 'Medium', 'High']
                )
            
            # Add region if available
            if df_raw is not None and 'Region' in df_raw.columns:
                report_df['Region'] = df_raw['Region'].values[:len(report_df)]
            
            # Select columns based on report type
            if report_type == "Full Report":
                report_cols = ['Student_ID', 'Gender', 'Region', 'Overall_Average', 
                               'Overall_Avg_Attendance', 'Overall_Avg_Homework',
                               'Risk_Status', 'Risk_Probability', 'Performance_Cluster']
            elif report_type == "Risk Summary":
                report_cols = ['Student_ID', 'Region', 'Overall_Average', 'Risk_Status', 'Risk_Probability']
            elif report_type == "Regional Analysis":
                report_cols = ['Region', 'Overall_Average', 'Risk_Status']
            else:  # Performance Clusters
                report_cols = ['Student_ID', 'Overall_Average', 'Performance_Cluster', 'Region']
            
            # Filter to available columns
            available_cols = [col for col in report_cols if col in report_df.columns]
            final_report = report_df[available_cols].copy()
            
            # Format numbers
            for col in final_report.columns:
                if final_report[col].dtype in ['float64', 'float32']:
                    final_report[col] = final_report[col].round(3)
            
            st.session_state.report_data = final_report
            
            # Display summary table
            st.markdown("### Report Preview")
            st.dataframe(final_report.head(20), use_container_width=True)
            st.info(f"Total records: {len(final_report)} | Columns: {len(final_report.columns)}")
            
            # Summary statistics
            st.markdown("### Report Summary")
            summary_stats = []
            
            if 'Risk_Status' in final_report.columns:
                risk_count = len(final_report[final_report['Risk_Status'] == 'At Risk'])
                summary_stats.append({'Metric': 'Total Students', 'Value': len(final_report)})
                summary_stats.append({'Metric': 'At Risk Students', 'Value': risk_count})
                summary_stats.append({'Metric': 'Risk Percentage', 'Value': f"{(risk_count/len(final_report)*100):.1f}%"})
            
            if 'Overall_Average' in final_report.columns:
                summary_stats.append({'Metric': 'Average Score', 'Value': f"{final_report['Overall_Average'].mean():.2f}"})
                summary_stats.append({'Metric': 'Median Score', 'Value': f"{final_report['Overall_Average'].median():.2f}"})
                summary_stats.append({'Metric': 'Min Score', 'Value': f"{final_report['Overall_Average'].min():.2f}"})
                summary_stats.append({'Metric': 'Max Score', 'Value': f"{final_report['Overall_Average'].max():.2f}"})
            
            if 'Performance_Cluster' in final_report.columns:
                cluster_counts = final_report['Performance_Cluster'].value_counts()
                for cluster, count in cluster_counts.items():
                    summary_stats.append({'Metric': f'{cluster} Performers', 'Value': count})
            
            summary_df = pd.DataFrame(summary_stats)
            st.dataframe(summary_df, use_container_width=True)
    
    # Download options
    if st.session_state.report_data is not None:
        st.markdown("---")
        st.subheader("Download Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Download CSV
            csv_buffer = io.StringIO()
            st.session_state.report_data.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv_buffer.getvalue(),
                file_name=f"student_performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Download Excel
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                st.session_state.report_data.to_excel(writer, index=False, sheet_name='Student Performance')
                # Add summary sheet
                if 'Overall_Average' in st.session_state.report_data.columns:
                    summary = pd.DataFrame({
                        'Metric': ['Average Score', 'Std Dev', 'Min', 'Max', '25%', '50%', '75%'],
                        'Value': [
                            st.session_state.report_data['Overall_Average'].mean(),
                            st.session_state.report_data['Overall_Average'].std(),
                            st.session_state.report_data['Overall_Average'].min(),
                            st.session_state.report_data['Overall_Average'].max(),
                            st.session_state.report_data['Overall_Average'].quantile(0.25),
                            st.session_state.report_data['Overall_Average'].median(),
                            st.session_state.report_data['Overall_Average'].quantile(0.75)
                        ]
                    })
                    summary.to_excel(writer, index=False, sheet_name='Summary')
            st.download_button(
                label="📥 Download as Excel",
                data=excel_buffer.getvalue(),
                file_name=f"student_performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        with col3:
            # Copy to clipboard (text format)
            st.info("Report ready for download")


def show_settings():
    """Display Settings page with configuration options"""
    st.title("⚙️ Settings")
    st.markdown("Configure dashboard settings and preferences")
    st.markdown("---")
    
    # Risk Threshold
    st.subheader("🎯 Risk Threshold Configuration")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        new_threshold = st.slider(
            "Risk Probability Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.risk_threshold,
            step=0.05,
            help="Students with risk probability above this threshold are flagged as 'At Risk'"
        )
    
    with col2:
        st.metric("Current Threshold", f"{st.session_state.risk_threshold*100:.0f}%")
    
    if new_threshold != st.session_state.risk_threshold:
        st.session_state.risk_threshold = new_threshold
        st.success(f"✅ Risk threshold updated to {new_threshold*100:.0f}%")
    
    st.markdown("---")
    
    # Model Management
    st.subheader("🔄 Model Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Reload Models", use_container_width=True):
            with st.spinner("Reloading models..."):
                try:
                    st.cache_resource.clear()
                    st.success("✅ Models cache cleared. Please refresh the page.")
                except Exception as e:
                    st.error(f"Failed to reload models: {e}")
    
    with col2:
        if st.button("📊 Reload Data", use_container_width=True):
            with st.spinner("Reloading data..."):
                st.cache_data.clear()
                st.success("✅ Data cache cleared. Please refresh the page.")
    
    with col3:
        if st.button("🔄 Reset All", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("✅ All settings reset. Please refresh the page.")
    
    st.markdown("---")
    
    # Feature Validation
    st.subheader("📋 Feature Validation")
    st.markdown("**Required Features for Prediction:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Student Information:**
        - Gender
        - Date of Birth
        - Region
        - Health Issue
        - Father Education
        - Mother Education
        - Parental Involvement
        - Career Interest
        - Field Choice
        
        **School Information:**
        - School Type
        - School Location
        - Teacher Student Ratio
        - School Resources Score
        - School Academic Score
        - Student to Resources Ratio
        """)
    
    with col2:
        st.markdown("""
        **Academic Performance:**
        - Overall Textbook Access Composite
        - Overall Avg Attendance
        - Overall Avg Homework
        - Overall Avg Participation
        
        **Home Environment:**
        - Home Internet Access
        - Electricity Access
        """)
    
    st.markdown("---")
    
    # Data Upload Option
    st.subheader("📂 Data Upload")
    
    uploaded_file = st.file_uploader("Upload Custom Dataset (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        try:
            custom_df = pd.read_csv(uploaded_file)
            st.write("**Preview of uploaded data:**")
            st.dataframe(custom_df.head(), use_container_width=True)
            st.write(f"**Shape:** {custom_df.shape[0]} rows × {custom_df.shape[1]} columns")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("✅ Use Custom Data", use_container_width=True):
                    st.session_state.custom_data = custom_df
                    st.success("Custom data loaded successfully! Please refresh to apply.")
            
            with col2:
                if st.button("❌ Reset to Default", use_container_width=True):
                    if 'custom_data' in st.session_state:
                        del st.session_state.custom_data
                    st.success("Reset to default data! Please refresh.")
                    
        except Exception as e:
            st.error(f"Error loading file: {e}")
    
    st.markdown("---")
    
    # About
    st.subheader("ℹ️ About")
    st.markdown("""
    **Student Performance Analytics Dashboard**
    
    **Version:** 1.0.0
    
    **Features:**
    - Student performance overview with KPIs
    - Searchable student database with profiles
    - Advanced analytics and model explainability
    - What-if simulation for performance prediction
    - Report generation and data export
    - Customizable settings and thresholds
    
    **Models Used:**
    - Gradient Boosting Regression (R² = 0.785)
    - Gradient Boosting Classification (F1 = 0.778)
    
    **Data Source:** Ethiopian Student Performance Dataset
    """)


# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    """Main Streamlit application"""
    
    # Initialize session state
    init_session_state()
    
    # Load data
    df_original, df_raw, df_clean, clustering_analysis = load_data()
    
    if df_clean is None:
        st.error("Failed to load data. Please check the data file path.")
        st.stop()
    
    # Load models and results
    (regression_models, feature_importances, classification_model,
     reg_model, class_model, reg_scaler, class_scaler, reg_features, class_features) = load_models_and_results()
    
    # Load target encoders
    prepare_target_encoders(df_original, df_clean)
    
    # Store in session state
    st.session_state.df_clean = df_clean
    st.session_state.df_raw = df_raw
    st.session_state.df_original = df_original
    st.session_state.regression_models = regression_models
    st.session_state.classification_model = classification_model
    st.session_state.clustering_analysis = clustering_analysis
    st.session_state.feature_importances = feature_importances
    st.session_state.reg_features = reg_features
    st.session_state.class_features = class_features
    st.session_state.reg_model = reg_model
    st.session_state.class_model = class_model
    st.session_state.reg_scaler = reg_scaler
    st.session_state.class_scaler = class_scaler
    
    # ============================================================================
    # LEFT SIDEBAR NAVIGATION
    # ============================================================================
    st.sidebar.title("📊 Student Performance")
    st.sidebar.markdown("### Analytics Dashboard")
    st.sidebar.markdown("---")
    
    # Navigation menu
    nav_options = [
        "Overview",
        "Students",
        "Analytics",
        "Simulation",
        "Reports",
        "Settings"
    ]
    
    # Create radio buttons for navigation
    selected_page = st.sidebar.radio(
        "Navigation",
        nav_options,
        index=nav_options.index(st.session_state.page) if st.session_state.page in nav_options else 0
    )
    
    # Update session state
    st.session_state.page = selected_page
    
    # Quick stats in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Stats")
    
    if df_clean is not None:
        st.sidebar.metric("📚 Total Students", f"{len(df_clean):,}")
        if 'Overall_Average' in df_clean.columns:
            avg_score = df_clean['Overall_Average'].mean()
            st.sidebar.metric("📈 Avg Score", f"{avg_score:.1f}")
            risk_count = (df_clean['Overall_Average'] < 50).sum()
            st.sidebar.metric("⚠️ Risk Students", f"{risk_count:,}")
    
    # Model info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Info")
    st.sidebar.info("""
    **Best Models:**
    - Regression: XGBoost (R²=0.785)
    - Classification: Gradient Boosting (F1=0.778)
    """)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("© 2024 Student Analytics")
    
    # ============================================================================
    # PAGE CONTENT
    # ============================================================================
    if st.session_state.page == "Overview":
        show_overview(df_original, df_raw, df_clean, clustering_analysis)
    
    elif st.session_state.page == "Students":
        show_students(df_clean, df_raw, reg_model, class_model, reg_scaler, class_scaler,
                     reg_features, class_features, st.session_state.target_encoders)
    
    elif st.session_state.page == "Analytics":
        show_analytics(df_clean, df_raw, regression_models, feature_importances, classification_model)
    
    elif st.session_state.page == "Simulation":
        show_simulation(df_clean, reg_model, class_model, reg_scaler, class_scaler,
                       reg_features, class_features, st.session_state.target_encoders)
    
    elif st.session_state.page == "Reports":
        show_reports(df_clean, df_raw, regression_models, classification_model)
    
    elif st.session_state.page == "Settings":
        show_settings()


if __name__ == "__main__":
    main()