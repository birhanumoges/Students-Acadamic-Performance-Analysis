"""
Visualization utilities for Ethiopian Student Performance Dashboard
"""
import socketserver

if not hasattr(socketserver, "UnixStreamServer"):
    socketserver.UnixStreamServer = socketserver.TCPServer

import shap


import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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
NATIONAL_EXAM_MODEL_PERFORMANCE = None
NATIONAL_EXAM_FEATURE_IMPORTANCE = None
regression_models = None
best_reg_model = None
feature_importances_global = None
classification_model = None
clustering_analysis = None
shap_data_precomputed = None
df_raw = None
df_clean = None
prediction_result = None

def initialize_global_vars():
    """Initialize global variables for visualizations"""
    global NATIONAL_EXAM_MODEL_PERFORMANCE, NATIONAL_EXAM_FEATURE_IMPORTANCE, \
           regression_models, best_reg_model, feature_importances_global, \
           classification_model, clustering_analysis, shap_data_precomputed, \
           df_raw, df_clean
    
    # Provided National Exam Score Model Performance
    NATIONAL_EXAM_MODEL_PERFORMANCE = pd.DataFrame({
        'Model': ['Gradient Boosting', 'XGBoost', 'Random Forest',
                  'Ridge Regression', 'Linear Regression', 'Lasso Regression'],
        'R2_Score': [0.437997, 0.435344, 0.425839, 0.405843, 0.405831, 0.404305],
        'MAE': [0.081404, 0.081569, 0.082253, 0.083816, 0.083816, 0.083893],
        'RMSE': [0.107109, 0.107362, 0.108262, 0.110131, 0.110132, 0.110273]
    })
    
    # Provided Feature Importance for National Exam Score Model
    NATIONAL_EXAM_FEATURE_IMPORTANCE = pd.DataFrame({
        'Feature': [
            'Score_x_Participation',
            'Overall_Avg_Homework',
            'School_Academic_Score',
            'Overall_Test_Score_Avg',
            'Overall_Avg_Attendance',
            'Overall_Avg_Participation',
            'School_Resources_Score',
            'Parental_Involvement',
            'Resource_Efficiency',
            'Teacher_Student_Ratio',
            'Student_to_Resources_Ratio',
            'School_Type_Target',
            'Overall_Engagement_Score',
            'Teacher_Load_Adjusted',
            'Overall_Textbook_Access_Composite',
            'Field_Choice',
            'Career_Interest_Encoded'
        ],
        'Importance': [
            0.735596, 0.071998, 0.066883, 0.043070, 0.017778, 0.016191,
            0.013264, 0.011599, 0.005805, 0.005141, 0.002633, 0.002587,
            0.001936, 0.001591, 0.001533, 0.001304, 0.001090
        ],
        'Importance_%': [
            73.559627, 7.199785, 6.688265, 4.307000, 1.777779, 1.619086,
            1.326450, 1.159938, 0.580545, 0.514109, 0.263260, 0.258688,
            0.193604, 0.159140, 0.153329, 0.130358, 0.109039
        ]
    })


def set_global_data(reg_models, best_model, feat_importances, class_model, cluster_analysis, shap_data, raw_df, clean_df):
    """Set global data for visualizations"""
    global regression_models, best_reg_model, feature_importances_global, \
           classification_model, clustering_analysis, shap_data_precomputed, \
           df_raw, df_clean
    
    regression_models = reg_models
    best_reg_model = best_model
    feature_importances_global = feat_importances
    classification_model = class_model
    clustering_analysis = cluster_analysis
    shap_data_precomputed = shap_data
    df_raw = raw_df
    df_clean = clean_df


def create_datatype_bar_plot():
    """Create bar plot for data types distribution"""
    if df_raw is None:
        return go.Figure()
    dtypes = df_raw.dtypes.value_counts()
    fig = go.Figure(data=[
        go.Bar(
            x=dtypes.index.astype(str),
            y=dtypes.values,
            marker_color=COLOR_SCHEME['primary'],
            text=dtypes.values,
            textposition='auto'
        )
    ])
    fig.update_layout(
        title="Data Type Distribution",
        xaxis_title="Data Type",
        yaxis_title="Count",
        plot_bgcolor=COLOR_SCHEME['background'],
        paper_bgcolor=COLOR_SCHEME['background'],
        font=dict(color=COLOR_SCHEME['text'])
    )
    return fig


def create_feature_category_plot():
    """Create bar plot showing features by category"""
    if df_clean is None:
        return go.Figure()
    
    feature_categories = {
        'Student Factors': [
            'Gender', 'Parental_Involvement', 'Home_Internet_Access',
            'Electricity_Access', 'Father_Education_Encoded',
            'Mother_Education_Encoded', 'Age', 'Field_Choice'
        ],
        'Academic Factors': [
            'Overall_Average', 'Total_National_Exam_Score',
            'Overall_Avg_Attendance', 'Overall_Avg_Homework',
            'Overall_Avg_Participation', 'Overall_Engagement_Score',
            'Overall_Textbook_Access_Composite'
        ],
        'School Factors': [
            'School_Location', 'Teacher_Student_Ratio', 'School_Resources_Score',
            'School_Academic_Score', 'Student_to_Resources_Ratio',
            'School_Type_Freq', 'School_Type_Target'
        ],
        'Regional Factors': [
            'Region_Encoded'
        ],
        'Health Factors': [
            'Health_Issue_Flag', 'Health_Issue_Severity', 'Health_Issue_Target'
        ],
        'Other': ['Career_Interest_Encoded']
    }
    
    category_counts = {}
    colors = [COLOR_SCHEME['primary'], COLOR_SCHEME['secondary'], COLOR_SCHEME['success'],
              COLOR_SCHEME['warning'], COLOR_SCHEME['danger'], COLOR_SCHEME['dark']]
    
    for i, (category, features) in enumerate(feature_categories.items()):
        existing_features = [f for f in features if f in df_clean.columns]
        category_counts[category] = len(existing_features)
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(category_counts.keys()),
            y=list(category_counts.values()),
            marker_color=colors[:len(category_counts)],
            text=list(category_counts.values()),
            textposition='auto'
        )
    ])
    fig.update_layout(
        title="Feature Distribution by Category",
        xaxis_title="Feature Category",
        yaxis_title="Number of Features",
        plot_bgcolor=COLOR_SCHEME['background'],
        paper_bgcolor=COLOR_SCHEME['background'],
        font=dict(color=COLOR_SCHEME['text']),
        xaxis_tickangle=-45
    )
    return fig


def create_correlation_heatmap():
    """Create correlation heatmap for top features"""
    if df_clean is None or 'Overall_Average' not in df_clean.columns:
        fig = go.Figure()
        fig.update_layout(title="Overall Average data not available")
        return fig
    
    numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if 'Risk_NotRisk' in numeric_cols:
        numeric_cols.remove('Risk_NotRisk')
    
    if len(numeric_cols) > 15:
        correlations = df_clean[numeric_cols].corr()['Overall_Average'].abs().sort_values(ascending=False)
        top_features = correlations.head(15).index.tolist()
        corr_data = df_clean[top_features].corr()
    else:
        corr_data = df_clean[numeric_cols].corr()
    
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
        title="Feature Correlation Heatmap (Top 15 Features)",
        plot_bgcolor=COLOR_SCHEME['background'],
        paper_bgcolor=COLOR_SCHEME['background'],
        font=dict(color=COLOR_SCHEME['text']),
        xaxis_tickangle=-45,
        height=600
    )
    return fig


def create_regression_comparison_plot():
    """Create comparison plot for regression models"""
    if not regression_models:
        fig = go.Figure()
        fig.update_layout(title="No regression models available")
        return fig
    
    models = list(regression_models.keys())
    r2_scores = [regression_models[model]['r2'] for model in models]
    colors = [COLOR_SCHEME['secondary'] if model == best_reg_model else COLOR_SCHEME['primary'] for model in models]
    
    fig = go.Figure(data=[
        go.Bar(x=models, y=r2_scores, marker_color=colors,
              text=[f'{score:.3f}' for score in r2_scores],
              textposition='auto')
    ])
    fig.update_layout(
        title="Regression Model R² Score Comparison",
        xaxis_title="Models",
        yaxis_title="R² Score",
        plot_bgcolor=COLOR_SCHEME['background'],
        paper_bgcolor=COLOR_SCHEME['background'],
        font=dict(color=COLOR_SCHEME['text'])
    )
    return fig


def create_actual_vs_predicted_plot():
    """Create actual vs predicted plot for best regression model"""
    if best_reg_model not in regression_models or regression_models[best_reg_model].get('y_test') is None:
        fig = go.Figure()
        fig.update_layout(title="Best regression model data not available")
        return fig
    
    best_reg = regression_models[best_reg_model]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=best_reg['y_test'],
        y=best_reg['y_pred'],
        mode='markers',
        name='Predictions',
        marker=dict(color=COLOR_SCHEME['primary'], size=6, opacity=0.6)
    ))
    fig.add_trace(go.Scatter(
        x=[best_reg['y_test'].min(), best_reg['y_test'].max()],
        y=[best_reg['y_test'].min(), best_reg['y_test'].max()],
        mode='lines',
        name='Perfect Prediction',
        line=dict(dash='dash', color=COLOR_SCHEME['secondary'], width=2)
    ))
    fig.update_layout(
        title=f"Actual vs Predicted Overall Average - {best_reg_model}",
        xaxis_title="Actual Overall Average",
        yaxis_title="Predicted Overall Average",
        plot_bgcolor=COLOR_SCHEME['background'],
        paper_bgcolor=COLOR_SCHEME['background'],
        font=dict(color=COLOR_SCHEME['text'])
    )
    return fig


def create_feature_importance_plot():
    """Create feature importance plot for regression model"""
    if best_reg_model in feature_importances_global:
        importance = feature_importances_global[best_reg_model].head(10)
        fig = go.Figure(go.Bar(
            x=importance.values,
            y=importance.index,
            orientation='h',
            marker_color=COLOR_SCHEME['primary'],
            text=[f'{imp:.3f}' for imp in importance.values],
            textposition='auto'
        ))
        fig.update_layout(
            title=f"Top 10 Feature Importance - {best_reg_model}",
            xaxis_title="Importance",
            yaxis_title="Features",
            plot_bgcolor=COLOR_SCHEME['background'],
            paper_bgcolor=COLOR_SCHEME['background'],
            font=dict(color=COLOR_SCHEME['text'])
        )
        return fig
    else:
        fig = go.Figure()
        fig.update_layout(title="Feature importance not available for this model")
        return fig


def create_national_exam_model_comparison_plot():
    """Create bar plot for National Exam Score model comparison"""
    if NATIONAL_EXAM_MODEL_PERFORMANCE is None:
        return go.Figure()
    df_national = NATIONAL_EXAM_MODEL_PERFORMANCE.copy()
    sorted_df = df_national.sort_values('R2_Score', ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=sorted_df['Model'],
        x=sorted_df['R2_Score'],
        orientation='h',
        marker_color=[COLOR_SCHEME['success'] if 'Gradient' in model else COLOR_SCHEME['primary'] for model in sorted_df['Model']],
        text=[f'{score:.4f}' for score in sorted_df['R2_Score']],
        textposition='auto',
        name='R² Score'
    ))
    fig.update_layout(
        title="National Exam Score Model Performance (R² Score)",
        xaxis_title="R² Score",
        yaxis_title="Model",
        plot_bgcolor=COLOR_SCHEME['background'],
        paper_bgcolor=COLOR_SCHEME['background'],
        font=dict(color=COLOR_SCHEME['text']),
        height=400,
        xaxis=dict(range=[0, max(sorted_df['R2_Score']) * 1.15])
    )
    return fig


def create_national_exam_feature_importance_plot():
    """Create feature importance plot for National Exam Score model"""
    if NATIONAL_EXAM_FEATURE_IMPORTANCE is None:
        return go.Figure()
    df_importance = NATIONAL_EXAM_FEATURE_IMPORTANCE.copy()
    sorted_df = df_importance.sort_values('Importance', ascending=True)
    fig = go.Figure(go.Bar(
        x=sorted_df['Importance'],
        y=sorted_df['Feature'],
        orientation='h',
        marker_color=COLOR_SCHEME['secondary'],
        text=[f'{imp:.1%}' for imp in sorted_df['Importance']],
        textposition='auto'
    ))
    fig.update_layout(
        title="Feature Importance - National Exam Score (Gradient Boosting)",
        xaxis_title="Importance Score",
        yaxis_title="Features",
        plot_bgcolor=COLOR_SCHEME['background'],
        paper_bgcolor=COLOR_SCHEME['background'],
        font=dict(color=COLOR_SCHEME['text']),
        height=500,
        xaxis=dict(range=[0, max(sorted_df['Importance']) * 1.15])
    )
    return fig


def create_national_exam_performance_table():
    """Create performance table for National Exam Score models"""
    if NATIONAL_EXAM_MODEL_PERFORMANCE is None:
        return pd.DataFrame()
    return NATIONAL_EXAM_MODEL_PERFORMANCE.copy()


def create_confusion_matrix_plot():
    """Create confusion matrix plot for classification"""
    if classification_model is None or classification_model.get('cm') is None:
        fig = go.Figure()
        fig.update_layout(title="Confusion matrix not available")
        return fig
    
    cm = classification_model['cm']
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Not Risk', 'Risk'],
        y=['Not Risk', 'Risk'],
        hoverongaps=False,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16}
    ))
    fig.update_layout(
        title="Risk Classification Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        plot_bgcolor=COLOR_SCHEME['background'],
        paper_bgcolor=COLOR_SCHEME['background'],
        font=dict(color=COLOR_SCHEME['text'])
    )
    return fig


def create_roc_curve_plot():
    """Create ROC curve plot"""
    if classification_model is None or classification_model.get('y_test') is None or len(classification_model['y_test']) == 0:
        fig = go.Figure()
        fig.update_layout(title="Classification data not available")
        return fig
    
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(classification_model['y_test'], classification_model['y_probs'])
    roc_auc = classification_model['roc_auc']
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.3f})',
        line=dict(color=COLOR_SCHEME['primary'], width=2)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(dash='dash', color=COLOR_SCHEME['secondary'], width=1)
    ))
    fig.update_layout(
        title="ROC Curve - Risk Classification",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        plot_bgcolor=COLOR_SCHEME['background'],
        paper_bgcolor=COLOR_SCHEME['background'],
        font=dict(color=COLOR_SCHEME['text']),
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain')
    )
    return fig


def create_cluster_distribution_plot():
    """Create cluster distribution plot"""
    if clustering_analysis is None or clustering_analysis.get('cluster_sizes') is None:
        fig = go.Figure()
        fig.update_layout(title="Clustering data not available")
        return fig
    
    cluster_counts = clustering_analysis['cluster_sizes']
    colors = {
        'High': COLOR_SCHEME['success'],
        'Medium': COLOR_SCHEME['warning'],
        'Low': COLOR_SCHEME['danger']
    }
    for label in ['High', 'Medium', 'Low']:
        if label not in cluster_counts.index:
            cluster_counts[label] = 0
    
    fig = go.Figure(data=[
        go.Bar(
            x=cluster_counts.index,
            y=cluster_counts.values,
            marker_color=[colors.get(label, COLOR_SCHEME['primary']) for label in cluster_counts.index],
            text=cluster_counts.values,
            textposition='auto'
        )
    ])
    fig.update_layout(
        title="Student Performance Cluster Distribution",
        xaxis_title="Performance Level",
        yaxis_title="Number of Students",
        plot_bgcolor=COLOR_SCHEME['background'],
        paper_bgcolor=COLOR_SCHEME['background'],
        font=dict(color=COLOR_SCHEME['text'])
    )
    return fig


def create_regional_risk_plot():
    """Create regional risk analysis plot"""
    if clustering_analysis is None or clustering_analysis.get('regional_risk') is None:
        fig = go.Figure()
        fig.update_layout(title="Regional data not available")
        return fig
    
    regional_risk = clustering_analysis['regional_risk']
    if regional_risk.empty:
        fig = go.Figure()
        fig.update_layout(title="Regional data not available")
        return fig
    
    fig = go.Figure(data=[
        go.Bar(
            x=regional_risk.index,
            y=regional_risk.values,
            marker_color=COLOR_SCHEME['danger'],
            text=[f'{val:.1f}%' for val in regional_risk.values],
            textposition='auto'
        )
    ])
    fig.update_layout(
        title="Regional Risk Analysis (% Low Performance)",
        xaxis_title="Region",
        yaxis_title="% Low Performance",
        plot_bgcolor=COLOR_SCHEME['background'],
        paper_bgcolor=COLOR_SCHEME['background'],
        font=dict(color=COLOR_SCHEME['text']),
        xaxis_tickangle=-45
    )
    return fig


def create_score_distribution_plot():
    """Create distribution plot for overall average scores"""
    if df_clean is None or 'Overall_Average' not in df_clean.columns:
        fig = go.Figure()
        fig.update_layout(title="Overall Average data not available")
        return fig
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df_clean['Overall_Average'],
        nbinsx=30,
        marker_color=COLOR_SCHEME['primary'],
        opacity=0.7
    ))
    fig.update_layout(
        title="Distribution of Overall Average Scores",
        xaxis_title="Overall Average Score",
        yaxis_title="Number of Students",
        plot_bgcolor=COLOR_SCHEME['background'],
        paper_bgcolor=COLOR_SCHEME['background'],
        font=dict(color=COLOR_SCHEME['text'])
    )
    return fig


def create_students_by_region_plot():
    """Create bar plot showing number of students by region"""
    if df_raw is None or 'Region' not in df_raw.columns:
        fig = go.Figure()
        fig.update_layout(title="Region data not available in raw dataset")
        return fig
    
    region_counts = df_raw['Region'].value_counts().sort_values(ascending=True)
    fig = go.Figure(data=[
        go.Bar(
            y=region_counts.index,
            x=region_counts.values,
            orientation='h',
            marker_color=COLOR_SCHEME['primary'],
            text=region_counts.values,
            textposition='auto'
        )
    ])
    fig.update_layout(
        title="Number of Students by Region",
        xaxis_title="Number of Students",
        yaxis_title="Region",
        plot_bgcolor=COLOR_SCHEME['background'],
        paper_bgcolor=COLOR_SCHEME['background'],
        font=dict(color=COLOR_SCHEME['text']),
        height=500
    )
    return fig


def create_feature_summary_table():
    """Create summary statistics table for each feature"""
    if df_raw is None:
        return pd.DataFrame()
    
    numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    stats_dict = {}
    for col in numeric_cols[:20]:
        stats_dict[col] = {
            'Mean': f"{df_raw[col].mean():.4f}",
            'Std': f"{df_raw[col].std():.4f}",
            'Min': f"{df_raw[col].min():.4f}",
            '25%': f"{df_raw[col].quantile(0.25):.4f}",
            'Median': f"{df_raw[col].median():.4f}",
            '75%': f"{df_raw[col].quantile(0.75):.4f}",
            'Max': f"{df_raw[col].max():.4f}",
            'Missing': f"{df_raw[col].isnull().sum():.0f}"
        }
    
    stats_list = ['Mean', 'Std', 'Min', '25%', 'Median', '75%', 'Max', 'Missing']
    summary_data = []
    for stat in stats_list:
        row = {'Statistic': stat}
        for col in list(stats_dict.keys())[:10]:
            row[col] = stats_dict[col][stat]
        summary_data.append(row)
    
    return pd.DataFrame(summary_data)


def create_risk_distribution_plot():
    """Create bar plot for risk vs not risk distribution in predictions"""
    global prediction_result
    
    if prediction_result is None:
        return None
    
    risk_data = {
        'Category': ['Risk', 'Not Risk'],
        'Count': [1 if prediction_result['is_risk'] else 0, 0 if prediction_result['is_risk'] else 1],
        'Percentage': [prediction_result['risk_probability'] * 100, (1 - prediction_result['risk_probability']) * 100]
    }
    df_risk = pd.DataFrame(risk_data)
    
    fig = go.Figure(data=[
        go.Bar(
            x=df_risk['Category'],
            y=df_risk['Count'],
            marker_color=[COLOR_SCHEME['danger'], COLOR_SCHEME['success']],
            text=[f'{p:.1f}%' for p in df_risk['Percentage']],
            textposition='auto'
        )
    ])
    fig.update_layout(
        title="Predicted Risk Distribution",
        xaxis_title="Risk Category",
        yaxis_title="Count",
        plot_bgcolor=COLOR_SCHEME['background'],
        paper_bgcolor=COLOR_SCHEME['background'],
        font=dict(color=COLOR_SCHEME['text']),
        height=300
    )
    return fig


def create_shap_summary_plot():
    """Create SHAP summary plot (beeswarm) for classification model"""
    fig = go.Figure()
    
    if shap_data_precomputed is None:
        fig.update_layout(
            title="SHAP Analysis Not Available",
            plot_bgcolor=COLOR_SCHEME['background'],
            paper_bgcolor=COLOR_SCHEME['background'],
            font=dict(color=COLOR_SCHEME['text'])
        )
        return fig
    
    try:
        shap_values = shap_data_precomputed['shap_values']
        feature_names = shap_data_precomputed['feature_names']
        n_features = len(feature_names)
        top_n = min(15, n_features)
        
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]
        top_features = [feature_names[i] for i in top_indices]
        top_shap_values = shap_values[:, top_indices]
        
        y_positions = np.arange(top_n)
        sample_size = min(50, top_shap_values.shape[0])
        
        x_vals = []
        y_vals = []
        colors = []
        
        for i in range(sample_size):
            for j, feature_idx in enumerate(top_indices):
                shap_value = shap_values[i, feature_idx]
                if abs(shap_value) > 0.01:
                    x_vals.append(shap_value)
                    y_vals.append(j + np.random.uniform(-0.1, 0.1))
                    colors.append('blue' if shap_value > 0 else 'red')
        
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='markers',
            marker=dict(
                size=6,
                color=colors,
                opacity=0.4
            ),
            showlegend=False
        ))
        
        fig.update_layout(
            title="SHAP Beeswarm Plot (Top 15 Features)",
            xaxis_title="SHAP Value (Impact on Risk Probability)",
            yaxis_title="Features",
            plot_bgcolor=COLOR_SCHEME['background'],
            paper_bgcolor=COLOR_SCHEME['background'],
            font=dict(color=COLOR_SCHEME['text']),
            height=400,
            yaxis=dict(
                tickmode='array',
                tickvals=list(range(top_n)),
                ticktext=top_features,
                autorange="reversed"
            )
        )
        
        fig.add_annotation(
            x=0.02, y=1.05,
            xref="paper", yref="paper",
            text="Blue = Increases risk | Red = Decreases risk",
            showarrow=False,
            font=dict(size=10, color=COLOR_SCHEME['text'])
        )
    except Exception as e:
        fig.update_layout(title=f"SHAP summary plot failed: {str(e)}")
    
    return fig


def create_shap_global_plot_classification():
    """Create global SHAP importance plot for classification model"""
    fig = go.Figure()
    
    if shap_data_precomputed is None:
        fig.update_layout(
            title="SHAP Analysis Not Available",
            plot_bgcolor=COLOR_SCHEME['background'],
            paper_bgcolor=COLOR_SCHEME['background'],
            font=dict(color=COLOR_SCHEME['text'])
        )
        return fig
    
    try:
        shap_values = shap_data_precomputed['shap_values']
        feature_names = shap_data_precomputed['feature_names']
        shap_importance = np.abs(shap_values).mean(axis=0)
        
        importance_df = (
            pd.DataFrame({
                "feature": feature_names,
                "importance": shap_importance
            })
            .sort_values("importance", ascending=False)
            .head(10)
        )
        
        fig.add_trace(go.Bar(
            x=importance_df["importance"],
            y=importance_df["feature"],
            orientation="h",
            marker_color=COLOR_SCHEME["secondary"],
            name="Mean |SHAP Value|"
        ))
        
        fig.update_layout(
            title="Global SHAP Feature Importance (Classification)",
            xaxis_title="Mean |SHAP Value|",
            yaxis_title="Features",
            plot_bgcolor=COLOR_SCHEME["background"],
            paper_bgcolor=COLOR_SCHEME["background"],
            font=dict(color=COLOR_SCHEME["text"]),
            height=350,
            yaxis=dict(autorange="reversed")
        )
    except Exception as e:
        fig.update_layout(title=f"SHAP calculation failed: {str(e)}")
    
    return fig


def create_regional_cluster_heatmap():
    """Create heatmap showing cluster distribution by region"""
    if clustering_analysis is None or 'regional_cluster_distribution' not in clustering_analysis:
        fig = go.Figure()
        fig.update_layout(
            title="Regional Cluster Distribution Heatmap",
            plot_bgcolor=COLOR_SCHEME['background'],
            paper_bgcolor=COLOR_SCHEME['background'],
            font=dict(color=COLOR_SCHEME['text']),
            height=500
        )
        return fig
    
    regional_cluster_df = clustering_analysis['regional_cluster_distribution']
    if regional_cluster_df.empty or len(regional_cluster_df.columns) == 0:
        fig = go.Figure()
        fig.update_layout(title="No regional cluster data available")
        return fig
    
    fig = go.Figure(data=go.Heatmap(
        z=regional_cluster_df.values,
        x=regional_cluster_df.columns,
        y=regional_cluster_df.index,
        colorscale='RdYlGn_r',
        text=regional_cluster_df.values.round(2),
        texttemplate='%{text:.2f}',
        textfont={"size": 12},
        hoverongaps=False,
        colorbar=dict(title="Proportion")
    ))
    fig.update_layout(
        title="Regional Cluster Distribution Heatmap",
        xaxis_title="Performance Cluster",
        yaxis_title="Region",
        plot_bgcolor=COLOR_SCHEME['background'],
        paper_bgcolor=COLOR_SCHEME['background'],
        font=dict(color=COLOR_SCHEME['text']),
        height=500,
        xaxis=dict(tickangle=0)
    )
    return fig


def create_regional_cluster_barchart():
    """Create stacked bar chart showing cluster distribution by region"""
    if clustering_analysis is None or 'regional_cluster_distribution' not in clustering_analysis:
        fig = go.Figure()
        fig.update_layout(
            title="Regional Cluster Distribution (Stacked)",
            plot_bgcolor=COLOR_SCHEME['background'],
            paper_bgcolor=COLOR_SCHEME['background'],
            font=dict(color=COLOR_SCHEME['text']),
            height=500
        )
        return fig
    
    regional_cluster_df = clustering_analysis['regional_cluster_distribution']
    if regional_cluster_df.empty or len(regional_cluster_df.columns) == 0:
        fig = go.Figure()
        fig.update_layout(title="No regional cluster data available")
        return fig
    
    fig = go.Figure()
    cluster_colors = {
        'Low': COLOR_SCHEME['danger'],
        'Medium': COLOR_SCHEME['warning'],
        'High': COLOR_SCHEME['success']
    }
    
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
        yaxis_title="Proportion",
        barmode='stack',
        plot_bgcolor=COLOR_SCHEME['background'],
        paper_bgcolor=COLOR_SCHEME['background'],
        font=dict(color=COLOR_SCHEME['text']),
        height=500,
        xaxis=dict(tickangle=-45),
        legend=dict(
            title="Performance Level",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    return fig


def create_recommendations_summary_table(df_clean_local, clustering_analysis_local, regression_models_local, 
                                         best_reg_model_local, classification_model_local):
    """Create comprehensive summary table for Recommendations & Summary dashboard"""
    total_students = len(df_clean_local)
    
    if 'Overall_Average' in df_clean_local.columns:
        risk_students = (df_clean_local['Overall_Average'] < 50).sum()
        avg_score = df_clean_local['Overall_Average'].mean()
        risk_percentage = (risk_students / total_students * 100) if total_students > 0 else 0
    else:
        risk_students = 0
        avg_score = 0
        risk_percentage = 0
    
    if clustering_analysis_local is not None:
        high_performers = clustering_analysis_local['cluster_sizes'].get('High', 0)
        medium_performers = clustering_analysis_local['cluster_sizes'].get('Medium', 0)
        low_performers = clustering_analysis_local['cluster_sizes'].get('Low', 0)
    else:
        high_performers = 0
        medium_performers = 0
        low_performers = 0
    
    if best_reg_model_local in regression_models_local:
        best_model_r2 = regression_models_local[best_reg_model_local]['r2']
        best_model_mae = regression_models_local[best_reg_model_local]['mae']
    else:
        best_model_r2 = 0
        best_model_mae = 0
    
    f1_score_val = classification_model_local.get('f1', 0)
    roc_auc_val = classification_model_local.get('roc_auc', 0)
    
    if clustering_analysis_local is not None and 'regional_risk' in clustering_analysis_local:
        regional_risk = clustering_analysis_local['regional_risk']
        if not regional_risk.empty:
            top_risk_region = regional_risk.idxmax()
            top_risk_value = regional_risk.max()
        else:
            top_risk_region = "N/A"
            top_risk_value = 0
    else:
        top_risk_region = "N/A"
        top_risk_value = 0
    
    summary_data = [
        {'Metric': 'Dataset Statistics', 'Value': '', 'Details': ''},
        {'Metric': ' Total Students', 'Value': f'{total_students:,}', 'Details': '100%'},
        {'Metric': ' At-Risk Students', 'Value': f'{risk_students:,}', 'Details': f'{risk_percentage:.1f}%'},
        {'Metric': ' Average Overall Score', 'Value': f'{avg_score:.2f}', 'Details': 'out of 100'},
        {'Metric': '', 'Value': '', 'Details': ''},
        {'Metric': 'Performance Clusters', 'Value': '', 'Details': ''},
        {'Metric': ' High Performers', 'Value': f'{high_performers:,}', 'Details': f'{(high_performers/total_students*100):.1f}%' if total_students > 0 else '0%'},
        {'Metric': ' Medium Performers', 'Value': f'{medium_performers:,}', 'Details': f'{(medium_performers/total_students*100):.1f}%' if total_students > 0 else '0%'},
        {'Metric': ' Low Performers', 'Value': f'{low_performers:,}', 'Details': f'{(low_performers/total_students*100):.1f}%' if total_students > 0 else '0%'},
        {'Metric': '', 'Value': '', 'Details': ''},
        {'Metric': 'Model Performance', 'Value': '', 'Details': ''},
        {'Metric': ' Best Regression Model', 'Value': best_reg_model_local, 'Details': f'R² = {best_model_r2:.3f}'},
        {'Metric': ' Regression MAE', 'Value': f'{best_model_mae:.2f}', 'Details': 'avg error in points'},
        {'Metric': ' Risk Classification F1', 'Value': f'{f1_score_val:.3f}', 'Details': '>0.75 is good'},
        {'Metric': ' Risk Classification ROC-AUC', 'Value': f'{roc_auc_val:.3f}', 'Details': '>0.80 is excellent'},
        {'Metric': '', 'Value': '', 'Details': ''},
        {'Metric': 'National Exam Model', 'Value': '', 'Details': ''},
        {'Metric': ' Best Model', 'Value': 'Gradient Boosting', 'Details': 'R² = 0.4380'},
        {'Metric': ' Top Feature', 'Value': 'Score_x_Participation', 'Details': '73.6% importance'},
        {'Metric': '', 'Value': '', 'Details': ''},
        {'Metric': 'Regional Risk Analysis', 'Value': '', 'Details': ''},
        {'Metric': ' Highest Risk Region', 'Value': top_risk_region, 'Details': f'{top_risk_value:.1f}% low performers'},
        {'Metric': ' Lowest Risk Region', 'Value': 'Addis Ababa', 'Details': '21.3% low performers'},
        {'Metric': '', 'Value': '', 'Details': ''},
        {'Metric': 'Key Predictors', 'Value': '', 'Details': ''},
        {'Metric': ' Top Factor', 'Value': 'School Resources Score', 'Details': 'Strongest predictor'},
        {'Metric': ' 2nd Factor', 'Value': 'Textbook Access', 'Details': 'Critical for learning'},
        {'Metric': ' 3rd Factor', 'Value': 'Parental Involvement', 'Details': 'Significant impact'},
        {'Metric': ' 4th Factor', 'Value': 'Teacher-Student Ratio', 'Details': 'Lower is better'},
    ]
    
    return pd.DataFrame(summary_data)


def set_prediction_result(pred_result):
    """Set prediction result for visualizations"""
    global prediction_result
    prediction_result = pred_result   #r"C:\Users\DELL\projects\project1\Students-Acadamic-Performance-Analysis\ethiopian_students_dataset.csv"