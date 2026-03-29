# utils/visualizations.py
"""
Visualization utilities for Ethiopian Student Performance Dashboard
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve
import warnings
warnings.filterwarnings('ignore')


class Visualizer:
    """Visualization class for all plots"""
    
    COLOR_SCHEME = {
        'primary': '#2E86AB',
        'secondary': '#A23B72',
        'success': '#18A999',
        'warning': '#F18F01',
        'danger': '#C73E1D',
        'light': '#F0F3F5',
        'dark': '#2C3E50',
        'background': '#FFFFFF',
        'text': '#2C3E50'
    }
    
    @staticmethod
    def create_score_histogram(df, column='Overall_Average'):
        """Create distribution plot for overall average scores"""
        if column not in df.columns:
            fig = go.Figure()
            fig.update_layout(title="No data available")
            return fig
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df[column].dropna(),
            nbinsx=30,
            marker_color=Visualizer.COLOR_SCHEME['primary'],
            opacity=0.7,
            name='Score Distribution'
        ))
        
        mean_score = df[column].mean()
        fig.add_vline(x=mean_score, line_dash="dash", line_color=Visualizer.COLOR_SCHEME['danger'])
        fig.add_vline(x=50, line_dash="dash", line_color=Visualizer.COLOR_SCHEME['warning'])
        
        fig.update_layout(
            title="Distribution of Overall Average Scores",
            xaxis_title="Overall Average Score",
            yaxis_title="Number of Students",
            plot_bgcolor=Visualizer.COLOR_SCHEME['background'],
            paper_bgcolor=Visualizer.COLOR_SCHEME['background'],
            height=450,
            showlegend=True
        )
        return fig
    
    @staticmethod
    def create_risk_bar(df):
        """Create risk distribution bar chart"""
        if 'Overall_Average' not in df.columns:
            return go.Figure()
        
        risk_count = (df['Overall_Average'] < 50).sum()
        not_risk_count = len(df) - risk_count
        total = len(df)
        
        fig = go.Figure(data=[
            go.Bar(
                name='At Risk', 
                x=['Risk Status'], 
                y=[risk_count],
                marker_color=Visualizer.COLOR_SCHEME['danger'],
                text=[f"{risk_count} ({risk_count/total*100:.1f}%)"], 
                textposition='auto'
            ),
            go.Bar(
                name='Not at Risk', 
                x=['Risk Status'], 
                y=[not_risk_count],
                marker_color=Visualizer.COLOR_SCHEME['success'],
                text=[f"{not_risk_count} ({not_risk_count/total*100:.1f}%)"], 
                textposition='auto'
            )
        ])
        fig.update_layout(
            title="Student Risk Distribution",
            xaxis_title="Risk Category",
            yaxis_title="Number of Students",
            plot_bgcolor=Visualizer.COLOR_SCHEME['background'],
            paper_bgcolor=Visualizer.COLOR_SCHEME['background'],
            height=400,
            barmode='group'
        )
        return fig
    
    @staticmethod
    def create_region_bar(df):
        """Create average score by region bar chart"""
        if 'Region' not in df.columns or 'Overall_Average' not in df.columns:
            return go.Figure()
        
        region_avg = df.groupby('Region')['Overall_Average'].mean().sort_values()
        
        fig = go.Figure(data=[
            go.Bar(
                x=region_avg.values,
                y=region_avg.index,
                orientation='h',
                marker_color=Visualizer.COLOR_SCHEME['primary'],
                text=[f'{v:.1f}' for v in region_avg.values],
                textposition='auto'
            )
        ])
        fig.update_layout(
            title="Average Score by Region",
            xaxis_title="Average Score",
            yaxis_title="Region",
            plot_bgcolor=Visualizer.COLOR_SCHEME['background'],
            paper_bgcolor=Visualizer.COLOR_SCHEME['background'],
            height=500,
            margin=dict(l=120)
        )
        return fig
    
    @staticmethod
    def create_gender_bar(df):
        """Create average score by gender bar chart"""
        if 'Gender' not in df.columns or 'Overall_Average' not in df.columns:
            return go.Figure()
        
        gender_avg = df.groupby('Gender')['Overall_Average'].mean()
        
        if gender_avg.index.dtype in ['int64', 'float64']:
            labels = ['Male' if x == 0 else 'Female' for x in gender_avg.index]
        else:
            labels = [str(x) for x in gender_avg.index]
        
        fig = go.Figure(data=[
            go.Bar(
                x=labels,
                y=gender_avg.values,
                marker_color=Visualizer.COLOR_SCHEME['primary'],
                text=[f'{v:.1f}' for v in gender_avg.values],
                textposition='auto'
            )
        ])
        fig.update_layout(
            title="Average Score by Gender",
            xaxis_title="Gender",
            yaxis_title="Average Score",
            plot_bgcolor=Visualizer.COLOR_SCHEME['background'],
            paper_bgcolor=Visualizer.COLOR_SCHEME['background'],
            height=400
        )
        return fig
    
    @staticmethod
    def create_correlation_heatmap(df):
        """Create correlation heatmap for top features"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['Risk_NotRisk', 'Student_ID']
        numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
        
        if len(numeric_cols) > 15:
            if 'Overall_Average' in numeric_cols:
                corr = df[numeric_cols].corr()['Overall_Average'].abs().sort_values(ascending=False)
                top_features = corr.head(15).index.tolist()
                corr_data = df[top_features].corr()
            else:
                corr_data = df[numeric_cols[:15]].corr()
        else:
            corr_data = df[numeric_cols].corr()
        
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
            plot_bgcolor=Visualizer.COLOR_SCHEME['background'],
            paper_bgcolor=Visualizer.COLOR_SCHEME['background'],
            height=600,
            xaxis_tickangle=-45
        )
        return fig
    
    @staticmethod
    def create_scatter_plot(df, x_col, y_col):
        """Create scatter plot for two variables"""
        if x_col not in df.columns or y_col not in df.columns:
            return go.Figure()
        
        plot_df = df[[x_col, y_col]].dropna()
        
        if len(plot_df) == 0:
            return go.Figure()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=plot_df[x_col],
            y=plot_df[y_col],
            mode='markers',
            marker=dict(color=Visualizer.COLOR_SCHEME['primary'], size=6, opacity=0.6),
            name=f'{y_col} vs {x_col}'
        ))
        
        fig.update_layout(
            title=f"{y_col} vs {x_col}",
            xaxis_title=x_col,
            yaxis_title=y_col,
            plot_bgcolor=Visualizer.COLOR_SCHEME['background'],
            paper_bgcolor=Visualizer.COLOR_SCHEME['background'],
            height=400
        )
        return fig
    
    @staticmethod
    def create_boxplot(df, x_col, y_col):
        """Create boxplot by category"""
        if x_col not in df.columns or y_col not in df.columns:
            return go.Figure()
        
        fig = go.Figure()
        
        categories = df[x_col].unique()
        for cat in sorted(categories):
            data = df[df[x_col] == cat][y_col].dropna()
            if len(data) > 0:
                fig.add_trace(go.Box(
                    y=data,
                    name=str(cat),
                    marker_color=Visualizer.COLOR_SCHEME['primary'],
                    boxmean='sd'
                ))
        
        fig.update_layout(
            title=f"{y_col} Distribution by {x_col}",
            xaxis_title=x_col,
            yaxis_title=y_col,
            plot_bgcolor=Visualizer.COLOR_SCHEME['background'],
            paper_bgcolor=Visualizer.COLOR_SCHEME['background'],
            height=400
        )
        return fig
    
    @staticmethod
    def create_regression_comparison_plot(regression_metrics):
        """Create comparison plot for regression models"""
        if not regression_metrics:
            return go.Figure()
        
        models = list(regression_metrics.keys())
        r2_scores = [regression_metrics[m]['R2'] for m in models]
        
        fig = go.Figure(data=[
            go.Bar(
                x=models, 
                y=r2_scores,
                marker_color=Visualizer.COLOR_SCHEME['primary'],
                text=[f'{score:.3f}' for score in r2_scores],
                textposition='auto'
            )
        ])
        fig.update_layout(
            title="Regression Model R² Score Comparison",
            xaxis_title="Models",
            yaxis_title="R² Score",
            plot_bgcolor=Visualizer.COLOR_SCHEME['background'],
            paper_bgcolor=Visualizer.COLOR_SCHEME['background'],
            height=400
        )
        return fig
    
    @staticmethod
    def create_feature_importance_plot(importance_dict, title="Feature Importance"):
        """Create feature importance bar chart"""
        if not importance_dict:
            return go.Figure()
        
        importance_df = pd.DataFrame(list(importance_dict.items()), columns=['Feature', 'Importance'])
        importance_df = importance_df.sort_values('Importance', ascending=True).tail(10)
        
        fig = go.Figure(go.Bar(
            x=importance_df['Importance'],
            y=importance_df['Feature'],
            orientation='h',
            marker_color=Visualizer.COLOR_SCHEME['primary'],
            text=[f'{imp:.3f}' for imp in importance_df['Importance']],
            textposition='auto'
        ))
        fig.update_layout(
            title=title,
            xaxis_title="Importance Score",
            yaxis_title="Features",
            plot_bgcolor=Visualizer.COLOR_SCHEME['background'],
            paper_bgcolor=Visualizer.COLOR_SCHEME['background'],
            height=450,
            margin=dict(l=150)
        )
        return fig
    
    @staticmethod
    def create_confusion_matrix(cm):
        """Create confusion matrix heatmap"""
        if cm is None:
            return go.Figure()
        
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
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            plot_bgcolor=Visualizer.COLOR_SCHEME['background'],
            paper_bgcolor=Visualizer.COLOR_SCHEME['background'],
            height=400
        )
        return fig
    
    @staticmethod
    def create_roc_curve(fpr, tpr, roc_auc):
        """Create ROC curve plot"""
        if fpr is None or tpr is None:
            return go.Figure()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.3f})',
            line=dict(color=Visualizer.COLOR_SCHEME['primary'], width=2),
            fill='tozeroy',
            fillcolor='rgba(46, 134, 171, 0.2)'
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color=Visualizer.COLOR_SCHEME['secondary'], width=2)
        ))
        fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            plot_bgcolor=Visualizer.COLOR_SCHEME['background'],
            paper_bgcolor=Visualizer.COLOR_SCHEME['background'],
            height=450
        )
        return fig
    
    @staticmethod
    def create_cluster_distribution_plot(cluster_sizes):
        """Create cluster distribution plot"""
        if not cluster_sizes:
            return go.Figure()
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(cluster_sizes.keys()),
                y=list(cluster_sizes.values()),
                marker_color=[Visualizer.COLOR_SCHEME['success'], 
                              Visualizer.COLOR_SCHEME['warning'], 
                              Visualizer.COLOR_SCHEME['danger']],
                text=list(cluster_sizes.values()),
                textposition='auto'
            )
        ])
        fig.update_layout(
            title="Student Performance Cluster Distribution",
            xaxis_title="Performance Level",
            yaxis_title="Number of Students",
            plot_bgcolor=Visualizer.COLOR_SCHEME['background'],
            paper_bgcolor=Visualizer.COLOR_SCHEME['background'],
            height=400
        )
        return fig
    
    @staticmethod
    def create_regional_risk_plot(regional_risk):
        """Create regional risk analysis plot"""
        if not regional_risk:
            return go.Figure()
        
        risk_df = pd.DataFrame(list(regional_risk.items()), columns=['Region', 'Risk %'])
        risk_df = risk_df.sort_values('Risk %', ascending=True)
        
        fig = go.Figure(data=[
            go.Bar(
                x=risk_df['Risk %'],
                y=risk_df['Region'],
                orientation='h',
                marker_color=Visualizer.COLOR_SCHEME['danger'],
                text=[f'{v:.1f}%' for v in risk_df['Risk %']],
                textposition='auto'
            )
        ])
        fig.update_layout(
            title="Regional Risk Analysis (% Low Performance)",
            xaxis_title="% Low Performance",
            yaxis_title="Region",
            plot_bgcolor=Visualizer.COLOR_SCHEME['background'],
            paper_bgcolor=Visualizer.COLOR_SCHEME['background'],
            height=550,
            margin=dict(l=150)
        )
        return fig
    
    @staticmethod
    def create_risk_gauge(risk_prob):
        """Create risk gauge chart"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_prob * 100,
            title={'text': "Risk Probability (%)"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': Visualizer.COLOR_SCHEME['danger'] if risk_prob > 0.5 else Visualizer.COLOR_SCHEME['success']},
                'steps': [
                    {'range': [0, 30], 'color': Visualizer.COLOR_SCHEME['success']},
                    {'range': [30, 70], 'color': Visualizer.COLOR_SCHEME['warning']},
                    {'range': [70, 100], 'color': Visualizer.COLOR_SCHEME['danger']}
                ],
                'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 50}
            }
        ))
        fig.update_layout(height=250)
        return fig


# ============================================================================
# GLOBAL FUNCTIONS (for compatibility with __init__.py)
# ============================================================================

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
    global NATIONAL_EXAM_MODEL_PERFORMANCE, NATIONAL_EXAM_FEATURE_IMPORTANCE
    
    NATIONAL_EXAM_MODEL_PERFORMANCE = pd.DataFrame({
        'Model': ['Gradient Boosting', 'XGBoost', 'Random Forest',
                  'Ridge Regression', 'Linear Regression', 'Lasso Regression'],
        'R2_Score': [0.437997, 0.435344, 0.425839, 0.405843, 0.405831, 0.404305],
        'MAE': [0.081404, 0.081569, 0.082253, 0.083816, 0.083816, 0.083893],
        'RMSE': [0.107109, 0.107362, 0.108262, 0.110131, 0.110132, 0.110273]
    })
    
    NATIONAL_EXAM_FEATURE_IMPORTANCE = pd.DataFrame({
        'Feature': [
            'Score_x_Participation', 'Overall_Avg_Homework', 'School_Academic_Score',
            'Overall_Test_Score_Avg', 'Overall_Avg_Attendance', 'Overall_Avg_Participation',
            'School_Resources_Score', 'Parental_Involvement', 'Resource_Efficiency',
            'Teacher_Student_Ratio', 'Student_to_Resources_Ratio', 'School_Type_Target',
            'Overall_Engagement_Score', 'Teacher_Load_Adjusted',
            'Overall_Textbook_Access_Composite', 'Field_Choice', 'Career_Interest_Encoded'
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
    
    return NATIONAL_EXAM_MODEL_PERFORMANCE, NATIONAL_EXAM_FEATURE_IMPORTANCE


def set_global_data(reg_models, best_model, feat_importances, class_model, 
                    cluster_analysis, shap_data, raw_df, clean_df):
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


def set_prediction_result(pred_result):
    """Set prediction result for visualizations"""
    global prediction_result
    prediction_result = pred_result


def create_datatype_bar_plot():
    """Create bar plot for data types distribution"""
    if df_raw is None:
        return go.Figure()
    dtypes = df_raw.dtypes.value_counts()
    fig = go.Figure(data=[
        go.Bar(
            x=dtypes.index.astype(str),
            y=dtypes.values,
            marker_color=Visualizer.COLOR_SCHEME['primary'],
            text=dtypes.values,
            textposition='auto'
        )
    ])
    fig.update_layout(
        title="Data Type Distribution",
        xaxis_title="Data Type",
        yaxis_title="Count",
        plot_bgcolor=Visualizer.COLOR_SCHEME['background'],
        paper_bgcolor=Visualizer.COLOR_SCHEME['background'],
        font=dict(color=Visualizer.COLOR_SCHEME['text'])
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
    colors = [Visualizer.COLOR_SCHEME['primary'], Visualizer.COLOR_SCHEME['secondary'], 
              Visualizer.COLOR_SCHEME['success'], Visualizer.COLOR_SCHEME['warning'], 
              Visualizer.COLOR_SCHEME['danger'], Visualizer.COLOR_SCHEME['dark']]
    
    for category, features in feature_categories.items():
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
        plot_bgcolor=Visualizer.COLOR_SCHEME['background'],
        paper_bgcolor=Visualizer.COLOR_SCHEME['background'],
        font=dict(color=Visualizer.COLOR_SCHEME['text']),
        xaxis_tickangle=-45
    )
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
        marker_color=[Visualizer.COLOR_SCHEME['success'] if 'Gradient' in model else Visualizer.COLOR_SCHEME['primary'] 
                      for model in sorted_df['Model']],
        text=[f'{score:.4f}' for score in sorted_df['R2_Score']],
        textposition='auto',
        name='R² Score'
    ))
    fig.update_layout(
        title="National Exam Score Model Performance (R² Score)",
        xaxis_title="R² Score",
        yaxis_title="Model",
        plot_bgcolor=Visualizer.COLOR_SCHEME['background'],
        paper_bgcolor=Visualizer.COLOR_SCHEME['background'],
        font=dict(color=Visualizer.COLOR_SCHEME['text']),
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
        marker_color=Visualizer.COLOR_SCHEME['secondary'],
        text=[f'{imp:.1%}' for imp in sorted_df['Importance']],
        textposition='auto'
    ))
    fig.update_layout(
        title="Feature Importance - National Exam Score (Gradient Boosting)",
        xaxis_title="Importance Score",
        yaxis_title="Features",
        plot_bgcolor=Visualizer.COLOR_SCHEME['background'],
        paper_bgcolor=Visualizer.COLOR_SCHEME['background'],
        font=dict(color=Visualizer.COLOR_SCHEME['text']),
        height=500,
        xaxis=dict(range=[0, max(sorted_df['Importance']) * 1.15])
    )
    return fig


def create_national_exam_performance_table():
    """Create performance table for National Exam Score models"""
    if NATIONAL_EXAM_MODEL_PERFORMANCE is None:
        return pd.DataFrame()
    return NATIONAL_EXAM_MODEL_PERFORMANCE.copy()


def create_actual_vs_predicted_plot():
    """Create actual vs predicted plot for best regression model"""
    if best_reg_model not in regression_models or regression_models[best_reg_model].get('y_test') is None:
        return go.Figure()
    
    best_reg = regression_models[best_reg_model]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=best_reg['y_test'],
        y=best_reg['y_pred'],
        mode='markers',
        name='Predictions',
        marker=dict(color=Visualizer.COLOR_SCHEME['primary'], size=6, opacity=0.6)
    ))
    fig.add_trace(go.Scatter(
        x=[best_reg['y_test'].min(), best_reg['y_test'].max()],
        y=[best_reg['y_test'].min(), best_reg['y_test'].max()],
        mode='lines',
        name='Perfect Prediction',
        line=dict(dash='dash', color=Visualizer.COLOR_SCHEME['secondary'], width=2)
    ))
    fig.update_layout(
        title=f"Actual vs Predicted Overall Average - {best_reg_model}",
        xaxis_title="Actual Overall Average",
        yaxis_title="Predicted Overall Average",
        plot_bgcolor=Visualizer.COLOR_SCHEME['background'],
        paper_bgcolor=Visualizer.COLOR_SCHEME['background'],
        font=dict(color=Visualizer.COLOR_SCHEME['text'])
    )
    return fig


def create_roc_curve_plot():
    """Create ROC curve plot"""
    if classification_model is None or classification_model.get('y_test') is None:
        return go.Figure()
    
    fpr, tpr, _ = roc_curve(classification_model['y_test'], classification_model['y_probs'])
    roc_auc = classification_model['roc_auc']
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.3f})',
        line=dict(color=Visualizer.COLOR_SCHEME['primary'], width=2)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(dash='dash', color=Visualizer.COLOR_SCHEME['secondary'], width=1)
    ))
    fig.update_layout(
        title="ROC Curve - Risk Classification",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        plot_bgcolor=Visualizer.COLOR_SCHEME['background'],
        paper_bgcolor=Visualizer.COLOR_SCHEME['background'],
        font=dict(color=Visualizer.COLOR_SCHEME['text']),
        height=400
    )
    return fig


def create_shap_summary_plot():
    """Create SHAP summary plot"""
    fig = go.Figure()
    fig.update_layout(
        title="SHAP Analysis Not Available",
        plot_bgcolor=Visualizer.COLOR_SCHEME['background'],
        paper_bgcolor=Visualizer.COLOR_SCHEME['background']
    )
    return fig


def create_shap_global_plot_classification():
    """Create global SHAP importance plot"""
    fig = go.Figure()
    fig.update_layout(
        title="SHAP Analysis Not Available",
        plot_bgcolor=Visualizer.COLOR_SCHEME['background'],
        paper_bgcolor=Visualizer.COLOR_SCHEME['background']
    )
    return fig


def create_regional_cluster_heatmap():
    """Create regional cluster heatmap"""
    if clustering_analysis is None:
        return go.Figure()
    
    regional_cluster_df = clustering_analysis.get('regional_cluster_distribution')
    if regional_cluster_df is None or regional_cluster_df.empty:
        return go.Figure()
    
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
        plot_bgcolor=Visualizer.COLOR_SCHEME['background'],
        paper_bgcolor=Visualizer.COLOR_SCHEME['background'],
        height=500
    )
    return fig


def create_regional_cluster_barchart():
    """Create stacked bar chart showing cluster distribution by region"""
    if clustering_analysis is None:
        return go.Figure()
    
    regional_cluster_df = clustering_analysis.get('regional_cluster_distribution')
    if regional_cluster_df is None or regional_cluster_df.empty:
        return go.Figure()
    
    fig = go.Figure()
    cluster_colors = {
        'Low': Visualizer.COLOR_SCHEME['danger'],
        'Medium': Visualizer.COLOR_SCHEME['warning'],
        'High': Visualizer.COLOR_SCHEME['success']
    }
    
    for cluster in ['Low', 'Medium', 'High']:
        if cluster in regional_cluster_df.columns:
            fig.add_trace(go.Bar(
                name=cluster,
                x=regional_cluster_df.index,
                y=regional_cluster_df[cluster],
                marker_color=cluster_colors[cluster],
                text=regional_cluster_df[cluster].round(2),
                textposition='inside'
            ))
    
    fig.update_layout(
        title="Regional Cluster Distribution (Stacked)",
        xaxis_title="Region",
        yaxis_title="Proportion",
        barmode='stack',
        plot_bgcolor=Visualizer.COLOR_SCHEME['background'],
        paper_bgcolor=Visualizer.COLOR_SCHEME['background'],
        height=500,
        xaxis_tickangle=-45,
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


def create_feature_summary_table():
    """Create summary statistics table for each feature"""
    if df_raw is None:
        return pd.DataFrame()
    
    numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    stats_dict = {}
    for col in numeric_cols[:15]:
        stats_dict[col] = {
            'Mean': f"{df_raw[col].mean():.2f}",
            'Std': f"{df_raw[col].std():.2f}",
            'Min': f"{df_raw[col].min():.2f}",
            'Median': f"{df_raw[col].median():.2f}",
            'Max': f"{df_raw[col].max():.2f}",
            'Missing': f"{df_raw[col].isnull().sum()}"
        }
    
    stats_list = ['Mean', 'Std', 'Min', 'Median', 'Max', 'Missing']
    summary_data = []
    for stat in stats_list:
        row = {'Statistic': stat}
        for col in list(stats_dict.keys()):
            row[col] = stats_dict[col][stat]
        summary_data.append(row)
    
    return pd.DataFrame(summary_data)


def create_risk_distribution_plot():
    """Create risk distribution plot for prediction"""
    global prediction_result
    if prediction_result is None:
        return None
    
    fig = go.Figure(data=[
        go.Bar(
            x=['Risk', 'Not Risk'],
            y=[1 if prediction_result['is_risk'] else 0, 0 if prediction_result['is_risk'] else 1],
            marker_color=[Visualizer.COLOR_SCHEME['danger'], Visualizer.COLOR_SCHEME['success']],
            text=[f'{prediction_result["risk_probability"]*100:.1f}%', f'{(1-prediction_result["risk_probability"])*100:.1f}%'],
            textposition='auto'
        )
    ])
    fig.update_layout(
        title="Predicted Risk Distribution",
        xaxis_title="Risk Category",
        yaxis_title="Count",
        height=300
    )
    return fig


def create_recommendations_summary_table(df_clean_local, clustering_analysis_local, 
                                         regression_models_local, best_reg_model_local, 
                                         classification_model_local):
    """Create summary table for recommendations page"""
    total = len(df_clean_local)
    risk = (df_clean_local['Overall_Average'] < 50).sum() if 'Overall_Average' in df_clean_local.columns else 0
    avg = df_clean_local['Overall_Average'].mean() if 'Overall_Average' in df_clean_local.columns else 0
    
    cluster_sizes = clustering_analysis_local.get('cluster_sizes', {}) if clustering_analysis_local else {}
    high = cluster_sizes.get('High', 0)
    medium = cluster_sizes.get('Medium', 0)
    low = cluster_sizes.get('Low', 0)
    
    summary_data = [
        {'Metric': 'Total Students', 'Value': f'{total:,}', 'Details': '100%'},
        {'Metric': 'At-Risk Students', 'Value': f'{risk:,}', 'Details': f'{(risk/total*100):.1f}%'},
        {'Metric': 'Average Score', 'Value': f'{avg:.2f}', 'Details': 'out of 100'},
        {'Metric': 'High Performers', 'Value': f'{high:,}', 'Details': f'{(high/total*100):.1f}%'},
        {'Metric': 'Medium Performers', 'Value': f'{medium:,}', 'Details': f'{(medium/total*100):.1f}%'},
        {'Metric': 'Low Performers', 'Value': f'{low:,}', 'Details': f'{(low/total*100):.1f}%'},
        {'Metric': 'Best Regression Model', 'Value': 'XGBoost', 'Details': 'R² = 0.7855'},
        {'Metric': 'Risk Classification F1', 'Value': '0.778', 'Details': '>0.75 is good'},
        {'Metric': 'Risk Classification ROC-AUC', 'Value': '0.918', 'Details': '>0.80 is excellent'},
        {'Metric': 'Highest Risk Region', 'Value': 'Somali', 'Details': '47.4% low performers'},
        {'Metric': 'Lowest Risk Region', 'Value': 'Addis Ababa', 'Details': '21.3% low performers'}
    ]
    
    return pd.DataFrame(summary_data)


def create_score_distribution_plot():
    """Create distribution plot for overall average scores"""
    if df_clean is None or 'Overall_Average' not in df_clean.columns:
        return go.Figure()
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df_clean['Overall_Average'],
        nbinsx=30,
        marker_color=Visualizer.COLOR_SCHEME['primary'],
        opacity=0.7
    ))
    fig.update_layout(
        title="Distribution of Overall Average Scores",
        xaxis_title="Overall Average Score",
        yaxis_title="Number of Students",
        plot_bgcolor=Visualizer.COLOR_SCHEME['background'],
        paper_bgcolor=Visualizer.COLOR_SCHEME['background'],
        font=dict(color=Visualizer.COLOR_SCHEME['text'])
    )
    return fig


def create_students_by_region_plot():
    """Create bar plot showing number of students by region"""
    if df_raw is None or 'Region' not in df_raw.columns:
        return go.Figure()
    
    region_counts = df_raw['Region'].value_counts().sort_values(ascending=True)
    fig = go.Figure(data=[
        go.Bar(
            y=region_counts.index,
            x=region_counts.values,
            orientation='h',
            marker_color=Visualizer.COLOR_SCHEME['primary'],
            text=region_counts.values,
            textposition='auto'
        )
    ])
    fig.update_layout(
        title="Number of Students by Region",
        xaxis_title="Number of Students",
        yaxis_title="Region",
        plot_bgcolor=Visualizer.COLOR_SCHEME['background'],
        paper_bgcolor=Visualizer.COLOR_SCHEME['background'],
        font=dict(color=Visualizer.COLOR_SCHEME['text']),
        height=500
    )
    return fig


def create_confusion_matrix_plot():
    """Create confusion matrix plot for classification"""
    if classification_model is None or classification_model.get('cm') is None:
        return go.Figure()
    
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
        plot_bgcolor=Visualizer.COLOR_SCHEME['background'],
        paper_bgcolor=Visualizer.COLOR_SCHEME['background'],
        font=dict(color=Visualizer.COLOR_SCHEME['text'])
    )
    return fig


def create_cluster_distribution_plot():
    """Create cluster distribution plot"""
    if clustering_analysis is None or clustering_analysis.get('cluster_sizes') is None:
        return go.Figure()
    
    cluster_counts = clustering_analysis['cluster_sizes']
    colors = {
        'High': Visualizer.COLOR_SCHEME['success'],
        'Medium': Visualizer.COLOR_SCHEME['warning'],
        'Low': Visualizer.COLOR_SCHEME['danger']
    }
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(cluster_counts.keys()),
            y=list(cluster_counts.values()),
            marker_color=[colors.get(label, Visualizer.COLOR_SCHEME['primary']) for label in cluster_counts.keys()],
            text=list(cluster_counts.values()),
            textposition='auto'
        )
    ])
    fig.update_layout(
        title="Student Performance Cluster Distribution",
        xaxis_title="Performance Level",
        yaxis_title="Number of Students",
        plot_bgcolor=Visualizer.COLOR_SCHEME['background'],
        paper_bgcolor=Visualizer.COLOR_SCHEME['background'],
        font=dict(color=Visualizer.COLOR_SCHEME['text'])
    )
    return fig


def create_regional_risk_plot():
    """Create regional risk analysis plot"""
    if clustering_analysis is None or clustering_analysis.get('regional_risk') is None:
        return go.Figure()
    
    regional_risk = clustering_analysis['regional_risk']
    if regional_risk.empty:
        return go.Figure()
    
    fig = go.Figure(data=[
        go.Bar(
            x=regional_risk.index,
            y=regional_risk.values,
            marker_color=Visualizer.COLOR_SCHEME['danger'],
            text=[f'{val:.1f}%' for val in regional_risk.values],
            textposition='auto'
        )
    ])
    fig.update_layout(
        title="Regional Risk Analysis (% Low Performance)",
        xaxis_title="Region",
        yaxis_title="% Low Performance",
        plot_bgcolor=Visualizer.COLOR_SCHEME['background'],
        paper_bgcolor=Visualizer.COLOR_SCHEME['background'],
        font=dict(color=Visualizer.COLOR_SCHEME['text']),
        xaxis_tickangle=-45
    )
    return fig
# =============================================================================
# MODULE WRAPPERS FOR Visualizer STATIC METHODS (required by utils/__init__.py)
# =============================================================================

def create_correlation_heatmap(df):
    return Visualizer.create_correlation_heatmap(df)

def create_scatter_plot(df, x_col, y_col):
    return Visualizer.create_scatter_plot(df, x_col, y_col)

def create_boxplot(df, x_col, y_col):
    return Visualizer.create_boxplot(df, x_col, y_col)

def create_regression_comparison_plot(regression_metrics):
    return Visualizer.create_regression_comparison_plot(regression_metrics)

def create_feature_importance_plot(importance_dict, title="Feature Importance"):
    return Visualizer.create_feature_importance_plot(importance_dict, title)

def create_risk_gauge(risk_prob):
    return Visualizer.create_risk_gauge(risk_prob)