# utils/visualizations.py
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

class Visualizer:
    """Visualization class for all plots"""
    
    COLOR_SCHEME = {
        'primary': '#2E86AB',
        'secondary': '#A23B72',
        'success': '#18A999',
        'warning': '#F18F01',
        'danger': '#C73E1D',
        'light': '#F0F3F5',
        'dark': '#2C3E50'
    }
    
    @staticmethod
    def create_score_histogram(df, column='Overall_Average'):
        """Create score distribution histogram"""
        if column not in df.columns:
            return go.Figure()
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df[column],
            nbinsx=30,
            marker_color=Visualizer.COLOR_SCHEME['primary'],
            opacity=0.7,
            name='Score Distribution'
        ))
        
        # Add vertical lines
        mean_score = df[column].mean()
        fig.add_vline(x=mean_score, line_dash="dash", line_color=Visualizer.COLOR_SCHEME['danger'],
                      annotation_text=f"Mean: {mean_score:.1f}")
        fig.add_vline(x=50, line_dash="dash", line_color=Visualizer.COLOR_SCHEME['warning'],
                      annotation_text="Pass Threshold")
        
        fig.update_layout(
            title="Distribution of Overall Average Scores",
            xaxis_title="Score",
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
        
        fig = go.Figure(data=[
            go.Bar(name='At Risk', x=['Risk Status'], y=[risk_count],
                   marker_color=Visualizer.COLOR_SCHEME['danger'],
                   text=[f"{risk_count} ({risk_count/len(df)*100:.1f}%)"], 
                   textposition='auto'),
            go.Bar(name='Not at Risk', x=['Risk Status'], y=[not_risk_count],
                   marker_color=Visualizer.COLOR_SCHEME['success'],
                   text=[f"{not_risk_count} ({not_risk_count/len(df)*100:.1f}%)"], 
                   textposition='auto')
        ])
        fig.update_layout(
            title="Student Risk Distribution",
            xaxis_title="Risk Category",
            yaxis_title="Number of Students",
            height=400,
            barmode='group',
            showlegend=True
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
        gender_map = {0: 'Male', 1: 'Female'}
        gender_avg.index = gender_avg.index.map(gender_map)
        
        fig = go.Figure(data=[
            go.Bar(
                x=gender_avg.index,
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
            height=400
        )
        return fig
    
    @staticmethod
    def create_correlation_heatmap(df):
        """Create correlation heatmap"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target leakage columns
        exclude_cols = ['Risk_NotRisk', 'Health_Issue_Target', 'School_Type_Target']
        numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
        
        if len(numeric_cols) > 20:
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
            textfont={"size": 10},
            hoverongaps=False
        ))
        fig.update_layout(
            title="Feature Correlation Heatmap",
            xaxis_title="Features",
            yaxis_title="Features",
            height=600,
            xaxis_tickangle=-45
        )
        return fig
    
    @staticmethod
    def create_scatter_plot(df, x_col, y_col):
        """Create scatter plot for two variables"""
        if x_col not in df.columns or y_col not in df.columns:
            return go.Figure()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode='markers',
            marker=dict(color=Visualizer.COLOR_SCHEME['primary'], size=6, opacity=0.6),
            name=f'{y_col} vs {x_col}'
        ))
        
        # Add trend line
        z = np.polyfit(df[x_col], df[y_col], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(df[x_col].min(), df[x_col].max(), 100)
        fig.add_trace(go.Scatter(
            x=x_trend,
            y=p(x_trend),
            mode='lines',
            name='Trend Line',
            line=dict(color=Visualizer.COLOR_SCHEME['secondary'], dash='dash')
        ))
        
        fig.update_layout(
            title=f"{y_col} vs {x_col}",
            xaxis_title=x_col,
            yaxis_title=y_col,
            height=400
        )
        return fig
    
    @staticmethod
    def create_feature_importance_plot(importance_dict, title="Feature Importance"):
        """Create feature importance bar chart"""
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
            height=450,
            margin=dict(l=150)
        )
        return fig
    
    @staticmethod
    def create_confusion_matrix(cm):
        """Create confusion matrix heatmap"""
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
            title="Confusion Matrix - Risk Classification",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=400
        )
        return fig
    
    @staticmethod
    def create_roc_curve(fpr, tpr, roc_auc):
        """Create ROC curve plot"""
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
            name='Random Classifier (AUC = 0.5)',
            line=dict(dash='dash', color=Visualizer.COLOR_SCHEME['secondary'], width=2)
        ))
        fig.update_layout(
            title="ROC Curve - Risk Classification",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=450,
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1.05])
        )
        return fig
    
    @staticmethod
    def create_cluster_distribution_plot(cluster_sizes):
        """Create cluster distribution plot"""
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
            height=400
        )
        return fig
    
    @staticmethod
    def create_regional_risk_plot(regional_risk):
        """Create regional risk analysis plot"""
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
            title="Regional Risk Analysis (% Low Performance Students)",
            xaxis_title="% Low Performance",
            yaxis_title="Region",
            height=550,
            margin=dict(l=150)
        )
        return fig
    
    @staticmethod
    def create_regional_cluster_heatmap(regional_cluster_distribution):
        """Create heatmap showing cluster distribution by region"""
        regional_cluster_df = pd.DataFrame(regional_cluster_distribution).T
        
        fig = go.Figure(data=go.Heatmap(
            z=regional_cluster_df.values,
            x=regional_cluster_df.columns,
            y=regional_cluster_df.index,
            colorscale='RdYlGn_r',
            text=regional_cluster_df.values.round(2),
            texttemplate='%{text:.2f}',
            textfont={"size": 11},
            hoverongaps=False,
            colorbar=dict(title="Proportion", titleside="right")
        ))
        fig.update_layout(
            title="Regional Cluster Distribution Heatmap",
            xaxis_title="Performance Cluster",
            yaxis_title="Region",
            height=550,
            xaxis=dict(tickangle=0)
        )
        return fig
    
    @staticmethod
    def create_boxplot(df, x_col, y_col):
        """Create boxplot by category"""
        if x_col not in df.columns or y_col not in df.columns:
            return go.Figure()
        
        fig = go.Figure()
        
        categories = df[x_col].unique()
        for cat in categories:
            fig.add_trace(go.Box(
                y=df[df[x_col] == cat][y_col],
                name=str(cat),
                marker_color=Visualizer.COLOR_SCHEME['primary']
            ))
        
        fig.update_layout(
            title=f"{y_col} Distribution by {x_col}",
            xaxis_title=x_col,
            yaxis_title=y_col,
            height=400
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
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "dark"},
                'bar': {'color': Visualizer.COLOR_SCHEME['danger'] if risk_prob > 0.5 else Visualizer.COLOR_SCHEME['success']},
                'steps': [
                    {'range': [0, 30], 'color': Visualizer.COLOR_SCHEME['success']},
                    {'range': [30, 70], 'color': Visualizer.COLOR_SCHEME['warning']},
                    {'range': [70, 100], 'color': Visualizer.COLOR_SCHEME['danger']}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig.update_layout(height=250)
        return fig