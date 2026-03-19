import streamlit as st
import backend
from backend import COLOR_SCHEME
import plotly.graph_objects as go
from sklearn.metrics import roc_curve

st.set_page_config(page_title="Models Analysis", layout="wide", initial_sidebar_state="expanded")

# Fetch Custom CSS and nav
st.markdown("""<style>[data-testid="stSidebarNav"] {display: none;} .stPageLink { margin-bottom: 0 !important; } .block-container { padding-top: 2rem !important; }</style>""", unsafe_allow_html=True)
def top_navigation():
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.page_link("main.py", label="Overview", icon="📊")
    with col2: st.page_link("pages/02_📈_Models_Analysis.py", label="Models", icon="📈")
    with col3: st.page_link("pages/03_🔮_Make_Prediction.py", label="Prediction", icon="🔮")
    with col4: st.page_link("pages/04_🧩_Student_Clustering.py", label="Clustering", icon="🧩")
    with col5: st.page_link("pages/05_💡_Recommendation_Summary.py", label="Summary", icon="💡")
    st.markdown("---")
    st.link_button("🌐 See More Visualization on Power BI", "https://app.powerbi.com", use_container_width=True)

st.title("📈 Models Analysis Dashboard")
top_navigation()

with st.spinner("Loading models..."):
    df_original = backend.get_original_data()
    df_raw = backend.load_and_preprocess_data(df_original)
    df_clean = backend.encode_categorical_features(df_raw)
    
    reg_results, reg_feat_imps, best_reg_model, best_model, scaler, cols, X_train, X_test, trained_models = backend.train_regression_models(df_clean)
    cls_results = backend.train_risk_classification(df_clean)

# Part 1: National Exam Model (Static DataFrame)
st.subheader("National Exam Score Model Analysis")
df_national = backend.NATIONAL_EXAM_MODEL_PERFORMANCE.copy()
st.dataframe(df_national, use_container_width=True)

fig_nat_imp = go.Figure(go.Bar(
    x=backend.NATIONAL_EXAM_FEATURE_IMPORTANCE['Importance'],
    y=backend.NATIONAL_EXAM_FEATURE_IMPORTANCE['Feature'],
    orientation='h', marker_color=COLOR_SCHEME['secondary']
))
fig_nat_imp.update_layout(title="Feature Importance - National Exam Score (Gradient Boosting)", height=500)
st.plotly_chart(fig_nat_imp, use_container_width=True)

st.markdown("---")

# Part 2: Overall Average Regression
st.subheader("Regression Model Performance")
col1, col2 = st.columns([2, 1])
if reg_results and best_reg_model in reg_results:
    models = list(reg_results.keys())
    r2_scores = [reg_results[m]['r2'] for m in models]
    colors = [COLOR_SCHEME['secondary'] if m == best_reg_model else COLOR_SCHEME['primary'] for m in models]
    fig_reg = go.Figure(data=[go.Bar(x=models, y=r2_scores, marker_color=colors, text=[f'{s:.3f}' for s in r2_scores], textposition='auto')])
    fig_reg.update_layout(title="Regression Model R² Score Comparison")
    
    with col1:
        st.plotly_chart(fig_reg, use_container_width=True)
    with col2:
        st.info(f"**Best Model:** {best_reg_model}\n\n**R² Score:** {reg_results[best_reg_model]['r2']:.3f}\n\n**MAE:** {reg_results[best_reg_model]['mae']:.2f}")

    fig_imp = go.Figure(go.Bar(
        x=reg_feat_imps[best_reg_model].head(10).values,
        y=reg_feat_imps[best_reg_model].head(10).index,
        orientation='h', marker_color=COLOR_SCHEME['primary']
    ))
    fig_imp.update_layout(title=f"Top 10 Feature Importance - {best_reg_model}")
    st.plotly_chart(fig_imp, use_container_width=True)

st.markdown("---")

# Part 3: Risk Classification
st.subheader("Risk Classification Performance")
if cls_results:
    col1, col2 = st.columns(2)
    cm = cls_results['cm']
    fig_cm = go.Figure(data=go.Heatmap(z=cm, x=['Not Risk', 'Risk'], y=['Not Risk', 'Risk'], colorscale='Blues', text=cm, texttemplate='%{text}'))
    fig_cm.update_layout(title="Risk Classification Confusion Matrix")
    
    with col1:
        st.info(f"**F1-Score:** {cls_results['f1']:.3f}\n\n**ROC-AUC:** {cls_results['roc_auc']:.3f}")
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with col2:
        fpr, tpr, _ = roc_curve(cls_results['y_test'], cls_results['y_probs'])
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC = {cls_results["roc_auc"]:.3f})', line=dict(color=COLOR_SCHEME['primary'], width=2)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color=COLOR_SCHEME['secondary'])))
        fig_roc.update_layout(title="ROC Curve - Risk Classification")
        st.plotly_chart(fig_roc, use_container_width=True)
    
st.markdown("---")
st.subheader("Global Feature Explainability (SHAP Analysis)")
import shap
import matplotlib.pyplot as plt

if best_model is not None:
    st.info("Calculating SHAP values logically explains how the gradient booster makes internal decisions across the entire test set.")
    try:
        # Bypass XGBoost 2.0+ JSON parsing bug in SHAP by using identical Scikit-Learn Native Model
        shap_model = trained_models.get('GradientBoosting', best_model) 
        X_test_scaled = scaler.transform(X_test)
        explainer = shap.TreeExplainer(shap_model)
        shap_values = explainer.shap_values(X_test_scaled)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**SHAP Summary Plot**")
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            shap.summary_plot(shap_values, X_test, feature_names=cols, show=False)
            st.pyplot(fig1, clear_figure=True)
            
        with col2:
            st.markdown("**SHAP Bar Plot**")
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            shap.summary_plot(shap_values, X_test, feature_names=cols, plot_type="bar", show=False)
            st.pyplot(fig2, clear_figure=True)
            
    except Exception as e:
        st.error(f"SHAP explicitly omitted: {str(e)}")
