import streamlit as st
import backend
from backend import COLOR_SCHEME

st.set_page_config(page_title="Ethiopian Student Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
[data-testid="stSidebarNav"] {display: none;}
.stPageLink { margin-bottom: 0 !important; }
.block-container { padding-top: 2rem !important; }
.stMetric { border: 2px solid """ + COLOR_SCHEME['primary'] + """; border-radius: 10px; padding: 15px; }
</style>
""", unsafe_allow_html=True)

def top_navigation():
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.page_link("main.py", label="Overview", icon="📊")
    with col2: st.page_link("pages/02_📈_Models_Analysis.py", label="Models", icon="📈")
    with col3: st.page_link("pages/03_🔮_Make_Prediction.py", label="Prediction", icon="🔮")
    with col4: st.page_link("pages/04_🧩_Student_Clustering.py", label="Clustering", icon="🧩")
    with col5: st.page_link("pages/05_💡_Recommendation_Summary.py", label="Summary", icon="💡")
    st.markdown("---")
    st.link_button("🌐 See More Visualization on Power BI", "https://app.powerbi.com", use_container_width=True)

st.title("📊 Ethiopian Student Analytics Dashboard")
st.markdown("Comprehensive analysis of Ethiopian students' academic performance")
top_navigation()

with st.spinner("Loading and processing data..."):
    df_original = backend.get_original_data()
    df_raw = backend.load_and_preprocess_data(df_original)
    df_clean = backend.encode_categorical_features(df_raw)

# Overview Content
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Students", f"{len(df_clean):,}")
col2.metric("All Columns", f"{df_original.shape[1]}")
col3.metric("Avg Overall Score", f"{df_clean['Overall_Average'].mean():.1f}" if 'Overall_Average' in df_clean.columns else "N/A")
risk_count = (df_clean['Overall_Average'] < 50).sum() if 'Overall_Average' in df_clean.columns else 0
col4.metric("Risk Students", f"{risk_count:,}")

st.markdown("### 🚨 Alerts & Insights")
if risk_count > 10000:
    st.error(f"High risk volume detected: {risk_count:,} students are predicted to score below 50. Immediate resource allocation required.")
else:
    st.success(f"Risk volume is manageable: {risk_count:,} students are predicted to score below 50.")
if 'Overall_Average' in df_clean.columns:
    avg_score = df_clean['Overall_Average'].mean()
    if avg_score < 60:
        st.warning(f"National Average is at a critical {avg_score:.1f}. Focus needed on upper-primary textbook access.")

st.markdown("### 🎯 Dashboard Objectives")
st.info("Analyze Ethiopian student performance patterns, Predict individual student academic outcomes, Identify at-risk students for early intervention, Understand school and regional disparities.")

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(backend.create_score_distribution_plot(df_clean), use_container_width=True)
with col2:
    st.plotly_chart(backend.create_students_by_region_plot(df_raw), use_container_width=True)

st.plotly_chart(backend.create_correlation_heatmap(df_clean), use_container_width=True)
