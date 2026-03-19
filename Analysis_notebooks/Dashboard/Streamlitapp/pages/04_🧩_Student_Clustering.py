import streamlit as st
import backend
from backend import COLOR_SCHEME
import plotly.graph_objects as go

st.set_page_config(page_title="Student Clustering", layout="wide", initial_sidebar_state="expanded")
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

st.title("🧩 Student Clustering Analysis")
top_navigation()

with st.spinner("Loading clustering data..."):
    cluster_sizes, regional_risk, regional_cluster_df = backend.perform_clustering()

st.subheader("Performance Cluster Distribution")
fig_dist = go.Figure(data=[go.Bar(
    x=cluster_sizes.index, y=cluster_sizes.values,
    marker_color=[COLOR_SCHEME['danger'] if i=='Low' else COLOR_SCHEME['warning'] if i=='Medium' else COLOR_SCHEME['success'] for i in cluster_sizes.index]
)])
fig_dist.update_layout(title="Student Performance Cluster Distribution", plot_bgcolor=COLOR_SCHEME['background'])
st.plotly_chart(fig_dist, use_container_width=True)

st.markdown("---")
st.subheader("Regional Risk Analysis")

col1, col2 = st.columns(2)
with col1:
    fig_risk = go.Figure(data=[go.Bar(
        x=regional_risk.index, y=regional_risk.values,
        marker_color=COLOR_SCHEME['danger']
    )])
    fig_risk.update_layout(title="Regional Risk (% Low Performance)", xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig_risk, use_container_width=True)

with col2:
    fig_heat = go.Figure(data=go.Heatmap(
        z=regional_cluster_df.values, x=regional_cluster_df.columns, y=regional_cluster_df.index,
        colorscale='RdYlGn_r', text=regional_cluster_df.values.round(2), texttemplate='%{text:.2f}'
    ))
    fig_heat.update_layout(title="Regional Cluster Heatmap", height=500)
    st.plotly_chart(fig_heat, use_container_width=True)
