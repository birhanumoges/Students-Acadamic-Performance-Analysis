import streamlit as st
import backend
from backend import COLOR_SCHEME
import pandas as pd

st.set_page_config(page_title="Recommendation & Summary", layout="wide", initial_sidebar_state="expanded")
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

st.title("💡 Recommendation & Comprehensive Summary")
top_navigation()

# Create recommendations summary manually mirroring Dash logic
st.subheader("Key Findings")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    #### 🔴 High Risk Interventions
    - Implement personalized learning pipelines immediately for students predicting an overall average below 50.
    - Provide rapid access to missing textbooks or resources based on explicit model SHAP importance.
    - Organize targeted parent engagement workshops.
    """)

with col2:
    st.markdown("""
    #### 🟢 High Performance Maintainers
    - Target enrichment programs for individuals with predicted averages over 80.
    - Sustain extracurricular encouragements.
    """)

st.markdown("---")

st.subheader("Summary Table")
summary_data = [
    {'Metric': 'Dataset Statistics', 'Value': '', 'Details': ''},
    {'Metric': '  Total Students', 'Value': '100,000+', 'Details': '100%'},
    {'Metric': '  Average Overall Score', 'Value': '54.28', 'Details': 'out of 100'},
    {'Metric': 'Model Performance', 'Value': '', 'Details': ''},
    {'Metric': '  Best Regression Model', 'Value': 'Gradient Boosting', 'Details': 'R² = 0.785'},
    {'Metric': '  Risk Classification F1', 'Value': '0.850', 'Details': '>0.75 is good'},
    {'Metric': 'Key Predictors', 'Value': '', 'Details': ''},
    {'Metric': '  Top Factor', 'Value': 'School Resources Score', 'Details': 'Strongest predictor'},
    {'Metric': '  2nd Factor', 'Value': 'Textbook Access', 'Details': 'Critical for learning'}
]

df_summary = pd.DataFrame(summary_data)
st.table(df_summary)

st.info("This concludes the porting of the Ethiopian Student Analytics Dashboard to Streamlit. Use the navigation buttons above to explore all modules.")
