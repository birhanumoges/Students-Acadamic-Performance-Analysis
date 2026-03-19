import streamlit as st
import backend

st.set_page_config(page_title="Make Prediction", layout="wide", initial_sidebar_state="expanded")
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

st.title("🔮 Make Student Performance Prediction")
top_navigation()

st.info("Input a student's metrics to predict their academic performance and risk level.")

regions = ['Addis Ababa', 'Afar', 'Amhara', 'Benishangul-Gumuz', 'Dire Dawa', 'Gambela', 'Harari', 'Oromia', 'Sidama', 'SNNP', 'Somali', 'South West Ethiopia', 'Tigray']

with st.form("prediction_form"):
    col1, col2, col3, col4 = st.columns(4)
    gender = col1.selectbox("Gender", ['Male', 'Female'])
    dob = col2.text_input("Date of Birth", "2005-06-15")
    region = col3.selectbox("Region", regions, index=7)
    health = col4.selectbox("Health Issue", ['No Issue', 'Vision Issues', 'Anemia', 'Chronic Illness'])

    col1, col2, col3, col4 = st.columns(4)
    f_edu = col1.selectbox("Father Education", ['Unknown', 'Primary', 'High School', 'College', 'University'])
    m_edu = col2.selectbox("Mother Education", ['Unknown', 'Primary', 'High School', 'College', 'University'])
    parent_inv = col3.number_input("Parental Involvement", 0.0, 1.0, 0.5, 0.1)
    internet = col4.selectbox("Home Internet Access", ["No", "Yes"])

    col1, col2, col3, col4 = st.columns(4)
    electricity = col1.selectbox("Electricity Access", ["No", "Yes"])
    school_type = col2.selectbox("School Type", ["Public", "Private"])
    school_loc = col3.selectbox("School Location", ["Rural", "Urban"])
    ts_ratio = col4.number_input("Teacher Student Ratio", 1, 100, 40)

    col1, col2, col3, col4 = st.columns(4)
    res_score = col1.slider("School Resources Score", 0.0, 1.0, 0.5)
    acad_score = col2.slider("School Academic Score", 0.0, 1.0, 0.5)
    sr_ratio = col3.number_input("Student/Resources Ratio", 1, 100, 20)
    field = col4.selectbox("Field Choice", ["Social", "Natural"])
    
    col1, col2, col3, col4 = st.columns(4)
    attendance = col1.slider("Attendance Avg", 0, 100, 75)
    homework = col2.slider("Homework Avg", 0, 100, 65)
    participation = col3.slider("Participation Avg", 0, 100, 70)
    textbook = col4.slider("Textbook Access Composite", 0.0, 1.0, 0.5)

    submit = st.form_submit_button("Predict Performance")

if submit:
    st.markdown("### Prediction Results")
    st.info("Since machine learning routing logic requires the full pipeline, we generate a representative projection based on your inputs:")
    risk_factor = "High" if (internet == "No" and attendance < 50) else "Low"
    pred_score = (attendance * 0.4) + (homework * 0.3) + (participation * 0.3)
    
    col1, col2 = st.columns(2)
    if pred_score < 50:
        col1.error(f"🔴 Predicted Overall Average: {pred_score:.1f}/100")
        col2.error("Risk Level: HIGH - Intervention Recommended")
        st.markdown("**Interventions:**\n- Provide internet access support\n- Implement personalized learning plan\n- Schedule academic counseling")
    else:
        col1.success(f"🟢 Predicted Overall Average: {pred_score:.1f}/100")
        col2.success("Risk Level: LOW - Performing Well")
        st.markdown("**Interventions:**\n- Maintain current study habits\n- Encourage extra-curricular activities")
