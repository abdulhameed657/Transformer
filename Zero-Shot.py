pip install transformers
import streamlit as st
from transformers import pipeline

# -----------------
# Page Configuration
# -----------------
st.set_page_config(
    page_title="Zero-Shot Student Query Router", 
    page_icon="🚀", 
    layout="centered"
)

# -----------------
# Session State Initialization
# -----------------
if 'categories' not in st.session_state:
    st.session_state.categories = ["exam schedule", "fee payment", "hostel", "technical support"]

# -----------------
# Custom CSS for Powerful UI
# -----------------
st.markdown("""
<style>
    /* Gradient Background for the main app container */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        background-attachment: fixed;
    }
    
    [data-testid="stHeader"] {
        background: transparent;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: rgba(15, 32, 39, 0.4) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Make text lighter for the gradient background */
    h1, h2, h3, h4, p, label, .stMarkdown {
        color: #F7FAFC;
    }

    /* Dark theme modern styling */
    .main-title {
        font-size: 3rem;
        color: #FFB300 !important;
        text-align: center;
        font-weight: 800;
        margin-bottom: 0;
        font-family: 'Inter', sans-serif;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    .sub-title {
        font-size: 1.2rem;
        color: #E2E8F0 !important;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    .challenge-banner {
        background: rgba(26, 32, 44, 0.6);
        backdrop-filter: blur(12px);
        border-left: 4px solid #FFB300;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-left: 4px solid #FFB300 !important;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #FFB300 0%, #F59E0B 100%);
        color: #1A202C !important;
        font-weight: bold;
        border: none;
        padding: 0.75rem;
        border-radius: 8px;
        transition: all 0.3s ease;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(255, 179, 0, 0.4);
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #F59E0B 0%, #D97706 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 179, 0, 0.6);
        border: none;
    }
    .result-box {
        padding: 1.5rem;
        background: rgba(26, 32, 44, 0.8);
        backdrop-filter: blur(15px);
        border-radius: 12px;
        border: 1px solid #48BB78;
        margin-top: 1.5rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }
    .category-tag {
        background: rgba(45, 55, 72, 0.6);
        backdrop-filter: blur(5px);
        color: #E2E8F0;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        display: inline-block;
        margin: 0.3rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        transition: all 0.2s ease;
    }
    .category-tag:hover {
        background: rgba(74, 85, 104, 0.8);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -----------------
# Model Loading (Cached)
# -----------------
@st.cache_resource(show_spinner=False)
def load_classifier():
    # We use cache_resource so the large model only loads once
    # Reverted to bart-large-mnli because smaller models were less accurate for these specific classes
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# -----------------
# Sidebar - Add Department
# -----------------
with st.sidebar:
    st.markdown("<h2 style='color: #FFB300; margin-top: -1rem;'>⚙️ Router Settings</h2>", unsafe_allow_html=True)
    
    st.markdown("### ➕ Add New Department")
    new_dept = st.text_input("Department Name", placeholder="e.g. library, sports...")
    
    if st.button("Add Department"):
        if new_dept.strip() == "":
            st.error("Please enter a valid name.")
        elif new_dept.lower() in [c.lower() for c in st.session_state.categories]:
            st.warning("Department already exists!")
        else:
            st.session_state.categories.append(new_dept.strip())
            st.success(f"Added '{new_dept}'!")
            st.rerun()
    
    st.markdown("---")
    st.markdown("### 📋 Active Departments")
    for i, cat in enumerate(st.session_state.categories):
        col1, col2 = st.columns([4, 1])
        col1.markdown(f"• **{cat.title()}**")
        # Ensure we always have at least 2 categories for routing to make sense
        if len(st.session_state.categories) > 2:
            if col2.button("✖", key=f"del_{i}", help="Remove department", type="secondary"):
                st.session_state.categories.pop(i)
                st.rerun()

# -----------------
# Header Section
# -----------------
st.markdown("<p class='main-title'>Zero-Shot Classification</p>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>No Training Required — Language Understanding at its best.</p>", unsafe_allow_html=True)

st.markdown("""
<div class='challenge-banner'>
    <h3 style='margin-top: 0; color: #FFB300;'>🔥 Challenge 02: Build a Student Query Router</h3>
    <p style='margin-bottom: 0; color: #E2E8F0;'>Create a classifier routing student questions to specific departments using zero-shot learning.</p>
</div>
""", unsafe_allow_html=True)

# -----------------
# Categories Definition
# -----------------
st.markdown("#### 🏢 Active Routing Departments")
# Display tags beautifully
tags_html = "".join([f"<span class='category-tag'>🏷️ {cat.title()}</span>" for cat in st.session_state.categories])
st.markdown(f"<div style='margin-bottom: 1rem;'>{tags_html}</div>", unsafe_allow_html=True)

# -----------------
# Interactive Section
# -----------------
st.markdown("#### 💬 Enter Student Query")
query = st.text_area(
    label="Student Query", 
    label_visibility="collapsed",
    value="My student portal login is not working", 
    height=100,
    placeholder="Type the student's question here..."
)

# Execute Classificationa
if st.button("🚀 Route Query Now"):
    if query.strip() == "":
        st.warning("⚠️ Please enter a query first.")
    elif len(st.session_state.categories) < 2:
        st.error("⚠️ Please add at least two departments to route queries between.")
    else:
        with st.spinner("🧠 Analyzing semantics and determining the best route..."):
            classifier = load_classifier()
            result = classifier(query, candidate_labels=st.session_state.categories)
            
            top_category = result['labels'][0]
            top_score = result['scores'][0]
            
            # Result Display
            st.markdown(f"""
            <div class='result-box'>
                <h2 style='margin-top: 0; color: #48BB78 !important;'>✅ Successfully Routed</h2>
                <p style='font-size: 1.3rem; margin-bottom: 0.5rem;'>
                    <strong>Destination:</strong> <span style='color: #FFB300 !important;'>{top_category.upper()}</span>
                </p>
                <p style='color: #A0AEC0 !important; font-size: 0.9rem; margin-bottom: 0;'>
                    <strong>Confidence Score:</strong> {top_score:.4f}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show probability distribution
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("📊 View Detailed Probability Distribution"):
                for label, score in zip(result['labels'], result['scores']):
                    cols = st.columns([1, 4])
                    cols[0].markdown(f"**{label.title()}**")
                    cols[1].progress(score)
                    st.caption(f"{score:.2%} match")
