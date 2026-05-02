import streamlit as st
import pandas as pd
from transformers import pipeline

# -----------------
# Page Configuration
# -----------------
st.set_page_config(
    page_title="Sentiment Analysis App", 
    page_icon="🎭", 
    layout="wide"
)

# -----------------
# Custom CSS for Powerful UI
# -----------------
st.markdown("""
<style>
    /* Animated Dark Colorful Gradient Background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(-45deg, #1E0933, #59114D, #0B256B, #0A423C);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        background-attachment: fixed;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    [data-testid="stHeader"] {
        background: transparent;
    }
    
    /* Global Text Color */
    h1, h2, h3, h4, p, label, .stMarkdown {
        color: #F9FAFB !important;
    }

    /* Titles */
    .main-title {
        font-size: 4.8rem;
        color: #60A5FA !important;
        text-align: center;
        font-weight: 900;
        margin-bottom: 0;
        font-family: 'Inter', sans-serif;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    .sub-title {
        font-size: 1.8rem;
        color: #9CA3AF !important;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Challenge Banner */
    .challenge-banner {
        background: rgba(31, 41, 55, 0.7);
        backdrop-filter: blur(12px);
        border-left: 4px solid #F59E0B;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-left: 5px solid #F59E0B !important;
    }
    
    /* Result Cards */
    .result-card {
        background: rgba(31, 41, 55, 0.8);
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        border-left: 6px solid #4B5563;
    }
    .result-positive { border-left-color: #10B981; }
    .result-negative { border-left-color: #EF4444; }
    .result-uncertain { border-left-color: #F59E0B; }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #3B82F6 0%, #2563EB 100%);
        color: white !important;
        font-weight: bold;
        border: none;
        padding: 0.75rem;
        border-radius: 8px;
        transition: all 0.3s ease;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.5);
    }
    
    /* Responsive Design / Media Queries */
    @media screen and (max-width: 768px) {
        .main-title { font-size: 3.2rem !important; }
        .sub-title { font-size: 1.4rem !important; }
        .challenge-banner { padding: 1rem !important; }
        .result-card { padding: 1rem !important; }
    }
    @media screen and (max-width: 480px) {
        .main-title { font-size: 2.5rem !important; }
        .sub-title { font-size: 1.2rem !important; }
        .stButton>button { padding: 0.6rem !important; font-size: 1rem !important; }
    }
</style>
""", unsafe_allow_html=True)

# -----------------
# Model Loading (Cached)
# -----------------
@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    # By default, pipeline("sentiment-analysis") loads distilbert-base-uncased-finetuned-sst-2-english
    return pipeline("sentiment-analysis")

# -----------------
# Header Section
# -----------------
st.markdown("<p class='main-title'>Sentiment Analysis</p>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Classifying Emotion in Text — No training needed.</p>", unsafe_allow_html=True)

st.markdown("""
<div class='challenge-banner'>
    <h3 style='margin-top: 0; color: #F59E0B;'>🔥 Challenge 01: Real-World Sentiment App</h3>
    <ul style='color: #E5E7EB; margin-bottom: 0;'>
        <li>Add <strong>"UNCERTAIN"</strong> label for scores between <strong>0.45–0.55</strong>.</li>
        <li>Collect 10 sentences from product reviews or cricket match tweets.</li>
        <li>Analyze them and find which has a score closest to <strong>0.5</strong>.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# -----------------
# Data: 10 Real World Sentences
# -----------------
sentences = [
    "The new phone has an amazing camera, but the battery life is terrible.",  # Product
    "What an unbelievable catch by the fielder! Game changing moment!",        # Cricket
    "The food was okay, nothing special but not bad either.",                  # Product/Review
    "I waited two hours for customer service. Completely unacceptable.",       # Product/Review
    "Kohli's century was an absolute masterclass in batting under pressure.",  # Cricket
    "The app crashes every time I try to open the settings menu.",             # Product/Tech
    "It's a decent laptop for the price, though the screen could be brighter.",# Product
    "Rain stopped play again. This tournament is ruined.",                     # Cricket
    "Absolutely love the new features in this software update!",               # Product/Tech
    "The delivery was delayed by a week, but the product is fine I guess."     # Product
]

# -----------------
# Tabs Setup
# -----------------
tab1, tab2 = st.tabs(["💬 Test Own Text", "📊 Run Challenge Batch Analysis"])

classifier = load_sentiment_model()

with tab1:
    st.markdown("### ✍️ Test Your Own Sentence")
    user_input = st.text_area("Enter text here:", "I am really excited to learn about Transformers!", height=100)
    
    if st.button("🎭 Analyze Sentiment"):
        if user_input.strip():
            with st.spinner("Analyzing emotion..."):
                res = classifier(user_input)[0]
                score = res['score']
                original_label = res['label']
                
                # Apply Challenge Logic
                if score <= 0.55:
                    final_label = "UNCERTAIN"
                    emoji = "🤔"
                    css_class = "result-uncertain"
                    color_hex = "#F59E0B"
                elif original_label == "POSITIVE":
                    final_label = "POSITIVE"
                    emoji = "😊"
                    css_class = "result-positive"
                    color_hex = "#10B981"
                else:
                    final_label = "NEGATIVE"
                    emoji = "😠"
                    css_class = "result-negative"
                    color_hex = "#EF4444"
                    
                st.markdown(f"""
                <div class='result-card {css_class}'>
                    <h2 style='margin-top: 0; color: {color_hex} !important;'>{emoji} {final_label}</h2>
                    <p style='font-size: 1.1rem;'>Confidence Score: <strong>{score:.4f}</strong></p>
                    <p style='font-size: 0.9rem; color: #9CA3AF !important;'>Original Model Output: {original_label}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Please enter some text to analyze.")

with tab2:
    st.markdown("### 🏏 10 Sentences: Product Reviews & Cricket Tweets")
    
    if st.button("🚀 Run Batch Analysis & Find Closest to 0.5"):
        with st.spinner("Analyzing all 10 sentences..."):
            results = classifier(sentences)
            
            closest_idx = 0
            min_diff = 1.0
            
            display_data = []
            
            for i, (s, r) in enumerate(zip(sentences, results)):
                score = r['score']
                original_label = r['label']
                
                # Challenge Logic
                final_label = original_label
                if score <= 0.55:
                    final_label = "UNCERTAIN"
                
                # Check distance to 0.5
                diff = abs(score - 0.5)
                if diff < min_diff:
                    min_diff = diff
                    closest_idx = i
                
                display_data.append({
                    "Sentence": s,
                    "Final Label": final_label,
                    "Score": round(score, 4),
                    "Distance from 0.5": round(diff, 4),
                    "Original Label": original_label
                })
            
            # Display Highlighted Result
            closest_sent = display_data[closest_idx]
            st.markdown(f"""
            <div style='background: rgba(245, 158, 11, 0.2); border: 1px solid #F59E0B; padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;'>
                <h3 style='color: #F59E0B !important; margin-top: 0;'>🏆 Sentence Closest to 0.5 (Most Uncertain)</h3>
                <p style='font-size: 1.2rem; font-style: italic;'>"{closest_sent['Sentence']}"</p>
                <p style='margin-bottom: 0;'>Score: <strong>{closest_sent['Score']}</strong> 
                (Distance from 0.5: {closest_sent['Distance from 0.5']})</p>
                <p>Assigned Label: <strong>{closest_sent['Final Label']}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display full DataFrame
            st.markdown("#### Full Analysis Data")
            df = pd.DataFrame(display_data)
            
            # Pandas Styler to highlight UNCERTAIN rows
            def highlight_uncertain(row):
                if row['Final Label'] == 'UNCERTAIN':
                    return ['background-color: rgba(245, 158, 11, 0.4)'] * len(row)
                return [''] * len(row)
                
            st.dataframe(df.style.apply(highlight_uncertain, axis=1), use_container_width=True)
