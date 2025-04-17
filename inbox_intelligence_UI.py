import streamlit as st
st.set_page_config(
    page_title="Inbox Intelligence",
    page_icon="/mnt/data/A_2D_digital_graphic_logo_design_displays_the_bran.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

import joblib
import os
import time
from PIL import Image
import streamlit.components.v1 as components

# Load the model and vectorizer
MODEL_PATH = "inbox_intelligence_model.pkl"
model = None
vectorizer = None

try:
    model_data = joblib.load(MODEL_PATH)
    model = model_data["model"]
    vectorizer = model_data["vectorizer"]
except Exception as e:
    st.error(f"🛠️ Error loading model: {str(e)}")

# Sidebar
with st.sidebar:
    st.image("/mnt/data/A_2D_digital_graphic_logo_design_displays_the_bran.png", width=100)
    st.title("📧 Inbox Intelligence")
    st.markdown("Detect spam messages with machine learning magic.")
    st.markdown("---")
    st.markdown("### 🔍 What does it do?")
    st.markdown("This app uses a Naive Bayes classifier to analyze email messages and determine whether they are **spam** or **not spam**.")
    st.markdown("---")
    st.markdown("### ⚙️ How it works")
    st.markdown("1. Text is vectorized using TF-IDF.\n2. The model evaluates the features.\n3. It returns a prediction with a confidence score.")
    st.markdown("---")
    st.markdown("### 💬 Example Uses")
    st.markdown("- Check suspicious emails\n- Test your spam filters\n- Educational demo for ML beginners")
    st.markdown("---")
    st.markdown("### 📚 More Info")
    st.markdown("Naive Bayes is a probabilistic classifier based on Bayes' Theorem. It's fast, efficient, and often used for spam detection.")
    st.markdown("Learn more about it [here](https://en.wikipedia.org/wiki/Naive_Bayes_classifier).")
    st.markdown("---")
    st.markdown("Developed with 💡 using Naive Bayes and Streamlit.")

# Header with styles
st.markdown("""
    <style>
    body {
        background: white;
        color: #000000;
    }
    .stApp {
        background: white !important;
    }
    .main-title {
        font-size: 48px;
        font-weight: bold;
        color: #000000;
        animation: fadeIn 2s ease-in-out;
    }
    .subtitle {
        font-size: 20px;
        color: #333333;
        animation: fadeIn 3s ease-in-out;
    }
    .footer {
        margin-top: 50px;
        font-size: 14px;
        color: #444444;
        animation: fadeIn 4s ease-in-out;
    }
    .stButton button {
        border: 2px solid #833AB4;
        background-color: #833AB4;
        color: #ffffff;
        padding: 0.5em 1em;
        font-size: 16px;
        font-weight: bold;
        transition: 0.3s ease;
        border-radius: 8px;
    }
    .stButton button:hover {
        background-color: #ffffff;
        color: #833AB4;
    }
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(-10px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0; }
    }
    .battery-container {
        height: 30px;
        width: 100%;
        background: #eee;
        border-radius: 10px;
        margin: 10px 0;
        overflow: hidden;
    }
    .battery-fill {
        height: 100%;
        text-align: right;
        padding-right: 10px;
        line-height: 30px;
        color: white;
        font-weight: bold;
        transition: width 1s ease-in-out;
    }
    .caution-animated {
        animation: blink 1s ease-in-out infinite;
        font-size: 26px;
        font-weight: bold;
        text-align: center;
    }
    .spam {
        color: #FF4136;
    }
    .not-spam {
        color: #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>📧 Inbox Intelligence</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Smart Spam Detection powered by Naive Bayes</div>", unsafe_allow_html=True)

# [The rest of the app logic remains unchanged]
