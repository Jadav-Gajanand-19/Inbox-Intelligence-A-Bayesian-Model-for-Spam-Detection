import streamlit as st
import joblib
import os
import time
from PIL import Image

# Set up the Streamlit page
st.set_page_config(
    page_title="Inbox Intelligence",
    page_icon="https://raw.githubusercontent.com/Jadav-Gajanand-19/Inbox-Intelligence-A-Bayesian-Model-for-Spam-Detection/main/inbox_intelligence_logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Floating About Developer Button with Fade-In
st.markdown("""
    <style>
    .about-dev-btn {
        position: fixed;
        top: 15px;
        right: 20px;
        background-color: #6a1b9a;
        color: white;
        padding: 8px 16px;
        font-size: 14px;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        z-index: 9999;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
        opacity: 0;
        animation: fadeIn 1.5s ease-in-out forwards;
    }
    .about-dev-btn:hover {
        background-color: #833AB4;
    }
    @keyframes fadeIn {
        to {
            opacity: 1;
        }
    }
    </style>
    <button class="about-dev-btn" onclick="window.open('https://www.aiip.in/profile/j.gajanand1123', '_blank')">
        👨‍💻 About Developer
    </button>
""", unsafe_allow_html=True)

# Load model
MODEL_PATH = "inbox_intelligence_model.pkl"
model = None
vectorizer = None

try:
    model_data = joblib.load(MODEL_PATH)
    model = model_data["model"]
    vectorizer = model_data["vectorizer"]
except Exception as e:
    st.error(f"💠 Error loading model: {str(e)}")

# Sidebar with stylish small logo
with st.sidebar:
    st.markdown("""
        <style>
        .logo-container {
            text-align: center;
            margin-bottom: 10px;
        }
        .logo-img {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
        }
        </style>
        <div class="logo-container">
            <img class="logo-img" src="https://raw.githubusercontent.com/Jadav-Gajanand-19/Inbox-Intelligence-A-Bayesian-Model-for-Spam-Detection/main/inbox_intelligence_logo.png">
        </div>
    """, unsafe_allow_html=True)

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

# Main page styling
st.markdown("""
    <style>
    .stApp {
        background: white !important;
        font-family: 'Trebuchet MS', sans-serif;
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
        margin-top: 80px;
        font-size: 14px;
        color: #444444;
        animation: fadeIn 4s ease-in-out;
        text-align: center;
    }
    .social-icons {
        margin-top: 20px;
        text-align: center;
    }
    .social-icons a {
        margin: 0 10px;
        display: inline-block;
    }
    .social-icons img {
        width: 32px;
        height: 32px;
        object-fit: contain;
        filter: grayscale(100%);
        transition: filter 0.3s ease;
    }
    .social-icons img:hover {
        filter: none;
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
        float: right;
    }
    .stButton button:hover {
        background-color: #ffffff;
        color: #833AB4;
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
    .feedback-box {
        padding: 10px;
        border-radius: 10px;
        margin-top: 20px;
        color: white;
        font-weight: bold;
        text-align: center;
    }
    .spam {
        background-color: #FFCCCC;
        color: #b30000;
        border: 2px solid #ff6666;
    }
    .not-spam {
        background-color: #CCFFCC;
        color: #006600;
        border: 2px solid #66ff66;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<div class='main-title'>📧 Inbox Intelligence</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Smart Spam Detection powered by Naive Bayes</div>", unsafe_allow_html=True)

# Input
uploaded_file = st.file_uploader("📤 Upload a .txt or .md file:", type=["txt", "md"], key="file_upload")
email_text = ""
disable_textarea = uploaded_file is not None

if uploaded_file is not None:
    file_contents = uploaded_file.read().decode("utf-8")
    email_text = file_contents
    st.markdown("### 📄 Email Content")
    st.code(file_contents, language='markdown')

textarea_input = st.text_area(
    "✏️ Paste your email content here:",
    value="" if disable_textarea else email_text,
    height=200,
    key="paste_text",
    disabled=disable_textarea
)

if not disable_textarea:
    email_text = textarea_input

analyze_btn = st.button("⚙️ Analyze Email", key="analyze_button")

# Prediction logic
if analyze_btn and model and vectorizer and email_text:
    with st.spinner("Analyzing the email..."):
        time.sleep(2.5)
        transformed = vectorizer.transform([email_text])
        prediction = model.predict(transformed)[0]
        confidence = max(model.predict_proba(transformed)[0]) * 100
        color = "#FF4136" if prediction == 1 else "#4CAF50"
        label = "🚨 Caution: This email is suspected to be spam." if prediction == 1 else "✅ This email is not suspected to be spam."

        box_class = "spam" if prediction == 1 else "not-spam"
        st.markdown(f"<div class='feedback-box {box_class}'>{label}</div>", unsafe_allow_html=True)

        st.markdown("**Confidence Meter**")
        st.markdown(f"""
            <div class="battery-container">
                <div class="battery-fill" style="background:{color}; width:{confidence}%">{confidence:.2f}%</div>
            </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class='footer'>
        Built with 💡 by Gajanand | Inbox Intelligence 2025
        <div class='social-icons'>
            <a href='https://www.linkedin.com/in/jadav-gajanand-3aa946290/' target='_blank'>
                <img src='https://cdn-icons-png.flaticon.com/512/174/174857.png'/>
            </a>
            <a href='https://github.com/Jadav-Gajanand-19' target='_blank'>
                <img src='https://cdn-icons-png.flaticon.com/512/25/25231.png'/>
            </a>
            <a href='https://www.instagram.com/cadet_x9/' target='_blank'>
                <img src='https://cdn-icons-png.flaticon.com/512/1384/1384031.png'/>
            </a>
            <a href='https://www.kaggle.com/jadavgajanand' target='_blank'>
                <img src='https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/kaggle.svg'/>
            </a>
            <a href='https://leetcode.com/u/Jadav_Gajanand/' target='_blank'>
                <img src='https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/leetcode.svg'/>
            </a>
        </div>
    </div>
""", unsafe_allow_html=True)
