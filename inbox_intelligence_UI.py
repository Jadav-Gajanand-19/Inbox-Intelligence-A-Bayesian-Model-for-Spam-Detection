import streamlit as st
st.set_page_config(
    page_title="Inbox Intelligence",
    page_icon="📬",
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
    st.error(f"🔧 Error loading model: {str(e)}")

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/561/561188.png", width=100)
    st.title("📬 Inbox Intelligence")
    st.markdown("Detect spam messages with machine learning magic.")
    st.markdown("---")
    st.markdown("Developed with ❤️ using Naive Bayes and Streamlit.")

# Header with animation
st.markdown("""
    <style>
    .main-title {
        font-size: 48px;
        font-weight: bold;
        color: #3b82f6;
        animation: fadeIn 2s ease-in-out;
    }
    .subtitle {
        font-size: 20px;
        color: #555;
        animation: fadeIn 3s ease-in-out;
    }
    .footer {
        margin-top: 50px;
        font-size: 14px;
        color: #999;
        animation: fadeIn 4s ease-in-out;
    }
    .stButton button {
        border: 2px solid #3b82f6;
        background-color: white;
        color: #3b82f6;
        padding: 0.5em 1em;
        font-size: 16px;
        font-weight: bold;
        transition: 0.3s ease;
        border-radius: 8px;
    }
    .stButton button:hover {
        background-color: #3b82f6;
        color: white;
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
        background: #e5e5e5;
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

st.markdown("<div class='main-title'>📥 Inbox Intelligence</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Smart Spam Detection powered by Naive Bayes</div>", unsafe_allow_html=True)

st.markdown("---")

# File upload first
st.subheader("📌 Upload an Email File (.txt)")
uploaded_file = st.file_uploader("Choose a .txt file", type=["txt"])
file_email_text = None

# Input area disabled based on file upload
st.subheader("✉️ Or Paste Your Email Message Below")
disable_text_input = uploaded_file is not None
email_text = st.text_area(
    "",
    height=200,
    placeholder="Subject: Hello\nBody: This is a test message...",
    key="email_input",
    disabled=disable_text_input
)

if uploaded_file is not None:
    try:
        file_email_text = uploaded_file.read().decode("utf-8")
        st.success("✅ File uploaded successfully!")
        st.text_area("📩 Email Content from File", file_email_text, height=150, disabled=True)
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")

if uploaded_file is not None and file_email_text:
    email_text = file_email_text

# Check button
col1, col2 = st.columns([3, 1])
with col2:
    check = st.button("🚀 Check Spam", use_container_width=True)

# Prediction result
if check and email_text.strip():
    if model is None or vectorizer is None:
        st.error("🔧 Model or vectorizer not loaded properly.")
    else:
        try:
            with st.spinner("Analyzing message..."):
                time.sleep(1.5)
                input_vector = vectorizer.transform([email_text])
                prediction = model.predict(input_vector)[0]
                confidence = max(model.predict_proba(input_vector)[0]) * 100

            st.markdown("---")
            if prediction == 1:
                st.error(f"🚨 This email is classified as **SPAM**.")
                st.markdown("Confidence Meter :")
                st.markdown("""
                    <div class='battery-container'>
                        <div class='battery-fill' style='width: {:.2f}%; background: #ff4d4f;'>
                            {:.2f}%
                        </div>
                    </div>
                """.format(confidence, confidence), unsafe_allow_html=True)
                st.markdown("""
                    <div class='caution-animated spam'>⚠️ WARNING: SPAM Detected! ⚠️</div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.success(f"✅ This email is **NOT SPAM**.")
                st.markdown("Confidence Meter :")
                st.markdown("""
                    <div class='battery-container'>
                        <div class='battery-fill' style='width: {:.2f}%; background: #4caf50;'>
                            {:.2f}%
                        </div>
                    </div>
                """.format(confidence, confidence), unsafe_allow_html=True)
                st.markdown("""
                    <div class='caution-animated not-spam'>✅ All clear: Not Spam!</div>
                """, unsafe_allow_html=True)
                st.snow()

            st.markdown("---")
            st.info("You can modify the email and re-check instantly.")
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")

# Footer
st.markdown("""
    <div class='footer'>
    Built with ❤️ by Gajanand. This is a demo of spam detection using machine learning.
    </div>
""", unsafe_allow_html=True)
