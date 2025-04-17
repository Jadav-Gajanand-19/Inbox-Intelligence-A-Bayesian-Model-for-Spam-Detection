import streamlit as st
import joblib
import os
import time
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
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

# Page config - must be the first Streamlit command
st.set_page_config(
    page_title="Inbox Intelligence",
    page_icon="📬",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>📥 Inbox Intelligence</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Smart Spam Detection powered by Naive Bayes</div>", unsafe_allow_html=True)

st.markdown("---")

# Input area
st.subheader("✉️ Paste Your Email Message Below")
email_text = st.text_area(
    "",
    height=200,
    placeholder="Subject: Hello\nBody: This is a test message...",
    key="email_input"
)

# Check button
col1, col2 = st.columns([3, 1])
with col2:
    check = st.button("🚀 Check Spam", use_container_width=True)

# Pie chart function for confidence
def display_confidence_pie(confidence):
    labels = ["Confidence", "Remaining"]
    sizes = [confidence, 100 - confidence]
    colors = ["#4CAF50", "#f0f0f0"] if confidence > 50 else ["#FF4136", "#f0f0f0"]
    explode = (0.1, 0)

    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        shadow=True,
        startangle=140
    )
    ax.axis('equal')
    st.pyplot(fig)

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
                st.error(f"🚨 This email is classified as **SPAM** with {confidence:.2f}% confidence.")
                st.markdown("""
                    <div style="animation: fadeIn 2s ease-in-out; font-size: 24px; color: #FF4136;">
                        ⚠️ Be cautious of such messages!
                    </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.success(f"✅ This email is **NOT SPAM** with {confidence:.2f}% confidence.")
                st.markdown("""
                    <div style="animation: fadeIn 2s ease-in-out; font-size: 24px; color: #4CAF50;">
                        🎉 Looks safe and clean!
                    </div>
                """, unsafe_allow_html=True)
                st.snow()

            display_confidence_pie(confidence)

            st.markdown("---")
            st.info("You can modify the email and re-check instantly.")
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")

# Footer
st.markdown("""
    <div class='footer'>
    Built with ❤️ by [Your Name]. This is a demo of spam detection using machine learning.
    </div>
""", unsafe_allow_html=True)


    





