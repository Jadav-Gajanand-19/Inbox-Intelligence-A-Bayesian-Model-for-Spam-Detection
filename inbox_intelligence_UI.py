import streamlit as st
import joblib
import os
from PIL import Image

# Load the model and vectorizer
MODEL_PATH = "inbox intelligence model.pkl"
model = None
vectorizer = None

try:
    model_data = joblib.load(MODEL_PATH)
    model = model_data["model"]
    vectorizer = model_data["vectorizer"]
except Exception as e:
    st.error(f"🔧 Error loading model: {str(e)}")

# Page config
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

# Header
st.markdown("""
    <style>
    .main-title {
        font-size: 48px;
        font-weight: bold;
        color: #3b82f6;
    }
    .subtitle {
        font-size: 20px;
        color: #555;
    }
    .footer {
        margin-top: 50px;
        font-size: 14px;
        color: #999;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>📥 Inbox Intelligence</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Smart Spam Detection powered by Naive Bayes</div>", unsafe_allow_html=True)

st.markdown("---")

# State for pre-filled examples
if "preset_email" not in st.session_state:
    st.session_state["preset_email"] = ""

# Use preset_email for text area
st.subheader("✉️ Paste Your Email Message Below")
email_text = st.text_area(
    "",
    height=200,
    placeholder="Subject: Hello\nBody: This is a test message...",
    value=st.session_state.get("preset_email", ""),
    key="email_input"
)

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
            input_vector = vectorizer.transform([email_text])
            prediction = model.predict(input_vector)[0]
            confidence = max(model.predict_proba(input_vector)[0]) * 100

            st.markdown("---")
            if prediction == 1:
                st.error(f"🚨 This email is classified as **SPAM** with {confidence:.2f}% confidence.")
            else:
                st.success(f"✅ This email is **NOT SPAM** with {confidence:.2f}% confidence.")

            st.markdown("---")
            st.info("You can modify the email and re-check instantly.")
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")

# Example emails
with st.expander("🔍 Show Sample Emails"):
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🏆 Non-Spam Example"):
            st.session_state["preset_email"] = "Subject: Meeting Reminder\nBody: Don’t forget our 10AM sync tomorrow."
            st.experimental_rerun()
    with col2:
        if st.button("💸 Spam Example"):
            st.session_state["preset_email"] = "Subject: You won a prize!\nBody: Click here to claim your $10,000 now!"
            st.experimental_rerun()

# Footer
st.markdown("""
    <div class='footer'>
    Built with ❤️ by [Your Name]. This is a demo of spam detection using machine learning.
    </div>
""", unsafe_allow_html=True))



