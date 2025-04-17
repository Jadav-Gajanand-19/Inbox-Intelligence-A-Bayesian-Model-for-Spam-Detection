import streamlit as st
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer

# Page settings
st.set_page_config(
    page_title="Inbox Intelligence",
    page_icon="📬",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "This app detects spam emails using a Naive Bayes classifier."
    }
)

# Load model and vectorizer
@st.cache_resource
def load_model():
    with open("inbox intelligence model.pkl", "rb") as f:
        model_data = pickle.load(f)
    return model_data['model'], model_data['vectorizer']

model, vectorizer = load_model()

# App title
st.title("📬 Inbox Intelligence: Spam Detection")
st.markdown("This smart spam filter uses machine learning to classify emails as **Spam** or **Not Spam**.")

# Input box
email_input = st.text_area("📥 Paste your email content here:", height=200)

# Predict button
if st.button("🔍 Check Spam"):
    if not email_input.strip():
        st.warning("Please enter an email message to classify.")
    else:
        # Transform input and predict
        features = vectorizer.transform([email_input])
        prediction = model.predict(features)[0]
        prediction_proba = model.predict_proba(features)[0]

        # Output result
        if prediction == 1:
            st.error(f"🚨 This email is likely SPAM (Confidence: {prediction_proba[1]*100:.2f}%)")
        else:
            st.success(f"✅ This email is NOT spam (Confidence: {prediction_proba[0]*100:.2f}%)")

# Example emails
with st.expander("📄 Try sample emails"):
    samples = [
        "Subject: Congratulations! You've won a free cruise! Click here to claim your prize.",
        "Subject: Meeting agenda for tomorrow's 10 AM sync.",
        "Subject: Get rich fast with this one simple trick."
    ]
    for i, sample in enumerate(samples, start=1):
        if st.button(f"Load Example #{i}"):
            st.session_state["email_input"] = sample
            email_input = sample
