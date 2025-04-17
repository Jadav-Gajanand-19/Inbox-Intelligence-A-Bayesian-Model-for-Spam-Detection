# Inbox-Intelligence-A-Bayesian-Model-for-Spam-Detection

## 📬 Inbox Intelligence: Smart Spam Detection Web App

Welcome to **Inbox Intelligence**, a machine learning-powered web application that classifies email messages as **Spam** or **Not Spam**. It leverages a **Naive Bayes classifier** trained on email text data and is deployed through a user-friendly **Streamlit** interface.

Whether you're exploring email classification, testing a custom dataset, or building a production-ready spam filter prototype, this app makes it easy to detect spam in real-time.

---

## 🚀 Features

- 🧠 **Machine Learning-Based Spam Detection**  
  Uses a pre-trained Naive Bayes classifier for reliable spam classification.

- 🔍 **Real-Time Email Analysis**  
  Instantly checks if an email is spam or not and displays the prediction confidence.

- 📨 **Interactive Streamlit UI**  
  Paste email text directly into the app and view results in a clean web interface.

- 🧪 **Sample Emails Included**  
  Load test samples with one click to see the model in action without needing your own input.

- ✅ **Lightweight and Fast**  
  Requires minimal setup, loads quickly, and is easy to understand and extend.

---

## 🧰 Technologies Used

### 🔬 Machine Learning
- **Scikit-learn**: For training and using the Naive Bayes model (`MultinomialNB`) and handling text vectorization via `CountVectorizer`.

- **Pickle** or **Joblib**: Used to serialize the trained model and vectorizer into a `.pkl` file for reuse without retraining.

### 💻 Web Interface
- **Streamlit**: Enables the creation of an interactive web application with just Python code—no front-end experience required.

- **Pandas** *(optional for further data handling)*

---

## 🛠️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Jadav-Gajanand-19/inbox-intelligence.git
cd inbox-intelligence
