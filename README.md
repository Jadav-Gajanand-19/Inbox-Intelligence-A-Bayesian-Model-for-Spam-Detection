# 📧 Inbox Intelligence - A Bayesian Model for Spam Detection

![Logo](https://raw.githubusercontent.com/Jadav-Gajanand-19/Inbox-Intelligence-A-Bayesian-Model-for-Spam-Detection/main/inbox_intelligence_logo.png)

A sleek and smart web app that uses a **Naive Bayes classifier** to detect whether an email is spam or not — with real-time prediction, confidence analysis, and a polished user interface built using **Streamlit**.

🌐 **Live Demo**:  
🔗 [Click to try the app](https://inbox-intelligence-a-bayesian-model-for-spam-detection-cadet-07.streamlit.app/)

---

## 🚀 Features

- 🧠 Built on **Multinomial Naive Bayes** model
- ✍️ **Paste** or **upload** email files for analysis
- ⚙️ Displays prediction with **confidence score**
- 📊 Confidence shown as an animated **battery meter**
- ✅ Clear message output: "Not Spam" or "Caution: This is likely spam"
- ⬇️ Downloadable predictions (optional future feature)
- 🎨 Sleek UI with modern styles and interactive components
- 👨‍💻 About Developer button in top-right corner  
- 🔗 Social media icons in the footer (GitHub, LinkedIn, Instagram, Kaggle, LeetCode)

---

## 🧠 Model Info

- Trained on a labeled dataset of spam and ham emails
- Uses **TF-IDF Vectorizer** for feature extraction
- Accuracy and F1-score optimized
- Saved using `joblib` as `inbox_intelligence_model.pkl`

---

## 📂 File Upload Support

You can upload plain `.txt` or `.md` files. Upon selection:
- File content is shown instantly
- Text area input is disabled to avoid confusion

---

## 👨‍💻 About the Developer

Hi! I'm **Gajanand Jadav**, an ML enthusiast passionate about real-world AI applications.  
🔗 [Portfolio](https://www.aiip.in/profile/j.gajanand1123)

---

## 🛠️ Tech Stack

- Python 🐍
- Scikit-learn ⚙️
- Streamlit 🌐
- PIL 🖼️
- Joblib 📦

---

## 🧑‍🏫 Example Use Cases

- Teachers explaining ML with Naive Bayes
- Companies verifying email safety
- Learners testing basic email spam filters

---


## 📦 Installation

```bash
git clone https://github.com/Jadav-Gajanand-19/Inbox-Intelligence-A-Bayesian-Model-for-Spam-Detection
cd Inbox-Intelligence-A-Bayesian-Model-for-Spam-Detection
pip install -r requirements.txt
streamlit run inbox_intelligence_UI.py
```

---

## ✅ Requirements

`requirements.txt` includes:

```txt
streamlit
scikit-learn
joblib
pillow
```

---

## 🤝 Contributions

Open to contributions and suggestions. Fork, enhance, and submit a PR!
