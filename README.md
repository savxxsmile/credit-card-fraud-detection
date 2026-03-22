# 💳 Credit Card Fraud Detection System

A machine learning-based web application to detect fraudulent credit card transactions in real time using predictive analytics and data-driven modeling.

---

## 🚀 Project Overview

This project focuses on identifying fraudulent transactions using supervised machine learning techniques. It handles highly imbalanced data and provides an interactive interface for prediction and dataset exploration.

---

## 🧠 Key Features

- 🔍 Real-time fraud prediction
- 📊 Interactive data exploration dashboard
- ⚖️ Handles class imbalance using SMOTE
- 📈 Displays fraud probability for each transaction
- ⚡ Fast and lightweight deployment using Streamlit

---

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Libraries:** pandas, numpy, scikit-learn, imbalanced-learn, joblib
- **Visualization:** matplotlib, seaborn, plotly
- **Frontend:** Streamlit
- **Model:** Logistic Regression

---

## 📂 Project Structure

```
CreditCard_Fraud_Detection/
│
├── app.py                  # Streamlit web app
├── requirements.txt        # Dependencies
├── README.md
│
├── models/
│   ├── scaler.pkl
│   └── logistic_regression.pkl
│
├── src/
│   ├── train.py            # Model training pipeline
│   └── utils.py
│
├── notebooks/
│   └── fraud_detection.ipynb
```

---

## ⚙️ How It Works

1. Data is preprocessed and scaled using StandardScaler
2. SMOTE is applied to balance the dataset
3. Logistic Regression model is trained
4. Model predicts fraud probability for new transactions
5. Streamlit app provides an interactive interface

---

## ▶️ Run Locally

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train model
python src/train.py

# Run app
streamlit run app.py
```

---

## 🌐 Deployment

This app is deployed using Streamlit Community Cloud.

👉 The dataset is fetched from an online source to avoid large file issues during deployment.

---

## 📊 Dataset

- Based on the popular credit card fraud dataset
- Contains anonymized features (V1–V28), Time, Amount, and Class
- Highly imbalanced dataset

---

## 🎯 Future Improvements

- 🔥 Add XGBoost / LightGBM for better performance
- 📉 Add ROC curve & confusion matrix visualization
- 🌐 Connect to real-time transaction APIs
- 🧠 Implement anomaly detection (Autoencoder / Isolation Forest)

---

## 👩‍💻 Author

**Sadhvi**
B.Tech CSE (2023–2027)
Aspiring Data Scientist

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and feel free to fork and improve!
