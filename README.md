# 💳 Credit Card Fraud Detection System

A Machine Learning-powered web application that detects fraudulent credit card transactions in real-time using predictive analytics and an interactive dashboard.

---

## 🌐 Live Demo

👉 https://credit-card-fraud-detection-projectapp.streamlit.app/

---

## 🚀 Project Overview

This project aims to identify fraudulent transactions from highly imbalanced financial data using machine learning techniques. It includes an end-to-end pipeline — from data preprocessing and model training to deployment via a web interface.

---

## 🧠 Key Features

- 🔍 Real-time fraud prediction
- 📊 Interactive dataset exploration
- 📈 Fraud probability visualization
- ⚖️ Handles class imbalance using SMOTE
- 🔎 Feature importance insights
- ⚡ Fast and lightweight Streamlit deployment

---

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** pandas, numpy, scikit-learn, imbalanced-learn, joblib
- **Visualization:** matplotlib, seaborn, plotly
- **Frontend:** Streamlit
- **Model:** Logistic Regression

---

## 📂 Project Structure

```
credit-card-fraud-detection/
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

1. Data is loaded from an online dataset
2. Features are scaled using StandardScaler
3. SMOTE is applied to handle class imbalance
4. Logistic Regression model is trained
5. Predictions are made with probability scores
6. Results are displayed via an interactive dashboard

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

## 📊 Dataset

- Based on the widely used credit card fraud dataset
- Contains anonymized features (V1–V28), Time, Amount, and Class
- Highly imbalanced dataset (fraud cases are rare)
- Loaded dynamically from an online source for deployment

---

## 🎯 Results & Insights

- Successfully detects fraudulent transactions with high recall
- Provides probability-based risk classification
- Feature importance helps interpret model decisions

---

## 🔮 Future Improvements

- 🔥 Add XGBoost / LightGBM for improved performance
- 📉 Include ROC Curve and Confusion Matrix
- 🌐 Real-time transaction API integration
- 🧠 Add anomaly detection (Isolation Forest / Autoencoder)

---

## 👩‍💻 Author

**Sadhvi**
B.Tech Computer Science (2023–2027)
Aspiring Data Scientist

---

## ⭐ Support

If you found this project useful:

- ⭐ Star this repository
- 🍴 Fork it and build upon it
- 📢 Share it with others

---

## 📌 Note

This project is built for educational and demonstration purposes and simulates real-world fraud detection systems.
