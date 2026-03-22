import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")

# Load models
scaler = joblib.load("models/scaler.pkl")
lr_model = joblib.load("models/logistic_regression.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv")
    return df.sample(10000, random_state=42)

df = load_data()
features = df.columns.drop("Class")

# Sidebar
st.sidebar.title("⚙️ Options")
option = st.sidebar.radio("Choose an option:", ["🔍 Predict Transaction", "📊 Explore Dataset"])

# --- Top Dashboard Metrics ---
col1, col2, col3 = st.columns(3)
total_tx = len(df)
fraud_count = df["Class"].sum()
legit_count = total_tx - fraud_count

col1.metric("Total Transactions", f"{total_tx:,}")
col2.metric("Legitimate", f"{legit_count:,}", delta=f"{(legit_count/total_tx)*100:.2f}%")
col3.metric("Fraudulent", f"{fraud_count:,}", delta=f"{(fraud_count/total_tx)*100:.2f}%")

# ---------------- DATA EXPLORATION ----------------
if option == "📊 Explore Dataset":
    st.title("📊 Dataset Overview")

    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### Class Distribution")
    st.bar_chart(df["Class"].value_counts())

# ---------------- PREDICTION ----------------
if option == "🔍 Predict Transaction":
    st.title("🔍 Predict Transaction Fraud")

    # Sample row for default values
    sample_row = df.sample(1, random_state=np.random.randint(0, 10000))
    default_values = sample_row[features].iloc[0].to_dict()

    with st.form("transaction_form"):
        cols = st.columns(4)
        input_data = {}

        for i, feature in enumerate(features):
            with cols[i % 4]:
                input_data[feature] = st.number_input(
                    feature,
                    value=float(default_values[feature])
                )

        submitted = st.form_submit_button("Predict 🚀")

    if submitted:
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)

        # Prediction
        lr_pred = lr_model.predict(input_scaled)[0]
        lr_prob = lr_model.predict_proba(input_scaled)[0][1]

        st.subheader("🔎 Results")

        st.write(f"Fraud Probability: **{lr_prob:.4f}**")

        if lr_pred == 1:
            st.error("🚨 Fraud Detected")
        else:
            st.success("✅ Legitimate Transaction")