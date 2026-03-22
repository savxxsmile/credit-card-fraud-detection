import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page config
st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")

# Load models
scaler = joblib.load("models/scaler.pkl")
lr_model = joblib.load("models/logistic_regression.pkl")

# Load dataset from online source
@st.cache_data
def load_data():
    df = pd.read_csv("https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv")
    return df.sample(10000, random_state=42)

df = load_data()
features = df.columns.drop("Class")

# ================= HEADER =================
st.title("💳 Credit Card Fraud Detection System")
st.markdown("Detect fraudulent transactions using Machine Learning 🚀")

# ================= SIDEBAR =================
st.sidebar.title("⚙️ Options")
option = st.sidebar.radio("Choose an option:", ["🔍 Predict Transaction", "📊 Explore Dataset"])

# ================= METRICS =================
col1, col2, col3 = st.columns(3)
total_tx = len(df)
fraud_count = df["Class"].sum()
legit_count = total_tx - fraud_count

col1.metric("Total Transactions", f"{total_tx:,}")
col2.metric("Legitimate", f"{legit_count:,}", delta=f"{(legit_count/total_tx)*100:.2f}%")
col3.metric("Fraudulent", f"{fraud_count:,}", delta=f"{(fraud_count/total_tx)*100:.2f}%")

# ================= DATA EXPLORATION =================
if option == "📊 Explore Dataset":
    st.title("📊 Dataset Overview")

    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### Class Distribution")
    st.bar_chart(df["Class"].value_counts())

    # Feature Importance
    st.markdown("### 🔍 Feature Importance")

    importance = lr_model.coef_[0]
    feature_importance = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(feature_importance.set_index("Feature").head(10))

# ================= PREDICTION =================
if option == "🔍 Predict Transaction":
    st.title("🔍 Predict Transaction Fraud")

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

        # Probability bar
        st.markdown("### 📊 Fraud Probability")
        st.progress(int(lr_prob * 100))
        st.write(f"Probability of Fraud: **{lr_prob:.2%}**")

        # Risk level
        if lr_prob > 0.8:
            st.error("🚨 High Risk Transaction")
        elif lr_prob > 0.4:
            st.warning("⚠️ Medium Risk Transaction")
        else:
            st.success("✅ Low Risk Transaction")

# ================= FOOTER =================
st.markdown("---")
st.markdown("Built by Sadhvi 🚀 | Data Science Project")