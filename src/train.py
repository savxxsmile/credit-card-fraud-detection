import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import joblib

# Load dataset
df = pd.read_csv("https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv")

# Features and target
X = df.drop("Class", axis=1).values
y = df["Class"].values

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")

# Handle class imbalance using SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_scaled, y)

print(f"Before SMOTE: {np.bincount(y)}")
print(f"After SMOTE: {np.bincount(y_res)}")

# Train Logistic Regression model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_res, y_res)

# Save trained model
joblib.dump(lr_model, "models/logistic_regression.pkl")

print("✅ Training completed successfully!")