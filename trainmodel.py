# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load data
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Select top 6 features for a lightweight demo
selected_features = [
    'mean concave points',
    'worst perimeter',
    'worst radius',
    'mean perimeter',
    'worst concavity',
    'mean radius'
]

X = df[selected_features]
y = df['target']

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(model, 'breast_cancer_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("breast_cancer_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🩺 Breast Cancer Prediction (Lightweight Demo)")
st.write("Enter the values below to predict whether the tumor is benign or malignant.")

# Features used in the model
selected_features = [
    'mean concave points',
    'worst perimeter',
    'worst radius',
    'mean perimeter',
    'worst concavity',
    'mean radius'
]

# Input section
user_input = []
for feature in selected_features:
    val = st.number_input(f"{feature}", format="%.5f")
    user_input.append(val)

# Prediction
if st.button("Predict"):
    input_df = pd.DataFrame([user_input], columns=selected_features)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    label = "Benign" if prediction[0] == 1 else "Malignant"
    st.success(f"🔍 Prediction: **{label}**")
