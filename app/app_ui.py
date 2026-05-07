import streamlit as st
import joblib
import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
model = joblib.load(BASE_DIR / "model" / "model.pkl")
scaler = joblib.load(BASE_DIR / "model" / "scaler.pkl")


# load model + scaler
# model = joblib.load(model)
# scaler = joblib.load(scaler)

st.title("🩺 Diabetes Prediction App")

# ---- inputs ----
age = st.number_input("Age")
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
bmi = st.number_input("BMI")
hba1c = st.number_input("HbA1c Level")
glucose = st.number_input("Blood Glucose Level")

gender = st.selectbox("Gender", ["Male", "Female"])
smoking = st.selectbox("Smoking History", [
    "never", "former", "current", "ever", "not current"
])

# ---- encode manually (must match training features order!) ----
input_data = [
    age,
    hypertension,
    heart_disease,
    bmi,
    hba1c,
    glucose,
    1 if gender == "Other" else 0,
    1 if gender == "Male" else 0,
    1 if smoking == "current" else 0,
    1 if smoking == "ever" else 0,
    1 if smoking == "former" else 0,
    1 if smoking == "never" else 0,
    1 if smoking == "not current" else 0
]

# ---- predict ----
if st.button("Predict"):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    result = model.predict(input_scaled)

    if result[0] == 1:
        st.error("⚠️ High risk of diabetes")
    else:
        st.success("✅ Low risk of diabetes")