import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.title("Student Performance Prediction")


# ---------------- Load Models Safely ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
encoder_path = os.path.join(BASE_DIR, "label_encoder.pkl")

if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(encoder_path)):
    st.error("Model files not found! Make sure model.pkl, scaler.pkl, label_encoder.pkl are in the same folder.")
    st.stop()

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
le_grade = joblib.load(encoder_path)

# ---------------- Inputs ----------------
study_hours = st.slider("Study Hours per Day", 0.0, 10.0, 5.0)
attendance = st.slider("Attendance %", 0.0, 100.0, 75.0)
assignment = st.slider("Assignment Score", 0.0, 100.0, 70.0)
midterm = st.slider("Midterm Score", 0.0, 100.0, 65.0)
final = st.slider("Final Exam Score", 0.0, 100.0, 70.0)
participation = st.slider("Participation Score", 0.0, 100.0, 60.0)
sleep = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
overall = st.slider("Overall Score", 0.0, 100.0, 70.0)

gender = st.selectbox("Gender", ["Female", "Male"])
internet = st.selectbox("Internet Access", ["No", "Yes"])
extra = st.selectbox("Extra Classes", ["No", "Yes"])
parent = st.selectbox("Parent Education", ["High School", "Bachelor", "Master", "PhD"])

# ---------------- Encoding ----------------
gender_map = {'Female': 0, 'Male': 1}
internet_map = {'No': 0, 'Yes': 1}
extra_map = {'No': 0, 'Yes': 1}

# FIX: consistent ordinal encoding
parent_map = {
    'High School': 0,
    'Bachelor': 1,
    'Master': 2,
    'PhD': 3
}

# ---------------- Input Vector ----------------
input_data = np.array([[
    study_hours,
    attendance,
    assignment,
    midterm,
    final,
    participation,
    sleep,
    overall,
    gender_map[gender],
    internet_map[internet],
    extra_map[extra],
    parent_map[parent]
]])

# Safety check
if input_data.shape[1] != scaler.n_features_in_:
    st.error("Feature mismatch! Check training features vs input features.")
    st.stop()

# ---------------- Scale ----------------
input_scaled = scaler.transform(input_data)

# ---------------- Prediction ----------------
if st.button("Predict Grade 🔮"):
    pred = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]

    grade = le_grade.inverse_transform([pred])[0]

    st.success(f"Predicted Grade: {grade}")
    st.write(f"Confidence: {np.max(proba)*100:.2f}%")

    st.subheader("Probability Breakdown")

    for i, cls in enumerate(le_grade.classes_):
        st.write(f"Grade {cls}: {proba[i]*100:.2f}%")