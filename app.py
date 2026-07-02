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

st.subheader("Enter student data")

FIELDS = [
    # key,            label,                    min,   max,   default, step
    ("study_hours",   "Study Hours per Day",     0.0,   10.0,  5.0,  0.1),
    ("attendance",    "Attendance %",            0.0,   100.0, 75.0, 0.5),
    ("assignment",    "Assignment Score",        0.0,   100.0, 70.0, 0.5),
    ("midterm",       "Midterm Score",           0.0,   100.0, 65.0, 0.5),
    ("participation", "Participation Score",     0.0,   100.0, 60.0, 0.5),
    ("sleep",         "Sleep Hours",             0.0,   12.0,  7.0,  0.1),
]

# Initialize session state once so slider and number input start in sync
for key, _, _, _, default, _ in FIELDS:
    slider_key = f"{key}_slider"
    num_key = f"{key}_num"
    if slider_key not in st.session_state:
        st.session_state[slider_key] = default
    if num_key not in st.session_state:
        st.session_state[num_key] = default


def make_sync(slider_key, num_key):
    def sync_from_slider():
        st.session_state[num_key] = st.session_state[slider_key]

    def sync_from_number():
        st.session_state[slider_key] = st.session_state[num_key]

    return sync_from_slider, sync_from_number


values = {}
for key, label, mn, mx, default, step in FIELDS:
    slider_key = f"{key}_slider"
    num_key = f"{key}_num"
    sync_from_slider, sync_from_number = make_sync(slider_key, num_key)

    col1, col2 = st.columns([3, 1])
    with col1:
        st.slider(label, mn, mx, key=slider_key, step=step, on_change=sync_from_slider)
    with col2:
        st.number_input(" ", mn, mx, key=num_key, step=step, on_change=sync_from_number,
                         label_visibility="collapsed")

    values[key] = st.session_state[slider_key]

study_hours = values["study_hours"]
attendance = values["attendance"]
assignment = values["assignment"]
midterm = values["midterm"]
participation = values["participation"]
sleep = values["sleep"]

gender = st.selectbox("Gender", ["Female", "Male"])
internet = st.selectbox("Internet Access", ["No", "Yes"])
extra = st.selectbox("Extra Classes", ["No", "Yes"])
parent = st.selectbox("Parent Education", ["High School", "Bachelor", "Master", "PhD"])

# ---------------- Encoding ----------------
gender_map = {'Female': 0, 'Male': 1}
internet_map = {'No': 0, 'Yes': 1}
extra_map = {'No': 0, 'Yes': 1}


parent_map = {
    'Bachelor': 0,
    'High School': 1,
    'Master': 2,
    'PhD': 3
}

# ---------------- Input Vector ----------------

input_data = np.array([[
    study_hours,
    attendance,
    assignment,
    midterm,
    participation,
    sleep,
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
