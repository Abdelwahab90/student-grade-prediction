import streamlit as st
import numpy as np
import joblib
import os

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Student Performance Predictor",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# =============================================================================
#  UI 
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

:root {
    --bg-deep: #05060a;
    --panel: rgba(255, 255, 255, 0.04);
    --panel-border: rgba(255, 255, 255, 0.08);
    --text-main: #eef0f8;
    --text-dim: #9aa0b4;
}

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
    color: var(--text-main);
}

/* Animated RGB backdrop */
.stApp {
    background:
        radial-gradient(circle at 15% 20%, rgba(255, 0, 128, 0.10), transparent 40%),
        radial-gradient(circle at 85% 15%, rgba(0, 200, 255, 0.10), transparent 40%),
        radial-gradient(circle at 50% 90%, rgba(140, 0, 255, 0.12), transparent 45%),
        var(--bg-deep);
}

/* Hide default streamlit chrome */
#MainMenu, footer, header {visibility: hidden;}

/* ---------------- Title block ---------------- */
.hero {
    text-align: center;
    padding: 28px 20px 22px 20px;
    margin-bottom: 22px;
    border-radius: 22px;
    background: var(--panel);
    border: 1px solid var(--panel-border);
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(14px);
}
.hero::before {
    content: "";
    position: absolute;
    inset: -2px;
    border-radius: 22px;
    padding: 2px;
    background: conic-gradient(from var(--angle, 0deg),
        #ff2ec4, #ffb84d, #58ffb0, #4dd2ff, #b06bff, #ff2ec4);
    -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
    -webkit-mask-composite: xor;
    mask-composite: exclude;
    animation: spin 6s linear infinite;
    opacity: 0.9;
}
@keyframes spin { to { --angle: 360deg; } }
@property --angle {
    syntax: '<angle>';
    initial-value: 0deg;
    inherits: false;
}
.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 2.1rem;
    margin: 0;
    background: linear-gradient(90deg, #ff6ec7, #7dd3fc, #a78bfa, #ff6ec7);
    background-size: 300% auto;
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    animation: shine 5s linear infinite;
}
@keyframes shine { to { background-position: 300% center; } }
.hero p {
    color: var(--text-dim);
    margin-top: 6px;
    font-size: 0.95rem;
    letter-spacing: 0.3px;
}

/* ---------------- Section cards ---------------- */
.section-card {
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 18px;
    padding: 18px 22px 6px 22px;
    margin-bottom: 20px;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 30px rgba(0,0,0,0.25);
}
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1.05rem;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-title .dot {
    width: 9px; height: 9px; border-radius: 50%;
    background: linear-gradient(135deg, #ff6ec7, #4dd2ff);
    box-shadow: 0 0 10px 2px rgba(160, 120, 255, 0.7);
}

/* ---------------- Sliders ---------------- */
div[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, #ff6ec7, #4dd2ff) !important;
}
div[data-testid="stSlider"] div[role="slider"] {
    box-shadow: 0 0 0 6px rgba(160, 120, 255, 0.25), 0 0 14px rgba(77, 210, 255, 0.9) !important;
    background: #ffffff !important;
}
label {
    color: var(--text-main) !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
}

/* ---------------- Number inputs / selects ---------------- */
div[data-testid="stNumberInput"] input, div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
    color: var(--text-main) !important;
}
div[data-testid="stNumberInput"] input:focus {
    border: 1px solid #4dd2ff !important;
    box-shadow: 0 0 0 3px rgba(77, 210, 255, 0.25) !important;
}

/* ---------------- Predict button ---------------- */
div.stButton > button {
    width: 100%;
    padding: 0.85em 0;
    font-size: 1.05rem;
    font-weight: 700;
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: 0.5px;
    color: #05060a;
    border: none;
    border-radius: 14px;
    background: linear-gradient(90deg, #ff6ec7, #ffb84d, #58ffb0, #4dd2ff, #b06bff, #ff6ec7);
    background-size: 300% auto;
    animation: shine 4s linear infinite;
    box-shadow: 0 0 25px rgba(160, 120, 255, 0.45);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
div.stButton > button:hover {
    transform: translateY(-2px) scale(1.01);
    box-shadow: 0 0 40px rgba(160, 120, 255, 0.65);
    color: #05060a;
}
div.stButton > button:active { transform: translateY(0) scale(0.99); }

/* ---------------- Result panel ---------------- */
.result-card {
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 20px;
    padding: 22px 24px;
    margin-top: 18px;
    backdrop-filter: blur(14px);
    box-shadow: 0 0 40px rgba(120, 80, 255, 0.18);
}
.grade-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 800;
    font-size: 2.4rem;
    width: 84px; height: 84px;
    border-radius: 50%;
    color: #05060a;
    margin-right: 18px;
    box-shadow: 0 0 30px rgba(160,120,255,0.55);
}
.result-header { display: flex; align-items: center; margin-bottom: 14px; }
.confidence-text { color: var(--text-dim); font-size: 0.95rem; }
.confidence-text b { color: var(--text-main); }

.prob-row { margin-bottom: 10px; }
.prob-label { display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 4px; color: var(--text-dim); }
.prob-track { background: rgba(255,255,255,0.07); border-radius: 8px; height: 10px; overflow: hidden; }
.prob-fill { height: 100%; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HERO HEADER
# =============================================================================
st.markdown(
    '<div class="hero">'
    '<h1>Student Performance Predictor</h1>'
    '</div>',
    unsafe_allow_html=True
)

# =============================================================================
# LOAD MODEL FILES SAFELY
# =============================================================================
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

# Colors used to paint each grade badge / probability bar consistently
GRADE_COLORS = {
    "A": "linear-gradient(135deg, #58ffb0, #22c98a)",
    "B": "linear-gradient(135deg, #4dd2ff, #2f8fff)",
    "C": "linear-gradient(135deg, #ffd166, #ffb84d)",
    "D": "linear-gradient(135deg, #ff9e6e, #ff6e6e)",
    "F": "linear-gradient(135deg, #ff6ec7, #ff3d81)",
}
DEFAULT_GRADIENT = "linear-gradient(135deg, #b06bff, #4dd2ff)"

# =============================================================================
# NUMERIC FIELDS 
# =============================================================================
FIELDS = [
    # key,            label,                       min,   max,   default, step
    ("study_hours",   "Study Hours per Day",     0.0,   10.0,  5.0,  0.1),
    ("attendance",    "Attendance %",            0.0,   100.0, 75.0, 0.5),
    ("assignment",    "Assignment Score",        0.0,   100.0, 70.0, 0.5),
    ("midterm",       "Midterm Score",           0.0,   100.0, 65.0, 0.5),
    ("final_exam",    "Final Exam Score",        0.0,   100.0, 65.0, 0.5),
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


st.markdown('<div class="section-card"><div class="section-title"><span class="dot"></span>Academic & Lifestyle Metrics</div>', unsafe_allow_html=True)

values = {}
for key, label, mn, mx, default, step in FIELDS:
    slider_key = f"{key}_slider"
    num_key = f"{key}_num"
    sync_from_slider, sync_from_number = make_sync(slider_key, num_key)

    col1, col2 = st.columns([3, 1])
    with col1:
        st.slider(label, mn, mx, key=slider_key, step=step, on_change=sync_from_slider)
    with col2:
        st.number_input("Manual entry", mn, mx, key=num_key, step=step, on_change=sync_from_number,
                         label_visibility="collapsed")

    values[key] = st.session_state[slider_key]

st.markdown('</div>', unsafe_allow_html=True)

study_hours = values["study_hours"]
attendance = values["attendance"]
assignment = values["assignment"]
midterm = values["midterm"]
final_exam = values["final_exam"]
participation = values["participation"]
sleep = values["sleep"]

# =============================================================================
# CATEGORICAL FIELDS
# =============================================================================
st.markdown('<div class="section-card"><div class="section-title"><span class="dot"></span>Student Profile</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    extra = st.selectbox("Extra Classes", ["No", "Yes"])
with c2:
    internet = st.selectbox("Internet Access", ["No", "Yes"])
    parent = st.selectbox("Parent Education", ["High School", "Bachelor", "Master", "PhD"])

st.markdown('</div>', unsafe_allow_html=True)

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

# ---------------- Input Vector (order must match training feature order) ----------------
input_data = np.array([[
    study_hours,
    attendance,
    assignment,
    midterm,
    final_exam,
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

# =============================================================================
# PREDICTION
# =============================================================================
if st.button("Predict Grade"):
    pred = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]
    grade = le_grade.inverse_transform([pred])[0]
    confidence = np.max(proba) * 100
    badge_gradient = GRADE_COLORS.get(grade, DEFAULT_GRADIENT)


    parts = [
        '<div class="result-card">',
        '<div class="result-header">',
        f'<div class="grade-badge" style="background:{badge_gradient};">{grade}</div>',
        '<div>',
        f'<div style="font-family:\'Space Grotesk\',sans-serif; font-weight:700; font-size:1.3rem;">Predicted Grade: {grade}</div>',
        f'<div class="confidence-text">Confidence: <b>{confidence:.2f}%</b></div>',
        '</div>',
        '</div>',
    ]

    for i, cls in enumerate(le_grade.classes_):
        pct = proba[i] * 100
        bar_gradient = GRADE_COLORS.get(cls, DEFAULT_GRADIENT)
        parts.append(
            f'<div class="prob-row"><div class="prob-label"><span>Grade {cls}</span>'
            f'<span>{pct:.2f}%</span></div><div class="prob-track">'
            f'<div class="prob-fill" style="width:{pct}%; background:{bar_gradient};"></div></div></div>'
        )

    parts.append('</div>')
    st.markdown("".join(parts), unsafe_allow_html=True)
