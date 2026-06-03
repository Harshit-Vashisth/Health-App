# -*- coding: utf-8 -*-
"""
Health App — Multiple Disease Prediction System
Modernised UI (dark theme) + hardened inputs.

@author: HP
"""

import pickle
import time
from pathlib import Path

import requests
import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu

BASE_DIR = Path(__file__).parent


# ----------------------------------------------------------------------
# Cached resource / data helpers
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_models():
    """Load and cache the pickled ML models (loaded once per session)."""
    return {
        "heart": pickle.load(open(BASE_DIR / "heartmodel.sav", "rb")),
        "parkinsons": pickle.load(open(BASE_DIR / "parkinsons_model.sav", "rb")),
        "diabetes": pickle.load(open(BASE_DIR / "diabet_model.sav", "rb")),
    }


@st.cache_data(show_spinner=False)
def load_lottie_url(url: str):
    """Fetch a Lottie animation JSON, cached so we don't re-download on rerun."""
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()
    except requests.RequestException:
        return None
    return None


def render_lottie(url, *, key, height=300, speed=1.0, loop=True, reverse=True):
    """Render a Lottie animation if it could be fetched; silently skip otherwise."""
    data = load_lottie_url(url)
    if data:
        st_lottie(data, speed=speed, height=height, key=key, loop=loop, reverse=reverse)


def load_css():
    """Inject the dark-theme stylesheet."""
    css_path = BASE_DIR / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Small UI building blocks
# ----------------------------------------------------------------------
def hero(title, subtitle, emoji=""):
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-emoji">{emoji}</div>
          <div>
            <h1 class="hero-title">{title}</h1>
            <p class="hero-sub">{subtitle}</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_result(is_positive, positive_msg, negative_msg):
    if is_positive:
        st.markdown(
            f'<div class="result-card result-danger">⚠️ {positive_msg}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="result-card result-safe">✅ {negative_msg}</div>',
            unsafe_allow_html=True,
        )


# Lottie animation URLs
LOTTIE = {
    "main": "https://lottie.host/114e906b-1add-4784-b94a-a38be8eed867/ScwUP2rhYG.json",
    "heart": "https://lottie.host/ade65593-1277-4972-9414-b30559ea958f/7eY58mtlPi.json",
    "parkinsons": "https://lottie.host/8d3128ff-32ce-4e13-b1ef-2c05de418e13/Yj8xS4ej8G.json",
    "diabetes": "https://lottie.host/aa7dd75d-bbf1-45b2-a21e-58ff70782e45/r0Pvq0kPmF.json",
    "result": "https://lottie.host/952cbc9d-4174-4430-862c-2e800877a3cf/ayjbjIGcM9.json",
    "soon": "https://lottie.host/58fc64b0-ae86-4010-8d95-4de59ff47780/XgbzsKyeLG.json",
}


# ----------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------
def page_main():
    hero("HEALTH APP", "AI-powered screening for common conditions — fast, private, on-device.", "🩺")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="card"><h3>❤️ Heart</h3><p class="hero-sub">Cardiac risk from clinical markers.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><h3>🧠 Parkinson\'s</h3><p class="hero-sub">Voice-measure based screening.</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card"><h3>🩸 Diabetes</h3><p class="hero-sub">Diabetes likelihood from vitals.</p></div>', unsafe_allow_html=True)
    render_lottie(LOTTIE["main"], key="lottie_main", height=420, speed=0.6)
    st.caption("⚕️ For educational use only — not a substitute for professional medical advice.")


def page_heart(model):
    hero("Heart Disease Prediction", "Enter the clinical measurements below.", "❤️")
    render_lottie(LOTTIE["heart"], key="lottie_heart", height=220, speed=0.8)

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", 0, 130, 45)
            cp = st.selectbox("Chest pain type", [0, 1, 2, 3], index=1,
                              help="0: typical angina · 1: atypical · 2: non-anginal · 3: asymptomatic")
            chol = st.number_input("Serum cholesterol (mg/dl)", 0, 700, 200)
            restecg_label = st.radio("Resting ECG results",
                                     ["Normal", "Borderline abnormality", "Abnormality / arrhythmia"])
            thalach = st.number_input("Max heart rate achieved", 0, 250, 150)
            slope = st.selectbox("Slope of peak exercise ST segment", [0, 1, 2], index=1)
            thal = st.selectbox("Thal", [0, 1, 2], index=2,
                                help="0: normal · 1: fixed defect · 2: reversible defect")
        with c2:
            sex = 1 if st.radio("Gender", ["Male", "Female"]) == "Male" else 0
            trestbps = st.number_input("Resting blood pressure (mm Hg)", 50, 250, 120)
            fbs_value = st.number_input("Fasting blood sugar (mg/dl)", 0, 600, 100)
            exang = 1 if st.radio("Exercise-induced angina", ["No", "Yes"]) == "Yes" else 0
            oldpeak = st.number_input("ST depression (oldpeak)", 0.0, 10.0, 1.0, step=0.1)
            ca = st.slider("Major vessels colored by fluoroscopy", 0, 3, 0)

        restecg = {"Normal": 0, "Borderline abnormality": 1, "Abnormality / arrhythmia": 2}[restecg_label]
        fbs = 1 if fbs_value > 120 else 0  # model expects the binary >120 mg/dl flag

        if st.button("📊  Predict"):
            features = [[age, sex, cp, trestbps, chol, fbs, restecg,
                         thalach, exang, oldpeak, slope, ca, thal]]
            try:
                positive = model.predict(features)[0] == 1
                render_lottie(LOTTIE["result"], key="lottie_heart_res", height=160, speed=2.0, loop=False)
                time.sleep(1.2)
                show_result(positive,
                            "Signs of heart disease detected. Please consult a cardiologist.",
                            "No signs of heart disease detected.")
            except Exception as e:
                st.error(f"Could not run prediction: {e}")
        st.markdown('</div>', unsafe_allow_html=True)


def page_parkinsons(model):
    hero("Parkinson's Disease Prediction", "Enter the voice-measurement features.", "🧠")
    render_lottie(LOTTIE["parkinsons"], key="lottie_park", height=220, speed=1.0)

    labels = [
        "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)",
        "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)",
        "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR",
        "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE",
    ]
    st.markdown('<div class="card">', unsafe_allow_html=True)
    values = []
    cols = st.columns(4)
    for i, label in enumerate(labels):
        with cols[i % 4]:
            values.append(st.number_input(label, value=0.0, step=0.0001, format="%.5f"))

    if st.button("📊  Predict"):
        try:
            positive = model.predict([values])[0] == 1
            render_lottie(LOTTIE["result"], key="lottie_park_res", height=160, speed=2.0, loop=False)
            time.sleep(1.2)
            show_result(positive,
                        "Indicators consistent with Parkinson's detected. Please consult a neurologist.",
                        "No indicators of Parkinson's detected.")
        except Exception as e:
            st.error(f"Could not run prediction: {e}")
    st.markdown('</div>', unsafe_allow_html=True)


def page_diabetes(model):
    hero("Diabetes Prediction", "Enter the patient vitals below.", "🩸")
    render_lottie(LOTTIE["diabetes"], key="lottie_dia", height=200, speed=1.5, loop=False)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        preg = st.number_input("Pregnancies", 0, 20, 1)
        bp = st.number_input("Diastolic blood pressure (mm Hg)", 0, 200, 70)
        insul = st.number_input("Insulin (2-hour serum, mu U/ml)", 0, 900, 80)
        pedi = st.number_input("Diabetes pedigree function", 0.0, 3.0, 0.5, step=0.01)
    with c2:
        glu = st.number_input("Glucose (plasma concentration)", 0, 300, 110)
        tricep = st.number_input("Triceps skin fold thickness (mm)", 0, 100, 20)
        bmi = st.number_input("Body mass index (BMI)", 0.0, 70.0, 25.0, step=0.1)
        age = st.number_input("Age", 0, 130, 33)

    if st.button("📊  Predict"):
        features = [[preg, glu, bp, tricep, insul, bmi, pedi, age]]
        try:
            positive = model.predict(features)[0] == 1
            render_lottie(LOTTIE["result"], key="lottie_dia_res", height=160, speed=2.0, loop=False)
            time.sleep(1.2)
            show_result(positive,
                        "Signs of diabetes detected. Please consult a physician.",
                        "No signs of diabetes detected.")
        except Exception as e:
            st.error(f"Could not run prediction: {e}")
    st.markdown('</div>', unsafe_allow_html=True)


def page_coming_soon(title, emoji):
    hero(title, "This feature is on the way — thank you for your patience.", emoji)
    render_lottie(LOTTIE["soon"], key=f"lottie_soon_{title}", height=420, speed=0.6)
    st.markdown('<div class="coming-soon">🚧  Available soon</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------
# App shell
# ----------------------------------------------------------------------
def main_code():
    load_css()
    models = load_models()

    with st.sidebar:
        selected = option_menu(
            "Multiple Disease Prediction",
            ["Main Page",
             "Heart Disease Prediction",
             "Parkinsons Disease Prediction",
             "Diabetes Prediction",
             "AI Medical Bot",
             "Know My Medicine details"],
            icons=["house", "activity", "person", "clipboard", "robot", "upload"],
            default_index=0,
            styles={
                "container": {"padding": "6px", "background-color": "transparent"},
                "icon": {"color": "#00d9b8", "font-size": "18px"},
                "nav-link": {"font-size": "15px", "color": "#cfd6e4",
                             "border-radius": "10px", "--hover-color": "#1d2230"},
                "nav-link-selected": {
                    "background": "linear-gradient(90deg,#00d9b8,#3b82f6)",
                    "color": "#04121f", "font-weight": "700"},
            },
        )

    if selected == "Main Page":
        page_main()
    elif selected == "Heart Disease Prediction":
        page_heart(models["heart"])
    elif selected == "Parkinsons Disease Prediction":
        page_parkinsons(models["parkinsons"])
    elif selected == "Diabetes Prediction":
        page_diabetes(models["diabetes"])
    elif selected == "AI Medical Bot":
        page_coming_soon("AI Medical Bot", "🤖")
    elif selected == "Know My Medicine details":
        page_coming_soon("Know My Medicine details", "💊")


def main():
    st.set_page_config(
        page_title="Health App",
        page_icon="https://cdn-icons-png.flaticon.com/512/2966/2966327.png",
        layout="wide",
    )
    main_code()


if __name__ == "__main__":
    main()
