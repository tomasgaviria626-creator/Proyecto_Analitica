import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ============================
# CONFIGURACIÓN BÁSICA
# ============================
st.set_page_config(
    page_title="Predicción de Enfermedad Cardíaca",
    page_icon="❤️",
    layout="wide",
)

# ============================
# ESTILOS (CSS)
# ============================
st.markdown("""
<style>
/* Fondo general y color del texto */
.stApp {
    background: radial-gradient(circle at top, #1e293b 0, #020617 45%, #000000 100%);
    color: #f9fafb;
}

/* Centrar contenido y limitar ancho */
section.main > div {
    max-width: 1100px;
    margin: 0 auto;
}

/* Tarjetas */
.card {
    background: rgba(15,23,42,0.96);
    padding: 1.5rem 1.75rem;
    border-radius: 18px;
    border: 1px solid rgba(148,163,184,0.45);
    box-shadow: 0 18px 45px rgba(15,23,42,0.85);
}

/* Texto pequeño y aclaraciones */
.small-label {
    font-size: 0.85rem;
    color: #e5e7eb;
}

/* Número grande de la probabilidad */
.big-number {
    font-size: 3.6rem;
    font-weight: 700;
}

/* Texto secundario */
.subtle {
    color: #cbd5f5;
}

/* Títulos */
h1, h2, h3, h4 {
    color: #f9fafb;
}

/* Quitar el borde gris de los inputs y hacerlos más claros */
div[data-baseweb="input"] > div {
    background-color: rgba(15,23,42,0.85);
    border-radius: 10px;
    border: 1px solid rgba(148,163,184,0.7);
    color: #f9fafb;
}

label {
    color: #e5e7eb !important;
}

/* Botón principal */
.stButton>button {
    background: linear-gradient(90deg,#ec4899,#ef4444);
    color: white;
    border-radius: 999px;
    padding: 0.6rem 1.6rem;
    font-weight: 600;
    border: none;
}
.stButton>button:hover {
    background: linear-gradient(90deg,#f97316,#ef4444);
}

/* Slider color */
.stSlider > div > div > div[data-baseweb="slider"] > div {
    background-color: #38bdf8;
}
</style>
""", unsafe_allow_html=True)

# ============================
# CARGAR MODELO
# ============================
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error("No se pudo cargar el modelo `model.pkl`.")
    st.exception(e)
    st.stop()

# Columnas que espera el modelo
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalch", "exang", "oldpeak", "slope",
    "ca", "thal"
]

# ============================
# TÍTULO Y DESCRIPCIÓN
# ============================
st.markdown(
    "<h1 style='margin-bottom:0.1rem;'>❤️ Predicción de Enfermedad Cardíaca</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p class='subtle' style='margin-bottom:1.2rem;'>"
    "Completa la información de la persona y estima la probabilidad de tener "
    "enfermedad cardiaca"
    "</p>",
    unsafe_allow_html=True,
)

# ============================
# FORMULARIO DE ENTRADA
# ============================
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Datos de la persona")

    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.slider("Edad", 18, 90, 55,
                        help="Edad en años.")

        trestbps = st.slider(
            "Presión arterial en reposo",
            80, 200, 130,
            help="Presión arterial en reposo en mmHg."
        )

        chol = st.slider(
            "Colesterol total",
            100, 600, 240,
            help="Colesterol sérico en mg/dl."
        )

        fbs = st.selectbox(
            "Glucosa en ayunas > 120 mg/dl",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Sí",
            help="si la glucosa en ayunas es mayor a 120 mg/dl; en otro caso No."
        )

    with c2:
        sex = st.selectbox(
            "Sexo biológico",
            options=[0, 1],
            format_func=lambda x: "Femenino" if x == 0 else "Masculino",
            help="Sexo biológico."
        )

        cp = st.selectbox(
            "Tipo de dolor en el pecho",
            options=[0, 1, 2, 3],
            format_func=lambda x: [
                "Angina típica", "Angina atípica",
                "Dolor no anginoso", "Asintomático"
            ][x],
            help="Categoría que describe el tipo de dolor torácico."
        )

        restecg = st.selectbox(
            "ECG en reposo",
            options=[0, 1, 2],
            format_func=lambda x: [
                "Normal",
                "Anormalidad de ST-T",
                "Hipertrofia ventricular izquierda"
            ][x],
            help="Resultado del electrocardiograma en reposo."
        )

        slope = st.selectbox(
            "Pendiente del segmento ST",
            options=[0, 1, 2],
            format_func=lambda x: [
                "Ascendente",
                "Plano",
                "Descendente"
            ][x],
            help="Pendiente del segmento ST en la prueba de esfuerzo."
        )

    with c3:
        thalch = st.slider(
            "Frecuencia cardiaca máxima",
            60, 220, 150,
            help="Frecuencia cardiaca máxima alcanzada (latidos por minuto)."
        )

        exang = st.selectbox(
            "Angina inducida por ejercicio",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Sí",
            help="si aparece angina durante el ejercicio; en otro caso No."
        )

        oldpeak = st.slider(
            "Depresión del ST",
            0.0, 6.0, 1.0, 0.1,
            help="Depresión del segmento ST respecto al reposo."
        )

        ca = st.selectbox(
            "Número de vasos principales",
            options=[0, 1, 2, 3],
            help="Cantidad de vasos principales coloreados por fluoroscopia (0-3)."
        )

        thal = st.selectbox(
            "Talemia (thal)",
            options=[0, 1, 2, 3],
            format_func=lambda x: [
                "Desconocido", "Normal", "Defecto fijo", "Defecto reversible"
            ][x],
            help="Resultado de la prueba de talemia."
        )

    st.markdown("</div>", unsafe_allow_html=True)  # cierra card

# ============================
# ARMAR DATAFRAME CON LOS DATOS
# ============================
input_data = pd.DataFrame([{
    "age": age,
    "sex": sex,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs,
    "restecg": restecg,
    "thalch": thalch,
    "exang": exang,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": ca,
    "thal": thal,
}])[FEATURES]  # asegura el orden correcto

st.markdown("<br>", unsafe_allow_html=True)

# Mostrar tabla de datos que se mandan al modelo
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("#### Datos que se enviarán al modelo:")
st.dataframe(input_data.style.format(precision=2), use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ============================
# BOTÓN Y PREDICCIÓN
# ============================
st.markdown("<br>", unsafe_allow_html=True)

col_btn, col_empty = st.columns([1, 3])
with col_btn:
    pressed = st.button("🔍 Predecir riesgo")

if pressed:
    try:
        proba_enfermo = float(model.predict_proba(input_data)[0, 1])

        # Elegir color según riesgo
        if proba_enfermo < 0.30:
            color = "#22c55e"
            msg = "Bajo"
            detalle = "Según el modelo, la probabilidad de enfermedad es baja."
        elif proba_enfermo < 0.70:
            color = "#eab308"
            msg = "Moderado"
            detalle = "Según el modelo, la probabilidad de enfermedad es moderada."
        else:
            color = "#ef4444"
            msg = "Alto"
            detalle = (
                "Según el modelo, la probabilidad de enfermedad es alta. "
                "Se recomienda una valoración médica detallada."
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        st.markdown(
            "<h3 style='margin-top:0;'>Probabilidad estimada de enfermedad</h3>",
            unsafe_allow_html=True,
        )

        numero_html = (
            f"<p class='big-number' style='color:{color};'>"
            f"{proba_enfermo*100:.1f} %</p>"
        )
        badge_html = (
            f"<span style='background-color:{color}33;"
            f"color:{color};padding:0.25rem 0.9rem;border-radius:999px;"
            f"font-weight:600;font-size:0.95rem;'>"
            f"Riesgo {msg}</span>"
        )

        st.markdown(numero_html + badge_html, unsafe_allow_html=True)
        st.markdown(
            f"<p class='subtle' style='margin-top:0.7rem;'>{detalle}</p>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<p class='small-label'>⚠️ Este resultado se basa en datos históricos y "
            "no sustituye una evaluación médica profesional.</p>",
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error("Ocurrió un error al hacer la predicción.")
        st.exception(e)