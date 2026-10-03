import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import load_iris

# ==========================================
# ⚙️ Page Config
# ==========================================

st.set_page_config(
    page_title="Iris Flower Prediction",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 🎨 Custom CSS
# ==========================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    /* Background gradient */
    .stApp {
        background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 50%, #e0eafc 100%);
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Hero header */
    .hero {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        margin-bottom: 2rem;
    }
    .hero h1 { font-size: 2.8rem; font-weight: 700; margin: 0; color: white; }
    .hero p  { font-size: 1.1rem; margin-top: 0.8rem; opacity: 0.95; font-weight: 300; color: white; }

    /* Section titles */
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #2d3748;
        margin: 1rem 0 1rem 0;
        padding-left: 0.8rem;
        border-left: 4px solid #667eea;
    }

    /* Prediction result box */
    .result-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.35);
        margin-top: 1rem;
    }
    .result-box h2 { color: white; font-size: 2rem; margin: 0.5rem 0; font-weight: 700; }
    .result-box p  { color: rgba(255,255,255,0.9); font-size: 1rem; margin: 0.3rem 0; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1.05rem;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        color: white;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #a0aec0;
        font-size: 0.85rem;
        padding: 2rem 0 1rem 0;
    }

    hr {
        border: none;
        height: 1px;
        background: linear-gradient(to right, transparent, #cbd5e0, transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🤖 Load Model & Scaler
# ==========================================

@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_artifacts()
except FileNotFoundError:
    st.error("❌ Model files not found! Please make sure `iris_model.pkl` and `scaler.pkl` are in the app directory.")
    st.stop()

# ==========================================
# 📊 Load Dataset (for preview only)
# ==========================================

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target
df["species_name"] = df["species"].map(dict(enumerate(iris.target_names)))

# ==========================================
# 🌸 Hero Header
# ==========================================

st.markdown("""
<div class="hero">
    <h1>🌸 Iris Flower Prediction</h1>
    <p>Enter the flower measurements below and let the ML model identify the species</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 🎛️ Sidebar Inputs - NUMBER INPUTS
# ==========================================

with st.sidebar:
    st.markdown("## 🎛️ Input Parameters")
    st.markdown("Enter the flower measurements below.")

    sepal_length = st.number_input(
        "Sepal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=5.1,
        step=0.1,
        format="%.1f"
    )

    sepal_width = st.number_input(
        "Sepal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=3.5,
        step=0.1,
        format="%.1f"
    )

    petal_length = st.number_input(
        "Petal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=1.4,
        step=0.1,
        format="%.1f"
    )

    petal_width = st.number_input(
        "Petal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=0.2,
        step=0.1,
        format="%.1f"
    )

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Species")

# ==========================================
# ⚙️ Feature Engineering (must match training!)
# ==========================================
# The model was trained on 6 features:
# [sepal_length, sepal_width, petal_length, petal_width, sepal_area, petal_area]

sepal_area = sepal_length * sepal_width
petal_area = petal_length * petal_width

# ==========================================
# 🎯 Prediction Section
# ==========================================

st.markdown('<div class="section-title">🎯 Prediction</div>', unsafe_allow_html=True)

if predict_btn:
    # Prepare input in SAME order as training:
    # [sepal_length, sepal_width, petal_length, petal_width, sepal_area, petal_area]
    input_data = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width,
        sepal_area,
        petal_area
    ]])

    # Scale features using the loaded scaler
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)[0]
    species_name = iris.target_names[prediction]
    emoji = {0: "🌼", 1: "🌺", 2: "🌸"}[prediction]

    # Get probabilities
    try:
        proba = model.predict_proba(input_scaled)[0]
        confidence = proba[prediction] * 100
    except Exception:
        proba = None
        confidence = None

    # Display result box
    conf_text = f"<p>Confidence: <b>{confidence:.2f}%</b></p>" if confidence else ""
    st.markdown(f"""
    <div class="result-box">
        <p>Predicted Species</p>
        <h2>{emoji} {species_name.capitalize()}</h2>
        {conf_text}
    </div>
    """, unsafe_allow_html=True)

    # Probability distribution table
    if proba is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Probability Distribution**")
        proba_df = pd.DataFrame({
            "Species": [s.capitalize() for s in iris.target_names],
            "Probability": proba
        })
        st.dataframe(
            proba_df.style.format({"Probability": "{:.2%}"}).background_gradient(
                cmap="Purples", subset=["Probability"]
            ),
            use_container_width=True,
            hide_index=True
        )
else:
    st.markdown("""
    <div style="background:white; padding: 3rem 1.5rem; border-radius: 16px;
                text-align:center; box-shadow: 0 4px 20px rgba(0,0,0,0.06);">
        <h3 style="color:#4a5568; margin:0;">👈 Ready when you are</h3>
        <p style="color:#718096;">Enter the measurements and click <b>Predict Species</b>.</p>
    </div>
    """, unsafe_allow_html=True)



# ==========================================
# 🦶 Footer
# ==========================================

st.markdown("""
<div class="footer">
    Built with ❤️ using Streamlit &nbsp;|&nbsp; Powered by scikit-learn
</div>
""", unsafe_allow_html=True)