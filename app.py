import streamlit as st
import numpy as np
import joblib
import pandas as pd

# ============================================
GITHUB_LINK  = "https://github.com/Vineeth-Muraleedharan/Bean-classifier"
YOUR_NAME    = "Vineeth Muraleedharan"
MODEL_ACC    = "92.58%"
# ============================================

# Load model, scaler and encoder
model  = joblib.load('bean_model.pkl')
scaler = joblib.load('scaler.pkl')
le     = joblib.load('label_encoder.pkl')

# Bean info
bean_info = {
    'SEKER'    : {'color': '🟡', 'desc': 'Small, round, yellowish bean'},
    'BARBUNYA' : {'color': '🔴', 'desc': 'Medium, speckled pinkish-red bean'},
    'BOMBAY'   : {'color': '⚫', 'desc': 'Large, dark brown bean'},
    'CALI'     : {'color': '🟤', 'desc': 'Medium-large, light brown bean'},
    'DERMASON' : {'color': '🟢', 'desc': 'Small, oval-shaped green bean'},
    'HOROZ'    : {'color': '🟠', 'desc': 'Large, elongated orange-brown bean'},
    'SIRA'     : {'color': '🔵', 'desc': 'Medium, oval-shaped pale bean'}
}

# Page config
st.set_page_config(
    page_title = "🫘 Bean Classifier",
    page_icon  = "🫘",
    layout     = "wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1a1a2e, #16213e);
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .result-box {
        text-align: center;
        padding: 1.5rem;
        border-radius: 10px;
        background: linear-gradient(90deg, #0f3460, #533483);
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class='main-header'>
        <h1>🫘 Dry Bean Type Classifier</h1>
        <p>Predict bean type from physical measurements using Machine Learning</p>
    </div>
""", unsafe_allow_html=True)

# Top metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("🎯 Model",     "SVM (RBF)")
col2.metric("📊 Accuracy",  MODEL_ACC)
col3.metric("🫘 Classes",   "7")
col4.metric("📐 Features",  "16")

st.markdown("---")

# Input section using expanders
st.subheader("📐 Enter Bean Measurements")

with st.expander("📏 Size Features", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        Area            = st.slider("Area",             20000,  260000, 50000,  100)
        Perimeter       = st.slider("Perimeter",        500.0,  2000.0, 800.0,  1.0)
        MajorAxisLength = st.slider("Major Axis Length",150.0,  750.0,  300.0,  1.0)
        ConvexArea      = st.slider("Convex Area",      20000,  270000, 52000,  100)
    with col2:
        MinorAxisLength = st.slider("Minor Axis Length",100.0,  500.0,  200.0,  1.0)
        AspectRation    = st.slider("Aspect Ratio",     1.0,    2.5,    1.5,    0.01)
        Eccentricity    = st.slider("Eccentricity",     0.0,    1.0,    0.7,    0.01)
        EquivDiameter   = st.slider("Equiv Diameter",   150.0,  600.0,  250.0,  1.0)

with st.expander("🔵 Shape Features", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        Extent       = st.slider("Extent",       0.50,   0.90,   0.75,   0.001)
        Solidity     = st.slider("Solidity",     0.90,   1.00,   0.98,   0.001)
        roundness    = st.slider("Roundness",    0.40,   1.00,   0.85,   0.001)
        Compactness  = st.slider("Compactness",  0.60,   1.00,   0.90,   0.001)
    with col2:
        ShapeFactor1 = st.slider("ShapeFactor1", 0.002,  0.012,  0.007,  0.0001)
        ShapeFactor2 = st.slider("ShapeFactor2", 0.0005, 0.007,  0.003,  0.0001)
        ShapeFactor3 = st.slider("ShapeFactor3", 0.40,   1.00,   0.85,   0.001)
        ShapeFactor4 = st.slider("ShapeFactor4", 0.94,   1.00,   0.99,   0.0001)

st.markdown("---")

# Predict button
if st.button("🔍 Predict Bean Type", use_container_width=True, type="primary"):

    # Prepare input
    input_data = np.array([[
        Area, Perimeter, MajorAxisLength, MinorAxisLength,
        AspectRation, Eccentricity, ConvexArea, EquivDiameter,
        Extent, Solidity, roundness, Compactness,
        ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4
    ]])

    # Apply log transformation
    input_data = np.log1p(input_data)

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction       = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)
    predicted_class  = le.inverse_transform(prediction)[0]
    confidence       = prediction_proba.max() * 100

    # Result box
    emoji = bean_info[predicted_class]['color']
    desc  = bean_info[predicted_class]['desc']

    st.markdown(f"""
        <div class='result-box'>
            <h2>{emoji} Predicted Bean Type : {predicted_class}</h2>
            <p>{desc}</p>
            <h3>📊 Confidence : {confidence:.2f}%</h3>
        </div>
    """, unsafe_allow_html=True)

    # Probability metrics
    st.markdown("### 🎯 Class Probabilities")
    proba_dict = dict(zip(le.classes_, prediction_proba[0].round(4)))

    col1, col2, col3, col4 = st.columns(4)
    col5, col6, col7, _    = st.columns(4)

    cols = [col1, col2, col3, col4, col5, col6, col7]
    for i, (cls, prob) in enumerate(proba_dict.items()):
        cols[i].metric(
            label = f"{bean_info[cls]['color']} {cls}",
            value = f"{prob*100:.1f}%"
        )

# Footer
st.markdown("---")
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown(f"**👤 Developer:** {YOUR_NAME}")
with col_b:
    st.markdown(f"**🎯 Best Model:** SVM | Accuracy: {MODEL_ACC}")
with col_c:
    st.markdown(f"**💻 [GitHub Repo]({GITHUB_LINK})**")