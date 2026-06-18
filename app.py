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

# ShapeFactor dropdown options
sf1_options = {"Low (0.003)"   : 0.003,
               "Medium (0.007)": 0.007,
               "High (0.010)"  : 0.010}

sf2_options = {"Low (0.001)"   : 0.001,
               "Medium (0.003)": 0.003,
               "High (0.006)"  : 0.006}

sf3_options = {"Low (0.45)"    : 0.45,
               "Medium (0.70)" : 0.70,
               "High (0.90)"   : 0.90}

sf4_options = {"Low (0.95)"    : 0.95,
               "Medium (0.97)" : 0.97,
               "High (0.99)"   : 0.99}

# Page config
st.set_page_config(
    page_title = "🫘 Bean Classifier",
    page_icon  = "🫘",
    layout     = "wide"
)

# Custom CSS - Minimalist bean themed background
st.markdown("""
    <style>

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0d1b0e 0%, #0a1628 50%, #0d1b0e 100%);
    }

    /* Subtle bean pattern overlay */
    .stApp::before {
        content: "🫘";
        position: fixed;
        font-size: 200px;
        opacity: 0.03;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        pointer-events: none;
        z-index: 0;
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #1a2f1a, #0d2137);
        border: 1px solid #2d5a27;
        border-radius: 15px;
        margin-bottom: 1.5rem;
    }

    /* Result box */
    .result-box {
        text-align: center;
        padding: 2rem;
        border-radius: 15px;
        background: linear-gradient(135deg, #1a2f1a, #0d2137);
        border: 1px solid #2d5a27;
        margin: 1rem 0;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: #1a2f1a !important;
        border-radius: 8px !important;
        border: 1px solid #2d5a27 !important;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #4CAF50 !important;
        font-size: 1.2rem !important;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(90deg, #2d5a27, #1a3a6e) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-size: 1.1rem !important;
        font-weight: bold !important;
        padding: 0.75rem !important;
    }

    .stButton > button:hover {
        background: linear-gradient(90deg, #3d7a37, #2a4a8e) !important;
        transform: scale(1.01);
    }

    /* Slider color */
    [data-testid="stSlider"] > div > div > div {
        background: #2d5a27 !important;
    }

    /* Select box */
    [data-testid="stSelectbox"] > div > div {
        background: #1a2f1a !important;
        border: 1px solid #2d5a27 !important;
        border-radius: 8px !important;
    }

    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class='main-header'>
        <h1 style='color: #7CFC00;'>🫘 Dry Bean Type Classifier</h1>
        <p style='color: #aaa;'>Predict bean variety from physical measurements using SVM</p>
    </div>
""", unsafe_allow_html=True)

# Top metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("🎯 Model",    "SVM (RBF)")
col2.metric("📊 Accuracy", MODEL_ACC)
col3.metric("🫘 Classes",  "7")
col4.metric("📐 Features", "16")

st.markdown("---")

# Input section
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
        Extent      = st.slider("Extent",      0.50, 0.90, 0.75, 0.001)
        Solidity    = st.slider("Solidity",    0.90, 1.00, 0.98, 0.001)
        roundness   = st.slider("Roundness",   0.40, 1.00, 0.85, 0.001)
        Compactness = st.slider("Compactness", 0.60, 1.00, 0.90, 0.001)
    with col2:
        SF1_label = st.selectbox("ShapeFactor1", list(sf1_options.keys()), index=1)
        SF2_label = st.selectbox("ShapeFactor2", list(sf2_options.keys()), index=1)
        SF3_label = st.selectbox("ShapeFactor3", list(sf3_options.keys()), index=1)
        SF4_label = st.selectbox("ShapeFactor4", list(sf4_options.keys()), index=2)

        ShapeFactor1 = sf1_options[SF1_label]
        ShapeFactor2 = sf2_options[SF2_label]
        ShapeFactor3 = sf3_options[SF3_label]
        ShapeFactor4 = sf4_options[SF4_label]

st.markdown("---")

# Predict button
if st.button("🔍 Predict Bean Type", use_container_width=True, type="primary"):

    input_data = np.array([[
        Area, Perimeter, MajorAxisLength, MinorAxisLength,
        AspectRation, Eccentricity, ConvexArea, EquivDiameter,
        Extent, Solidity, roundness, Compactness,
        ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4
    ]])

    input_data   = np.log1p(input_data)
    input_scaled = scaler.transform(input_data)

    prediction       = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)
    predicted_class  = le.inverse_transform(prediction)[0]
    confidence       = prediction_proba.max() * 100

    emoji = bean_info[predicted_class]['color']
    desc  = bean_info[predicted_class]['desc']

    st.markdown(f"""
        <div class='result-box'>
            <h2 style='color: #7CFC00;'>{emoji} Predicted Bean Type : {predicted_class}</h2>
            <p style='color: #aaa;'>{desc}</p>
            <h3 style='color: #4CAF50;'>📊 Confidence : {confidence:.2f}%</h3>
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