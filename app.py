import streamlit as st
import numpy as np
import joblib

# ============================================
#         EDIT YOUR LINKS HERE
# ============================================
GITHUB_LINK  = "https://github.com/Vineeth-Muraleedharan/Bean-classifier"
YOUR_NAME    = "Vineeth Muraleedharan"
MODEL_ACC    = "92.58%"
# ============================================

# Load model, scaler and encoder
model  = joblib.load('bean_model.pkl')
scaler = joblib.load('scaler.pkl')
le     = joblib.load('label_encoder.pkl')

# Bean info dictionary
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
    .metric-box {
        text-align: center;
        padding: 0.5rem;
        border-radius: 8px;
        background: #1a1a2e;
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
col1.metric("🎯 Model", "SVM (RBF)")
col2.metric("📊 Accuracy", MODEL_ACC)
col3.metric("🫘 Bean Classes", "7")
col4.metric("📐 Features", "16")

st.markdown("---")

# Sidebar for inputs
st.sidebar.title("📐 Bean Measurements")
st.sidebar.markdown("Adjust the sliders to input bean measurements:")

st.sidebar.subheader("📏 Size Features")
Area            = st.sidebar.slider("Area",             min_value=20000,  max_value=260000, value=50000,  step=100)
Perimeter       = st.sidebar.slider("Perimeter",        min_value=500.0,  max_value=2000.0, value=800.0,  step=1.0)
MajorAxisLength = st.sidebar.slider("Major Axis Length",min_value=150.0,  max_value=750.0,  value=300.0,  step=1.0)
MinorAxisLength = st.sidebar.slider("Minor Axis Length",min_value=100.0,  max_value=500.0,  value=200.0,  step=1.0)
AspectRation    = st.sidebar.slider("Aspect Ratio",     min_value=1.0,    max_value=2.5,    value=1.5,    step=0.01)
Eccentricity    = st.sidebar.slider("Eccentricity",     min_value=0.0,    max_value=1.0,    value=0.7,    step=0.01)
ConvexArea      = st.sidebar.slider("Convex Area",      min_value=20000,  max_value=270000, value=52000,  step=100)
EquivDiameter   = st.sidebar.slider("Equiv Diameter",   min_value=150.0,  max_value=600.0,  value=250.0,  step=1.0)

st.sidebar.subheader("🔵 Shape Features")
Extent       = st.sidebar.slider("Extent",       min_value=0.50, max_value=0.90, value=0.75, step=0.001)
Solidity     = st.sidebar.slider("Solidity",     min_value=0.90, max_value=1.00, value=0.98, step=0.001)
roundness    = st.sidebar.slider("Roundness",    min_value=0.40, max_value=1.00, value=0.85, step=0.001)
Compactness  = st.sidebar.slider("Compactness",  min_value=0.60, max_value=1.00, value=0.90, step=0.001)
ShapeFactor1 = st.sidebar.slider("ShapeFactor1", min_value=0.002, max_value=0.012, value=0.007, step=0.0001)
ShapeFactor2 = st.sidebar.slider("ShapeFactor2", min_value=0.0005, max_value=0.007, value=0.003, step=0.0001)
ShapeFactor3 = st.sidebar.slider("ShapeFactor3", min_value=0.40, max_value=1.00, value=0.85, step=0.001)
ShapeFactor4 = st.sidebar.slider("ShapeFactor4", min_value=0.94, max_value=1.00, value=0.99, step=0.0001)

# Main area - show current values
st.subheader("📋 Current Input Values")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**📏 Size Features**")
    size_data = {
        'Feature'  : ['Area', 'Perimeter', 'Major Axis', 'Minor Axis',
                      'Aspect Ratio', 'Eccentricity', 'Convex Area', 'Equiv Diameter'],
        'Value'    : [Area, Perimeter, MajorAxisLength, MinorAxisLength,
                      AspectRation, Eccentricity, ConvexArea, EquivDiameter]
    }
    import pandas as pd
    st.dataframe(pd.DataFrame(size_data), use_container_width=True, hide_index=True)

with col2:
    st.markdown("**🔵 Shape Features**")
    shape_data = {
        'Feature'  : ['Extent', 'Solidity', 'Roundness', 'Compactness',
                      'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4'],
        'Value'    : [Extent, Solidity, roundness, Compactness,
                      ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4]
    }
    st.dataframe(pd.DataFrame(shape_data), use_container_width=True, hide_index=True)

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

    # Result
    emoji  = bean_info[predicted_class]['color']
    desc   = bean_info[predicted_class]['desc']

    st.markdown(f"""
        <div class='result-box'>
            <h2>{emoji} Predicted Bean Type: {predicted_class}</h2>
            <p>{desc}</p>
            <h3>📊 Confidence: {confidence:.2f}%</h3>
        </div>
    """, unsafe_allow_html=True)

    # Probability chart
    st.markdown("### 📈 Class Probabilities")
    proba_dict = dict(zip(le.classes_, prediction_proba[0].round(4)))
    st.bar_chart(proba_dict)

    # Confidence columns
    st.markdown("### 🎯 Detailed Probabilities")
    cols = st.columns(7)
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