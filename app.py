import streamlit as st
import numpy as np
import joblib


GITHUB_LINK   = "https://github.com/Vineeth-Muraleedharan/Bean-classifier"
STREAMLIT_LINK = "https://your-app-link.streamlit.app"
YOUR_NAME     = "VINEETH MURALEEDHARAN"
MODEL_ACC     = "92.58%"


# Load model, scaler and encoder
model  = joblib.load('bean_model.pkl')
scaler = joblib.load('scaler.pkl')
le     = joblib.load('label_encoder.pkl')

# Page config
st.set_page_config(
    page_title = " Bean Classifier",
    page_icon  = "🫘",
    layout     = "wide"
)

# Title
st.title(" Dry Bean Type Classifier")
st.markdown(f"**Developed by:** {YOUR_NAME}")
st.markdown("Enter the physical measurements of a bean to predict its type.")
st.markdown("---")

# Input features - 2 columns layout
col1, col2 = st.columns(2)

with col1:
    st.subheader(" Size Features")
    Area            = st.number_input("Area",              min_value=0.0, value=50000.0)
    Perimeter       = st.number_input("Perimeter",         min_value=0.0, value=800.0)
    MajorAxisLength = st.number_input("Major Axis Length", min_value=0.0, value=300.0)
    MinorAxisLength = st.number_input("Minor Axis Length", min_value=0.0, value=200.0)
    AspectRation    = st.number_input("Aspect Ratio",      min_value=0.0, value=1.5)
    Eccentricity    = st.number_input("Eccentricity",      min_value=0.0, value=0.7)
    ConvexArea      = st.number_input("Convex Area",       min_value=0.0, value=52000.0)
    EquivDiameter   = st.number_input("Equiv Diameter",    min_value=0.0, value=250.0)

with col2:
    st.subheader(" Shape Features")
    Extent       = st.number_input("Extent",       min_value=0.0, value=0.75)
    Solidity     = st.number_input("Solidity",     min_value=0.0, value=0.98)
    roundness    = st.number_input("Roundness",    min_value=0.0, value=0.85)
    Compactness  = st.number_input("Compactness",  min_value=0.0, value=0.90)
    ShapeFactor1 = st.number_input("ShapeFactor1", min_value=0.0, value=0.007, format="%.4f")
    ShapeFactor2 = st.number_input("ShapeFactor2", min_value=0.0, value=0.003, format="%.4f")
    ShapeFactor3 = st.number_input("ShapeFactor3", min_value=0.0, value=0.85)
    ShapeFactor4 = st.number_input("ShapeFactor4", min_value=0.0, value=0.99)

st.markdown("---")

# Predict button
if st.button("🔍 Predict Bean Type", use_container_width=True):

    # Prepare input
    input_data = np.array([[
        Area, Perimeter, MajorAxisLength, MinorAxisLength,
        AspectRation, Eccentricity, ConvexArea, EquivDiameter,
        Extent, Solidity, roundness, Compactness,
        ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4
    ]])

    # Apply log transformation (same as Step 4)
    input_data = np.log1p(input_data)

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction       = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)
    predicted_class  = le.inverse_transform(prediction)[0]
    confidence       = prediction_proba.max() * 100

    # Display result
    st.success(f"🫘 Predicted Bean Type : **{predicted_class}**")
    st.info(f" Confidence          : **{confidence:.2f}%**")

    # Show all class probabilities
    st.markdown("###  Class Probabilities")
    proba_dict = dict(zip(le.classes_, prediction_proba[0].round(4)))
    st.bar_chart(proba_dict)

# Footer
st.markdown("---")
col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown(f"**👤 Developer:** {YOUR_NAME}")
with col_b:
    st.markdown(f"**💻 [GitHub Repo]({https://github.com/Vineeth-Muraleedharan/Bean-classifier})**")