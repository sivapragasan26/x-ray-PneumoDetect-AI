import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import os
import time

# ----------------------------
# MODEL LOGIC (unchanged)
# ----------------------------
def load_pneumonia_model():
    possible_paths = [
        'best_chest_xray_model.h5',
        './best_chest_xray_model.h5',
        'api/streamlit_api_folder/best_chest_xray_model.h5',
        '/mount/src/chest-xray-pneumonia-detection-ai/api/streamlit_api_folder/best_chest_xray_model.h5',
        './models/best_chest_xray_model.h5',
        'models/pneumonia_model.h5'
    ]
    for model_path in possible_paths:
        try:
            if os.path.exists(model_path):
                model = tf.keras.models.load_model(model_path, compile=False)
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                return model
        except:
            continue
    return None

def preprocess_image(image_input):
    if isinstance(image_input, str):
        image = Image.open(image_input)
    else:
        image = image_input
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((224, 224))
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def interpret_prediction(prediction_score):
    if prediction_score > 0.5:
        diagnosis = "PNEUMONIA"
        confidence = float(prediction_score * 100)
        if confidence >= 80:
            confidence_level = "High"
            recommendation = "🚨 Strong indication of pneumonia. Seek immediate medical attention."
        elif confidence >= 60:
            confidence_level = "Moderate"
            recommendation = "⚠️ Moderate indication of pneumonia. Medical review recommended."
        else:
            confidence_level = "Low"
            recommendation = "💡 Possible pneumonia detected. Further examination advised."
    else:
        diagnosis = "NORMAL"
        confidence = float((1 - prediction_score) * 100)
        if confidence >= 80:
            confidence_level = "High"
            recommendation = "✅ No signs of pneumonia detected. Chest X-ray appears normal."
        elif confidence >= 60:
            confidence_level = "Moderate"
            recommendation = "👍 Likely normal chest X-ray. Routine follow-up if symptoms persist."
        else:
            confidence_level = "Low"
            recommendation = "🤔 Unclear result. Manual review by radiologist recommended."
    return {
        "diagnosis": diagnosis,
        "confidence": round(confidence, 2),
        "confidence_level": confidence_level,
        "recommendation": recommendation,
        "raw_score": float(prediction_score),
        "threshold": 0.5,
        "model_architecture": "MobileNetV2"
    }

def predict_pneumonia(image_input, model=None):
    try:
        if model is None:
            model = load_pneumonia_model()
            if model is None:
                raise Exception("Could not load pneumonia detection model")
        processed_image = preprocess_image(image_input)
        prediction = model.predict(processed_image, verbose=0)[0][0]
        result = interpret_prediction(prediction)
        return {"success": True, "result": result, "error": None}
    except Exception as e:
        return {"success": False, "result": None, "error": str(e)}

MODEL_SPECS = {
    "name": "PneumoDetect AI",
    "version": "v2.0",
    "architecture": "MobileNetV2",
    "accuracy": 86.0,
    "sensitivity": 96.4,
    "specificity": 74.8,
    "avg_prediction_time": "2.5 seconds",
    "developer": "Ayushi Rathour",
    "supported_formats": ["JPG", "JPEG", "PNG"],
    "max_file_size_mb": 200
}

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.set_page_config(page_title="PneumoDetect AI", page_icon="🫁", layout="wide")

# Inject custom CSS for gradient + glassmorphism
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0c1c44, #1e3a8a, #0a1238, #4ade80);
    background-size: 400% 400%;
    animation: gradientAnimation 15s ease infinite;
}
@keyframes gradientAnimation {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.glass-card {
    background: rgba(255, 255, 255, 0.15);
    border-radius: 15px;
    padding: 20px;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    box-shadow: 0 4px 30px rgba(0,0,0,0.1);
    color: white;
}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Hero Section
# ----------------------------
st.markdown("<h1 style='text-align:center; color:white;'>🫁 PneumoDetect AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:white; font-size:20px;'>Fast, Accurate & Reliable Pneumonia Detection in 2.5 Seconds</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Stats Section
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"<div class='glass-card'><h3>🎯 {MODEL_SPECS['accuracy']}%</h3><p>Model Accuracy</p></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='glass-card'><h3>🔍 {MODEL_SPECS['sensitivity']}%</h3><p>Sensitivity Rate</p></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='glass-card'><h3>⏱ {MODEL_SPECS['avg_prediction_time']}</h3><p>Avg Prediction Time</p></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Upload Section
uploaded_file = st.file_uploader("📤 Upload Chest X-Ray for Analysis", type=["jpg", "jpeg", "png"])

# Prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Chest X-Ray", use_column_width=True)

    with st.spinner("🔬 Analyzing X-ray with AI..."):
        time.sleep(1)  # mimic processing delay
        prediction = predict_pneumonia(image)
    
    if prediction["success"]:
        result = prediction["result"]
        st.markdown(f"<div class='glass-card'><h2>Diagnosis: {result['diagnosis']}</h2><p>Confidence: {result['confidence']}% ({result['confidence_level']})</p><p>{result['recommendation']}</p></div>", unsafe_allow_html=True)
    else:
        st.error("Error: " + prediction["error"])

# Footer
st.markdown("""
<hr>
<div style='text-align:center; color:white;'>
PneumoDetect AI v2.0 | © 2025 Ayushi Rathour<br>
<a href='https://github.com/ayushirathour/chest-xray-pneumonia-detection-ai' style='color:#4ade80;'>GitHub</a> |
<a href='https://twitter.com/ayushirathour' style='color:#4ade80;'>Twitter</a> |
<a href='https://linkedin.com/in/ayushirathour' style='color:#4ade80;'>LinkedIn</a> |
<a href='https://instagram.com/ayushirathour' style='color:#4ade80;'>Instagram</a>
</div>
""", unsafe_allow_html=True)
