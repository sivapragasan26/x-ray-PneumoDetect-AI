import streamlit as st
from PIL import Image
import numpy as np
import io
import time
import os
import tensorflow as tf
from PIL import Image as PILImage

# -----------------------------
# MODEL LOGIC (kept intact)
# -----------------------------
@st.cache_resource
def load_pneumonia_model():
    """Load H5 pneumonia detection model with fallback paths (cached)."""
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
        except Exception:
            continue
    return None

def preprocess_image(image_input):
    """
    Preprocess image for pneumonia detection model
    Input: PIL Image or image path
    Output: Preprocessed numpy array ready for prediction
    """
    if isinstance(image_input, str):
        image = PILImage.open(image_input)
    else:
        image = image_input
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize((224, 224))
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def interpret_prediction(prediction_score):
    """
    Interpret model prediction score into medical diagnosis
    Input: Float prediction score (0-1)
    Output: Dictionary with diagnosis details
    """
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
    """
    Complete pneumonia prediction pipeline
    Input: Image (PIL or path) and optional model
    Output: Prediction results dictionary
    """
    try:
        if model is None:
            model = load_pneumonia_model()
            if model is None:
                raise Exception("Could not load pneumonia detection model")
        
        processed_image = preprocess_image(image_input)
        prediction = model.predict(processed_image, verbose=0)[0][0]
        result = interpret_prediction(prediction)
        
        return {
            "success": True,
            "result": result,
            "error": None
        }
        
    except Exception as e:
        return {
            "success": False,
            "result": None,
            "error": str(e)
        }

MODEL_SPECS = {
    "name": "PneumoDetect AI",
    "version": "v2.0",
    "architecture": "MobileNetV2",
    "input_size": (224, 224, 3),
    "threshold": 0.5,
    "accuracy": 86.0,
    "sensitivity": 96.4,
    "specificity": 74.8,
    "validation_samples": 485,
    "avg_prediction_time": "2.5 sec",
    "developer": "Ayushi Rathour",
    "supported_formats": ["JPG", "JPEG", "PNG"],
    "max_file_size_mb": 200
}

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="🫁 PneumoDetect AI", page_icon="🫁", layout="wide", initial_sidebar_state="collapsed")

# Custom gradient background and styling - Python/Streamlit friendly
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        color: #e6eef8;
    }

    /* Full-screen gradient background with animation */
    .stApp {
        background: linear-gradient(135deg, #0c1c44, #1e3a8a, #0a1238, #4ade80);
        background-size: 400% 400%;
        animation: gradientAnimation 15s ease infinite;
        min-height: 100vh;
    }

    @keyframes gradientAnimation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }

    /* Container */
    .app-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 36px 20px;
    }

    /* Header with floating animation */
    .hero {
        text-align: center;
        margin-bottom: 40px;
    }
    .hero-emoji { 
        font-size: 64px; 
        animation: float 3s ease-in-out infinite; 
        display: block;
        margin-bottom: 20px;
    }
    .hero-title { 
        font-size: 48px; 
        font-weight: 800; 
        margin: 16px 0; 
        color: #ffffff; 
        text-shadow: 2px 2px 10px rgba(0,0,0,0.3);
    }
    .hero-sub { 
        color: rgba(230,238,248,0.9); 
        font-size: 20px; 
        margin-bottom: 16px; 
        font-weight: 500;
    }
    .hero-tagline {
        color: rgba(230,238,248,0.85);
        font-size: 18px;
        font-weight: 600;
        font-style: italic;
        margin-top: 12px;
    }

    /* Stats grid with emojis */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 24px;
        margin: 40px 0;
    }
    .stat-card {
        background: rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 30px;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255,255,255,0.12);
        box-shadow: 0 15px 35px rgba(2,6,23,0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        text-align: center;
    }
    .stat-card:hover { 
        transform: translateY(-8px); 
        box-shadow: 0 25px 50px rgba(2,6,23,0.4); 
    }
    .stat-value { 
        font-size: 36px; 
        font-weight: 900; 
        color: #fff; 
        margin: 0;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    .stat-label { 
        color: rgba(230,238,248,0.8); 
        margin-top: 12px; 
        font-size: 16px; 
        text-transform: uppercase; 
        letter-spacing: 1px;
        font-weight: 600;
    }

    /* Upload section styling */
    .upload-section {
        margin: 40px 0;
        text-align: center;
    }
    .upload-title {
        font-size: 24px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 20px;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }

    /* File uploader styling */
    .stFileUploader {
        background: rgba(255,255,255,0.08);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        margin: 20px auto;
        max-width: 600px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
        border: 2px dashed rgba(255,255,255,0.3);
        padding: 20px;
        transition: all 0.3s ease;
    }
    .stFileUploader:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.25);
        border-color: rgba(255,255,255,0.5);
    }

    /* Button styling - responsive and centered */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 16px 32px;
        border-radius: 50px;
        font-weight: 800;
        font-size: 18px;
        box-shadow: 0 10px 25px rgba(102,126,234,0.3);
        transition: all 0.3s ease;
        width: 100%;
        max-width: 300px;
        margin: 0 auto;
        display: block;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(102,126,234,0.5);
        background: linear-gradient(135deg, #5a67d8, #6b46c1);
    }

    /* Social icons styling */
    .social-section {
        text-align: center;
        margin: 50px 0 30px 0;
    }
    .social-icons {
        display: flex;
        justify-content: center;
        gap: 30px;
        margin: 20px 0;
    }
    .social-icon {
        font-size: 32px;
        color: rgba(255,255,255,0.8);
        transition: all 0.3s ease;
        text-decoration: none;
    }
    .social-icon:hover {
        color: #4ade80;
        transform: translateY(-5px);
    }

    /* Disclaimer box */
    .disclaimer-box {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 15px;
        padding: 25px;
        margin: 40px auto;
        max-width: 800px;
        color: rgba(255,255,255,0.9);
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .disclaimer-title {
        font-size: 20px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 15px;
    }

    /* Footer styling - kept simple */
    .footer {
        text-align: center;
        margin-top: 60px;
        padding: 30px;
        color: rgba(230,238,248,0.7);
        font-size: 16px;
        font-weight: 500;
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .stats-grid { grid-template-columns: 1fr; gap: 16px; }
        .hero-title { font-size: 36px; }
        .hero-emoji { font-size: 48px; }
        .social-icons { gap: 20px; }
        .social-icon { font-size: 28px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App content wrapper
st.markdown('<div class="app-container">', unsafe_allow_html=True)

# Header / hero section with emojis
st.markdown(
    """
    <div class="hero">
        <div class="hero-emoji">🫁</div>
        <div class="hero-title">PneumoDetect AI</div>
        <div class="hero-sub">Advanced Chest X-Ray Analysis | Clinical-Grade Artificial Intelligence</div>
        <div class="hero-tagline">⚡ Fast. Accurate. Reliable. AI-powered pneumonia detection in just 2.5 seconds.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Stats grid with emojis
st.markdown(
    f"""
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value">🎯 {int(MODEL_SPECS['accuracy'])}%</div>
            <div class="stat-label">Model Accuracy</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">🔍 {MODEL_SPECS['sensitivity']}%</div>
            <div class="stat-label">Sensitivity Rate</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">⏱ {MODEL_SPECS['avg_prediction_time']}</div>
            <div class="stat-label">Average Prediction Time</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Upload section with emojis
st.markdown(
    """
    <div class="upload-section">
        <div class="upload-title">📤 Upload Chest X-Ray for Instant AI Analysis</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# File uploader - FIXED: use_container_width instead of use_column_width
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="upload")

# If file uploaded: preview & analyze
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
    except Exception:
        st.error("⚠️ Unable to open image. Please upload a valid JPG/PNG file.")
        image = None

    if image is not None:
        # Preview with FIXED parameter
        st.image(image, caption="🖼️ Uploaded Chest X-Ray - Ready for Analysis", use_container_width=True, output_format='PNG')

        # Analyze button - responsive and centered using columns
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze = st.button("🔬 Analyze X-Ray", key="analyze_btn")
            
        if analyze:
            # Spinner while model loads & predicts
            with st.spinner("🧠 Analyzing image, please wait..."):
                t0 = time.time()
                model = load_pneumonia_model()
                prediction_data = predict_pneumonia(image, model)
                elapsed = time.time() - t0

            if not prediction_data["success"]:
                st.error(f"❌ Prediction failed: {prediction_data['error']}")
            else:
                res = prediction_data["result"]

                # Render result with emojis
                if res["diagnosis"] == "PNEUMONIA":
                    st.error(f"""
                    🩺 **Diagnosis Result: Pneumonia Detected**
                    
                    **Confidence:** {res['confidence']}% ({res['confidence_level']} level)
                    
                    **Recommendation:** {res['recommendation']}
                    """)
                else:
                    st.success(f"""
                    ✅ **Diagnosis Result: Normal Chest X-Ray**
                    
                    **Confidence:** {res['confidence']}% ({res['confidence_level']} level)
                    
                    **Recommendation:** {res['recommendation']}
                    """)

                # Technical analysis
                st.info(f"""
                📊 **Technical Analysis:**
                - Model Architecture: {res['model_architecture']}
                - Raw Score: {res['raw_score']:.4f}
                - Threshold: {res['threshold']}
                - ⏱ Prediction Time: {elapsed:.2f} sec
                """)

# Social media icons section
st.markdown(
    """
    <div class="social-section">
        <div class="social-icons">
            <a href="https://github.com/ayushirathour" target="_blank" class="social-icon" title="GitHub">
                🐙
            </a>
            <a href="https://huggingface.co/ayushirathour" target="_blank" class="social-icon" title="Hugging Face">
                🤗
            </a>
            <a href="mailto:ayushirathour1804@gmail.com" class="social-icon" title="Gmail">
                📧
            </a>
            <a href="https://www.linkedin.com/in/ayushirathour" target="_blank" class="social-icon" title="LinkedIn">
                💼
            </a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Disclaimer in box - content unchanged
st.markdown(
    """
    <div class="disclaimer-box">
        <div class="disclaimer-title">⚠️ Medical Disclaimer</div>
        <p>This AI tool is intended for preliminary screening purposes only. Always seek advice from qualified healthcare professionals before making medical decisions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Footer - kept simple as requested, removed developer line
st.markdown(
    f"""
    <div class="footer">
        <strong>{MODEL_SPECS['name']} {MODEL_SPECS['version']}</strong> | © 2025 {MODEL_SPECS['developer']}
    </div>
    """,
    unsafe_allow_html=True,
)

# Close container
st.markdown("</div>", unsafe_allow_html=True)
