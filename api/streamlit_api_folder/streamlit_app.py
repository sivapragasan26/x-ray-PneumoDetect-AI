# streamlit_app.py

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
st.set_page_config(page_title="PneumoDetect AI", page_icon="🫁", layout="wide", initial_sidebar_state="collapsed")

# CSS: full-screen fixed gradient background + glassmorphism + responsiveness
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        color: #e6eef8;
    }

    /* Full-screen background layer so Streamlit elements sit above it */
    .bg-aurora {
        position: fixed;
        inset: 0;
        z-index: -999;
        background: linear-gradient(135deg, #0c1c44, #1e3a8a, #0a1238, #4ade80);
        background-size: 400% 400%;
        animation: gradientAnimation 15s ease infinite;
        filter: blur(20px) saturate(0.9);
        opacity: 0.95;
        pointer-events: none;
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

    /* Header */
    .hero {
        text-align: center;
        margin-bottom: 22px;
    }
    .hero-emoji { font-size: 56px; animation: float 3s ease-in-out infinite; }
    .hero-title { font-size: 34px; font-weight: 700; margin: 8px 0; color: #ffffff; }
    .hero-sub { color: rgba(230,238,248,0.85); font-size: 16px; margin-bottom: 8px; }
    .dev-badge { display:inline-block; padding:8px 14px; border-radius:999px; background: rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.03); color: rgba(230,238,248,0.95); font-weight:600; }

    /* Stats grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 18px;
        margin-top: 20px;
    }
    .stat-card {
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 22px;
        backdrop-filter: blur(6px);
        border:1px solid rgba(255,255,255,0.04);
        box-shadow: 0 10px 30px rgba(2,6,23,0.45);
        transition: transform .18s ease, box-shadow .18s ease;
    }
    .stat-card:hover { transform: translateY(-6px); box-shadow: 0 18px 42px rgba(2,6,23,0.6); }
    .stat-value { font-size: 28px; font-weight: 800; color: #fff; margin:0; }
    .stat-label { color: rgba(230,238,248,0.8); margin-top:8px; font-size:13px; text-transform:uppercase; letter-spacing:1px; }

    /* Upload area: removed drag-drop box requirements. Use simple decorated area. */
    .upload-area {
        margin-top: 28px;
        display:flex;
        justify-content:center;
    }
    .upload-card {
        width:100%;
        max-width:900px;
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.04);
        padding: 22px;
        text-align:center;
        backdrop-filter: blur(5px);
        box-shadow: 0 12px 36px rgba(2,6,23,0.45);
    }
    .upload-title { font-size: 18px; font-weight:700; color:#ffffff; }
    .upload-sub { margin-top:6px; color: rgba(230,238,248,0.8); font-size:14px; }

    /* Preview card */
    .preview-card {
        margin-top: 18px;
        max-width: 1000px;
        border-radius: 12px;
        overflow:hidden;
        border: 1px solid rgba(255,255,255,0.05);
        background: rgba(255,255,255,0.02);
        box-shadow: 0 12px 36px rgba(2,6,23,0.45);
    }
    .preview-caption {
        padding:12px;
        text-align:center;
        color: rgba(230,238,248,0.9);
        background: rgba(255,255,255,0.01);
    }
    .preview-img { width:100%; height:auto; display:block; }

    /* Button */
    .analyze-btn {
        display:block;
        width:100%;
        max-width:420px;
        margin: 18px auto 0;
        padding: 12px 18px;
        border-radius: 10px;
        border: none;
        background: linear-gradient(90deg,#1981ff,#2ac2d8);
        color: white;
        font-weight:700;
        box-shadow: 0 10px 26px rgba(26,115,255,0.16);
        cursor:pointer;
        transition: transform .12s ease, box-shadow .12s ease;
    }
    .analyze-btn:hover { transform: translateY(-3px); box-shadow: 0 18px 36px rgba(26,115,255,0.22); }

    /* Result */
    .result-block { margin-top:18px; }
    .result-panel { border-radius:10px; padding:18px; }
    .result-title { font-weight:800; font-size:20px; margin-bottom:6px; color:#fff; }
    .result-confidence { font-size:26px; font-weight:900; margin:6px 0; color:#fff; }
    .result-reco { color: rgba(230,238,248,0.95); margin-top:6px; }

    .tech-grid { display:grid; grid-template-columns: repeat(3,1fr); gap:12px; margin-top:12px; }
    .tech-card { background: rgba(255,255,255,0.02); padding:12px; border-radius:8px; text-align:center; border:1px solid rgba(255,255,255,0.03); }
    .tech-label { color: rgba(230,238,248,0.8); font-size:13px; }
    .tech-value { color:#fff; font-weight:700; margin-top:6px; }

    /* Footer */
    .footer { margin-top:28px; text-align:center; color: rgba(230,238,248,0.78); font-size:14px; }
    .footer a { color: rgba(172,216,255,0.95); text-decoration:none; }

    /* Responsive */
    @media (max-width: 920px) {
        .stats-grid { grid-template-columns: 1fr; }
        .tech-grid { grid-template-columns: 1fr; }
        .hero-title { font-size: 24px; }
        .hero-emoji { font-size:44px; }
    }
    </style>

    <div class="bg-aurora"></div>
    """,
    unsafe_allow_html=True,
)

# App content wrapper
st.markdown('<div class="app-container">', unsafe_allow_html=True)

# Header / hero
st.markdown(
    """
    <div class="hero">
        <div class="hero-emoji">🫁</div>
        <div class="hero-title">PneumoDetect AI</div>
        <div class="hero-sub">Advanced Chest X-Ray Analysis for Pneumonia Detection</div>
        <div class="dev-badge">Developed by Ayushi Rathour — Biotechnology Graduate</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Stats (three)
st.markdown(
    f"""
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value">{int(MODEL_SPECS['accuracy'])}%</div>
            <div class="stat-label">Model Accuracy</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{MODEL_SPECS['sensitivity']}%</div>
            <div class="stat-label">Sensitivity Rate</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{MODEL_SPECS['avg_prediction_time']}</div>
            <div class="stat-label">Average Prediction Time</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Upload area (no drag-and-drop box)
st.markdown(
    """
    <div class="upload-area">
      <div class="upload-card">
        <div class="upload-title">Upload Chest X-Ray Image</div>
        <div class="upload-sub">Supported formats: JPG, PNG, JPEG • Max {mb}MB</div>
        <div style="height:6px;"></div>
    """.format(mb=MODEL_SPECS["max_file_size_mb"]),
    unsafe_allow_html=True,
)

# Streamlit file uploader (regular) - placed inside the styled card
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="upload")

# Close upload card container (will be reopened when previewing)
st.markdown("</div></div>", unsafe_allow_html=True)

# If file uploaded: preview & analyze
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
    except Exception:
        st.error("Unable to open image. Please upload a valid JPG/PNG file.")
        image = None

    if image is not None:
        # Preview card
        bio = io.BytesIO()
        image.save(bio, format="PNG")
        img_bytes = bio.getvalue()

        st.markdown('<div class="preview-card">', unsafe_allow_html=True)
        st.image(img_bytes, use_column_width=True, output_format='PNG')
        st.markdown('<div class="preview-caption">Uploaded Chest X-Ray</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Analyze button
        analyze = st.button("Analyze X-Ray", key="analyze_btn")
        if analyze:
            # Spinner while model loads & predicts. No model path info on UI.
            with st.spinner("Analyzing — please wait..."):
                t0 = time.time()
                model = load_pneumonia_model()  # cached resource
                prediction_data = predict_pneumonia(image, model)
                elapsed = time.time() - t0

            if not prediction_data["success"]:
                st.error(f"Prediction failed: {prediction_data['error']}")
            else:
                res = prediction_data["result"]

                # Render result card
                if res["diagnosis"] == "PNEUMONIA":
                    # red-accent result
                    st.markdown(
                        f"""
                        <div class="result-block">
                            <div class="result-panel" style="border-left:4px solid rgba(255,90,90,0.95); background: linear-gradient(90deg, rgba(255,255,255,0.015), rgba(255,255,255,0.01));">
                                <div class="result-title" style="color: #ffb6b6;">Pneumonia Detected</div>
                                <div class="result-confidence">{res['confidence']}% ({res['confidence_level']})</div>
                                <div class="result-reco">{res['recommendation']}</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    # green-accent result
                    st.markdown(
                        f"""
                        <div class="result-block">
                            <div class="result-panel" style="border-left:4px solid rgba(106,230,141,0.9); background: linear-gradient(90deg, rgba(255,255,255,0.015), rgba(255,255,255,0.01));">
                                <div class="result-title" style="color: #9ff3b3;">Normal Chest X-Ray</div>
                                <div class="result-confidence">{res['confidence']}% ({res['confidence_level']})</div>
                                <div class="result-reco">{res['recommendation']}</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # technical analysis
                st.markdown(
                    f"""
                    <div class="glass" style="margin-top:12px;">
                        <div style="font-weight:700; color: rgba(230,238,248,0.95);">Technical Analysis</div>
                        <div class="tech-grid">
                            <div class="tech-card"><div class="tech-label">Raw Score</div><div class="tech-value">{res['raw_score']:.4f}</div></div>
                            <div class="tech-card"><div class="tech-label">Threshold</div><div class="tech-value">{res['threshold']}</div></div>
                            <div class="tech-card"><div class="tech-label">Architecture</div><div class="tech-value">{res['model_architecture']}</div></div>
                        </div>
                        <div style="margin-top:10px; color: rgba(230,238,248,0.78); font-size:0.92rem;">
                            ⏱ Prediction Time: {elapsed:.2f} sec
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# If no file, show hint card
else:
    st.markdown(
        """
        <div style="margin-top:18px;">
            <div class="glass" style="padding:16px; text-align:center; max-width:820px; margin-left:auto; margin-right:auto;">
                <div style="font-weight:700; color: rgba(230,238,248,0.95);">Ready to analyze a chest X-ray?</div>
                <div style="color: rgba(230,238,248,0.78); margin-top:6px;">Upload a supported image to start. The model runs on the server and you'll see results here.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Footer with updated GitHub link
st.markdown(
    f"""
    <div class="footer">
        <div style="font-weight:700; color: #fff; margin-bottom:6px;">Medical Disclaimer</div>
        <div>This AI tool is for preliminary screening only. Consult a healthcare professional for medical advice.</div>
        <div style="margin-top:10px;">For model details and source code, visit <a href="https://github.com/ayushirathour/chest-xray-pneumonia-detection-ai" target="_blank">GitHub</a></div>
        <div style="margin-top:10px; font-weight:600;">Developed by {MODEL_SPECS['developer']}</div>
        <div style="margin-top:6px;">© 2025 PneumoDetect AI</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# close container
st.markdown("</div>", unsafe_allow_html=True)
