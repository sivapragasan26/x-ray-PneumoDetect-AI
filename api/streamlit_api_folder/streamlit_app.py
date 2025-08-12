# streamlit_app.py

import streamlit as st
from PIL import Image
import numpy as np
import time
import os
import io

# ---------------------------
# MODEL LOGIC (exact & integrated)
# ---------------------------
import tensorflow as tf
from PIL import Image as PILImage

# MODEL LOADING WITH MULTIPLE PATHS
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
                # Use print (server-side) for debugging only; DO NOT use st.write here to avoid UI messages.
                print(f"Loading model from: {model_path}")
                model = tf.keras.models.load_model(model_path, compile=False)
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                print("Model loaded successfully.")
                return model
        except Exception as e:
            print(f"Failed to load from {model_path}: {e}")
            continue
    print("Could not load model from any path.")
    return None

# IMAGE PREPROCESSING
def preprocess_image(image_input):
    """
    Preprocess image for pneumonia detection model.
    Accepts PIL Image or path string.
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

# PREDICTION INTERPRETATION
def interpret_prediction(prediction_score):
    """
    Interpret model prediction score into diagnosis dictionary.
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

# MAIN PREDICTION FUNCTION
def predict_pneumonia(image_input, model=None):
    """
    Full pipeline. Returns dict {success, result, error}.
    """
    try:
        if model is None:
            model = load_pneumonia_model()
            if model is None:
                raise Exception("Could not load pneumonia detection model")
        processed = preprocess_image(image_input)
        prediction = model.predict(processed, verbose=0)[0][0]
        result = interpret_prediction(prediction)
        return {"success": True, "result": result, "error": None}
    except Exception as e:
        return {"success": False, "result": None, "error": str(e)}

# MODEL SPECS (display)
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

# ---------------------------
# STREAMLIT UI (glassmorphic + animated gradient)
# ---------------------------
st.set_page_config(page_title="PneumoDetect AI", page_icon="🫁", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS: aurora gradient + glassmorphism + responsive styles
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Poppins', sans-serif;
        color: #e6eef8;
    }}

    /* Background gradient (animated aurora) */
    body {{
        background: linear-gradient(135deg, #0c1c44, #1e3a8a, #0a1238, #4ade80);
        background-size: 400% 400%;
        animation: gradientAnimation 15s ease infinite;
    }}
    @keyframes gradientAnimation {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    @keyframes float {{
        0%, 100% {{ transform: translateY(0); }}
        50% {{ transform: translateY(-10px); }}
    }}

    /* Container spacing */
    .block-container {{
        max-width: 1200px;
        margin-left: auto;
        margin-right: auto;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }}

    /* Header */
    .hero {{
        text-align: center;
        margin-bottom: 2.5rem;
    }}
    .hero-emoji {{
        font-size: 4.5rem;
        animation: float 3s ease-in-out infinite;
        margin-bottom: 0.6rem;
    }}
    .hero-title {{
        font-size: 2.6rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0.2rem 0;
    }}
    .hero-sub {{
        color: rgba(226,236,255,0.85);
        margin-top: 0.5rem;
        font-size: 1.05rem;
    }}
    .developer-badge {{
        display:inline-block;
        margin-top: 0.8rem;
        background: rgba(255,255,255,0.03);
        padding: 10px 18px;
        border-radius: 999px;
        color: rgba(226,236,255,0.9);
        font-weight: 600;
        border: 1px solid rgba(255,255,255,0.04);
    }}

    /* Glass card */
    .glass {{
        background: rgba(255,255,255,0.04);
        border-radius: 14px;
        padding: 20px;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: 0 8px 30px rgba(2,6,23,0.45);
        transition: transform .25s ease, box-shadow .25s ease;
    }}
    .glass:hover {{
        transform: translateY(-6px);
        box-shadow: 0 14px 40px rgba(2,6,23,0.55);
    }}

    /* Stats grid */
    .stats-grid {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 18px;
        margin-top: 1.4rem;
    }}
    .stat-value {{
        font-size: 2.25rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
    }}
    .stat-label {{
        font-size: 0.9rem;
        color: rgba(226,236,255,0.75);
        margin-top: 6px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}

    /* Upload box */
    .upload-box {{
        margin-top: 2rem;
        border: 2px dashed rgba(255,255,255,0.12);
        border-radius: 14px;
        padding: 28px;
        text-align: center;
        cursor: pointer;
        transition: border-color .2s ease, box-shadow .2s ease, transform .2s ease;
        background: linear-gradient(180deg, rgba(255,255,255,0.015), rgba(255,255,255,0.01));
    }}
    .upload-box:hover {{
        border-color: rgba(0,184,212,0.45);
        box-shadow: 0 12px 40px rgba(0,184,212,0.06);
        transform: translateY(-4px);
    }}
    .upload-title {{
        font-size: 1.15rem;
        font-weight: 600;
        color: #ffffff;
    }}
    .upload-sub {{
        color: rgba(226,236,255,0.75);
        margin-top: 6px;
        font-size: 0.95rem;
    }}

    /* Preview */
    .preview-card {{
        margin-top: 1.2rem;
        border-radius: 12px;
        overflow: hidden;
        padding: 0;
        border: 1px solid rgba(255,255,255,0.05);
    }}
    .preview-caption {{
        background: rgba(255,255,255,0.03);
        padding: 12px;
        text-align: center;
        color: rgba(226,236,255,0.85);
    }}
    img.preview-img {{
        width: 100%;
        display:block;
    }}

    /* Analyze Button */
    .analyze-btn {{
        margin-top: 1.2rem;
        width: 100%;
        padding: 14px;
        border-radius: 10px;
        background: linear-gradient(90deg, #1981ff, #2ac2d8);
        color: white;
        font-weight: 700;
        border: none;
        box-shadow: 0 10px 30px rgba(26,115,255,0.18);
        cursor: pointer;
        transition: transform .15s ease, box-shadow .15s ease, opacity .15s;
    }}
    .analyze-btn:hover {{ transform: translateY(-3px); box-shadow: 0 16px 36px rgba(26,115,255,0.26); }}

    /* Result cards */
    .result-panel {{
        margin-top: 1.6rem;
        padding: 22px;
        border-radius: 12px;
    }}
    .result-header {{
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 6px;
    }}
    .result-confidence {{
        font-size: 2.2rem;
        font-weight: 800;
        margin: 8px 0;
    }}
    .result-recommend {{
        font-size: 1rem;
        color: rgba(226,236,255,0.95);
    }}

    /* Technical analysis grid */
    .tech-grid {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin-top: 12px;
    }}
    .tech-card {{
        background: rgba(255,255,255,0.02);
        padding: 14px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.03);
    }}
    .tech-label {{ color: rgba(226,236,255,0.75); font-size: 0.85rem; }}
    .tech-value {{ color: #ffffff; font-weight: 700; font-size: 1.05rem; margin-top: 6px; }}

    /* Footer */
    .footer {{
        margin-top: 2.5rem;
        text-align: center;
        color: rgba(226,236,255,0.78);
        font-size: 0.95rem;
    }}
    .footer a {{ color: rgba(172,216,255,0.95); text-decoration: none; }}

    /* Responsive */
    @media (max-width: 900px) {{
        .stats-grid {{ grid-template-columns: 1fr; gap: 12px; }}
        .tech-grid {{ grid-template-columns: 1fr; }}
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Page container
st.markdown('<div class="block-container">', unsafe_allow_html=True)

# Header / Hero
st.markdown(
    """
    <div class="hero">
        <div class="hero-emoji">🫁</div>
        <h1 class="hero-title">PneumoDetect AI</h1>
        <div class="hero-sub">Advanced Chest X-Ray Analysis for Pneumonia Detection</div>
        <div class="developer-badge">Developed by Ayushi Rathour — Biotechnology Graduate</div>
    </div>
    """, unsafe_allow_html=True
)

# Stats (3 boxes as requested: Accuracy, Sensitivity, Avg Time)
st.markdown(
    """
    <div class="stats-grid">
        <div class="glass">
            <p class="stat-value">{accuracy}%</p>
            <div class="stat-label">Model Accuracy</div>
        </div>
        <div class="glass">
            <p class="stat-value">{sensitivity}%</p>
            <div class="stat-label">Sensitivity Rate</div>
        </div>
        <div class="glass">
            <p class="stat-value">{avg_time}</p>
            <div class="stat-label">Average Prediction Time</div>
        </div>
    </div>
    """.format(
        accuracy=int(MODEL_SPECS["accuracy"]),
        sensitivity=MODEL_SPECS["sensitivity"],
        avg_time=MODEL_SPECS["avg_prediction_time"]
    ),
    unsafe_allow_html=True
)

# Upload box (styled). We'll still use streamlit's file_uploader but with decoration.
st.markdown(
    """
    <div style="margin-top:20px;"></div>
    <label for="file_uploader" class="upload-box glass" tabindex="0">
        <div class="upload-title">Upload Chest X-Ray Image</div>
        <div class="upload-sub">Supported: JPG, PNG, JPEG • Max {max_mb}MB</div>
    </label>
    """.format(max_mb=MODEL_SPECS["max_file_size_mb"]),
    unsafe_allow_html=True
)

# Hide default streamlit uploader label (we still use its widget for functionality)
hide_streamlit_uploader_style = """
    <style>
        /* Move the uploader off-screen but still usable */
        #file_uploader { position: relative; z-index: 2; }
        /* Hide extraneous default text that Streamlit sometimes shows */
        .stFileUploader>div>div { background: transparent; border: none; }
    </style>
"""
st.markdown(hide_streamlit_uploader_style, unsafe_allow_html=True)

# Actual Streamlit uploader (invisible label above triggers it visually)
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="file_uploader", accept_multiple_files=False)

# If file uploaded: show preview and analyze button
if uploaded_file is not None:
    # Load as PIL image and show preview in glass card
    try:
        image = Image.open(uploaded_file)
    except Exception:
        st.error("Unable to open image. Make sure file is valid.")
        image = None

    if image:
        # Preview card
        st.markdown('<div class="glass preview-card">', unsafe_allow_html=True)
        # Convert image to bytes for inlined display sizing if needed
        bio = io.BytesIO()
        image.save(bio, format="PNG")
        b64 = bio.getvalue()
        # Show image
        st.image(b64, use_column_width=True, caption=None, output_format="PNG")
        st.markdown('<div class="preview-caption">Uploaded Chest X-Ray</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Analyze button
        analyze_clicked = st.button("Analyze X-Ray", key="analyze", help="Run AI analysis on the uploaded X-ray")

        if analyze_clicked:
            # Spinner while model loads & prediction runs.
            with st.spinner("Analyzing X-ray — this may take a few seconds..."):
                start_time = time.time()
                model = load_pneumonia_model()  # cached; doesn't print to UI
                prediction_data = predict_pneumonia(image, model)
                elapsed = time.time() - start_time

            if not prediction_data["success"]:
                st.error(f"Prediction failed: {prediction_data['error']}")
            else:
                res = prediction_data["result"]

                # Result panel (glass)
                if res["diagnosis"] == "PNEUMONIA":
                    # Red accent
                    st.markdown(
                        f"""
                        <div class="glass result-panel" style="border-left:4px solid rgba(255,90,90,0.95);">
                            <div class="result-header" style="color: #ff8a8a;">Pneumonia Detected</div>
                            <div class="result-confidence" style="color: #fff;">{res['confidence']}% Confidence ({res['confidence_level']})</div>
                            <div class="result-recommend"> {res['recommendation']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    # Green accent
                    st.markdown(
                        f"""
                        <div class="glass result-panel" style="border-left:4px solid rgba(106,230,141,0.9);">
                            <div class="result-header" style="color: #6ae68d;">Normal Chest X-Ray</div>
                            <div class="result-confidence" style="color: #fff;">{res['confidence']}% Confidence ({res['confidence_level']})</div>
                            <div class="result-recommend"> {res['recommendation']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Technical analysis
                st.markdown(
                    f"""
                    <div class="glass" style="margin-top:12px;">
                        <div style="font-weight:700; color: rgba(226,236,255,0.95); margin-bottom:8px;">Technical Analysis</div>
                        <div class="tech-grid">
                            <div class="tech-card"><div class="tech-label">Raw Score</div><div class="tech-value">{res['raw_score']:.4f}</div></div>
                            <div class="tech-card"><div class="tech-label">Threshold</div><div class="tech-value">{res['threshold']}</div></div>
                            <div class="tech-card"><div class="tech-label">Architecture</div><div class="tech-value">{res['model_architecture']}</div></div>
                        </div>
                        <div style="margin-top:10px; color: rgba(226,236,255,0.75); font-size:0.9rem;">
                            ⏱ Prediction Time: {elapsed:.2f} sec
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# If no file uploaded, show helpful hint
else:
    st.markdown(
        """
        <div style="margin-top:12px;">
            <div class="glass" style="padding:18px; text-align:center;">
                <div style="font-weight:700; color: rgba(226,236,255,0.95);">Ready to analyze a chest X-ray?</div>
                <div style="color: rgba(226,236,255,0.78); margin-top:6px;">Click the upload box above or drag & drop a supported image.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Footer / Disclaimer
st.markdown(
    """
    <div class="footer">
        <div style="margin-top:18px; font-weight:700; color: #ffffff;">Medical Disclaimer</div>
        <div style="margin-top:8px;">This AI tool is for preliminary screening only. Consult a healthcare professional for definitive medical advice.</div>
        <div style="margin-top:10px;">For model details and source code, visit <a href="https://github.com/ayushirathour" target="_blank">GitHub</a> • <a href="https://huggingface.co/ayushirathour" target="_blank">Hugging Face</a></div>
        <div style="margin-top:12px; font-weight:600;">Developed by {dev}</div>
        <div style="margin-top:4px;">© 2025 PneumoDetect AI</div>
    </div>
    """.format(dev=MODEL_SPECS["developer"]),
    unsafe_allow_html=True
)

# Close container
st.markdown('</div>', unsafe_allow_html=True)
