import streamlit as st
from PIL import Image
import numpy as np
import time
import os
import io
import base64

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
                model = tf.keras.models.load_model(model_path, compile=False)
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                return model
        except Exception as e:
            continue
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
            recommendation = "🚨 Strong indication — seek immediate medical attention."
        elif confidence >= 60:
            confidence_level = "Moderate"
            recommendation = "⚠ Moderate indication — medical review advised."
        else:
            confidence_level = "Low"
            recommendation = "💡 Possible pneumonia — further examination advised."
    else:
        diagnosis = "NORMAL"
        confidence = float((1 - prediction_score) * 100)
        if confidence >= 80:
            confidence_level = "High"
            recommendation = "✅ No signs of pneumonia detected — chest appears normal."
        elif confidence >= 60:
            confidence_level = "Moderate"
            recommendation = "👍 Likely normal — routine follow-up if symptoms persist."
        else:
            confidence_level = "Low"
            recommendation = "🤔 Unclear result — manual review recommended."

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
st.set_page_config(
    page_title="PneumoDetect AI", 
    page_icon="🫁", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# Custom CSS: aurora gradient + glassmorphism + responsive styles
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        color: #e6eef8;
    }}

    /* Background gradient (animated aurora) */
    body {{
        background: linear-gradient(300deg, #211c6a, #17594a, #08045b, #264422, #b7b73d);
        background-size: 400% 400%;
        animation: gradientAnimation 25s ease infinite;
    }}
    @keyframes gradientAnimation {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}

    /* Container spacing */
    .block-container {{
        max-width: 1000px;
        margin-left: auto;
        margin-right: auto;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}

    /* Header */
    .hero {{
        text-align: center;
        margin-bottom: 2rem;
    }}
    .hero-emoji {{
        font-size: 4.5rem;
        margin-bottom: 0.6rem;
    }}
    .hero-title {{
        font-size: 2.2rem;
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
        background: rgba(255,255,255,0.08);
        border-radius: 16px;
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
        gap: 15px;
        margin-top: 1.4rem;
    }}
    .stat-value {{
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
    }}
    .stat-label {{
        font-size: 0.85rem;
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
        background: rgba(255,255,255,0.05);
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
    .preview-container {{
        border-radius: 12px;
        overflow: hidden;
        padding: 0;
        border: 1px solid rgba(255,255,255,0.05);
        margin-top: 20px;
    }}
    .preview-img {{
        width: 100%;
        display: block;
    }}
    .preview-caption {{
        background: rgba(255,255,255,0.03);
        padding: 12px;
        text-align: center;
        color: rgba(226,236,255,0.85);
        font-size: 0.9rem;
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
        margin: 30px auto;
        max-width: 800px;
        padding: 25px;
        border-radius: 16px;
        text-align: center;
    }}
    .result-header {{
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 10px;
    }}
    .result-confidence {{
        font-size: 2.5rem;
        font-weight: 800;
        margin: 15px 0;
        background: rgba(255,255,255,0.1);
        padding: 15px;
        border-radius: 12px;
    }}
    .result-recommend {{
        font-size: 1.1rem;
        color: rgba(226,236,255,0.95);
        margin-top: 15px;
        padding: 15px;
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
    }}

    /* Technical analysis grid */
    .tech-grid {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 15px;
        margin-top: 20px;
    }}
    .tech-card {{
        background: rgba(255,255,255,0.05);
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.03);
    }}
    .tech-label {{ 
        color: rgba(226,236,255,0.75); 
        font-size: 0.85rem; 
        margin-bottom: 8px;
    }}
    .tech-value {{ 
        color: #ffffff; 
        font-weight: 700; 
        font-size: 1.1rem; 
    }}

    /* Footer */
    .footer {{
        margin-top: 3rem;
        text-align: center;
        color: rgba(226,236,255,0.78);
        font-size: 0.95rem;
    }}
    .footer a {{ 
        color: rgba(172,216,255,0.95); 
        text-decoration: none;
        font-weight: 600;
    }}
    .footer-icons {{
        display: flex;
        justify-content: center;
        gap: 20px;
        margin: 15px 0;
    }}
    .footer-icon {{
        font-size: 1.5rem;
        color: #fff;
        transition: transform 0.3s ease;
    }}
    .footer-icon:hover {{
        transform: translateY(-5px);
    }}

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
        <div class="hero-sub">Clinical-Grade Artificial Intelligence</div>
        <div class="developer-badge">Fast. Accurate. Reliable. AI-powered pneumonia detection - externally validated on 400+ scans</div>
    </div>
    """, unsafe_allow_html=True
)

# Stats (3 boxes as requested: Accuracy, Sensitivity, Avg Time)
st.markdown(
    """
    <div class="stats-grid">
        <div class="glass">
            <p class="stat-value">{accuracy}%</p>
            <div class="stat-label">🎯 Accuracy</div>
        </div>
        <div class="glass">
            <p class="stat-value">{sensitivity}%</p>
            <div class="stat-label">🔍 Sensitivity</div>
        </div>
        <div class="glass">
            <p class="stat-value">{avg_time}</p>
            <div class="stat-label">⏱ Avg. Prediction Time</div>
        </div>
    </div>
    """.format(
        accuracy=int(MODEL_SPECS["accuracy"]),
        sensitivity=MODEL_SPECS["sensitivity"],
        avg_time=MODEL_SPECS["avg_prediction_time"]
    ),
    unsafe_allow_html=True
)

# Upload box
st.markdown(
    """
    <div class="glass">
        <h3 class="upload-title">📤 Upload Chest X-Ray for Instant AI Analysis</h3>
        <div class="upload-box">
            <div style="font-size: 2.5rem; margin-bottom: 15px;">🫁</div>
            <div>Drag & drop your file or click to browse</div>
            <div class="upload-sub">Supported formats: JPG, PNG, JPEG | Max {max_mb}MB</div>
        </div>
    </div>
    """.format(max_mb=MODEL_SPECS["max_file_size_mb"]),
    unsafe_allow_html=True
)

# Hide default streamlit uploader label
hide_streamlit_uploader_style = """
    <style>
        #file_uploader { position: relative; z-index: 2; }
        .stFileUploader>div>div { background: transparent; border: none; }
    </style>
"""
st.markdown(hide_streamlit_uploader_style, unsafe_allow_html=True)

# Actual Streamlit uploader
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="file_uploader", accept_multiple_files=False)

# If file uploaded: show preview and analyze button
if uploaded_file is not None:
    # Load as PIL image
    try:
        image = Image.open(uploaded_file)
        # Show preview in glass container
        st.markdown('<div class="glass preview-container">', unsafe_allow_html=True)
        
        # Convert image to bytes for display
        bio = io.BytesIO()
        image.save(bio, format="PNG")
        b64 = base64.b64encode(bio.getvalue()).decode()
        
        # Display image with container width
        st.markdown(
            f'<img src="data:image/png;base64,{b64}" class="preview-img">',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="preview-caption">📸 Uploaded Chest X-Ray - Ready for AI Analysis</div>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # Analyze button
        analyze_clicked = st.button("🔬 Analyze X-ray with AI", key="analyze", 
                                   use_container_width=True, 
                                   type="primary", 
                                   help="Run AI analysis on the uploaded X-ray")

        if analyze_clicked:
            # Show loading spinner without text
            with st.spinner(""):
                # Add custom loading animation
                st.markdown(
                    """
                    <div style="text-align:center; margin:30px 0;">
                        <div class="loading-spinner"></div>
                        <style>
                        .loading-spinner {
                            width: 50px;
                            height: 50px;
                            border: 5px solid rgba(255,255,255,0.3);
                            border-radius: 50%;
                            border-top-color: #fff;
                            animation: spin 1s linear infinite;
                            margin: 0 auto;
                        }
                        @keyframes spin {
                            to { transform: rotate(360deg); }
                        }
                        </style>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Run prediction
                start_time = time.time()
                model = load_pneumonia_model()
                prediction_data = predict_pneumonia(image, model)
                elapsed = time.time() - start_time

            if not prediction_data["success"]:
                st.error(f"Prediction failed: {prediction_data['error']}")
            else:
                res = prediction_data["result"]
                
                # Determine result styling
                if res["diagnosis"] == "PNEUMONIA":
                    result_style = "background: linear-gradient(135deg, #ff6b6b, #ff5252); color: white;"
                    icon = "🩺"
                else:
                    result_style = "background: linear-gradient(135deg, #51cf66, #40c057); color: white;"
                    icon = "✅"
                
                # Result panel
                st.markdown(
                    f"""
                    <div class="result-panel glass" style="{result_style}">
                        <div class="result-header">
                            {icon} Diagnosis Result: {"Pneumonia Detected" if res["diagnosis"] == "PNEUMONIA" else "Normal Chest X-Ray"}
                        </div>
                        <div class="result-confidence">
                            {res['confidence']}% confidence
                        </div>
                        <div class="result-recommend">
                            {res['recommendation']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Technical analysis
                st.markdown(
                    f"""
                    <div class="glass" style="margin: 0 auto; max-width: 800px;">
                        <div class="analysis-title" style="text-align:center; font-weight:700; margin-bottom:20px;">
                            🔬 Technical Analysis Dashboard
                        </div>
                        <div class="tech-grid">
                            <div class="tech-card">
                                <div class="tech-label">Model Architecture</div>
                                <div class="tech-value">{res['model_architecture']}</div>
                            </div>
                            <div class="tech-card">
                                <div class="tech-label">Threshold</div>
                                <div class="tech-value">{res['threshold']}</div>
                            </div>
                            <div class="tech-card">
                                <div class="tech-label">Raw Score</div>
                                <div class="tech-value">{res['raw_score']:.4f}</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

# If no file uploaded, show helpful hint
else:
    st.markdown(
        """
        <div class="glass" style="padding:20px; text-align:center; margin-top:20px;">
            <div style="font-weight:600; color: rgba(226,236,255,0.95);">Ready to analyze a chest X-ray?</div>
            <div style="color: rgba(226,236,255,0.78); margin-top:10px;">
                Click the upload box above or drag & drop a supported image
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Footer / Disclaimer
st.markdown(
    """
    <div class="footer">
        <div class="glass" style="padding:20px; text-align:center;">
            <div style="font-weight:700; color: #ffffff; margin-bottom:15px;">⚠ Medical Disclaimer</div>
            <div style="margin-bottom:15px;">
                This AI tool is intended for preliminary screening purposes only.<br>
                Always seek advice from qualified healthcare professionals before making medical decisions.
            </div>
            
            <div style="margin:20px 0;">
                <div style="font-weight:600; margin-bottom:10px;">👩‍💻 Trained, developed and deployed by Ayushi Rathour</div>
                <div>For model info visit:</div>
                <div class="footer-icons">
                    <a href="https://github.com/ayushirathour" target="_blank" class="footer-icon">
                        <i class="fab fa-github"></i>
                    </a>
                    <a href="https://huggingface.co/ayushirathour" target="_blank" class="footer-icon">
                        <i class="fas fa-robot"></i>
                    </a>
                </div>
            </div>
            
            <div style="font-weight:600; margin-top:20px;">
                PneumoDetect AI v2.0 | © 2025 Ayushi Rathour
            </div>
        </div>
    </div>
    
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    """,
    unsafe_allow_html=True
)

# Close container
st.markdown('</div>', unsafe_allow_html=True)
