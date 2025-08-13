import streamlit as st
from PIL import Image
import numpy as np
import io
import time
import os
import tensorflow as tf
from PIL import Image as PILImage
import base64
from fpdf import FPDF
from datetime import datetime


# -----------------------------
# MODEL LOGIC (kept intact - unchanged)
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
    "total_scans": "800+",  # ADDED: New stat
    "developer": "Ayushi Rathour",
    "supported_formats": ["JPG", "JPEG", "PNG"],
    "max_file_size_mb": 200
}
# -----------------------------
# PDF GENERATION FUNCTIONS (NEW - ADD THIS SECTION)
# -----------------------------
def generate_medical_pdf_report(prediction_result, analysis_time):
    """
    Generate professional medical PDF report (Unicode-safe + Version-safe)
    """
    
    def clean_text_for_pdf(text):
        """
        Clean text to remove Unicode characters that cause PDF errors
        """
        if not text:
            return ""
        
        # Replace problematic Unicode characters
        replacements = {
            # Smart quotes to regular quotes
            '"': '"', '"': '"', ''': "'", ''': "'",
            # Em/en dashes to regular dashes
            '—': '-', '–': '-', '−': '-',
            # Remove emojis and special symbols
            '✅': '[✓]', '❌': '[✗]', '🚨': '[!]', 
            '⚠️': '[!]', '💡': '[i]', '🔬': '',
            '📊': '', '🩺': '', '👍': '', '🤔': '',
            # Other problematic characters
            '…': '...', '•': '-', '→': '->', 
            '°': 'deg', '±': '+/-'
        }
        
        cleaned_text = text
        for unicode_char, replacement in replacements.items():
            cleaned_text = cleaned_text.replace(unicode_char, replacement)
        
        # Remove any remaining non-ASCII characters
        cleaned_text = cleaned_text.encode('ascii', 'ignore').decode('ascii')
        
        return cleaned_text
    
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'PneumoDetect AI - Medical Analysis Report', 0, 1, 'C')
    pdf.ln(10)
    
    # Report Details
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Report Information:', 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 8, f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1)
    pdf.cell(0, 8, f'Model: {MODEL_SPECS["name"]} {MODEL_SPECS["version"]}', 0, 1)
    pdf.cell(0, 8, f'Architecture: {MODEL_SPECS["architecture"]}', 0, 1)
    pdf.cell(0, 8, f'Analysis Time: {analysis_time:.2f} seconds', 0, 1)
    pdf.ln(5)
    
    # Analysis Results
    result = prediction_result['result']
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Analysis Results:', 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 8, f'Diagnosis: {clean_text_for_pdf(result["diagnosis"])}', 0, 1)
    pdf.cell(0, 8, f'Confidence: {result["confidence"]}%', 0, 1)
    pdf.cell(0, 8, f'Confidence Level: {clean_text_for_pdf(result["confidence_level"])}', 0, 1)
    pdf.ln(5)
    
    # Recommendation - FIXED (Unicode-safe)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Recommendation:', 0, 1)
    pdf.set_font('Arial', '', 10)
    
    # Clean the recommendation text before adding to PDF
    clean_recommendation = clean_text_for_pdf(result["recommendation"])
    pdf.multi_cell(0, 8, clean_recommendation)
    pdf.ln(5)
    
    # Technical Details
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Technical Details:', 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 8, f'Raw Score: {result["raw_score"]:.4f}', 0, 1)
    pdf.cell(0, 8, f'Decision Threshold: {result["threshold"]}', 0, 1)
    pdf.cell(0, 8, f'Model Accuracy: {MODEL_SPECS["accuracy"]}%', 0, 1)
    pdf.cell(0, 8, f'Model Sensitivity: {MODEL_SPECS["sensitivity"]}%', 0, 1)
    pdf.ln(10)
    
    # Medical Disclaimer
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 8, 'MEDICAL DISCLAIMER:', 0, 1)
    pdf.set_font('Arial', '', 8)
    
    disclaimer_text = 'This AI analysis is for preliminary screening purposes only. Always seek advice from qualified healthcare professionals before making medical decisions. This tool is not approved for clinical diagnosis.'
    clean_disclaimer = clean_text_for_pdf(disclaimer_text)
    pdf.multi_cell(0, 6, clean_disclaimer)
    
    # FIXED: Version-safe return statement
    pdf_output = pdf.output(dest='S')
    if isinstance(pdf_output, str):
        return pdf_output.encode('latin-1')
    else:
        return pdf_output  # Already bytes
        
def create_pdf_download_link(pdf_data, filename):
    """
    Create download link for PDF file
    """
    b64 = base64.b64encode(pdf_data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}" style="color: #74b9ff; text-decoration: none; font-weight: bold;">📄 Download Medical Report (PDF)</a>'
    return href




# -----------------------------
# STREAMLIT UI (same design, updated content)
# -----------------------------
# Complete CSS styling with FIXED HEADER REPOSITIONED
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        color: #e6eef8;
    }

   .stApp {
    background: linear-gradient(300deg, #0c0634, #17082f, #0a021f, #030108, #120a27, #271653, #0faba9);
    background-size: 420% 420%;
    animation: gradientAnimation 10s ease infinite;
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

    /* FIXED GLASSMORPHIC HEADER - REPOSITIONED BELOW STREAMLIT HEADER */
    /* BLACK GLASSMORPHIC HEADER - NO BORDER + SANS-SERIF */
.fixed-header {
    position: fixed;
    top: 48px;
    left: 0;
    right: 0;
    height: 70px;
    background: rgba(0, 0, 0, 0.25);
    backdrop-filter: blur(30px);
    -webkit-backdrop-filter: blur(30px);
    border: none;
    z-index: 999;
    display: flex;
    align-items: center;
    padding: 0 40px;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
}

.header-brand {
    display: flex;
    align-items: center;
    gap: 0px;
    color: #ffffff;
    text-decoration: none !important;
    font-weight: 300;
    font-size: 20px;
    font-family: sans-serif;
    letter-spacing: 0.5px;
    text-shadow: 0 1px 3px rgba(0, 0, 0, 0.5);
    cursor: default;
    border: none !important;
    outline: none !important;
    pointer-events: none;
}

.header-brand:hover, 
.header-brand:focus, 
.header-brand:active, 
.header-brand:visited {
    color: #ffffff !important;
    text-decoration: none !important;
    outline: none !important;
    transform: none !important;
    text-shadow: 0 1px 3px rgba(0, 0, 0, 0.5);
}

.header-text {
    font-family: sans-serif;
    letter-spacing: 0.5px;
    font-weight: 300;
    text-decoration: none !important;
    font-size: 20px;
    color: #ffffff;
}

/* Remove any hover effects completely */
.fixed-header:hover {
    background: rgba(0, 0, 0, 0.25);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    transform: none;
}


    .header-brand {
        display: flex;
        align-items: center;
        gap: 0px; /* REMOVED: No gap needed without emoji */
        color: white;
        text-decoration: none !important; /* ENHANCED: Force no underline */
        font-weight: 400; /* CHANGED: From 800 to 400 (normal weight) */
        font-size: 28px;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.3); /* REDUCED: Lighter shadow */
        transition: all 0.3s ease;
        border: none !important; /* ADDED: Remove any borders */
        outline: none !important; /* ADDED: Remove outline */
    }

    .header-brand:hover {
        transform: scale(1.03); /* REDUCED: Subtle scale */
        color: #74b9ff;
        text-shadow: 0 3px 12px rgba(116, 185, 255, 0.3);
        text-decoration: none !important; /* ENSURE: No underline on hover */
    }

    .header-brand:focus, .header-brand:active, .header-brand:visited {
        text-decoration: none !important; /* ADDED: Remove underline for all states */
        outline: none !important;
    }

    .header-emoji {
        display: none; /* HIDDEN: Lung icon removed */
    }

    .header-text {
        font-family: 'Poppins', sans-serif;
        letter-spacing: -0.3px; /* REDUCED: Less tight spacing */
        font-weight: 400; /* CHANGED: Light weight */
        text-decoration: none !important; /* ADDED: Ensure no underline */
    }

    .app-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 150px 20px 36px 20px;
    }

    /* REST OF YOUR EXISTING CSS REMAINS THE SAME... */
    .hero {
        text-align: center;
        margin-bottom: 50px;
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
    .hero-tagline { 
        color: rgba(230,238,248,0.95); 
        font-size: 24px; 
        margin-bottom: 12px; 
        font-weight: 600;
    }
    .hero-subline {
        color: rgba(230,238,248,0.85);
        font-size: 18px;
        font-weight: 500;
        margin-bottom: 30px;
        letter-spacing: 2px;
    }

    .value-prop {
        text-align: center;
        margin: 50px 0;
    }
    .value-prop-title {
        color: #ffffff;
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 30px;
    }

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

    .upload-section {
        margin: 50px 0;
        text-align: center;
    }
    .upload-title {
        font-size: 24px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 10px;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    .upload-subtitle {
        color: rgba(230,238,248,0.8);
        font-size: 16px;
        margin-bottom: 30px;
    }

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
    
 .stButton > button {
    background: linear-gradient(135deg, #4c1d95, #3730a3, #6366f1);
    color: white;
    border: none;
    padding: 16px 32px;
    border-radius: 12px; 
    font-weight: 800;
    font-size: 18px;
    box-shadow: 0 10px 25px rgba(76, 29, 149, 0.4); 
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
    box-shadow: 0 15px 35px rgba(76, 29, 149, 0.6); 
    background: linear-gradient(135deg, #5b21b6, #4338ca, #7c3aed);
}




    .tech-section {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 40px;
        margin: 60px 0;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .tech-title {
        color: #ffffff;
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 20px;
        text-align: center;
    }
    .tech-description {
        color: rgba(230,238,248,0.9);
        font-size: 16px;
        line-height: 1.6;
        text-align: center;
        margin-bottom: 30px;
    }

    .about-developer {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 40px;
        margin: 50px 0;
        border: 1px solid rgba(255,255,255,0.1);
        text-align: center;
    }
    .developer-title {
        color: #ffffff;
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 15px;
    }
    .developer-bio {
        color: rgba(230,238,248,0.9);
        font-size: 16px;
        line-height: 1.6;
        max-width: 600px;
        margin: 0 auto;
    }

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

    .social-section {
        text-align: center;
        margin: 60px 0 30px 0;
        padding-top: 40px;
        border-top: 1px solid rgba(255,255,255,0.1);
    }
    .developed-by {
        color: rgba(230,238,248,0.7);
        font-size: 14px;
        font-weight: 500;
        margin-bottom: 20px;
        text-transform: lowercase;
    }
    .social-icons {
        display: flex;
        justify-content: center;
        gap: 30px;
        margin: 20px 0;
        flex-wrap: wrap;
    }
    .social-icon {
        width: 40px;
        height: 40px;
        transition: all 0.3s ease;
        filter: brightness(0) invert(1);
        opacity: 0.8;
    }
    .social-icon:hover {
        transform: translateY(-5px) scale(1.1);
        opacity: 1;
        filter: brightness(0) invert(1) drop-shadow(0 0 10px rgba(255,255,255,0.5));
    }

    .footer {
        text-align: center;
        margin-top: 40px;
        padding: 30px;
        color: rgba(230,238,248,0.6);
        font-size: 14px;
        font-weight: 400;
    }
    .footer-links {
        margin: 15px 0;
    }
    .footer-links a {
        color: rgba(172,216,255,0.9);
        text-decoration: none;
        margin: 0 15px;
        transition: color 0.3s ease;
    }
    .footer-links a:hover {
        color: #ffffff;
    }

    /* RESPONSIVE DESIGN FOR REPOSITIONED HEADER */
 /* RESPONSIVE DESIGN FOR BLACK GLASSMORPHIC HEADER */
@media (max-width: 768px) {
    .fixed-header {
        top: 40px;
        padding: 0 20px;
        height: 60px;
        background: rgba(0, 0, 0, 0.3);
        border: none;
    }
    .header-brand {
        font-size: 18px;
        gap: 0px;
        font-weight: 300;
        letter-spacing: 0.3px;
        font-family: sans-serif;
        color: #ffffff;
    }
    .header-text {
        font-size: 18px;
        font-weight: 300;
        letter-spacing: 0.3px;
        font-family: sans-serif;
        color: #ffffff;
    }
    .header-emoji {
        display: none;
    }
    .app-container {
        padding-top: 110px;
    }
    .stats-grid { grid-template-columns: 1fr; gap: 16px; }
    .hero-title { font-size: 36px; }
    .hero-emoji { font-size: 48px; }
    .social-icons { gap: 20px; }
    .social-icon { width: 35px; height: 35px; }
    .tech-section, .about-developer { padding: 25px; }
}


    </style>
    """,
    unsafe_allow_html=True,
)


# Fixed Glassmorphic Header - UPDATED (Remove emoji)
st.markdown(
    """
    <div class="fixed-header">
        <a href="#" class="header-brand">
            <span class="header-text">PneumoDetect AI</span>
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)

# App content wrapper
st.markdown('<div class="app-container">', unsafe_allow_html=True)

# 1. Hero / Intro Section
st.markdown(
    """
    <div class="hero">
        <div class="hero-emoji">🫁</div>
        <div class="hero-title">PneumoDetect AI</div>
        <div class="hero-tagline">Clinical-Grade AI for Rapid Pneumonia Detection</div>
        <div class="hero-subline">Fast • Accurate • Trusted</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 2. Key Value Proposition
st.markdown(
    """
    <div class="value-prop">
        <div class="value-prop-title">Designed for doctors, researchers, and patients. Instant AI-powered insights.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 3. Stats grid
st.markdown(
    f"""
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value">🎯 {int(MODEL_SPECS['accuracy'])}%</div>
            <div class="stat-label">Accuracy</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">🔍 {MODEL_SPECS['sensitivity']}%</div>
            <div class="stat-label">Detection Rate</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">⏱ {MODEL_SPECS['avg_prediction_time']}</div>
            <div class="stat-label">Avg Analysis Time</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">📊 {MODEL_SPECS['total_scans']}</div>
            <div class="stat-label">Scans Complete</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# 4. File Upload & Analysis Panel
st.markdown(
    """
    <div class="upload-section">
        <div class="upload-title">📤 Upload & Analyze Chest X-Ray</div>
        <div class="upload-subtitle">Secure & Private Processing</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# File uploader
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="upload")

# Simple upload and preview
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="🖼️ Uploaded Chest X-Ray - Ready for Analysis", use_container_width=True)
        
        # Analyze button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            analyze = st.button("🔬 Analyze X-Ray", key="analyze_btn")
        
        # Analysis processing
        if analyze:
            with st.spinner("🧠 AI Analysis in Progress..."):
                t0 = time.time()
                model = load_pneumonia_model()
                prediction_data = predict_pneumonia(image, model)
                elapsed = time.time() - t0
                
                st.session_state["prediction_results"] = prediction_data
                st.session_state["analysis_time"] = elapsed
                st.session_state["analyzed_image"] = image
                
    except Exception:
        st.error("⚠️ Unable to open image. Please upload a valid JPG/PNG file.")

# Results display section
if "prediction_results" in st.session_state and st.session_state["prediction_results"] is not None:
    prediction_data = st.session_state["prediction_results"]
    elapsed = st.session_state["analysis_time"]
    
    if not prediction_data["success"]:
        st.error(f"❌ Analysis failed: {prediction_data['error']}")
    else:
        res = prediction_data["result"]

        # Use native Streamlit container with border styling
        with st.container(border=True):
            
            # WORKING PROGRESS BAR VERSION
            if res["diagnosis"] == "PNEUMONIA":
                st.markdown(f"""
                <div style="background: rgba(255,0,0,0.1); border: 1px solid rgba(255,0,0,0.3); border-radius: 12px; padding: 20px; margin-bottom: 20px;">
                    <h3 style="color: #d32f2f; margin-bottom: 10px;">🩺 DIAGNOSIS: PNEUMONIA DETECTED</h3>
                    <p style="color: #ffffff; margin-bottom: 8px;"><strong>Confidence:</strong> {res['confidence_level']} ({res['confidence']}%)</p>
                    <p style="color: #ffffff; margin-bottom: 20px;"><strong>Recommendation:</strong> {res['recommendation']}</p>
                    <div style="background-color: rgba(255,255,255,0.2); border-radius: 8px; height: 12px; overflow: hidden; margin-bottom: 8px;">
                        <div style="background-color: #d32f2f; height: 100%; width: {res['confidence']}%; border-radius: 8px; transition: width 0.5s ease;"></div>
                    </div>
                    <div style="text-align: center; color: #ffffff; font-size: 13px; font-weight: 500;">{res['confidence']}% Confidence Level</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: rgba(0,255,0,0.1); border: 1px solid rgba(0,255,0,0.3); border-radius: 12px; padding: 20px; margin-bottom: 20px;">
                    <h3 style="color: #388e3c; margin-bottom: 10px;">✅ DIAGNOSIS: NORMAL CHEST X-RAY</h3>
                    <p style="color: #ffffff; margin-bottom: 8px;"><strong>Confidence:</strong> {res['confidence_level']} ({res['confidence']}%)</p>
                    <p style="color: #ffffff; margin-bottom: 20px;"><strong>Recommendation:</strong> {res['recommendation']}</p>
                    <div style="background-color: rgba(255,255,255,0.2); border-radius: 8px; height: 12px; overflow: hidden; margin-bottom: 8px;">
                        <div style="background-color: #388e3c; height: 100%; width: {res['confidence']}%; border-radius: 8px; transition: width 0.5s ease;"></div>
                    </div>
                    <div style="text-align: center; color: #ffffff; font-size: 13px; font-weight: 500;">{res['confidence']}% Confidence Level</div>
                </div>
                """, unsafe_allow_html=True)

            # PDF GENERATION SECTION
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("📄 Generate PDF Report", key="pdf_btn"):
                    try:
                        with st.spinner("Generating PDF..."):
                            pdf_data = generate_medical_pdf_report(prediction_data, elapsed)
                            filename = f"PneumoDetect_Report_{int(time.time())}.pdf"
                            download_link = create_pdf_download_link(pdf_data, filename)
                            st.session_state["pdf_generated"] = True
                            st.session_state["pdf_download_link"] = download_link
                            st.success("✅ PDF generated successfully!")
                    except Exception as e:
                        st.error(f"❌ Failed to generate PDF: {str(e)}")

            # DOWNLOAD LINK
            if "pdf_generated" in st.session_state and st.session_state.get("pdf_generated", False):
                st.markdown('<div style="text-align: center; margin-top: 15px;">', unsafe_allow_html=True)
                st.markdown(st.session_state.get("pdf_download_link", ""), unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)




# 7. Medical Disclaimer - ENHANCED CENTERED
st.markdown(
    """
    <div class="disclaimer-box" style="text-align: center;">
        <div class="disclaimer-title" style="text-align: center;">⚠️ Medical Disclaimer</div>
        <p style="text-align: center; margin: 15px auto 0 auto; line-height: 1.6;">
            This AI tool is intended for preliminary screening purposes only.<br>
            Always seek advice from qualified healthcare professionals before making medical decisions.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# 8. About the Developer
st.markdown(
    """
    <div class="about-developer">
        <div class="developer-title">👩‍💻 Developed By</div>
        <div class="developer-bio">
          Ayushi Rathour | Biotechnology Graduate bridging AI & healthcare
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 9. Social Media Icons & Developer Credit (At the very end)
st.markdown(
    """
    <div class="social-section">
        <div class="developed-by"></div>
        <div class="social-icons">
            <a href="https://github.com/ayushirathour" target="_blank" title="GitHub">
                <svg class="social-icon" xmlns="http://www.w3.org/2000/svg" x="0px" y="0px" width="100" height="100" viewBox="0 0 48 48">
                    <path d="M44,24c0,8.96-5.88,16.54-14,19.08V38c0-1.71-0.72-3.24-1.86-4.34c5.24-0.95,7.86-4,7.86-9.66c0-2.45-0.5-4.39-1.48-5.9 c0.44-1.71,0.7-4.14-0.52-6.1c-2.36,0-4.01,1.39-4.98,2.53C27.57,14.18,25.9,14,24,14c-1.8,0-3.46,0.2-4.94,0.61 C18.1,13.46,16.42,12,14,12c-1.42,2.28-0.84,4.74-0.3,6.12C12.62,19.63,12,21.57,12,24c0,5.66,2.62,8.71,7.86,9.66 c-0.67,0.65-1.19,1.44-1.51,2.34H16c-1.44,0-2-0.64-2.77-1.68c-0.77-1.04-1.6-1.74-2.59-2.03c-0.53-0.06-0.89,0.37-0.42,0.75 c1.57,1.13,1.68,2.98,2.31,4.19C13.1,38.32,14.28,39,15.61,39H18v4.08C9.88,40.54,4,32.96,4,24C4,12.95,12.95,4,24,4 S44,12.95,44,24z"></path>
                </svg>
            </a>
            <a href="mailto:ayushirathour1804@gmail.com" title="Gmail">
                <svg class="social-icon" xmlns="http://www.w3.org/2000/svg" x="0px" y="0px" width="100" height="100" viewBox="0 0 64 64">
                    <path d="M47 34.837V52h7.533C56.448 52 58 50.448 58 48.533V29.486L47 34.837zM47 32l11-7.333v-5.426c0-1.914-.812-3.781-2.325-4.953-2.336-1.809-5.515-1.673-7.665.151L47 15.232V32zM19.814 33.822L32 41 44.349 33.726 43.443 18.023 32 27 20.718 18.149zM17.153 32.102v-16.75L15.99 14.44c-2.15-1.823-5.329-1.961-7.664-.151C6.812 15.46 6 17.328 6 19.243v5.424L17.153 32.102zM6 29.486v19.047C6 50.448 7.552 52 9.467 52H17V34.837L6 29.486z"></path>
                </svg>
            </a>
            <a href="https://www.linkedin.com/in/ayushirathour" target="_blank" title="LinkedIn">
                <svg class="social-icon" xmlns="http://www.w3.org/2000/svg" x="0px" y="0px" width="100" height="100" viewBox="0 0 30 30">
                    <path d="M24,4H6C4.895,4,4,4.895,4,6v18c0,1.105,0.895,2,2,2h18c1.105,0,2-0.895,2-2V6C26,4.895,25.105,4,24,4z M10.954,22h-2.95 v-9.492h2.95V22z M9.449,11.151c-0.951,0-1.72-0.771-1.72-1.72c0-0.949,0.77-1.719,1.72-1.719c0.948,0,1.719,0.771,1.719,1.719 C11.168,10.38,10.397,11.151,9.449,11.151z M22.004,22h-2.948v-4.616c0-1.101-0.02-2.517-1.533-2.517 c-1.535,0-1.771,1.199-1.771,2.437V22h-2.948v-9.492h2.83v1.297h0.04c0.394-0.746,1.356-1.533,2.791-1.533 c2.987,0,3.539,1.966,3.539,4.522V22z"></path>
                </svg>
            </a>
            <a href="https://huggingface.co/ayushirathour" target="_blank" title="Hugging Face">
                <svg class="social-icon" xmlns="http://www.w3.org/2000/svg" x="0px" y="0px" width="100" height="100" viewBox="0 0 50 50">
                    <path d="M 25 4 C 13.954 4 5 12.954 5 24 C 5 26.206 5.37125 28.322641 6.03125 30.306641 C 6.13425 30.328641 6.2367031 30.350859 6.3457031 30.380859 C 6.3477031 30.219859 6.3598594 30.051953 6.3808594 29.876953 C 6.5538594 28.428953 7.6701563 27.376953 9.0351562 27.376953 C 9.4001563 27.376953 9.7642344 27.451609 10.115234 27.599609 C 10.763234 27.872609 11.695781 28.335359 12.300781 29.193359 C 12.410781 29.349359 12.519906 29.505109 12.628906 29.662109 C 13.345906 28.496109 14.423297 28.286906 14.904297 28.253906 C 14.972297 28.248906 15.038469 28.246094 15.105469 28.246094 C 16.225469 28.246094 17.221578 28.897281 17.767578 29.988281 C 17.964578 30.382281 18.146078 30.782594 18.330078 31.183594 C 18.538078 31.637594 18.733313 32.066656 18.945312 32.472656 C 19.526312 33.583656 20.266531 34.582406 21.144531 35.441406 C 21.193531 35.489406 21.240109 35.537891 21.287109 35.587891 C 21.943109 36.300891 23.939063 38.47025 22.789062 41.53125 C 22.454062 42.42325 22.025734 43.117641 21.552734 43.681641 C 22.674734 43.877641 23.822 44 25 44 C 26.114 44 27.199672 43.885938 28.263672 43.710938 C 27.780672 43.141938 27.341 42.43925 27 41.53125 C 25.851 38.47025 27.845953 36.300891 28.501953 35.587891 C 28.547953 35.537891 28.595531 35.488406 28.644531 35.441406 C 29.522531 34.581406 30.26275 33.583656 30.84375 32.472656 C 31.05575 32.066656 31.252938 31.637594 31.460938 31.183594 C 31.643937 30.782594 31.826437 30.382281 32.023438 29.988281 C 32.569437 28.897281 33.565547 28.246094 34.685547 28.246094 C 34.751547 28.246094 34.818719 28.248906 34.886719 28.253906 C 35.367719 28.286906 36.446109 28.495156 37.162109 29.660156 C 37.271109 29.503156 37.380234 29.347406 37.490234 29.191406 C 38.095234 28.332406 39.027781 27.871609 39.675781 27.599609 C 40.026781 27.451609 40.390859 27.376953 40.755859 27.376953 C 42.120859 27.376953 43.237156 28.427 43.410156 29.875 C 43.431156 30.05 43.443312 30.217906 43.445312 30.378906 C 43.637313 30.326906 43.816328 30.290766 43.986328 30.259766 C 44.636328 28.289766 45 26.189 45 24 C 45 12.954 36.046 4 25 4 z M 18.296875 15.279297 C 18.574687 15.263672 18.856516 15.294297 19.134766 15.373047 C 20.290766 15.700047 21.0645 16.724484 21.0625 18.021484 C 21.0605 18.057484 21.061016 18.191266 21.041016 18.322266 C 20.884016 19.334266 20.6385 19.478234 19.6875 19.115234 C 18.8925 18.812234 18.697687 18.872078 18.179688 19.580078 C 17.528688 20.469078 17.190031 20.481531 16.457031 19.644531 C 15.699031 18.779531 15.589359 17.488109 16.193359 16.537109 C 16.674109 15.780359 17.463437 15.326172 18.296875 15.279297 z M 31.503906 15.287109 C 31.885633 15.268295 32.272219 15.33175 32.636719 15.484375 C 33.637719 15.902375 34.247047 16.817812 34.248047 18.382812 C 34.242047 18.860813 33.821062 19.588 33.039062 20.125 C 32.699062 20.358 32.422688 20.308281 32.179688 19.988281 C 32.159687 19.962281 32.140141 19.935203 32.119141 19.908203 C 31.303141 18.848203 31.300719 18.838672 30.011719 19.263672 C 29.573719 19.407672 29.329969 19.280469 29.167969 18.855469 C 28.808969 17.908469 29.042344 16.771922 29.777344 16.044922 C 30.247344 15.579922 30.867695 15.318467 31.503906 15.287109 z M 12.5 20 C 13.328 20 14 20.672 14 21.5 C 14 22.328 13.328 23 12.5 23 C 11.672 23 11 22.328 11 21.5 C 11 20.672 11.672 20 12.5 20 z M 37.5 20 C 38.328 20 39 20.672 39 21.5 C 39 22.328 38.328 23 37.5 23 C 36.672 23 36 22.328 36 21.5 C 36 20.672 36.672 20 37.5 20 z M 18.726562 23.896484 C 18.859422 23.903656 19.016969 23.944328 19.199219 24.017578 C 19.937219 24.314578 20.656188 24.660531 21.367188 25.019531 C 23.947188 26.323531 26.487719 26.178422 29.011719 24.857422 C 29.584719 24.557422 30.169531 24.276156 30.769531 24.035156 C 31.566531 23.715156 31.878656 23.927438 31.847656 24.773438 C 31.714656 28.384438 29.205484 31.636094 25.021484 31.621094 C 20.841484 31.639094 18.327828 28.416328 18.173828 24.736328 C 18.149828 24.151328 18.327984 23.874969 18.726562 23.896484 z"></path>
                </svg>
            </a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 10. Footer
st.markdown(
    f"""
    <div class="footer">
        <div class="footer-links">
            <a href="https://github.com/ayushirathour/chest-xray-pneumonia-detection-ai" target="_blank">Source Code & Model Details</a>
            <a href="mailto:ayushirathour1804@gmail.com">Contact</a>
            <a href="#" onclick="alert('Privacy: No data is stored. All processing is done locally.')">Privacy Policy</a>
        </div>
        <div>© 2025 {MODEL_SPECS['name']} {MODEL_SPECS['version']} | All Rights Reserved</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Close container
st.markdown("</div>", unsafe_allow_html=True)

















































