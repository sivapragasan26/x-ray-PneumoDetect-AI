import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

@st.cache_resource
def load_model():
    """Load your trained pneumonia detection model - Silent Loading"""
    
    # Multiple possible paths to try
    possible_paths = [
        'best_chest_xray_model.h5',
        './best_chest_xray_model.h5',
        'api/streamlit_api_folder/best_chest_xray_model.h5',
        '/mount/src/chest-xray-pneumonia-detection-ai/api/streamlit_api_folder/best_chest_xray_model.h5'
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

def preprocess_image(image):
    """Preprocess image for model prediction"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((224, 224))
    img_array = np.array(image).astype(np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

def interpret_prediction(prediction_score):
    """Interpret model prediction with confidence levels"""
    if prediction_score > 0.5:
        diagnosis = "PNEUMONIA"
        confidence = float(prediction_score * 100)
        
        if confidence >= 80:
            level = "High"
            recommendation = "🚨 Strong indication of pneumonia. Seek immediate medical attention."
        elif confidence >= 60:
            level = "Moderate" 
            recommendation = "⚠️ Moderate indication of pneumonia. Medical review recommended."
        else:
            level = "Low"
            recommendation = "💡 Possible pneumonia detected. Further examination advised."
    else:
        diagnosis = "NORMAL"
        confidence = float((1 - prediction_score) * 100)
        
        if confidence >= 80:
            level = "High"
            recommendation = "✅ No signs of pneumonia detected. Chest X-ray appears normal."
        elif confidence >= 60:
            level = "Moderate"
            recommendation = "👍 Likely normal chest X-ray. Routine follow-up if symptoms persist."
        else:
            level = "Low"
            recommendation = "🤔 Unclear result. Manual review by radiologist recommended."
    
    return {
        "diagnosis": diagnosis,
        "confidence": round(confidence, 2),
        "confidence_level": level,
        "recommendation": recommendation
    }

def display_premium_results(result, prediction, image):
    """Display results with ENHANCED premium styling"""
    
    if result['diagnosis'] == 'PNEUMONIA':
        st.markdown(f"""
        <div style='
            background: linear-gradient(135deg, #ff4757, #ff3742, #ff6b7a);
            color: white;
            border: 3px solid #ff6b7a;
            padding: 3rem;
            border-radius: 30px;
            margin: 2rem 0;
            box-shadow: 0 25px 80px rgba(255, 71, 87, 0.5);
            animation: pulseAlert 2s infinite, slideInUp 1s ease-out;
            position: relative;
            overflow: hidden;
            backdrop-filter: blur(20px);
        '>
            <div style='position: relative; z-index: 2; text-align: center;'>
                <div style='font-size: 5rem; margin-bottom: 1.5rem; animation: bounce 2s infinite; filter: drop-shadow(0 0 20px rgba(255,255,255,0.5));'>🚨</div>
                <h1 style='margin: 0; font-size: 3.2rem; font-weight: 900; text-shadow: 3px 3px 8px rgba(0,0,0,0.4); letter-spacing: 2px;'>
                    PNEUMONIA DETECTED
                </h1>
                <div style='
                    background: rgba(255,255,255,0.2);
                    backdrop-filter: blur(15px);
                    border-radius: 25px;
                    padding: 30px;
                    margin: 2.5rem 0;
                    border: 3px solid rgba(255,255,255,0.4);
                    box-shadow: inset 0 0 30px rgba(255,255,255,0.1);
                '>
                    <div style='font-size: 5rem; font-weight: 900; margin: 0; text-shadow: 2px 2px 10px rgba(0,0,0,0.3);'>{result['confidence']}%</div>
                    <div style='font-size: 1.6rem; margin: 15px 0; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;'>
                        Confidence Level: {result['confidence_level']}
                    </div>
                </div>
                <div style='font-size: 1.4rem; line-height: 1.9; padding: 25px; border-radius: 20px; font-weight: 600; background: rgba(255,255,255,0.15); backdrop-filter: blur(10px);'>
                    {result['recommendation']}
                </div>
            </div>
            <div style='position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%); animation: shimmer 3s infinite;'></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='
            background: linear-gradient(135deg, #2ed573, #1e9c57, #5af593);
            color: white;
            border: 3px solid #5af593;
            padding: 3rem;
            border-radius: 30px;
            margin: 2rem 0;
            box-shadow: 0 25px 80px rgba(46, 213, 115, 0.5);
            animation: pulseSuccess 3s infinite, slideInUp 1s ease-out;
            position: relative;
            overflow: hidden;
            backdrop-filter: blur(20px);
        '>
            <div style='position: relative; z-index: 2; text-align: center;'>
                <div style='font-size: 5rem; margin-bottom: 1.5rem; animation: heartbeat 2s infinite; filter: drop-shadow(0 0 20px rgba(255,255,255,0.5));'>✅</div>
                <h1 style='margin: 0; font-size: 3.2rem; font-weight: 900; text-shadow: 3px 3px 8px rgba(0,0,0,0.3); letter-spacing: 2px;'>
                    NORMAL CHEST X-RAY
                </h1>
                <div style='
                    background: rgba(255,255,255,0.2);
                    backdrop-filter: blur(15px);
                    border-radius: 25px;
                    padding: 30px;
                    margin: 2.5rem 0;
                    border: 3px solid rgba(255,255,255,0.4);
                    box-shadow: inset 0 0 30px rgba(255,255,255,0.1);
                '>
                    <div style='font-size: 5rem; font-weight: 900; margin: 0; text-shadow: 2px 2px 10px rgba(0,0,0,0.3);'>{result['confidence']}%</div>
                    <div style='font-size: 1.6rem; margin: 15px 0; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;'>
                        Confidence Level: {result['confidence_level']}
                    </div>
                </div>
                <div style='font-size: 1.4rem; line-height: 1.9; padding: 25px; border-radius: 20px; font-weight: 600; background: rgba(255,255,255,0.15); backdrop-filter: blur(10px);'>
                    {result['recommendation']}
                </div>
            </div>
            <div style='position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%); animation: shimmer 3s infinite;'></div>
        </div>
        """, unsafe_allow_html=True)

    # Enhanced Technical Analysis Section
    st.markdown(f"""
    <div style='
        background: linear-gradient(145deg, #2c3e50, #34495e, #2c3e50);
        color: white;
        padding: 35px;
        border-radius: 30px;
        margin: 35px 0;
        position: relative;
        overflow: hidden;
        border: 3px solid #3498db;
        box-shadow: 0 20px 60px rgba(52,152,219,0.3);
    '>
        <div style='position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: linear-gradient(45deg, transparent 30%, rgba(52,152,219,0.1) 50%, transparent 70%); animation: shimmer 4s infinite;'></div>
        <h3 style='color: #74b9ff; margin-bottom: 30px; font-size: 2rem; text-align: center; font-weight: 800; text-transform: uppercase; letter-spacing: 2px; position: relative; z-index: 2;'>
            🔬 Technical Analysis Dashboard
        </h3>
        <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 25px; position: relative; z-index: 2;'>
            <div style='background: rgba(255,255,255,0.15); padding: 25px; border-radius: 20px; text-align: center; backdrop-filter: blur(15px); border: 2px solid rgba(255,255,255,0.2); transition: all 0.3s ease;'>
                <h4 style='color: #74b9ff; margin: 0 0 15px 0; font-size: 1.2rem; font-weight: 700;'>Raw Score</h4>
                <p style='color: #fff; font-size: 1.4rem; font-weight: 800; text-shadow: 1px 1px 3px rgba(0,0,0,0.5);'>{prediction:.4f}</p>
            </div>
            <div style='background: rgba(255,255,255,0.15); padding: 25px; border-radius: 20px; text-align: center; backdrop-filter: blur(15px); border: 2px solid rgba(255,255,255,0.2); transition: all 0.3s ease;'>
                <h4 style='color: #55a3ff; margin: 0 0 15px 0; font-size: 1.2rem; font-weight: 700;'>Threshold</h4>
                <p style='color: #fff; font-size: 1.4rem; font-weight: 800; text-shadow: 1px 1px 3px rgba(0,0,0,0.5);'>0.5</p>
            </div>
            <div style='background: rgba(255,255,255,0.15); padding: 25px; border-radius: 20px; text-align: center; backdrop-filter: blur(15px); border: 2px solid rgba(255,255,255,0.2); transition: all 0.3s ease;'>
                <h4 style='color: #a29bfe; margin: 0 0 15px 0; font-size: 1.2rem; font-weight: 700;'>Architecture</h4>
                <p style='color: #fff; font-size: 1.1rem; font-weight: 700; text-shadow: 1px 1px 3px rgba(0,0,0,0.5);'>MobileNetV2</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Streamlit App Interface
st.set_page_config(
    page_title="PneumoDetect AI | Ayushi Rathour - Healthcare Innovation",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced Premium CSS with Modern Design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Poppins:wght@300;400;500;600;700;800;900&display=swap');
    
    * {
        font-family: 'Poppins', 'Inter', sans-serif;
        box-sizing: border-box;
    }
    
    /* Hide Streamlit elements */
    .css-1d391kg, [data-testid="stSidebar"] {
        display: none !important;
    }
    
    /* Your Custom Gradient Background */
    .main, .stApp {
        background: linear-gradient(300deg, #211c6a, #17594a, #08045b, #264422, #b7b73d);
        background-size: 300% 300%;
        animation: gradient-animation 25s ease infinite;
        min-height: 100vh;
        position: relative;
        overflow-x: hidden;
    }
    
    .main::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="rgba(255,255,255,0.1)"/><circle cx="75" cy="75" r="1" fill="rgba(255,255,255,0.1)"/><circle cx="50" cy="10" r="0.5" fill="rgba(255,255,255,0.1)"/><circle cx="10" cy="60" r="0.5" fill="rgba(255,255,255,0.1)"/><circle cx="90" cy="40" r="0.5" fill="rgba(255,255,255,0.1)"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
        opacity: 0.3;
        z-index: -1;
    }
    
    @keyframes gradient-animation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Remove default Streamlit padding */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1400px;
    }
    
    /* Enhanced Title Section */
    .title-section {
        text-align: center;
        padding: 4rem 0 3rem 0;
        margin-bottom: 3rem;
        position: relative;
    }
    
    .title-section::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 200px;
        height: 200px;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        border-radius: 50%;
        animation: pulse 4s ease-in-out infinite;
    }
    
    .main-title {
        font-size: 5rem;
        font-weight: 900;
        color: white;
        text-shadow: 3px 3px 15px rgba(0,0,0,0.5), 0 0 30px rgba(255,255,255,0.3);
        margin-bottom: 1.5rem;
        animation: titleGlow 3s ease-in-out infinite alternate;
        position: relative;
        z-index: 2;
        letter-spacing: 3px;
        text-transform: uppercase;
    }
    
    @keyframes titleGlow {
        from { text-shadow: 3px 3px 15px rgba(0,0,0,0.5), 0 0 30px rgba(255,255,255,0.3); }
        to { text-shadow: 3px 3px 25px rgba(0,0,0,0.7), 0 0 50px rgba(255,255,255,0.5); }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.3; transform: translate(-50%, -50%) scale(1); }
        50% { opacity: 0.6; transform: translate(-50%, -50%) scale(1.1); }
    }
    
    .subtitle {
        font-size: 1.6rem;
        color: rgba(255,255,255,0.95);
        margin-bottom: 2.5rem;
        font-weight: 500;
        position: relative;
        z-index: 2;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.3);
    }
    
    .developer-badge {
        background: rgba(255,255,255,0.2);
        backdrop-filter: blur(20px);
        padding: 25px 50px;
        border-radius: 60px;
        display: inline-block;
        color: white;
        font-weight: 700;
        font-size: 1.3rem;
        border: 3px solid rgba(255,255,255,0.4);
        animation: developerPulse 4s ease-in-out infinite;
        box-shadow: 0 15px 40px rgba(0,0,0,0.3);
        position: relative;
        z-index: 2;
        transition: all 0.3s ease;
    }
    
    .developer-badge:hover {
        transform: scale(1.05);
        box-shadow: 0 20px 50px rgba(0,0,0,0.4);
        border-color: rgba(255,255,255,0.6);
    }
    
    @keyframes developerPulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    /* Enhanced Stats Grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 30px;
        margin: 4rem 0;
        padding: 0 1rem;
    }
    
    .stat-card {
        background: rgba(255,255,255,0.95);
        backdrop-filter: blur(25px);
        padding: 2.5rem;
        border-radius: 30px;
        text-align: center;
        box-shadow: 0 25px 60px rgba(0,0,0,0.2);
        border: 3px solid rgba(255,255,255,0.4);
        transition: all 0.5s ease;
        position: relative;
        overflow: hidden;
        cursor: pointer;
    }
    
    .stat-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(52,152,219,0.3), transparent);
        transition: left 0.8s;
    }
    
    .stat-card:hover::before {
        left: 100%;
    }
    
    .stat-card:hover {
        transform: translateY(-15px) scale(1.02);
        box-shadow: 0 35px 80px rgba(0,0,0,0.3);
        border-color: rgba(52,152,219,0.6);
    }
    
    .stat-number {
        font-size: 4rem;
        font-weight: 900;
        color: #2c3e50;
        margin: 0;
        position: relative;
        z-index: 2;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .stat-label {
        color: #7f8c8d;
        font-weight: 700;
        margin-top: 15px;
        font-size: 1.3rem;
        position: relative;
        z-index: 2;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Enhanced Upload Container */
    .enhanced-upload-container {
        background: rgba(255,255,255,0.98);
        backdrop-filter: blur(25px);
        border-radius: 30px;
        margin: 3rem auto;
        max-width: 1000px;
        box-shadow: 0 25px 70px rgba(0,0,0,0.2);
        border: 3px solid #3498db;
        overflow: hidden;
        transition: all 0.4s ease;
        position: relative;
    }
    
    .enhanced-upload-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 5px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb, #f5576c, #4facfe);
        animation: rainbow 3s linear infinite;
    }
    
    @keyframes rainbow {
        0% { background-position: 0% 50%; }
        100% { background-position: 100% 50%; }
    }
    
    .enhanced-upload-container:hover {
        transform: translateY(-8px);
        box-shadow: 0 30px 80px rgba(0,0,0,0.25);
        border-color: #2980b9;
    }
    
    .upload-header {
        background: linear-gradient(135deg, #3498db, #2980b9, #1abc9c);
        color: white;
        padding: 25px;
        text-align: center;
        font-weight: 800;
        font-size: 1.4rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        position: relative;
        overflow: hidden;
    }
    
    .upload-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        animation: shimmer 2s infinite;
    }
    
    .upload-content {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0;
        min-height: 350px;
    }
    
    .drop-zone {
        padding: 50px 30px;
        text-align: center;
        border-right: 3px solid #e0e0e0;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        background: linear-gradient(135deg, rgba(52,152,219,0.05), rgba(26,188,156,0.05));
        cursor: pointer;
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }
    
    .drop-zone::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(52,152,219,0.1) 50%, transparent 70%);
        transform: translateX(-100%);
        transition: transform 0.6s ease;
    }
    
    .drop-zone:hover::before {
        transform: translateX(100%);
    }
    
    .drop-zone:hover {
        background: linear-gradient(135deg, rgba(52,152,219,0.1), rgba(26,188,156,0.1));
        transform: scale(1.02);
    }
    
    .drop-zone-icon {
        font-size: 4rem;
        color: #3498db;
        margin-bottom: 20px;
        animation: float 3s ease-in-out infinite;
        filter: drop-shadow(0 5px 15px rgba(52,152,219,0.3));
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .drop-zone-text {
        color: #3498db;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 15px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .drop-zone-subtext {
        color: #7f8c8d;
        font-size: 1rem;
        font-weight: 500;
    }
    
    .preview-zone {
        padding: 30px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        background: linear-gradient(135deg, rgba(255,255,255,0.8), rgba(236,240,241,0.8));
        position: relative;
    }
    
    .preview-placeholder {
        color: #bdc3c7;
        font-size: 1.2rem;
        text-align: center;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .preview-image {
        max-width: 100%;
        max-height: 280px;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        border: 3px solid #3498db;
        transition: all 0.3s ease;
    }
    
    .preview-image:hover {
        transform: scale(1.05);
        box-shadow: 0 15px 40px rgba(0,0,0,0.3);
    }
    
    /* Hide default file uploader */
    .stFileUploader {
        display: none !important;
    }
    
    /* Enhanced Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 18px 40px;
        border-radius: 30px;
        font-weight: 800;
        font-size: 1.2rem;
        box-shadow: 0 15px 35px rgba(102,126,234,0.4);
        transition: all 0.4s ease;
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 2px;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.6s;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 45px rgba(102,126,234,0.6);
        background: linear-gradient(135deg, #764ba2, #667eea);
    }
    
    /* Enhanced Footer */
    .footer-section {
        background: linear-gradient(135deg, rgba(44, 62, 80, 0.95), rgba(52, 73, 94, 0.95));
        backdrop-filter: blur(25px);
        color: white;
        padding: 4rem 2rem;
        border-radius: 30px;
        margin-top: 5rem;
        text-align: center;
        border: 3px solid rgba(255,255,255,0.2);
        box-shadow: 0 20px 50px rgba(0,0,0,0.3);
        position: relative;
        overflow: hidden;
    }
    
    .footer-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb, #f5576c, #4facfe);
        animation: rainbow 3s linear infinite;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-title { font-size: 3.5rem; letter-spacing: 2px; }
        .stats-grid { grid-template-columns: 1fr; gap: 20px; }
        .stat-card { padding: 2rem; }
        .developer-badge { font-size: 1.1rem; padding: 20px 40px; }
        .upload-content { grid-template-columns: 1fr; }
        .drop-zone { border-right: none; border-bottom: 3px solid #e0e0e0; }
        .enhanced-upload-container { margin: 2rem 1rem; }
        .block-container { padding: 1rem; }
    }
    
    @media (max-width: 480px) {
        .main-title { font-size: 2.8rem; }
        .subtitle { font-size: 1.3rem; }
        .developer-badge { font-size: 1rem; padding: 15px 30px; }
        .stat-number { font-size: 3rem; }
        .upload-header { font-size: 1.2rem; padding: 20px; }
    }
    
    /* Enhanced Animation Keyframes */
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    @keyframes pulseAlert {
        0%, 100% { box-shadow: 0 25px 80px rgba(255, 71, 87, 0.5); }
        50% { box-shadow: 0 30px 100px rgba(255, 71, 87, 0.7); }
    }
    
    @keyframes pulseSuccess {
        0%, 100% { box-shadow: 0 25px 80px rgba(46, 213, 115, 0.5); }
        50% { box-shadow: 0 30px 100px rgba(46, 213, 115, 0.7); }
    }
    
    @keyframes slideInUp {
        from { opacity: 0; transform: translateY(80px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-20px); }
        60% { transform: translateY(-10px); }
    }
    
    @keyframes heartbeat {
        0% { transform: scale(1); }
        14% { transform: scale(1.3); }
        28% { transform: scale(1); }
        42% { transform: scale(1.3); }
        70% { transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)

# Enhanced Title Section
st.markdown("""
<div class='title-section'>
    <div style='font-size: 7rem; margin-bottom: 2rem; animation: float 4s ease-in-out infinite; filter: drop-shadow(0 0 30px rgba(255,255,255,0.5));'>🫁</div>
    <h1 class='main-title'>PneumoDetect AI</h1>
    <p class='subtitle'>🔬 Advanced Chest X-Ray Analysis | 🎯 Clinical-Grade Artificial Intelligence</p>
    <div class='developer-badge'>
        👩‍💻 Developed by Ayushi Rathour - Biotechnology Graduate | Exploring AI in Healthcare
    </div>
</div>
""", unsafe_allow_html=True)

# Enhanced Performance Statistics
st.markdown("""
<div class='stats-grid'>
    <div class='stat-card'>
        <p class='stat-number'>86%</p>
        <p class='stat-label'>🎯 Model Accuracy</p>
    </div>
    <div class='stat-card'>
        <p class='stat-number'>96.4%</p>
        <p class='stat-label'>🔍 Sensitivity Rate</p>
    </div>
    <div class='stat-card'>
        <p class='stat-number'>74.8%</p>
        <p class='stat-label'>📊 Specificity Rate</p>
    </div>
    <div class='stat-card'>
        <p class='stat-number'>485</p>
        <p class='stat-label'>🧪 Validation Samples</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Load Model (Silent)
model = load_model()

if model is not None:
    # Enhanced Upload Container with Working Preview
    st.markdown("""
    <div class='enhanced-upload-container'>
        <div class='upload-header'>
            📤 Upload Chest X-Ray for AI Analysis
        </div>
        <div class='upload-content'>
            <div class='drop-zone'>
                <div class='drop-zone-icon'>🫁</div>
                <div class='drop-zone-text'>Drag and Drop File Here</div>
                <div class='drop-zone-subtext'>Limit 200MB per file • JPG, PNG, JPEG</div>
            </div>
            <div class='preview-zone'>
                <div class='preview-placeholder'>📋 Image Preview Will Appear Here</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader (styled to be invisible but functional)
    uploaded_file = st.file_uploader("", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        # Display image with enhanced styling - FIXED VERSION
        image = Image.open(uploaded_file)
        
        # Create a centered container for the image
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style='
                background: rgba(255,255,255,0.98);
                backdrop-filter: blur(25px);
                padding: 30px;
                border-radius: 25px;
                margin: 30px 0;
                box-shadow: 0 20px 50px rgba(0,0,0,0.2);
                border: 3px solid rgba(52,152,219,0.4);
                text-align: center;
                position: relative;
                overflow: hidden;
            '>
                <div style='
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    height: 3px;
                    background: linear-gradient(90deg, #3498db, #2980b9, #3498db);
                    animation: shimmer 2s infinite;
                '></div>
            """, unsafe_allow_html=True)
            
            # FIXED: Proper image display without JavaScript
            st.image(image, caption="📸 Uploaded Chest X-Ray - Ready for AI Analysis", use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

        # Enhanced Analyze Button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔬 Analyze X-ray with AI", type="primary"):
                with st.spinner("🧠 Advanced AI analysis in progress..."):
                    processed_image = preprocess_image(image)
                    prediction = model.predict(processed_image, verbose=0)[0][0]
                    result = interpret_prediction(prediction)

                display_premium_results(result, prediction, image)

    # Enhanced Footer Section - WORKING VERSION
    st.markdown("""
    <div class='footer-section'>
        <div style='margin-bottom: 3rem;'>
            <h3 style='color: #74b9ff; margin-bottom: 1.5rem; font-size: 2rem; font-weight: 800; text-transform: uppercase; letter-spacing: 2px;'>⚠️ Medical Disclaimer</h3>
            <p style='font-size: 1.2rem; line-height: 1.9; margin-bottom: 2.5rem; color: rgba(255,255,255,0.95); font-weight: 500;'>
                This AI system is designed for <strong>preliminary screening purposes only</strong>.<br>
                Always consult qualified healthcare professionals for definitive medical decisions.<br>
                This tool serves as a supportive diagnostic aid, not a replacement for professional medical judgment.
            </p>
        </div>
        
        <div style='
            border-top: 3px solid rgba(255,255,255,0.2); 
            padding-top: 3rem;
            text-align: center;
        '>
            <div style='
                background: rgba(255,255,255,0.15);
                padding: 2.5rem;
                border-radius: 25px;
                margin: 1.5rem 0;
                border: 2px solid rgba(255,255,255,0.3);
                backdrop-filter: blur(15px);
            '>
                <h4 style='color: #74b9ff; margin-bottom: 1.5rem; font-size: 1.5rem; font-weight: 800; text-transform: uppercase; letter-spacing: 1px;'>👩‍💻 Developer</h4>
                <p style='margin: 0; font-size: 1.5rem; font-weight: 800; color: white; text-shadow: 1px 1px 3px rgba(0,0,0,0.5);'>Ayushi Rathour</p>
                <p style='margin: 12px 0; color: #ddd; font-size: 1.2rem; font-weight: 600;'>Biotechnology Graduate</p>
                <p style='margin: 12px 0; color: #ddd; font-size: 1.2rem; font-weight: 600;'>Exploring AI in Healthcare</p>
                <p style='margin: 20px 0; color: #74b9ff; font-weight: 700; font-size: 1.2rem; text-transform: uppercase; letter-spacing: 1px;'>🚀 Bridging Biology & Technology</p>
            </div>
            
            <div style='
                margin-top: 3rem; 
                padding-top: 3rem; 
                border-top: 2px solid rgba(255,255,255,0.15);
            '>
                <p style='color: #bdc3c7; font-size: 1.1rem; line-height: 1.9; font-weight: 500;'>
                    🏥 Developed with ❤️ for Healthcare Innovation<br>
                    🔬 Powered by TensorFlow & Streamlit<br>
                    <strong>PneumoDetect AI v2.0</strong> | © 2024 Ayushi Rathour<br>
                    Biotechnology × Artificial Intelligence
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("❌ Model failed to load. Please check the model file.")
