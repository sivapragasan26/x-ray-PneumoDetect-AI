import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

@st.cache_resource
def load_model():
    """Load your trained pneumonia detection model - Silent Loading"""
    
    # Multiple possible paths to try (VERIFIED - All paths correct)
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
    """Preprocess image for model prediction - VERIFIED LOGIC"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((224, 224))
    img_array = np.array(image).astype(np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

def interpret_prediction(prediction_score):
    """Interpret model prediction with confidence levels - VERIFIED LOGIC"""
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
    """Display results with enhanced styling - UPDATED CONTENT"""
    
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
                <div style='font-size: 5rem; margin-bottom: 1.5rem; animation: bounce 2s infinite;'>🩺</div>
                <h1 style='margin: 0; font-size: 3.2rem; font-weight: 900; letter-spacing: 2px;'>
                    Diagnosis Result: Pneumonia Detected
                </h1>
                <div style='
                    background: rgba(255,255,255,0.2);
                    backdrop-filter: blur(15px);
                    border-radius: 25px;
                    padding: 30px;
                    margin: 2.5rem 0;
                    border: 3px solid rgba(255,255,255,0.4);
                '>
                    <div style='font-size: 5rem; font-weight: 900; margin: 0;'>{result['confidence']}% confidence</div>
                    <div style='font-size: 1.6rem; margin: 15px 0; font-weight: 700; text-transform: uppercase;'>
                        Confidence Level: {result['confidence_level']}
                    </div>
                </div>
                <div style='font-size: 1.4rem; line-height: 1.9; padding: 25px; border-radius: 20px; font-weight: 600; background: rgba(255,255,255,0.15);'>
                    ⚠ Recommendation: {result['recommendation']}
                </div>
            </div>
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
                <div style='font-size: 5rem; margin-bottom: 1.5rem; animation: heartbeat 2s infinite;'>🩺</div>
                <h1 style='margin: 0; font-size: 3.2rem; font-weight: 900; letter-spacing: 2px;'>
                    Diagnosis Result: Normal Chest X-Ray
                </h1>
                <div style='
                    background: rgba(255,255,255,0.2);
                    backdrop-filter: blur(15px);
                    border-radius: 25px;
                    padding: 30px;
                    margin: 2.5rem 0;
                    border: 3px solid rgba(255,255,255,0.4);
                '>
                    <div style='font-size: 5rem; font-weight: 900; margin: 0;'>{result['confidence']}% confidence</div>
                    <div style='font-size: 1.6rem; margin: 15px 0; font-weight: 700; text-transform: uppercase;'>
                        Confidence Level: {result['confidence_level']}
                    </div>
                </div>
                <div style='font-size: 1.4rem; line-height: 1.9; padding: 25px; border-radius: 20px; font-weight: 600; background: rgba(255,255,255,0.15);'>
                    ⚠ Recommendation: {result['recommendation']}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Technical Summary - UPDATED CONTENT
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
        <h3 style='color: #74b9ff; margin-bottom: 30px; font-size: 2rem; text-align: center; font-weight: 800; text-transform: uppercase; letter-spacing: 2px; position: relative; z-index: 2;'>
            🔬 Technical Summary
        </h3>
        <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 25px; position: relative; z-index: 2;'>
            <div style='background: rgba(255,255,255,0.15); padding: 25px; border-radius: 20px; text-align: center; border: 2px solid rgba(255,255,255,0.2);'>
                <h4 style='color: #74b9ff; margin: 0 0 15px 0; font-size: 1.2rem; font-weight: 700;'>Model Architecture</h4>
                <p style='color: #fff; font-size: 1.4rem; font-weight: 800;'>MobileNetV2</p>
            </div>
            <div style='background: rgba(255,255,255,0.15); padding: 25px; border-radius: 20px; text-align: center; border: 2px solid rgba(255,255,255,0.2);'>
                <h4 style='color: #55a3ff; margin: 0 0 15px 0; font-size: 1.2rem; font-weight: 700;'>Threshold</h4>
                <p style='color: #fff; font-size: 1.4rem; font-weight: 800;'>0.5</p>
            </div>
            <div style='background: rgba(255,255,255,0.15); padding: 25px; border-radius: 20px; text-align: center; border: 2px solid rgba(255,255,255,0.2);'>
                <h4 style='color: #a29bfe; margin: 0 0 15px 0; font-size: 1.2rem; font-weight: 700;'>Raw Score</h4>
                <p style='color: #fff; font-size: 1.1rem; font-weight: 700;'>{prediction:.4f}</p>
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

# Your Custom Gradient Background + Enhanced CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
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
    }
    
    @keyframes gradient-animation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
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
    
    .main-title {
        font-size: 5rem;
        font-weight: 900;
        color: white;
        text-shadow: 3px 3px 15px rgba(0,0,0,0.5), 0 0 30px rgba(255,255,255,0.3);
        margin-bottom: 1.5rem;
        position: relative;
        z-index: 2;
        letter-spacing: 3px;
        text-transform: uppercase;
    }
    
    .subtitle {
        font-size: 1.6rem;
        color: rgba(255,255,255,0.95);
        margin-bottom: 1.5rem;
        font-weight: 500;
        position: relative;
        z-index: 2;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.3);
    }
    
    .tagline {
        font-size: 1.8rem;
        color: rgba(255,255,255,0.95);
        margin-bottom: 2.5rem;
        font-weight: 700;
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
        box-shadow: 0 15px 40px rgba(0,0,0,0.3);
        position: relative;
        z-index: 2;
    }
    
    /* Enhanced Stats Grid - UPDATED TO 3 CARDS */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
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
    
    .stat-card:hover {
        transform: translateY(-15px) scale(1.02);
        box-shadow: 0 35px 80px rgba(0,0,0,0.3);
    }
    
    .stat-number {
        font-size: 4rem;
        font-weight: 900;
        color: #2c3e50;
        margin: 0;
        position: relative;
        z-index: 2;
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
    
    /* WORKING DRAG & DROP STYLING */
    .stFileUploader {
        background: rgba(255,255,255,0.98);
        backdrop-filter: blur(25px);
        border-radius: 30px;
        margin: 3rem auto;
        max-width: 800px;
        box-shadow: 0 25px 70px rgba(0,0,0,0.2);
        border: 3px dashed #3498db;
        padding: 2rem;
        transition: all 0.4s ease;
    }
    
    .stFileUploader:hover {
        transform: translateY(-8px);
        box-shadow: 0 30px 80px rgba(0,0,0,0.25);
        border-color: #2980b9;
    }
    
    .stFileUploader label {
        color: #3498db !important;
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    .stFileUploader label::before {
        content: "🫁 ";
        font-size: 3rem;
        display: block;
        margin-bottom: 1rem;
    }
    
    .stFileUploader label::after {
        content: "\ADrag & drop your file or click to browse.\ASupported formats: JPG, PNG, JPEG | Max 200MB";
        white-space: pre;
        display: block;
        color: #7f8c8d;
        font-size: 1rem;
        font-weight: 500;
        margin-top: 15px;
        text-transform: none;
        letter-spacing: normal;
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
    }
    
    .stButton > button:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 45px rgba(102,126,234,0.6);
    }
    
    /* Animation Keyframes */
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
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-title { font-size: 3.5rem; letter-spacing: 2px; }
        .stats-grid { grid-template-columns: 1fr; gap: 20px; }
        .stat-card { padding: 2rem; }
        .developer-badge { font-size: 1.1rem; padding: 20px 40px; }
        .stFileUploader { margin: 2rem 1rem; }
    }
</style>
""", unsafe_allow_html=True)

# UPDATED HEADER SECTION
st.markdown("""
<div class='title-section'>
    <div style='font-size: 7rem; margin-bottom: 2rem; filter: drop-shadow(0 0 30px rgba(255,255,255,0.5));'>🫁</div>
    <h1 class='main-title'>PneumoDetect AI</h1>
    <p class='subtitle'>Advanced Chest X-Ray Analysis | Clinical-Grade Artificial Intelligence</p>
    <div class='tagline'>Fast. Accurate. Reliable.<br>AI-powered pneumonia detection in just 2.5 seconds.</div>
</div>
""", unsafe_allow_html=True)

# UPDATED PERFORMANCE STATISTICS - 3 CARDS ONLY
st.markdown("""
<div class='stats-grid'>
    <div class='stat-card'>
        <p class='stat-number'>86%</p>
        <p class='stat-label'>🎯 Accuracy</p>
    </div>
    <div class='stat-card'>
        <p class='stat-number'>96.4%</p>
        <p class='stat-label'>🔍 Sensitivity</p>
    </div>
    <div class='stat-card'>
        <p class='stat-number'>2.5 sec</p>
        <p class='stat-label'>⏱ Avg. Prediction Time</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Load Model (Silent) - VERIFIED LOGIC
model = load_model()

if model is not None:
    # UPDATED UPLOAD SECTION
    st.markdown("""
    <div style='text-align: center; margin: 2rem 0;'>
        <h2 style='color: white; font-weight: 700; margin-bottom: 1rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);'>
            📤 Upload Chest X-Ray for Instant AI Analysis
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # FUNCTIONAL file uploader with drag & drop styling
    uploaded_file = st.file_uploader(
        "Upload Chest X-Ray for Instant AI Analysis",
        type=['jpg', 'png', 'jpeg'],
        help="Drag & drop your file or click to browse"
    )
    
    if uploaded_file is not None:
        # Display image with enhanced styling
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
            '>
            """, unsafe_allow_html=True)
            
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

    # UPDATED FOOTER SECTION
    st.markdown("""
    <div style='
        background: linear-gradient(135deg, rgba(44, 62, 80, 0.95), rgba(52, 73, 94, 0.95));
        backdrop-filter: blur(25px);
        color: white;
        padding: 4rem 2rem;
        border-radius: 30px;
        margin-top: 5rem;
        text-align: center;
        border: 3px solid rgba(255,255,255,0.2);
        box-shadow: 0 20px 50px rgba(0,0,0,0.3);
    '>
        <div style='margin-bottom: 3rem;'>
            <h3 style='color: #74b9ff; margin-bottom: 1.5rem; font-size: 2rem; font-weight: 800; text-transform: uppercase; letter-spacing: 2px;'>⚠ Medical Disclaimer</h3>
            <p style='font-size: 1.2rem; line-height: 1.9; margin-bottom: 2.5rem; color: rgba(255,255,255,0.95); font-weight: 500;'>
                This AI tool is intended for preliminary screening only.<br>
                Always seek advice from qualified healthcare professionals before making medical decisions.
            </p>
        </div>
        
        <div style='border-top: 3px solid rgba(255,255,255,0.2); padding-top: 3rem; text-align: center;'>
            <div style='background: rgba(255,255,255,0.15); padding: 2.5rem; border-radius: 25px; margin: 1.5rem 0; border: 2px solid rgba(255,255,255,0.3);'>
                <h4 style='color: #74b9ff; margin-bottom: 1.5rem; font-size: 1.5rem; font-weight: 800; text-transform: uppercase; letter-spacing: 1px;'>👩‍💻 About</h4>
                <p style='margin: 0; font-size: 1.5rem; font-weight: 800; color: white; text-shadow: 1px 1px 3px rgba(0,0,0,0.5);'>Ayushi Rathour — Biotechnology Graduate | Exploring AI in Healthcare</p>
                <p style='margin: 20px 0; color: #74b9ff; font-weight: 700; font-size: 1.2rem; text-transform: uppercase; letter-spacing: 1px;'>🚀 Powered by TensorFlow & Modern Web Technologies</p>
            </div>
            
            <div style='margin-top: 3rem; padding-top: 3rem; border-top: 2px solid rgba(255,255,255,0.15);'>
                <p style='color: #bdc3c7; font-size: 1.1rem; line-height: 1.9; font-weight: 500;'>
                    <strong>PneumoDetect AI v2.0</strong> | © 2025 Ayushi Rathour
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("❌ Model failed to load. Please check the model file.")
