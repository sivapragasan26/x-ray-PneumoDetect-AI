import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Page Configuration
st.set_page_config(
    page_title="PneumoDetect AI | Ayushi Rathour - Healthcare Innovation",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS Styling - PROPERLY WRAPPED
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
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
    
    /* Hide Streamlit elements */
    .css-1d391kg, [data-testid="stSidebar"] {
        display: none !important;
    }
    
    /* Title Section */
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
    
    /* Stats Grid */
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
    
    /* File Upload Styling */
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
    
    /* Button Styling */
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
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-title { font-size: 3.5rem; letter-spacing: 2px; }
        .stats-grid { grid-template-columns: 1fr; gap: 20px; }
        .stat-card { padding: 2rem; }
        .stFileUploader { margin: 2rem 1rem; }
    }
</style>
""", unsafe_allow_html=True)

# Model Loading Functions
@st.cache_resource
def load_model():
    """Load your trained pneumonia detection model"""
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

# Title Section
st.markdown("""
<div class='title-section'>
    <div style='font-size: 7rem; margin-bottom: 2rem; filter: drop-shadow(0 0 30px rgba(255,255,255,0.5));'>🫁</div>
    <h1 class='main-title'>PneumoDetect AI</h1>
    <p class='subtitle'>Advanced Chest X-Ray Analysis | Clinical-Grade Artificial Intelligence</p>
    <div class='tagline'>Fast. Accurate. Reliable.<br>AI-powered pneumonia detection in just 2.5 seconds.</div>
</div>
""", unsafe_allow_html=True)

# Performance Statistics
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

# Load Model
model = load_model()

if model is not None:
    # Upload Section
    st.markdown("""
    <div style='text-align: center; margin: 2rem 0;'>
        <h2 style='color: white; font-weight: 700; margin-bottom: 1rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);'>
            📤 Upload Chest X-Ray for Instant AI Analysis
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload Chest X-Ray for Instant AI Analysis",
        type=['jpg', 'png', 'jpeg'],
        help="Drag & drop your file or click to browse"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="📸 Uploaded Chest X-Ray - Ready for AI Analysis", use_container_width=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔬 Analyze X-ray with AI", type="primary"):
                with st.spinner("🧠 Advanced AI analysis in progress..."):
                    processed_image = preprocess_image(image)
                    prediction = model.predict(processed_image, verbose=0)[0][0]
                    result = interpret_prediction(prediction)

                # Display Results
                if result['diagnosis'] == 'PNEUMONIA':
                    st.error(f"🩺 **Diagnosis Result: Pneumonia Detected**\n\n**{result['confidence']}% confidence ({result['confidence_level']} level)**\n\n{result['recommendation']}")
                else:
                    st.success(f"✅ **Diagnosis Result: Normal Chest X-Ray**\n\n**{result['confidence']}% confidence ({result['confidence_level']} level)**\n\n{result['recommendation']}")

                # Technical Summary
                st.info(f"🔬 **Technical Summary**\n\nModel Architecture: MobileNetV2 | Threshold: 0.5 | Raw Score: {prediction:.4f}")

    # Footer
    st.markdown("""
    <div style='
        background: linear-gradient(135deg, rgba(44, 62, 80, 0.95), rgba(52, 73, 94, 0.95));
        color: white;
        padding: 3rem 2rem;
        border-radius: 30px;
        margin-top: 5rem;
        text-align: center;
        border: 3px solid rgba(255,255,255,0.2);
    '>
        <h3 style='color: #74b9ff; margin-bottom: 1.5rem;'>⚠ Medical Disclaimer</h3>
        <p style='margin-bottom: 2rem;'>This AI tool is intended for preliminary screening only. Always seek advice from qualified healthcare professionals before making medical decisions.</p>
        
        <div style='background: rgba(255,255,255,0.15); padding: 2rem; border-radius: 25px; margin: 1.5rem 0;'>
            <h4 style='color: #74b9ff;'>👩‍💻 Ayushi Rathour</h4>
            <p>Biotechnology Graduate | Exploring AI in Healthcare</p>
            <p style='color: #74b9ff; font-weight: 700;'>🚀 Powered by TensorFlow & Modern Web Technologies</p>
        </div>
        
        <p style='color: #bdc3c7;'><strong>PneumoDetect AI v2.0</strong> | © 2025 Ayushi Rathour</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("❌ Model failed to load. Please check the model file.")
