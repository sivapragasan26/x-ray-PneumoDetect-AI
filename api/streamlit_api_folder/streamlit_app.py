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

# FIXED CSS - All Issues Resolved
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    /* Hide Streamlit Elements */
    .css-1d391kg, [data-testid="stSidebar"] {
        display: none !important;
    }
    
    /* Main Background - FIXED GRADIENT */
    .main, .stApp {
        background: linear-gradient(300deg, #211c6a, #17594a, #08045b, #264422, #b7b73d);
        background-size: 300% 300%;
        animation: gradient-animation 25s ease infinite;
        min-height: 100vh;
        font-family: 'Inter', sans-serif;
    }
    
    @keyframes gradient-animation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Block Container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Enhanced CSS Specificity for Streamlit */
    div[data-testid="stMarkdownContainer"] h1 {
        font-size: 5rem !important;
        font-weight: 900 !important;
        color: white !important;
        text-shadow: 3px 3px 15px rgba(0,0,0,0.5) !important;
        text-align: center !important;
        margin-bottom: 1.5rem !important;
        letter-spacing: 3px !important;
        text-transform: uppercase !important;
    }
    
    div[data-testid="stMarkdownContainer"] h2 {
        color: white !important;
        font-weight: 700 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5) !important;
        text-align: center !important;
    }
    
    div[data-testid="stMarkdownContainer"] p {
        color: rgba(255,255,255,0.95) !important;
        font-weight: 500 !important;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.3) !important;
        text-align: center !important;
    }
    
    /* File Uploader Styling */
    .stFileUploader {
        background: rgba(255,255,255,0.98) !important;
        backdrop-filter: blur(25px) !important;
        border-radius: 30px !important;
        margin: 3rem auto !important;
        max-width: 800px !important;
        box-shadow: 0 25px 70px rgba(0,0,0,0.2) !important;
        border: 3px dashed #3498db !important;
        padding: 2rem !important;
    }
    
    .stFileUploader label {
        color: #3498db !important;
        font-size: 1.3rem !important;
        font-weight: 700 !important;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        padding: 18px 40px !important;
        border-radius: 30px !important;
        font-weight: 800 !important;
        font-size: 1.2rem !important;
        width: 100% !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-5px) !important;
        box-shadow: 0 20px 45px rgba(102,126,234,0.6) !important;
    }
    
    /* WORKING FOOTER STYLES - HIGH SPECIFICITY */
    .custom-footer {
        background: linear-gradient(135deg, rgba(44, 62, 80, 0.95), rgba(52, 73, 94, 0.95)) !important;
        backdrop-filter: blur(25px) !important;
        color: white !important;
        padding: 4rem 2rem !important;
        border-radius: 30px !important;
        margin-top: 5rem !important;
        text-align: center !important;
        border: 3px solid rgba(255,255,255,0.2) !important;
        box-shadow: 0 20px 50px rgba(0,0,0,0.3) !important;
    }
    
    .custom-footer h3 {
        color: #74b9ff !important;
        margin-bottom: 1.5rem !important;
        font-size: 2rem !important;
        font-weight: 800 !important;
    }
    
    .custom-footer .developer-card {
        background: rgba(255,255,255,0.15) !important;
        padding: 2.5rem !important;
        border-radius: 25px !important;
        margin: 1.5rem 0 !important;
        border: 2px solid rgba(255,255,255,0.3) !important;
    }
    
    .custom-footer .developer-name {
        color: white !important;
        font-size: 1.6rem !important;
        font-weight: 800 !important;
        margin-bottom: 8px !important;
    }
    
    .custom-footer .developer-title {
        color: #ddd !important;
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        margin: 12px 0 !important;
    }
    
    .custom-footer .tech-stack {
        color: #74b9ff !important;
        font-weight: 700 !important;
        font-size: 1.2rem !important;
        margin: 20px 0 !important;
    }
    
    .custom-footer .copyright {
        color: #bdc3c7 !important;
        font-size: 1.1rem !important;
        margin-top: 3rem !important;
        padding-top: 3rem !important;
        border-top: 2px solid rgba(255,255,255,0.15) !important;
    }
</style>
""", unsafe_allow_html=True)

# Model Functions (YOUR EXISTING LOGIC - VERIFIED)
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

# WORKING HEADER - USING STREAMLIT COMPONENTS
st.markdown('<div style="font-size: 7rem; text-align: center; margin-bottom: 2rem; filter: drop-shadow(0 0 30px rgba(255,255,255,0.5));">🫁</div>', unsafe_allow_html=True)

st.markdown("# PneumoDetect AI")
st.markdown("## Advanced Chest X-Ray Analysis | Clinical-Grade Artificial Intelligence")
st.markdown("### Fast. Accurate. Reliable. AI-powered pneumonia detection in just 2.5 seconds.")

# STATS USING STREAMLIT COLUMNS
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🎯 Accuracy", "86%")
with col2:
    st.metric("🔍 Sensitivity", "96.4%") 
with col3:
    st.metric("⏱ Avg. Time", "2.5 sec")

# Load Model
model = load_model()

if model is not None:
    st.markdown("## 📤 Upload Chest X-Ray for Instant AI Analysis")
    
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

    # WORKING FOOTER - USING HIGH SPECIFICITY CSS
    st.markdown("""
    <div class="custom-footer">
        <h3>⚠ Medical Disclaimer</h3>
        <p style="margin-bottom: 2rem;">This AI tool is intended for preliminary screening only. Always seek advice from qualified healthcare professionals before making medical decisions.</p>
        
        <div class="developer-card">
            <div class="developer-name">👩‍💻 Ayushi Rathour</div>
            <div class="developer-title">Biotechnology Graduate | Exploring AI in Healthcare</div>
            <div class="tech-stack">🚀 Powered by TensorFlow & Modern Web Technologies</div>
        </div>
        
        <div class="copyright">
            <strong>PneumoDetect AI v2.0</strong> | © 2025 Ayushi Rathour
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("❌ Model failed to load. Please check the model file.")
