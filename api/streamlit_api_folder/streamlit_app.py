import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

@st.cache_resource
def load_model():
    """Load your trained pneumonia detection model"""
    
    # Multiple possible paths to try (YOUR WORKING LOGIC - UNCHANGED)
    possible_paths = [
        'best_chest_xray_model.h5',
        './best_chest_xray_model.h5',
        'api/streamlit_api_folder/best_chest_xray_model.h5',
        '/mount/src/chest-xray-pneumonia-detection-ai/api/streamlit_api_folder/best_chest_xray_model.h5'
    ]
    
    for model_path in possible_paths:
        try:
            if os.path.exists(model_path):
                st.info(f"✅ Found model at: {model_path}")
                model = tf.keras.models.load_model(model_path, compile=False)
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                st.success("✅ Trained model loaded successfully")
                return model
        except Exception as e:
            st.warning(f"❌ Failed to load from {model_path}: {e}")
            continue
    
    st.error("❌ Model file not found in any expected location")
    st.write(f"🔍 Current working directory: {os.getcwd()}")
    st.write(f"🔍 Files in current directory: {os.listdir('.')}")
    
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
            background: linear-gradient(145deg, #ff4757, #ff3742);
            color: white;
            border: 3px solid #ff6b7a;
            padding: 2.5rem;
            border-radius: 25px;
            margin: 2rem 0;
            box-shadow: 0 20px 60px rgba(255, 71, 87, 0.4);
            animation: pulseAlert 2s infinite, slideInUp 0.8s ease-out;
            position: relative;
            overflow: hidden;
        '>
            <div style='position: relative; z-index: 2; text-align: center;'>
                <div style='font-size: 4rem; margin-bottom: 1rem; animation: bounce 2s infinite;'>🚨</div>
                <h1 style='margin: 0; font-size: 2.8rem; font-weight: 800; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>
                    PNEUMONIA DETECTED
                </h1>
                <div style='
                    background: rgba(255,255,255,0.25);
                    backdrop-filter: blur(10px);
                    border-radius: 20px;
                    padding: 25px;
                    margin: 2rem 0;
                    border: 2px solid rgba(255,255,255,0.3);
                '>
                    <div style='font-size: 4rem; font-weight: 900; margin: 0;'>{result['confidence']}%</div>
                    <div style='font-size: 1.4rem; margin: 10px 0; font-weight: 600;'>
                        Confidence Level: {result['confidence_level']}
                    </div>
                </div>
                <div style='font-size: 1.3rem; line-height: 1.8; padding: 20px; border-radius: 15px; font-weight: 500;'>
                    {result['recommendation']}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='
            background: linear-gradient(145deg, #2ed573, #1e9c57);
            color: white;
            border: 3px solid #5af593;
            padding: 2.5rem;
            border-radius: 25px;
            margin: 2rem 0;
            box-shadow: 0 20px 60px rgba(46, 213, 115, 0.4);
            animation: pulseSuccess 3s infinite, slideInUp 0.8s ease-out;
            position: relative;
            overflow: hidden;
        '>
            <div style='position: relative; z-index: 2; text-align: center;'>
                <div style='font-size: 4rem; margin-bottom: 1rem; animation: heartbeat 2s infinite;'>✅</div>
                <h1 style='margin: 0; font-size: 2.8rem; font-weight: 800; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);'>
                    NORMAL CHEST X-RAY
                </h1>
                <div style='
                    background: rgba(255,255,255,0.25);
                    backdrop-filter: blur(10px);
                    border-radius: 20px;
                    padding: 25px;
                    margin: 2rem 0;
                    border: 2px solid rgba(255,255,255,0.3);
                '>
                    <div style='font-size: 4rem; font-weight: 900; margin: 0;'>{result['confidence']}%</div>
                    <div style='font-size: 1.4rem; margin: 10px 0; font-weight: 600;'>
                        Confidence Level: {result['confidence_level']}
                    </div>
                </div>
                <div style='font-size: 1.3rem; line-height: 1.8; padding: 20px; border-radius: 15px; font-weight: 500;'>
                    {result['recommendation']}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Technical Analysis Section
    st.markdown(f"""
    <div style='
        background: linear-gradient(145deg, #2c3e50, #34495e);
        color: white;
        padding: 30px;
        border-radius: 25px;
        margin: 30px 0;
        position: relative;
        overflow: hidden;
        border: 2px solid #3498db;
    '>
        <h3 style='color: #3498db; margin-bottom: 25px; font-size: 1.8rem; text-align: center; font-weight: 700;'>
            🔬 Technical Analysis Dashboard
        </h3>
        <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px;'>
            <div style='background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px; text-align: center;'>
                <h4 style='color: #74b9ff; margin: 0 0 10px 0;'>Raw Score</h4>
                <p style='color: #ddd; font-size: 1.2rem; font-weight: 700;'>{prediction:.4f}</p>
            </div>
            <div style='background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px; text-align: center;'>
                <h4 style='color: #55a3ff; margin: 0 0 10px 0;'>Threshold</h4>
                <p style='color: #ddd; font-size: 1.2rem; font-weight: 700;'>0.5</p>
            </div>
            <div style='background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px; text-align: center;'>
                <h4 style='color: #a29bfe; margin: 0 0 10px 0;'>Architecture</h4>
                <p style='color: #ddd; font-size: 1rem; font-weight: 600;'>MobileNetV2</p>
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

# PREMIUM CSS WITH ACETERNITY-STYLE GRADIENT BACKGROUND
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit elements */
    .css-1d391kg, [data-testid="stSidebar"] {
        display: none !important;
    }
    
    /* Full screen animated gradient background - Aceternity style */
    .main, .stApp {
        background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #f5576c, #4facfe, #00f2fe);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        min-height: 100vh;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Floating particles effect */
    .main::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(circle at 20% 80%, rgba(255,255,255,0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(255,255,255,0.1) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(255,255,255,0.05) 0%, transparent 50%);
        animation: particleFloat 20s ease-in-out infinite;
        pointer-events: none;
        z-index: -1;
    }
    
    @keyframes particleFloat {
        0%, 100% { opacity: 0.3; transform: translateY(0px); }
        50% { opacity: 0.6; transform: translateY(-20px); }
    }
    
    /* Remove default Streamlit padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Title section without box */
    .title-section {
        text-align: center;
        padding: 3rem 0;
        margin-bottom: 2rem;
    }
    
    .main-title {
        font-size: 4.5rem;
        font-weight: 800;
        color: white;
        text-shadow: 3px 3px 15px rgba(0,0,0,0.5);
        margin-bottom: 1rem;
        animation: titleGlow 3s ease-in-out infinite alternate;
    }
    
    @keyframes titleGlow {
        from { text-shadow: 3px 3px 15px rgba(0,0,0,0.5); }
        to { text-shadow: 3px 3px 25px rgba(255,255,255,0.3); }
    }
    
    .subtitle {
        font-size: 1.5rem;
        color: rgba(255,255,255,0.9);
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    .developer-badge {
        background: rgba(255,255,255,0.25);
        backdrop-filter: blur(15px);
        padding: 20px 40px;
        border-radius: 50px;
        display: inline-block;
        color: white;
        font-weight: 700;
        font-size: 1.2rem;
        border: 2px solid rgba(255,255,255,0.3);
        animation: developerPulse 4s ease-in-out infinite;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    @keyframes developerPulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    /* Responsive stats grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 25px;
        margin: 3rem 0;
        padding: 0 1rem;
    }
    
    .stat-card {
        background: rgba(255,255,255,0.95);
        backdrop-filter: blur(20px);
        padding: 2rem;
        border-radius: 25px;
        text-align: center;
        box-shadow: 0 20px 50px rgba(0,0,0,0.15);
        border: 2px solid rgba(255,255,255,0.3);
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }
    
    .stat-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(52,152,219,0.2), transparent);
        transition: left 0.6s;
    }
    
    .stat-card:hover::before {
        left: 100%;
    }
    
    .stat-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 30px 70px rgba(0,0,0,0.25);
    }
    
    .stat-number {
        font-size: 3.5rem;
        font-weight: 800;
        color: #2c3e50;
        margin: 0;
        position: relative;
        z-index: 2;
    }
    
    .stat-label {
        color: #7f8c8d;
        font-weight: 600;
        margin-top: 10px;
        font-size: 1.2rem;
        position: relative;
        z-index: 2;
    }
    
    /* Upload container */
    .upload-container {
        background: rgba(255,255,255,0.95);
        backdrop-filter: blur(20px);
        padding: 2rem;
        border-radius: 25px;
        margin: 2rem auto;
        max-width: 800px;
        box-shadow: 0 20px 50px rgba(0,0,0,0.15);
        border: 3px dashed #3498db;
    }
    
    .upload-container:hover {
        border-color: #2980b9;
        transform: translateY(-5px);
        box-shadow: 0 25px 60px rgba(0,0,0,0.2);
    }
    
    /* Image preview container */
    .image-preview {
        background: rgba(255,255,255,0.98);
        backdrop-filter: blur(20px);
        padding: 25px;
        border-radius: 25px;
        margin: 25px auto;
        max-width: 600px;
        box-shadow: 0 15px 40px rgba(0,0,0,0.15);
        border: 2px solid rgba(52,152,219,0.3);
    }
    
    /* Animation keyframes */
    @keyframes pulseAlert {
        0%, 100% { box-shadow: 0 20px 60px rgba(255, 71, 87, 0.4); }
        50% { box-shadow: 0 25px 80px rgba(255, 71, 87, 0.6); }
    }
    
    @keyframes pulseSuccess {
        0%, 100% { box-shadow: 0 20px 60px rgba(46, 213, 115, 0.4); }
        50% { box-shadow: 0 25px 80px rgba(46, 213, 115, 0.6); }
    }
    
    @keyframes slideInUp {
        from { opacity: 0; transform: translateY(50px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-15px); }
        60% { transform: translateY(-7px); }
    }
    
    @keyframes heartbeat {
        0% { transform: scale(1); }
        14% { transform: scale(1.3); }
        28% { transform: scale(1); }
        42% { transform: scale(1.3); }
        70% { transform: scale(1); }
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(145deg, #3498db, #2980b9);
        color: white;
        border: none;
        padding: 15px 35px;
        border-radius: 25px;
        font-weight: 700;
        font-size: 1.1rem;
        box-shadow: 0 10px 25px rgba(52,152,219,0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(52,152,219,0.5);
    }
    
    /* Hide file uploader label */
    .stFileUploader > label {
        display: none;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-title { font-size: 3rem; }
        .stats-grid { grid-template-columns: 1fr; }
        .stat-card { padding: 1.5rem; }
        .developer-badge { font-size: 1rem; padding: 15px 30px; }
    }
</style>
""", unsafe_allow_html=True)

# TITLE SECTION (NO CONTAINER BOX)
st.markdown("""
<div class='title-section'>
    <div style='font-size: 6rem; margin-bottom: 1rem; animation: bounce 3s infinite;'>🫁</div>
    <h1 class='main-title'>PneumoDetect AI</h1>
    <p class='subtitle'>🔬 Advanced Chest X-Ray Analysis | 🎯 Clinical-Grade Artificial Intelligence</p>
    <div class='developer-badge'>
        👩‍💻 Developed by Ayushi Rathour - Biotechnology Graduate | Exploring AI in Healthcare
    </div>
</div>
""", unsafe_allow_html=True)

# RESPONSIVE PERFORMANCE STATISTICS (MENTIONED ONLY ONCE)
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

# LOAD MODEL
model = load_model()

if model is not None:
    # UPLOAD SECTION WITH PROPER BORDER
    st.markdown("""
    <div class='upload-container'>
        <h2 style='text-align: center; color: #2c3e50; margin-bottom: 1rem; font-weight: 700;'>
            📤 Upload Chest X-Ray for AI Analysis
        </h2>
        <p style='text-align: center; color: #7f8c8d; font-size: 1.1rem; margin-bottom: 1rem;'>
            Supported formats: JPG, PNG, JPEG | Maximum file size: 200MB
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Select chest X-ray image", 
        type=['jpg', 'png', 'jpeg'],
        help="Upload a chest X-ray image for AI-powered pneumonia detection"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        # IMAGE PREVIEW WITH PROPER CONTAINER
        st.markdown("""<div class='image-preview'>""", unsafe_allow_html=True)
        st.image(image, caption="📸 Uploaded Chest X-Ray - Ready for AI Analysis", use_column_width=True)
        st.markdown("""</div>""", unsafe_allow_html=True)

        # ANALYZE BUTTON
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔬 Analyze X-ray with AI", type="primary", use_container_width=True):
                with st.spinner("🧠 Advanced AI analysis in progress..."):
                    processed_image = preprocess_image(image)
                    prediction = model.predict(processed_image, verbose=0)[0][0]
                    result = interpret_prediction(prediction)

                display_premium_results(result, prediction, image)

    # FOOTER WITH PROPER HTML RENDERING
    st.markdown("""
    <div style='
        background: linear-gradient(145deg, rgba(44, 62, 80, 0.95), rgba(52, 73, 94, 0.95));
        backdrop-filter: blur(20px);
        color: white;
        padding: 3rem 2rem;
        border-radius: 25px;
        margin-top: 4rem;
        text-align: center;
        border: 2px solid rgba(255,255,255,0.1);
    '>
        <div style='margin-bottom: 2rem;'>
            <h3 style='color: #3498db; margin-bottom: 1rem; font-size: 1.8rem; font-weight: 700;'>⚠️ Medical Disclaimer</h3>
            <p style='font-size: 1.1rem; line-height: 1.8; margin-bottom: 2rem;'>
                This AI system is designed for <strong>preliminary screening purposes only</strong>.<br>
                Always consult qualified healthcare professionals for definitive medical decisions.<br>
                This tool serves as a supportive diagnostic aid, not a replacement for professional medical judgment.
            </p>
        </div>
        
        <div style='border-top: 2px solid rgba(255,255,255,0.2); padding-top: 2rem;'>
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 2rem; text-align: left;'>
                <div style='background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 15px;'>
                    <h4 style='color: #74b9ff; margin-bottom: 1rem;'>👩‍💻 Developer</h4>
                    <p style='margin: 0; font-size: 1.1rem; font-weight: 700;'><strong>Ayushi Rathour</strong></p>
                    <p style='margin: 5px 0; color: #ddd;'>Biotechnology Graduate</p>
                    <p style='margin: 5px 0; color: #ddd;'>Exploring AI in Healthcare</p>
                    <p style='margin: 10px 0; color: #74b9ff; font-weight: 600;'>🚀 Bridging Biology & Technology</p>
                </div>
                
                <div style='background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 15px;'>
                    <h4 style='color: #55dd88; margin-bottom: 1rem;'>🔬 Technical Stack</h4>
                    <p style='margin: 5px 0;'>• TensorFlow & Deep Learning</p>
                    <p style='margin: 5px 0;'>• Streamlit & Python</p>
                    <p style='margin: 5px 0;'>• Computer Vision & CNN</p>
                    <p style='margin: 5px 0;'>• Medical Image Analysis</p>
                </div>
                
                <div style='background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 15px;'>
                    <h4 style='color: #ff7675; margin-bottom: 1rem;'>🏥 Healthcare Impact</h4>
                    <p style='margin: 5px 0;'>• Early Pneumonia Detection</p>
                    <p style='margin: 5px 0;'>• Clinical Decision Support</p>
                    <p style='margin: 5px 0;'>• Accessible AI Diagnostics</p>
                    <p style='margin: 5px 0;'>• Evidence-Based Medicine</p>
                </div>
            </div>
            
            <div style='text-align: center; margin-top: 2rem; padding-top: 2rem; border-top: 1px solid rgba(255,255,255,0.1);'>
                <p style='color: #bdc3c7; font-size: 1rem; line-height: 1.6;'>
                    🏥 Developed with ❤️ for Healthcare Innovation | 🔬 Powered by TensorFlow & Streamlit<br>
                    <strong>PneumoDetect AI v2.0</strong> | © 2024 Ayushi Rathour | Biotechnology × Artificial Intelligence
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("❌ Model failed to load. Please check the model file.")
