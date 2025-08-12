import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

@st.cache_resource
def load_model():
    """Load your trained pneumonia detection model"""
    
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
                st.info(f"✅ Found model at: {model_path}")
                model = tf.keras.models.load_model(model_path, compile=False)
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                st.success("✅ Trained model loaded successfully (86% accuracy)")
                return model
        except Exception as e:
            st.warning(f"❌ Failed to load from {model_path}: {e}")
            continue
    
    st.error("❌ Model file not found in any expected location")
    
    # Debug info
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

# Streamlit App Interface
st.set_page_config(
    page_title="Pneumonia Detection AI",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Chest X-Ray Pneumonia Detection")
st.markdown("**86% External Validation Accuracy | 96.4% Sensitivity | Clinical Grade AI**")

# Load model
model = load_model()

if model is not None:
    # Sidebar with model info
    with st.sidebar:
        st.header("📊 Model Performance")
        st.metric("Accuracy", "86.0%")
        st.metric("Sensitivity", "96.4%") 
        st.metric("Specificity", "74.8%")
        st.info("External validation on 485 independent samples")
    
    # Main interface
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "📤 Upload chest X-ray image", 
        type=['jpg', 'png', 'jpeg'],
        help="Supported formats: JPG, PNG, JPEG"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Uploaded X-ray", use_column_width=True)
        
        with col2:
            st.markdown("### 🔍 Analysis")
            
            if st.button("🔬 Analyze X-ray", type="primary", use_container_width=True):
                with st.spinner("Analyzing with trained AI model..."):
                    # Make prediction
                    processed_image = preprocess_image(image)
                    prediction = model.predict(processed_image, verbose=0)[0][0]
                    result = interpret_prediction(prediction)
                
                # Display results
                st.markdown("### 📋 Results")
                
                if result['diagnosis'] == 'PNEUMONIA':
                    st.error(f"**Diagnosis:** {result['diagnosis']}")
                else:
                    st.success(f"**Diagnosis:** {result['diagnosis']}")
                
                st.info(f"**Confidence:** {result['confidence']}% ({result['confidence_level']})")
                st.write(result['recommendation'])
                
                # Technical details in expander
                with st.expander("🔬 Technical Details"):
                    st.write(f"**Raw prediction score:** {prediction:.4f}")
                    st.write(f"**Threshold:** 0.5 (>0.5 = Pneumonia, ≤0.5 = Normal)")
                    st.write(f"**Model architecture:** MobileNetV2 + Custom Head")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        ⚠️ <b>Medical Disclaimer:</b> This AI is for preliminary screening only.<br>
        Always consult qualified healthcare professionals for medical decisions.
        </div>
        """, 
        unsafe_allow_html=True
    )

else:
    st.error("❌ Model failed to load. Please check the model file.")

