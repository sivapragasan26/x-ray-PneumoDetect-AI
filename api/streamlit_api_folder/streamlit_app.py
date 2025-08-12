# streamlit_app.py

import streamlit as st
from PIL import Image
import time

# ======== IMPORT MODEL LOGIC ========
import tensorflow as tf
from PIL import Image as PILImage
import numpy as np
import os

# MODEL LOADING WITH MULTIPLE PATHS
def load_pneumonia_model():
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
                st.write(f"✅ Loading model from: `{model_path}`")
                model = tf.keras.models.load_model(model_path, compile=False)
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                return model
        except Exception as e:
            st.write(f"❌ Failed from `{model_path}`: {str(e)}")
    st.error("❌ Could not load model from any path")
    return None

# IMAGE PREPROCESSING
def preprocess_image(image_input):
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
    if prediction_score > 0.5:
        diagnosis = "PNEUMONIA"
        confidence = float(prediction_score * 100)
        if confidence >= 80:
            level = "High"
            rec = "🚨 Strong indication of pneumonia. Seek immediate medical attention."
        elif confidence >= 60:
            level = "Moderate"
            rec = "⚠️ Moderate indication of pneumonia. Medical review recommended."
        else:
            level = "Low"
            rec = "💡 Possible pneumonia detected. Further examination advised."
    else:
        diagnosis = "NORMAL"
        confidence = float((1 - prediction_score) * 100)
        if confidence >= 80:
            level = "High"
            rec = "✅ No signs of pneumonia detected. Chest X-ray appears normal."
        elif confidence >= 60:
            level = "Moderate"
            rec = "👍 Likely normal chest X-ray. Routine follow-up if symptoms persist."
        else:
            level = "Low"
            rec = "🤔 Unclear result. Manual review by radiologist recommended."
    return {
        "diagnosis": diagnosis,
        "confidence": round(confidence, 2),
        "confidence_level": level,
        "recommendation": rec,
        "raw_score": float(prediction_score),
        "threshold": 0.5,
        "model_architecture": "MobileNetV2"
    }

# MAIN PREDICTION FUNCTION
def predict_pneumonia(image_input, model=None):
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

# MODEL SPECS
MODEL_SPECS = {
    "name": "PneumoDetect AI",
    "version": "v2.0",
    "architecture": "MobileNetV2",
    "accuracy": 86.0,
    "sensitivity": 96.4,
    "avg_prediction_time": "2.5 seconds",
    "developer": "Ayushi Rathour",
    "supported_formats": ["JPG", "JPEG", "PNG"],
    "max_file_size_mb": 200
}

# ======== STREAMLIT UI ========

st.set_page_config(page_title="PneumoDetect AI", page_icon="🫁", layout="centered")
st.title("🫁 PneumoDetect AI")
st.markdown("### Advanced Chest X-Ray Analysis | Clinical-Grade AI")

# Metrics Section
col1, col2, col3 = st.columns(3)
col1.metric("🎯 Accuracy", f"{MODEL_SPECS['accuracy']}%")
col2.metric("🔍 Sensitivity", f"{MODEL_SPECS['sensitivity']}%")
col3.metric("⏱ Avg. Time", MODEL_SPECS['avg_prediction_time'])

st.divider()

# Upload Section
uploaded_file = st.file_uploader("📤 Upload Chest X-Ray", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📸 Uploaded Chest X-Ray", use_column_width=True)

    if st.button("🔬 Analyze X-ray with AI"):
        with st.spinner("Analyzing... Please wait"):
            model = load_pneumonia_model()
            start_time = time.time()
            prediction_data = predict_pneumonia(image, model)
            end_time = time.time()

        if prediction_data["success"]:
            res = prediction_data["result"]
            st.subheader(f"Diagnosis: {res['diagnosis']}")
            st.write(f"**Confidence:** {res['confidence']}% ({res['confidence_level']})")
            st.write(res["recommendation"])
            st.markdown("---")
            st.caption(f"Raw Score: {res['raw_score']:.4f} | Threshold: {res['threshold']} | Architecture: {res['model_architecture']}")
            st.caption(f"⏱ Prediction Time: {end_time - start_time:.2f} sec")
        else:
            st.error(f"Error: {prediction_data['error']}")

st.markdown("---")
st.caption("⚠️ This AI system is for preliminary screening purposes only. Always consult a healthcare professional.")
st.caption(f"Developed by {MODEL_SPECS['developer']} | Powered by TensorFlow")
