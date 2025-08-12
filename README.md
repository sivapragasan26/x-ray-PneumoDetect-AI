# 🏥 Chest X-Ray Pneumonia Detection - Externally Validated AI System

[![Python](https://img.shields.io/badge/Python-3.11.9-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-orange.svg)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live_Demo-red.svg)](https://pneumodetectai.streamlit.app/)
[![External Validation](https://img.shields.io/badge/External_Validation-485_Samples-success.svg)]()
[![Model](https://img.shields.io/badge/🤗_Model-Live_on_HuggingFace-yellow.svg)](https://huggingface.co/ayushirathour/chest-xray-pneumonia-detection)

### Deep Learning Model with Clinical-Grade Performance & Real-World Validation

**A professionally validated medical AI system for pneumonia detection with comprehensive external validation demonstrating real-world generalization.**

*Developed by **Ayushi Rathour***

---

## 📋 **Quick Navigation**
- [🌐 Live Demo](#-live-application)
- [🎯 Project Overview](#-project-overview)
- [🤖 Pre-trained Model](#-pre-trained-model-access)
- [📊 Performance Results](#-performance-results---dual-validation-approach)
- [🚀 Quick Start](#-quick-start)
- [🧠 Technical Architecture](#-technical-architecture)
- [🏗️ Repository Structure](#️-repository-structure)
- [📊 Dataset Information](#-dataset-information)
- [⚠️ Limitations](#️-limitations--important-considerations)
- [🎯 Responsible AI](#-responsible-ai-disclaimer)
- [📞 Contact](#-contact--collaboration)

---

## 🌐 **Live Application**
![PneumoDetect AI Demo](demo/pneumodetect_demo.gif)

### **🎯 Interactive Web Interface**
**🔗 [PneumoDetect AI - Live Demo](https://pneumodetectai.streamlit.app/)**

Experience the complete AI-powered pneumonia detection system with:
- **Professional Medical Interface** - Clean, clinical-grade glassmorphic design
- **Real-time Analysis** - Upload chest X-rays and get instant results in 2.5 seconds
- **Clinical Metrics Display** - 86% accuracy, 96.4% sensitivity, 1K+ training images
- **Comprehensive Reporting** - Detailed technical analysis with confidence scores
- **Mobile Responsive** - Works seamlessly across all devices
- **Privacy Focused** - No data storage, secure local processing

> **⚡ TL;DR:** AI system detects pneumonia in chest X-rays with 86% accuracy on external validation (485 samples). Try the live demo above or download the pre-trained model from HuggingFace.

---

## 🎯 **Project Overview**

This project implements an end-to-end deep learning system for automated pneumonia detection in chest X-ray images. The system underwent rigorous **internal and external validation** to ensure real-world applicability, achieving strong performance across independent datasets.

### **🏆 Key Achievements**
- ✅ **Comprehensive External Validation** on 485 independent samples
- ✅ **Strong Generalization** with only 8.8% accuracy drop on external data
- ✅ **Clinical-Grade Sensitivity** of 96.4% for pneumonia detection
- ✅ **Complete ML Pipeline** from training to deployment
- ✅ **Production-Ready Web Application** with Streamlit frontend
- ✅ **Professional Model Hosting** on Hugging Face Hub
- ✅ **Live Deployment** accessible globally via web interface

### **🎯 Performance Summary**

| **External Validation** | **Clinical Significance** |
|------------------------|---------------------------|
| **86% Accuracy** | Strong generalization (8.8% drop) |
| **96.4% Sensitivity** | Excellent for screening applications |
| **74.8% Specificity** | Acceptable false positive rate |
| **485 Samples** | Statistically significant validation |
| **1K+ Training Images** | Robust model training dataset |

---

## 🤖 **Pre-trained Model Access**

**The trained model is hosted externally due to file size limitations:**

**🔗 [ayushirathour/chest-xray-pneumonia-detection](https://huggingface.co/ayushirathour/chest-xray-pneumonia-detection)**

### **Model Download Options:**

```python
# Option 1: Download from Hugging Face Hub
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(
    repo_id="ayushirathour/chest-xray-pneumonia-detection", 
    filename="best_chest_xray_model.h5"
)

# Option 2: Load and use the model
import tensorflow as tf
model = tf.keras.models.load_model(model_path)
```

---

## 📊 **Performance Results - Dual Validation Approach**

### **🔬 Validation Methodology**

This system underwent **two-tier validation** to ensure reliable real-world performance:

| Validation Type | Dataset Source | Sample Size | Purpose |
|----------------|---------------|-------------|---------|
| **Internal Validation** | [Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) | 269 samples | Model development & tuning |
| **External Validation** | [Independent Pneumonia Dataset](https://www.kaggle.com/datasets/iamtanmayshukla/pneumonia-radiography-dataset) | 485 samples | Real-world generalization testing |

### **📈 Comprehensive Performance Metrics**

| Metric | Internal Validation | External Validation | Performance Drop | Clinical Significance |
|--------|-------------------|-------------------|------------------|----------------------|
| **Accuracy** | 94.8% | 86.0% | 8.8% ↓ | ✅ Good generalization |
| **Sensitivity** | 89.6% | 96.4% | 6.8% ↑ | ✅ Improved detection |
| **Specificity** | 100.0% | 74.8% | 25.2% ↓ | ⚠️ More false positives |
| **Precision** | 100.0% | 80.4% | 19.6% ↓ | ✅ Acceptable accuracy |
| **F1-Score** | 94.5% | 87.7% | 6.8% ↓ | ✅ Balanced performance |
| **ROC-AUC** | 98.8% | 96.4% | 2.4% ↓ | ✅ Excellent discrimination |

### **🏥 Clinical Interpretation**
- **High External Sensitivity (96.4%):** Excellent for pneumonia screening - catches 96% of cases
- **Moderate External Specificity (74.8%):** Acceptable false positive rate for screening applications
- **Strong Generalization:** 8.8% accuracy drop indicates robust model performance across datasets
- **Clinical Applicability:** Performance suitable for preliminary screening and triage

---

## 🚀 **Quick Start**

### **🌐 Option 1: Use Live Web Application (Recommended)**

**🔗 [Try PneumoDetect AI Now](https://pneumodetectai.streamlit.app/)**
- No installation required
- Upload chest X-rays instantly  
- Get professional AI analysis in 2.5 seconds
- View external validation metrics
- Mobile-friendly responsive design

### **💻 Option 2: Run Locally**

```bash
# Clone repository
git clone https://github.com/ayushirathour/chest-xray-pneumonia-detection
cd chest-xray-pneumonia-detection

# Setup environment
python -m venv tf_env
tf_env\Scripts\activate # Windows
source tf_env/bin/activate # Linux/Mac

# Install dependencies
cd api
pip install -r requirements.txt

# Run Streamlit app
streamlit run streamlit_api_folder/streamlit_app.py
```

### **🤖 Option 3: Use Pre-trained Model in Code**

```python
from huggingface_hub import hf_hub_download
import tensorflow as tf
import numpy as np
from PIL import Image

# Download model from Hugging Face Hub
model_path = hf_hub_download(
    repo_id="ayushirathour/chest-xray-pneumonia-detection", 
    filename="best_chest_xray_model.h5"
)
model = tf.keras.models.load_model(model_path)

# Preprocess image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Make prediction
image_path = "your_xray_image.jpg"
processed_img = preprocess_image(image_path)
prediction = model.predict(processed_img)
result = "Pneumonia" if prediction[0][0] > 0.5 else "Normal"
confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]

print(f"Prediction: {result} (Confidence: {confidence:.2%})")
```

### **🔬 Option 4: Train Your Own Model**

```bash
# Prepare your dataset (download links provided in scripts)
python scripts/analyze_and_balance.py
python scripts/create_balanced_dataset.py

# Train the model
python scripts/train_model.py

# Evaluate performance
python scripts/evaluate_model.py
python scripts/external_validation.py
```

---

## 🧠 **Technical Architecture**

### **Model Design**
- **Architecture:** Transfer Learning with MobileNetV2
- **Input Processing:** 224×224 RGB normalization
- **Training Strategy:** Balanced dataset with data augmentation
- **Optimization:** Adam optimizer with early stopping
- **Validation:** Stratified train-test split + external validation

### **Web Application System**
- **Frontend:** Streamlit with glassmorphic design
- **Model Serving:** TensorFlow 2.19.0
- **Deployment:** Streamlit Cloud hosting
- **Response Format:** Real-time predictions with confidence scores
- **UI/UX:** Professional medical interface with animations

---

## 🏗️ **Repository Structure**

```
chest-xray-pneumonia-detection/
├── api/
│   ├── streamlit_api_folder/
│   │   └── streamlit_app.py      # Main Streamlit application
│   ├── requirements.txt          # Application dependencies  
│   ├── packages.txt             # System packages
│   └── best_chest_xray_model.h5 # Trained model weights
├── scripts/
│   ├── train_model.py           # Model training pipeline
│   ├── external_validation.py   # Comprehensive validation
│   ├── create_balanced_dataset.py # Dataset balancing
│   ├── analyze_and_balance.py   # Dataset analysis
│   └── evaluate_model.py        # Model evaluation
├── results/
│   ├── internal_validation/     # Internal validation results
│   │   ├── confusion_matrix.png
│   │   ├── roc_curve.png
│   │   ├── precision_recall_curve.png
│   │   └── evaluation_metrics.json
│   └── external_validation/     # External validation suite
│       ├── 1_enhanced_confusion_matrix.png
│       ├── 2_roc_curve.png
│       ├── 3_precision_recall_curve.png
│       ├── 4_performance_comparison.png
│       ├── 5_class_distribution.png
│       ├── 6_prediction_confidence_distribution.png
│       ├── 7_calibration_plot.png
│       └── 8_comprehensive_metrics_dashboard.png
├── demo/
│   └── pneumodetect_demo.gif    # Application demonstration
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

## 📊 **Dataset Information**

### **Data Access**

**Datasets are NOT included in this repository.** Download instructions:

#### **Training Data**
- **Source:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Manual Download Required:** ~1GB dataset
- **Preprocessing:** Use `scripts/analyze_and_balance.py` to prepare
- **Final Training Set:** 1K+ balanced images

#### **External Validation Data**
- **Source:** [Pneumonia Radiography Dataset](https://www.kaggle.com/datasets/iamtanmayshukla/pneumonia-radiography-dataset)
- **Size:** 485 samples (234 Normal, 251 Pneumonia)
- **Access:** Download separately and place in `external_validation_dataset/`

### **Preprocessing Steps**
1. Download datasets from provided links
2. Run `python scripts/analyze_and_balance.py`
3. Run `python scripts/create_balanced_dataset.py`
4. Processed data will be created locally (not committed to repo)

---

## ⚠️ **Limitations & Important Considerations**

### **🔬 Technical Limitations**
- **Training Data Size:** Limited to balanced subset of Kaggle dataset (1K+ images)
- **Image Quality Dependency:** Performance degrades with poor quality images
- **Binary Classification Only:** Cannot detect specific pneumonia types
- **Resolution Constraints:** Input images resized to 224×224 may lose detail

### **🏥 Clinical Limitations**
- **Screening Tool Only:** NOT suitable for definitive diagnosis
- **Professional Oversight Required:** All results must be reviewed by radiologists
- **Clinical Context Missing:** Cannot consider patient history or symptoms
- **Legal Liability:** Not approved by medical regulatory bodies

### **📊 Performance Limitations**
- **False Positive Rate:** 25.2% means 1 in 4 normal cases flagged incorrectly
- **Dataset Bias:** Limited testing across populations and imaging protocols
- **Temporal Stability:** No longitudinal validation over time

---

## 🎯 **Responsible AI Disclaimer**

### **⚠️ Important Notices**
- **Experimental System:** This is a research prototype, not a medical device
- **No Clinical Approval:** Not validated by FDA, CE, or other regulatory bodies
- **Educational Purpose:** Designed for learning and AI validation methodology
- **Professional Supervision:** Clinical use requires qualified radiologist oversight

### **Recommended Use Cases**
- ✅ **Academic Research:** Demonstrating external validation methodology
- ✅ **Educational Training:** Teaching AI in healthcare concepts
- ✅ **Technical Portfolio:** Showcasing end-to-end ML pipeline
- ❌ **Clinical Diagnosis:** Not suitable for patient care decisions
- ❌ **Regulatory Submission:** Not ready for medical device approval

---

## 📄 **Citation & License**

```bibtex
@misc{rathour2025pneumonia,
    title={Chest X-Ray Pneumonia Detection: Externally Validated AI System with Live Web Interface},
    author={Rathour, Ayushi},
    year={2025},
    note={External validation on 485 independent samples, Live demo at https://pneumodetectai.streamlit.app/},
    url={https://github.com/ayushirathour/chest-xray-pneumonia-detection}
}
```

**License:** MIT - See LICENSE for details

---

## 🏆 **Acknowledgments**
- **Training Dataset:** [Paul Timothy Mooney - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **External Validation Dataset:** [Tanmay Shukla - Pneumonia Radiography Dataset](https://www.kaggle.com/datasets/iamtanmayshukla/pneumonia-radiography-dataset)
- **Frameworks:** TensorFlow, scikit-learn, Streamlit
- **Model Hosting:** Hugging Face Hub
- **Web Hosting:** Streamlit Cloud

---

## 📞 **Contact & Collaboration**

**Ayushi Rathour** - *Biotechnology Graduate | AI in Healthcare Researcher*
- **🌐 Live Demo:** [PneumoDetect AI](https://pneumodetectai.streamlit.app/)
- **GitHub:** [@ayushirathour](https://github.com/ayushirathour)
- **LinkedIn:** [Ayushi Rathour](https://linkedin.com/in/ayushi-rathour)
- **Email:** ayushirathour1804@gmail.com
- **HuggingFace:** [ayushirathour](https://huggingface.co/ayushirathour)

---

**⚡ Advancing AI in Healthcare Through Rigorous Validation & Accessible Deployment**

*This project demonstrates the complete journey from AI research to production deployment, emphasizing the importance of external validation in medical AI while providing an accessible web interface for global use.*

---

**⭐ If you find this project helpful, please consider starring the repository and sharing it with others interested in AI in healthcare!**
