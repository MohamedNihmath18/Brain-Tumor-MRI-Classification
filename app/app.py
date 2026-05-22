# ============================================================
# BRAIN TUMOR MRI IMAGE CLASSIFICATION
# Streamlit Web Application — TensorFlow version
# ============================================================

import streamlit as st
import numpy as np
from PIL import Image
import os
import time
import pandas as pd

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title = "Brain Tumor MRI Classifier",
    page_icon  = "🧠",
    layout     = "wide"
)

# ── Constants ──────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__),
                          "../models/efficientnetb0.h5")
CLASSES    = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
IMG_SIZE   = (224, 224)

CLASS_INFO = {
    'Glioma': {
        'color'      : '#FF4B4B',
        'description': 'Glioma is a tumor that occurs in the brain and '
                       'spinal cord, originating in glial cells. It is '
                       'the most common type of malignant brain tumor.',
        'severity'   : '🔴 High Risk',
        'action'     : 'Immediate consultation with a neurosurgeon is '
                       'strongly recommended.'
    },
    'Meningioma': {
        'color'      : '#FFA500',
        'description': 'Meningioma arises from the meninges — the membranes '
                       'surrounding the brain and spinal cord. Most '
                       'meningiomas are benign (non-cancerous).',
        'severity'   : '🟡 Moderate Risk',
        'action'     : 'Schedule an appointment with a neurologist '
                       'for further evaluation.'
    },
    'No Tumor': {
        'color'      : '#00CC00',
        'description': 'No tumor was detected in this MRI scan. The brain '
                       'appears normal based on the classification model.',
        'severity'   : '🟢 Normal',
        'action'     : 'No immediate action required. '
                       'Continue regular health checkups.'
    },
    'Pituitary': {
        'color'      : '#4B9EFF',
        'description': 'Pituitary tumors form in the pituitary gland at '
                       'the base of the brain. Most are benign and can '
                       'affect hormone production.',
        'severity'   : '🔵 Moderate Risk',
        'action'     : 'Consult an endocrinologist or neurosurgeon '
                       'for further assessment.'
    }
}

# ── Load model ─────────────────────────────────────────────
@st.cache_resource
def load_classifier():
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    model = load_model(MODEL_PATH)
    return model

# ── Predict function ───────────────────────────────────────
def predict(image, model):
    from tensorflow.keras.applications.efficientnet import preprocess_input
    img       = image.convert('RGB')
    img       = img.resize(IMG_SIZE, Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array, verbose=0)
    pred_class  = CLASSES[np.argmax(predictions[0])]
    confidence  = float(np.max(predictions[0])) * 100
    all_probs   = {
        CLASSES[i]: float(predictions[0][i]) * 100
        for i in range(len(CLASSES))
    }
    return pred_class, confidence, all_probs

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 20px;
        border-radius: 12px;
        border: 2px solid;
        margin: 10px 0;
        text-align: center;
        color: #ffffff !important;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #dee2e6;
        color: #1a1a2e !important;
        font-size: 0.95rem;
    }
    .disclaimer {
        background: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        font-size: 0.9rem;
        color: #856404 !important;
    }
    .info-box {
        background: #e8f4fd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 10px 0;
        color: #1a1a2e !important;
        font-size: 0.95rem;
    }
    .stProgress > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────
st.markdown(
    '<p class="main-title">🧠 Brain Tumor MRI Classifier</p>',
    unsafe_allow_html=True
)
st.markdown(
    '<p class="sub-title">AI-powered brain tumor classification '
    'using Deep Learning (EfficientNetB0)</p>',
    unsafe_allow_html=True
)
st.markdown("---")

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/brain.png", width=80)
    st.title("About This App")
    st.markdown("""
    This application uses **EfficientNetB0** deep learning model
    trained on brain MRI images to classify tumor types.

    **Model Performance:**
    - ✅ Accuracy  : 91.46%
    - ✅ Precision : 91.52%
    - ✅ Recall    : 91.46%
    - ✅ F1 Score  : 91.40%

    **Tumor Classes:**
    - 🔴 Glioma
    - 🟡 Meningioma
    - 🟢 No Tumor
    - 🔵 Pituitary

    **How to use:**
    1. Upload a brain MRI image
    2. Click Analyze
    3. View prediction & confidence

    ---
    ⚠️ **Disclaimer:** This tool is for
    educational purposes only. Always
    consult a medical professional.
    """)
    st.markdown("---")
    st.markdown("**Model:** EfficientNetB0")
    st.markdown("**Framework:** TensorFlow/Keras")
    st.markdown("**Developer:** Mohamed Nihmath")

# ── Main content ───────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📤 Upload MRI Image")
    uploaded_file = st.file_uploader(
        "Choose a brain MRI image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a brain MRI scan image (JPG, JPEG, or PNG)"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI Scan", width=400)
        st.markdown(f"""
        <div class="metric-card">
            <b>File:</b> {uploaded_file.name}<br>
            <b>Size:</b> {image.size[0]} x {image.size[1]} px<br>
            <b>Mode:</b> {image.mode}
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button(
            "🔍 Analyze MRI Scan",
            use_container_width=True,
            type="primary"
        )
    else:
        st.markdown("""
        <div class="info-box">
            👆 Please upload a brain MRI image to get started.<br><br>
            Supported formats: <b>JPG, JPEG, PNG</b>
        </div>
        """, unsafe_allow_html=True)
        analyze_btn = False

with col2:
    st.subheader("📊 Classification Results")

    if uploaded_file is not None and analyze_btn:
        with st.spinner("🧠 Analyzing MRI scan..."):
            model = load_classifier()
            time.sleep(0.5)
            pred_class, confidence, all_probs = predict(image, model)
            info = CLASS_INFO[pred_class]

        st.markdown(f"""
        <div class="result-box"
             style="border-color:{info['color']};
                    background-color:{info['color']}15;">
            <h2 style="color:{info['color']}; margin:0;">
                {pred_class}
            </h2>
            <h3 style="color:{info['color']}; margin:5px 0;">
                Confidence: {confidence:.2f}%
            </h3>
            <p style="margin:5px 0; color:{info['color']}">
                {info['severity']}
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Confidence Scores")
        for cls, prob in sorted(
            all_probs.items(), key=lambda x: x[1], reverse=True
        ):
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.progress(max(0, min(100, int(prob))))
                st.caption(cls)
            with col_b:
                st.markdown(f"**{prob:.1f}%**")

        st.markdown("#### 📋 About This Condition")
        st.markdown(f"""
        <div class="info-box">
            {info['description']}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 🏥 Recommended Action")
        st.markdown(f"""
        <div class="info-box">
            {info['action']}
        </div>
        """, unsafe_allow_html=True)

    elif uploaded_file is None:
        st.markdown("""
        <div class="info-box">
            Results will appear here after you upload
            an MRI image and click Analyze.
        </div>
        """, unsafe_allow_html=True)

# ── Model Performance ──────────────────────────────────────
st.markdown("---")
st.subheader("📈 Model Performance Overview")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Accuracy",  "91.46%", "+26.42% vs baseline")
with c2:
    st.metric("Precision", "91.52%")
with c3:
    st.metric("Recall",    "91.46%")
with c4:
    st.metric("F1 Score",  "91.40%")

st.markdown("#### All Models Comparison")
df = pd.DataFrame([
    {"Model": "Custom CNN",     "Accuracy": "69.11%",
     "F1 Score": "0.6419", "Status": "❌ Baseline"},
    {"Model": "EfficientNetB0", "Accuracy": "91.46%",
     "F1 Score": "0.9140", "Status": "✅ Deployed"},
    {"Model": "MobileNetV2",    "Accuracy": "91.06%",
     "F1 Score": "0.9086", "Status": "✅ Trained"},
    {"Model": "ResNet50",       "Accuracy": "80.08%",
     "F1 Score": "0.7821", "Status": "✅ Trained"},
])
st.dataframe(df, use_container_width=True, hide_index=True)

# ── Disclaimer ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class="disclaimer">
    ⚠️ <b>Medical Disclaimer:</b> This application is developed for
    educational and research purposes only. The predictions made by
    this AI model should NOT be used as a substitute for professional
    medical diagnosis. Always consult a qualified medical professional
    for proper diagnosis and treatment.
</div>
""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#888;'>"
    "Built with TensorFlow & Streamlit | "
    "Brain Tumor MRI Classification | "
    "Mohamed Nihmath</p>",
    unsafe_allow_html=True
)