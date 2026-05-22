# 🧠 Brain Tumor MRI Image Classification

An end-to-end deep learning project for classifying brain MRI images into 4 tumor categories using CNN and Transfer Learning.

## 🎯 Project Overview
- **Domain:** Medical Imaging — Image Classification
- **Framework:** TensorFlow/Keras
- **Deployment:** Streamlit Cloud

## 📊 Classes
| Class | Description |
|-------|-------------|
| Glioma | Most common malignant brain tumor |
| Meningioma | Outer brain membrane tumor |
| No Tumor | Normal brain scan |
| Pituitary | Hormone gland tumor |

## 🏆 Model Performance
| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Custom CNN | 69.11% | 0.6419 |
| EfficientNetB0 | 91.46% | 0.9140 ✅ |
| MobileNetV2 | 91.06% | 0.9086 |
| ResNet50 | 80.08% | 0.7821 |

## 🚀 Live Demo
[Click here to try the app](<your-streamlit-url>)

## 📁 Project Structure

Brain-Tumor-MRI-Classification/
├── notebooks/          # Jupyter notebooks
├── models/             # Trained model files
├── app/                # Streamlit application
├── requirements.txt    # Dependencies
└── README.md

## 🛠️ Tech Stack
- Python 3.12
- TensorFlow/Keras
- EfficientNetB0 (Transfer Learning)
- Streamlit
- scikit-learn
- NumPy, Pandas, Matplotlib

## 📌 Dataset
- Source: Roboflow Brain Tumor Dataset
- Total Images: 2,443
- Split: Train (1695) / Valid (502) / Test (246)

## 👨‍💻 Developer
Mohamed Nihmath