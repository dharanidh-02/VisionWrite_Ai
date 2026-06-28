<div align="center">

# ✍️ VisionWrite AI

### AI-Powered Handwritten Character Recognition using Deep Learning

A premium Computer Vision application that recognizes handwritten characters in real time using a **Convolutional Neural Network (CNN)** trained on the **EMNIST Balanced Dataset**.

<br>

### 🌐 Live Demo

**https://visionwrite.streamlit.app/**

### 📓 Kaggle Training Notebook

**https://www.kaggle.com/code/dharanidharant/handwritten-character-recognition-using-cnn**

<br>

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge\&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep_Learning-FF6F00?style=for-the-badge\&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-FF4B4B?style=for-the-badge\&logo=streamlit)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=for-the-badge\&logo=opencv)
![EMNIST](https://img.shields.io/badge/Dataset-EMNIST_Balanced-success?style=for-the-badge)

</div>

---

# 📌 Overview

**VisionWrite AI** is a production-ready handwritten character recognition platform powered by Deep Learning.

The application uses a **Convolutional Neural Network (CNN)** trained on the **EMNIST Balanced Dataset** to recognize handwritten digits, uppercase letters, and selected lowercase letters in real time.

The project demonstrates a complete Deep Learning workflow including data preprocessing, CNN training, model evaluation, deployment, and an interactive web application.

---

# ✨ Features

## 🎨 Real-Time Drawing Canvas

* Interactive handwriting canvas
* Instant CNN prediction
* Confidence score
* Top-5 predicted characters
* Probability visualization

---

## 📤 Image Upload Prediction

Supports

* PNG
* JPG
* JPEG
* BMP

Automatic

* Grayscale conversion
* Noise reduction
* Background inversion
* Character normalization
* CNN preprocessing

---

## 📊 AI Analytics Dashboard

Interactive pages include

* Training Accuracy
* Validation Accuracy
* Loss Curves
* Confusion Matrix
* Classification Report
* Dataset Statistics
* Model Information

---

## 🌓 Premium UI

* Light Theme
* Dark Theme
* Glassmorphism Cards
* Gradient Buttons
* Responsive Layout
* Plotly Charts
* Interactive Animations

---

# 🧠 Deep Learning Pipeline

```text
EMNIST Balanced Dataset
          │
          ▼
Data Loading
          │
          ▼
Image Preprocessing
          │
          ▼
Normalization
          │
          ▼
CNN Training
          │
          ▼
Model Evaluation
          │
          ▼
Model Export
          │
          ▼
Streamlit Deployment
          │
          ▼
Real-Time Character Prediction
```

---

# 🏗 CNN Architecture

```text
Input (28×28×1)

↓

Conv2D (32)

↓

Batch Normalization

↓

MaxPooling

↓

Conv2D (64)

↓

Batch Normalization

↓

MaxPooling

↓

Conv2D (128)

↓

Batch Normalization

↓

MaxPooling

↓

Flatten

↓

Dense (256)

↓

Dropout

↓

Dense (47)

↓

Softmax
```

---

# 📂 Project Structure

```text
VisionWriteAI/

├── app.py
├── config.py
├── utils.py
├── README.md
├── requirements.txt
│
├── Ai_model/
│   ├── visionwrite_ai_emnist.keras
│   ├── label_mapping.pkl
│   └── training_history.pkl
│
├── assets/
│
└── pages/
    ├── Home
    ├── Draw & Predict
    ├── Upload Image
    ├── Model Performance
    ├── About Project
    └── Developer
```

---

# 📊 Dataset

**Dataset:** EMNIST Balanced

* 47 Character Classes
* Digits
* Uppercase Letters
* Selected Lowercase Letters

Image Size

```
28 × 28
```

---

# 💻 Technology Stack

### Frontend

* Streamlit

### Deep Learning

* TensorFlow
* Keras

### Computer Vision

* OpenCV
* Pillow

### Data Processing

* NumPy
* Pandas

### Visualization

* Plotly
* Matplotlib

---

# 🚀 Getting Started

Clone Repository

```bash
git clone https://github.com/dharanidharan-t/VisionWriteAI.git
```

Install Dependencies

```bash
pip install -r requirements.txt
```

Run

```bash
streamlit run app.py
```

---

# 📈 Model Performance

Evaluation includes

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Classification Report
* Prediction Confidence
* Top-5 Probability Distribution

---

# 📸 Application Modules

* 🏠 Home
* ✍ Draw & Predict
* 📤 Upload Image
* 📊 Model Performance
* 📖 About Project
* 👨‍💻 Developer

---

# 🔮 Future Enhancements

* OCR Word Recognition
* Sentence Recognition
* Mobile Application
* REST API
* Multi-language Character Recognition
* Explainable AI (Grad-CAM)
* ONNX Model Export
* Docker Deployment

---

# 👨‍💻 Developer

**Dharanidharan T**

B.E. Computer Science and Business Systems

Sri Eshwar College of Engineering

**CodeAlpha Machine Learning Intern**

---

# 📄 License

Licensed under the **MIT License**

---

<div align="center">

### ⭐ If you found this project useful, consider giving it a Star!

**Made with ❤️ using TensorFlow, Streamlit and Computer Vision**

</div>
