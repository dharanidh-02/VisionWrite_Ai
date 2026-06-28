import streamlit as st
import os

# Set page title
st.markdown("<h1 class='gradient-text'>📖 About Project</h1>", unsafe_allow_html=True)
st.markdown("Detailed documentation of VisionWrite AI's research, methodology, and implementation.")

# Table of Contents
st.markdown(
    """
    ### Table of Contents
    1. [Introduction](#introduction)
    2. [Project Objectives](#project-objectives)
    3. [Deep Learning Workflow](#deep-learning-workflow)
    4. [CNN Architecture Details](#cnn-architecture-details)
    5. [Dataset Information](#dataset-information)
    6. [Technology Stack](#technology-stack)
    7. [Future Scope](#future-scope)
    8. [References](#references)
    """
)

st.markdown("---")

# 1. Introduction
st.markdown("<a name='introduction'></a>", unsafe_allow_html=True)
st.markdown("## 1. Introduction")
st.markdown(
    """
    **VisionWrite AI** is a state-of-the-art handwritten character recognition (HCR) system. 
    Character recognition has been a focal point in the field of Computer Vision and Pattern Recognition for decades. 
    While Optical Character Recognition (OCR) for printed text has reached near-perfect accuracy, recognizing handwritten text 
    remains a challenging task due to the infinite variety in human writing styles, stroke thicknesses, slants, and shapes.
    
    By utilizing a **Convolutional Neural Network (CNN)** trained on the extensive **EMNIST Balanced** dataset, VisionWrite AI 
    bridges the gap between physical handwriting and digital text. The application provides an interactive platform for 
    real-time prediction via canvas drawing or image uploads, making it a viable solution for commercial and archival use cases.
    """
)

# 2. Project Objectives
st.markdown("<a name='project-objectives'></a>", unsafe_allow_html=True)
st.markdown("## 2. Project Objectives")
st.markdown(
    """
    - **High-Accuracy Recognition:** Develop a robust deep learning model capable of distinguishing between 47 distinct character classes (digits, uppercase, and lowercase letters).
    - **Real-Time Inference:** Build a low-latency pipeline that processes drawings or uploads and outputs predictions in milliseconds.
    - **Premium User Experience:** Design an intuitive, commercial-grade SaaS interface using Streamlit, custom CSS, and interactive Plotly charts.
    - **Clean, Modular Architecture:** Maintain a production-ready codebase separating UI, configurations, and core business logic.
    - **Performance Auditing:** Provide comprehensive visual feedback of model performance, including training history curves and a detailed confusion matrix.
    """
)

# 3. Deep Learning Workflow
st.markdown("<a name='deep-learning-workflow'></a>", unsafe_allow_html=True)
st.markdown("## 3. Deep Learning Workflow")
st.markdown(
    """
    The workflow of VisionWrite AI is divided into five core phases:
    
    1. **Data Preparation:** The EMNIST Balanced dataset is loaded, containing 131,600 pre-labeled grayscale images. The images are split into training (85%) and testing (15%) subsets.
    2. **Preprocessing Pipeline:** Input images (whether drawn on the canvas or uploaded) are converted to grayscale, resized to `28 x 28` pixels, binarized via thresholding to remove noise, and transposed. Transposition is necessary because EMNIST images are stored in a column-major format.
    3. **CNN Feature Extraction:** The preprocessed image is passed through convolutional layers where filters extract spatial features (curves, lines, loops). Max pooling layers reduce dimensionality, and dropout layers prevent overfitting.
    4. **Softmax Classification:** The fully connected dense layers map the extracted features to a 47-dimensional output vector. The Softmax function converts these outputs into probability scores.
    5. **Visual Output & Logging:** The character with the highest probability is displayed along with confidence metrics. The prediction is saved in the session history, allowing users to export results as a CSV report.
    """
)

# 4. CNN Architecture Details
st.markdown("<a name='cnn-architecture-details'></a>", unsafe_allow_html=True)
st.markdown("## 4. CNN Architecture Details")
st.markdown(
    """
    The neural network is structured specifically for image pattern recognition. Below is the layer-by-layer explanation:
    
    - **Input Layer:** Accepts a 3D tensor of shape `(28, 28, 1)` representing a single-channel grayscale image.
    - **Convolutional Layers (Conv2D):** Apply learnable filters to the input. The early layers detect simple features like horizontal and vertical edges, while deeper layers capture complex shapes (loops, intersections).
    - **Activation Function (ReLU):** Introduces non-linearity ($f(x) = \max(0, x)$), allowing the network to learn complex non-linear decision boundaries.
    - **Max Pooling Layers (MaxPooling2D):** Downsample the feature maps by taking the maximum value in a sliding window (typically `2 x 2`). This reduces computational complexity and provides translation invariance (recognizing the character even if it is slightly shifted).
    - **Dropout Layers:** A regularization technique where a fraction of neurons (e.g., 25% to 50%) are randomly deactivated during training. This forces the network to learn redundant representations and prevents overfitting.
    - **Flatten Layer:** Flattens the 2D feature maps into a 1D vector to prepare for the classification layers.
    - **Dense (Fully Connected) Layers:** Fully connect all inputs to all outputs. The final dense layer has **47 units** (one for each class).
    - **Softmax Activation:** Applied to the final layer to output a probability distribution summing to 1.
    """
)

# 5. Dataset Information
st.markdown("<a name='dataset-information'></a>", unsafe_allow_html=True)
st.markdown("## 5. Dataset Information")
st.markdown(
    """
    VisionWrite AI utilizes the **EMNIST (Extended MNIST) Balanced** dataset. 
    While the classic MNIST dataset contains only digits (0-9), EMNIST extends this to include both uppercase and lowercase letters.
    
    The **Balanced** split is specifically designed to address the class imbalance present in the full EMNIST dataset (where letters like 'E' are far more common than 'Q'). 
    - **Total Classes:** 47
      - Digits: `0 - 9` (10 classes)
      - Uppercase Letters: `A - Z` (26 classes)
      - Lowercase Letters: `a, b, d, e, f, g, h, n, q, r, t` (11 classes). The remaining lowercase letters are excluded because they are visually indistinguishable from their uppercase counterparts in handwriting (e.g., 'C' and 'c', 'S' and 's', 'O' and 'o').
    - **Samples per Class:** Exactly 2,400 training samples and 400 testing samples per class, ensuring zero bias towards any character.
    - **Image Format:** `28 x 28` pixels, grayscale, inverted (white character on black background), and transposed.
    """
)

# 6. Technology Stack
st.markdown("<a name='technology-stack'></a>", unsafe_allow_html=True)
st.markdown("## 6. Technology Stack")

tech_col1, tech_col2 = st.columns(2)

with tech_col1:
    st.markdown(
        """
        #### 🖥️ Frontend & UI
        - **Streamlit:** Fast, pythonic web application framework.
        - **Streamlit Drawable Canvas:** Custom component enabling smooth drawing interactions.
        - **Plotly:** Interactive, high-performance charting library for gauges and bar charts.
        - **HTML5/CSS3:** Custom styles and animations for a premium dark/light mode SaaS feel.
        """
    )

with tech_col2:
    st.markdown(
        """
        #### 🧠 Machine Learning & Processing
        - **TensorFlow / Keras:** Backend deep learning framework for CNN inference.
        - **OpenCV (cv2):** Image processing library for resizing, thresholding, and transposing.
        - **Pillow (PIL):** Image loading and format conversions.
        - **NumPy & Pandas:** High-performance numerical computations and data manipulation.
        - **Joblib / Pickle:** Serializing and loading metadata (mappings, training history).
        """
    )

# 7. Future Scope
st.markdown("<a name='future-scope'></a>", unsafe_allow_html=True)
st.markdown("## 7. Future Scope")
st.markdown(
    """
    While VisionWrite AI is fully functional for individual character recognition, several advancements are planned for future releases:
    
    1. **Cursive Handwriting OCR:** Moving from isolated character recognition to continuous word and sentence recognition using **Recurrent Neural Networks (RNN)** or **Transformers** (e.g., Vision Transformers).
    2. **Multilingual Support:** Extending the dataset to support accented characters, symbols, and non-Latin scripts (such as Devanagari, Hanzi, or Arabic).
    3. **Layout Analysis:** Integrating **Object Detection** (e.g., YOLO or Faster R-CNN) to segment pages, detect paragraphs, lines, and words before classifying individual characters.
    4. **On-Device Mobile App:** Porting the trained Keras model to **TensorFlow Lite (TFLite)** for lightweight, offline inference on iOS and Android devices.
    """
)

# 8. References
st.markdown("<a name='references'></a>", unsafe_allow_html=True)
st.markdown("## 8. References")
st.markdown(
    """
    - **EMNIST Dataset Paper:** Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). *EMNIST: Extending MNIST to handwritten letters.* IEEE International Joint Conference on Neural Networks (IJCNN).
    - **CNN Architecture:** LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). *Gradient-based learning applied to document recognition.* Proceedings of the IEEE.
    - **Streamlit Documentation:** [https://docs.streamlit.io](https://docs.streamlit.io)
    - **TensorFlow Keras Guide:** [https://www.tensorflow.org/guide/keras](https://www.tensorflow.org/guide/keras)
    - **OpenCV Tutorials:** [https://docs.opencv.org](https://docs.opencv.org)
    """
)
