import streamlit as st
import os
import config

# Title of the page (Premium centered typography)
st.markdown("<h1 style='text-align: center; margin-bottom: 0;' class='gradient-text'>VisionWrite AI</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; font-weight: 400; margin-top: 5px; color: var(--text-color);'>AI-Powered Handwritten Character Recognition using Deep Learning</h3>", unsafe_allow_html=True)

# Hero Banner (Centered and scaled down)
hero_path = os.path.join("assets", "hero.png")
if os.path.exists(hero_path):
    h_col1, h_col2, h_col3 = st.columns([1.2, 1.6, 1.2])
    with h_col2:
        st.image(hero_path, use_container_width=True)

st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

# Four Statistics Cards in a 4-column layout (Premium Glassmorphism)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-card-title">Model Architecture</div>
            <div class="metric-card-value">CNN</div>
            <p style="margin: 8px 0 0 0; font-size: 13px; color: var(--text-muted);">4-Layer Convolutional</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-card-title">Dataset Used</div>
            <div class="metric-card-value">EMNIST</div>
            <p style="margin: 8px 0 0 0; font-size: 13px; color: var(--text-muted);">Balanced Character Split</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-card-title">Total Classes</div>
            <div class="metric-card-value">47</div>
            <p style="margin: 8px 0 0 0; font-size: 13px; color: var(--text-muted);">Digits (0-9) & Letters (A-Z)</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-card-title">Framework</div>
            <div class="metric-card-value">TensorFlow</div>
            <p style="margin: 8px 0 0 0; font-size: 13px; color: var(--text-muted);">Keras Engine (CPU/GPU)</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

# Project Overview & Applications
st.markdown("## 🔍 Project Overview")

left_col, right_col = st.columns([3, 2], gap="large")

with left_col:
    st.markdown(
        """
        ### What is Handwritten Character Recognition?
        Handwritten Character Recognition (HCR) is a specialized subset of Computer Vision that focuses on converting handwritten text from paper documents, images, or touchscreens into machine-readable digital formats. 
        
        Unlike printed text, handwritten characters present significant challenges due to infinite variations in writing styles, stroke thicknesses, slants, and individual hand movements. 
        
        **VisionWrite AI** addresses these challenges using a custom **Convolutional Neural Network (CNN)**. By training on the **EMNIST Balanced dataset**, the model learns high-dimensional topological representations of letters and numbers, allowing it to perform real-time, high-accuracy predictions on both drawn canvases and camera feeds.
        """
    )
    
with right_col:
    handwriting_path = os.path.join("assets", "handwriting.png")
    if os.path.exists(handwriting_path):
        st.image(handwriting_path, use_container_width=True)

st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

st.markdown("### 🌐 Real-World Applications")
app_col1, app_col2, app_col3 = st.columns(3)

with app_col1:
    st.markdown(
        """
        <div class="metric-card" style="height: 100%;">
            <h4 style="margin-top: 0; color: var(--primary-color);">🏦 Financial Services</h4>
            <p style="font-size: 14px; margin-bottom: 0; color: var(--text-muted);">Automating bank cheque processing, signature verification, and financial form digitization with high throughput and safety.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with app_col2:
    st.markdown(
        """
        <div class="metric-card" style="height: 100%;">
            <h4 style="margin-top: 0; color: var(--primary-color);">📬 Logistics & Postal</h4>
            <p style="font-size: 14px; margin-bottom: 0; color: var(--text-muted);">Sorting mail and packages automatically by scanning handwritten addresses and PIN codes from envelopes.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with app_col3:
    st.markdown(
        """
        <div class="metric-card" style="height: 100%;">
            <h4 style="margin-top: 0; color: var(--primary-color);">📚 Education & Archives</h4>
            <p style="font-size: 14px; margin-bottom: 0; color: var(--text-muted);">Digitizing historical manuscripts, old books, library archives, and grading handwritten student assessments.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<div style='margin-top: 35px;'></div>", unsafe_allow_html=True)

# ==================== HORIZONTAL WORKFLOW TIMELINE ====================
st.markdown("## ⚙️ VisionWrite AI Workflow")
st.markdown("The sequential flow of the handwritten character recognition pipeline:")

st.markdown(
    """
    <div class="timeline-wrapper">
        <div class="timeline-line"></div>
        <div class="timeline-container">
            <div class="timeline-step">
                <div class="timeline-node">1</div>
                <div class="timeline-label">📊 Dataset</div>
                <div class="timeline-desc">EMNIST Balanced dataset containing 131,600 samples across 47 classes.</div>
            </div>
            <div class="timeline-step">
                <div class="timeline-node">2</div>
                <div class="timeline-label">🔧 Preprocessing</div>
                <div class="timeline-desc">Convert to grayscale, resize to 28x28, binarize, and mirror.</div>
            </div>
            <div class="timeline-step">
                <div class="timeline-node">3</div>
                <div class="timeline-label">🧠 CNN Inference</div>
                <div class="timeline-desc">A 4-layer Convolutional Neural Network extracts local topological features.</div>
            </div>
            <div class="timeline-step">
                <div class="timeline-node">4</div>
                <div class="timeline-label">🔮 Prediction</div>
                <div class="timeline-desc">A Softmax activation layer outputs a probability distribution over the 47 classes.</div>
            </div>
            <div class="timeline-step">
                <div class="timeline-node">5</div>
                <div class="timeline-label">🎯 Confidence</div>
                <div class="timeline-desc">Winning class probability determines prediction strength (High/Medium/Low).</div>
            </div>
            <div class="timeline-step">
                <div class="timeline-node">6</div>
                <div class="timeline-label">🏆 Results</div>
                <div class="timeline-desc">Displays predicted character, Plotly gauge, bar chart, and logs history to CSV.</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)
