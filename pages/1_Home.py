import os

import streamlit as st

import config
import utils

utils.render_page_hero(
    "Handwritten Character Intelligence",
    "A premium AI workspace for recognizing handwritten symbols with fast, reliable CNN inference.",
    stats=[
        ("Model", "CNN"),
        ("Classes", str(config.EMNIST_METADATA["total_classes"])),
        ("Dataset", "EMNIST Balanced"),
        ("Session Predictions", str(st.session_state.total_predictions)),
    ],
)

hero_col, hero_action = st.columns([3, 1.2], gap="large")
with hero_col:
    st.markdown(
        """
        <div class='surface-card'>
            <h3 style='margin-top:0;'>Why VisionWrite AI</h3>
            <p>VisionWrite AI transforms raw handwritten input into machine-readable characters using a production CNN stack. The experience is designed as a modern AI product with structured analytics, rich prediction feedback, and high usability across devices.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with hero_action:
    st.markdown(
        """
        <div class='surface-card'>
            <h3 style='margin-top:0;'>Quick Start</h3>
            <p>Go to Draw & Predict or Upload Image to run live predictions and export your session history.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("### Platform Metrics")
metric_cols = st.columns(4)
metrics = [
    ("Training Samples", "112.8K", "Balanced training split"),
    ("Test Samples", "18.8K", "Evaluation partition"),
    ("Input Shape", "28×28×1", "Optimized grayscale pipeline"),
    ("Approx Accuracy", "87.5%", "Validated on EMNIST balanced"),
]
for col, (label, value, desc) in zip(metric_cols, metrics):
    with col:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-label'>{label}</div>
                <div class='metric-number'>{value}</div>
                <div class='metric-desc'>{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("### Interactive Workflow Timeline")
steps = [
    ("1", "Input", "Canvas, uploaded image, or camera-based air writing input."),
    ("2", "Preprocess", "Grayscale conversion, resize, thresholding, and optional alignment."),
    ("3", "Inference", "CNN predicts class probabilities across 47 character classes."),
    ("4", "AI Response", "Premium response card displays prediction, confidence, and explanation."),
    ("5", "Analytics", "Top-5 probabilities and session history table are updated instantly."),
]
step_cols = st.columns(len(steps))
for col, (idx, title, text) in zip(step_cols, steps):
    with col:
        st.markdown(
            f"""
            <div class='surface-card' style='height:100%;'>
                <div class='metric-label'>Step {idx}</div>
                <h4 style='margin:8px 0;'>{title}</h4>
                <p style='margin:0;'>{text}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("### Applications")
app_cols = st.columns(3)
applications = [
    ("Finance", "Digitize cheques and handwritten forms with low-latency character extraction."),
    ("Education", "Evaluate handwritten assessments and convert notes into searchable text."),
    ("Archives", "Modernize manuscript indexing with AI-assisted transcription workflows."),
]
for col, (title, desc) in zip(app_cols, applications):
    with col:
        st.markdown(
            f"""
            <div class='surface-card' style='height:100%;'>
                <h4 style='margin-top:0;'>{title}</h4>
                <p>{desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("### Technology & Dataset Overview")
info_left, info_right = st.columns(2, gap="large")
with info_left:
    st.markdown(
        """
        <div class='surface-card'>
            <h4 style='margin-top:0;'>Core Stack</h4>
            <p><strong>Frontend:</strong> Streamlit + custom design system</p>
            <p><strong>Model:</strong> TensorFlow / Keras CNN</p>
            <p><strong>Vision:</strong> OpenCV preprocessing pipeline</p>
            <p><strong>Analytics:</strong> Plotly interactive charts</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with info_right:
    st.markdown(
        """
        <div class='surface-card'>
            <h4 style='margin-top:0;'>Dataset Snapshot</h4>
            <p>EMNIST Balanced provides 47 classes with balanced class distribution suitable for robust handwritten character recognition.</p>
            <p>All prediction pages keep model behavior unchanged while improving UX quality and visual hierarchy.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

if os.path.exists("assets/hero.png"):
    st.image("assets/hero.png", use_container_width=True)

st.markdown(
    """
    <div style='margin-top:24px; text-align:center; color:var(--text-muted); font-size:14px;'>
        VisionWrite AI • Premium Handwritten Character Recognition Workspace
    </div>
    """,
    unsafe_allow_html=True,
)
