import streamlit as st

import utils

utils.render_page_hero(
    "About the Project",
    "VisionWrite AI combines practical CNN modeling with a production-style user experience for handwriting recognition.",
    stats=[("Sections", "7"), ("Architecture", "CNN"), ("Dataset", "EMNIST Balanced")],
)

st.subheader("Project Timeline")
steps = [
    ("Research", "Problem framing, dataset validation, and architecture selection."),
    ("Training", "CNN training with EMNIST balanced classes and regularization."),
    ("Inference", "Preprocessing and low-latency prediction execution path."),
    ("Experience", "Premium UI system with responsive and theme-aware design."),
]
cols = st.columns(4)
for col, (title, text) in zip(cols, steps):
    with col:
        st.markdown(
            f"""
            <div class='surface-card' style='height:100%;'>
                <h4 style='margin-top:0;'>{title}</h4>
                <p style='margin-bottom:0;'>{text}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

left, right = st.columns(2, gap="large")
with left:
    st.markdown(
        """
        <div class='surface-card'>
            <h3 style='margin-top:0;'>CNN Workflow</h3>
            <p>Input image → grayscale conversion → resize to 28×28 → thresholding → alignment → CNN inference → softmax probabilities.</p>
            <h4>Technology Stack</h4>
            <p>Frontend: Streamlit<br>AI: TensorFlow/Keras<br>Vision: OpenCV<br>Analytics: Plotly<br>Data: Pandas/NumPy</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with right:
    st.markdown(
        """
        <div class='surface-card'>
            <h3 style='margin-top:0;'>Dataset</h3>
            <p>EMNIST Balanced includes 47 character classes with uniform support, enabling robust handwritten character classification across digits and letters.</p>
            <h4>Model Objective</h4>
            <p>Deliver fast and consistent single-character recognition for real-world UI flows such as scanned forms, notes, and digitization tasks.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

ref_col, future_col = st.columns(2, gap="large")
with ref_col:
    st.markdown(
        """
        <div class='surface-card'>
            <h3 style='margin-top:0;'>References</h3>
            <ul>
                <li>EMNIST dataset paper (IJCNN, 2017)</li>
                <li>LeNet/CNN foundational work</li>
                <li>TensorFlow and Streamlit documentation</li>
                <li>OpenCV and Plotly documentation</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
with future_col:
    st.markdown(
        """
        <div class='surface-card'>
            <h3 style='margin-top:0;'>Future Scope</h3>
            <ul>
                <li>Word and line-level OCR pipeline</li>
                <li>Multilingual script expansion</li>
                <li>On-device inference using TFLite</li>
                <li>Explainability modules for prediction trust</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
