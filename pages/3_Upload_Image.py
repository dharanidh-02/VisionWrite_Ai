import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

import config
import utils

utils.render_page_hero(
    "Upload and Predict",
    "Upload a single handwritten character image and receive a premium AI response with confidence analytics.",
    stats=[
        ("Mode", "Image Upload"),
        ("Model", "CNN"),
        ("History", str(st.session_state.total_predictions)),
    ],
)

model = utils.load_model()
mapping = utils.load_label_mapping()

if model is None or mapping is None:
    st.error("Failed to load AI model assets from /Ai_models.")
    st.stop()

left, right = st.columns([1, 1.15], gap="large")
with left:
    st.markdown("<div class='surface-card'>", unsafe_allow_html=True)
    st.subheader("Input")
    align_toggle = st.toggle("Model alignment", value=True)
    uploaded_file = st.file_uploader(
        "Upload image",
        type=["png", "jpg", "jpeg", "bmp"],
        help="Use a single character on a clean background for best results.",
    )
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown(
        """
        <div class='surface-card'>
            <h3 style='margin-top:0;'>AI Response</h3>
            <p>The response card updates after image upload and includes confidence, timing, top-5 classes, and explanation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

if uploaded_file is None:
    st.info("Upload an image to start prediction.")
    st.stop()

image = Image.open(uploaded_file)
image_np = np.array(image)
processed_img, thresholded_preview = utils.preprocess_image(image_np, align_model=align_toggle)
pred_char, confidence, top_5, inference_time = utils.predict_character(model, mapping, processed_img)

signature = f"upload_{uploaded_file.name}_{pred_char}_{confidence:.4f}"
if st.session_state.get("last_upload_sig") != signature:
    utils.add_to_history(pred_char, confidence, inference_time, f"Upload ({uploaded_file.name})")
    st.session_state.last_upload_sig = signature

level, message, status_cls = utils.confidence_message(confidence)

res_col1, res_col2, res_col3 = st.columns(3, gap="large")
with res_col1:
    st.markdown(
        f"""
        <div class='panel-card' style='text-align:center;'>
            <div class='metric-label'>Predicted Character</div>
            <div class='prediction-number'>{pred_char}</div>
            <div class='metric-desc'>Inference: {inference_time * 1000:.1f} ms</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with res_col2:
    colors = config.get_current_colors()
    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            number={"suffix": "%", "font": {"family": "Outfit", "size": 24}},
            title={"text": "Confidence Circle", "font": {"family": "Poppins", "size": 14}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": colors["accent_primary"]},
                "steps": [
                    {"range": [0, 50], "color": "rgba(248,113,113,0.2)"},
                    {"range": [50, 80], "color": "rgba(251,191,36,0.2)"},
                    {"range": [80, 100], "color": "rgba(34,197,94,0.2)"},
                ],
            },
        )
    )
    fig_gauge.update_layout(**utils.get_plotly_layout("", 220, False))
    st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})
with res_col3:
    st.markdown(
        f"""
        <div class='status-card {status_cls}'>
            <div class='metric-label'>Prediction Quality</div>
            <h4 style='margin:8px 0 4px 0;'>{level} Confidence</h4>
            <p style='margin:0;'>{message}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

viz_col1, viz_col2 = st.columns(2, gap="large")
with viz_col1:
    st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
    st.subheader("Processing Preview")
    img_cols = st.columns(2)
    with img_cols[0]:
        st.image(image, caption="Original", use_container_width=True)
    with img_cols[1]:
        st.image(thresholded_preview, caption="Model Input (28×28)", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
with viz_col2:
    st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
    st.subheader("Top-5 Probabilities")
    top_5_df = pd.DataFrame(top_5, columns=["Character", "Probability"])
    top_5_df["Percentage"] = top_5_df["Probability"] * 100
    fig_bar = px.bar(
        top_5_df,
        y="Character",
        x="Percentage",
        orientation="h",
        text=top_5_df["Percentage"].map(lambda x: f"{x:.1f}%"),
        color="Percentage",
        color_continuous_scale=[colors["accent_secondary"], colors["accent_primary"]],
    )
    fig_bar.update_layout(**utils.get_plotly_layout("", 320, False))
    fig_bar.update_layout(yaxis={"autorange": "reversed"}, coloraxis_showscale=False)
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

st.subheader("Prediction History")
if st.session_state.prediction_history:
    st.markdown(utils.render_prediction_transcript(st.session_state.prediction_history), unsafe_allow_html=True)
    history_df = pd.DataFrame(st.session_state.prediction_history).iloc[::-1][
        ["Timestamp", "Source", "Predicted Character", "Confidence", "Inference Time"]
    ]
    utils.render_searchable_table(history_df, "upload_history", page_size=6)
    csv_data = utils.export_history_to_csv()
    if csv_data:
        st.download_button(
            "Export History CSV",
            data=csv_data,
            file_name="visionwrite_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )
