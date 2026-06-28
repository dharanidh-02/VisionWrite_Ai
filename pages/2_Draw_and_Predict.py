import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_drawable_canvas import st_canvas

import config
import utils

utils.render_page_hero(
    "Draw and Predict",
    "Use the interactive canvas or a live camera snapshot to get premium AI character predictions.",
    stats=[
        ("Modes", "Canvas + Camera"),
        ("Realtime", "Enabled"),
        ("Session Predictions", str(st.session_state.total_predictions)),
    ],
)

model = utils.load_model()
mapping = utils.load_label_mapping()
if model is None or mapping is None:
    st.error("Failed to load AI model assets from /Ai_models.")
    st.stop()

tab_canvas, tab_camera = st.tabs(["Canvas", "Camera Snapshot"])

with tab_canvas:
    controls, output = st.columns([1, 1.2], gap="large")
    with controls:
        st.markdown("<div class='surface-card'>", unsafe_allow_html=True)
        st.subheader("Drawing Controls")
        stroke_width = st.slider("Brush thickness", min_value=8, max_value=30, value=18, step=1)
        align_toggle = st.toggle("Model alignment", value=True)

        if "canvas_reset_key" not in st.session_state:
            st.session_state.canvas_reset_key = 0

        canvas_result = st_canvas(
            fill_color="rgba(255,255,255,0)",
            stroke_width=stroke_width,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=340,
            width=340,
            drawing_mode="freedraw",
            update_streamlit=True,
            key=f"canvas_{st.session_state.canvas_reset_key}",
        )

        btn1, btn2 = st.columns(2)
        with btn1:
            if st.button("Clear Canvas", use_container_width=True):
                st.session_state.canvas_reset_key += 1
                st.rerun()
        with btn2:
            if st.button("Reset Session", use_container_width=True):
                st.session_state.canvas_reset_key += 1
                st.session_state.prediction_history = []
                st.session_state.total_predictions = 0
                st.session_state.avg_confidence = 0.0
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with output:
        if canvas_result.image_data is not None and np.any(canvas_result.image_data[:, :, :3] > 0):
            raw_img = canvas_result.image_data
            processed_img, thresholded_preview = utils.preprocess_image(raw_img, align_model=align_toggle)
            pred_char, confidence, top_5, inference_time = utils.predict_character(model, mapping, processed_img)

            curr_signature = f"canvas_{pred_char}_{confidence:.4f}"
            if st.session_state.get("last_sig") != curr_signature:
                utils.add_to_history(pred_char, confidence, inference_time, "Canvas Drawing")
                st.session_state.last_sig = curr_signature

            level, message, status_cls = utils.confidence_message(confidence)
            r1, r2 = st.columns([1, 1.15], gap="large")
            with r1:
                st.markdown(
                    f"""
                    <div class='panel-card' style='text-align:center;'>
                        <div class='metric-label'>Predicted Character</div>
                        <div class='prediction-number'>{pred_char}</div>
                        <div class='metric-desc'>Inference: {inference_time*1000:.1f} ms</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.image(thresholded_preview, caption="Model Input (28×28)", width=180)

            with r2:
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
                st.markdown(
                    f"""
                    <div class='status-card {status_cls}'>
                        <div class='metric-label'>{level} Confidence</div>
                        <p style='margin:6px 0 0 0;'>{message}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

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
            fig_bar.update_layout(**utils.get_plotly_layout("", 300, False))
            fig_bar.update_layout(yaxis={"autorange": "reversed"}, coloraxis_showscale=False)
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Draw a character in the canvas to run prediction.")

with tab_camera:
    st.markdown("<div class='surface-card'>", unsafe_allow_html=True)
    st.subheader("Camera Capture")
    st.write("Capture a frame from your camera and run the same character recognition pipeline.")
    align_toggle_cam = st.toggle("Model alignment", value=True, key="camera_align_toggle")
    camera_image = st.camera_input("Take a snapshot")
    st.markdown("</div>", unsafe_allow_html=True)

    if camera_image is not None:
        file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_img, thresholded_preview = utils.preprocess_image(frame_rgb, align_model=align_toggle_cam)
        pred_char, confidence, top_5, inference_time = utils.predict_character(model, mapping, processed_img)

        signature = f"camera_{pred_char}_{confidence:.4f}"
        if st.session_state.get("last_camera_sig") != signature:
            utils.add_to_history(pred_char, confidence, inference_time, "Camera Snapshot")
            st.session_state.last_camera_sig = signature

        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(frame_rgb, caption="Captured Frame", use_container_width=True)
        with c2:
            st.image(thresholded_preview, caption="Processed Input", use_container_width=True)
        with c3:
            st.markdown(
                f"""
                <div class='panel-card' style='text-align:center;'>
                    <div class='metric-label'>Prediction</div>
                    <div class='prediction-number'>{pred_char}</div>
                    <div class='metric-desc'>Confidence: {confidence*100:.1f}%</div>
                    <div class='metric-desc'>Time: {inference_time*1000:.1f} ms</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.subheader("Prediction History")
if st.session_state.prediction_history:
    st.markdown(utils.render_prediction_transcript(st.session_state.prediction_history), unsafe_allow_html=True)
    history_df = pd.DataFrame(st.session_state.prediction_history).iloc[::-1][
        ["Timestamp", "Source", "Predicted Character", "Confidence", "Inference Time"]
    ]
    utils.render_searchable_table(history_df, "draw_history", page_size=6)
    csv_data = utils.export_history_to_csv()
    if csv_data:
        st.download_button(
            "Export History CSV",
            data=csv_data,
            file_name="visionwrite_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )
else:
    st.info("No predictions yet in this session.")
