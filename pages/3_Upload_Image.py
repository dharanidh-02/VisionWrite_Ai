import streamlit as st
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import utils
import config

# Set page title
st.markdown("<h1 class='gradient-text'>📤 Upload Image</h1>", unsafe_allow_html=True)
st.markdown("Upload a photo or scanned image of a handwritten character (e.g., from a notebook or form) to recognize it using the CNN model.")

# Load Model and Mapping
model = utils.load_model()
mapping = utils.load_label_mapping()

if model is None or mapping is None:
    st.error("⚠️ Failed to load the AI model or label mapping. Please check the 'Ai_models' directory.")
else:
    # Option toggles
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        align_toggle = st.toggle("Enable Model Alignment (Recommended)", value=True,
                                 help="Empirical testing shows this model expects input characters to be horizontally flipped (mirrored). Leaving this enabled aligns your image with the model's expected orientation.")
    with col_opt2:
        st.write("") # Spacer

    # File Uploader
    uploaded_file = st.file_uploader(
        "Drag and drop or browse a handwritten character image", 
        type=["png", "jpg", "jpeg", "bmp"],
        help="Upload a single character on a clean background. The system will auto-invert colors if the background is light."
    )

    if uploaded_file is not None:
        try:
            # Load image
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            st.markdown("---")
            st.markdown("### 🔄 Visual Processing Pipeline")
            
            # Preprocess the uploaded image
            processed_img, thresholded_preview = utils.preprocess_image(image_np, align_model=align_toggle)
            
            # Run Inference
            pred_char, confidence, top_5, inference_time = utils.predict_character(model, mapping, processed_img)
            
            # Add to history (only if this is a new upload and not already added)
            curr_signature = f"upload_{uploaded_file.name}_{pred_char}_{confidence:.4f}"
            if "last_upload_sig" not in st.session_state or st.session_state.last_upload_sig != curr_signature:
                utils.add_to_history(pred_char, confidence, inference_time, f"Upload ({uploaded_file.name})")
                st.session_state.last_upload_sig = curr_signature
                st.toast(f"Uploaded character recognized: '{pred_char}'", icon="📤")

            # Three-column comparison layout (Original -> Processed -> Prediction)
            col_orig, col_proc, col_pred = st.columns(3)
            
            with col_orig:
                st.markdown("<div style='text-align: center; height: 100%; border: 1px solid var(--border-color); border-radius: 12px; padding: 15px;'>", unsafe_allow_html=True)
                st.markdown("##### 1. Original Image")
                st.image(image, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col_proc:
                st.markdown("<div style='text-align: center; height: 100%; border: 1px solid var(--border-color); border-radius: 12px; padding: 15px;'>", unsafe_allow_html=True)
                st.markdown("##### 2. Processed Model Input")
                st.image(thresholded_preview, caption="Resized & Binarized (28x28)", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col_pred:
                st.markdown("<div style='text-align: center; height: 100%; border: 1px solid var(--border-color); border-radius: 12px; padding: 15px;'>", unsafe_allow_html=True)
                st.markdown("##### 3. CNN Prediction")
                st.markdown(f"<div class='prediction-large' style='margin-top: 40px;'>{pred_char}</div>", unsafe_allow_html=True)
                st.markdown(f"**Confidence:** `{confidence * 100:.2f}%`")
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("---")
            
            # Detailed metrics below the pipeline
            col_metrics, col_chart = st.columns([1, 1.2], gap="large")
            
            with col_metrics:
                st.markdown("### 📊 Metrics & Status")
                
                # Metric cards
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-card-title">Prediction</div>
                            <div class="metric-card-value">{pred_char}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with m_col2:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-card-title">Time Taken</div>
                            <div class="metric-card-value" style="font-size: 22px; padding: 3px 0;">{inference_time * 1000:.1f} ms</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                # Confidence Status Card
                if confidence >= 0.8:
                    st.markdown(
                        f"""
                        <div class="status-card status-high">
                            <span style="font-size: 20px;">🟢</span>
                            <div>
                                <strong>High Confidence: {confidence*100:.1f}%</strong><br>
                                The model successfully matched the features.
                            </div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                elif confidence >= 0.5:
                    st.markdown(
                        f"""
                        <div class="status-card status-medium">
                            <span style="font-size: 20px;">🟡</span>
                            <div>
                                <strong>Medium Confidence: {confidence*100:.1f}%</strong><br>
                                Features are slightly ambiguous. Auto-inversion or transposition might need adjustments.
                            </div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="status-card status-low">
                            <span style="font-size: 20px;">🔴</span>
                            <div>
                                <strong>Low Confidence: {confidence*100:.1f}%</strong><br>
                                The character is unrecognized. Verify if the image contains multiple characters or complex backgrounds.
                            </div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                
                # Fetch current theme colors for Plotly styling
                theme_colors = config.get_current_colors()
                primary_color = theme_colors["primary"]
                secondary_color = theme_colors["secondary"]
                text_color = theme_colors["text"]
                
                # Gauge plot
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence * 100,
                    number={'suffix': "%", 'font': {'size': 22, 'family': 'Outfit', 'color': text_color}},
                    title={'text': "Confidence", 'font': {'size': 14, 'family': 'Poppins', 'color': text_color}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': text_color},
                        'bar': {'color': primary_color},
                        'bgcolor': "rgba(0,0,0,0.05)",
                        'steps': [
                            {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.15)'},
                            {'range': [50, 80], 'color': 'rgba(245, 158, 11, 0.15)'},
                            {'range': [80, 100], 'color': 'rgba(22, 197, 94, 0.15)'}
                        ],
                    }
                ))
                fig_gauge.update_layout(
                    height=145, 
                    margin=dict(l=10, r=10, t=25, b=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})
 
            with col_chart:
                st.markdown("### 📈 Class Probabilities")
                st.markdown("Top 5 predictions and their relative softmax probabilities:")
                
                top_5_df = pd.DataFrame(top_5, columns=["Character", "Probability"])
                top_5_df["Percentage"] = top_5_df["Probability"] * 100
                
                fig_bar = px.bar(
                    top_5_df, 
                    x="Percentage", 
                    y="Character", 
                    orientation='h',
                    text=top_5_df['Percentage'].apply(lambda x: f"{x:.1f}%"),
                    color="Percentage",
                    color_continuous_scale=[secondary_color, primary_color]
                )
                
                fig_bar.update_layout(
                    height=260,
                    margin=dict(l=10, r=40, t=10, b=10),
                    xaxis_title="Probability (%)",
                    yaxis_title="Character",
                    coloraxis_showscale=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(autorange="reversed", tickfont=dict(color=text_color)),
                    xaxis=dict(tickfont=dict(color=text_color), title=dict(font=dict(color=text_color)))
                )
                fig_bar.update_traces(
                    textposition='outside', 
                    textfont_size=11, 
                    textfont_color=text_color,
                    marker_line_color='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

            # Download history button
            st.markdown("---")
            st.markdown("### 📜 Prediction History")
            if st.session_state.prediction_history:
                history_df = pd.DataFrame(st.session_state.prediction_history)
                st.dataframe(
                    history_df.iloc[::-1],
                    column_order=["Timestamp", "Source", "Predicted Character", "Confidence", "Inference Time"],
                    use_container_width=True
                )
                csv_data = utils.export_history_to_csv()
                if csv_data:
                    st.download_button(
                        label="📥 Export History as CSV",
                        data=csv_data,
                        file_name="visionwrite_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
        except Exception as e:
            st.error(f"⚠️ Error processing the image: {e}. Please ensure the file is a valid image.")
            
    else:
        st.info("📤 Upload an image file to begin. For best results, use a single character with clear boundaries on a clean background.")
        
        # Default placeholder panel
        st.markdown(
            """
            <div style="border: 1px dashed var(--border-color); border-radius: 12px; height: 250px; display: flex; align-items: center; justify-content: center; color: var(--text-muted); margin-top: 20px;">
                <div style="text-align: center;">
                    <span style="font-size: 48px;">📤</span>
                    <p style="margin-top: 12px;">Awaiting file upload...</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
