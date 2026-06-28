import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import utils
import config

# Set page title
st.markdown("<h1 class='gradient-text'>✍️ Draw & Predict</h1>", unsafe_allow_html=True)
st.markdown("Choose between drawing on the interactive canvas or writing in the air using your webcam.")

# Load Model and Mapping
model = utils.load_model()
mapping = utils.load_label_mapping()

# Define HSV Color Ranges for Webcam Tracking (Fallback)
COLOR_RANGES = {
    "Blue": {
        "lower": np.array([94, 80, 20]),
        "upper": np.array([126, 255, 255]),
        "color_bgr": (255, 0, 0)
    },
    "Red": {
        "lower": np.array([160, 100, 100]),
        "upper": np.array([180, 255, 255]),
        "color_bgr": (0, 0, 255)
    },
    "Green": {
        "lower": np.array([36, 50, 50]),
        "upper": np.array([89, 255, 255]),
        "color_bgr": (0, 255, 0)
    },
    "Yellow": {
        "lower": np.array([15, 100, 100]),
        "upper": np.array([35, 255, 255]),
        "color_bgr": (0, 255, 255)
    }
}

if model is None or mapping is None:
    st.error("⚠️ Failed to load the AI model or label mapping. Please check the 'Ai_models' directory.")
else:
    # Create Tabs
    tab_canvas, tab_camera = st.tabs(["🎨 Canvas Drawing", "📷 Air Writing (Webcam)"])

    # ==================== TAB 1: CANVAS DRAWING ====================
    with tab_canvas:
        # Set up layout: Left column for canvas, Right column for results
        col_canvas, col_results = st.columns([1, 1.2], gap="large")

        with col_canvas:
            st.markdown("### 🎨 Drawing Board")
            
            # Stroke width slider
            stroke_width = st.slider("Brush Thickness", min_value=8, max_value=30, value=18, step=1, key="canvas_stroke_slider")
            
            # Use a key that we can increment to force-reset the canvas
            if "canvas_reset_key" not in st.session_state:
                st.session_state.canvas_reset_key = 0
                
            # Optional Model Alignment toggle
            align_toggle = st.toggle("Enable Model Alignment (Recommended)", value=True, 
                                     help="Empirical testing shows this model was trained on horizontally mirrored images. Leaving this enabled aligns your drawing with the model's expected orientation.",
                                     key="canvas_align_toggle")

            # Drawing canvas widget
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0.0)",  # Fixed fill color
                stroke_width=stroke_width,
                stroke_color="#FFFFFF",
                background_color="#000000",
                height=320,
                width=320,
                drawing_mode="freedraw",
                key=f"canvas_{st.session_state.canvas_reset_key}",
                update_streamlit=True,
            )
            
            # Action buttons in a row
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                clear_btn = st.button("🧹 Clear Canvas", use_container_width=True, key="canvas_clear_btn")
                if clear_btn:
                    st.session_state.canvas_reset_key += 1
                    st.rerun()
                    
            with btn_col2:
                reset_btn = st.button("🔄 Reset Page", use_container_width=True, key="canvas_reset_page_btn")
                if reset_btn:
                    st.session_state.canvas_reset_key += 1
                    st.session_state.prediction_history = []
                    st.session_state.total_predictions = 0
                    st.session_state.avg_confidence = 0.0
                    st.toast("Application state has been reset!", icon="🔄")
                    st.rerun()

        # Process and Predict
        with col_results:
            st.markdown("### 🔮 Recognition Results")
            
            # Check if drawing has occurred
            if canvas_result.image_data is not None and np.any(canvas_result.image_data[:, :, :3] > 0):
                # Preprocess the canvas image
                raw_img = canvas_result.image_data
                processed_img, thresholded_preview = utils.preprocess_image(raw_img, align_model=align_toggle)
                
                # Predict
                pred_char, confidence, top_5, inference_time = utils.predict_character(model, mapping, processed_img)
                
                # Add to history (only if this is a new drawing and not already added)
                curr_signature = f"{pred_char}_{confidence:.4f}"
                if "last_sig" not in st.session_state or st.session_state.last_sig != curr_signature:
                    utils.add_to_history(pred_char, confidence, inference_time, "Canvas Drawing")
                    st.session_state.last_sig = curr_signature
                    st.toast(f"Character recognized: '{pred_char}'", icon="🎯")
                
                # 1. Main prediction display
                res_col1, res_col2 = st.columns([1, 1.5])
                with res_col1:
                    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                    st.markdown("##### Predicted Character")
                    st.markdown(f"<div class='prediction-large'>{pred_char}</div>", unsafe_allow_html=True)
                    st.markdown(f"**Inference Time:** `{inference_time * 1000:.1f} ms`")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                with res_col2:
                    # Fetch current theme colors for Plotly styling
                    theme_colors = config.get_current_colors()
                    primary_color = theme_colors["primary"]
                    secondary_color = theme_colors["secondary"]
                    text_color = theme_colors["text"]
                    
                    # Plotly gauge for confidence
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=confidence * 100,
                        number={'suffix': "%", 'font': {'size': 24, 'family': 'Outfit', 'color': text_color}},
                        title={'text': "Confidence Score", 'font': {'size': 14, 'family': 'Poppins', 'color': text_color}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': text_color},
                            'bar': {'color': primary_color},
                            'bgcolor': "rgba(0,0,0,0.05)",
                            'borderwidth': 2,
                            'bordercolor': text_color,
                            'steps': [
                                {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.15)'},
                                {'range': [50, 80], 'color': 'rgba(245, 158, 11, 0.15)'},
                                {'range': [80, 100], 'color': 'rgba(22, 197, 94, 0.15)'}
                            ],
                        }
                    ))
                    fig_gauge.update_layout(
                        height=180, 
                        margin=dict(l=20, r=20, t=35, b=10),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})
                
                # 2. Confidence Status Card
                if confidence >= 0.8:
                    st.markdown(
                        f"""
                        <div class="status-card status-high">
                            <span style="font-size: 20px;">🟢</span>
                            <div>
                                <strong>High Confidence: {confidence*100:.1f}%</strong><br>
                                The model is highly confident in this prediction. The strokes match class patterns closely.
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
                                The model has detected some ambiguity. Try drawing with clearer strokes or adjust brush thickness.
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
                                The model is uncertain. This could be due to a complex character, unusual angle, or noise.
                            </div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    
                # 3. Top 5 Probabilities Bar Chart
                st.markdown("##### Top 5 Predictions")
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
                    height=200,
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
                
                # 4. Preview of EMNIST preprocessed image
                with st.expander("🔍 View Preprocessed Model Input (28x28)"):
                    st.write("This is exactly how the Convolutional Neural Network sees your drawing:")
                    col_preview1, col_preview2 = st.columns(2)
                    with col_preview1:
                        st.image(thresholded_preview, caption="Resized & Thresholded", width=120)
                    with col_preview2:
                        # Show aligned model input
                        st.image(processed_img[0, :, :, 0], caption="EMNIST Input (Aligned)", width=120)
                        
            else:
                st.info("🎨 Use your mouse or touchscreen to draw a character in the board on the left. The AI will analyze it instantly.")
                
                st.markdown(
                    """
                    <div style="border: 1px dashed var(--border-color); border-radius: 12px; height: 250px; display: flex; align-items: center; justify-content: center; color: var(--text-muted);">
                        <div style="text-align: center;">
                            <span style="font-size: 48px;">🔮</span>
                            <p style="margin-top: 12px;">Awaiting drawing input...</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # 5. Prediction History
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
                    use_container_width=True,
                    key="canvas_export_btn"
                )
        else:
            st.write("No predictions recorded in this session yet.")

    # ==================== TAB 2: AIR WRITING (WEBCAM) ====================
    with tab_camera:
        st.markdown("### 📷 Air Writing (Virtual Canvas)")
        st.write("Write in the air using your index finger (tracked by AI) or fall back to tracking a colored object.")

        # Layout: Left column for controls, Right column for camera and HUD
        cam_col_left, cam_col_right = st.columns([1, 2.2], gap="large")

        with cam_col_left:
            st.markdown("##### ⚙️ Settings")
            
            # Start/Stop toggle
            cam_active = st.toggle("Activate Webcam", value=False, key="webcam_active_toggle")
            
            # Tracking Mode Selection
            tracking_mode = st.selectbox(
                "Tracking Mode",
                options=["👉 Finger Tracking (MediaPipe)", "🎨 Color Tracking (HSV)"],
                index=0,
                key="webcam_tracking_mode",
                help="Finger Tracking uses your webcam to detect your bare hand. Color Tracking tracks a specific colored object."
            )
            
            # Conditionally show color selector
            if tracking_mode == "🎨 Color Tracking (HSV)":
                track_color = st.selectbox(
                    "Select Object Color",
                    options=list(COLOR_RANGES.keys()),
                    index=0,
                    key="webcam_color_select"
                )
                
                # Color Bar indicator
                st.markdown(
                    """
                    <div style="display: flex; gap: 10px; margin-top: 5px; margin-bottom: 15px;">
                        <div style="background-color: #3b82f6; width: 24px; height: 24px; border-radius: 50%; border: 2px solid white;" title="Blue"></div>
                        <div style="background-color: #ef4444; width: 24px; height: 24px; border-radius: 50%; border: 2px solid white;" title="Red"></div>
                        <div style="background-color: #10b981; width: 24px; height: 24px; border-radius: 50%; border: 2px solid white;" title="Green"></div>
                        <div style="background-color: #fbbc05; width: 24px; height: 24px; border-radius: 50%; border: 2px solid white;" title="Yellow"></div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                track_color = "Blue" # Default fallback
                st.info("💡 **How to Draw:** Pinch your **index finger** and **thumb** together to draw. Separate them to hover (pause drawing).")

            # Brush Thickness
            air_brush_size = st.slider("Air Brush Thickness", min_value=4, max_value=24, value=12, step=1, key="webcam_brush_slider")
            
            # Clear Word
            if st.button("🗑️ Clear Word", use_container_width=True, key="webcam_clear_word_btn"):
                st.session_state.predicted_word = ""
                st.toast("Word cleared!")
                st.rerun()

            st.markdown(
                """
                **💡 Gestures & Features:**
                - **Virtual CLEAR Button**: Move your finger into the red **CLEAR** box in the top-left of the camera feed to clear the drawing.
                - **Auto-Append**: Finish drawing a letter and hide your hand/object from the camera for 1.5 seconds. It will auto-predict and append the character to the word.
                """
            )

        with cam_col_right:
            # HUD Card at the top (Futuristic Dashboard)
            if "predicted_word" not in st.session_state:
                st.session_state.predicted_word = ""
                
            # Placeholders for HUD values
            hud_container = st.empty()
            
            def update_hud(word, current_char, status, color):
                # Fetch current theme dynamically
                theme = st.session_state.get("theme", "dark")
                if theme == "light":
                    hud_bg = "#ffffff"
                    hud_border = "#e2e8f0"
                    hud_label = "#64748b"
                    hud_word_color = "#1e40af"  # Deep blue for high contrast in light mode
                    hud_char_color = "#6d28d9"  # Deep purple for high contrast in light mode
                    hud_shadow = "0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -2px rgba(0,0,0,0.05)"
                else:
                    hud_bg = "#1e293b"
                    hud_border = "#334155"
                    hud_label = "#94a3b8"
                    hud_word_color = "#00f2fe"  # Neon cyan for ultra-high visibility in dark mode
                    hud_char_color = "#c084fc"  # Bright purple for high contrast in dark mode
                    hud_shadow = "0 10px 15px -3px rgba(0,0,0,0.3), 0 4px 6px -4px rgba(0,0,0,0.3)"
                
                hud_container.markdown(
                    f"""
                    <div style="background: {hud_bg}; border: 1px solid {hud_border}; border-radius: 16px; padding: 15px; box-shadow: {hud_shadow}; margin-bottom: 15px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid {hud_border}; padding-bottom: 10px; margin-bottom: 10px;">
                            <span style="font-size: 11px; color: {hud_label}; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em;">SYSTEM STATUS</span>
                            <span style="background-color: {color}; color: #ffffff; font-size: 11px; font-weight: 700; padding: 3px 10px; border-radius: 20px; text-transform: uppercase;">
                                {status}
                            </span>
                        </div>
                        <div style="display: flex; gap: 20px; align-items: center;">
                            <div style="flex: 2; border-right: 1px solid {hud_border}; padding-right: 10px;">
                                <span style="font-size: 11px; color: {hud_label}; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em;">Recognized Word</span>
                                <h2 style="margin: 5px 0 0 0; font-family: 'Outfit', sans-serif; letter-spacing: 1px; font-size: 28px; color: {hud_word_color}; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                                    {word if word else "[AWAITING DRAWING]"}
                                </h2>
                            </div>
                            <div style="flex: 1; text-align: center;">
                                <span style="font-size: 11px; color: {hud_label}; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em;">Active Letter</span>
                                <h2 style="margin: 5px 0 0 0; font-family: 'Outfit', sans-serif; font-size: 32px; color: {hud_char_color};">
                                    {current_char if current_char else "-"}
                                </h2>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Initial HUD state
            update_hud(st.session_state.predicted_word, "", "SYSTEM INACTIVE", "#64748b")
            
            if cam_active:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("🔌 Could not access your webcam. Make sure it is plugged in and not in use by another app.")
                else:
                    # Optimize camera settings for high speed, low latency, and lower CPU usage
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                    
                    # Streamlit image placeholder
                    cam_frame_placeholder = st.empty()
                    
                    # Read first frame to get dimensions
                    ret, test_frame = cap.read()
                    if ret:
                        h, w, _ = test_frame.shape
                    else:
                        h, w = 240, 320
                        
                    # Initialize air canvas
                    air_canvas = np.zeros((h, w, 3), dtype=np.uint8)
                    
                    prev_x, prev_y = None, None
                    frames_empty = 0
                    running_prediction = ""
                    running_confidence = 0.0
                    
                    # Initialize MediaPipe Hands if selected
                    hands = None
                    mp_hands = None
                    mp_draw = None
                    
                    if tracking_mode == "👉 Finger Tracking (MediaPipe)":
                        import mediapipe as mp
                        import mediapipe.python.solutions.hands as mp_hands
                        import mediapipe.python.solutions.drawing_utils as mp_draw
                        hands = mp_hands.Hands(
                            static_image_mode=False,
                            max_num_hands=1,
                            model_complexity=0,  # Use the fastest, most lightweight model for high FPS
                            min_detection_confidence=0.45,
                            min_tracking_confidence=0.45
                        )
                    
                    # Capture loop variables
                    frame_counter = 0
                    
                    while cam_active:
                        ret, frame = cap.read()
                        if not ret:
                            break
                            
                        frame_counter += 1
                        
                        # Mirror frame for intuitive mirrored movement
                        frame = cv2.flip(frame, 1)
                        
                        center = None
                        gesture_status = "✋ HOVERING"
                        gesture_color = "#38bdf8"
                        
                        # ---------------- FINGER TRACKING MODE ----------------
                        if tracking_mode == "👉 Finger Tracking (MediaPipe)" and hands is not None:
                            # Convert BGR to RGB for MediaPipe
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            results = hands.process(rgb_frame)
                            
                            if results.multi_hand_landmarks:
                                # Get landmarks of the first hand
                                hand_landmarks = results.multi_hand_landmarks[0]
                                
                                # Draw skeletal joints
                                mp_draw.draw_landmarks(
                                    frame, 
                                    hand_landmarks, 
                                    mp_hands.HAND_CONNECTIONS,
                                    mp_draw.DrawingSpec(color=(192, 132, 252), thickness=2, circle_radius=2), # Purple joints
                                    mp_draw.DrawingSpec(color=(96, 165, 250), thickness=2)                 # Blue bones
                                )
                                
                                # Get Index Finger Tip (8) and Thumb Tip (4) coordinates
                                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                                
                                ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                                tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
                                
                                center = (ix, iy)
                                
                                # Distance between index tip and thumb tip (for pinch gesture)
                                distance = np.hypot(ix - tx, iy - ty)
                                
                                # If pinched, activate drawing. Threshold is relative to frame width (6% of width)
                                is_drawing = distance < (w * 0.06)
                                
                                if is_drawing:
                                    # Draw green indicator line between thumb and index
                                    cv2.line(frame, (ix, iy), (tx, ty), (16, 185, 129), 3)
                                    cv2.circle(frame, center, 8, (16, 185, 129), -1)
                                    
                                    # Draw on canvas
                                    if prev_x is not None and prev_y is not None:
                                        cv2.line(air_canvas, (prev_x, prev_y), center, (255, 255, 255), air_brush_size)
                                        
                                    prev_x, prev_y = center
                                    frames_empty = 0
                                    gesture_status = "🖊️ WRITING"
                                    gesture_color = "#10b981"
                                else:
                                    # Draw red warning line (not pinching)
                                    cv2.line(frame, (ix, iy), (tx, ty), (239, 68, 68), 1)
                                    cv2.circle(frame, center, 6, (96, 165, 250), -1)
                                    
                                    prev_x, prev_y = None, None
                                    frames_empty += 1
                                    gesture_status = "✋ HOVERING"
                                    gesture_color = "#38bdf8"
                            else:
                                prev_x, prev_y = None, None
                                frames_empty += 1
                                gesture_status = "💤 NO HAND DETECTED"
                                gesture_color = "#64748b"
                                
                        # ---------------- COLOR TRACKING MODE ----------------
                        else:
                            # Convert to HSV color space
                            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                            
                            # Apply color mask
                            color_data = COLOR_RANGES[track_color]
                            mask = cv2.inRange(hsv, color_data["lower"], color_data["upper"])
                            
                            # Morphological operations to clean noise
                            mask = cv2.erode(mask, None, iterations=2)
                            mask = cv2.dilate(mask, None, iterations=2)
                            
                            # Find contours of the tracked color
                            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
                            if len(contours) > 0:
                                c = max(contours, key=cv2.contourArea)
                                ((x, y), radius) = cv2.minEnclosingCircle(c)
                                
                                if radius > 12:
                                    M = cv2.moments(c)
                                    if M["m00"] > 0:
                                        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                                        
                                        # Draw cursor and tracking ring on webcam feed
                                        cv2.circle(frame, center, int(radius), (255, 255, 255), 2)
                                        cv2.circle(frame, center, 6, color_data["color_bgr"], -1)
                                        
                                        # Draw on canvas if we have a continuous line
                                        if prev_x is not None and prev_y is not None:
                                            cv2.line(air_canvas, (prev_x, prev_y), center, (255, 255, 255), air_brush_size)
                                            
                                        prev_x, prev_y = center
                                        frames_empty = 0
                                        gesture_status = "🖊️ WRITING"
                                        gesture_color = "#10b981"
                            else:
                                prev_x, prev_y = None, None
                                frames_empty += 1
                                gesture_status = "💤 OBJECT LOST"
                                gesture_color = "#64748b"
                                
                        # Overlay the air drawing onto the webcam frame in color
                        if tracking_mode == "👉 Finger Tracking (MediaPipe)":
                            draw_color = (168, 85, 247) # Glowing Purple for Finger Drawing
                        else:
                            draw_color = COLOR_RANGES[track_color]["color_bgr"]
                            
                        gray_canvas = cv2.cvtColor(air_canvas, cv2.COLOR_BGR2GRAY)
                        _, thresh_canvas = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY)
                        frame[thresh_canvas > 0] = draw_color
                        
                        # Draw Virtual CLEAR Button (top-left)
                        cv2.rectangle(frame, (15, 15), (120, 65), (239, 68, 68), -1)
                        cv2.putText(frame, "CLEAR", (32, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                        
                        # Check if pointer hovers over virtual CLEAR button
                        if center is not None:
                            if 15 <= center[0] <= 120 and 15 <= center[1] <= 65:
                                air_canvas = np.zeros((h, w, 3), dtype=np.uint8)
                                prev_x, prev_y = None, None
                                running_prediction = ""
                                running_confidence = 0.0
                                st.toast("Air Canvas Cleared!", icon="🧹")
                                
                        # Run real-time background prediction if canvas has drawing
                        if np.any(air_canvas > 0):
                            # Predict only once every 20 frames (approx. every 1 second) to maintain high FPS.
                            # This prevents Keras from blocking the video capture loop on every frame.
                            if frame_counter % 20 == 0:
                                # Preprocess the air canvas drawing
                                processed_air_img, _ = utils.preprocess_image(air_canvas, align_model=True)
                                pred_char_air, conf_air, _, _ = utils.predict_character(model, mapping, processed_air_img)
                                running_prediction = pred_char_air
                                running_confidence = conf_air
                                
                        # Update the OCR HUD at the top
                        update_hud(
                            st.session_state.predicted_word, 
                            running_prediction if np.any(air_canvas > 0) else "", 
                            gesture_status, 
                            gesture_color
                        )
                        
                        # Auto-Append logic:
                        # If no drawing detected for ~1.5 seconds (45 empty frames) and canvas contains drawings
                        if frames_empty > 45 and np.any(air_canvas > 0):
                            # Run a final, dedicated prediction on the complete drawing
                            processed_air_img, _ = utils.preprocess_image(air_canvas, align_model=True)
                            final_pred, final_conf, _, _ = utils.predict_character(model, mapping, processed_air_img)
                            
                            if final_pred:
                                if final_conf > 0.30:  # Forgiving threshold
                                    st.session_state.predicted_word += final_pred
                                    st.toast(f"Appended: '{final_pred}' ({final_conf*100:.1f}%)", icon="🎯")
                                    utils.add_to_history(final_pred, final_conf, 0.1, "Air Writing")
                                else:
                                    st.toast(f"Prediction discarded (low confidence: {final_conf*100:.1f}%)", icon="⚠️")
                                
                            # ALWAYS clear canvas and reset state so it doesn't get stuck!
                            air_canvas = np.zeros((h, w, 3), dtype=np.uint8)
                            running_prediction = ""
                            running_confidence = 0.0
                            prev_x, prev_y = None, None
                            st.rerun()
                                
                        # Display the processed frame
                        cam_frame_placeholder.image(frame, channels="BGR", use_container_width=True)
                        
                    cap.release()
            else:
                # Inactive camera state placeholder
                st.markdown(
                    """
                    <div style="border: 1px dashed var(--border-color); border-radius: 12px; height: 350px; display: flex; align-items: center; justify-content: center; color: var(--text-muted);">
                        <div style="text-align: center;">
                            <span style="font-size: 64px;">📷</span>
                            <p style="margin-top: 15px; font-weight: 500;">Toggle "Activate Webcam" on the left to start Air Writing.</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
