import os
import pickle
import time
import html
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st
import config

@st.cache_resource
def load_model():
    """Loads and caches the trained Keras CNN model."""
    try:
        model = tf.keras.models.load_model(config.MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_label_mapping():
    """Loads and caches the pickle label mapping."""
    try:
        with open(config.MAPPING_PATH, "rb") as f:
            mapping = pickle.load(f)
        return mapping
    except Exception as e:
        st.error(f"Error loading label mapping: {e}")
        return None

@st.cache_data
def load_training_history():
    """Loads and caches the training history."""
    try:
        with open(config.HISTORY_PATH, "rb") as f:
            history = pickle.load(f)
        return history
    except Exception as e:
        st.error(f"Error loading training history: {e}")
        return None

def preprocess_image(image_np, align_model=True):
    """
    Preprocesses a numpy image (from canvas or file upload) for the EMNIST CNN model.
    1. Converts to grayscale.
    2. Resizes to 28x28.
    3. Auto-inverts if it has a light background (EMNIST is white text on black background).
    4. Applies binary thresholding to denoise.
    5. Mirrors horizontally if align_model is True (empirical testing shows this model
       expects horizontally flipped inputs).
    6. Normalizes pixels to [0, 1].
    7. Reshapes to (1, 28, 28, 1).
    """
    # 1. Convert to grayscale if image is color
    if len(image_np.shape) == 3:
        if image_np.shape[2] == 4:  # RGBA
            # If it's canvas drawing, we want the drawn part.
            # In RGBA, if background is transparent (A=0), we can use the alpha channel
            # to isolate the drawing, or convert RGB to Grayscale.
            # Let's check if the alpha channel is active (some pixels have A < 255)
            alpha = image_np[:, :, 3]
            if np.min(alpha) < 255:
                # Use alpha channel as grayscale mask (white stroke on black background)
                gray = alpha.copy()
            else:
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGBA2GRAY)
        else:  # RGB
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np.copy()

    # 2. Resize to 28x28 using cubic interpolation (preserves stroke quality better than linear)
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_CUBIC)

    # 3. Auto-invert: EMNIST requires white text (255) on a black background (0).
    # If the average pixel value is high (>127), it is likely a dark-on-light image, so we invert.
    mean_val = np.mean(resized)
    if mean_val > 127:
        resized = cv2.bitwise_not(resized)

    # 4. Apply Otsu's thresholding or binary thresholding to denoise and binarize
    _, thresholded = cv2.threshold(resized, 30, 255, cv2.THRESH_BINARY)

    # 5. Model Alignment: This specific model was trained on horizontally flipped images.
    # We mirror the drawing horizontally to align with the model's learned weights.
    if align_model:
        processed = cv2.flip(thresholded, 1)
    else:
        processed = thresholded.copy()

    # 6. Normalize to [0, 1]
    normalized = processed.astype("float32") / 255.0

    # 7. Reshape to (1, 28, 28, 1)
    final_image = np.expand_dims(normalized, axis=(0, -1))

    # Return both the model input and the thresholded image (for visualization)
    return final_image, thresholded

def predict_character(model, mapping, processed_image):
    """
    Runs model inference on the preprocessed image.
    Returns:
        - predicted_char: The predicted character (str).
        - confidence: The confidence score (float, 0.0 to 1.0).
        - top_5: A list of tuples (char, probability) for the top 5 predictions.
        - prediction_time: Time taken for inference in seconds.
    """
    start_time = time.time()
    predictions = model.predict(processed_image)[0]
    prediction_time = time.time() - start_time

    # Get the index of the highest probability
    predicted_idx = int(np.argmax(predictions))
    predicted_char = mapping.get(predicted_idx, str(predicted_idx))
    confidence = float(predictions[predicted_idx])

    # Get top 5 predictions
    top_indices = np.argsort(predictions)[::-1][:5]
    top_5 = []
    for idx in top_indices:
        char = mapping.get(int(idx), str(idx))
        prob = float(predictions[idx])
        top_5.append((char, prob))

    return predicted_char, confidence, top_5, prediction_time

def add_to_history(char, confidence, inference_time, source):
    """Adds a prediction to the session state history and updates statistics."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    prediction_item = {
        "Timestamp": timestamp,
        "Source": source,
        "Predicted Character": char,
        "Confidence": f"{confidence * 100:.2f}%",
        "Raw Confidence": confidence,
        "Inference Time": f"{inference_time * 1000:.1f} ms",
        "Raw Time": inference_time
    }
    
    st.session_state.prediction_history.append(prediction_item)
    st.session_state.total_predictions = len(st.session_state.prediction_history)
    
    # Recalculate average confidence
    total_conf = sum(item["Raw Confidence"] for item in st.session_state.prediction_history)
    st.session_state.avg_confidence = total_conf / st.session_state.total_predictions

def export_history_to_csv():
    """Converts prediction history to a CSV string."""
    if not st.session_state.prediction_history:
        return None
    df = pd.DataFrame(st.session_state.prediction_history)
    return df.to_csv(index=False).encode('utf-8')

def render_prediction_transcript(history_items, max_items=12):
    """Renders prediction history as a chat-like transcript UI."""
    if not history_items:
        return ""

    message_blocks = []
    for item in list(reversed(history_items[-max_items:])):
        source = html.escape(str(item.get("Source", "Input")))
        timestamp = html.escape(str(item.get("Timestamp", "")))
        predicted_char = html.escape(str(item.get("Predicted Character", "-")))
        confidence = html.escape(str(item.get("Confidence", "-")))
        inference_time = html.escape(str(item.get("Inference Time", "-")))

        message_blocks.append(
            f"""
            <div class="message-row user">
                <div class="message-bubble">
                    <div class="message-meta">User Input • {timestamp}</div>
                    <div class="message-text">{source}</div>
                </div>
            </div>
            <div class="message-row assistant">
                <div class="message-bubble">
                    <div class="message-meta">VisionWrite AI</div>
                    <div class="message-text">
                        Predicted character: <strong>{predicted_char}</strong><br>
                        Confidence: {confidence} • Inference: {inference_time}
                    </div>
                </div>
            </div>
            """
        )

    return f"<div class='message-feed'>{''.join(message_blocks)}</div>"
