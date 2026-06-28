import html
import pickle
import time
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

import config


@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model(config.MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_resource
def load_label_mapping():
    try:
        with open(config.MAPPING_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading label mapping: {e}")
        return None


@st.cache_data
def load_training_history():
    try:
        with open(config.HISTORY_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading training history: {e}")
        return None


def preprocess_image(image_np, align_model=True):
    if len(image_np.shape) == 3:
        if image_np.shape[2] == 4:
            alpha = image_np[:, :, 3]
            gray = alpha.copy() if np.min(alpha) < 255 else cv2.cvtColor(image_np, cv2.COLOR_RGBA2GRAY)
        else:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np.copy()

    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_CUBIC)
    if np.mean(resized) > 127:
        resized = cv2.bitwise_not(resized)

    _, thresholded = cv2.threshold(resized, 30, 255, cv2.THRESH_BINARY)
    processed = cv2.flip(thresholded, 1) if align_model else thresholded.copy()
    normalized = processed.astype("float32") / 255.0
    final_image = np.expand_dims(normalized, axis=(0, -1))
    return final_image, thresholded


def predict_character(model, mapping, processed_image):
    start_time = time.time()
    predictions = model.predict(processed_image, verbose=0)[0]
    prediction_time = time.time() - start_time

    predicted_idx = int(np.argmax(predictions))
    predicted_char = mapping.get(predicted_idx, str(predicted_idx))
    confidence = float(predictions[predicted_idx])

    top_indices = np.argsort(predictions)[::-1][:5]
    top_5 = [(mapping.get(int(i), str(i)), float(predictions[i])) for i in top_indices]
    return predicted_char, confidence, top_5, prediction_time


def add_to_history(char, confidence, inference_time, source):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.prediction_history.append(
        {
            "Timestamp": timestamp,
            "Source": source,
            "Predicted Character": char,
            "Confidence": f"{confidence * 100:.2f}%",
            "Raw Confidence": confidence,
            "Inference Time": f"{inference_time * 1000:.1f} ms",
            "Raw Time": inference_time,
        }
    )
    st.session_state.total_predictions = len(st.session_state.prediction_history)
    total_conf = sum(item["Raw Confidence"] for item in st.session_state.prediction_history)
    st.session_state.avg_confidence = total_conf / st.session_state.total_predictions


def export_history_to_csv():
    if not st.session_state.prediction_history:
        return None
    return pd.DataFrame(st.session_state.prediction_history).to_csv(index=False).encode("utf-8")


def render_prediction_transcript(history_items, max_items=12):
    if not history_items:
        return ""

    blocks = []
    for item in list(reversed(history_items[-max_items:])):
        source = html.escape(str(item.get("Source", "Input")))
        timestamp = html.escape(str(item.get("Timestamp", "")))
        predicted_char = html.escape(str(item.get("Predicted Character", "-")))
        confidence = html.escape(str(item.get("Confidence", "-")))
        inference_time = html.escape(str(item.get("Inference Time", "-")))

        blocks.append(
            f"""
            <div class="message-row user"><div class="message-bubble"><div class="message-meta">Input • {timestamp}</div><div class="message-text">{source}</div></div></div>
            <div class="message-row assistant"><div class="message-bubble"><div class="message-meta">VisionWrite AI</div><div class="message-text">Prediction: <strong>{predicted_char}</strong><br>Confidence: {confidence} • Time: {inference_time}</div></div></div>
            """
        )

    return f"<div class='message-feed'>{''.join(blocks)}</div>"


def render_page_hero(title: str, subtitle: str, stats: list[tuple[str, str]] | None = None):
    stats_html = ""
    if stats:
        stats_html = "".join(
            [
                f"<div class='hero-stat'><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>"
                for label, value in stats
            ]
        )

    st.markdown(
        f"""
        <section class="hero-card">
            <div class="hero-content">
                <p class="hero-kicker">VisionWrite AI</p>
                <h1>{html.escape(title)}</h1>
                <p>{html.escape(subtitle)}</p>
                <div class="hero-stats">{stats_html}</div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def get_plotly_layout(title: str = "", height: int = 320, show_legend: bool = True) -> dict[str, Any]:
    colors = config.get_current_colors()
    return {
        "template": colors["plotly_template"],
        "title": {"text": title, "font": {"family": "Poppins", "size": 18, "color": colors["text_primary"]}},
        "height": height,
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Inter", "color": colors["text_secondary"]},
        "margin": {"l": 16, "r": 16, "t": 50, "b": 24},
        "showlegend": show_legend,
        "legend": {"orientation": "h", "y": 1.02, "x": 0},
    }


def confidence_message(confidence: float):
    if confidence >= 0.8:
        return "High", "The model found a strong feature match.", "status-high"
    if confidence >= 0.5:
        return "Medium", "Prediction is usable but slightly ambiguous.", "status-medium"
    return "Low", "Input is noisy or unclear for reliable classification.", "status-low"


def render_searchable_table(df: pd.DataFrame, key: str, page_size: int = 8):
    if df.empty:
        st.info("No data available yet.")
        return

    search_query = st.text_input("Search", key=f"{key}_search", placeholder="Filter rows...")
    filtered = df.copy()
    if search_query:
        mask = filtered.astype(str).apply(lambda c: c.str.contains(search_query, case=False, na=False))
        filtered = filtered[mask.any(axis=1)]

    total_rows = len(filtered)
    total_pages = max((total_rows - 1) // page_size + 1, 1)
    page = st.selectbox("Page", options=list(range(1, total_pages + 1)), key=f"{key}_page")
    start = (page - 1) * page_size
    end = start + page_size

    st.dataframe(filtered.iloc[start:end], use_container_width=True, hide_index=True)
    st.caption(f"Showing {min(end, total_rows)} of {total_rows} row(s)")
