import os
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "Ai_models")

MODEL_PATH = os.path.join(MODEL_DIR, "visionwrite_ai_emnist.keras")
MAPPING_PATH = os.path.join(MODEL_DIR, "label_mapping.pkl")
HISTORY_PATH = os.path.join(MODEL_DIR, "training_history.pkl")

LIGHT_THEME = {
    "bg": "#FFFFFF",
    "bg_secondary": "#F8FAFC",
    "sidebar_bg": "#F9FAFB",
    "card_bg": "#FFFFFF",
    "border": "#E5E7EB",
    "text_primary": "#111827",
    "text_secondary": "#4B5563",
    "text_muted": "#6B7280",
    "accent_primary": "#2563EB",
    "accent_secondary": "#4F46E5",
    "success": "#22C55E",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "hover": "#F3F4F6",
    "selection": "#DBEAFE",
    "shadow": "0 12px 30px -24px rgba(15, 23, 42, 0.25)",
    "plotly_template": "plotly_white",
}

DARK_THEME = {
    "bg": "#0F172A",
    "bg_secondary": "#111827",
    "sidebar_bg": "#020617",
    "card_bg": "#1E293B",
    "border": "#334155",
    "text_primary": "#F8FAFC",
    "text_secondary": "#CBD5E1",
    "text_muted": "#94A3B8",
    "accent_primary": "#60A5FA",
    "accent_secondary": "#A78BFA",
    "success": "#22C55E",
    "warning": "#FBBF24",
    "danger": "#F87171",
    "hover": "#1F2937",
    "selection": "#1E3A8A",
    "shadow": "0 14px 34px -22px rgba(2, 6, 23, 0.6)",
    "plotly_template": "plotly_dark",
}

EMNIST_METADATA = {
    "name": "EMNIST Balanced",
    "total_classes": 47,
    "total_samples": 131600,
    "train_samples": 112800,
    "test_samples": 18800,
    "model_type": "Convolutional Neural Network (CNN)",
    "framework": "TensorFlow / Keras",
    "input_shape": "28 x 28 x 1 (Grayscale)",
    "accuracy": "87.5% (approx)",
}


def init_session_state():
    if "theme" not in st.session_state:
        st.session_state.theme = "light"
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []
    if "total_predictions" not in st.session_state:
        st.session_state.total_predictions = 0
    if "avg_confidence" not in st.session_state:
        st.session_state.avg_confidence = 0.0


def get_current_colors():
    return DARK_THEME if st.session_state.get("theme", "light") == "dark" else LIGHT_THEME
