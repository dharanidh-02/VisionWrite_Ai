import os
import streamlit as st

# Project Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "Ai_models")

MODEL_PATH = os.path.join(MODEL_DIR, "visionwrite_ai_emnist.keras")
MAPPING_PATH = os.path.join(MODEL_DIR, "label_mapping.pkl")
HISTORY_PATH = os.path.join(MODEL_DIR, "training_history.pkl")

# Theme Colors (Professional neutral palettes)
LIGHT_THEME = {
    "bg": "#F4F7FB",
    "bg_gradient_1": "#F7F9FC",
    "bg_gradient_2": "#FFFFFF",
    "card_bg": "#FFFFFF",
    "text": "#1F2937",
    "text_muted": "#6B7280",
    "primary": "#2563EB",
    "secondary": "#1D4ED8",
    "accent": "#0EA5E9",
    "success": "#16A34A",
    "warning": "#F59E0B",
    "danger": "#DC2626",
    "border": "#DCE3EC",
    "shadow": "0 14px 30px -24px rgba(15, 23, 42, 0.45)",
    "sidebar_bg": "#EEF3FA",
    "skill_bg": "rgba(37, 99, 235, 0.08)",
    "skill_color": "#2563EB",
    "skill_border": "rgba(37, 99, 235, 0.18)",
    "heading": "#0F172A",
    "btn_hover": "#1D4ED8",
    "toggle_bg": "#334155"
}

# Dark Mode Colors (enterprise dark)
DARK_THEME = {
    "bg": "#0B1220",
    "bg_gradient_1": "#0B1220",
    "bg_gradient_2": "#111B2E",
    "card_bg": "#111A2B",
    "text": "#E2E8F0",
    "text_muted": "#94A3B8",
    "primary": "#60A5FA",
    "secondary": "#3B82F6",
    "accent": "#38BDF8",
    "success": "#22C55E",
    "warning": "#FBBF24",
    "danger": "#FB7185",
    "border": "#263244",
    "shadow": "0 18px 36px -26px rgba(2, 6, 23, 0.9)",
    "sidebar_bg": "#070E1C",
    "skill_bg": "rgba(96, 165, 250, 0.1)",
    "skill_color": "#60A5FA",
    "skill_border": "rgba(96, 165, 250, 0.24)",
    "heading": "#F8FAFC",
    "btn_hover": "#2563EB",
    "toggle_bg": "#334155"
}

# Dataset Metadata
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
    """Initializes global session states for theme, prediction history, and stats."""
    if "theme" not in st.session_state:
        st.session_state.theme = "light"
    
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []
    
    if "total_predictions" not in st.session_state:
        st.session_state.total_predictions = 0
        
    if "avg_confidence" not in st.session_state:
        st.session_state.avg_confidence = 0.0

def get_current_colors():
    """Returns the HSL color palette based on current theme selection."""
    if st.session_state.get("theme", "light") == "dark":
        return DARK_THEME
    return LIGHT_THEME
