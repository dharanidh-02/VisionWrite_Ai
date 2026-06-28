import os
import streamlit as st

# Project Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "Ai_models")

MODEL_PATH = os.path.join(MODEL_DIR, "visionwrite_ai_emnist.keras")
MAPPING_PATH = os.path.join(MODEL_DIR, "label_mapping.pkl")
HISTORY_PATH = os.path.join(MODEL_DIR, "training_history.pkl")

# Theme Colors (Premium HSL-based palettes)
# Light Mode Colors (SaaS Minimalist + Healthcare AI)
LIGHT_THEME = {
    "bg": "#F8FAFC",
    "bg_gradient_1": "#F8FAFC",
    "bg_gradient_2": "#FFFFFF",
    "card_bg": "#FFFFFF",
    "text": "#334155",
    "text_muted": "#64748B",
    "primary": "#2563EB",
    "secondary": "#4F46E5",
    "accent": "#7C3AED",
    "success": "#16A34A",
    "warning": "#F59E0B",
    "danger": "#DC2626",
    "border": "#E2E8F0",
    "shadow": "0 10px 15px -3px rgba(0, 0, 0, 0.03), 0 4px 6px -4px rgba(0, 0, 0, 0.03)",
    "sidebar_bg": "#F1F5F9",
    "skill_bg": "rgba(37, 99, 235, 0.08)",
    "skill_color": "#2563EB",
    "skill_border": "rgba(37, 99, 235, 0.15)",
    "heading": "#0F172A",
    "btn_hover": "#1D4ED8",
    "toggle_bg": "#1E293B"
}

# Dark Mode Colors (SaaS Cyber AI + Premium Dark)
DARK_THEME = {
    "bg": "#0F172A",
    "bg_gradient_1": "#0F172A",
    "bg_gradient_2": "#111827",
    "card_bg": "#1E293B",
    "text": "#CBD5E1",
    "text_muted": "#94A3B8",
    "primary": "#60A5FA",
    "secondary": "#818CF8",
    "accent": "#A78BFA",
    "success": "#22C55E",
    "warning": "#FBBF24",
    "danger": "#F87171",
    "border": "#334155",
    "shadow": "0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -4px rgba(0, 0, 0, 0.3)",
    "sidebar_bg": "#020617",
    "skill_bg": "rgba(96, 165, 250, 0.1)",
    "skill_color": "#60A5FA",
    "skill_border": "rgba(96, 165, 250, 0.2)",
    "heading": "#F8FAFC",
    "btn_hover": "#2563EB",
    "toggle_bg": "#475569"
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
