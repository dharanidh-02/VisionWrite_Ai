# 0. TensorFlow & MediaPipe Protobuf Compatibility Hack
# TensorFlow 2.20.0+ requires protobuf 5.x+ (for runtime_version).
# MediaPipe 0.10.x requires protobuf 4.x (due to C-extension FieldDescriptor.label).
# We use protobuf 4.25.3, and dynamically mock 'google.protobuf.runtime_version'
# at runtime so TensorFlow imports it successfully without crashing.
import sys
import types

# Create a dummy runtime_version module
runtime_version_module = types.ModuleType("google.protobuf.runtime_version")
runtime_version_module.Domain = type("Domain", (), {"PUBLIC": 0})
runtime_version_module.ValidateProtobufRuntimeVersion = lambda *args, **kwargs: None

# Inject it into sys.modules
sys.modules["google.protobuf.runtime_version"] = runtime_version_module
sys.modules["google.protobuf.runtime_version.Domain"] = runtime_version_module.Domain

# Inject into the google.protobuf namespace
try:
    import google.protobuf
    google.protobuf.runtime_version = runtime_version_module
except ImportError:
    pass

import os
import streamlit as st
import config
import utils

# 1. Initialize session state
config.init_session_state()

# 2. Set page config
st.set_page_config(
    page_title="VisionWrite AI",
    page_icon="✍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 3. Custom CSS Loading
def load_css():
    css_path = os.path.join("assets", "styles.css")
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            css = f.read()
    else:
        css = ""
    
    # Get current theme colors
    colors = config.get_current_colors()
    
    # Generate CSS overrides based on the selected theme
    theme_vars = f"""
    :root {{
        --bg-color: {colors["bg"]};
        --bg-gradient-1: {colors.get("bg_gradient_1", colors["bg"])};
        --bg-gradient-2: {colors.get("bg_gradient_2", colors["bg"])};
        --card-bg: {colors["card_bg"]};
        --text-color: {colors["text"]};
        --text-muted: {colors["text_muted"]};
        --primary-color: {colors["primary"]};
        --secondary-color: {colors["secondary"]};
        --accent-color: {colors.get("accent", "#7C3AED")};
        --success-color: {colors["success"]};
        --warning-color: {colors["warning"]};
        --danger-color: {colors["danger"]};
        --border-color: {colors["border"]};
        --shadow: {colors["shadow"]};
        --sidebar-bg: {colors.get("sidebar_bg", "#F1F5F9")};
        --skill-bg: {colors.get("skill_bg", "rgba(37, 99, 235, 0.08)")};
        --skill-color: {colors.get("skill_color", "#2563EB")};
        --skill-border: {colors.get("skill_border", "rgba(37, 99, 235, 0.15)")};
        --heading-color: {colors.get("heading", "#0F172A")};
        --btn-hover: {colors.get("btn_hover", "#1D4ED8")};
        --toggle-bg: {colors.get("toggle_bg", "#cbd5e1")};
    }}
    """
    st.markdown(f"<style>{theme_vars}\n{css}</style>", unsafe_allow_html=True)

# Apply styling
load_css()

# 4. Sidebar Top Section (Brand Title only, no image)
st.sidebar.markdown(
    '<div style="padding: 10px 0 5px 0;"><h2 style="margin:0; background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-family:\'Poppins\', sans-serif; font-size: 24px; font-weight: 700;">VisionWrite AI</h2></div>',
    unsafe_allow_html=True
)
st.sidebar.markdown("<p style='font-size:11px; margin-top:-15px; margin-bottom:10px; font-weight:500; color:var(--text-muted);'>Handwritten Character Recognition</p>", unsafe_allow_html=True)

# 5. Define Pages for Navigation
home_page = st.Page("pages/1_Home.py", title="Home", icon=":material/home:")
draw_page = st.Page("pages/2_Draw_and_Predict.py", title="Draw & Predict", icon=":material/edit:")
upload_page = st.Page("pages/3_Upload_Image.py", title="Upload Image", icon=":material/upload:")
perf_page = st.Page("pages/4_Model_Performance.py", title="Model Performance", icon=":material/bar_chart:")
about_page = st.Page("pages/5_About_Project.py", title="About Project", icon=":material/info:")
dev_page = st.Page("pages/6_Developer.py", title="Developer", icon=":material/person:")

# 6. Setup Navigation
pg = st.navigation({
    "VisionWrite AI": [home_page, draw_page, upload_page, perf_page, about_page, dev_page]
})

# 7. Sidebar Bottom Section (Settings, Stats, and Footer - Compact Layout)
st.sidebar.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)

# Theme Toggle Switch
dark_mode_active = st.sidebar.toggle(
    "🌙 Dark Mode", 
    value=(st.session_state.theme == "dark")
)

new_theme = "dark" if dark_mode_active else "light"
if st.session_state.theme != new_theme:
    st.session_state.theme = new_theme
    st.rerun()

# Session Stats (Inline & Compact)
st.sidebar.markdown(
    f"""
    <div style="font-size: 12px; margin: 8px 0; color: var(--text-muted);">
        Predictions: <strong style="color: var(--text-color);">{st.session_state.total_predictions}</strong> | 
        Avg Conf: <strong style="color: var(--text-color);">{st.session_state.avg_confidence * 100:.1f}%</strong>
    </div>
    """,
    unsafe_allow_html=True
)

# Footer & Developer Profile Card (Compact & Integrated)
st.sidebar.markdown(
    """
    <div style="display: flex; align-items: center; gap: 10px; padding: 8px 12px; background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 10px; margin-top: 10px; box-shadow: var(--shadow);">
        <div style="width: 32px; height: 32px; border-radius: 50%; background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%); display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 11px; font-family: 'Outfit', sans-serif;">
            DT
        </div>
        <div style="flex: 1; min-width: 0;">
            <div style="font-weight: 600; font-size: 12px; color: var(--text-color); text-overflow: ellipsis; overflow: hidden; white-space: nowrap; font-family: 'Poppins', sans-serif;">Dharanidharan T</div>
            <div style="font-size: 10px; color: var(--text-muted); text-overflow: ellipsis; overflow: hidden; white-space: nowrap; font-family: 'Inter', sans-serif;">AI & ML Developer</div>
        </div>
    </div>
    <p style='font-size:9px; text-align:center; color:var(--text-muted); margin: 6px 0 0 0;'>VisionWrite AI © 2026 | CodeAlpha</p>
    """, 
    unsafe_allow_html=True
)

# 8. Run active page
pg.run()
