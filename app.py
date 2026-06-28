import os
import sys
import types

import streamlit as st

import config

runtime_version_module = types.ModuleType("google.protobuf.runtime_version")
runtime_version_module.Domain = type("Domain", (), {"PUBLIC": 0})
runtime_version_module.ValidateProtobufRuntimeVersion = lambda *args, **kwargs: None
sys.modules["google.protobuf.runtime_version"] = runtime_version_module
sys.modules["google.protobuf.runtime_version.Domain"] = runtime_version_module.Domain

try:
    import google.protobuf

    google.protobuf.runtime_version = runtime_version_module
except ImportError:
    pass

config.init_session_state()

st.set_page_config(
    page_title="VisionWrite AI",
    page_icon="assets/logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css():
    css_path = os.path.join("assets", "styles.css")
    css = ""
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()

    c = config.get_current_colors()
    theme_vars = f"""
    :root {{
        --bg: {c['bg']};
        --bg-secondary: {c['bg_secondary']};
        --sidebar-bg: {c['sidebar_bg']};
        --card-bg: {c['card_bg']};
        --border: {c['border']};
        --text-primary: {c['text_primary']};
        --text-secondary: {c['text_secondary']};
        --text-muted: {c['text_muted']};
        --accent-primary: {c['accent_primary']};
        --accent-secondary: {c['accent_secondary']};
        --success: {c['success']};
        --warning: {c['warning']};
        --danger: {c['danger']};
        --hover: {c['hover']};
        --selection: {c['selection']};
        --shadow: {c['shadow']};
    }}
    """
    st.markdown(f"<style>{theme_vars}\n{css}</style>", unsafe_allow_html=True)


load_css()

st.sidebar.markdown(
    """
    <div class='sidebar-brand'>
        <div class='sidebar-brand-mark'>VW</div>
        <div>
            <h2>VisionWrite AI</h2>
            <p>Character Intelligence Platform</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

home_page = st.Page("pages/1_Home.py", title="Home", icon=":material/home:")
draw_page = st.Page("pages/2_Draw_and_Predict.py", title="Draw & Predict", icon=":material/draw:")
upload_page = st.Page("pages/3_Upload_Image.py", title="Upload Image", icon=":material/upload:")
perf_page = st.Page("pages/4_Model_Performance.py", title="Model Performance", icon=":material/monitoring:")
about_page = st.Page("pages/5_About_Project.py", title="About Project", icon=":material/description:")
dev_page = st.Page("pages/6_Developer.py", title="Developer", icon=":material/person:")

pg = st.navigation(
    {
        "Workspace": [home_page, draw_page, upload_page, perf_page, about_page, dev_page],
    }
)

with st.sidebar:
    st.markdown("<div class='sidebar-section-label'>Appearance</div>", unsafe_allow_html=True)
    dark_mode_active = st.toggle("Dark Theme", value=(st.session_state.theme == "dark"))
    new_theme = "dark" if dark_mode_active else "light"
    if st.session_state.theme != new_theme:
        st.session_state.theme = new_theme
        st.rerun()

    st.markdown(
        f"""
        <div class='sidebar-stats'>
            <div><span>Predictions</span><strong>{st.session_state.total_predictions}</strong></div>
            <div><span>Avg Confidence</span><strong>{st.session_state.avg_confidence * 100:.1f}%</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class='sidebar-profile'>
            <div class='avatar'>DT</div>
            <div>
                <h4>Dharanidharan T</h4>
                <p>AI & ML Developer</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

pg.run()
