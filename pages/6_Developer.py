import streamlit as st

import utils

utils.render_page_hero(
    "Developer Portfolio",
    "Professional profile of the VisionWrite AI developer, including skills, projects, and contact channels.",
    stats=[("Role", "AI/ML Developer"), ("Focus", "Computer Vision"), ("Primary Stack", "Python + TensorFlow")],
)

left, right = st.columns([1, 1.6], gap="large")

with left:
    st.markdown(
        """
        <div class='profile-card' style='text-align:center;'>
            <div style='width:100px;height:100px;border-radius:50%;margin:0 auto 14px;display:grid;place-items:center;background:linear-gradient(135deg,var(--accent-primary),var(--accent-secondary));color:#fff;font-family:Outfit,sans-serif;font-size:34px;font-weight:700;'>DT</div>
            <h3 style='margin:0;'>Dharanidharan T</h3>
            <p style='margin:8px 0 0 0;'>AI & Machine Learning Developer</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class='surface-card'>
            <h4 style='margin-top:0;'>Contact</h4>
            <p>Email: <a href='mailto:dharanidh02@gmail.com'>dharanidh02@gmail.com</a></p>
            <p>GitHub: <a href='https://github.com/dharanidh-02' target='_blank'>github.com/dharanidh-02</a></p>
            <p>LinkedIn: <a href='https://www.linkedin.com/in/dharanidharan-t-180a79321/' target='_blank'>LinkedIn Profile</a></p>
            <p>Portfolio: <a href='https://my-portfolio-delta-seven-lmmpyciz39.vercel.app/#' target='_blank'>View Portfolio</a></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    st.markdown(
        """
        <div class='surface-card'>
            <h3 style='margin-top:0;'>About</h3>
            <p>Computer Science and Business Systems student focused on AI products that combine deep learning with polished user-facing systems.</p>
            <h4>Education</h4>
            <p>Sri Eshwar College of Engineering — Computer Science & Business Systems</p>
            <h4>Experience</h4>
            <p>CodeAlpha Machine Learning Internship</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    skills = [
        "Python",
        "TensorFlow",
        "Keras",
        "OpenCV",
        "Deep Learning",
        "Computer Vision",
        "Streamlit",
        "Plotly",
        "Machine Learning",
    ]
    skill_html = "".join([
        f"<span style='display:inline-block;margin:6px 6px 0 0;padding:8px 12px;border-radius:999px;border:1px solid var(--border);background:var(--bg-secondary);font-size:14px;'>{s}</span>"
        for s in skills
    ])
    st.markdown(
        f"""
        <div class='surface-card'>
            <h3 style='margin-top:0;'>Skills</h3>
            <div>{skill_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class='surface-card'>
            <h3 style='margin-top:0;'>Highlighted Projects</h3>
            <p><strong>VisionWrite AI:</strong> Handwritten character recognition workspace with responsive premium UI and CNN inference.</p>
            <p><strong>Portfolio Projects:</strong> Additional AI and web projects are available in the public portfolio and GitHub profile.</p>
            <a href='https://my-portfolio-delta-seven-lmmpyciz39.vercel.app/#' target='_blank'>Open Resume / Portfolio</a>
        </div>
        """,
        unsafe_allow_html=True,
    )
