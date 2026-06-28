import streamlit as st

# Set page title (Compact and centered)
st.markdown("<h2 class='gradient-text' style='text-align: center; margin-bottom: 0;'>👨‍💻 Developer Profile</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 14px; margin-top: 5px; color: var(--text-muted);'>Meet the developer behind VisionWrite AI</p>", unsafe_allow_html=True)

st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

# Centered single-card layout that fits on the screen
col_space1, col_card, col_space2 = st.columns([0.5, 3, 0.5])

with col_card:
    # Glassmorphic Profile Card
    st.markdown("<div class='profile-card' style='padding: 24px;'>", unsafe_allow_html=True)
    
    # Split content horizontally to prevent scrolling
    left_col, right_col = st.columns([1.2, 1.8], gap="large")
    
    with left_col:
        # Avatar (Styled circle with theme gradient)
        st.markdown(
            """
            <div style="display: flex; justify-content: center; margin-top: 5px; margin-bottom: 15px;">
                <div style="width: 90px; height: 90px; border-radius: 50%; background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%); display: flex; align-items: center; justify-content: center; color: white; font-size: 32px; font-weight: 700; box-shadow: var(--shadow);">
                    DT
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Name & Title
        st.markdown("<h3 style='text-align: center; margin-bottom: 0; font-size: 20px; color: var(--heading-color);'>Dharanidharan T</h3>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: var(--text-muted); font-weight: 600; font-size: 13px; margin-top: 4px; margin-bottom: 15px;'>AI & Machine Learning Developer</p>", unsafe_allow_html=True)
        
        # Social & Professional Links (Shield Badges in a Row)
        st.markdown(
            """
            <div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 6px; margin-top: 10px;">
                <a href="https://my-portfolio-delta-seven-lmmpyciz39.vercel.app/#" target="_blank"><img src="https://img.shields.io/badge/Portfolio-0d1117?style=flat-square&logo=vercel&logoColor=white" alt="Portfolio"></a>
                <a href="mailto:dharanidh02@gmail.com"><img src="https://img.shields.io/badge/dharanidh02%40gmail.com-EA4335?style=flat-square&logo=gmail&logoColor=white" alt="Gmail"></a>
                <a href="https://www.linkedin.com/in/dharanidharan-t-180a79321/" target="_blank"><img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white" alt="LinkedIn"></a>
                <a href="https://leetcode.com/u/dharanidh_02/" target="_blank"><img src="https://img.shields.io/badge/LeetCode-FFA116?style=flat-square&logo=leetcode&logoColor=black" alt="LeetCode"></a>
                <a href="https://www.kaggle.com/dharanidharant" target="_blank"><img src="https://img.shields.io/badge/Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white" alt="Kaggle"></a>
                <a href="https://www.skillrack.com/faces/resume.xhtml?id=514797&key=79cd5fc9a977158524d4eaee835e8e0bcb367584" target="_blank"><img src="https://img.shields.io/badge/SkillRack-00C853?style=flat-square&logo=codeforces&logoColor=white" alt="SkillRack"></a>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with right_col:
        # Academic & Professional Details
        st.markdown("<h4 style='margin-top: 0; margin-bottom: 12px; color: var(--primary-color); font-size: 16px;'>🏫 Academic & Professional</h4>", unsafe_allow_html=True)
        
        # Details in a clean, compact layout
        st.markdown(
            """
            <div style="font-size: 13px; line-height: 1.6; color: var(--text-color);">
                <div style="margin-bottom: 6px;">🎓 <b>College:</b> Sri Eshwar College of Engineering</div>
                <div style="margin-bottom: 6px;">💼 <b>Department:</b> Computer Science & Business Systems</div>
                <div style="margin-bottom: 6px;">🏢 <b>Internship:</b> CodeAlpha Machine Learning Internship</div>
                <div style="margin-bottom: 6px;">🎯 <b>Specialization:</b> Deep Learning & Computer Vision</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown("<div style='margin-top: 15px; border-top: 1px solid var(--border-color); padding-top: 12px;'></div>", unsafe_allow_html=True)
        
        # Technical Skills Grid (using the custom CSS skill badges)
        st.markdown("<h4 style='margin-top: 0; margin-bottom: 10px; color: var(--primary-color); font-size: 16px;'>🛠️ Technical Expertise</h4>", unsafe_allow_html=True)
        skills = [
            "Python", "TensorFlow", "Keras", "OpenCV", 
            "Deep Learning", "Computer Vision", "Streamlit", 
            "Plotly", "Machine Learning"
        ]
        skills_html = "".join([f'<span class="skill-badge" style="margin: 2px; font-size: 11.5px; padding: 4px 10px;">{skill}</span>' for skill in skills])
        st.markdown(f'<div style="line-height: 1.8;">{skills_html}</div>', unsafe_allow_html=True)
        
    st.markdown("</div>", unsafe_allow_html=True)
