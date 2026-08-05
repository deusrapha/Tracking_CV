import streamlit as st
import streamlit.components.v1 as components
import os

st.set_page_config(
    page_title="Beyond the Line of Sight - Occlusion Tracking Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Header style tuning
st.markdown("""
<style>
    .reportview-container .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("Beyond the Line of Sight")
st.markdown("""
**Counterfactual Amodal Anchors for Occlusion-Aware Multi-Animal Tracking in UAV-Based Precision Livestock Monitoring**

This Streamlit application serves the interactive hardware-in-the-loop (HIL) tracking demonstration. It runs a CTRV kinematic prediction model, EKF state tracker, and Counterfactual Amodal Anchor reasoning system entirely client-side.
""")

# Load and bundle HTML, CSS, and JS files dynamically to render inside components.html iframe
current_dir = os.path.dirname(os.path.abspath(__file__))
demo_dir = os.path.join(current_dir, "demo")

html_path = os.path.join(demo_dir, "index.html")
css_path = os.path.join(demo_dir, "style.css")
js_path = os.path.join(demo_dir, "simulation.js")

if os.path.exists(html_path) and os.path.exists(css_path) and os.path.exists(js_path):
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    with open(css_path, "r", encoding="utf-8") as f:
        css_content = f.read()
    with open(js_path, "r", encoding="utf-8") as f:
        js_content = f.read()

    # 1. Inline CSS contents into style tag
    css_tag = f"<style>\n{css_content}\n</style>"
    html_content = html_content.replace('<link rel="stylesheet" href="style.css">', css_tag)

    # 2. Inline Javascript contents into script tag
    js_tag = f"<script>\n{js_content}\n</script>"
    html_content = html_content.replace('<script src="simulation.js?v=2"></script>', js_tag)

    # 3. Render self-contained application iframe
    components.html(html_content, height=1000, scrolling=True)
else:
    st.error("Error: Demo files (index.html, style.css, or simulation.js) could not be located in 'demo/' folder.")
