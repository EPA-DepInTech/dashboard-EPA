import os

import streamlit as st


st.set_page_config(page_title="Dashboard - Grupo EPA", layout="wide")

CSS_PATH = os.path.join(os.path.dirname(__file__), "style", "style.css")
logo_path = os.path.join(os.path.dirname(__file__), "style", "epa_logo.png")

if os.path.exists(CSS_PATH):
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.warning(f"CSS não encontrado em: {CSS_PATH}")

if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        logo_bytes = f.read()
    st.logo(logo_bytes, icon_image=logo_bytes)

pages = [
    st.Page("pages/dashboard_page.py", title="Dashboard", icon="📊"),
    st.Page("pages/create_graph.py", title="Graficos", icon="📈"),
]

nav = st.navigation(pages)
nav.run()
