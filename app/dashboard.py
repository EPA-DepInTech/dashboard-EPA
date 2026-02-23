from pathlib import Path

import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
CSS_PATH = BASE_DIR / "style" / "style.css"
LOGO_PATH = BASE_DIR / "style" / "epa_logo.png"

st.set_page_config(page_title="Dashboard - Grupo EPA", layout="wide")

if CSS_PATH.exists():
    st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    st.session_state["_global_css_loaded"] = True
else:
    st.warning(f"CSS não encontrado em: {CSS_PATH}")

if LOGO_PATH.exists():
    logo_bytes = LOGO_PATH.read_bytes()
    st.logo(logo_bytes, icon_image=logo_bytes)
    st.session_state["_global_logo_loaded"] = True

if not hasattr(st, "Page") or not hasattr(st, "navigation"):
    st.error(
        "A versão do Streamlit não suporta st.Page/st.navigation. "
        "Atualize para uma versão compatível para usar o app multipáginas."
    )
    st.stop()

home_page = st.Page(str(BASE_DIR / "pages" / "dashboard_page.py"), title="Dashboard", icon=":material/dashboard:")
graph_page = st.Page(str(BASE_DIR / "pages" / "create_graph.py"), title="Criar gráfico", icon=":material/query_stats:")

pg = st.navigation({"Pages": [home_page, graph_page]})
pg.run()
