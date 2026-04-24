import base64
import time
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

if "splash_shown" not in st.session_state:
    st.session_state["splash_shown"] = False

if not st.session_state["splash_shown"]:
    if LOGO_PATH.exists():
        logo_b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode()
        splash_html = f"""
        <div class="splash-container">
            <img src="data:image/png;base64,{logo_b64}" class="splash-logo" />
        </div>
        """
        splash_placeholder = st.empty()
        splash_placeholder.markdown(splash_html, unsafe_allow_html=True)
        time.sleep(2)
        splash_placeholder.empty()

    st.session_state["splash_shown"] = True
    st.rerun()

if not hasattr(st, "Page") or not hasattr(st, "navigation"):
    st.error(
        "A versão do Streamlit não suporta st.Page/st.navigation. "
        "Atualize para uma versão compatível para usar o app multipáginas."
    )
    st.stop()

home_page = st.Page(str(BASE_DIR / "pages" / "dashboard_page.py"), title="Dashboard", icon=":material/dashboard:")
graph_operacional_page = st.Page(str(BASE_DIR / "pages" / "create_graph_operacional.py"), title="Operacional", icon=":material/query_stats:")
graph_visualizacao_page = st.Page(str(BASE_DIR / "pages" / "create_graph_visualizacao_aprofundada.py"), title="Visualizacao aprofundada", icon=":material/analytics:")
graph_in_situ_page = st.Page(str(BASE_DIR / "pages" / "create_graph_in_situ.py"), title="In situ", icon=":material/science:")
graph_in_situ_aprofundado_page = st.Page(str(BASE_DIR / "pages" / "create_graph_in_situ_aprofundado.py"), title="In situ aprofundado", icon=":material/monitoring:")
graph_laboratorial_page = st.Page(str(BASE_DIR / "pages" / "create_graph_laboratorial.py"), title="Laboratorial", icon=":material/biotech:")

pg = st.navigation(
    [
        home_page,
        graph_operacional_page,
        graph_visualizacao_page,
        graph_in_situ_page,
        graph_in_situ_aprofundado_page,
        graph_laboratorial_page,
    ],
    position="top",
)
pg.run()
