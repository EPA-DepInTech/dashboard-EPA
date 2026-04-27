from pathlib import Path
import runpy

import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
st.set_page_config(initial_sidebar_state="expanded")
st.session_state["create_graph_hide_sidebar"] = False
st.session_state["create_graph_subpage"] = "In situ aprofundado"
runpy.run_path(str(BASE_DIR / "create_graph.py"), run_name="__main__")
