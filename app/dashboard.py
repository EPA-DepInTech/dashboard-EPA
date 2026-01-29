# dashboard.py
import streamlit as st

from core.state import get_uploaded_file, init_session_state, set_uploaded_file
from services.dataset_service import (
    build_dataset_from_excel,
)

st.set_page_config(page_title="Dashboard de Amostras", layout="wide")
init_session_state()

st.title("📊 Dashboard de Amostras")
def _process_upload(uploaded_file):
    """Processa o excel enviado e atualiza estado + feedback."""
    if uploaded_file is None:
        return

    set_uploaded_file(uploaded_file)
    result = build_dataset_from_excel(uploaded_file)

    if result.errors:
        st.error("Não foi possível processar o Excel:")
        for e in result.errors:
            st.write("-", e)
        return

    st.session_state["df_dict"] = result.df_dict  # padronizado
    if result.warnings:
        st.warning("Arquivo carregado, mas com avisos:")
        for w in result.warnings:
            st.write("-", w)
    if result.skipped:
        with st.expander("⚠️ Abas ignoradas"):
            for s in result.skipped:
                st.caption(f"- **{s.sheet}** → {s.reason}")
    st.success("Arquivo carregado e processado")


# ================ SIDE BAR ===============
with st.sidebar:
    st.header("Entrada de dados")
    st.info("Use o botão na tela principal para enviar o Excel.")

st.divider()

st.subheader("Carregar Excel")
uploaded_main = st.file_uploader(
    "Selecione o arquivo (.xlsx)",
    type=["xlsx"],
    accept_multiple_files=False,
    key="uploader_main",
    label_visibility="visible",
)
_process_upload(uploaded_main)

file_in_state = get_uploaded_file()

if file_in_state is None:
    st.info("Nenhum arquivo carregado ainda. Use o menu lateral para enviar seu Excel.")
    st.stop()

df_dict = st.session_state.get("df_dict")
if df_dict is None:
    st.info("Arquivo foi carregado, mas ainda não há dataset processado em memória.")
    st.stop()

if isinstance(df_dict, dict):
    st.success(" Excel Operacional Carregado")
    if st.button("📈 Criar gráfico", use_container_width=True):
        try:
            st.switch_page("pages/create_graph.py")
        except Exception:
            st.error("Erro ao abrir página de gráficos.")
