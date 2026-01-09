# app.py
import streamlit as st
import pandas as pd
from core.state import init_session_state, set_uploaded_file, get_uploaded_file

st.set_page_config(
    page_title="Dashboard de Amostras",
    layout="wide",
)

init_session_state()

st.title("Dashboard de Amostras (Fase 1)")
st.caption("Envie um Excel para iniciar. Em breve: validação, gráficos e layouts customizáveis.")


# ================ SIDE BAR ===============

with st.sidebar:
    st.header("Entrada de dados")
    uploaded = st.file_uploader(
        "Upload do Excel (.xlsx)",
        type=["xlsx"],
        accept_multiple_files=False
    )

    if uploaded is not None:
        set_uploaded_file(uploaded)
        st.success("Arquivo carregado")


st.divider()

file_in_state = get_uploaded_file()

if file_in_state is None:
    st.info("Nenhum arquivo carregado ainda. Use o menu lateral para enviar seu Excel.")
else:
    st.subheader("Informações do arquivo")
    st.write(
        {
            "nome": file_in_state.name,
            "tamanho_kb": round(file_in_state.size / 1024, 2),
            "tipo": file_in_state.type,
        }
    )

    df = pd.read_excel(file_in_state)

    st.dataframe(df)
