# app.py
import streamlit as st
import pandas as pd
from core.state import init_session_state, set_uploaded_file, get_uploaded_file
from services.dataset_service import build_dataset_from_excel
import numpy as np

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
        result = build_dataset_from_excel(uploaded)
        if result.errors:
            st.error("Não foi possível processar o Excel:")
            for e in result.errors: st.write("-", e)
        else:
            st.session_state["df"] = result.df_dict
            st.success("Arquivo carregado e processado")


st.divider()

st.subheader("Navegação rápida")
st.page_link("pages/create_graph.py", label="📈 Ir para o Criador de Gráfico")

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

    # st.dataframe(st.session_state['df'])
    df = st.session_state["df"]

    st.dataframe(df)