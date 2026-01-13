# app.py
import streamlit as st
from core.state import init_session_state, set_uploaded_file, get_uploaded_file
from services.dataset_service import build_dataset_from_excel, format_datetime_columns_for_display, remove_accumulated_rows

st.set_page_config(page_title="Dashboard de Amostras", layout="wide")
init_session_state()

st.title("📊 Dashboard de Amostras")

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
            for e in result.errors:
                st.write("-", e)
        else:
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

st.divider()

if st.button("📈 Criar gráfico", use_container_width=True):
    try:
        st.switch_page("pages/create_graph.py")
    except Exception:
        st.error("Erro ao abrir página de gráficos.")

file_in_state = get_uploaded_file()

if file_in_state is None:
    st.info("Nenhum arquivo carregado ainda. Use o menu lateral para enviar seu Excel.")
    st.stop()

df_dict = st.session_state.get("df_dict")
if not df_dict:
    st.info("Arquivo foi carregado, mas ainda não há dataset processado em memória.")
    st.stop()

selected = st.selectbox("Tabela:", list(df_dict.keys()))
df = df_dict[selected]

st.caption(f"{selected} — {len(df)} linhas | {len(df.columns)} colunas")
# Remove linhas com "Acumulado" e formata datas para exibição (remove hora)
df_display = remove_accumulated_rows(df)
df_display = format_datetime_columns_for_display(df_display)
st.dataframe(df_display, use_container_width=True)
