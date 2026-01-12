# app.py
import streamlit as st
from core.state import init_session_state, set_uploaded_file, get_uploaded_file
from services.dataset_service import build_dataset_from_excel, format_datetime_columns_for_display, remove_accumulated_rows

st.set_page_config(page_title="Dashboard de Amostras", layout="wide")
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
            for e in result.errors:
                st.write("-", e)
        else:
            st.session_state["df_dict"] = result.df_dict  # padronizado
            if result.warnings:
                st.warning("Arquivo carregado, mas com avisos:")
                for w in result.warnings:
                    st.write("-", w)
            st.success("Arquivo carregado e processado")

st.divider()

st.subheader("Navegação rápida")

# ✅ Evita o crash do st.page_link em alguns cenários (se registry falhar)
# Preferimos switch_page quando disponível.
col1, col2 = st.columns([1, 3])
with col1:
    go = st.button("📈 Criar gráfico", use_container_width=True)
with col2:
    st.caption("Abrir a página de criação de gráficos.")

if go:
    try:
        st.switch_page("pages/4_Criar_Grafico.py")
    except Exception:
        # fallback (não derruba o app)
        st.warning("Não foi possível navegar automaticamente. Abra a página 'Criar gráfico' pelo menu lateral do Streamlit.")

file_in_state = get_uploaded_file()

if file_in_state is None:
    st.info("Nenhum arquivo carregado ainda. Use o menu lateral para enviar seu Excel.")
    st.stop()

st.subheader("Informações do arquivo")
st.write(
    {
        "nome": file_in_state.name,
        "tamanho_kb": round(file_in_state.size / 1024, 2),
        "tipo": file_in_state.type,
    }
)

df_dict = st.session_state.get("df_dict")
if not df_dict:
    st.info("Arquivo foi carregado, mas ainda não há dataset processado em memória.")
    st.stop()

st.subheader("Tabelas carregadas")
st.write(list(df_dict.keys()))

# Preview selecionável
selected = st.selectbox("Selecione uma tabela para visualizar:", list(df_dict.keys()))
df = df_dict[selected]

st.caption(f"Preview: `{selected}` — linhas: {len(df)} | colunas: {len(df.columns)}")
# Remove linhas com "Acumulado" e formata datas para exibição (remove hora)
df_display = remove_accumulated_rows(df)
df_display = format_datetime_columns_for_display(df_display)
st.dataframe(df_display, use_container_width=True)

result = build_dataset_from_excel(uploaded)

if result.errors:
    ...
else:
    st.session_state["df_dict"] = result.df_dict
    if result.warnings:
        ...
    if result.skipped:
        with st.expander("Abas ignoradas (não tabulares / gráficos)"):
            for s in result.skipped:
                st.write(f"- **{s.sheet}** → {s.reason} (charts={s.has_charts}, sample={s.non_empty_cells_sample})")

if result.skipped:
    with st.expander("Abas ignoradas (gráficos/sem tabela)"):
        for s in result.skipped:
            st.write(f"- **{s.sheet}** → {s.reason}")
