import pandas as pd
import plotly.express as px
import re
import streamlit as st
import time 

from core.state import get_uploaded_file, init_session_state, set_uploaded_file
from services.dataset_service import build_dataset_from_excels
from PIL import Image

#=========== Configurações da página ============
st.set_page_config(
    page_title="Dashboard - Grupo EPA",page_icon='🌳',layout="wide")

# ===== Splash Screen (executa só uma vez por sessão) =====
if "splash_done" not in st.session_state:
    st.session_state.splash_done = True

    logo = Image.open("app/logo.png")
    placeholder = st.empty()

    with placeholder.container():
        st.markdown("##")  # espaçamento
        col1, col2, col3 = st.columns([2, 2, 1])
        with col2:
            st.image(logo, width=350)

    time.sleep(2.5)

    placeholder.empty()
#=========== Título da página ============
st.title("Dashboard Operacional - Grupo EPA")

def _process_upload(uploaded_files):
    if not uploaded_files:
        return

    set_uploaded_file(uploaded_files)
    result = build_dataset_from_excels(uploaded_files)

    if result.errors:
        st.error("Não foi possível processar os Excels:")
        for e in result.errors:
            st.write("-", e)
        return

    st.session_state["df_dict"] = result.df_dict
    st.success("Excel operacional carregado com sucesso!")

def _norm_text(value: object) -> str:
    s = str(value).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def _find_col(df: pd.DataFrame, tokens: list[str]) -> str | None:
    for col in df.columns:
        n = _norm_text(col)
        if all(t in n for t in tokens):
            return col
    return None

# ================== OVERVIEW ==================
def _build_overview(df_dict: dict[str, pd.DataFrame]) -> None:
    na_raw = df_dict.get("NA Semanal")
    if na_raw is None or na_raw.empty:
        st.info("NA Semanal não encontrado.")
        return

    df = na_raw.copy()
    date_col = _find_col(df, ["data"])
    poco_col = _find_col(df, ["poco"]) or _find_col(df, ["ponto"])

    if not date_col or not poco_col:
        st.info("Colunas de Data/Poço não encontradas.")
        return

    df["Data"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Data"])
    df["poco_key"] = df[poco_col].astype(str).str.strip().str.upper()

    measure_cols = [c for c in ["NA (m)", "NO (m)", "FL (m)"] if c in df.columns]
    df["has_measure"] = df[measure_cols].notna().any(axis=1) if measure_cols else True

    total_samples = int(df["has_measure"].sum())

    date_min = df["Data"].min()
    date_max = df["Data"].max()

    if pd.notna(date_min) and pd.notna(date_max):
        period_months = (date_max.to_period("M") - date_min.to_period("M")).n + 1
        period_label = f"{date_min.date()} a {date_max.date()}"
    else:
        period_months = 0
        period_label = "Sem datas válidas"

    st.subheader("Resumo Operacional")

    left, right = st.columns(2)

    with left:
        st.metric("Período", f"{period_months} meses", period_label)
        st.metric("Amostras de NA", f"{total_samples} medições")

    with right:
        st.info("Gráficos adicionais carregados após processamento.")

# ================== SIDEBAR ==================
with st.sidebar:
    st.header("Entrada de dados")
    st.info("Use o botão abaixo para enviar o Excel.")

st.subheader("Carregar Excel")
uploaded_main = st.file_uploader(
    "Selecione os arquivos (.xlsx)",
    type=["xlsx"],
    accept_multiple_files=True,
    key="uploader_main",
)

_process_upload(uploaded_main)

file_in_state = get_uploaded_file()

if not file_in_state:
    st.info("Nenhum arquivo carregado ainda.")
    st.stop()

df_dict = st.session_state.get("df_dict")

if isinstance(df_dict, dict):
    _build_overview(df_dict)

    if st.button("📈 Criar gráfico", use_container_width=True):
        try:
            st.switch_page("pages/create_graph.py")
        except Exception:
            st.error("Erro ao abrir página de gráficos.")
