import pandas as pd
import plotly.express as px
import re
import streamlit as st
import os
import base64
import time 

from core.state import get_uploaded_file, init_session_state, set_uploaded_file
from services.dataset_service import build_dataset_from_excels

# ================== CONFIG ==================
st.set_page_config(page_title="Dashboard - Grupo EPA", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSS_PATH = os.path.join(BASE_DIR, "style.css")
logo_path = os.path.join(BASE_DIR, "epa_logo.png")

# ================== FUNÇÕES ==================
def load_image_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# ================== CSS ==================
if os.path.exists(CSS_PATH):
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.warning(f"CSS não encontrado em: {CSS_PATH}")

# ================== LOGO ==================
if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        logo_bytes = f.read()

    st.logo(
        logo_bytes,
        icon_image=logo_bytes,
    )

# ================== SESSION STATE ==================
init_session_state()

# ================== SPLASH SCREEN (2s) ==================
if "splash_shown" not in st.session_state:
    st.session_state["splash_shown"] = False

if not st.session_state["splash_shown"]:
    if os.path.exists(logo_path):
        logo_b64 = load_image_base64(logo_path)

        splash_html = f"""
        <div class="splash-container">
            <img src="data:image/png;base64,{logo_b64}" class="splash-logo" />
        </div>
        <div data-testid="stSidebarContent" 
        class="st-emotion-cache-155jwzh"></div>
        """
        placeholder = st.empty()
        placeholder.markdown(splash_html, unsafe_allow_html=True)
        time.sleep(2)
        placeholder.empty()

    st.session_state["splash_shown"] = True
    st.rerun()

# ================== TÍTULO ==================
st.title("📊 Dashboard de Amostras")

# ================== HELPERS ==================
def _process_upload(uploaded_files):
    if not uploaded_files:
        return

    # mantém referência aos uploads originais (para leitura dedicada em abas específicas)
    st.session_state["uploaded_files"] = uploaded_files

    set_uploaded_file(uploaded_files)
    result = build_dataset_from_excels(uploaded_files)

    if result.errors:
        st.error("Não foi possível processar os Excels:")
        for e in result.errors:
            st.write("-", e)
        return

    st.session_state["df_dict"] = result.df_dict  # padronizado
    # guarda datasets por arquivo para permitir seleção na página de gráficos
    df_dict_by_file: dict[str, dict] = {}
    for f in uploaded_files:
        single = build_dataset_from_excels([f])
        if single.errors:
            continue
        name = getattr(f, "name", f"arquivo_{len(df_dict_by_file)+1}")
        df_dict_by_file[name] = single.df_dict or {}
    if df_dict_by_file:
        st.session_state["df_dict_by_file"] = df_dict_by_file
    else:
        st.session_state.pop("df_dict_by_file", None)
    skipped_charts = [s for s in result.skipped if s.has_charts]
    if skipped_charts:
        st.success(f"Excel operacional carregado: {len(skipped_charts)} abas de grafico ignoradas.")
    else:
        st.success("Excel operacional carregado: nenhuma aba de grafico ignorada.")


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

