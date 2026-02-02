# dashboard.py
import pandas as pd
import plotly.express as px
import re
import streamlit as st

from core.state import get_uploaded_file, init_session_state, set_uploaded_file
from services.dataset_service import (
    build_dataset_from_excels,
)

st.set_page_config(page_title="Dashboard de Amostras", layout="wide")
init_session_state()

st.title("📊 Dashboard de Amostras")
def _process_upload(uploaded_files):
    """Processa os excels enviados e atualiza estado + feedback."""
    if not uploaded_files:
        return

    set_uploaded_file(uploaded_files)
    result = build_dataset_from_excels(uploaded_files)

    if result.errors:
        st.error("Não foi possível processar os Excels:")
        for e in result.errors:
            st.write("-", e)
        return

    st.session_state["df_dict"] = result.df_dict  # padronizado
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


def _build_overview(df_dict: dict[str, pd.DataFrame]) -> None:
    na_raw = df_dict.get("NA Semanal")
    if na_raw is None or na_raw.empty:
        st.info("Nao foi possivel gerar o resumo: NA Semanal nao encontrado.")
        return

    df = na_raw.copy()
    date_col = _find_col(df, ["data"])
    poco_col = _find_col(df, ["poco"]) or _find_col(df, ["ponto"])
    if not date_col or not poco_col:
        st.info("Nao foi possivel gerar o resumo: colunas de Data/Poço nao encontradas.")
        return

    df["Data"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Data"])
    df["poco_key"] = df[poco_col].astype(str).str.strip().str.upper()

    measure_cols = [c for c in ["NA (m)", "NO (m)", "FL (m)"] if c in df.columns]
    if measure_cols:
        df["has_measure"] = df[measure_cols].notna().any(axis=1)
    else:
        df["has_measure"] = True

    total_samples = int(df["has_measure"].sum())

    date_min = df["Data"].min()
    date_max = df["Data"].max()
    if pd.notna(date_min) and pd.notna(date_max):
        period_months = (date_max.to_period("M") - date_min.to_period("M")).n + 1
        period_label = f"{date_min.date()} a {date_max.date()}"
    else:
        period_months = 0
        period_label = "Sem datas validas"

    def _row_has_seco(row: pd.Series) -> bool:
        check_cols = ["NA (m)", "NO (m)", "FL (m)", "Status", "Observacao", "Observação"]
        for col in check_cols:
            if col not in row:
                continue
            v = row.get(col)
            if pd.isna(v):
                continue
            if "seco" in _norm_text(v):
                return True
        return False

    last_rows = df.sort_values("Data").groupby("poco_key", as_index=False).tail(1)
    last_rows["is_dry"] = last_rows.apply(_row_has_seco, axis=1)
    dry_count = int(last_rows["is_dry"].sum())
    normal_count = max(0, int(last_rows.shape[0]) - dry_count)

    df["tipo_poco"] = df["poco_key"].map(lambda x: str(x)[:2] if x else "")
    type_counts = (
        df[df["tipo_poco"] != ""]
        .drop_duplicates(subset=["poco_key"])
        .groupby("tipo_poco")["poco_key"]
        .count()
        .reset_index()
        .rename(columns={"poco_key": "quantidade"})
    )

    measurements_by_month = (
        df.assign(Mes=df["Data"].dt.to_period("M").astype(str))
        .groupby("Mes")["has_measure"]
        .sum()
        .reset_index()
        .rename(columns={"has_measure": "Amostras"})
    )

    st.subheader("Resumo Operacional")
    left, right = st.columns(2, gap="large")

    with left:
        st.metric("Periodo", f"{period_months} meses", period_label)
        st.metric("Amostras de NA", f"{total_samples} medicoes")
        if not measurements_by_month.empty:
            fig_month = px.bar(
                measurements_by_month,
                x="Mes",
                y="Amostras",
                title="Amostras por mes",
            )
            fig_month.update_layout(margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_month, use_container_width=True)

    with right:
        dry_df = pd.DataFrame(
            {"Status": ["Seco", "Normal"], "Quantidade": [dry_count, normal_count]}
        )
        fig_dry = px.pie(
            dry_df,
            names="Status",
            values="Quantidade",
            hole=0.45,
            title="Pocos na ultima medicao",
        )
        fig_dry.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_dry, use_container_width=True)

        if not type_counts.empty:
            fig_types = px.pie(
                type_counts,
                names="tipo_poco",
                values="quantidade",
                hole=0.4,
                title="Tipos de pocos (prefixo)",
            )
            fig_types.update_layout(margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_types, use_container_width=True)

    # with st.expander("Sugestoes de outras visualizacoes para esta pagina"):
    #     st.write(
    #         "- Linha do tempo de eventos (ex.: seco, odor, oleoso) por poco\n"
    #         "- Heatmap de amostras por poco x mes\n"
    #         "- Top 10 pocos com maior variacao de NA\n"
    #         "- Distribuicao de profundidade (boxplot) por tipo de poco\n"
    #         "- Comparativo de NA medio vs volume bombeado por mes"
    #     )


# ================ SIDE BAR ===============
with st.sidebar:
    st.header("Entrada de dados")
    st.info("Use o botão na tela principal para enviar o Excel.")

st.divider()

st.subheader("Carregar Excel")
uploaded_main = st.file_uploader(
    "Selecione os arquivos (.xlsx)",
    type=["xlsx"],
    accept_multiple_files=True,
    key="uploader_main",
    label_visibility="visible",
)
_process_upload(uploaded_main)

file_in_state = get_uploaded_file()

if not file_in_state:
    st.info("Nenhum arquivo carregado ainda. Use o menu lateral para enviar seu Excel.")
    st.stop()

df_dict = st.session_state.get("df_dict")
if df_dict is None:
    st.info("Arquivo foi carregado, mas ainda não há dataset processado em memória.")
    st.stop()

if isinstance(df_dict, dict):
    _build_overview(df_dict)
    if st.button("📈 Criar gráfico", use_container_width=True):
        try:
            st.switch_page("pages/create_graph.py")
        except Exception:
            st.error("Erro ao abrir página de gráficos.")
