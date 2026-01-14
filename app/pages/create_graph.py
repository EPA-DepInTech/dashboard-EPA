# pages/create_graph.py
import streamlit as st
import pandas as pd
from pandas.api.types import is_numeric_dtype
import re

from charts.builder import dual_axis_chart, dissolved_dual_axis_chart, status_timeline_heatmap
from services.dataset_service import format_datetime_columns_for_display, remove_accumulated_rows


def guess_first_existing(cols: list[str], candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


st.title("📈 Criar Gráfico")

# ✅ Carregar dataset real
df_dict = st.session_state.get("df_dict")

if type(df_dict) == None or not isinstance(df_dict, (dict, pd.DataFrame)) or len(df_dict) == 0:
    st.error("❌ Nenhum Excel carregado. Importe um arquivo na página inicial.")
    st.stop()

if isinstance(df_dict, dict):
    table_name = st.selectbox("Tabela:", list(df_dict.keys()))
    df = df_dict[table_name].copy()
elif isinstance(df_dict, pd.DataFrame):
    table_name = "Monitoramento Laboratorial"
    df = df_dict

# ✅ Mapear colunas (timestamp e poço)
cols = list(df.columns)

default_time = guess_first_existing(cols, ["timestamp", "data", "Data", "DATA", "date", "Date"])
default_poco = guess_first_existing(cols, ["poco", "Poço", "poço", "ponto", "Ponto", "well", "Well"])

if not default_time:
    st.warning("Não encontrei uma coluna de data/hora automaticamente. Selecione manualmente abaixo.")

if not default_poco:
    st.warning("Não encontrei uma coluna de poço/ponto automaticamente. Selecione manualmente abaixo.")

with st.expander("⚙️ Colunas"):
    time_col = st.selectbox("Data/Hora:", options=cols, index=cols.index(default_time) if default_time in cols else 0)
    group_col_base = st.selectbox("Poço/Ponto:", options=cols, index=cols.index(default_poco) if default_poco in cols else 0)

# tenta converter a coluna de tempo (sem quebrar)
try:
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce", dayfirst=True)
except Exception:
    pass

st.divider()

# X mode
x_mode = st.selectbox("Eixo X", ["Temporal", "Por poço"])

if x_mode == "Temporal":
    x_col = time_col

    # se a coluna de grupo não existir, cria uma dummy
    if group_col_base not in df.columns:
        df["poco"] = "Série"
        group_col = "poco"
    else:
        group_col = group_col_base

    # filtros de poço
    all_groups = sorted(df[group_col].dropna().astype(str).unique())
    if all_groups:
        sel_groups = st.multiselect("Poços:", all_groups, default=all_groups)
    else:
        sel_groups = []

    # filtro período
    valid_dt = df[df[x_col].notna()]
    if len(valid_dt) == 0:
        st.error("A coluna de tempo está vazia/ inválida. Verifique o mapeamento.")
        st.stop()

    min_dt, max_dt = valid_dt[x_col].min(), valid_dt[x_col].max()
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Data inicial:",
            value=min_dt.date(),
            min_value=min_dt.date(),
            max_value=max_dt.date(),
            key="start_date_temporal"
        )
    with col2:
        end_date = st.date_input(
            "Data final:",
            value=max_dt.date(),
            min_value=min_dt.date(),
            max_value=max_dt.date(),
            key="end_date_temporal"
        )
    
    # Converter datas para datetime com hora
    dt_range = (
        pd.Timestamp(start_date).to_pydatetime(),
        pd.Timestamp(end_date).replace(hour=23, minute=59, second=59).to_pydatetime()
    )
    
    if start_date > end_date:
        st.error("❌ Data inicial não pode ser maior que data final!")
        st.stop()

    dff = df[df[x_col].between(dt_range[0], dt_range[1])].copy()
    if sel_groups:
        dff = dff[dff[group_col].astype(str).isin(sel_groups)].copy()

    allowed_chart_types = ["Auto", "Linha", "Dispersão"]
else:
    x_col = group_col_base if group_col_base in df.columns else None
    if not x_col:
        st.error("Para 'Por poço', você precisa mapear uma coluna categórica (poço/ponto).")
        st.stop()

    valid_dt = df[df[time_col].notna()]
    if len(valid_dt) > 0:
        min_dt, max_dt = valid_dt[time_col].min(), valid_dt[time_col].max()
        
        st.markdown("### 📅 Período (para agregação)")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Data inicial:",
                value=min_dt.date(),
                min_value=min_dt.date(),
                max_value=max_dt.date(),
                key="start_date_poço"
            )
        with col2:
            end_date = st.date_input(
                "Data final:",
                value=max_dt.date(),
                min_value=min_dt.date(),
                max_value=max_dt.date(),
                key="end_date_poço"
            )
        
        # Converter datas para datetime com hora
        dt_range = (
            pd.Timestamp(start_date).to_pydatetime(),
            pd.Timestamp(end_date).replace(hour=23, minute=59, second=59).to_pydatetime()
        )
        
        if start_date > end_date:
            st.error("❌ Data inicial não pode ser maior que data final!")
            st.stop()
        
        dff = df[df[time_col].between(dt_range[0], dt_range[1])].copy()
    else:
        dff = df.copy()

    allowed_chart_types = ["Auto", "Barra", "Box"]

# ===========================
#  Y + PLOT (com compatibilidade)
# ===========================

# garante group_col definido (no modo "Por poço" ele pode não existir)
if x_mode != "Temporal":
    group_col = None

# Detecta se existe modo laboratório (colunas __num e __status)
status_num_cols = [c for c in dff.columns if isinstance(c, str) and c.endswith("__num")]
status_params = sorted({c[:-5] for c in status_num_cols})  # remove "__num"
has_status_mode = len(status_params) > 0 and any(f"{p}__status" in dff.columns for p in status_params)

enable_status_mode = (x_mode == "Temporal") and has_status_mode

st.markdown("#### 📊 Parâmetros do Gráfico")

if enable_status_mode:
    plot_mode = st.radio(
        "Modo de plot:",
        options=["Padrão (numérico)", "Laboratório (SECO/FASE LIVRE/<)"],
        horizontal=True,
        index=1,
    )
else:
    plot_mode = "Padrão (numérico)"


# -------------------------
# MODO PADRÃO (igual ao seu)
# -------------------------
if plot_mode == "Padrão (numérico)":
    numeric_cols = [c for c in dff.columns if is_numeric_dtype(dff[c]) and c != time_col]
    default_y = [c for c in ["ph", "condutividade"] if c in numeric_cols] or numeric_cols[:2]

    y_cols = st.multiselect(
        "Selecione parâmetros:",
        options=numeric_cols,
        default=default_y,
        key="y_params_select_numeric",
    )

    if not y_cols:
        st.info("Selecione pelo menos um parâmetro.")
        st.stop()

    # Detectar se todos os parâmetros têm a mesma ordem de grandeza
    def get_scale(col_name: str) -> str:
        vals = dff[col_name].dropna()
        if len(vals) == 0:
            return "unknown"
        range_val = vals.max() - vals.min()
        if range_val == 0:
            return "unknown"
        import math
        magnitude = math.floor(math.log10(abs(range_val))) if range_val != 0 else 0
        return f"10^{magnitude}"

    scales = {col: get_scale(col) for col in y_cols}
    unique_scales = set(scales.values())

    if len(unique_scales) == 1 and list(unique_scales)[0] != "unknown":
        y_left = y_cols
        y_right = []
        st.success("✅ Um eixo Y (escalas similares)")
    else:
        st.markdown("##### 📐 Distribuição dos Eixos")
        col_y1, col_y2 = st.columns(2, gap="medium")
        with col_y1:
            st.markdown("**Eixo Y1 (esquerdo)**")
            y_left = st.multiselect(
                "Selecione:",
                options=y_cols,
                default=y_cols[:1],
                key="y_left_select_numeric",
                label_visibility="collapsed"
            )
        with col_y2:
            st.markdown("**Eixo Y2 (direito)**")
            y_right_display = [c for c in y_cols if c not in y_left]
            if y_right_display:
                st.markdown(" ")
                for param in y_right_display:
                    st.caption(f"• {param}")
            else:
                st.caption("—")
            y_right = y_right_display

    st.markdown("") 

    col_tipo, col_agg = st.columns([2, 1], gap="medium") 
    with col_tipo: chart_type = st.selectbox("Tipo de gráfico:", options=allowed_chart_types, index=0)
    agg = "mean" 

    if x_mode == "Por poço" and (chart_type in ("Auto", "Barra")):
        with col_agg: agg = st.selectbox("Agregação:", ["mean", "median", "min", "max", "sum"], index=0)
    
    st.markdown("")

    fig = dual_axis_chart(
        df=dff,
        x_col=x_col,
        y_left=y_left,
        y_right=y_right,
        chart_type=chart_type,
        agg=agg,
        group_col=group_col if x_mode == "Temporal" else None,
    )
    st.plotly_chart(fig, use_container_width=True)


# -------------------------
# MODO LABORATÓRIO (status)
# -------------------------
else:
    # filtro opcional por categoria
    GROUPS = {
        "Todos": None,
        "TPH": r"(?i)\bTPH\b",
        "BTEX": r"(?i)\b(benzeno|tolueno|etilbenzeno|xilen)\b",
        "VOC (geral)": r"(?i)\b(cloro|dicloro|tricloro|tetracloro|benzen|tolu|xilen)\b",
    }
    sel_group = st.selectbox("Categoria (opcional):", options=list(GROUPS.keys()), index=0)
    patt = GROUPS[sel_group]

    candidate_params = status_params
    if patt:
        candidate_params = [p for p in status_params if re.search(patt, p)]

    # remove parâmetros sem números no recorte (com base no __num)
    available_params = []
    for p in candidate_params:
        col_num = f"{p}__num"
        if col_num in dff.columns and pd.to_numeric(dff[col_num], errors="coerce").notna().any():
            available_params.append(p)

    y_params = st.multiselect(
        "Selecione parâmetros:",
        options=available_params,
        default=available_params[:1] if available_params else [],
        key="y_params_select_status",
    )

    if not y_params:
        st.info("Selecione pelo menos um parâmetro.")
        st.stop()

    # escala usando __num
    def get_scale_param(param: str) -> str:
        col_num = f"{param}__num"
        vals = pd.to_numeric(dff[col_num], errors="coerce").dropna() if col_num in dff.columns else pd.Series([], dtype=float)
        if len(vals) == 0:
            return "unknown"
        range_val = vals.max() - vals.min()
        if range_val == 0:
            return "unknown"
        import math
        magnitude = math.floor(math.log10(abs(range_val))) if range_val != 0 else 0
        return f"10^{magnitude}"

    scales = {p: get_scale_param(p) for p in y_params}
    unique_scales = set(scales.values())

    if len(unique_scales) == 1 and list(unique_scales)[0] != "unknown":
        y_left = y_params
        y_right = []
        st.success("✅ Um eixo Y (escalas similares)")
    else:
        st.markdown("##### 📐 Distribuição dos Eixos")
        col_y1, col_y2 = st.columns(2, gap="medium")
        with col_y1:
            st.markdown("**Eixo Y1 (esquerdo)**")
            y_left = st.multiselect(
                "Selecione:",
                options=y_params,
                default=y_params[:1],
                key="y_left_select_status",
                label_visibility="collapsed"
            )
        with col_y2:
            st.markdown("**Eixo Y2 (direito)**")
            y_right_display = [p for p in y_params if p not in y_left]
            if y_right_display:
                st.markdown(" ")
                for param in y_right_display:
                    st.caption(f"• {param}")
            else:
                st.caption("—")
            y_right = y_right_display

    tab1, tab2 = st.tabs(["Valores dissolvidos", "Timeline (status)"])

    with tab1:
        fig1 = dissolved_dual_axis_chart(
            df=dff,
            x_col=x_col,
            params_left=y_left,
            params_right=y_right,
            group_col=group_col,
        )
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        # timeline fica mais clara mostrando 1 parâmetro por vez
        timeline_param = st.selectbox(
            "Parâmetro para timeline:",
            options=y_left + y_right,
            index=0,
            key="timeline_param_select",
        )
        fig2 = status_timeline_heatmap(
            df=dff,
            x_col=x_col,
            group_col=group_col,
            param=timeline_param,
        )
        st.plotly_chart(fig2, use_container_width=True)
