# pages/4_Criar_Grafico.py
import streamlit as st
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

from charts.builder import dual_axis_chart
from services.dataset_service import format_datetime_columns_for_display, remove_accumulated_rows


def demo_df() -> pd.DataFrame:
    ts = pd.date_range(end=pd.Timestamp.now().floor("H"), periods=72, freq="H")
    pocos = ["Poço A", "Poço B", "Poço C"]

    rows = []
    rng = np.random.default_rng(7)

    for poco in pocos:
        bias = {"Poço A": 0.0, "Poço B": 0.2, "Poço C": -0.15}[poco]
        for t in ts:
            rows.append(
                {
                    "timestamp": t,
                    "poco": poco,
                    "ph": 6.8 + bias + 0.15 * np.sin(t.hour / 24 * 2 * np.pi) + rng.normal(0, 0.05),
                    "condutividade": 250 + (20 if poco == "Poço B" else 0) + rng.normal(0, 8),
                    "turbidez": 1.2 + (0.4 if poco == "Poço C" else 0) + abs(rng.normal(0, 0.25)),
                }
            )

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def guess_first_existing(cols: list[str], candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


st.title("📈 Criar Gráfico")

# ✅ 1) tentar usar dataset real
df_dict = st.session_state.get("df_dict")
use_demo = False

if df_dict and isinstance(df_dict, dict) and len(df_dict) > 0:
    table_name = st.selectbox("Tabela:", list(df_dict.keys()))
    df = df_dict[table_name].copy()
else:
    st.info("Usando dataset de demonstração (sem Excel carregado).")
    df = demo_df()
    use_demo = True

# ✅ 2) mapear colunas (timestamp e poço) quando for dataset real
cols = list(df.columns)

default_time = guess_first_existing(cols, ["timestamp", "data", "Data", "DATA", "date", "Date"])
default_poco = guess_first_existing(cols, ["poco", "Poço", "poço", "ponto", "Ponto", "well", "Well"])

if not default_time:
    if use_demo:
        default_time = "timestamp"
    else:
        st.warning("Não encontrei uma coluna de data/hora automaticamente. Selecione manualmente abaixo.")

if not default_poco:
    if use_demo:
        default_poco = "poco"
    else:
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

# Y múltiplo (somente numéricos)
numeric_cols = [c for c in dff.columns if is_numeric_dtype(dff[c]) and c != time_col]
default_y = [c for c in ["ph", "condutividade"] if c in numeric_cols] or numeric_cols[:2]

y_cols = st.multiselect("Parâmetros:", options=numeric_cols, default=default_y)

if not y_cols:
    st.info("Selecione pelo menos um parâmetro.")
    st.stop()

default_left = y_cols[:1]
y_left = st.multiselect("Y1 (esquerdo):", options=y_cols, default=default_left, key="y_left_select")
y_right = [c for c in y_cols if c not in y_left]

chart_type = st.selectbox("Tipo:", options=allowed_chart_types, index=0)

agg = "mean"
if x_mode == "Por poço" and (chart_type in ("Auto", "Barra")):
    agg = st.selectbox("Agregação:", ["mean", "median", "min", "max", "sum"], index=0)

fig = dual_axis_chart(
    df=dff,
    x_col=x_col,
    y_left=y_left,
    y_right=y_right,
    chart_type=chart_type,
    agg=agg,
    group_col=group_col,
)

st.plotly_chart(fig, use_container_width=True)

with st.expander("Resumo das escalas"):
    st.write({"Y1 (esquerdo)": y_left, "Y2 (direito)": y_right})
