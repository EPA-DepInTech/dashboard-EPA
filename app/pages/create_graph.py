# pages/4_Criar_Grafico.py
import streamlit as st
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

from charts.builder import dual_axis_chart


def demo_df() -> pd.DataFrame:
    # Dados hardcoded (demo) para testar o construtor de gráfico
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


st.title("Criar gráfico (X temporal ou por poço, Y múltiplo)")

# Por enquanto vamos usar df hardcoded para você testar o builder
# Depois substituímos por st.session_state["df"] quando o Excel estiver definido.
df = demo_df()

st.caption(f"Dataset de demonstração | Linhas: {len(df)} | Colunas: {len(df.columns)}")
with st.expander("Prévia dos dados"):
    st.dataframe(df.head(40), use_container_width=True)

# X hardcoded
x_mode = st.selectbox("Eixo X", ["Temporal", "Por poço"])

# Define coluna X e filtros auxiliares
if x_mode == "Temporal":
    x_col = "timestamp"
    all_pocos = sorted(df["poco"].unique())
    sel_pocos = st.multiselect("Poços", all_pocos, default=all_pocos)

    # Filtro por período (simples)
    min_dt, max_dt = df["timestamp"].min(), df["timestamp"].max()
    dt_range = st.slider(
        "Período",
        min_value=min_dt.to_pydatetime(),
        max_value=max_dt.to_pydatetime(),
        value=(min_dt.to_pydatetime(), max_dt.to_pydatetime()),
    )

    dff = df[(df["poco"].isin(sel_pocos)) & (df["timestamp"].between(dt_range[0], dt_range[1]))].copy()

    allowed_chart_types = ["Auto", "Linha", "Dispersão"]  # temporal
else:
    x_col = "poco"
    # opcional: filtro temporal mesmo no modo "por poço"
    min_dt, max_dt = df["timestamp"].min(), df["timestamp"].max()
    dt_range = st.slider(
        "Considerar dados no período (para agregação por poço)",
        min_value=min_dt.to_pydatetime(),
        max_value=max_dt.to_pydatetime(),
        value=(min_dt.to_pydatetime(), max_dt.to_pydatetime()),
    )

    dff = df[df["timestamp"].between(dt_range[0], dt_range[1])].copy()

    allowed_chart_types = ["Auto", "Barra", "Box"]  # categórico


# Y múltiplo
numeric_cols = [c for c in dff.columns if is_numeric_dtype(dff[c]) and c not in ("timestamp",)]
default_y = [c for c in ["ph", "condutividade"] if c in numeric_cols] or numeric_cols[:2]

y_cols = st.multiselect(
    "Parâmetros (selecione 1 ou mais)",
    options=numeric_cols,
    default=default_y,
)

if not y_cols:
    st.info("Selecione pelo menos um parâmetro para gerar o gráfico.")
    st.stop()

st.markdown("### Escalas do eixo Y (duas escalas)")
st.caption("Coloque no Y1 (esquerdo) as séries de mesma ordem de grandeza; o restante vai pro Y2 (direito).")

# Escolha do eixo esquerdo; o direito vira o complemento
default_left = y_cols[:1]  # por padrão, o primeiro vai pro Y1
y_left = st.multiselect(
    "Eixo Y1 (esquerdo)",
    options=y_cols,
    default=default_left,
    key="y_left_select",
)

y_right = [c for c in y_cols if c not in y_left]

chart_type = st.selectbox("Tipo de gráfico", options=allowed_chart_types, index=0)

agg = "mean"
if x_mode == "Por poço" and (chart_type in ("Auto", "Barra")):
    agg = st.selectbox("Agregação (Barra por poço)", ["mean", "median", "min", "max", "sum"], index=0)

group_col = "poco" if x_mode == "Temporal" else None

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

# Ajuda visual pro usuário
with st.expander("Resumo das escalas"):
    st.write({"Y1 (esquerdo)": y_left, "Y2 (direito)": y_right})
