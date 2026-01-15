# charts/builder.py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

########################################################
#          GRAPH BUILDER FOR NON NUM PARAMS            #
########################################################

DISSOLVED_STATUSES = {"MEASURED", "MEASURED_QUAL", "LT_RL"}
EVENT_MARKERS = {
    "SECO": ("red", "x", 9, "SECO"),
    "FASE_LIVRE": ("orange", "triangle-up", 9, "FASE LIVRE"),
    "MISSING": ("gray", "circle-open", 7, "SEM MEDICAO"),
}

LEGEND_STYLE = dict(
    x=1.02,
    y=1,
    xanchor="left",
    yanchor="top",
    orientation="v",
    bordercolor="rgba(0, 0, 0, 0.2)",
    borderwidth=1,
    itemsizing="constant",
    tracegroupgap=6,
)

# --- 1) Valores dissolvidos: linha/markers usando <param>__num, mas SEM estourar escala por FASE LIVRE ---
def dissolved_dual_axis_chart(
    df: pd.DataFrame,
    x_col: str,
    params_left: list[str],
    params_right: list[str],
    group_col: str | None = None,
):
    """
    Usa colunas <param>__num e <param>__status.
    Plota apenas valores dissolvidos (MEASURED, MEASURED_QUAL, LT_RL),
    e coloca marcadores em y=0 para SECO e FASE LIVRE (sem afetar escala).
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    def add_param(sub: pd.DataFrame, param: str, secondary: bool, prefix: str):
        num_col = f"{param}__num"
        st_col = f"{param}__status"
        if num_col not in sub.columns or st_col not in sub.columns:
            return

        s = sub[[x_col, num_col, st_col]].copy().sort_values(by=x_col)

        # Linha: só dissolve (inclui LT_RL=0). Quebra onde SECO/MISSING/FASE_LIVRE.
        y = pd.to_numeric(s[num_col], errors="coerce").copy()
        y[~s[st_col].isin(DISSOLVED_STATUSES)] = np.nan

        fig.add_trace(
            go.Scatter(
                x=s[x_col],
                y=y,
                mode="lines+markers",
                name=f"{prefix}{param}",
                marker=dict(size=5),
            ),
            secondary_y=secondary,
        )

        # Marcadores de eventos em y=0 (nao alteram a escala)
        for status, (color, symbol, size, label) in EVENT_MARKERS.items():
            mask = s[st_col] == status
            if not mask.any():
                continue
            fig.add_trace(
                go.Scatter(
                    x=s.loc[mask, x_col],
                    y=np.zeros(mask.sum()),
                    mode="markers",
                    name=f"{prefix}{param} - {label}",
                    marker=dict(color=color, size=size, symbol=symbol),
                ),
                secondary_y=secondary,
            )

    if group_col and group_col in df.columns:
        for g, sub in df.groupby(group_col, sort=True):
            prefix = f"{g} · "
            for p in params_left:
                add_param(sub, p, secondary=False, prefix=prefix)
            for p in params_right:
                add_param(sub, p, secondary=True, prefix=prefix)
    else:
        for p in params_left:
            add_param(df, p, secondary=False, prefix="")
        for p in params_right:
            add_param(df, p, secondary=True, prefix="")

    fig.update_layout(
        xaxis_title=x_col,
        legend_title_text="Series / Status",
        legend=LEGEND_STYLE,
        margin=dict(r=250),
    )
    fig.update_yaxes(title_text="Y1", secondary_y=False)
    fig.update_yaxes(title_text="Y2", secondary_y=True)
    return fig


# --- 2) Timeline de status (heatmap) para 1 parâmetro ---
def status_timeline_heatmap(
    df: pd.DataFrame,
    x_col: str,
    group_col: str,
    param: str,
):
    """
    Heatmap Poço x Data usando <param>__status.
    Consolida duplicatas por (poço,data) usando prioridade:
    FASE_LIVRE > SECO > MEASURED/MEASURED_QUAL > LT_RL > MISSING > outros
    """
    st_col = f"{param}__status"
    if st_col not in df.columns:
        raise ValueError(f"Coluna {st_col} não encontrada.")

    tmp = df[[group_col, x_col, st_col]].copy()
    tmp[x_col] = pd.to_datetime(tmp[x_col], errors="coerce")
    tmp = tmp[tmp[x_col].notna()].copy()
    tmp["data"] = tmp[x_col].dt.date

    priority = {
        "FASE_LIVRE": 5,
        "SECO": 4,
        "MEASURED": 3,
        "MEASURED_QUAL": 3,
        "LT_RL": 2,
        "MISSING": 1,
    }
    tmp["prio"] = tmp[st_col].astype(str).str.strip().map(priority).fillna(0).astype(int)

    # consolida por (poço,data): pega maior prioridade
    idx = tmp.groupby([group_col, "data"])["prio"].idxmax()
    tmp2 = tmp.loc[idx].copy()

    pivot = tmp2.pivot(index=group_col, columns="data", values="prio").fillna(0)

    # texto com status
    txt = tmp2.pivot(index=group_col, columns="data", values=st_col).fillna("")

    # escala discreta (0..5)
    colorscale = [
        [0.00, "rgb(30,30,30)"],   # 0 = vazio/outros
        [0.20, "gray"],            # 1 = missing
        [0.40, "lightgray"],       # 2 = <RL (se quiser diferenciar; aqui mapeamos 2 como claro)
        [0.60, "deepskyblue"],     # 3 = medido
        [0.80, "red"],             # 4 = seco
        [1.00, "orange"],          # 5 = fase livre
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[str(d) for d in pivot.columns],
            y=pivot.index.astype(str),
            colorscale=colorscale,
            zmin=0,
            zmax=5,
            showscale=False,
            text=txt.values,
            hovertemplate="Poço=%{y}<br>Data=%{x}<br>Status=%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title="Data",
        yaxis_title="Poço",
        title=f"Timeline de status — {param}",
        height=350 + 12 * len(pivot.index),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig
    
########################################################
#                 NORMAL GRAPH BUILDER                 #
########################################################

def dual_axis_chart(
    df: pd.DataFrame,
    x_col: str,
    y_left: list[str],
    y_right: list[str],
    chart_type: str = "Auto",     # Auto | Linha | Dispersão | Barra
    agg: str = "mean",            # usado quando X é categórico e chart_type == Barra
    group_col: str | None = None, # ex.: "poco" para separar linhas por poço
):
    if not y_left and not y_right:
        raise ValueError("Selecione ao menos 1 coluna para o eixo Y.")

    cols = [x_col] + y_left + y_right + ([group_col] if group_col else [])
    d = df[cols].copy()

    x_is_datetime = pd.api.types.is_datetime64_any_dtype(d[x_col])

    if chart_type == "Auto":
        chart_type = "Linha" if x_is_datetime else "Barra"

    # --- Caso Barra (categórico): mantém seu comportamento com agregação ---
    if chart_type == "Barra" and not x_is_datetime:
        if agg not in ("mean", "median", "min", "max", "sum"):
            agg = "mean"

        long = d.melt(
            id_vars=[x_col] + ([group_col] if group_col else []),
            value_vars=y_left + y_right,
            var_name="serie",
            value_name="valor",
        )

        # agrega por X (+ grupo se existir)
        group_keys = [x_col, "serie"] + ([group_col] if group_col else [])
        grouped = (
            long.groupby(group_keys, as_index=False)["valor"]
            .agg(agg)
            .rename(columns={"valor": f"valor_{agg}"})
        )

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        # Se tiver group_col, incluímos no nome pra não misturar
        def _name(row_serie, row_group):
            return f"{row_group} · {row_serie}" if row_group else f"{row_serie}"

        if group_col:
            groups = sorted(grouped[group_col].unique())
        else:
            groups = [None]

        for g in groups:
            gdf = grouped[grouped[group_col] == g] if g is not None else grouped
            for col in y_left:
                sub = gdf[gdf["serie"] == col]
                fig.add_trace(
                    go.Bar(x=sub[x_col], y=sub[f"valor_{agg}"], name=_name(col, g)),
                    secondary_y=False,
                )
            for col in y_right:
                sub = gdf[gdf["serie"] == col]
                fig.add_trace(
                    go.Bar(x=sub[x_col], y=sub[f"valor_{agg}"], name=_name(col, g)),
                    secondary_y=True,
                )

        fig.update_layout(
            barmode="group",
            xaxis_title=x_col,
            legend_title_text="Parâmetros",
            legend=LEGEND_STYLE,
            margin=dict(r=250),
            hovermode="x unified"
        )
        fig.update_yaxes(title_text=" / ".join(y_left) if y_left else "—", secondary_y=False)
        fig.update_yaxes(title_text=" / ".join(y_right) if y_right else "—", secondary_y=True)
        return fig

    # --- Linha / Dispersão: aqui é onde resolvemos o seu problema ---
    if chart_type not in ("Linha", "Dispersão"):
        raise ValueError("chart_type inválido para este modo.")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    mode = "lines" if chart_type == "Linha" else "markers"

    def add_traces(sub: pd.DataFrame, group_value: str | None):
        sub = sub.sort_values(by=x_col)
        prefix = f"{group_value} · " if group_value is not None else ""
        for col in y_left:
            fig.add_trace(
                go.Scatter(x=sub[x_col], y=sub[col], name=f"{prefix}{col}", mode=mode),
                secondary_y=False,
            )
        for col in y_right:
            fig.add_trace(
                go.Scatter(x=sub[x_col], y=sub[col], name=f"{prefix}{col}", mode=mode),
                secondary_y=True,
            )

    if group_col:
        for g, sub in d.groupby(group_col, sort=True):
            add_traces(sub, str(g))
    else:
        add_traces(d, None)

    fig.update_layout(
        xaxis_title=x_col,
        legend_title_text="Parâmetros",
        legend=LEGEND_STYLE,
        margin=dict(r=250),  # Espaco para legenda a direita
        hovermode="x unified"
    )
    fig.update_yaxes(title_text=" / ".join(y_left) if y_left else "—", secondary_y=False)
    fig.update_yaxes(title_text=" / ".join(y_right) if y_right else "—", secondary_y=True)
    return fig
