# charts/builder.py
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

########################################################
#          GRAPH BUILDER FOR NON NUM PARAMS            #
########################################################

def dual_axis_status_chart(
    df: pd.DataFrame,
    x_col: str,
    params_left: list[str],
    params_right: list[str],
    group_col: str | None = None,
):
    """
    Espera colunas:
      <param>__num
      <param>__status

    Faz:
    - linha com __num (quebra onde status = SECO ou MISSING)
    - markers coloridos:
        SECO: vermelho em y=0 (quebra linha)
        FASE_LIVRE: laranja em y=cap (cap calculado por param)
        MISSING: cinza em y=0 (opcional)
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    def add_param_traces(sub: pd.DataFrame, param: str, secondary: bool, prefix: str):
        num_col = f"{param}__num"
        st_col = f"{param}__status"
        if num_col not in sub.columns or st_col not in sub.columns:
            return

        # ordena por tempo
        s = sub[[x_col, num_col, st_col]].copy()
        s = s.sort_values(by=x_col)

        # cap para FASE LIVRE: baseado no máximo observado (MEASURED/LT_RL)
        base = s[s[st_col].isin(["MEASURED", "MEASURED_QUAL", "LT_RL"])][num_col].dropna()
        cap = (base.max() * 1.2) if len(base) else 1.0

        # Linha: só onde não é SECO/MISSING/FASE_LIVRE
        y_line = s[num_col].copy()
        y_line[s[st_col].isin(["SECO", "MISSING", "FASE_LIVRE", "TEXT"])] = np.nan

        fig.add_trace(
            go.Scatter(
                x=s[x_col],
                y=y_line,
                mode="lines",
                name=f"{prefix}{param}",
            ),
            secondary_y=secondary,
        )

        # SECO marker (vermelho) em y=0
        mask_seco = s[st_col] == "SECO"
        if mask_seco.any():
            fig.add_trace(
                go.Scatter(
                    x=s.loc[mask_seco, x_col],
                    y=np.zeros(mask_seco.sum()),
                    mode="markers",
                    name=f"{prefix}{param} · SECO",
                    marker=dict(color="red", size=8),
                ),
                secondary_y=secondary,
            )

        # FASE LIVRE marker (laranja) em y=cap
        mask_fl = s[st_col] == "FASE_LIVRE"
        if mask_fl.any():
            fig.add_trace(
                go.Scatter(
                    x=s.loc[mask_fl, x_col],
                    y=np.full(mask_fl.sum(), cap),
                    mode="markers",
                    name=f"{prefix}{param} · FASE LIVRE",
                    marker=dict(color="orange", size=8),
                ),
                secondary_y=secondary,
            )

        # MISSING marker (cinza) em y=0 (opcional)
        mask_miss = s[st_col] == "MISSING"
        if mask_miss.any():
            fig.add_trace(
                go.Scatter(
                    x=s.loc[mask_miss, x_col],
                    y=np.zeros(mask_miss.sum()),
                    mode="markers",
                    name=f"{prefix}{param} · SEM MEDIÇÃO",
                    marker=dict(color="gray", size=6),
                ),
                secondary_y=secondary,
            )

    # agrupa por poço (se existir)
    if group_col and group_col in df.columns:
        for g, sub in df.groupby(group_col, sort=True):
            prefix = f"{g} · "
            for p in params_left:
                add_param_traces(sub, p, secondary=False, prefix=prefix)
            for p in params_right:
                add_param_traces(sub, p, secondary=True, prefix=prefix)
    else:
        for p in params_left:
            add_param_traces(df, p, secondary=False, prefix="")
        for p in params_right:
            add_param_traces(df, p, secondary=True, prefix="")

    fig.update_layout(
        xaxis_title=x_col,
        legend_title_text="Séries / Status",
    )
    fig.update_yaxes(title_text="Y1", secondary_y=False)
    fig.update_yaxes(title_text="Y2", secondary_y=True)
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
        def _name(row_serie, row_group, axis_label):
            return f"{row_group} · {row_serie} ({axis_label})" if row_group else f"{row_serie} ({axis_label})"

        if group_col:
            groups = sorted(grouped[group_col].unique())
        else:
            groups = [None]

        for g in groups:
            gdf = grouped[grouped[group_col] == g] if g is not None else grouped
            for col in y_left:
                sub = gdf[gdf["serie"] == col]
                fig.add_trace(
                    go.Bar(x=sub[x_col], y=sub[f"valor_{agg}"], name=_name(col, g, "Y1")),
                    secondary_y=False,
                )
            for col in y_right:
                sub = gdf[gdf["serie"] == col]
                fig.add_trace(
                    go.Bar(x=sub[x_col], y=sub[f"valor_{agg}"], name=_name(col, g, "Y2")),
                    secondary_y=True,
                )

        fig.update_layout(
            barmode="group",
            xaxis_title=x_col,
            legend_title_text="Parâmetros",
            legend=dict(
                x=1.02,
                y=1,
                xanchor="left",
                yanchor="top",
                bordercolor="rgba(0, 0, 0, 0.2)",
                borderwidth=1,
            ),
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
                go.Scatter(x=sub[x_col], y=sub[col], name=f"{prefix}{col} (Y1)", mode=mode),
                secondary_y=False,
            )
        for col in y_right:
            fig.add_trace(
                go.Scatter(x=sub[x_col], y=sub[col], name=f"{prefix}{col} (Y2)", mode=mode),
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
        legend=dict(
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="top",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1,
        ),
        margin=dict(r=250),  # Espaço para legenda à direita
        hovermode="x unified"
    )
    fig.update_yaxes(title_text=" / ".join(y_left) if y_left else "—", secondary_y=False)
    fig.update_yaxes(title_text=" / ".join(y_right) if y_right else "—", secondary_y=True)
    return fig
