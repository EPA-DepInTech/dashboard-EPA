from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Sequence

import numpy as np
import pandas as pd


ChartKind = Literal["line", "scatter", "bar", "area", "step", "status_step"]
Agg = Literal["none", "mean", "sum", "median", "min", "max", "count"]

<<<<<<< HEAD
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
=======

@dataclass
class SeriesSpec:
    """
    Descreve UMA série (um Y) no gráfico temporal.
    Você passa uma lista disso no builder para sobrepor várias séries no mesmo chart.
    """
    y: str
    label: Optional[str] = None

    # Visual
    kind: ChartKind = "line"
    color: Optional[str] = None          # ex: "#1f77b4"
    marker: Optional[str] = None         # plotly: "circle", "square", "x", ...
    line_dash: Optional[str] = None      # "solid", "dash", "dot", "dashdot"

    # Eixos
    axis: Literal["y", "y2"] = "y"

    # Transformações
    agg: Agg = "none"
    resample_rule: Optional[str] = None  # "D", "W", "M", "H"...
    rolling_window: Optional[int] = None
    clip_quantiles: Optional[tuple[float, float]] = None  # (0.01, 0.99)

    # Qualitativo (status_step)
    category_order: Optional[list[str]] = None

    # Texto extra no hover
    hover_cols: list[str] = field(default_factory=list)


def build_time_chart_plotly(
>>>>>>> origin/experimental-rework
    df: pd.DataFrame,
    x: str,
    series: Sequence[SeriesSpec],
    *,
    title: Optional[str] = None,

    # --- Filtros ---
    date_range: Optional[tuple[Any, Any]] = None,
    query: Optional[str] = None,
    category_filters: Optional[dict[str, Sequence[Any]]] = None,

    # --- Tratamento ---
    sort: bool = True,
    dropna_x: bool = True,
    limit_points: Optional[int] = None,

    # --- Layout ---
    height: int = 420,
    show_range_slider: bool = False,
    legend: bool = True,

    # --- Interpretação ---
    return_insights: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """
    Retorna (fig, insights). Em Streamlit:
      st.plotly_chart(fig, use_container_width=True)
    """
    data = df.copy()

    # 1) Filtros (query e filtros categóricos)
    if query:
        data = data.query(query)

    if category_filters:
        for col, allowed in category_filters.items():
            data = data[data[col].isin(list(allowed))]

    # 2) Eixo temporal robusto
    if dropna_x:
        data = data.dropna(subset=[x])

    data[x] = pd.to_datetime(data[x], errors="coerce")
    data = data.dropna(subset=[x])

    if date_range:
        start, end = date_range
        start = pd.to_datetime(start) if start is not None else None
        end = pd.to_datetime(end) if end is not None else None
        if start is not None:
            data = data[data[x] >= start]
        if end is not None:
            data = data[data[x] <= end]

    if sort:
        data = data.sort_values(by=x)

    if limit_points is not None and len(data) > limit_points:
        data = data.tail(limit_points)

    # 3) Prepara cada série (transformações por série)
    prepped: list[dict[str, Any]] = []
    any_y2 = any(s.axis == "y2" for s in series)

    for spec in series:
        if spec.y not in data.columns:
            continue

        cols = [x, spec.y] + (spec.hover_cols or [])
        sdata = data[cols].copy()

        # 3.1) clip de outliers por quantis (só se for numérico)
        if spec.clip_quantiles and pd.api.types.is_numeric_dtype(sdata[spec.y]):
            qlo, qhi = spec.clip_quantiles
            lo = sdata[spec.y].quantile(qlo)
            hi = sdata[spec.y].quantile(qhi)
            sdata[spec.y] = sdata[spec.y].clip(lo, hi)

        # 3.2) resample + agregação (se solicitado)
        if spec.resample_rule:
            sdata = sdata.set_index(x)

            agg = "mean" if spec.agg == "none" else spec.agg

            if agg == "count":
                sres = sdata[spec.y].resample(spec.resample_rule).count()
            else:
                if pd.api.types.is_numeric_dtype(sdata[spec.y]):
                    sres = getattr(sdata[spec.y].resample(spec.resample_rule), agg)()
                else:
                    # qualitativo -> último valor do período
                    sres = sdata[spec.y].resample(spec.resample_rule).last()

            sdata = sres.reset_index().rename(columns={0: spec.y})

            # hover_cols após resample: normalmente não faz sentido manter (perde granularidade)
            # se você quiser, dá pra agregar hover_cols também (mas aí precisa definir regras).
            if spec.hover_cols:
                for c in spec.hover_cols:
                    if c in sdata.columns:
                        continue

        # 3.3) rolling (média móvel) após agregação (só numérico)
        if spec.rolling_window and pd.api.types.is_numeric_dtype(sdata[spec.y]):
            sdata[spec.y] = sdata[spec.y].rolling(spec.rolling_window, min_periods=1).mean()

        prepped.append({"spec": spec, "data": sdata})

    # 4) Renderização Plotly
    fig = _render_plotly(prepped, x=x, title=title, height=height,
                         any_y2=any_y2, show_range_slider=show_range_slider,
                         legend=legend)

    # 5) Insights
    insights = interpret_time_series(data, x=x, series=series, date_range=date_range) if return_insights else {}

    return fig, insights


def _render_plotly(
    prepped: list[dict[str, Any]],
    *,
    x: str,
    title: Optional[str],
    height: int,
    any_y2: bool,
    show_range_slider: bool,
    legend: bool,
):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as e:
        raise ImportError("Instale plotly: pip install plotly") from e

    fig = make_subplots(specs=[[{"secondary_y": any_y2}]])
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=legend,
    )
    fig.update_xaxes(rangeslider=dict(visible=show_range_slider))

    for item in prepped:
        spec: SeriesSpec = item["spec"]
        df = item["data"]
        label = spec.label or spec.y

        customdata = None
        hovertemplate = None

        if spec.hover_cols:
            customdata = df[spec.hover_cols].to_numpy()
            parts = [f"<b>{label}</b><br>", "%{x}<br>", f"{spec.y}: %{y}<br>"]
            for i, c in enumerate(spec.hover_cols):
                parts.append(f"{c}: %{{customdata[{i}]}}<br>")
            parts.append("<extra></extra>")
            hovertemplate = "".join(parts)

        # --- Qualitativo como degraus (status_step)
        if spec.kind == "status_step" and not pd.api.types.is_numeric_dtype(df[spec.y]):
            cat = df[spec.y].astype(str)

            order = spec.category_order or list(pd.unique(cat.dropna()))
            mapping = {k: i for i, k in enumerate(order)}
            y_num = cat.map(mapping)

            fig.add_trace(
                go.Scatter(
                    x=df[x],
                    y=y_num,
                    mode="lines+markers" if spec.marker else "lines",
                    name=label,
                    line=dict(color=spec.color, dash=spec.line_dash, shape="hv"),
                    marker=dict(symbol=spec.marker, color=spec.color) if spec.marker or spec.color else None,
                    customdata=cat.to_numpy()[:, None],
                    hovertemplate=f"<b>{label}</b><br>%{{x}}<br>Status: %{{customdata[0]}}<extra></extra>",
                ),
                secondary_y=(spec.axis == "y2"),
            )

            axis_name = "yaxis2" if (spec.axis == "y2") else "yaxis"
            fig.update_layout(**{
                axis_name: dict(
                    tickmode="array",
                    tickvals=list(mapping.values()),
                    ticktext=list(mapping.keys()),
                )
            })
            continue

        # --- Numérico padrão
        y_vals = df[spec.y]

        if spec.kind == "bar":
            trace = go.Bar(
                x=df[x],
                y=y_vals,
                name=label,
                marker=dict(color=spec.color) if spec.color else None,
                customdata=customdata,
                hovertemplate=hovertemplate,
            )
            fig.add_trace(trace, secondary_y=(spec.axis == "y2"))
            continue

        mode = "lines"
        if spec.kind == "scatter":
            mode = "markers"
        if spec.marker:
            mode = "lines+markers" if spec.kind in ("line", "area", "step") else "markers"

        line_shape = "hv" if spec.kind == "step" else "linear"
        fill = "tozeroy" if spec.kind == "area" else None

        trace = go.Scatter(
            x=df[x],
            y=y_vals,
            mode=mode,
            name=label,
            line=dict(color=spec.color, dash=spec.line_dash, shape=line_shape),
            marker=dict(symbol=spec.marker, color=spec.color) if spec.marker or spec.color else None,
            fill=fill,
            customdata=customdata,
            hovertemplate=hovertemplate,
            connectgaps=False,
        )

        fig.add_trace(trace, secondary_y=(spec.axis == "y2"))

<<<<<<< HEAD
    fig.update_layout(
        xaxis_title=x_col,
        legend_title_text="Series / Status",
        legend=LEGEND_STYLE,
        margin=dict(r=250),
    )
    fig.update_yaxes(title_text="Y1", secondary_y=False)
    fig.update_yaxes(title_text="Y2", secondary_y=True)
=======
>>>>>>> origin/experimental-rework
    return fig


def interpret_time_series(
    df: pd.DataFrame,
    x: str,
    series: Sequence[SeriesSpec],
    *,
    date_range: Optional[tuple[Any, Any]] = None,
) -> dict[str, Any]:
    """
    Insights simples:
      - numérico: first/last/delta/%/min/max + tendência linear aproximada
      - categórico: counts, transições, último status, tempo aproximado por estado
    """
    data = df.copy()
    data[x] = pd.to_datetime(data[x], errors="coerce")
    data = data.dropna(subset=[x])

    if date_range:
        start, end = date_range
        start = pd.to_datetime(start) if start is not None else None
        end = pd.to_datetime(end) if end is not None else None
        if start is not None:
            data = data[data[x] >= start]
        if end is not None:
            data = data[data[x] <= end]

    data = data.sort_values(x)

    out: dict[str, Any] = {
        "window": {"start": data[x].min(), "end": data[x].max(), "rows": int(len(data))},
        "series": {}
    }

    for spec in series:
        if spec.y not in data.columns:
            continue

        y = data[spec.y]
        sx = data[x]
        label = spec.label or spec.y

        if pd.api.types.is_numeric_dtype(y):
            y_clean = y.dropna()
            if y_clean.empty:
                out["series"][label] = {"type": "numeric", "empty": True}
                continue

            last = float(y_clean.iloc[-1])
            first = float(y_clean.iloc[0])
            delta = last - first
            pct = (delta / first * 100.0) if first != 0 else None

            t = (sx.loc[y_clean.index] - sx.loc[y_clean.index].iloc[0]).dt.total_seconds().to_numpy()
            yy = y_clean.to_numpy()
            slope = None
            if len(yy) >= 2 and np.nanstd(t) > 0:
                slope = float(np.polyfit(t, yy, 1)[0])

<<<<<<< HEAD
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
=======
            out["series"][label] = {
                "type": "numeric",
                "first": first,
                "last": last,
                "delta": float(delta),
                "delta_pct": float(pct) if pct is not None else None,
                "min": float(np.nanmin(yy)),
                "max": float(np.nanmax(yy)),
                "trend_slope_per_day": (slope * 86400.0) if slope is not None else None,
            }
>>>>>>> origin/experimental-rework
        else:
            y_clean = y.dropna().astype(str)
            if y_clean.empty:
                out["series"][label] = {"type": "categorical", "empty": True}
                continue

<<<<<<< HEAD
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
=======
            counts = y_clean.value_counts(dropna=True).to_dict()
            last = y_clean.iloc[-1]
            transitions = int((y_clean != y_clean.shift(1)).sum() - 1) if len(y_clean) > 1 else 0

            durations = {}
            if len(y_clean) > 1:
                times = sx.loc[y_clean.index]
                deltas = (times.shift(-1) - times).dt.total_seconds()
                for state, dt_sec in zip(y_clean.iloc[:-1], deltas.iloc[:-1]):
                    if pd.isna(dt_sec):
                        continue
                    durations[state] = durations.get(state, 0.0) + float(dt_sec)
>>>>>>> origin/experimental-rework

            out["series"][label] = {
                "type": "categorical",
                "last": last,
                "counts": counts,
                "transitions": transitions,
                "duration_seconds_by_state_approx": durations,
            }

<<<<<<< HEAD
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
=======
    return out
>>>>>>> origin/experimental-rework
