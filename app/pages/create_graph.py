import re
import unicodedata

import pandas as pd
import streamlit as st

from charts.builder import SeriesSpec, build_time_chart_plotly
from services.date_num_prep import add_pr_for_pb_pi, normalize_dates, parse_ptbr_number


def _norm_key(value: object) -> str:
    s = str(value).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _find_col(df: pd.DataFrame, tokens: list[str]) -> str | None:
    for col in df.columns:
        norm = _norm_key(col)
        if all(t in norm for t in tokens):
            return col
    return None


def _get_poco_col(df: pd.DataFrame) -> str | None:
    return _find_col(df, ["poco"]) or _find_col(df, ["ponto"])

def _is_pr(value: object) -> bool:
    if value is None:
        return False
    return str(value).strip().upper().startswith("PR-")


def prep_vol_bombeado(vol_bombeado: pd.DataFrame) -> pd.DataFrame:
    df = normalize_dates(vol_bombeado, "Data")
    vol_col = _find_col(df, ["volume", "bombeado"])
    if vol_col:
        df[vol_col] = df[vol_col].map(parse_ptbr_number)
        df = df.rename(columns={vol_col: "bombeado_vol"})
    else:
        df["bombeado_vol"] = pd.NA

    poco_col = _get_poco_col(df)
    df["poco_key"] = df[poco_col] if poco_col else pd.NA
    return df


def prep_vol_infiltrado(vol_infiltrado: pd.DataFrame) -> pd.DataFrame:
    df = normalize_dates(vol_infiltrado, "Data")
    vol_col = _find_col(df, ["volume", "infiltrado"])
    if vol_col:
        df[vol_col] = df[vol_col].map(parse_ptbr_number)
        df = df.rename(columns={vol_col: "infiltrado_vol"})
    else:
        df["infiltrado_vol"] = pd.NA
    return df


def prep_na_semanal(na_semanal: pd.DataFrame) -> pd.DataFrame:
    df = normalize_dates(na_semanal, "Data")
    for c in ["NA (m)", "NO (m)", "FL (m)"]:
        if c in df.columns:
            df[c] = df[c].map(parse_ptbr_number)
    if "NA (m)" in df.columns:
        df = df.rename(columns={"NA (m)": "na_m"})
    else:
        df["na_m"] = pd.NA

    poco_col = _get_poco_col(df)
    df["poco_key"] = df[poco_col] if poco_col else pd.NA
    df["PR"] = df["poco_key"].map(add_pr_for_pb_pi)
    df["entity_key"] = df["PR"].fillna(df["poco_key"])
    return df


def build_na_pr_vs_infiltrado(na: pd.DataFrame, vi: pd.DataFrame) -> pd.DataFrame:
    na_pr = na[na["PR"].notna()].copy()
    na_pr = na_pr.groupby("Data", as_index=False)["na_m"].mean()

    vi_daily = vi.groupby("Data", as_index=False)["infiltrado_vol"].sum()

    out = na_pr.merge(vi_daily, on="Data", how="outer")
    return out.sort_values("Data")


def build_point_series(vb: pd.DataFrame, na: pd.DataFrame, points: list[str]) -> pd.DataFrame:
    vb_f = vb[vb["poco_key"].isin(points)].copy()
    na_f = na[na["entity_key"].isin(points)].copy()

    vb_wide = (
        vb_f.groupby(["Data", "poco_key"], as_index=False)["bombeado_vol"]
        .sum()
        .pivot_table(index="Data", columns="poco_key", values="bombeado_vol", aggfunc="sum")
    )
    vb_wide = vb_wide.add_prefix("bombeado__")

    na_wide = (
        na_f.groupby(["Data", "entity_key"], as_index=False)["na_m"]
        .mean()
        .pivot_table(index="Data", columns="entity_key", values="na_m", aggfunc="mean")
    )
    na_wide = na_wide.add_prefix("na__")

    wide = pd.concat([vb_wide, na_wide], axis=1).reset_index()
    return wide

def _point_color_map(points: list[str]) -> dict[str, str]:
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    return {p: palette[i % len(palette)] for i, p in enumerate(points)}


st.title("NA e Volume - Visualizacoes")

df_dict = st.session_state.get("df_dict")
if df_dict is None or not isinstance(df_dict, dict):
    st.info("Arquivo foi carregado, mas ainda nao ha dataset em memoria.")
    st.stop()

for required in ["Volume Bombeado", "Volume Infiltrado", "NA Semanal"]:
    if required not in df_dict:
        st.error(f"Planilha obrigatoria ausente: {required}")
        st.stop()

vb = prep_vol_bombeado(df_dict["Volume Bombeado"])
vi = prep_vol_infiltrado(df_dict["Volume Infiltrado"])
na = prep_na_semanal(df_dict["NA Semanal"])

st.subheader("Media do NA (PR) vs Volume Infiltrado")

na_vi = build_na_pr_vs_infiltrado(na, vi)

series = [
    SeriesSpec(
        y="na_m",
        label="NA medio PR (m)",
        kind="line",
        marker="circle",
        color="#2ca02c",
    ),
    SeriesSpec(
        y="infiltrado_vol",
        label="Volume Infiltrado",
        kind="bar",
        axis="y2",
        color="#1f77b4",
    ),
]

fig, _ = build_time_chart_plotly(
    na_vi,
    x="Data",
    series=series,
    title="NA medio PR vs Volume Infiltrado",
    show_range_slider=False,
    limit_points=200000,
    return_insights=False,
)
fig.update_yaxes(title_text="NA medio (m)", secondary_y=False)
fig.update_yaxes(title_text="Volume Infiltrado (m³)", secondary_y=True)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Volume Bombeado vs NA (por ponto)")

all_points = sorted(
    set(vb["poco_key"].dropna().unique().tolist())
    | set(na["entity_key"].dropna().unique().tolist())
)
all_points = [p for p in all_points if p and p != "Acumulado" and _is_pr(p)]
if not all_points:
    st.info("Nao ha pontos PR para visualizar.")
    st.stop()

mode = st.radio(
    "Modo de selecao",
    ["Selecionar pontos (checkboxes)", "Um ponto por vez"],
    horizontal=True,
)

if mode == "Um ponto por vez":
    selected_points = [st.selectbox("Ponto", all_points)] if all_points else []
else:
    selected_points = st.multiselect("Pontos", all_points, default=all_points)

if not selected_points:
    st.info("Selecione ao menos um ponto para visualizar.")
    st.stop()

wide = build_point_series(vb, na, selected_points)

color_map = _point_color_map(selected_points)
single_point = len(selected_points) == 1

series = []
for point in selected_points:
    col_na = f"na__{point}"
    if col_na in wide.columns:
        if single_point:
            na_color = "rgba(46, 139, 87, 0.7)"
        else:
            na_color = color_map.get(point)
        series.append(
            SeriesSpec(
                y=col_na,
                label=f"NA (m) - {point}",
                kind="line",
                marker="circle",
                line_dash="solid",
                color=na_color,
                axis="y",
            )
        )
    col_vb = f"bombeado__{point}"
    if col_vb in wide.columns:
        if single_point:
            vb_color = "rgba(31, 119, 180, 0.45)"
        else:
            vb_color = color_map.get(point)
        series.append(
            SeriesSpec(
                y=col_vb,
                label=f"Volume Bombeado - {point}",
                kind="bar",
                color=vb_color,
                axis="y2",
            )
        )

if not series:
    st.info("Sem dados de NA ou volume bombeado para os pontos selecionados.")
    st.stop()

fig2, _ = build_time_chart_plotly(
    wide,
    x="Data",
    series=series,
    title="Volume Bombeado vs NA por ponto",
    show_range_slider=False,
    limit_points=200000,
    return_insights=False,
)
fig2.update_layout(barmode="group")
fig2.update_yaxes(title_text="NA (m)", secondary_y=False)
fig2.update_yaxes(title_text="Volume Bombeado (m³)", secondary_y=True)

st.plotly_chart(fig2, use_container_width=True)
