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


def _norm_text(value: object) -> str:
    s = str(value).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s)
    return s


def _parse_na_value(value: object):
    if value is None or pd.isna(value):
        return pd.NA
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if s == "":
        return pd.NA

    n = _norm_text(s)
    if n in {"nm", "n m", "nao medido"}:
        return "Não medido"

    if re.match(r"^[-+]?\d{1,3}(\.\d{3})*(,\d+)?$|^[-+]?\d+([.,]\d+)?$", s):
        s_num = s.replace(" ", "").replace(".", "").replace(",", ".")
        try:
            return float(s_num)
        except Exception:
            return pd.NA

    if "odor" in n:
        return "Odor"
    if "oleos" in n:
        return "Oleoso"
    if "pelicul" in n:
        return "Pelicula"

    return s


def _is_text_value(value: object) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _parse_hidrometro_value(value: object):
    if value is None or pd.isna(value):
        return "Não medido"
    s = str(value).strip()
    if s == "":
        return "Não medido"
    num = parse_ptbr_number(s)
    if pd.isna(num):
        return s
    return num


def _first_non_null(series: pd.Series):
    for v in series:
        if pd.notna(v):
            return v
    return pd.NA


def _slug_status(status: str) -> str:
    s = _norm_text(status)
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "status"


def _fl_phase(status: object) -> str | None:
    if not _is_text_value(status):
        return None
    n = _norm_text(status)
    if "odor" in n:
        return "Odor"
    if "oleos" in n:
        return "Oleoso"
    if "pelicul" in n:
        return "Pelicula"
    return None


def _collect_statuses(row: pd.Series) -> list[str]:
    statuses: list[str] = []
    phase = row.get("fl_phase")
    if pd.isna(phase):
        phase = None
    if any(pd.notna(row.get(k)) for k in ("na_val", "no_val", "fl_val")):
        return []
    for key in ("na_status", "no_status", "fl_status"):
        v = row.get(key)
        if pd.isna(v):
            continue
        if phase == v:
            continue
        statuses.append(str(v).strip())

    seen: set[str] = set()
    out: list[str] = []
    for s in statuses:
        if s in seen:
            continue
        seen.add(s)
        out.append(s)

    if phase is not None:
        return []

    if len(out) > 1:
        drop_keys = {"nd", "n d", "nao medido"}
        filtered = [s for s in out if _norm_text(s) not in drop_keys]
        if filtered:
            out = filtered
        if len(out) > 1:
            out = [out[0]]
    return out


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
    hm_col = _find_col(df, ["hidrometro", "manha"])
    ht_col = _find_col(df, ["hidrometro", "tarde"])
    if hm_col:
        df[hm_col] = df[hm_col].map(_parse_hidrometro_value)
    if ht_col:
        df[ht_col] = df[ht_col].map(_parse_hidrometro_value)
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
    hm_col = _find_col(df, ["hidrometro", "manha"])
    ht_col = _find_col(df, ["hidrometro", "tarde"])
    if hm_col:
        df[hm_col] = df[hm_col].map(_parse_hidrometro_value)
    if ht_col:
        df[ht_col] = df[ht_col].map(_parse_hidrometro_value)
    vol_col = _find_col(df, ["volume", "infiltrado"])
    if vol_col:
        df[vol_col] = df[vol_col].map(parse_ptbr_number)
        df = df.rename(columns={vol_col: "infiltrado_vol"})
    else:
        df["infiltrado_vol"] = pd.NA
    return df


def prep_na_semanal(na_semanal: pd.DataFrame) -> pd.DataFrame:
    df = normalize_dates(na_semanal, "Data")
    poco_col = _get_poco_col(df)
    df["poco_key"] = df[poco_col] if poco_col else pd.NA
    df["PR"] = df["poco_key"].map(add_pr_for_pb_pi)
    df["entity_key"] = df["PR"].fillna(df["poco_key"])

    col_map = {"NA (m)": "na", "NO (m)": "no", "FL (m)": "fl"}
    for col, prefix in col_map.items():
        if col in df.columns:
            raw = df[col].map(_parse_na_value)
        else:
            raw = pd.Series(pd.NA, index=df.index)
        df[f"{prefix}_raw"] = raw
        df[f"{prefix}_val"] = pd.to_numeric(raw, errors="coerce")
        df[f"{prefix}_status"] = raw.where(raw.map(_is_text_value), pd.NA)

        missing_mask = raw.isna()
        if missing_mask.any():
            filled = df[f"{prefix}_status"].groupby(df["poco_key"]).ffill()
            df[f"{prefix}_status"] = df[f"{prefix}_status"].where(~missing_mask, filled)

    fl_phase_raw = df["fl_status"].map(_fl_phase)
    df["fl_phase"] = fl_phase_raw.where(
        df["no_val"].isna() & df["fl_val"].isna(),
        pd.NA,
    )
    return df


def build_na_pr_vs_infiltrado(na: pd.DataFrame, vi: pd.DataFrame) -> pd.DataFrame:
    na_pr = na[na["PR"].notna()].copy()
    na_pr = na_pr.groupby("Data", as_index=False)["na_val"].mean()

    vi_daily = vi.groupby("Data", as_index=False)["infiltrado_vol"].sum()

    out = na_pr.merge(vi_daily, on="Data", how="outer")
    return out.sort_values("Data")


def build_point_series(
    vb: pd.DataFrame,
    na: pd.DataFrame,
    points: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    vb_f = vb[vb["poco_key"].isin(points)].copy()
    na_f = na[na["entity_key"].isin(points)].copy()

    vb_wide = (
        vb_f.groupby(["Data", "poco_key"], as_index=False)["bombeado_vol"]
        .sum()
        .pivot_table(index="Data", columns="poco_key", values="bombeado_vol", aggfunc="sum")
    )
    vb_wide = vb_wide.add_prefix("bombeado__")

    if na_f.empty:
        na_wide = pd.DataFrame(index=vb_wide.index)
        na_flat = pd.DataFrame(columns=["Data", "entity_key"])
    else:
        na_flat = na_f.groupby(["Data", "entity_key"], as_index=False).agg(
            na_val=("na_val", "mean"),
            no_val=("no_val", "mean"),
            fl_val=("fl_val", "mean"),
            na_status=("na_status", _first_non_null),
            no_status=("no_status", _first_non_null),
            fl_status=("fl_status", _first_non_null),
            fl_phase=("fl_phase", _first_non_null),
        )
        na_flat["fl_phase"] = na_flat["fl_phase"].where(
            na_flat["no_val"].isna() & na_flat["fl_val"].isna(),
            pd.NA,
        )
        na_flat["na_base"] = na_flat["no_val"].where(na_flat["no_val"].notna(), na_flat["na_val"])

        na_wide = (
            na_flat.pivot_table(index="Data", columns="entity_key", values="na_base", aggfunc="mean")
            .add_prefix("na_base__")
        )
        fl_wide = (
            na_flat.pivot_table(index="Data", columns="entity_key", values="fl_val", aggfunc="mean")
            .add_prefix("fl_num__")
        )
        na_wide = pd.concat([na_wide, fl_wide], axis=1)

    if "Data" in na_flat.columns and not na_flat.empty:
        na_dates = pd.Index(pd.to_datetime(na_flat["Data"].unique()), name="Data")
        full_index = vb_wide.index.union(na_dates)
    else:
        full_index = vb_wide.index

    if full_index.name is None:
        full_index = full_index.rename("Data")

    vb_wide = vb_wide.reindex(full_index)
    na_wide = na_wide.reindex(full_index)
    vb_wide.index.name = "Data"
    na_wide.index.name = "Data"

    wide = pd.concat([vb_wide, na_wide], axis=1).reset_index()
    return wide, na_flat

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

<<<<<<< HEAD
if isinstance(df_dict, dict):
    table_name = st.selectbox("Tabela:", list(df_dict.keys()))
    df = df_dict[table_name].copy()
elif isinstance(df_dict, pd.DataFrame):
    table_name = "Monitoramento Laboratorial"
    df = df_dict

# De-duplica nomes de colunas para evitar groupby com colunas ambiguas
if df.columns.duplicated().any():
    seen = {}
    new_cols = []
    for col in df.columns:
        if col in seen:
            seen[col] += 1
            new_cols.append(f"{col}__dup{seen[col]}")
        else:
            seen[col] = 0
            new_cols.append(col)
    df.columns = new_cols
    st.warning("Colunas duplicadas detectadas; renomeadas com sufixo __dupN.")

# ✅ Mapear colunas (timestamp e poço)
cols = list(df.columns)

default_time = guess_first_existing(cols, ["timestamp", "data", "Data", "DATA", "date", "Date"])
default_poco = guess_first_existing(cols, ["poco", "Poço", "poço", "ponto", "Ponto", "well", "Well"])

if not default_time:
    st.warning("Não encontrei uma coluna de data/hora automaticamente. Selecione manualmente abaixo.")

if not default_poco:
    st.warning("Não encontrei uma coluna de poço/ponto automaticamente. Selecione manualmente abaixo.")

with st.expander("⚙️ Colunas"):
    time_col = st.selectbox(
        "Data/Hora:",
        options=cols,
        index=cols.index(default_time) if default_time in cols else 0,
    )

    group_options = [c for c in cols if c != time_col]
    if not group_options:
        st.error("Selecione uma coluna de data/hora diferente para liberar a coluna de poço/ponto.")
        st.stop()

    group_default = default_poco if default_poco in group_options else group_options[0]
    group_col_base = st.selectbox(
        "Poço/Ponto:",
        options=group_options,
        index=group_options.index(group_default),
    )

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
=======
for required in ["Volume Bombeado", "Volume Infiltrado", "NA Semanal"]:
    if required not in df_dict:
        st.error(f"Planilha obrigatoria ausente: {required}")
>>>>>>> origin/experimental-rework
        st.stop()

vb = prep_vol_bombeado(df_dict["Volume Bombeado"])
vi = prep_vol_infiltrado(df_dict["Volume Infiltrado"])
na = prep_na_semanal(df_dict["NA Semanal"])

st.subheader("Media do NA (PR) vs Volume Infiltrado")

na_vi = build_na_pr_vs_infiltrado(na, vi)

series = [
    SeriesSpec(
        y="na_val",
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

wide, na_flat = build_point_series(vb, na, selected_points)

color_map = _point_color_map(selected_points)

if "Data" not in wide.columns and "index" in wide.columns:
    wide = wide.rename(columns={"index": "Data"})
if "Data" not in wide.columns:
    st.info("Sem dados de data para os pontos selecionados.")
    st.stop()
wide = wide.set_index("Data")

phase_colors = {
    "Odor": "#f1c40f",
    "Oleoso": "#f39c12",
    "Pelicula": "#e74c3c",
}
na_palette = [
    "rgba(46, 139, 87, 0.6)",
    "rgba(40, 161, 152, 0.6)",
    "rgba(35, 139, 214, 0.6)",
    "rgba(58, 105, 214, 0.6)",
    "rgba(94, 88, 214, 0.6)",
]
na_color_map = {p: na_palette[i % len(na_palette)] for i, p in enumerate(selected_points)}
fl_bar_color = "rgba(214, 39, 40, 0.6)"
vb_palette = [
    "#ffcc00",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#bcbd22",
]
vb_color_map = {p: vb_palette[i % len(vb_palette)] for i, p in enumerate(selected_points)}
status_marker_color = "rgba(130, 130, 130, 0.8)"

phase_specs: list[SeriesSpec] = []
status_specs: list[SeriesSpec] = []

for point in selected_points:
    base_col = f"na_base__{point}"
    fl_col = f"fl_num__{point}"

    if base_col not in wide.columns and fl_col not in wide.columns:
        continue

    base_vals = wide[base_col] if base_col in wide.columns else pd.Series(pd.NA, index=wide.index)
    fl_vals = wide[fl_col] if fl_col in wide.columns else pd.Series(0.0, index=wide.index)

    bar_top = base_vals
    if fl_col in wide.columns:
        bar_top = bar_top.where(bar_top.notna(), 0) + fl_vals.fillna(0)
        bar_top = bar_top.where(base_vals.notna() | fl_vals.notna(), pd.NA)

    if bar_top.notna().any():
        span = float(bar_top.max() - bar_top.min())
    else:
        span = 1.0
    offset = max(0.05, span * 0.03)

    marker_y = bar_top.where(bar_top.notna(), 0) + offset

    point_status = na_flat[na_flat["entity_key"] == point]
    if point_status.empty:
        continue
    point_status = point_status.set_index("Data")

<<<<<<< HEAD
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
=======
    for phase in ("Odor", "Oleoso", "Pelicula"):
        phase_col = f"fl_phase__{point}__{phase}"
        phase_series = pd.Series(pd.NA, index=wide.index)
        mask = point_status["fl_phase"] == phase
        if mask.any():
            mask_idx = mask[mask].index.intersection(wide.index)
            phase_series.loc[mask_idx] = marker_y.reindex(mask_idx)
            wide[phase_col] = phase_series
            phase_specs.append(
                SeriesSpec(
                    y=phase_col,
                    label=f"FL {phase} - {point}",
                    kind="scatter",
                    marker="triangle-up",
                    color=phase_colors.get(phase, "#e74c3c"),
                    axis="y",
                )
>>>>>>> origin/experimental-rework
            )

    status_lists = point_status.apply(_collect_statuses, axis=1)
    for date_key, statuses in status_lists.items():
        for status in statuses:
            col_key = f"status__{point}__{_slug_status(status)}"
            if col_key not in wide.columns:
                wide[col_key] = pd.NA
                status_specs.append(
                    SeriesSpec(
                        y=col_key,
                        label=f"{status} - {point}",
                        kind="scatter",
                        marker="square",
                        color=status_marker_color,
                        axis="y",
                    )
                )
            if date_key in wide.index:
                wide.at[date_key, col_key] = marker_y.get(date_key, offset)

wide = wide.reset_index()

<<<<<<< HEAD
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
=======
base_series: list[SeriesSpec] = []
fl_series: list[SeriesSpec] = []
line_series: list[SeriesSpec] = []
for point in selected_points:
    col_base = f"na_base__{point}"
    if col_base in wide.columns:
        na_color = na_color_map.get(point, na_palette[0])
        base_series.append(
            SeriesSpec(
                y=col_base,
                label=f"NA/NO (m) - {point}",
                kind="bar",
                color=na_color,
                axis="y",
            )
        )
    col_fl = f"fl_num__{point}"
    if col_fl in wide.columns:
        fl_series.append(
            SeriesSpec(
                y=col_fl,
                label=f"Fase livre (m) - {point}",
                kind="bar",
                color=fl_bar_color,
                axis="y",
            )
        )
    col_vb = f"bombeado__{point}"
    if col_vb in wide.columns:
        vb_color = vb_color_map.get(point, vb_palette[0])
        line_series.append(
            SeriesSpec(
                y=col_vb,
                label=f"Volume Bombeado - {point}",
                kind="line",
                marker="circle",
                color=vb_color,
                axis="y2",
>>>>>>> origin/experimental-rework
            )
        )

series = base_series + fl_series + line_series

series.extend(phase_specs)
series.extend(status_specs)

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
fig2.update_layout(barmode="stack", bargap=0.0, bargroupgap=0.05)

for trace in fig2.data:
    if getattr(trace, "type", None) != "bar":
        continue
    if not trace.name or " - " not in trace.name:
        continue
    trace.offsetgroup = trace.name.split(" - ", 1)[-1]

bar_width = 1000 * 60 * 60 * 24 * 3
fig2.update_traces(width=bar_width, selector=dict(type="bar"))
fig2.update_yaxes(title_text="NA (m)", secondary_y=False)
fig2.update_yaxes(title_text="Volume Bombeado (m3)", secondary_y=True)

st.plotly_chart(fig2, use_container_width=True)
