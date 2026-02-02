import re
import unicodedata

import pandas as pd
import streamlit as st

from charts.builder import SeriesSpec, build_time_chart_plotly
from services.date_num_prep import normalize_dates, parse_ptbr_number


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
    if s == "" or s == "-":
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
    if "oleos" in n or "oleosid" in n:
        return "Oleoso"
    if "pelicul" in n:
        return "Pelicula"
    if "iridescen" in n:
        return "Iridescencia"
    if "seco" in n:
        return "Seco"

    return s


def _is_text_value(value: object) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _extract_terms(value: object) -> tuple[list[str], list[str], float | None]:
    if value is None or pd.isna(value):
        return [], [], None
    s = str(value).strip()
    if s == "" or s == "-":
        return [], [], None

    n_full = _norm_text(s)
    depth = None
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*prof", n_full)
    if m:
        try:
            depth = float(m.group(1).replace(",", "."))
        except Exception:
            depth = None

    if " e " in n_full or " | " in n_full:
        parts = re.split(r"\s+e\s+|\s+\|\s+", s, flags=re.I)
    else:
        parts = [s]

    phases: list[str] = []
    statuses: list[str] = []

    for part in parts:
        n = _norm_text(part)
        found = False

        for key, label in (
            ("odor", "Odor"),
            ("oleos", "Oleoso"),
            ("oleosid", "Oleoso"),
            ("pelicul", "Pelicula"),
            ("iridescen", "Iridescencia"),
        ):
            if key in n:
                phases.append(label)
                n = n.replace(key, " ")
                found = True

        for key, label in (
            ("seco", "Seco"),
            ("satur", "Saturado"),
        ):
            if key in n:
                statuses.append(label)
                n = n.replace(key, " ")
                found = True

        if not found:
            statuses.append(part.strip())

    def _dedupe(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in items:
            key = _norm_text(item)
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    return _dedupe(phases), _dedupe(statuses), depth


def _parse_observacao_text(value: object) -> tuple[str | None, str | None, float | None]:
    phases, statuses, depth = _extract_terms(value)
    phase_text = " | ".join(phases) if phases else None
    status_text = " | ".join(statuses) if statuses else None
    return phase_text, status_text, depth


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
    if "oleos" in n or "oleosid" in n:
        return "Oleoso"
    if "pelicul" in n:
        return "Pelicula"
    if "iridescen" in n:
        return "Iridescencia"
    return None


def _collect_statuses(row: pd.Series) -> list[str]:
    statuses: list[str] = []
    phase = row.get("fl_phase")
    if pd.isna(phase):
        phase = None
    if any(pd.notna(row.get(k)) for k in ("na_val", "no_val", "fl_val")):
        return []
    for key in ("na_status", "no_status", "fl_status", "obs_status"):
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
        drop_keys = {"nd", "n d", "nao medido", "seco", "abaixar bomba", "abaixar a bomba"}
        filtered = [s for s in out if _norm_text(s) not in drop_keys]
        if filtered:
            out = filtered
    else:
        out = [s for s in out if _norm_text(s) not in {"abaixar bomba", "abaixar a bomba"}]
    expanded: list[str] = []
    for s in out:
        _, statuses, _ = _extract_terms(s)
        expanded.extend(statuses)
    return expanded


def _find_col(df: pd.DataFrame, tokens: list[str]) -> str | None:
    for col in df.columns:
        norm = _norm_key(col)
        if all(t in norm for t in tokens):
            return col
    return None


def _get_poco_col(df: pd.DataFrame) -> str | None:
    return _find_col(df, ["poco"]) or _find_col(df, ["ponto"])

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
    df["entity_key"] = df["poco_key"]

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
    fl_phase = fl_phase_raw.where(df["no_val"].isna() & df["fl_val"].isna(), pd.NA)

    obs_phase = pd.Series(pd.NA, index=df.index)
    obs_status = pd.Series(pd.NA, index=df.index)
    obs_depth = pd.Series(pd.NA, index=df.index)

    obs_text = pd.Series(pd.NA, index=df.index)
    if "Observacao" in df.columns:
        obs_text = df["Observacao"]
    if "Observação" in df.columns:
        obs_text = obs_text.fillna(df["Observação"])

    if obs_text.notna().any():
        parsed = obs_text.map(_parse_observacao_text)
        obs_phase = parsed.map(lambda x: x[0])
        obs_status = parsed.map(lambda x: x[1])
        obs_depth = parsed.map(lambda x: x[2])

    if "Status" in df.columns:
        status_parsed = df["Status"].map(_parse_observacao_text)
        status_phase = status_parsed.map(lambda x: x[0])
        status_status = status_parsed.map(lambda x: x[1])
        status_depth = status_parsed.map(lambda x: x[2])
        obs_phase = obs_phase.fillna(status_phase)
        obs_status = obs_status.fillna(status_status)
        obs_depth = obs_depth.fillna(status_depth)

    fl_phase = fl_phase.fillna(obs_phase)
    df["obs_status"] = obs_status
    df["dry_depth"] = obs_depth.where(obs_status.map(lambda x: _norm_text(x) if pd.notna(x) else "") == "seco")

    if "NA (m)" in df.columns:
        na_terms = df["NA (m)"].map(_extract_terms)
        na_status_list = na_terms.map(lambda x: x[1])
        na_depth = na_terms.map(lambda x: x[2])
        na_has_seco = na_status_list.map(
            lambda items: any(_norm_text(s) == "seco" for s in items) if isinstance(items, list) else False
        )
        df["dry_depth"] = df["dry_depth"].where(df["dry_depth"].notna(), na_depth.where(na_has_seco))

    df["fl_phase"] = fl_phase.where(df["no_val"].isna() & df["fl_val"].isna(), pd.NA)
    return df


def prep_in_situ(in_situ: pd.DataFrame) -> pd.DataFrame:
    df = normalize_dates(in_situ, "Data")
    point_col = _get_poco_col(df) or ("Ponto" if "Ponto" in df.columns else None)
    if point_col and point_col != "Ponto":
        df = df.rename(columns={point_col: "Ponto"})
    if "Ponto" in df.columns:
        df["Ponto"] = df["Ponto"].astype(str).str.strip().str.upper()

    for col in df.columns:
        if col in ("Data", "Ponto"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_na_pr_vs_infiltrado(na: pd.DataFrame, vi: pd.DataFrame) -> pd.DataFrame:
    if "Data" not in na.columns or "na_val" not in na.columns or "entity_key" not in na.columns:
        na_pr = pd.DataFrame(columns=["Data", "na_val"])
    else:
        na_pr = na[na["entity_key"].notna()].copy()
        na_pr = na_pr.groupby("Data", as_index=False)["na_val"].mean()

    if "Data" not in vi.columns or "infiltrado_vol" not in vi.columns:
        vi_daily = pd.DataFrame(columns=["Data", "infiltrado_vol"])
    else:
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
            dry_depth=("dry_depth", _first_non_null),
            obs_status=("obs_status", _first_non_null),
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
        na_val_wide = (
            na_flat.pivot_table(index="Data", columns="entity_key", values="na_val", aggfunc="mean")
            .add_prefix("na_val__")
        )
        no_val_wide = (
            na_flat.pivot_table(index="Data", columns="entity_key", values="no_val", aggfunc="mean")
            .add_prefix("no_val__")
        )
        na_wide = pd.concat([na_wide, fl_wide, na_val_wide, no_val_wide], axis=1)

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


st.title("NA e Volume - Visualizacoes")

df_dict = st.session_state.get("df_dict")
if df_dict is None or not isinstance(df_dict, dict):
    st.info("Arquivo foi carregado, mas ainda nao ha dataset em memoria.")
    st.stop()
    df_dict = {}

for required in ["Volume Bombeado", "Volume Infiltrado", "NA Semanal"]:
    if required not in df_dict:
        st.error(f"Planilha obrigatoria ausente: {required}")
        st.stop()
        df_dict[required] = pd.DataFrame()

vb = prep_vol_bombeado(df_dict["Volume Bombeado"])
vi = prep_vol_infiltrado(df_dict["Volume Infiltrado"])
na = prep_na_semanal(df_dict["NA Semanal"])

in_situ_pontos = None
in_situ_geral = None
for key in ("In Situ (Pontos)", "In Situ"):
    if key in df_dict:
        try:
            prepared = prep_in_situ(df_dict[key])
            if not prepared.empty:
                in_situ_pontos = prepared
                break
        except Exception as e:
            st.warning(f"Falha ao preparar dados de In Situ (Pontos): {e}")

if "In Situ (Geral)" in df_dict:
    try:
        prepared = prep_in_situ(df_dict["In Situ (Geral)"])
        if not prepared.empty:
            in_situ_geral = prepared
    except Exception as e:
        st.warning(f"Falha ao preparar dados de In Situ (Geral): {e}")

subpage_options = ["Media NA vs Volume Infiltrado", "Visualizacao aprofundada"]
if in_situ_pontos is not None or in_situ_geral is not None:
    subpage_options.append("In situ")

with st.sidebar:
    st.subheader("Abas")
    subpage = st.radio(
        "Selecionar grafico",
        subpage_options,
        index=0,
        horizontal=False,
    )

if subpage == "Media NA vs Volume Infiltrado":
    st.subheader("Media do NA (PR) vs Volume Infiltrado")

    na_vi = build_na_pr_vs_infiltrado(na, vi)

    insitu_val_col = None
    insitu_param_avg = None
    insitu_label = None
    if in_situ_pontos is not None or in_situ_geral is not None:
        insitu_params = []
        if in_situ_pontos is not None:
            insitu_params = [c for c in in_situ_pontos.columns if c not in ("Data", "Ponto")]
        if not insitu_params and in_situ_geral is not None:
            insitu_params = [c for c in in_situ_geral.columns if c not in ("Data", "Ponto")]

        if insitu_params:
            add_insitu = st.checkbox("Exibir In situ", value=False, key="avg_insitu_toggle")
            if add_insitu:
                modes = []
                if in_situ_pontos is not None:
                    modes.append("Por ponto")
                if in_situ_geral is not None:
                    modes.append("Geral")
                insitu_mode = modes[0]
                if len(modes) > 1:
                    insitu_mode = st.radio("Tipo de In situ", modes, horizontal=True)

                insitu_param_avg = st.selectbox(
                    "Parametro In situ",
                    insitu_params,
                    key="avg_insitu_param",
                )

                if insitu_mode == "Por ponto":
                    insitu_points = sorted(in_situ_pontos["Ponto"].dropna().unique().tolist())
                    default_points = insitu_points if len(insitu_points) <= 8 else insitu_points[:8]
                    selected_insitu_points = st.multiselect(
                        "Pontos In situ",
                        insitu_points,
                        default=default_points,
                        key="avg_insitu_points",
                    )
                    if not selected_insitu_points:
                        st.info("Selecione ao menos um ponto para exibir o In situ.")
                    else:
                        df_insitu = in_situ_pontos[in_situ_pontos["Ponto"].isin(selected_insitu_points)].copy()
                        df_insitu["Data"] = pd.to_datetime(df_insitu["Data"], errors="coerce")
                        df_insitu = df_insitu.dropna(subset=["Data"])
                        if not df_insitu.empty and insitu_param_avg in df_insitu.columns:
                            insitu_daily = (
                                df_insitu.groupby("Data", as_index=False)[insitu_param_avg]
                                .mean()
                                .rename(columns={insitu_param_avg: "insitu_val"})
                            )
                            na_vi = na_vi.merge(insitu_daily, on="Data", how="outer")
                            insitu_val_col = "insitu_val"
                            insitu_label = f"{insitu_param_avg} (In situ)"
                        else:
                            st.info("Nenhum dado In situ encontrado para o parametro/pontos escolhidos.")
                else:
                    insitu_points = []
                    if "Ponto" in in_situ_geral.columns:
                        insitu_points = sorted(in_situ_geral["Ponto"].dropna().unique().tolist())
                    if insitu_points:
                        selected_insitu_point = st.selectbox(
                            "Ponto In situ (geral)",
                            insitu_points,
                            key="avg_insitu_point_geral",
                        )
                        df_insitu = in_situ_geral[in_situ_geral["Ponto"] == selected_insitu_point].copy()
                    else:
                        df_insitu = in_situ_geral.copy()
                    df_insitu["Data"] = pd.to_datetime(df_insitu["Data"], errors="coerce")
                    df_insitu = df_insitu.dropna(subset=["Data"])
                    if not df_insitu.empty and insitu_param_avg in df_insitu.columns:
                        insitu_daily = (
                            df_insitu.groupby("Data", as_index=False)[insitu_param_avg]
                            .mean()
                            .rename(columns={insitu_param_avg: "insitu_val"})
                        )
                        na_vi = na_vi.merge(insitu_daily, on="Data", how="outer")
                        insitu_val_col = "insitu_val"
                        if insitu_points:
                            insitu_label = f"{insitu_param_avg} (In situ geral - {selected_insitu_point})"
                        else:
                            insitu_label = f"{insitu_param_avg} (In situ geral)"
                    else:
                        st.info("Nenhum dado In situ geral encontrado para o parametro escolhido.")

    series = [
        SeriesSpec(
            y="na_val",
            label="NA medio PR (m)",
            kind="line",
            marker="circle",
            color="#146c43",
            connect_gaps=True,
        ),
        SeriesSpec(
            y="infiltrado_vol",
            label="Volume Infiltrado",
            kind="bar",
            axis="y2",
            color="rgba(133, 193, 233, 0.45)",
        ),
    ]
    if insitu_val_col and insitu_val_col in na_vi.columns:
        series.append(
            SeriesSpec(
                y=insitu_val_col,
                label=insitu_label or f"{insitu_param_avg} (In situ)",
                kind="line",
                marker="diamond",
                color="#9467bd",
                axis="y",
                connect_gaps=True,
            )
        )

    fig, _ = build_time_chart_plotly(
        na_vi,
        x="Data",
        series=series,
        title="NA medio PR vs Volume Infiltrado",
        show_range_slider=False,
        limit_points=200000,
    )
    fig.update_yaxes(title_text="NA medio (m)", secondary_y=False)
    if any(s.axis == "y2" for s in series):
        fig.update_yaxes(title_text="Volume Infiltrado (m3)", secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)
elif subpage == "Visualizacao aprofundada":
    st.subheader("Visualizacao aprofundada")

    all_points = sorted(
        set(vb["poco_key"].dropna().unique().tolist())
        | set(na["entity_key"].dropna().unique().tolist())
    )
    all_points = [p for p in all_points if p and p != "Acumulado"]
    if not all_points:
        st.info("Nao ha pontos para visualizar.")
        st.stop()

    prefixes = sorted({str(p).strip().upper()[:2] for p in all_points if isinstance(p, str)})
    if not prefixes:
        st.info("Nao ha tipos de poco para selecionar.")
        st.stop()

    selected_prefix = st.selectbox("Tipo de poco", prefixes)
    all_points = [p for p in all_points if str(p).strip().upper().startswith(selected_prefix)]
    if not all_points:
        st.info("Nao ha pontos para o tipo selecionado.")
        st.stop()

    mode = st.radio(
        "Modo de selecao",
        ["Poco individual", "Multiplos pocos"],
        horizontal=True,
    )

    param_options = ["NA", "Volume bombeado"]
    if in_situ_pontos is not None:
        param_options.append("In situ (pontos)")
    if in_situ_geral is not None:
        param_options.append("In situ (geral)")
    insitu_param = None

    # Define palettes outside the mode conditional so they're available in both branches
    phase_colors = {
        "Odor": "#f1c40f",
        "Oleoso": "#f39c12",
        "Iridescencia": "#fff2a8",
        "Pelicula": "#e74c3c",
    }
    na_palette = [
        "#0b5d1e",
        "#136f63",
        "#1d6fb8",
        "#6c3fb4",
        "#b23c8a",
    ]
    vb_palette = [
        "#f59f00",
        "#f76707",
        "#f03e3e",
        "#e03131",
        "#c92a2a",
        "#b02525",
        "#862e9c",
    ]
    insitu_palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#17becf",
    ]

    if mode == "Poco individual":
        if "point_index" not in st.session_state:
            st.session_state["point_index"] = 0
        st.session_state["point_index"] = min(st.session_state["point_index"], len(all_points) - 1)

        sel_cols = st.columns([5, 2], gap="small")
        selected_point = sel_cols[0].selectbox(
            "Poco",
            all_points,
            index=st.session_state["point_index"],
        )
        st.session_state["point_index"] = all_points.index(selected_point)

        if "show_params_single" not in st.session_state:
            st.session_state["show_params_single"] = False
        if sel_cols[1].button("Adicionar parametros", use_container_width=True):
            st.session_state["show_params_single"] = not st.session_state["show_params_single"]

        nav_cols = st.columns([1, 1, 8], gap="small")
        if nav_cols[0].button("⏪", use_container_width=True, key="prev_point"):
            st.session_state["point_index"] = max(0, st.session_state["point_index"] - 1)
            st.rerun()
        if nav_cols[1].button("⏩", use_container_width=True, key="next_point"):
            st.session_state["point_index"] = min(len(all_points) - 1, st.session_state["point_index"] + 1)
            st.rerun()

        if st.session_state["show_params_single"]:
            selected_params = st.multiselect(
                "Parametros do grafico",
                param_options,
                default=["NA"],
                key="single_params",
            )
        else:
            selected_params = ["NA"]

        selected_points = [selected_point]
    else:
        if "multi_points" not in st.session_state:
            st.session_state["multi_points"] = []
        st.session_state["multi_points"] = [
            p for p in st.session_state["multi_points"] if p in all_points
        ]

        available_points = [p for p in all_points if p not in st.session_state["multi_points"]]
        if available_points:
            add_cols = st.columns([5, 2], gap="small")
            to_add = add_cols[0].selectbox("Adicionar poco", available_points, key="multi_add")
            if add_cols[1].button("Adicionar", use_container_width=True):
                st.session_state["multi_points"].append(to_add)
                st.rerun()
        else:
            st.info("Todos os pocos do tipo selecionado ja foram adicionados.")

        if st.session_state["multi_points"]:
            st.caption("Pocos selecionados:")
            for point in st.session_state["multi_points"]:
                row = st.columns([6, 1], gap="small")
                row[0].write(point)
                if row[1].button("X", key=f"rm_{point}"):
                    st.session_state["multi_points"] = [
                        p for p in st.session_state["multi_points"] if p != point
                    ]
                    st.rerun()

        selected_params = st.multiselect(
            "Parametros do grafico",
            param_options,
            default=["NA"],
            key="multi_params",
        )

        selected_points = list(st.session_state["multi_points"])

    insitu_param = None
    insitu_param_geral = None
    insitu_point_geral = None

    if "In situ (pontos)" in selected_params:
        if in_situ_pontos is None:
            st.warning("Aba 'In Situ (Pontos)' nao carregada. Parametro removido.")
            selected_params = [p for p in selected_params if p != "In situ (pontos)"]
        else:
            insitu_options = [c for c in in_situ_pontos.columns if c not in ("Data", "Ponto")]
            if not insitu_options:
                st.warning("Nenhum parametro numerico encontrado na aba In Situ (Pontos). Parametro removido.")
                selected_params = [p for p in selected_params if p != "In situ (pontos)"]
            else:
                insitu_param = st.selectbox(
                    "Parametro In situ (pontos)",
                    insitu_options,
                    key="insitu_param_selector_pontos",
                )

    if "In situ (geral)" in selected_params:
        if in_situ_geral is None:
            st.warning("Aba 'In Situ (Geral)' nao carregada. Parametro removido.")
            selected_params = [p for p in selected_params if p != "In situ (geral)"]
        else:
            insitu_options = [c for c in in_situ_geral.columns if c not in ("Data", "Ponto")]
            if not insitu_options:
                st.warning("Nenhum parametro numerico encontrado na aba In Situ (Geral). Parametro removido.")
                selected_params = [p for p in selected_params if p != "In situ (geral)"]
            else:
                insitu_param_geral = st.selectbox(
                    "Parametro In situ (geral)",
                    insitu_options,
                    key="insitu_param_selector_geral",
                )
                if "Ponto" in in_situ_geral.columns:
                    points_geral = sorted(in_situ_geral["Ponto"].dropna().unique().tolist())
                    if points_geral:
                        insitu_point_geral = st.selectbox(
                            "Ponto In situ (geral)",
                            points_geral,
                            key="insitu_point_selector_geral",
                        )

    if "NA" not in selected_params:
        selected_params = ["NA"] + selected_params

    if not selected_points:
        st.info("Selecione ao menos um poco para visualizar.")
        st.stop()

    wide, na_flat = build_point_series(vb, na, selected_points)

    if "Data" not in wide.columns and "index" in wide.columns:
        wide = wide.rename(columns={"index": "Data"})
    if "Data" not in wide.columns:
        st.info("Sem dados de data para os pocos selecionados.")
        st.stop()
    wide = wide.set_index("Data")
    insitu_cols_map: dict[str, str] = {}
    insitu_geral_col = None
    if insitu_param and "In situ (pontos)" in selected_params and in_situ_pontos is not None:
        insitu_data = in_situ_pontos[in_situ_pontos["Ponto"].isin(selected_points)].copy()
        insitu_data["Data"] = pd.to_datetime(insitu_data["Data"], errors="coerce")
        insitu_data = insitu_data.dropna(subset=["Data"])
        if insitu_param not in insitu_data.columns or insitu_data.empty:
            st.info("Nenhum dado In situ disponivel para o parametro/pontos selecionados.")
            selected_params = [p for p in selected_params if p != "In situ (pontos)"]
        else:
            pivot = insitu_data.pivot_table(
                index="Data",
                columns="Ponto",
                values=insitu_param,
                aggfunc="mean",
            )
            pivot.index = pd.to_datetime(pivot.index)
            insitu_cols_map = {pt: f"in_situ__{pt}" for pt in pivot.columns}
            pivot = pivot.rename(columns=insitu_cols_map)
            full_index = wide.index.union(pivot.index)
            wide = wide.reindex(full_index)
            pivot = pivot.reindex(full_index)
            wide = pd.concat([wide, pivot], axis=1)

    if insitu_param_geral and "In situ (geral)" in selected_params and in_situ_geral is not None:
        insitu_data = in_situ_geral.copy()
        if insitu_point_geral and "Ponto" in insitu_data.columns:
            insitu_data = insitu_data[insitu_data["Ponto"] == insitu_point_geral]
        insitu_data["Data"] = pd.to_datetime(insitu_data["Data"], errors="coerce")
        insitu_data = insitu_data.dropna(subset=["Data"])
        if insitu_param_geral not in insitu_data.columns or insitu_data.empty:
            st.info("Nenhum dado In situ geral disponivel para o parametro selecionado.")
            selected_params = [p for p in selected_params if p != "In situ (geral)"]
        else:
            daily = (
                insitu_data.groupby("Data", as_index=True)[insitu_param_geral]
                .mean()
                .rename("in_situ_geral")
            )
            insitu_geral_col = "in_situ_geral"
            full_index = wide.index.union(daily.index)
            wide = wide.reindex(full_index)
            wide[insitu_geral_col] = daily.reindex(full_index)

    if mode == "Poco individual":
        phase_colors = {
            "Odor": "#f1c40f",
            "Oleoso": "#f39c12",
            "Iridescencia": "#fff2a8",
            "Pelicula": "#e74c3c",
        }
        na_palette = [
            "#0b5d1e",
            "#136f63",
            "#1d6fb8",
            "#6c3fb4",
            "#b23c8a",
        ]
        na_color_map = {p: na_palette[i % len(na_palette)] for i, p in enumerate(selected_points)}
        fl_bar_color = "rgba(214, 39, 40, 0.35)"
        air_fill = "rgba(0, 0, 0, 0)"
        air_line = "rgba(150, 150, 150, 0.65)"
        water_tail_color = "rgba(76, 139, 245, 0.55)"
        water_fade_mid = "rgba(76, 139, 245, 0.32)"
        water_fade_low = "rgba(76, 139, 245, 0.16)"
        dry_marker_color = "rgba(200, 70, 70, 0.9)"
        vb_palette = [
            "#f59f00",
            "#f76707",
            "#f03e3e",
            "#e03131",
            "#c92a2a",
            "#b02525",
            "#862e9c",
        ]
        vb_color_map = {p: vb_palette[i % len(vb_palette)] for i, p in enumerate(selected_points)}
        insitu_palette = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#17becf",
        ]
        status_marker_color = "rgba(130, 130, 130, 0.8)"

        phase_specs: list[SeriesSpec] = []
        status_specs: list[SeriesSpec] = []
        use_offset = True

        for point in selected_points:
            na_val_col = f"na_val__{point}"
            no_val_col = f"no_val__{point}"
            fl_val_col = f"fl_num__{point}"

            point_status = na_flat[na_flat["entity_key"] == point]
            has_dry = not point_status.empty and point_status["dry_depth"].notna().any()
            if na_val_col not in wide.columns and no_val_col not in wide.columns and not has_dry:
                continue

            na_vals = wide[na_val_col] if na_val_col in wide.columns else pd.Series(pd.NA, index=wide.index)
            no_vals = wide[no_val_col] if no_val_col in wide.columns else pd.Series(pd.NA, index=wide.index)
            fl_vals = wide[fl_val_col] if fl_val_col in wide.columns else pd.Series(pd.NA, index=wide.index)

            if point_status.empty:
                continue
            point_status = point_status.set_index("Data")

            dry_dates = point_status.index[point_status["dry_depth"].notna()]
            if len(dry_dates) > 0:
                na_vals = na_vals.copy()
                no_vals = no_vals.copy()
                na_vals.loc[dry_dates.intersection(na_vals.index)] = pd.NA
                no_vals.loc[dry_dates.intersection(no_vals.index)] = pd.NA

            pump_mask = point_status["obs_status"].astype(str).map(_norm_text).str.contains("abaix")
            pump_dates = point_status.index[pump_mask]

            air_len = no_vals.where(no_vals.notna(), pd.NA)
            air_len = air_len.where(air_len.notna(), na_vals.where(na_vals.notna(), pd.NA))
            if len(dry_dates) > 0:
                air_len.loc[dry_dates.intersection(air_len.index)] = point_status.loc[
                    dry_dates.intersection(point_status.index),
                    "dry_depth",
                ]
            has_na = na_vals.notna()
            has_no = no_vals.notna()
            has_fl = fl_vals.notna()
            fl_len = fl_vals.where(has_fl, pd.NA)
            fl_len = fl_len.where(fl_len.notna(), (na_vals - no_vals))
            fl_len = fl_len.where(has_na & (has_no | has_fl), pd.NA)
            fl_len = fl_len.where(fl_len > 0, pd.NA)
            water_top_len = pd.Series(0.12, index=wide.index).where(na_vals.notna(), pd.NA)
            water_mid_len = pd.Series(0.08, index=wide.index).where(na_vals.notna(), pd.NA)
            water_low_len = pd.Series(0.05, index=wide.index).where(na_vals.notna(), pd.NA)

            if len(pump_dates) > 0:
                for d in pump_dates:
                    if d in wide.index:
                        water_top_len.loc[d] = pd.NA
                        water_mid_len.loc[d] = pd.NA
                        water_low_len.loc[d] = pd.NA

            air_col = f"air__{point}"
            fl_seg_col = f"fl_seg__{point}"
            fl_base_col = f"fl_base__{point}"
            water_top_col = f"water_top__{point}"
            water_mid_col = f"water_mid__{point}"
            water_low_col = f"water_low__{point}"
            dry_col = f"dry__{point}"
            pump_col = f"pump__{point}"
            pump_missing_col = f"pump_missing__{point}"

            wide[air_col] = air_len
            wide[fl_seg_col] = fl_len
            fl_base_vals = no_vals.where(has_no, pd.NA)
            fl_base_vals = fl_base_vals.where(fl_base_vals.notna(), (na_vals - fl_len))
            wide[fl_base_col] = fl_base_vals
            wide[water_top_col] = water_top_len
            wide[water_mid_col] = water_mid_len
            wide[water_low_col] = water_low_len
            if len(dry_dates) > 0:
                wide[dry_col] = pd.NA
                for d in dry_dates:
                    if d in wide.index:
                        wide.at[d, dry_col] = point_status.at[d, "dry_depth"]
            if len(pump_dates) > 0:
                wide[pump_col] = pd.NA
                wide[pump_missing_col] = pd.NA
                for d in pump_dates:
                    if d in wide.index:
                        if pd.notna(na_vals.get(d, pd.NA)):
                            wide.at[d, pump_col] = na_vals.get(d, pd.NA)
                        else:
                            wide.at[d, pump_missing_col] = 0.08
                            wide.at[d, pump_col] = 0.12

            base_vals = na_vals
            if base_vals.notna().any():
                span = float(base_vals.max() - base_vals.min())
            else:
                span = 1.0
            offset = max(0.05, span * 0.03)

            marker_y = base_vals.where(base_vals.notna(), 0) - offset

            for phase in ("Odor", "Oleoso", "Iridescencia", "Pelicula"):
                phase_col = f"fl_phase__{point}__{phase}"
                phase_x_col = f"x__{phase_col}"
                phase_series = pd.Series(pd.NA, index=wide.index)
                mask = (
                    point_status["fl_phase"]
                    .astype(str)
                    .map(_norm_text)
                    .str.contains(_norm_text(phase), regex=False)
                )
                if mask.any():
                    mask_idx = mask[mask].index.intersection(wide.index)
                    phase_series.loc[mask_idx] = marker_y.reindex(mask_idx)
                    wide[phase_col] = phase_series
                    phase_specs.append(
                        SeriesSpec(
                            y=phase_col,
                            x=phase_x_col if use_offset else None,
                            label=f"FL {phase} - {point}",
                            kind="scatter",
                            marker="triangle-up",
                            color=phase_colors.get(phase, "#e74c3c"),
                            marker_line_color="#ffffff" if phase == "Iridescencia" else None,
                            marker_line_width=1.5 if phase == "Iridescencia" else None,
                            axis="y",
                        )
                    )
                    if use_offset:
                        for d in mask_idx:
                            if d not in wide.index:
                                continue
                            phases_here = [
                                p for p in ("Odor", "Oleoso", "Iridescencia", "Pelicula")
                                if _norm_text(p) in _norm_text(str(point_status.at[d, "fl_phase"]))
                            ]
                            if not phases_here:
                                phases_here = [phase]
                            idx = phases_here.index(phase) if phase in phases_here else 0
                            offset_hours = (idx - (len(phases_here) - 1) / 2) * 36
                            wide.at[d, phase_x_col] = d + pd.Timedelta(hours=offset_hours)

            status_lists = point_status.apply(_collect_statuses, axis=1)
            for date_key, statuses in status_lists.items():
                if not statuses:
                    continue
                total = len(statuses)
                for idx, status in enumerate(statuses):
                    if _norm_text(status) == "seco" and date_key in dry_dates:
                        continue
                    col_key = f"status__{point}__{_slug_status(status)}"
                    x_col = f"x__{col_key}"
                    if col_key not in wide.columns:
                        wide[col_key] = pd.NA
                        if use_offset:
                            wide[x_col] = wide.index
                        norm_status = _norm_text(status)
                        marker = "square"
                        color = status_marker_color
                        line_color = None
                        line_width = None
                        if "iridescen" in norm_status:
                            marker = "triangle-up"
                            color = "#fff2a8"
                            line_color = "#ffffff"
                            line_width = 1.5
                        status_specs.append(
                            SeriesSpec(
                                y=col_key,
                                x=x_col if use_offset else None,
                                label=f"{status} - {point}",
                                kind="scatter",
                                marker=marker,
                                color=color,
                                marker_line_color=line_color,
                                marker_line_width=line_width,
                                axis="y",
                            )
                        )
                    if date_key in wide.index:
                        wide.at[date_key, col_key] = marker_y.get(date_key, offset)
                        if use_offset and total > 1:
                            offset_hours = (idx - (total - 1) / 2) * 36
                            wide.at[date_key, x_col] = date_key + pd.Timedelta(hours=offset_hours)

            if dry_col in wide.columns and wide[dry_col].notna().any():
                status_specs.append(
                    SeriesSpec(
                        y=dry_col,
                        label=f"Seco - {point}",
                        kind="scatter",
                        marker="x",
                        color=dry_marker_color,
                        axis="y",
                    )
                )
            if pump_col in wide.columns and wide[pump_col].notna().any():
                status_specs.append(
                    SeriesSpec(
                        y=pump_col,
                        label=f"Abaixar bomba - {point}",
                        kind="scatter",
                        marker="square",
                        color="rgba(255, 255, 255, 0.9)",
                        axis="y",
                    )
                )
            if pump_missing_col in wide.columns and wide[pump_missing_col].notna().any():
                status_specs.append(
                    SeriesSpec(
                        y=pump_missing_col,
                        label=f"Nao medido - {point}",
                        kind="bar",
                        color="rgba(170, 170, 170, 0.7)",
                        axis="y",
                    )
                )

        wide = wide.reset_index()

        air_series: list[SeriesSpec] = []
        fl_series: list[SeriesSpec] = []
        water_series: list[SeriesSpec] = []
        line_series: list[SeriesSpec] = []
        insitu_series: list[SeriesSpec] = []
        for point in selected_points:
            col_air = f"air__{point}"
            if col_air in wide.columns:
                air_series.append(
                    SeriesSpec(
                        y=col_air,
                        label=f"Ar (m) - {point}",
                        kind="bar",
                        color=air_fill,
                        axis="y",
                    )
                )
            col_fl = f"fl_seg__{point}"
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
            col_water_top = f"water_top__{point}"
            if col_water_top in wide.columns:
                water_series.append(
                    SeriesSpec(
                        y=col_water_top,
                        label=f"Agua (m) - {point}",
                        kind="bar",
                        color=water_tail_color,
                        axis="y",
                    )
                )
            col_water_mid = f"water_mid__{point}"
            if col_water_mid in wide.columns:
                water_series.append(
                    SeriesSpec(
                        y=col_water_mid,
                        label=f"Agua (fade) - {point}",
                        kind="bar",
                        color=water_fade_mid,
                        axis="y",
                    )
                )
            col_water_low = f"water_low__{point}"
            if col_water_low in wide.columns:
                water_series.append(
                    SeriesSpec(
                        y=col_water_low,
                        label=f"Agua (fade2) - {point}",
                        kind="bar",
                        color=water_fade_low,
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
                    )
                )
            if "In situ (pontos)" in selected_params and insitu_param and insitu_cols_map:
                col_insitu = insitu_cols_map.get(point)
                if col_insitu and col_insitu in wide.columns:
                    insitu_series.append(
                        SeriesSpec(
                            y=col_insitu,
                            label=f"{insitu_param} (In situ) - {point}",
                            kind="line",
                            marker="diamond",
                            color=insitu_palette[ selected_points.index(point) % len(insitu_palette)],
                            axis="y",
                            connect_gaps=True,
                        )
                    )
        if "In situ (geral)" in selected_params and insitu_param_geral and insitu_geral_col:
            if insitu_geral_col in wide.columns:
                label = f"{insitu_param_geral} (In situ geral)"
                if insitu_point_geral:
                    label = f"{insitu_param_geral} (In situ geral - {insitu_point_geral})"
                insitu_series.append(
                    SeriesSpec(
                        y=insitu_geral_col,
                        label=label,
                        kind="line",
                        marker="diamond",
                        color="#8c564b",
                        axis="y",
                        connect_gaps=True,
                    )
                )

        series = air_series + fl_series + water_series + line_series + insitu_series

        series.extend(phase_specs)
        series.extend(status_specs)

        if not series:
            st.info("Sem dados de NA, status ou volume bombeado para os pocos selecionados.")
            st.stop()

        title_suffix = ""
        if len(selected_points) == 1 and not na_flat.empty:
            point_key = selected_points[0]
            status_series = (
                na_flat[na_flat["entity_key"] == point_key]
                .sort_values("Data")["obs_status"]
                .dropna()
            )
            if not status_series.empty:
                title_suffix = f" ({status_series.iloc[-1]})"

        fig2, _ = build_time_chart_plotly(
            wide,
            x="Data",
            series=series,
            title=f"NA por ponto{title_suffix}",
            show_range_slider=False,
            limit_points=200000,
            return_insights=False,
        )
        fig2.update_layout(barmode="group", bargap=0.0, bargroupgap=0.05)

        for trace in fig2.data:
            if getattr(trace, "type", None) != "bar":
                continue
            if not trace.name or " - " not in trace.name:
                continue
            label, point = trace.name.split(" - ", 1)
            trace.offsetgroup = point
            trace.alignmentgroup = point
            if label.startswith("Fase livre"):
                base_col = f"fl_base__{point}"
                if base_col in wide.columns:
                    trace.base = wide[base_col].tolist()
            if label.startswith("Agua"):
                base_col = f"na_val__{point}"
                base_series = wide[base_col] if base_col in wide.columns else pd.Series(0, index=wide.index)
                trace.base = base_series.tolist()
                if label.startswith("Agua (fade)"):
                    fade_base = base_series.add(wide.get(f"water_top__{point}", 0)).tolist()
                    trace.base = fade_base
                if label.startswith("Agua (fade2)"):
                    fade_base = base_series.add(wide.get(f"water_top__{point}", 0)).add(
                        wide.get(f"water_mid__{point}", 0)
                    ).tolist()
                    trace.base = fade_base
            if label.startswith("Ar"):
                trace.marker.line = dict(color=air_line, width=1.6)
                trace.marker.color = "rgba(0, 0, 0, 0)"

            if label.startswith("Fase livre"):
                vals = trace.y
                trace.text = [
                    f"{v:.2f}" if isinstance(v, (int, float)) and pd.notna(v) else ""
                    for v in vals
                ]
                trace.textposition = "inside"
                trace.textfont = dict(color="#ffffff", size=10)

        for trace in fig2.data:
            if not getattr(trace, "name", ""):
                continue
            if trace.name.startswith("Agua (fade)"):
                trace.showlegend = False
            if trace.name.startswith("Agua (fade2)"):
                trace.showlegend = False

        bar_width = 1000 * 60 * 60 * 24 * 3
        fig2.update_traces(width=bar_width, selector=dict(type="bar"))
        fig2.update_yaxes(title_text="NA (m)", secondary_y=False, autorange="reversed")
        fig2.update_yaxes(title_text="Volume Bombeado (m3)", secondary_y=True)

        st.plotly_chart(fig2, use_container_width=True)
    else:
        na_color_map = {p: na_palette[i % len(na_palette)] for i, p in enumerate(selected_points)}
        vb_color_map = {p: vb_palette[i % len(vb_palette)] for i, p in enumerate(selected_points)}
        status_marker_color = "rgba(120, 120, 120, 0.85)"
        dry_marker_color = "rgba(200, 70, 70, 0.9)"

        def _display_status_label(status: str) -> str:
            norm = _norm_text(status)
            if norm in {"nao medido", "naeo medido", "n m", "nm"}:
                return "Nao medido"
            return status

        series: list[SeriesSpec] = []
        phase_specs: list[SeriesSpec] = []
        status_specs: list[SeriesSpec] = []
        use_offset = True

        if "NA" in selected_params:
            for point in selected_points:
                na_val_col = f"na_val__{point}"
                if na_val_col not in wide.columns:
                    na_val_col = f"na_base__{point}"
                if na_val_col in wide.columns:
                    series.append(
                    SeriesSpec(
                        y=na_val_col,
                        label=f"NA - {point}",
                        kind="line",
                        marker="circle",
                        color=na_color_map.get(point, na_palette[0]),
                        axis="y",
                        connect_gaps=True,
                    )
                    )

        if "Volume bombeado" in selected_params:
            for point in selected_points:
                col_vb = f"bombeado__{point}"
                if col_vb in wide.columns:
                    series.append(
                        SeriesSpec(
                            y=col_vb,
                            label=f"Volume Bombeado - {point}",
                            kind="bar",
                            color=vb_color_map.get(point, vb_palette[0]),
                            axis="y2",
                        )
                    )

        if "In situ (pontos)" in selected_params and insitu_param and insitu_cols_map:
            for point in selected_points:
                col_insitu = insitu_cols_map.get(point)
                if col_insitu and col_insitu in wide.columns:
                    series.append(
                        SeriesSpec(
                            y=col_insitu,
                            label=f"{insitu_param} (In situ) - {point}",
                            kind="line",
                            marker="diamond",
                            color=insitu_palette[selected_points.index(point) % len(insitu_palette)],
                            axis="y",
                            connect_gaps=True,
                        )
                    )

        if "In situ (geral)" in selected_params and insitu_param_geral and insitu_geral_col:
            if insitu_geral_col in wide.columns:
                label = f"{insitu_param_geral} (In situ geral)"
                if insitu_point_geral:
                    label = f"{insitu_param_geral} (In situ geral - {insitu_point_geral})"
                series.append(
                    SeriesSpec(
                        y=insitu_geral_col,
                        label=label,
                        kind="line",
                        marker="diamond",
                        color="#8c564b",
                        axis="y",
                        connect_gaps=True,
                    )
                )

        if "NA" in selected_params and not na_flat.empty:
            for point in selected_points:
                na_val_col = f"na_val__{point}"
                if na_val_col not in wide.columns:
                    na_val_col = f"na_base__{point}"
                if na_val_col not in wide.columns:
                    continue

                point_status = na_flat[na_flat["entity_key"] == point]
                if point_status.empty:
                    continue
                point_status = point_status.set_index("Data")

                base_vals = wide[na_val_col]
                if base_vals.notna().any():
                    span = float(base_vals.max() - base_vals.min())
                else:
                    span = 1.0
                offset = max(0.05, span * 0.03)
                marker_y = base_vals.where(base_vals.notna(), pd.NA) - offset

                for phase in ("Odor", "Oleoso", "Iridescencia", "Pelicula"):
                    phase_col = f"fl_phase__{point}__{phase}"
                    phase_x_col = f"x__{phase_col}"
                    phase_series = pd.Series(pd.NA, index=wide.index)
                    mask = (
                        point_status["fl_phase"]
                        .astype(str)
                        .map(_norm_text)
                        .str.contains(_norm_text(phase), regex=False)
                    )
                    if not mask.any():
                        continue
                    mask_idx = mask[mask].index.intersection(wide.index)
                    phase_series.loc[mask_idx] = marker_y.reindex(mask_idx)
                    wide[phase_col] = phase_series
                    phase_specs.append(
                        SeriesSpec(
                            y=phase_col,
                            x=phase_x_col if use_offset else None,
                            label=f"FL {phase} - {point}",
                            kind="scatter",
                            marker="triangle-up",
                            color=phase_colors.get(phase, "#e74c3c"),
                            marker_line_color="#ffffff" if phase == "Iridescencia" else None,
                            marker_line_width=1.5 if phase == "Iridescencia" else None,
                            axis="y",
                        )
                    )
                    if use_offset:
                        for d in mask_idx:
                            if d not in wide.index:
                                continue
                            phases_here = [
                                p for p in ("Odor", "Oleoso", "Iridescencia", "Pelicula")
                                if _norm_text(p) in _norm_text(str(point_status.at[d, "fl_phase"]))
                            ]
                            if not phases_here:
                                phases_here = [phase]
                            idx = phases_here.index(phase) if phase in phases_here else 0
                            offset_hours = (idx - (len(phases_here) - 1) / 2) * 36
                            wide.at[d, phase_x_col] = d + pd.Timedelta(hours=offset_hours)

                status_lists = point_status.apply(_collect_statuses, axis=1)
                for date_key, statuses in status_lists.items():
                    if not statuses:
                        obs = point_status.at[date_key, "obs_status"] if date_key in point_status.index else pd.NA
                        if (
                            isinstance(obs, str)
                            and "abaix" in _norm_text(obs)
                            and pd.isna(base_vals.get(date_key, pd.NA))
                        ):
                            statuses = ["Nao medido"]
                        else:
                            continue
                    total = len(statuses)
                    for idx, status in enumerate(statuses):
                        if _norm_text(status) == "seco":
                            continue
                        label = _display_status_label(status)
                        col_key = f"status__{point}__{_slug_status(label)}"
                        x_col = f"x__{col_key}"
                        if col_key not in wide.columns:
                            wide[col_key] = pd.NA
                            if use_offset:
                                wide[x_col] = wide.index
                            norm_status = _norm_text(label)
                            marker = "square"
                            color = status_marker_color
                            line_color = None
                            line_width = None
                            if "iridescen" in norm_status:
                                marker = "triangle-up"
                                color = "#fff2a8"
                                line_color = "#ffffff"
                                line_width = 1.5
                            status_specs.append(
                                SeriesSpec(
                                    y=col_key,
                                    x=x_col if use_offset else None,
                                    label=f"{label} - {point}",
                                    kind="scatter",
                                    marker=marker,
                                    color=color,
                                    marker_line_color=line_color,
                                    marker_line_width=line_width,
                                    axis="y",
                                )
                            )
                        if date_key in wide.index:
                            wide.at[date_key, col_key] = marker_y.get(date_key, offset)
                            if use_offset and total > 1:
                                offset_hours = (idx - (total - 1) / 2) * 36
                                wide.at[date_key, x_col] = date_key + pd.Timedelta(hours=offset_hours)

                if point_status["dry_depth"].notna().any():
                    dry_col = f"dry__{point}"
                    wide[dry_col] = pd.NA
                    dry_dates = point_status.index[point_status["dry_depth"].notna()]
                    for d in dry_dates:
                        if d in wide.index:
                            wide.at[d, dry_col] = point_status.at[d, "dry_depth"]
                    status_specs.append(
                        SeriesSpec(
                            y=dry_col,
                            label=f"Seco - {point}",
                            kind="scatter",
                            marker="x",
                            color=dry_marker_color,
                            axis="y",
                        )
                    )

        wide = wide.reset_index()
        series.extend(phase_specs)
        series.extend(status_specs)

        if not series:
            st.info("Sem dados suficientes para os pocos selecionados.")
            st.stop()

        fig2, _ = build_time_chart_plotly(
            wide,
            x="Data",
            series=series,
            title="NA por poco",
            show_range_slider=False,
            limit_points=200000,
            return_insights=False,
        )
        fig2.update_layout(barmode="group", bargap=0.0, bargroupgap=0.05)
        fig2.update_yaxes(title_text="NA (m)", secondary_y=False, autorange="reversed")
        if any(s.axis == "y2" for s in series):
            fig2.update_yaxes(title_text="Volume Bombeado (m3)", secondary_y=True)

        st.plotly_chart(fig2, use_container_width=True)
elif subpage == "In situ":
    st.subheader("In situ")

    if (in_situ_pontos is None or in_situ_pontos.empty) and (in_situ_geral is None or in_situ_geral.empty):
        st.info("Planilha 'In Situ' nao encontrada ou sem dados.")
        st.stop()

    modes = []
    if in_situ_pontos is not None and not in_situ_pontos.empty:
        modes.append("Por ponto")
    if in_situ_geral is not None and not in_situ_geral.empty:
        modes.append("Geral")
    mode = modes[0]
    if len(modes) > 1:
        mode = st.radio("Tipo de In situ", modes, horizontal=True)

    data_source = in_situ_pontos if mode == "Por ponto" else in_situ_geral
    params = [c for c in data_source.columns if c not in ("Data", "Ponto")]
    if not params:
        st.info("Nenhum parametro numerico identificado na aba In Situ.")
        st.stop()

    date_min = pd.to_datetime(data_source["Data"], errors="coerce").min()
    date_max = pd.to_datetime(data_source["Data"], errors="coerce").max()
    default_range = None
    if pd.notna(date_min) and pd.notna(date_max):
        default_range = (date_min.date(), date_max.date())

    if mode == "Por ponto":
        points = sorted(data_source["Ponto"].dropna().unique().tolist())
        if not points:
            st.info("Nenhum ponto encontrado na aba In Situ.")
            st.stop()

        ctrl_cols = st.columns([2, 2, 1], gap="small")
        selected_param = ctrl_cols[0].selectbox("Parametro", params)
        default_points = points if len(points) <= 8 else points[:8]
        selected_points = ctrl_cols[1].multiselect(
            "Pontos",
            points,
            default=default_points,
        )
        date_input = ctrl_cols[2].date_input("Periodo", value=default_range)

        if not selected_points:
            st.info("Selecione ao menos um ponto para exibir o grafico.")
            st.stop()

        data = data_source[data_source["Ponto"].isin(selected_points)].copy()
        data["Data"] = pd.to_datetime(data["Data"], errors="coerce")
        data = data.dropna(subset=["Data"])
    else:
        points = []
        if "Ponto" in data_source.columns:
            points = sorted(data_source["Ponto"].dropna().unique().tolist())

        if points:
            ctrl_cols = st.columns([2, 2, 1], gap="small")
            selected_param = ctrl_cols[0].selectbox("Parametro", params)
            default_points = points if len(points) <= 3 else points[:3]
            selected_points = ctrl_cols[1].multiselect(
                "Pontos gerais",
                points,
                default=default_points,
            )
            date_input = ctrl_cols[2].date_input("Periodo", value=default_range)
            if not selected_points:
                st.info("Selecione ao menos um ponto para exibir o grafico.")
                st.stop()
            data = data_source[data_source["Ponto"].isin(selected_points)].copy()
        else:
            ctrl_cols = st.columns([2, 1], gap="small")
            selected_param = ctrl_cols[0].selectbox("Parametro", params)
            date_input = ctrl_cols[1].date_input("Periodo", value=default_range)
            data = data_source.copy()

        data["Data"] = pd.to_datetime(data["Data"], errors="coerce")
        data = data.dropna(subset=["Data"])

    if isinstance(date_input, (list, tuple)) and len(date_input) == 2:
        start, end = date_input
        if start:
            data = data[data["Data"] >= pd.to_datetime(start)]
        if end:
            data = data[data["Data"] <= pd.to_datetime(end)]

    if mode == "Por ponto":
        chart_df = (
            data.pivot_table(index="Data", columns="Ponto", values=selected_param, aggfunc="mean")
            .reset_index()
            .sort_values("Data")
        )
        value_cols = [c for c in chart_df.columns if c != "Data"]
        if not value_cols:
            st.info("Nao ha valores para o parametro selecionado nos pontos escolhidos.")
            st.stop()

        palette = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#17becf",
        ]
        series = [
            SeriesSpec(
                y=col,
                label=f"{col} - {selected_param}",
                kind="line",
                marker="circle",
                color=palette[i % len(palette)],
                connect_gaps=True,
            )
            for i, col in enumerate(value_cols)
        ]
    else:
        if "Ponto" in data.columns:
            chart_df = (
                data.pivot_table(index="Data", columns="Ponto", values=selected_param, aggfunc="mean")
                .reset_index()
                .sort_values("Data")
            )
            value_cols = [c for c in chart_df.columns if c != "Data"]
            if not value_cols:
                st.info("Nao ha valores para o parametro selecionado no In situ geral.")
                st.stop()
            palette = [
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
                "#e377c2",
                "#17becf",
            ]
            series = [
                SeriesSpec(
                    y=col,
                    label=f"{col} - {selected_param}",
                    kind="line",
                    marker="circle",
                    color=palette[i % len(palette)],
                    connect_gaps=True,
                )
                for i, col in enumerate(value_cols)
            ]
        else:
            chart_df = (
                data.groupby("Data", as_index=False)[selected_param]
                .mean()
                .sort_values("Data")
            )
            if chart_df.empty:
                st.info("Nao ha valores para o parametro selecionado no In situ geral.")
                st.stop()
            series = [
                SeriesSpec(
                    y=selected_param,
                    label=f"{selected_param} - In situ geral",
                    kind="line",
                    marker="circle",
                    color="#8c564b",
                    connect_gaps=True,
                )
            ]

    fig3, _ = build_time_chart_plotly(
        chart_df,
        x="Data",
        series=series,
        title=f"{selected_param} - In situ",
        show_range_slider=False,
        limit_points=200000,
    )
    fig3.update_yaxes(title_text=selected_param, secondary_y=False)

    st.plotly_chart(fig3, use_container_width=True)
