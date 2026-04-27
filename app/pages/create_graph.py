import re
import unicodedata
import io

import pandas as pd
import streamlit as st

from charts.builder import SeriesSpec, build_time_chart_plotly
from components.period_selector import period_selector
from services.date_num_prep import normalize_dates, parse_ptbr_number
from services.in_situ_parser import read_in_situ_excel, pivot_in_situ_for_plot


def apply_graph_theme(fig):
    theme = st.session_state.get('graph_theme', 'light')
    if 'graph_theme' not in st.session_state:
        st.session_state['graph_theme'] = 'light'
    if theme == 'dark':
        dark_layout = dict(
            template='plotly_dark',
            paper_bgcolor="#000000",
            plot_bgcolor="#000000",
            font_color="#e5e7eb",
            title_font=dict(color='#f8fafc', size=20, family='Segoe UI, Arial'),
            xaxis=dict(
                gridcolor="#000000",
                zerolinecolor="#D1CECE",
                linecolor="#919191",
                tickfont=dict(color="#e5e7eb", size=13),
                title_font=dict(color="#f1f5f9", size=15),
                showline=True,
                showgrid=True,
            ),
            yaxis=dict(
                gridcolor="#BBBBBB",
                zerolinecolor="#B9B9B9",
                linecolor='#f8fafc',
                tickfont=dict(color="#e5e7eb", size=13),
                title_font=dict(color="#f1f5f9", size=15),
                showline=True,
                showgrid=True,
            ),
            legend=dict(
                bgcolor="rgba(2, 6, 23, 0.88)",
                bordercolor='#334155',
                font=dict(color='#f8fafc', size=13),
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ),
            colorway=[
                '#2563eb', '#16a34a', '#f59e42', '#e11d48', '#7c3aed',
                '#0ea5e9', '#facc15', '#f472b6', '#a3e635', '#f87171',
            ],
        )
        fig.update_layout(**dark_layout)
    else:
        light_layout = dict(
            template='plotly_white',
            paper_bgcolor='#f8fafc',
            plot_bgcolor='#f8fafc',
            font_color="#000000",
            title_font=dict(color="#181a1b", size=20, family='Segoe UI, Arial'),
            
            xaxis=dict(
                gridcolor='#e5e7eb',
                zerolinecolor='#e5e7eb',
                linecolor="#000000",
                tickfont=dict(color='#181c1f', size=13),
                title_font=dict(color="#0b0c0c", size=15),
                showline=True,
                showgrid=True,
            ),
            yaxis=dict(
                gridcolor='#e5e7eb',
                zerolinecolor='#e5e7eb',
                linecolor="#4B4B4B",
                tickfont=dict(color="#334155", size=12),
                title_font=dict(color="#334155", size=13),
                showline=True,
                showgrid=True,
            ),
            legend=dict(
                bgcolor='rgba(255, 255, 255, 0.92)',
                bordercolor='#cbd5e1',
                font=dict(color="#111827", size=13),
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ),
            colorway=[
                '#2563eb', "#003b16", '#f59e42', '#e11d48', "#2f0675",
                "#4597bd", '#facc15', '#f472b6', '#a3e635', '#f87171',
            ],
        )
        fig.update_layout(**light_layout)
    return fig


def render_graph_theme_toggle(key_suffix: str = "default") -> None:
    if "graph_theme" not in st.session_state:
        st.session_state["graph_theme"] = "light"
    current_theme = st.session_state["graph_theme"]
    if current_theme == "light":
        next_theme = "dark"
        button_label = "Dark Mode"
        button_icon = ":material/dark_mode:"
    else:
        next_theme = "light"
        button_label = "Light Mode"
        button_icon = ":material/light_mode:"
    if st.button(
        button_label,
        icon=button_icon,
        use_container_width=True,
        key=f"global_graph_theme_toggle_{key_suffix}",
    ):
        st.session_state["graph_theme"] = next_theme
        st.rerun()


def render_graph_theme_header_toggle() -> None:
    st.markdown(
        """
        <style>
            .st-key-graph_theme_header_toggle {
                position: fixed;
                top: 0.3rem;
                right: 3.75rem;
                z-index: 999992;
                width: auto;
                max-width: max-content;
                margin: 0;
                padding: 0;
                overflow: visible;
                writing-mode: horizontal-tb;
            }

            .st-key-graph_theme_header_toggle > div {
                width: auto !important;
                max-width: max-content;
            }

            .st-key-graph_theme_header_toggle div[data-testid="stVerticalBlock"] {
                gap: 0;
                width: auto !important;
                max-width: max-content;
            }

            .st-key-graph_theme_header_toggle .stButton {
                margin: 0;
                width: auto;
            }

            .st-key-graph_theme_header_toggle .stButton > button {
                width: auto;
                min-height: 2.5rem;
                padding: 0.45rem 0.9rem;
                border-radius: 14px;
                border: 1px solid #0f766e;
                background: #00352f;
                color: #e5e7eb;
                box-shadow: none;
                font-weight: 600;
                white-space: nowrap;
            }

            .st-key-graph_theme_header_toggle .stButton > button:hover {
                background: #004d44;
                color: #ffffff;
                transform: none;
                box-shadow: none;
            }

            .st-key-graph_theme_header_toggle .stButton > button span[data-testid="stIconMaterial"] {
                font-size: 1rem;
            }

            @media (max-width: 1100px) {
                .st-key-graph_theme_header_toggle {
                    right: 3.25rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.container(key="graph_theme_header_toggle"):
        render_graph_theme_toggle("header")


def render_create_graph_tabs(options: list[str]) -> str:
    st.markdown(
        """
        <style>
            .st-key-create_graph_header_bar {
                position: sticky;
                top: 0;
                z-index: 999989;
                margin: -5.95rem 10.75rem 0.85rem -1rem;
                padding: 0.15rem 0 0;
                background: #000000;
                border-bottom: 1px solid #0f766e;
            }

            .st-key-create_graph_header_bar > div[data-testid="stHorizontalBlock"] {
                align-items: end;
            }

            .st-key-create_graph_header_bar div[data-testid="stRadio"] > label {
                display: none;
            }

            .st-key-create_graph_header_bar div[data-testid="stRadio"] {
                margin: 0;
            }

            .st-key-create_graph_header_bar div[data-testid="stRadio"] > div {
                overflow-x: auto;
                overflow-y: hidden;
                scrollbar-width: none;
            }

            .st-key-create_graph_header_bar div[data-testid="stRadio"] > div::-webkit-scrollbar {
                display: none;
            }

            .st-key-create_graph_header_bar div[role="radiogroup"][aria-label="Visualizacao"] {
                align-items: flex-end;
                background: transparent;
                display: flex;
                flex-wrap: nowrap;
                gap: 0.25rem;
                margin: 0;
                min-width: max-content;
                padding: 0.2rem 0 0;
            }

            .st-key-create_graph_header_bar div[role="radiogroup"][aria-label="Visualizacao"] label {
                background: #00352f;
                border: 1px solid #0f766e;
                border-bottom: none;
                border-radius: 8px 8px 0 0;
                color: #e5e7eb;
                min-height: 2.5rem;
                padding: 0.55rem 1rem;
            }

            .st-key-create_graph_header_bar div[role="radiogroup"][aria-label="Visualizacao"] label:hover {
                background: #004d44;
                color: #ffffff;
            }

            .st-key-create_graph_header_bar div[role="radiogroup"][aria-label="Visualizacao"] label:has(input:checked) {
                background: #b7c0dd;
                border-color: #b7c0dd;
                box-shadow: 0 -2px 10px rgba(183, 192, 221, 0.18);
                color: #001f1c !important;
                font-weight: 700;
                position: relative;
                top: 1px;
            }

            .st-key-create_graph_header_bar div[role="radiogroup"][aria-label="Visualizacao"] label:has(input:checked) p {
                color: #001f1c !important;
                font-weight: 700;
            }

            .st-key-create_graph_header_bar div[role="radiogroup"][aria-label="Visualizacao"] label > div:first-child {
                display: none;
            }

            .st-key-create_graph_header_bar div[role="radiogroup"][aria-label="Visualizacao"] p {
                font-size: 0.92rem;
                margin: 0;
                white-space: nowrap;
            }

            .st-key-create_graph_header_bar .stButton > button {
                min-height: 2.65rem;
                margin-bottom: 0.15rem;
            }

            @media (max-width: 1100px) {
                .st-key-create_graph_header_bar {
                    margin: -5.65rem 8.75rem 0.85rem -1rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    return st.radio(
        "Visualizacao",
        options,
        index=0,
        horizontal=True,
        key="create_graph_subpage",
    )


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


def _looks_like_in_situ_aprofundado_df(df: pd.DataFrame) -> bool:
    """Heuristica para identificar planilha de in situ detalhado, mesmo sem coluna Data explícita."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return False
    poco_col = _get_poco_col(df) or _find_col(df, ["id", "poco"]) or _find_col(df, ["id", "ponto"])
    if not poco_col:
        return False
    def _is_numeric_like(series: pd.Series) -> bool:
        if not isinstance(series, pd.Series):
            return False
        sample = series.dropna().head(30)
        if sample.empty:
            return False
        try:
            parsed = sample.map(parse_ptbr_number)
            return parsed.notna().mean() >= 0.4
        except Exception:
            return False

    numeric_cols = [
        c
        for c in df.columns
        if c not in {"Data", "Ponto", poco_col, "Hora"} and _is_numeric_like(df[c])
    ]
    # aceita se houver pelo menos dois parâmetros numéricos (pH, Temp, ORP, OD, Cond, etc.)
    return len(numeric_cols) >= 2

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
    saida_col = _find_col(df, ["saida"]) or _get_poco_col(df)
    df["saida_key"] = df[saida_col] if saida_col else pd.NA
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


def prep_in_situ_aprofundado(in_situ: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza a planilha de In situ aprofundado para formato longo:
    colunas esperadas: Data, Poco/Ponto, Parametro*, Valor* (ou colunas numericas a serem derretidas).
    """
    df = normalize_dates(in_situ, "Data")
    poco_col = _get_poco_col(df) or ("Ponto" if "Ponto" in df.columns else None)
    if poco_col and poco_col != "Ponto":
        df = df.rename(columns={poco_col: "Ponto"})
    if "Ponto" not in df.columns:
        df["Ponto"] = pd.NA
    df["Ponto"] = df["Ponto"].astype(str).str.strip().str.upper()

    # Hora não é usada no eixo, mas pode existir; mantemos para referência opcional
    if "Hora" in df.columns:
        df["Hora"] = df["Hora"].astype(str).str.strip()

    param_col = None
    value_col = None
    for col in df.columns:
        n = _norm_key(col)
        if param_col is None and (n.startswith("param") or "parametro" in n or "parameter" in n):
            param_col = col
        if value_col is None and any(tok in n for tok in ["valor", "value", "resultado", "medicao", "medida", "dados"]):
            value_col = col

    if param_col and value_col:
        long_df = df.rename(columns={param_col: "Parametro", value_col: "Valor"})
        long_df["Valor"] = long_df["Valor"].map(parse_ptbr_number)
        long_df = long_df[["Data", "Ponto", "Parametro", "Valor"]]
    else:
        def _is_numeric_like(series: pd.Series) -> bool:
            if not isinstance(series, pd.Series):
                return False
            sample = series.dropna().head(30)
            if sample.empty:
                return False
            try:
                parsed = sample.map(parse_ptbr_number)
                return parsed.notna().mean() >= 0.4
            except Exception:
                return False

        numeric_cols = [
            c
            for c in df.columns
            if c not in {"Data", "Ponto", "Hora"} and not _norm_key(c).startswith("obs") and _is_numeric_like(df[c])
        ]
        if not numeric_cols:
            return pd.DataFrame(columns=["Data", "Ponto", "Parametro", "Valor"])
        long_df = df.melt(
            id_vars=["Data", "Ponto"],
            value_vars=numeric_cols,
            var_name="Parametro",
            value_name="Valor",
        )
        long_df["Valor"] = long_df["Valor"].map(parse_ptbr_number)

    long_df["Parametro"] = long_df["Parametro"].astype(str).str.strip()
    long_df = long_df.dropna(subset=["Data"])
    return long_df


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


def _flatten_excel_columns(columns) -> list[str]:
    flat: list[str] = []
    seen: dict[str, int] = {}
    for col in columns:
        if isinstance(col, tuple):
            parts = [str(p).strip() for p in col if p is not None and str(p).strip() and "unnamed" not in str(p).lower()]
            name = parts[-1] if parts else "col"
        else:
            name = str(col).strip()
        name = re.sub(r"\s+", " ", name).strip()
        if not name:
            name = "col"
        idx = seen.get(name, 0)
        seen[name] = idx + 1
        flat.append(f"{name}_{idx}" if idx else name)
    return flat


def _parse_laboratorio_result(value: object):
    if value is None or pd.isna(value):
        return pd.NA
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if s == "":
        return pd.NA
    m = re.search(r"[-+]?\d{1,3}(?:\.\d{3})*(?:,\d+)?|[-+]?\d+(?:[.,]\d+)?", s)
    if not m:
        return pd.NA
    num = m.group(0).replace(".", "").replace(",", ".")
    try:
        return float(num)
    except Exception:
        return pd.NA


def read_laboratorio_excel(file_obj) -> pd.DataFrame:
    if hasattr(file_obj, "getvalue"):
        xls = pd.ExcelFile(io.BytesIO(file_obj.getvalue()))
    else:
        xls = pd.ExcelFile(file_obj)

    target_sheet = None
    for sheet in xls.sheet_names:
        ns = _norm_text(sheet)
        if "resultado" in ns and "ensaio" in ns:
            target_sheet = sheet
            break
    if target_sheet is None:
        for sheet in xls.sheet_names:
            if "sumario" not in _norm_text(sheet):
                target_sheet = sheet
                break
    if target_sheet is None:
        return pd.DataFrame()

    df = xls.parse(target_sheet, header=[0, 1])
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = _flatten_excel_columns(df.columns)
    else:
        df.columns = _flatten_excel_columns([(c,) for c in df.columns])

    ignore_tokens = {
        "processocomercial",
        "situacao",
        "parecerdaamostra",
        "parecerdeamostra",
        "datadapublicacao",
        "datadepublicacao",
    }
    keep_cols = [
        c for c in df.columns
        if not any(tok in _norm_key(c) for tok in ignore_tokens)
    ]
    df = df[keep_cols].copy()

    date_col = _find_col(df, ["data", "coleta"])
    param_col = _find_col(df, ["param"])
    result_col = _find_col(df, ["result"])
    sample_col = _find_col(df, ["n", "amostra"])
    sample_id_col = _find_col(df, ["identificacao", "amostra"]) or _find_col(df, ["amostra"])
    method_col = _find_col(df, ["metodo"])
    unit_col = _find_col(df, ["unidad"])
    cas_col = _find_col(df, ["cas"])

    if not date_col or not param_col or not result_col:
        return pd.DataFrame()

    out = pd.DataFrame(
        {
            "data_coleta": pd.to_datetime(df[date_col], errors="coerce", dayfirst=True),
            "parametro": df[param_col].astype(str).str.strip(),
            "resultado": df[result_col],
            "resultado_num": df[result_col].map(_parse_laboratorio_result),
        }
    )
    if sample_col and sample_col in df.columns:
        out["amostra"] = df[sample_col].astype(str).str.strip()
    else:
        out["amostra"] = pd.NA
    if sample_id_col and sample_id_col in df.columns:
        out["identificacao_amostra"] = df[sample_id_col].astype(str).str.strip()
    else:
        out["identificacao_amostra"] = pd.NA
    if method_col and method_col in df.columns:
        out["metodo"] = df[method_col].astype(str).str.strip()
    else:
        out["metodo"] = pd.NA
    if unit_col and unit_col in df.columns:
        out["unidade"] = df[unit_col].astype(str).str.strip()
    else:
        out["unidade"] = pd.NA
    if cas_col and cas_col in df.columns:
        out["cas"] = df[cas_col].astype(str).str.strip()
    else:
        out["cas"] = pd.NA

    out = out.dropna(subset=["data_coleta"])
    out = out[out["parametro"] != ""]
    return out


# st.title("NA e Volume - Visualizacoes")

df_dict = st.session_state.get("df_dict")
df_by_file = st.session_state.get("df_dict_by_file")
selected_na_file = None
selected_insitu_file = None
in_situ_aprofundado = None

def _copy_keys(src: dict, keys: list[str], dest: dict):
    for k in keys:
        if src and k in src:
            dest[k] = src[k]

if isinstance(df_by_file, dict) and df_by_file:
    na_candidates = [f for f, d in df_by_file.items() if d and any(k in d for k in ("NA Semanal", "Volume Bombeado", "Volume Infiltrado"))]
    insitu_candidates = [f for f, d in df_by_file.items()]
    if na_candidates:
        selected_na_file = na_candidates[0]
    if insitu_candidates:
        default_insitu_idx = 0
        if selected_na_file in insitu_candidates and len(insitu_candidates) > 1:
            default_insitu_idx = 1
        selected_insitu_file = insitu_candidates[default_insitu_idx]

    if selected_na_file or selected_insitu_file:
        combined: dict[str, pd.DataFrame] = {}
        if selected_na_file and selected_na_file in df_by_file:
            src = df_by_file[selected_na_file]
            _copy_keys(src, ["Volume Bombeado", "Volume Infiltrado", "NA Semanal"], combined)
        if selected_insitu_file and selected_insitu_file in df_by_file:
            src = df_by_file[selected_insitu_file]
            _copy_keys(src, ["In Situ (Pontos)", "In Situ", "In Situ (Geral)"], combined)
        # fallback: se não houver df_by_file (ou nada selecionado), mantém o global
        if df_dict and isinstance(df_dict, dict):
            for k, v in df_dict.items():
                if k not in combined:
                    combined[k] = v
        df_dict = combined

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
        except Exception:
            continue

if "In Situ (Geral)" in df_dict:
    try:
        prepared = prep_in_situ(df_dict["In Situ (Geral)"])
        if not prepared.empty:
            in_situ_geral = prepared
    except Exception as e:
        st.warning(f"Falha ao preparar dados de In Situ (Geral): {e}")

subpage_options = ["Operacional", "Visualizacao aprofundada"]
if in_situ_pontos is not None or in_situ_geral is not None:
    subpage_options.append("In situ")
# deixa sempre visível; se faltar dado, mostra orientação dentro da aba
subpage_options.append("In situ aprofundado")
subpage_options.append("Laboratorial")

subpage = st.session_state.get("create_graph_subpage", "Operacional")
if subpage not in subpage_options:
    subpage = "Operacional"
st.session_state["create_graph_subpage"] = subpage

if subpage == "Operacional" and st.session_state.get("create_graph_hide_sidebar"):
    st.markdown(
        """
        <style>
            section[data-testid="stSidebar"] {
                display: none;
            }

            button[kind="header"],
            [data-testid="collapsedControl"] {
                display: none;
            }

            [data-testid="stAppViewContainer"] [data-testid="stMain"] > div:first-child {
                padding-top: 0.75rem !important;
            }

            [data-testid="stAppViewContainer"] [data-testid="stMain"] .stSubheader {
                margin-top: 0 !important;
                padding-top: 0 !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    render_graph_theme_header_toggle()
else:
    theme_spacer, theme_col = st.columns([0.84, 0.16], gap="small")
    with theme_col:
        render_graph_theme_toggle()

if subpage == "Operacional":
    st.subheader("Volume bombeado por poço")

    vb_plot = vb[["Data", "poco_key", "bombeado_vol"]].copy()
    vb_plot["Data"] = pd.to_datetime(vb_plot["Data"], errors="coerce")
    vb_plot["poco_key"] = vb_plot["poco_key"].astype(str).str.strip().str.upper()
    vb_plot = vb_plot.dropna(subset=["Data", "poco_key"])

    if vb_plot.empty:
        st.info("Nao ha dados de volume bombeado para exibir.")
        st.stop()

    non_acc_points = sorted([p for p in vb_plot["poco_key"].unique().tolist() if _norm_text(p) != "acumulado"])
    if not non_acc_points:
        st.info("Nao ha pocos de bombeamento para exibir.")
        st.stop()

    min_date = vb_plot["Data"].min()
    max_date = vb_plot["Data"].max()

    default_points = non_acc_points if len(non_acc_points) <= 12 else non_acc_points[:12]
    selected_points = st.multiselect(
        "Poços",
        non_acc_points,
        default=default_points,
        key="avg_vb_points",
    )
    vb_start, vb_end = period_selector("avg_vb", min_date, max_date, default_preset="1 mês")

    if not selected_points:
        st.info("Selecione ao menos um poco para exibir o grafico.")
        st.stop()

    vb_plot = vb_plot[vb_plot["poco_key"].isin(selected_points)].copy()
    if vb_start:
        vb_plot = vb_plot[vb_plot["Data"] >= pd.to_datetime(vb_start)]
    if vb_end:
        vb_plot = vb_plot[vb_plot["Data"] <= pd.to_datetime(vb_end)]

    if vb_plot.empty:
        st.info("Sem dados de volume bombeado apos aplicar os filtros.")
        st.stop()

    bars_daily = (
        vb_plot.groupby(["Data", "poco_key"], as_index=False)["bombeado_vol"]
        .sum()
        .pivot_table(index="Data", columns="poco_key", values="bombeado_vol", aggfunc="sum")
        .sort_index()
    )
    bars_daily = bars_daily.reindex(columns=selected_points)
    bars_daily = bars_daily.add_prefix("vb__")

    # Serie de acumulado: usa linha "Acumulado" quando existir; senao calcula acumulado do total diario.
    vb_acc = vb[["Data", "poco_key", "bombeado_vol"]].copy()
    vb_acc["Data"] = pd.to_datetime(vb_acc["Data"], errors="coerce")
    vb_acc["poco_key"] = vb_acc["poco_key"].astype(str).str.strip().str.upper()
    vb_acc = vb_acc.dropna(subset=["Data", "poco_key"])
    if vb_start:
        vb_acc = vb_acc[vb_acc["Data"] >= pd.to_datetime(vb_start)]
    if vb_end:
        vb_acc = vb_acc[vb_acc["Data"] <= pd.to_datetime(vb_end)]

    acc_line = (
        vb_acc[vb_acc["poco_key"].map(_norm_text) == "acumulado"]
        .groupby("Data", as_index=False)["bombeado_vol"]
        .sum()
        .sort_values("Data")
    )
    if acc_line.empty:
        acc_line = bars_daily.sum(axis=1).cumsum().rename("volume_acumulado").reset_index()
    else:
        acc_line = acc_line.rename(columns={"bombeado_vol": "volume_acumulado"})

    chart_df = bars_daily.reset_index().merge(acc_line, on="Data", how="outer").sort_values("Data")
    chart_df = chart_df.dropna(subset=["Data"])

    vb_palette = [
        "#f59f00", "#f76707", "#f03e3e", "#e03131", "#c92a2a",
        "#b02525", "#862e9c", "#5f3dc4", "#1f77b4", "#0b7285",
        "#2b8a3e", "#495057", "#7950f2", "#1971c2",
    ]
    series: list[SeriesSpec] = []
    for i, point in enumerate(selected_points):
        col_vb = f"vb__{point}"
        if col_vb not in chart_df.columns:
            continue
        series.append(
            SeriesSpec(
                y=col_vb,
                label=point,
                kind="bar",
                axis="y",
                color=vb_palette[i % len(vb_palette)],
            )
        )

    series.append(
        SeriesSpec(
            y="volume_acumulado",
            label="Volume Acumulado",
            kind="line",
            axis="y2",
            color="#f97316",
            line_dash="dash",
            connect_gaps=True,
        )
    )

    fig, _ = build_time_chart_plotly(
        chart_df,
        x="Data",
        series=series,
        title="Volume bombeado por poço e volume acumulado",
        show_range_slider=False,
        limit_points=200000,
        return_insights=False,
        height=520,
    )
    fig.update_layout(barmode="group", bargap=0.15, bargroupgap=0.05)
    fig.update_xaxes(
        tickformat="%d/%m/%Y",
        tickangle=-35,
        dtick="D1",
        tickmode="auto",
        nticks=20,
    )
    fig.update_yaxes(title_text="Volume Bombeado (m\u00b3)", secondary_y=False)
    fig.update_yaxes(title_text="Volume Acumulado (m\u00b3)", secondary_y=True)
    fig.for_each_trace(
        lambda tr: tr.update(line=dict(width=2.5, dash="dash"))
        if getattr(tr, "name", "") == "Volume Acumulado"
        else None
    )

    

    # Garante boa legibilidade das legendas dos eixos Y nos dois temas.
    fig.update_yaxes(
        title_standoff=12,
        secondary_y=False,
    )
    fig.update_yaxes(
        title_standoff=12,
        secondary_y=True,
    )

    # Posiciona a legenda abaixo do grafico sem sobrepor os rótulos do eixo X.
    num_legend_items = len(selected_points) + 1  # +1 para Volume Acumulado
    legend_rows = max(1, (num_legend_items + 7) // 8)  # ~8 itens por linha
    bottom_margin = 142 + legend_rows * 24
    legend_y = -0.32 - max(0, legend_rows - 1) * 0.10

    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="top",
            y=legend_y,
            xanchor="left",
            x=0,
            borderwidth=1,
            itemwidth=72,
        ),
        margin=dict(b=bottom_margin, t=60, l=60, r=60),
    )
    fig.update_xaxes(automargin=True)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Volume infiltrado por saída")

    vi_plot = vi[["Data", "saida_key", "infiltrado_vol"]].copy()
    vi_plot["Data"] = pd.to_datetime(vi_plot["Data"], errors="coerce")
    vi_plot["saida_key"] = vi_plot["saida_key"].astype(str).str.strip().str.upper()
    vi_plot = vi_plot.dropna(subset=["Data", "saida_key"])

    if vi_plot.empty:
        st.info("Nao ha dados de volume infiltrado para exibir.")
    else:
        non_acc_outputs = sorted([s for s in vi_plot["saida_key"].unique().tolist() if _norm_text(s) != "acumulado"])
        if not non_acc_outputs:
            st.info("Nao ha saídas de infiltracao para exibir.")
        else:
            min_vi_date = vi_plot["Data"].min()
            max_vi_date = vi_plot["Data"].max()

            default_outputs = non_acc_outputs if len(non_acc_outputs) <= 12 else non_acc_outputs[:12]
            selected_outputs = st.multiselect(
                "Saídas",
                non_acc_outputs,
                default=default_outputs,
                key="avg_vi_outputs",
            )
            vi_start, vi_end = period_selector("avg_vi", min_vi_date, max_vi_date, default_preset="1 mês")

            if not selected_outputs:
                st.info("Selecione ao menos uma saída para exibir o grafico.")
            else:
                vi_plot = vi_plot[vi_plot["saida_key"].isin(selected_outputs)].copy()
                if vi_start:
                    vi_plot = vi_plot[vi_plot["Data"] >= pd.to_datetime(vi_start)]
                if vi_end:
                    vi_plot = vi_plot[vi_plot["Data"] <= pd.to_datetime(vi_end)]

                if vi_plot.empty:
                    st.info("Sem dados de volume infiltrado apos aplicar os filtros.")
                else:
                    vi_bars_daily = (
                        vi_plot.groupby(["Data", "saida_key"], as_index=False)["infiltrado_vol"]
                        .sum()
                        .pivot_table(index="Data", columns="saida_key", values="infiltrado_vol", aggfunc="sum")
                        .sort_index()
                    )
                    vi_bars_daily = vi_bars_daily.reindex(columns=selected_outputs)
                    vi_bars_daily = vi_bars_daily.add_prefix("vi__")

                    vi_acc = vi[["Data", "saida_key", "infiltrado_vol"]].copy()
                    vi_acc["Data"] = pd.to_datetime(vi_acc["Data"], errors="coerce")
                    vi_acc["saida_key"] = vi_acc["saida_key"].astype(str).str.strip().str.upper()
                    vi_acc = vi_acc.dropna(subset=["Data", "saida_key"])
                    if vi_start:
                        vi_acc = vi_acc[vi_acc["Data"] >= pd.to_datetime(vi_start)]
                    if vi_end:
                        vi_acc = vi_acc[vi_acc["Data"] <= pd.to_datetime(vi_end)]

                    vi_acc_line = (
                        vi_acc[vi_acc["saida_key"].map(_norm_text) == "acumulado"]
                        .groupby("Data", as_index=False)["infiltrado_vol"]
                        .sum()
                        .sort_values("Data")
                    )
                    if vi_acc_line.empty:
                        vi_acc_line = vi_bars_daily.sum(axis=1).cumsum().rename("infiltrado_acumulado").reset_index()
                    else:
                        vi_acc_line = vi_acc_line.rename(columns={"infiltrado_vol": "infiltrado_acumulado"})

                    vi_chart_df = (
                        vi_bars_daily.reset_index()
                        .merge(vi_acc_line, on="Data", how="outer")
                        .sort_values("Data")
                        .dropna(subset=["Data"])
                    )

                    vi_palette = [
                        "#6a2c91", "#1f6f2f", "#1f8ac0", "#9547b8", "#2f9e44",
                        "#0c8599", "#862e9c", "#2b8a3e", "#1971c2", "#7048e8",
                    ]
                    vi_series: list[SeriesSpec] = []
                    for i, output in enumerate(selected_outputs):
                        col_vi = f"vi__{output}"
                        if col_vi not in vi_chart_df.columns:
                            continue
                        vi_series.append(
                            SeriesSpec(
                                y=col_vi,
                                label=output,
                                kind="bar",
                                axis="y",
                                color=vi_palette[i % len(vi_palette)],
                            )
                        )

                    vi_series.append(
                        SeriesSpec(
                            y="infiltrado_acumulado",
                            label="Acumulado",
                            kind="line",
                            axis="y2",
                            color="#c2410c",
                            line_dash="dash",
                            connect_gaps=True,
                        )
                    )

                    fig_vi, _ = build_time_chart_plotly(
                        vi_chart_df,
                        x="Data",
                        series=vi_series,
                        title="Volume infiltrado por saída e acumulado",
                        show_range_slider=False,
                        limit_points=200000,
                        return_insights=False,
                        height=520,
                    )
                    fig_vi.update_layout(barmode="group", bargap=0.15, bargroupgap=0.05)
                    fig_vi.update_xaxes(
                        tickformat="%d/%m/%Y",
                        tickangle=-35,
                        dtick="D1",
                        tickmode="auto",
                        nticks=20,
                    )
                    fig_vi.update_yaxes(title_text="Volume Infiltrado (m\u00b3)", secondary_y=False)
                    fig_vi.update_yaxes(title_text="Volume Acumulado (m\u00b3)", secondary_y=True)
                    fig_vi.for_each_trace(
                        lambda tr: tr.update(line=dict(width=2.5, dash="dash"))
                        if getattr(tr, "name", "") == "Acumulado"
                        else None
                    )

                   

                    fig_vi.update_yaxes(
                        title_standoff=12,
                        secondary_y=False,
                    )
                    fig_vi.update_yaxes(
                        title_standoff=12,
                        secondary_y=True,
                    )

                    num_vi_legend_items = len(selected_outputs) + 1
                    vi_legend_rows = max(1, (num_vi_legend_items + 7) // 8)
                    vi_bottom_margin = 142 + vi_legend_rows * 24
                    vi_legend_y = -0.32 - max(0, vi_legend_rows - 1) * 0.10
                
                    fig_vi.update_layout(
                        legend=dict(
                            orientation="h",
                            yanchor="top",
                            y=vi_legend_y,
                            xanchor="left",
                            x=0,
                            borderwidth=1,
                            itemwidth=72,
                        ),
                        margin=dict(b=vi_bottom_margin, t=60, l=60, r=60),
                    )
                    fig_vi.update_xaxes(automargin=True)

                    st.plotly_chart(fig_vi, use_container_width=True)

    st.markdown("---")
    st.subheader("Volume acumulado: bombeado vs infiltrado")

    vb_acc_src = vb[["Data", "poco_key", "bombeado_vol"]].copy()
    vb_acc_src["Data"] = pd.to_datetime(vb_acc_src["Data"], errors="coerce")
    vb_acc_src["poco_key"] = vb_acc_src["poco_key"].astype(str).str.strip().str.upper()
    vb_acc_src = vb_acc_src.dropna(subset=["Data", "poco_key"])

    vi_acc_src = vi[["Data", "saida_key", "infiltrado_vol"]].copy()
    vi_acc_src["Data"] = pd.to_datetime(vi_acc_src["Data"], errors="coerce")
    vi_acc_src["saida_key"] = vi_acc_src["saida_key"].astype(str).str.strip().str.upper()
    vi_acc_src = vi_acc_src.dropna(subset=["Data", "saida_key"])

    all_dates = pd.concat(
        [vb_acc_src["Data"], vi_acc_src["Data"]],
        ignore_index=True,
    ).dropna()

    if all_dates.empty:
        st.info("Nao ha dados para o grafico acumulado comparativo.")
    else:
        max_acc_date = all_dates.max()
        min_acc_date = all_dates.min()

        acc_start, acc_end = period_selector("avg_acc", min_acc_date, max_acc_date, default_preset="1 mês")

        if acc_start:
            vb_acc_src = vb_acc_src[vb_acc_src["Data"] >= pd.to_datetime(acc_start)]
            vi_acc_src = vi_acc_src[vi_acc_src["Data"] >= pd.to_datetime(acc_start)]
        if acc_end:
            vb_acc_src = vb_acc_src[vb_acc_src["Data"] <= pd.to_datetime(acc_end)]
            vi_acc_src = vi_acc_src[vi_acc_src["Data"] <= pd.to_datetime(acc_end)]

        vb_acc_line = (
            vb_acc_src[vb_acc_src["poco_key"].map(_norm_text) == "acumulado"]
            .groupby("Data", as_index=False)["bombeado_vol"]
            .sum()
            .sort_values("Data")
        )
        if vb_acc_line.empty:
            vb_daily = (
                vb_acc_src[vb_acc_src["poco_key"].map(_norm_text) != "acumulado"]
                .groupby("Data", as_index=False)["bombeado_vol"]
                .sum()
                .sort_values("Data")
            )
            vb_acc_line = vb_daily.assign(bombeado_acumulado=vb_daily["bombeado_vol"].cumsum())[["Data", "bombeado_acumulado"]]
        else:
            vb_acc_line = vb_acc_line.rename(columns={"bombeado_vol": "bombeado_acumulado"})

        vi_acc_line = (
            vi_acc_src[vi_acc_src["saida_key"].map(_norm_text) == "acumulado"]
            .groupby("Data", as_index=False)["infiltrado_vol"]
            .sum()
            .sort_values("Data")
        )
        if vi_acc_line.empty:
            vi_daily = (
                vi_acc_src[vi_acc_src["saida_key"].map(_norm_text) != "acumulado"]
                .groupby("Data", as_index=False)["infiltrado_vol"]
                .sum()
                .sort_values("Data")
            )
            vi_acc_line = vi_daily.assign(infiltrado_acumulado=vi_daily["infiltrado_vol"].cumsum())[["Data", "infiltrado_acumulado"]]
        else:
            vi_acc_line = vi_acc_line.rename(columns={"infiltrado_vol": "infiltrado_acumulado"})

        acc_chart_df = (
            vb_acc_line.merge(vi_acc_line, on="Data", how="outer")
            .sort_values("Data")
            .dropna(subset=["Data"])
        )

        if acc_chart_df.empty:
            st.info("Sem dados para o período selecionado.")
        else:
            acc_series = [
                SeriesSpec(
                    y="bombeado_acumulado",
                    label="Volume Bombeado Acumulado",
                    kind="line",
                    marker="circle",
                    color="#f97316",
                    line_dash="dash",
                    connect_gaps=True,
                ),
                SeriesSpec(
                    y="infiltrado_acumulado",
                    label="Volume Infiltrado Acumulado",
                    kind="line",
                    marker="diamond",
                    color="#2563eb",
                    line_dash="dash",
                    connect_gaps=True,
                ),
            ]

    fig, _ = build_time_chart_plotly(
        acc_chart_df,
        x="Data",
        series=acc_series,
        title="Comparativo de volumes acumulados",
        show_range_slider=False,
        limit_points=200000,
        return_insights=False,
        height=460,
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
elif subpage == "In situ aprofundado":
    st.subheader("In situ aprofundado")

    uploaded_files = st.session_state.get("uploaded_files", [])
    file_obj = None
    if uploaded_files and selected_insitu_file:
        for f in uploaded_files:
            if getattr(f, "name", "") == selected_insitu_file:
                file_obj = f
                break

    if file_obj is None:
        st.info("Nao foi encontrado um arquivo de In situ para esta visualizacao.")
        st.stop()

    try:
        df_long = read_in_situ_excel(file_obj)
    except Exception as e:
        st.error(f"Falha ao ler o arquivo selecionado: {e}")
        st.stop()

    if df_long.empty:
        st.info("Arquivo de In situ aprofundado vazio ou não reconhecido.")
        st.stop()

    # remove linhas agregadas/estatísticas eventualmente não filtradas
    df_long = df_long[
        ~df_long["poco_id"].str.contains(r"max|máx|min|média|media", case=False, regex=True)
    ]

    parametros = sorted(df_long["parametro"].dropna().unique().tolist())
    if not parametros:
        st.info("Nenhum parâmetro identificado no arquivo de In situ.")
        st.stop()

    date_min = df_long["DataHora"].min()
    date_max = df_long["DataHora"].max()

    ctrl_cols = st.columns([2, 2, 2], gap="small")
    param1 = ctrl_cols[0].selectbox("Parâmetro principal", parametros)
    param2_options = ["(nenhum)"] + [p for p in parametros if p != param1]
    param2 = ctrl_cols[1].selectbox("Segundo parâmetro (opcional)", param2_options, index=0)
    pontos = sorted(df_long["poco_id"].dropna().unique().tolist())
    default_pontos: list[str] = [pontos[0]] if pontos else []
    pontos_sel = ctrl_cols[2].multiselect("Poços", pontos, default=default_pontos)
    aps_start, aps_end = period_selector("aps_insitu", date_min, date_max, default_preset="12 meses")

    data_filt = df_long.copy()
    if pontos_sel:
        data_filt = data_filt[data_filt["poco_id"].isin(pontos_sel)]
    if aps_start:
        data_filt = data_filt[data_filt["DataHora"] >= pd.to_datetime(aps_start)]
    if aps_end:
        data_filt = data_filt[data_filt["DataHora"] <= pd.to_datetime(aps_end)]

    df_plot1 = pivot_in_situ_for_plot(data_filt, param1)
    if df_plot1.empty:
        st.info("Sem valores numéricos para o parâmetro selecionado.")
        st.stop()

    series: list[SeriesSpec] = []
    palette_main = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#17becf",
    ]
    plot1_cols = [c for c in df_plot1.columns if c != "DataHora"]
    df_plot = df_plot1.rename(columns={col: f"{col}__p1" for col in plot1_cols})
    for idx, col in enumerate(plot1_cols):
        series.append(
            SeriesSpec(
                y=f"{col}__p1",
                label=f"{col} - {param1}",
                kind="bar",
                color=palette_main[idx % len(palette_main)],
                axis="y",
            )
        )

    if param2 != "(nenhum)":
        df_plot2 = pivot_in_situ_for_plot(data_filt, param2)
        if not df_plot2.empty:
            palette_second = [
                "#d94801",
                "#e6550d",
                "#f16913",
                "#fd8d3c",
                "#9e9ac8",
                "#756bb1",
                "#dd3497",
                "#c51b8a",
            ]
            plot2_cols = [c for c in df_plot2.columns if c != "DataHora"]
            df_plot2 = df_plot2.rename(columns={col: f"{col}__p2" for col in plot2_cols})
            df_plot = df_plot.merge(df_plot2, on="DataHora", how="outer")
            for idx, col in enumerate(plot2_cols):
                series.append(
                    SeriesSpec(
                        y=f"{col}__p2",
                        label=f"{col} - {param2}",
                        kind="line",
                        marker="square",
                        line_dash="dash",
                        marker_line_color="#111111",
                        marker_line_width=1.0,
                        color=palette_second[idx % len(palette_second)],
                        axis="y2",
                    )
                )

    fig_ap, _ = build_time_chart_plotly(
        df_plot,
        x="DataHora",
        series=series,
        title="In situ aprofundado",
        show_range_slider=False,
        limit_points=200000,
    )
    fig_ap.update_layout(barmode="group", bargap=0.1, bargroupgap=0.05)
    fig_ap.update_xaxes(title_text="Data / Hora")
    fig_ap.update_yaxes(title_text=param1, secondary_y=False)
    if any(s.axis == "y2" for s in series):
        fig_ap.update_yaxes(title_text=param2, secondary_y=True)
    
    st.plotly_chart(fig_ap, use_container_width=True)

    with st.expander("Tabela detalhada", expanded=False):
        st.dataframe(
            data_filt[["DataHora", "poco_id", "parametro", "valor", "status", "sheet"]]
            .sort_values("DataHora"),
            use_container_width=True,
            hide_index=True,
        )

elif subpage == "Laboratorial":
    st.subheader("Laboratorial")

    uploaded_files = st.session_state.get("uploaded_files", [])
    if not uploaded_files:
        st.info("Carregue ao menos um Excel na página inicial para usar a aba laboratorial.")
        st.stop()

    file_names = [getattr(f, "name", f"arquivo_{i+1}") for i, f in enumerate(uploaded_files)]
    saved_lab_file = st.session_state.get("lab_file_select")
    if saved_lab_file in file_names:
        default_lab_file = saved_lab_file
    else:
        excluded_files = {selected_na_file, selected_insitu_file}
        preferred_files = [name for name in file_names if name not in excluded_files]
        if preferred_files:
            default_lab_file = preferred_files[0]
        elif selected_insitu_file and selected_insitu_file in file_names:
            default_lab_file = selected_insitu_file
        elif selected_na_file and selected_na_file in file_names:
            default_lab_file = selected_na_file
        else:
            default_lab_file = file_names[0]
    default_idx = file_names.index(default_lab_file)
    selected_lab_file = st.selectbox(
        "Arquivo laboratorial",
        file_names,
        index=default_idx,
        key="lab_file_select",
    )
    file_obj = next((f for f in uploaded_files if getattr(f, "name", "") == selected_lab_file), None)
    if file_obj is None:
        st.info("Arquivo laboratorial não encontrado na sessão.")
        st.stop()

    try:
        df_lab = read_laboratorio_excel(file_obj)
    except Exception as e:
        st.error(f"Falha ao ler a planilha laboratorial: {e}")
        st.stop()

    if df_lab.empty:
        st.info("Não foi possível identificar dados laboratoriais no arquivo selecionado.")
        st.stop()

    # ignora registros sem unidade útil
    if "unidade" in df_lab.columns:
        df_lab = df_lab[df_lab["unidade"].fillna("").astype(str).str.strip().str.lower() != "no unit"]
        if df_lab.empty:
            st.info("Sem dados laboratoriais após remover registros com unidade 'No Unit'.")
            st.stop()

    params = sorted(df_lab["parametro"].dropna().unique().tolist())
    if not params:
        st.info("Nenhum parâmetro laboratorial encontrado.")
        st.stop()

    date_min = df_lab["data_coleta"].min()
    date_max = df_lab["data_coleta"].max()

    sample_ids = sorted(df_lab["identificacao_amostra"].dropna().unique().tolist())

    lab_ctrl_cols = st.columns([1, 1], gap="small")
    default_params = params[: min(6, len(params))]
    if "lab_selected_params" not in st.session_state:
        st.session_state["lab_selected_params"] = default_params
    else:
        st.session_state["lab_selected_params"] = [
            p for p in st.session_state["lab_selected_params"] if p in params
        ]
        if not st.session_state["lab_selected_params"]:
            st.session_state["lab_selected_params"] = default_params
    if st.session_state.get("lab_select_all_trigger", False):
        st.session_state["lab_selected_params"] = params.copy()
        st.session_state["lab_select_all_trigger"] = False
    selected_params = lab_ctrl_cols[0].multiselect(
        "Parâmetros",
        params,
        key="lab_selected_params",
    )
    if lab_ctrl_cols[0].button("Selecionar todos parâmetros", use_container_width=True):
        st.session_state["lab_select_all_trigger"] = True
        st.rerun()
    selected_samples = lab_ctrl_cols[1].multiselect(
        "Identificação da amostra",
        sample_ids,
        default=sample_ids[:3] if sample_ids else [],
    )
    lab_start, lab_end = period_selector("lab", date_min, date_max, default_preset="Tudo")

    data_filt = df_lab.copy()
    if selected_params:
        data_filt = data_filt[data_filt["parametro"].isin(selected_params)]
    if selected_samples:
        data_filt = data_filt[data_filt["identificacao_amostra"].isin(selected_samples)]
    if lab_start:
        data_filt = data_filt[data_filt["data_coleta"] >= pd.to_datetime(lab_start)]
    if lab_end:
        data_filt = data_filt[data_filt["data_coleta"] <= pd.to_datetime(lab_end)]

    if data_filt.empty:
        st.info("Sem dados após os filtros selecionados.")
        st.stop()

    chart_df = (
        data_filt.pivot_table(
            index="data_coleta",
            columns="parametro",
            values="resultado_num",
            aggfunc="mean",
        )
        .reset_index()
        .sort_values("data_coleta")
    )
    value_cols = [c for c in chart_df.columns if c != "data_coleta"]
    if not value_cols:
        st.info("Nenhum resultado numérico disponível para os parâmetros selecionados.")
        st.stop()


        palette = [
            "#60a5fa",
            "#34d399",
            "#f59e0b",
            "#f87171",
            "#a78bfa",
            "#22d3ee",
            "#f472b6",
            "#bef264",
        ]
        marker_border = "#0f172a"
    else:
        palette = [
            "#1d4ed8",
            "#047857",
            "#d97706",
            "#dc2626",
            "#7c3aed",
            "#0e7490",
            "#be185d",
            "#65a30d",
        ]
        marker_border = "#ffffff"
    series = [
        SeriesSpec(
            y=col,
            label=col,
            kind="line",
            marker="circle",
            color=palette[i % len(palette)],
            marker_line_color=marker_border,
            marker_line_width=1.2,
            connect_gaps=True,
        )
        for i, col in enumerate(value_cols)
    ]

    fig_lab, _ = build_time_chart_plotly(
        chart_df,
        x="data_coleta",
        series=series,
        title="Resultados laboratoriais",
        show_range_slider=False,
        limit_points=200000,
    )
    fig_lab.update_xaxes(title_text="Data de Coleta")
    fig_lab.update_yaxes(title_text="Resultado")
   
    fig_lab.update_layout(
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=11),
        ),
    )
    st.plotly_chart(fig_lab, use_container_width=True)

    with st.expander("Tabela laboratorial", expanded=False):
        cols_view = [
            "data_coleta",
            "amostra",
            "identificacao_amostra",
            "metodo",
            "parametro",
            "resultado",
            "resultado_num",
            "unidade",
            "cas",
        ]
        cols_view = [c for c in cols_view if c in data_filt.columns]
        st.dataframe(
            data_filt[cols_view].sort_values("data_coleta"),
            use_container_width=True,
            hide_index=True,
        )

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

    if mode == "Por ponto":
        points = sorted(data_source["Ponto"].dropna().unique().tolist())
        if not points:
            st.info("Nenhum ponto encontrado na aba In Situ.")
            st.stop()

        is_ctrl_cols = st.columns([1, 1], gap="small")
        selected_param = is_ctrl_cols[0].selectbox("Parametro", params)
        default_points = points if len(points) <= 8 else points[:8]
        selected_points = is_ctrl_cols[1].multiselect(
            "Pontos",
            points,
            default=default_points,
        )
        is_start, is_end = period_selector("is_pontos", date_min, date_max, default_preset="Tudo")

        if not selected_points:
            st.info("Selecione ao menos um ponto para exibir o grafico.")
            st.stop()

        data = data_source[data_source["Ponto"].isin(selected_points)].copy()
        data["Data"] = pd.to_datetime(data["Data"], errors="coerce")
        data = data.dropna(subset=["Data"])
        if is_start:
            data = data[data["Data"] >= pd.to_datetime(is_start)]
        if is_end:
            data = data[data["Data"] <= pd.to_datetime(is_end)]

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
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
            "#9467bd", "#8c564b", "#e377c2", "#17becf",
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

        fig3, _ = build_time_chart_plotly(
            chart_df,
            x="Data",
            series=series,
            title=f"{selected_param} - In situ",
            show_range_slider=False,
            limit_points=200000,
        )
        fig3.update_yaxes(title_text=selected_param, secondary_y=False)
        apply_graph_theme(fig3)
        st.plotly_chart(fig3, use_container_width=True)

    else:
        if "Ponto" in data_source.columns:
            # Multiselect: cada parâmetro gera duas linhas (Entrada e Saída)
            selected_params_list = st.multiselect(
                "Parâmetros",
                params,
                default=[params[0]] if params else [],
                key="is_geral_params_multi",
            )
            is_start, is_end = period_selector("is_geral", date_min, date_max, default_preset="Tudo")

            if not selected_params_list:
                st.info("Selecione ao menos um parâmetro para exibir o gráfico.")
                st.stop()

            data = data_source.copy()
            data["Data"] = pd.to_datetime(data["Data"], errors="coerce")
            data = data.dropna(subset=["Data"])
            if is_start:
                data = data[data["Data"] >= pd.to_datetime(is_start)]
            if is_end:
                data = data[data["Data"] <= pd.to_datetime(is_end)]

            chart_frames = []
            param_col_map: dict[str, list[str]] = {}
            _SEP = " – "

            for param in selected_params_list:
                if param not in data.columns:
                    continue
                pivot = (
                    data.pivot_table(index="Data", columns="Ponto", values=param, aggfunc="mean")
                    .sort_index()
                )
                pivot.columns = [f"{param}{_SEP}{col}" for col in pivot.columns]
                param_col_map[param] = pivot.columns.tolist()
                chart_frames.append(pivot)

            if not chart_frames:
                st.info("Nao ha valores para os parâmetros selecionados.")
                st.stop()

            chart_df = pd.concat(chart_frames, axis=1).reset_index().sort_values("Data")

            # Uma cor por parâmetro; linha sólida/círculo para Entrada, tracejada/quadrado para Saída
            param_colors = [
                "#1f77b4", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#17becf", "#bcbd22",
            ]

            series = []
            for p_idx, param in enumerate(selected_params_list):
                color = param_colors[p_idx % len(param_colors)]
                for col in param_col_map.get(param, []):
                    ponto_part = col[len(param) + len(_SEP):]
                    is_entrada = "entr" in ponto_part.lower()
                    series.append(
                        SeriesSpec(
                            y=col,
                            label=col,
                            kind="line",
                            marker="circle" if is_entrada else "square",
                            line_dash="solid" if is_entrada else "dash",
                            color=color,
                            connect_gaps=True,
                        )
                    )

            if not series:
                st.info("Nenhuma série disponível para os parâmetros selecionados.")
                st.stop()

            fig3, _ = build_time_chart_plotly(
                chart_df,
                x="Data",
                series=series,
                title="In situ – Entrada vs Saída",
                show_range_slider=False,
                limit_points=200000,
            )
            apply_graph_theme(fig3)
            st.plotly_chart(fig3, use_container_width=True)

        else:
            selected_param = st.selectbox("Parametro", params)
            is_start, is_end = period_selector("is_geral", date_min, date_max, default_preset="Tudo")

            data = data_source.copy()
            data["Data"] = pd.to_datetime(data["Data"], errors="coerce")
            data = data.dropna(subset=["Data"])
            if is_start:
                data = data[data["Data"] >= pd.to_datetime(is_start)]
            if is_end:
                data = data[data["Data"] <= pd.to_datetime(is_end)]

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
            apply_graph_theme(fig3)
            st.plotly_chart(fig3, use_container_width=True)
