from __future__ import annotations

import streamlit as st


def ensure_theme_state() -> None:
    if "graph_theme" not in st.session_state:
        st.session_state["graph_theme"] = "light"


def get_current_theme() -> str:
    ensure_theme_state()
    return st.session_state.get("graph_theme", "light")


def apply_plotly_theme(fig):
    ensure_theme_state()
    theme = get_current_theme()
    if theme == "dark":
        axis_style = dict(
            gridcolor="#3f3f46",
            zerolinecolor="#71717a",
            linecolor="#e5e7eb",
            tickfont=dict(color="#e5e7eb", size=13),
            title_font=dict(color="#f8fafc", size=15),
            showline=True,
            showgrid=True,
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#000000",
            plot_bgcolor="#000000",
            font_color="#e5e7eb",
            title_font=dict(color="#f8fafc", size=20, family="Segoe UI, Arial"),
            legend=dict(
                bgcolor="rgba(2, 6, 23, 0.88)",
                bordercolor="#334155",
                font=dict(color="#f8fafc", size=13),
            ),
            colorway=[
                "#2563eb",
                "#16a34a",
                "#f59e42",
                "#e11d48",
                "#7c3aed",
                "#0ea5e9",
                "#facc15",
                "#f472b6",
                "#a3e635",
                "#f87171",
            ],
        )
    else:
        axis_style = dict(
            gridcolor="#e5e7eb",
            zerolinecolor="#e5e7eb",
            linecolor="#334155",
            tickfont=dict(color="#181c1f", size=13),
            title_font=dict(color="#111827", size=15),
            showline=True,
            showgrid=True,
        )
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="#f8fafc",
            plot_bgcolor="#f8fafc",
            font_color="#000000",
            title_font=dict(color="#181a1b", size=20, family="Segoe UI, Arial"),
            legend=dict(
                bgcolor="rgba(255, 255, 255, 0.92)",
                bordercolor="#cbd5e1",
                font=dict(color="#111827", size=13),
            ),
            colorway=[
                "#2563eb",
                "#003b16",
                "#f59e42",
                "#e11d48",
                "#2f0675",
                "#4597bd",
                "#facc15",
                "#f472b6",
                "#a3e635",
                "#f87171",
            ],
        )
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    fig.update_layout(
        yaxis2=dict(
            gridcolor=axis_style["gridcolor"],
            zerolinecolor=axis_style["zerolinecolor"],
            linecolor=axis_style["linecolor"],
            tickfont=axis_style["tickfont"],
            title=dict(font=axis_style["title_font"]),
            showline=axis_style["showline"],
            showgrid=axis_style["showgrid"],
        )
    )
    return fig


def apply_global_theme_styles() -> None:
    ensure_theme_state()
    theme = get_current_theme()
    if theme == "dark":
        palette = {
            "page_bg": "#000000",
            "surface": "#101820",
            "surface_alt": "#0d141b",
            "text": "#f8fafc",
            "muted": "#cbd5e1",
            "border": "#334155",
            "accent": "#004d44",
            "accent_border": "#0f766e",
            "accent_hover": "#006b5c",
            "header_bg": "#000000",
            "header_text": "#f8fafc",
        }
    else:
        palette = {
            "page_bg": "#f3f7fb",
            "surface": "#ffffff",
            "surface_alt": "#eef3f8",
            "text": "#0f172a",
            "muted": "#334155",
            "border": "#cbd5e1",
            "accent": "#0f766e",
            "accent_border": "#0f766e",
            "accent_hover": "#115e59",
            "header_bg": "#0f766e",
            "header_text": "#ffffff",
        }

    st.markdown(
        f"""
        <style>
            :root {{
                --global-page-bg: {palette["page_bg"]};
                --global-surface: {palette["surface"]};
                --global-surface-alt: {palette["surface_alt"]};
                --global-text: {palette["text"]};
                --global-muted: {palette["muted"]};
                --global-border: {palette["border"]};
                --global-accent: {palette["accent"]};
                --global-accent-border: {palette["accent_border"]};
                --global-accent-hover: {palette["accent_hover"]};
                --global-header-bg: {palette["header_bg"]};
                --global-header-text: {palette["header_text"]};
            }}

            [data-testid="stAppViewContainer"],
            [data-testid="stMain"],
            [data-testid="stMainBlockContainer"] {{
                background: var(--global-page-bg) !important;
                color: var(--global-text) !important;
            }}

            [data-testid="stHeader"] {{
                background: var(--global-header-bg) !important;
                border-bottom: 1px solid var(--global-accent-border) !important;
            }}

            [data-testid="stHeader"] * {{
                color: var(--global-header-text) !important;
            }}

            [data-testid="stHeader"] button[kind="header"] {{
                color: var(--global-header-text) !important;
                border: 1px solid transparent !important;
                border-radius: 8px !important;
            }}

            [data-testid="stHeader"] button[kind="header"]:hover {{
                background: rgba(255, 255, 255, 0.16) !important;
                border-color: rgba(255, 255, 255, 0.35) !important;
                color: var(--global-header-text) !important;
            }}

            [data-testid="stToolbar"] {{
                background: transparent !important;
            }}

            h1, h2, h3, h4, h5, h6, p, li, label, span, div {{
                color: var(--global-text);
            }}

            [data-testid="stSidebar"] {{
                background-color: var(--global-accent) !important;
            }}

            [data-testid="stSidebar"] * {{
                color: #ffffff !important;
            }}

            .stButton > button {{
                border-radius: 10px;
                font-weight: 600;
                padding: 0.6rem 1rem;
                border: 1px solid var(--global-accent-border) !important;
                background: var(--global-accent) !important;
                color: #ffffff !important;
                transition: all 0.2s ease;
            }}

            .stButton > button:hover {{
                background: var(--global-accent-hover) !important;
                color: #ffffff !important;
                box-shadow: 0 6px 18px rgba(0, 77, 68, 0.25);
            }}

            [data-testid="stMetric"] {{
                background: var(--global-surface) !important;
                border-radius: 14px;
                padding: 1rem;
                border: 1px solid var(--global-border) !important;
                box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
            }}

            [data-testid="stMetricLabel"] {{
                color: var(--global-muted) !important;
                font-weight: 600;
            }}

            [data-testid="stMetricValue"] {{
                color: var(--global-text) !important;
                font-weight: 700;
            }}

            .js-plotly-plot {{
                border-radius: 16px;
                overflow: hidden;
                box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08);
                background: var(--global-surface) !important;
            }}

            [data-testid="stFileUploader"] {{
                background-color: var(--global-surface-alt) !important;
                border-radius: 10px;
                border: 1px solid var(--global-border) !important;
                padding: 10px;
            }}

            [data-testid="stFileUploaderDropzone"] {{
                background: var(--global-surface) !important;
                border: 1px dashed var(--global-border) !important;
                border-radius: 10px !important;
            }}

            [data-testid="stFileUploaderDropzone"] * {{
                color: var(--global-text) !important;
            }}

            [data-testid="stFileUploader"] button {{
                background: var(--global-accent) !important;
                border: 1px solid var(--global-accent-border) !important;
                color: #ffffff !important;
            }}

            [data-testid="stFileUploader"] button:hover {{
                background: var(--global-accent-hover) !important;
                color: #ffffff !important;
            }}

            [data-testid="stFileUploaderFile"] {{
                background: var(--global-surface) !important;
                border: 1px solid var(--global-border) !important;
                border-radius: 8px !important;
            }}

            [data-testid="stFileUploaderFile"] * {{
                color: var(--global-text) !important;
            }}

            [data-testid="stSelectbox"] label,
            [data-testid="stMultiSelect"] label,
            [data-testid="stDateInput"] label,
            [data-testid="stPills"] label,
            [data-testid="stRadio"] label {{
                color: var(--global-text) !important;
                font-weight: 700 !important;
            }}

            [data-baseweb="select"] > div,
            [data-baseweb="popover"] [data-baseweb="menu"],
            [data-baseweb="input"] > div {{
                background: var(--global-surface-alt) !important;
                border: 1px solid var(--global-border) !important;
                border-radius: 10px !important;
                color: var(--global-text) !important;
                box-shadow: none !important;
            }}

            [data-baseweb="select"] input,
            [data-baseweb="select"] span,
            [data-baseweb="select"] svg,
            [data-baseweb="input"] input {{
                color: var(--global-text) !important;
                fill: var(--global-text) !important;
            }}

            [data-baseweb="popover"] [role="option"] {{
                background: var(--global-surface-alt) !important;
                color: var(--global-text) !important;
            }}

            [data-baseweb="popover"] [role="option"]:hover,
            [data-baseweb="popover"] [aria-selected="true"] {{
                background: var(--global-accent) !important;
                color: #ffffff !important;
            }}

            [data-testid="stPills"] button,
            [data-testid="stPillsButton"],
            [data-testid="stBaseButton-pills"] {{
                border-radius: 20px !important;
                font-size: 0.9rem !important;
                font-weight: 700 !important;
                padding: 0.45rem 1rem !important;
                border: 1.5px solid var(--global-border) !important;
                background: var(--global-surface-alt) !important;
                color: var(--global-text) !important;
                box-shadow: none !important;
            }}

            [data-testid="stPills"] button[aria-pressed="true"],
            [data-testid="stPills"] button[aria-selected="true"],
            [data-testid="stPillsButton"][aria-pressed="true"],
            [data-testid="stBaseButton-pillsActive"] {{
                background: var(--global-accent) !important;
                border-color: var(--global-accent-border) !important;
                color: #ffffff !important;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_global_theme_toggle(key_suffix: str = "default") -> None:
    ensure_theme_state()
    current_theme = get_current_theme()
    if current_theme == "light":
        next_theme = "dark"
        button_label = "Modo Escuro"
        button_icon = ":material/dark_mode:"
    else:
        next_theme = "light"
        button_label = "Modo Claro"
        button_icon = ":material/light_mode:"

    if st.button(
        button_label,
        icon=button_icon,
        use_container_width=True,
        key=f"global_graph_theme_toggle_{key_suffix}",
    ):
        st.session_state["graph_theme"] = next_theme
        st.rerun()


def render_global_theme_header_toggle() -> None:
    st.markdown(
        """
        <style>
            .st-key-global_theme_header_toggle {
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

            .st-key-global_theme_header_toggle > div {
                width: auto !important;
                max-width: max-content;
            }

            .st-key-global_theme_header_toggle div[data-testid="stVerticalBlock"] {
                gap: 0;
                width: auto !important;
                max-width: max-content;
            }

            .st-key-global_theme_header_toggle .stButton {
                margin: 0;
                width: auto;
            }

            .st-key-global_theme_header_toggle .stButton > button {
                width: auto;
                min-height: 2.5rem;
                padding: 0.45rem 0.9rem;
                border-radius: 14px;
                white-space: nowrap;
            }

            .st-key-global_theme_header_toggle .stButton > button span[data-testid="stIconMaterial"] {
                font-size: 1rem;
            }

            @media (max-width: 1100px) {
                .st-key-global_theme_header_toggle {
                    right: 3.25rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.container(key="global_theme_header_toggle"):
        render_global_theme_toggle("header")
