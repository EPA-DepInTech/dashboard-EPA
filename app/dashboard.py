from pathlib import Path

import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
CSS_PATH = BASE_DIR / "style" / "style.css"
LOGO_PATH = BASE_DIR / "style" / "epa_logo.png"

st.set_page_config(page_title="Dashboard - Grupo EPA", layout="wide")

if CSS_PATH.exists():
    st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    st.session_state["_global_css_loaded"] = True
else:
    st.warning(f"CSS não encontrado em: {CSS_PATH}")

# ================== LOGO ==================
if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        logo_bytes = f.read()

    st.logo(
        logo_bytes,
        icon_image=logo_bytes,
    )

init_session_state()

# ================== SPLASH SCREEN (2s) ==================
if "splash_shown" not in st.session_state:
    st.session_state["splash_shown"] = False

if not st.session_state["splash_shown"]:
    if os.path.exists(logo_path):
        logo_b64 = load_image_base64(logo_path)

        splash_html = f"""
        <div class="splash-container">
            <img src="data:image/png;base64,{logo_b64}" class="splash-logo" />
        </div>
        <div data-testid="stSidebarContent" 
        class="st-emotion-cache-155jwzh"></div>
        """
        placeholder = st.empty()
        placeholder.markdown(splash_html, unsafe_allow_html=True)
        time.sleep(2)
        placeholder.empty()

    st.session_state["splash_shown"] = True
    st.rerun()

# ================== TÍTULO ==================
st.title("📊 Dashboard de Amostras")

# ================== HELPERS ==================
def _process_upload(uploaded_files):
    if not uploaded_files:
        return

    set_uploaded_file(uploaded_files)
    result = build_dataset_from_excels(uploaded_files)

    if result.errors:
        st.error("Não foi possível processar os Excels:")
        for e in result.errors:
            st.write("-", e)
        return

    st.session_state["df_dict"] = result.df_dict  # padronizado
    # guarda datasets por arquivo para permitir seleção na página de gráficos
    df_dict_by_file: dict[str, dict] = {}
    for f in uploaded_files:
        single = build_dataset_from_excels([f])
        if single.errors:
            continue
        name = getattr(f, "name", f"arquivo_{len(df_dict_by_file)+1}")
        df_dict_by_file[name] = single.df_dict or {}
    if df_dict_by_file:
        st.session_state["df_dict_by_file"] = df_dict_by_file
    else:
        st.session_state.pop("df_dict_by_file", None)
    skipped_charts = [s for s in result.skipped if s.has_charts]
    if skipped_charts:
        st.success(f"Excel operacional carregado: {len(skipped_charts)} abas de gráfico ignoradas.")
    else:
        st.success("Excel operacional carregado: nenhuma aba de gráfico ignorada.")


def _norm_text(value: object) -> str:
    s = str(value).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def _find_col(df: pd.DataFrame, tokens: list[str]) -> str | None:
    for col in df.columns:
        n = _norm_text(col)
        if all(t in n for t in tokens):
            return col
    return None

# ================== OVERVIEW ==================
def _build_overview(df_dict: dict[str, pd.DataFrame]) -> None:
    na_raw = df_dict.get("NA Semanal")
    if na_raw is None or na_raw.empty:
        st.info("NA Semanal não encontrado.")
        return

    df = na_raw.copy()
    date_col = _find_col(df, ["data"])
    poco_col = _find_col(df, ["poco"]) or _find_col(df, ["ponto"])

if not hasattr(st, "Page") or not hasattr(st, "navigation"):
    st.error(
        "A versão do Streamlit não suporta st.Page/st.navigation. "
        "Atualize para uma versão compatível para usar o app multipáginas."
    )
    st.stop()

home_page = st.Page(str(BASE_DIR / "pages" / "dashboard_page.py"), title="Dashboard", icon=":material/dashboard:")
graph_page = st.Page(str(BASE_DIR / "pages" / "create_graph.py"), title="Criar gráfico", icon=":material/query_stats:")

pg = st.navigation({"Pages": [home_page, graph_page]})
pg.run()
