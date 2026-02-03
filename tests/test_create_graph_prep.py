import importlib
import os
import random
import sys
import types

import pandas as pd

os.environ["EPA_TESTING"] = "1"
APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
if "streamlit" not in sys.modules:
    class _DummySidebar:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def subheader(self, *args, **kwargs):
            return None

        def radio(self, *args, **kwargs):
            options = args[1] if len(args) > 1 else kwargs.get("options", [])
            return options[0] if options else None

    class _DummyStreamlit:
        def __init__(self):
            self.session_state = {}
            self.sidebar = _DummySidebar()

        def title(self, *args, **kwargs):
            return None

        def info(self, *args, **kwargs):
            return None

        def stop(self, *args, **kwargs):
            return None

        def error(self, *args, **kwargs):
            return None

        def warning(self, *args, **kwargs):
            return None

        def subheader(self, *args, **kwargs):
            return None

        def radio(self, *args, **kwargs):
            options = args[1] if len(args) > 1 else kwargs.get("options", [])
            return options[0] if options else None

        def checkbox(self, *args, **kwargs):
            return kwargs.get("value", False)

        def selectbox(self, *args, **kwargs):
            options = args[1] if len(args) > 1 else kwargs.get("options", [])
            return options[0] if options else None

        def multiselect(self, *args, **kwargs):
            default = kwargs.get("default", [])
            return default if default else []

    sys.modules["streamlit"] = _DummyStreamlit()
from app.pages import create_graph as cg  # noqa: E402
importlib.reload(cg)


def _rand_series(seed: int, n: int) -> list[float]:
    rng = random.Random(seed)
    return [round(rng.uniform(1.0, 5.0), 2) for _ in range(n)]


def test_prep_vol_bombeado_creates_keys_and_volume():
    df = pd.DataFrame(
        {
            "Data": ["01/01/2026", "02/01/2026"],
            "Poco": ["PR-01", "PR-02"],
            "Hidrometro Manha": ["1.000,0", "2.000,0"],
            "Hidrometro Tarde": ["1.200,0", "2.200,0"],
            "Volume Bombeado": ["10,5", "12,5"],
        }
    )
    out = cg.prep_vol_bombeado(df)
    assert "bombeado_vol" in out.columns
    assert out["bombeado_vol"].iloc[0] == 10.5
    assert out["poco_key"].tolist() == ["PR-01", "PR-02"]


def test_prep_na_semanal_parses_status_and_depth():
    df = pd.DataFrame(
        {
            "Data": ["01/01/2026", "08/01/2026"],
            "Poco": ["PR-01", "PR-01"],
            "NA (m)": ["1,5", "nm"],
            "NO (m)": ["", ""],
            "FL (m)": ["odor", ""],
            "Observacao": ["Seco 2,3 prof", ""],
        }
    )
    out = cg.prep_na_semanal(df)
    assert out["na_val"].iloc[0] == 1.5
    assert isinstance(out["na_status"].iloc[1], str)
    assert "medido" in out["na_status"].iloc[1].lower()
    assert out["dry_depth"].iloc[0] == 2.3
    assert out["fl_phase"].iloc[0] == "Odor"


def test_build_point_series_shapes_columns():
    vb = pd.DataFrame(
        {
            "Data": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "poco_key": ["PR-01", "PR-01"],
            "bombeado_vol": _rand_series(7, 2),
        }
    )
    na = pd.DataFrame(
        {
            "Data": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "entity_key": ["PR-01", "PR-01"],
            "na_val": _rand_series(8, 2),
            "no_val": [pd.NA, pd.NA],
            "fl_val": [pd.NA, pd.NA],
            "na_status": [pd.NA, pd.NA],
            "no_status": [pd.NA, pd.NA],
            "fl_status": [pd.NA, pd.NA],
            "fl_phase": [pd.NA, pd.NA],
            "dry_depth": [pd.NA, pd.NA],
            "obs_status": [pd.NA, pd.NA],
        }
    )
    wide, na_flat = cg.build_point_series(vb, na, ["PR-01"])
    assert "bombeado__PR-01" in wide.columns
    assert "na_val__PR-01" in wide.columns
    assert not na_flat.empty


def test_build_na_pr_vs_infiltrado_merges():
    na = pd.DataFrame(
        {
            "Data": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "entity_key": ["PR-01", "PR-01"],
            "na_val": [1.0, 2.0],
        }
    )
    vi = pd.DataFrame(
        {
            "Data": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "infiltrado_vol": [10.0, 11.0],
        }
    )
    out = cg.build_na_pr_vs_infiltrado(na, vi)
    assert list(out.columns) == ["Data", "na_val", "infiltrado_vol"]
    assert out["infiltrado_vol"].sum() == 21.0
