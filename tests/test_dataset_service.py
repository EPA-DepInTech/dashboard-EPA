from io import BytesIO
import random

import pandas as pd

from app.services import dataset_service


class _Upload:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def _rand_numbers(seed: int, n: int) -> list[float]:
    rng = random.Random(seed)
    return [round(rng.uniform(10.0, 50.0), 2) for _ in range(n)]


def test_clean_basic_removes_unnamed_and_converts_numbers():
    df = pd.DataFrame(
        {
            "Unnamed: 0": [None, None, None],
            " Data ": ["01/01/2026", "02/01/2026", "03/01/2026"],
            "Mes": [1, 1, 1],
            "Ano": [2026, 2026, 2026],
            "Valor": ["1.234,50", "2.000,00", "3.500,75"],
        }
    )
    out = dataset_service.clean_basic(df)
    assert "Unnamed: 0" not in out.columns
    assert "Mes" not in out.columns
    assert "Ano" not in out.columns
    assert pd.api.types.is_datetime64_any_dtype(out["Data"])
    assert out["Valor"].iloc[0] == 1234.50


def test_drop_month_year_if_full_date_exists():
    df = pd.DataFrame(
        {
            "Data": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "Mes": [1, 1, 1],
            "Ano": [2026, 2026, 2026],
            "Valor": [1, 2, 3],
        }
    )
    out = dataset_service.drop_month_year_if_full_date_exists(df)
    assert "Mes" not in out.columns
    assert "Ano" not in out.columns


def test_build_dataset_from_excel_ignores_chart_sheet():
    data = pd.DataFrame(
        {
            "Data": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "Poco": ["PR-01", "PR-01"],
            "Volume Bombeado": _rand_numbers(42, 2),
        }
    )
    chart = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        data.to_excel(writer, sheet_name="Volume Bombeado", index=False)
        chart.to_excel(writer, sheet_name="Grafico Volume Bombeado", index=False)

    result = dataset_service.build_dataset_from_excel(_Upload(bio.getvalue()))

    assert result.df_dict is not None
    assert "Volume Bombeado" in result.df_dict
    skipped_names = {s.sheet for s in result.skipped}
    assert "Grafico Volume Bombeado" in skipped_names
