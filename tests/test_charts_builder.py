import random

import pandas as pd

from app.charts.builder import SeriesSpec, build_time_chart_plotly


def test_build_time_chart_plotly_numeric_and_categorical():
    rng = random.Random(123)
    dates = pd.date_range("2026-01-01", periods=6, freq="D")
    df = pd.DataFrame(
        {
            "Data": dates,
            "Valor": [rng.uniform(1.0, 5.0) for _ in range(6)],
            "Status": ["OK", "OK", "ALTO", "ALTO", "OK", "BAIXO"],
        }
    )

    series = [
        SeriesSpec(y="Valor", label="Valor", kind="line"),
        SeriesSpec(y="Status", label="Status", kind="status_step"),
    ]

    fig, insights = build_time_chart_plotly(df, x="Data", series=series)
    assert len(fig.data) == 2
    assert "Valor" in insights["series"]
    assert insights["series"]["Valor"]["type"] == "numeric"
