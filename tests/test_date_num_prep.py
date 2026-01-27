import math

import pandas as pd

from app.services import date_num_prep


def test_parse_ptbr_number_basic():
    assert date_num_prep.parse_ptbr_number("1.234,50") == 1234.50
    assert date_num_prep.parse_ptbr_number("123,4") == 123.4
    assert date_num_prep.parse_ptbr_number("11.87") == 11.87
    assert math.isnan(date_num_prep.parse_ptbr_number(""))


def test_normalize_dates_dayfirst():
    df = pd.DataFrame({"Data": ["27/01/2026", "01/02/2026"]})
    out = date_num_prep.normalize_dates(df, "Data")
    assert out["Data"].dt.day.tolist() == [27, 1]
    assert out["Data"].dt.month.tolist() == [1, 2]


def test_add_prefix():
    assert date_num_prep.add_prefix("PR-15") == "PR"
    assert date_num_prep.add_prefix("pm 01") == "PM"
    assert date_num_prep.add_prefix("Saida 1") == "PONTO"
