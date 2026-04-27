from datetime import date

import pandas as pd
import streamlit as st

_PRESET_LABELS = [
    "1 sem", "2 sem", "1 mês", "2 meses",
    "6 meses", "12 meses", "Tudo", "Personalizado",
]

_PRESET_OFFSETS: dict[str, tuple[str, int]] = {
    "1 sem":    ("weeks",  1),
    "2 sem":    ("weeks",  2),
    "1 mês":    ("months", 1),
    "2 meses":  ("months", 2),
    "6 meses":  ("months", 6),
    "12 meses": ("months", 12),
}


def _to_date(d) -> date | None:
    if d is None:
        return None
    if isinstance(d, pd.Timestamp):
        return d.date()
    if isinstance(d, date):
        return d
    try:
        if pd.isna(d):
            return None
    except (TypeError, ValueError):
        pass
    return pd.Timestamp(d).date()


def period_selector(
    key: str,
    min_date=None,
    max_date=None,
    default_preset: str = "1 mês",
) -> tuple[date | None, date | None]:
    """
    Renders preset period pills + optional custom date range picker.
    Returns (start_date, end_date) as date objects or (None, None).
    """
    preset_key = f"ps_{key}_preset"
    custom_key = f"ps_{key}_custom"

    selected = st.pills(
        "Período",
        _PRESET_LABELS,
        default=default_preset,
        key=preset_key,
    )

    if selected is None:
        selected = default_preset

    max_d = _to_date(max_date)
    min_d = _to_date(min_date)

    if selected == "Personalizado":
        default_range = (min_d, max_d) if (min_d and max_d) else None
        period = st.date_input(
            "Intervalo",
            value=default_range,
            format="DD/MM/YYYY",
            key=custom_key,
            label_visibility="collapsed",
        )
        if isinstance(period, (list, tuple)) and len(period) == 2:
            return period[0], period[1]
        return None, None

    if selected == "Tudo":
        return min_d, max_d

    if max_d is None:
        return None, None

    kind, amount = _PRESET_OFFSETS[selected]
    max_ts = pd.Timestamp(max_d)
    if kind == "months":
        start_d = (max_ts - pd.DateOffset(months=amount)).date()
    else:
        start_d = (max_ts - pd.Timedelta(weeks=amount)).date()

    if min_d:
        start_d = max(start_d, min_d)

    return start_d, max_d
