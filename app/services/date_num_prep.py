import re
import numpy as np
import pandas as pd


def parse_ptbr_number(s):
    """
    Converte números vindos como string em formatos comuns BR:
    - "1.230" (pode ser 1.23 OU 1230) -> aqui assumimos decimal com ponto (seu exemplo tem 11.87 e 11.870)
    - "1.230,50" -> vira 1230.50
    - "123,4" -> vira 123.4
    Se já for número, devolve como float.
    """
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    if isinstance(s, (int, float, np.number)):
        return float(s)

    s = str(s).strip()
    if s == "" or s.lower() in {"nan", "none", "<na>"}:
        return np.nan

    # caso misto 1.234,56 -> remove milhares e troca decimal
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
        return pd.to_numeric(s, errors="coerce")

    # caso 123,45 -> troca decimal
    if "," in s:
        s = s.replace(",", ".")
        return pd.to_numeric(s, errors="coerce")

    # caso só ponto -> assume decimal
    return pd.to_numeric(s, errors="coerce")


def normalize_dates(df, date_col):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    return df


def add_prefix(entity_name: str) -> str:
    """
    PR-15 -> PR
    PM-01 -> PM
    PI-35 -> PI
    'Geral'/'Saída 1' -> PONTO
    """
    if entity_name is None:
        return "UNK"
    s = str(entity_name).strip()
    m = re.match(r"^([A-Za-z]{2})[-_ ]?\d+", s)
    if m:
        return m.group(1).upper()
    return "PONTO"


def add_pr_for_pb_pi(entity_name: str) -> str | None:
    """
    PB-1 -> PR-1
    PI-12 -> PR-12
    PM-3 -> None
    """
    if entity_name is None:
        return None
    s = str(entity_name).strip()
    m = re.match(r"^(PB|PI)[-_ ]?(\d+)$", s, flags=re.I)
    if not m:
        return None
    return f"PR-{m.group(2)}"
