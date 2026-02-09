import math
import pandas as pd
from typing import Iterable

from services.date_num_prep import parse_ptbr_number


def _norm(col: str) -> str:
    return str(col).strip().lower()


def _is_status_text(s: str) -> bool:
    n = _norm(s)
    keywords = ["seco", "poço seco", "poco seco", "na baixo", "não medido", "nao medido", "nm", "n m"]
    return any(k in n for k in keywords)


def _parse_value(cell) -> tuple[float | None, str | None]:
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        return None, None
    if isinstance(cell, (int, float)):
        return float(cell), None
    s = str(cell).strip()
    if s == "":
        return None, None
    num = parse_ptbr_number(s)
    if pd.notna(num):
        return float(num), None
    if _is_status_text(s):
        return None, s
    return None, s  # status genérico


def _combine_data_hora(data_series: pd.Series, hora_series: pd.Series | None) -> pd.Series:
    data = pd.to_datetime(data_series, errors="coerce", dayfirst=True)
    if hora_series is None:
        return data
    hora = pd.to_datetime(hora_series, errors="coerce").dt.time
    return pd.to_datetime(
        data.dt.date.astype(str) + " " + hora.astype(str),
        errors="coerce",
    ).fillna(data)


def _find_col(cols: Iterable[str], needles: list[str]) -> str | None:
    for c in cols:
        n = _norm(c)
        if all(k in n for k in needles):
            return c
    return None


def _parse_format_a(df: pd.DataFrame, sheet: str) -> pd.DataFrame | None:
    id_col = _find_col(df.columns, ["po"]) or _find_col(df.columns, ["id", "po"])
    date_col = _find_col(df.columns, ["data"])
    hora_col = _find_col(df.columns, ["hora"])
    if not id_col or not date_col:
        return None

    param_cols = [c for c in df.columns if c not in {id_col, date_col, hora_col}]
    if not param_cols:
        return None

    data = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    hora = df[hora_col] if hora_col else None
    datahora = _combine_data_hora(data, hora)

    records = []
    for _, row in df.iterrows():
        poco = row[id_col]
        if pd.isna(poco):
            continue
        poco = str(poco).strip()
        for col in param_cols:
            val_raw = row[col]
            valor, status = _parse_value(val_raw)
            records.append(
                dict(
                    poco_id=poco,
                    Data=data.iloc[_],
                    Hora=row[hora_col] if hora_col else None,
                    DataHora=datahora.iloc[_],
                    parametro=str(col).strip(),
                    valor=valor,
                    status=status,
                    sheet=sheet,
                )
            )
    out = pd.DataFrame(records)
    if not out.empty:
        out["parametro"] = out["parametro"].str.strip()
    return out


def _parse_format_b(df: pd.DataFrame, sheet: str) -> pd.DataFrame | None:
    # espera Data na célula (0,0) e Id do Poço na célula (1,0)
    if df.shape[0] < 3 or df.shape[1] < 3:
        return None
    first_cell = str(df.iat[0, 0]).lower()
    second_cell = str(df.iat[1, 0]).lower()
    if "data" not in first_cell or ("po" not in second_cell and "ponto" not in second_cell and "id" not in second_cell):
        return None

    records = []
    for col in range(1, df.shape[1]):
        data_cell = df.iat[0, col]
        if pd.isna(data_cell):
            continue
        date_val = pd.to_datetime(data_cell, errors="coerce", dayfirst=True)
        if pd.isna(date_val):
            continue
        param = df.iat[1, col]
        if param is None or str(param).strip() == "":
            continue
        for row_idx in range(2, df.shape[0]):
            poco = df.iat[row_idx, 0]
            if pd.isna(poco):
                continue
            val_raw = df.iat[row_idx, col]
            valor, status = _parse_value(val_raw)
            records.append(
                dict(
                    poco_id=str(poco).strip(),
                    Data=date_val,
                    Hora=None,
                    DataHora=date_val,
                    parametro=str(param).strip(),
                    valor=valor,
                    status=status,
                    sheet=sheet,
                )
            )
    if not records:
        return None
    out = pd.DataFrame.from_records(records)
    out["parametro"] = out["parametro"].str.strip()
    return out


def read_in_situ_excel(file_obj) -> pd.DataFrame:
    """
    Lê um Excel (formato A ou B) e devolve dataset long padronizado.
    file_obj: caminho ou file-like.
    """
    xls = pd.ExcelFile(file_obj)
    frames = []
    for sheet in xls.sheet_names:
        df = xls.parse(sheet, header=0)
        parsed = _parse_format_a(df, sheet)
        # tentativa com header dinâmico (primeira linha que contenha 'poco' e 'data')
        if (parsed is None or parsed.empty):
            for hdr in range(1, 6):
                try:
                    df_try = xls.parse(sheet, header=hdr)
                except Exception:
                    continue
                parsed = _parse_format_a(df_try, sheet)
                if parsed is not None and not parsed.empty:
                    break

        if parsed is None or parsed.empty:
            # tenta formato B lendo sem headers fixos
            raw = xls.parse(sheet, header=None)
            parsed = _parse_format_b(raw, sheet)
        if parsed is not None and not parsed.empty:
            frames.append(parsed)

    if not frames:
        return pd.DataFrame(columns=["poco_id", "Data", "Hora", "DataHora", "parametro", "valor", "status", "sheet"])

    out = pd.concat(frames, ignore_index=True, sort=False)
    out = out.dropna(subset=["poco_id", "DataHora"])
    out["poco_id"] = out["poco_id"].astype(str).str.strip().str.upper()
    # filtra poços não desejados
    out = out[~out["poco_id"].str.contains(r"^entrada|^saida|^saída|^max|^mín|^media", case=False, regex=True)]
    return out


def pivot_in_situ_for_plot(df_long: pd.DataFrame, parametro: str) -> pd.DataFrame:
    df = df_long[df_long["parametro"].str.lower() == parametro.lower()].copy()
    if df.empty:
        return pd.DataFrame()
    df = df[df["valor"].notna()]
    pivot = (
        df.pivot_table(index="DataHora", columns="poco_id", values="valor", aggfunc="mean")
        .sort_index()
    )
    pivot.index.name = "DataHora"
    return pivot.reset_index()
