# services/dataset_service.py
from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd
from openpyxl import load_workbook


# =======================
# Result / Metadata
# =======================

@dataclass
class SheetSkipInfo:
    sheet: str
    reason: str
    has_charts: bool = False
    non_empty_cells_sample: int = 0


@dataclass
class DatasetResult:
    df_dict: dict[str, pd.DataFrame] | None
    errors: list[str]
    warnings: list[str]
    skipped: list[SheetSkipInfo]


# =======================
# Normalization helpers
# =======================

def _is_empty_df(df: pd.DataFrame | None) -> bool:
    return df is None or df.empty


def format_datetime_columns_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove a hora de colunas datetime, mantendo apenas a data.
    Retorna uma cópia do DataFrame com as datas formatadas como strings (apenas data).
    """
    if _is_empty_df(df):
        return df
    
    df_display = df.copy()
    
    for col in df_display.columns:
        if pd.api.types.is_datetime64_any_dtype(df_display[col]):
            # Formata para string no padrão DD/MM/YYYY
            df_display[col] = df_display[col].dt.strftime("%d/%m/%Y")
    
    return df_display


def remove_accumulated_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove linhas onde a coluna de "Poço" contém "Acumulado".
    Procura por colunas que possam conter essa informação (Poço, poco, ponto, well, etc).
    """
    if _is_empty_df(df):
        return df
    
    df_filtered = df.copy()
    
    # Procura por coluna que contém informação de poço/ponto
    poco_col = None
    candidates = ["poço", "poco", "ponto", "well", "pocos", "pontos", "wells"]
    
    for col in df_filtered.columns:
        col_lower = col.lower()
        for candidate in candidates:
            if candidate in col_lower:
                poco_col = col
                break
        if poco_col:
            break
    
    # Se encontrou coluna de poço, remove linhas com "Acumulado"
    if poco_col:
        df_filtered = df_filtered[
            ~df_filtered[poco_col].astype(str).str.strip().str.lower().eq("acumulado")
        ]
    
    return df_filtered


def _strip_accents_basic(s: str) -> str:
    table = str.maketrans(
        "áàâãäéèêëíìîïóòôõöúùûüçÁÀÂÃÄÉÈÊËÍÌÎÏÓÒÔÕÖÚÙÛÜÇ",
        "aaaaaeeeeiiiiooooouuuucAAAAAEEEEIIIIOOOOOUUUUC",
    )
    return str(s).translate(table)


def _norm(s: str) -> str:
    s = _strip_accents_basic(str(s).strip().lower())
    s = re.sub(r"\s+", " ", s)
    return s


def _norm_colname(c: str) -> str:
    return _norm(c).replace("/", " ").replace("-", " ").replace("_", " ")


def is_chart_sheet_name(name: str) -> bool:
    n = _norm(name)
    return (
        n.startswith("grafico")
        or " grafico" in n
        or "chart" in n
        or n.startswith("gráfico")
        or " gráfico" in n
    )


def canonical_sheet_name(name: str) -> str:
    """
    Produz um "assunto" da aba para relacionar gráficos com abas tabulares.
    Ex:
      - "Gráfico Volume Bombeado" -> "bombeado"
      - "Volume Bombeado" -> "bombeado"
      - "Gráfico Infiltrado" -> "infiltrado"
      - "Volume Infiltrado" -> "infiltrado"
    """
    n = _norm(name)

    stop_words = [
        "grafico", "gráfico",
        "grafico de", "grafico do", "grafico da",
        "chart", "dashboard", "resumo", "capa",
        "volume", "volumes",
    ]
    for w in stop_words:
        n = n.replace(w, " ")

    n = re.sub(r"[^a-z0-9]+", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


# =======================
# Remove mês/ano se existe data completa
# =======================

def _pick_best_date_col(df: pd.DataFrame) -> str | None:
    if _is_empty_df(df):
        return None

    cols = list(df.columns)
    strong = []
    for c in cols:
        n = _norm_colname(str(c))
        if any(k in n for k in ["data", "date", "timestamp", "datetime"]):
            strong.append(c)

    candidates = strong + [c for c in cols if c not in strong]

    best_col = None
    best_score = 0.0

    for c in candidates:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            score = s.notna().mean()
        else:
            conv = pd.to_datetime(s, errors="coerce", dayfirst=True)
            score = conv.notna().mean()

        if score > best_score and score >= 0.50:
            best_score = score
            best_col = c

        if best_col is not None and best_score >= 0.85 and c in strong:
            break

    return best_col


def _find_redundant_month_year_cols(df: pd.DataFrame) -> list[str]:
    redundants = []
    for c in df.columns:
        n = _norm_colname(str(c))

        is_month = (n == "mes") or (n == "mês") or (n == "month") or n.startswith("mes ") or n.endswith(" mes") or " month" in n
        is_year = (n == "ano") or (n == "year") or n.startswith("ano ") or n.endswith(" ano") or " year" in n
        is_month_year_combo = (("mes" in n or "mês" in n or "month" in n) and ("ano" in n or "year" in n))

        if is_month or is_year or is_month_year_combo:
            redundants.append(c)

    return redundants


def drop_month_year_if_full_date_exists(df: pd.DataFrame) -> pd.DataFrame:
    if _is_empty_df(df):
        return df

    date_col = _pick_best_date_col(df)
    if not date_col:
        return df

    # garante datetime
    try:
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    except Exception:
        return df

    # se a data for válida em boa parte das linhas, aplica regra
    valid_rate = df[date_col].notna().mean()
    if valid_rate < 0.50:
        return df

    redundants = _find_redundant_month_year_cols(df)
    redundants = [c for c in redundants if c != date_col]

    if redundants:
        df = df.drop(columns=redundants, errors="ignore")

    return df


# =======================
# Cleaning
# =======================

def clean_basic(df: pd.DataFrame) -> pd.DataFrame:
    if _is_empty_df(df):
        return df

    df = df.copy()

    # remove colunas "Unnamed:*"
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed:.*")]
    # remove colunas/linhas totalmente vazias (isso remove "Coluna1" se for toda None)
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")

    # normaliza nomes de colunas
    new_cols = []
    for c in df.columns:
        c_str = str(c).replace("\n", " ").replace("\r", " ")
        c_str = re.sub(r"\s+", " ", c_str).strip()
        new_cols.append(c_str)
    df.columns = new_cols

    # tenta converter datas
    for col in df.columns:
        ncol = _norm_colname(col)
        if "data" in ncol or "date" in ncol or "timestamp" in ncol:
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
            except Exception:
                pass

    # tenta converter números com vírgula quando parece numérico
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().astype(str).head(60)
            if len(sample) == 0:
                continue

            looks_numeric = sample.str.match(
                r"^\s*[<>]?\s*[-+]?\d{1,3}(\.\d{3})*(,\d+)?\s*$|^\s*[<>]?\s*[-+]?\d+([.,]\d+)?\s*$"
            ).mean()

            if looks_numeric >= 0.6:
                s = df[col].astype(str).str.strip()
                s = s.str.replace(" ", "", regex=False)
                s_num = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
                s_num = s_num.str.replace("<", "", regex=False).str.replace(">", "", regex=False)
                df[col] = pd.to_numeric(s_num, errors="coerce")

    # ✅ remove Mês/Ano se existir Data completa válida
    df = drop_month_year_if_full_date_exists(df)

    # ✅ remove colunas que ficaram 100% vazias depois das conversões
    df = df.dropna(axis=1, how="all")

    return df


# =======================
# Sheet detection (charts vs table)
# =======================

def _looks_like_na_param(value: object) -> str | None:
    if value is None:
        return None
    n = _norm(str(value))
    n = n.replace("(", " ").replace(")", " ")
    n = re.sub(r"\s+", " ", n).strip()
    if n in ("na m", "na"):
        return "NA (m)"
    if n in ("no m", "no"):
        return "NO (m)"
    if n in ("fl m", "fl"):
        return "FL (m)"
    return None


def _is_na_semanal_sheet(ws) -> bool:
    try:
        row1 = list(ws.iter_rows(min_row=1, max_row=1, values_only=True))[0]
        row2 = list(ws.iter_rows(min_row=2, max_row=2, values_only=True))[0]
    except Exception:
        return False

    if not row1 or not row2:
        return False

    first = _norm(str(row1[0])) if row1[0] is not None else ""
    if "poco" not in first and "ponto" not in first:
        return False

    params = [
        _looks_like_na_param(v)
        for v in row1[1:10]
        if v is not None
    ]
    has_triplet = {p for p in params if p}
    if not {"NA (m)", "NO (m)", "FL (m)"}.issubset(has_triplet):
        return False

    parsed_dates = 0
    for v in row2[1:10]:
        parsed = pd.to_datetime(v, errors="coerce", dayfirst=True)
        if pd.notna(parsed):
            parsed_dates += 1
    return parsed_dates > 0


def _normalize_na_semanal_value(value: object):
    if value is None or pd.isna(value):
        return pd.NA

    if isinstance(value, (int, float)):
        return value

    s = str(value).strip()
    if s == "":
        return pd.NA

    n = _norm(s)
    if n in {"nm", "n m", "nao medido"}:
        return "Não medido"
    if n in {"nd", "n d", "nao localizado", "nao localiz"}:
        return s

    if re.match(r"^[-+]?\d{1,3}(\.\d{3})*(,\d+)?$|^[-+]?\d+([.,]\d+)?$", s):
        s_num = s.replace(" ", "").replace(".", "").replace(",", ".")
        try:
            return float(s_num)
        except Exception:
            return s

    return s


def _extract_na_semanal_ws(ws) -> pd.DataFrame | None:
    try:
        row1 = list(ws.iter_rows(min_row=1, max_row=1, values_only=True))[0]
        row2 = list(ws.iter_rows(min_row=2, max_row=2, values_only=True))[0]
    except Exception:
        return None

    max_col = max(len(row1), len(row2))
    row1 = list(row1) + [None] * (max_col - len(row1))
    row2 = list(row2) + [None] * (max_col - len(row2))

    data_cols: list[tuple[int, str, pd.Timestamp]] = []
    last_date = None
    for idx in range(1, max_col):
        param = _looks_like_na_param(row1[idx])
        if not param:
            continue

        raw_date = row2[idx]
        parsed_date = pd.to_datetime(raw_date, errors="coerce", dayfirst=True)
        if pd.notna(parsed_date):
            last_date = parsed_date
        if last_date is None:
            continue

        data_cols.append((idx, param, last_date))

    if not data_cols:
        return None

    records: list[dict[str, object]] = []
    empty_streak = 0
    for row in ws.iter_rows(min_row=3, values_only=True):
        if row is None:
            continue
        row_vals = list(row) + [None] * (max_col - len(row))
        if all(v is None or (isinstance(v, str) and v.strip() == "") for v in row_vals):
            empty_streak += 1
            if empty_streak >= 5:
                break
            continue
        empty_streak = 0

        poco = row_vals[0]
        if poco is None or str(poco).strip() == "":
            continue
        poco = str(poco).strip()

        by_date: dict[pd.Timestamp, dict[str, object]] = {}
        for idx, param, date_val in data_cols:
            raw_val = row_vals[idx] if idx < len(row_vals) else None
            val = _normalize_na_semanal_value(raw_val)
            if date_val not in by_date:
                by_date[date_val] = {"Poco": poco, "Data": date_val}
            by_date[date_val][param] = val

        for date_val in sorted(by_date.keys()):
            rec = by_date[date_val]
            if any(k in rec for k in ("NA (m)", "NO (m)", "FL (m)")):
                records.append(rec)

    if not records:
        return None

    return pd.DataFrame.from_records(records)


def _sheet_has_charts(ws) -> bool:
    try:
        return hasattr(ws, "_charts") and ws._charts and len(ws._charts) > 0
    except Exception:
        return False


def _count_non_empty_cells_sample(ws, max_rows: int = 120, max_cols: int = 40) -> int:
    non_empty = 0
    try:
        for row in ws.iter_rows(min_row=1, max_row=max_rows, max_col=max_cols, values_only=True):
            for v in row:
                if v is None:
                    continue
                if isinstance(v, str) and v.strip() == "":
                    continue
                non_empty += 1
    except Exception:
        return 0
    return non_empty


def _extract_table_from_ws(
    ws,
    header_search_rows: int = 40,
    max_cols: int = 80,
    max_data_rows: int = 6000,
) -> pd.DataFrame | None:
    rows: list[list[Any]] = []
    for row in ws.iter_rows(min_row=1, max_row=max_data_rows, max_col=max_cols, values_only=True):
        rows.append(list(row))

    def row_nonempty_count(r: list[Any]) -> int:
        c = 0
        for v in r:
            if v is None:
                continue
            if isinstance(v, str) and v.strip() == "":
                continue
            c += 1
        return c

    if not rows:
        return None

    search = rows[:header_search_rows]
    counts = [row_nonempty_count(r) for r in search]
    if not counts or max(counts) < 2:
        return None

    header_idx = int(max(range(len(counts)), key=lambda i: counts[i]))
    header = rows[header_idx]
    data_rows = rows[header_idx + 1 :]

    last_col = -1
    for i, v in enumerate(header):
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        last_col = i

    if last_col < 1:
        return None

    header = header[: last_col + 1]
    data_rows = [r[: last_col + 1] for r in data_rows]

    header = [("" if v is None else str(v).strip()) for v in header]
    if sum(1 for h in header if h != "") < 2:
        return None

    seen: dict[str, int] = {}
    fixed: list[str] = []
    for h in header:
        h2 = h if h else "col"
        h2 = re.sub(r"\s+", " ", h2).strip()
        if h2 in seen:
            seen[h2] += 1
            h2 = f"{h2}_{seen[h2]}"
        else:
            seen[h2] = 0
        fixed.append(h2)

    cleaned_rows: list[list[Any]] = []
    empty_streak = 0
    for r in data_rows:
        if row_nonempty_count(r) == 0:
            empty_streak += 1
            if empty_streak >= 10:
                break
            continue
        empty_streak = 0
        cleaned_rows.append(r)

    if not cleaned_rows:
        return None

    df = pd.DataFrame(cleaned_rows, columns=fixed)
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    if df.empty or len(df.columns) < 2:
        return None

    return df


# =======================
# Main entry
# =======================

def build_dataset_from_excel(uploaded_file) -> DatasetResult:
    errors: list[str] = []
    warnings: list[str] = []
    skipped: list[SheetSkipInfo] = []

    try:
        file_bytes = uploaded_file.getvalue()
        bio = io.BytesIO(file_bytes)
    except Exception as e:
        return DatasetResult(df_dict=None, errors=[f"Falha ao ler bytes do upload: {e}"], warnings=[], skipped=[])

    try:
        wb = load_workbook(bio, data_only=True, read_only=True)
    except Exception as e:
        return DatasetResult(df_dict=None, errors=[f"Falha ao abrir Excel (openpyxl): {e}"], warnings=[], skipped=[])

    sheet_names = list(wb.sheetnames)

    # cria mapa "assunto -> aba tabular" para ignorar "Gr?fico X" quando existe a irm? "X"
    canonical_to_data_sheet: dict[str, str] = {}
    for s in sheet_names:
        if not is_chart_sheet_name(s):
            canon = canonical_sheet_name(s)
            if canon:
                canonical_to_data_sheet[canon] = s

    df_dict: dict[str, pd.DataFrame] = {}

    for sheet_name in sheet_names:
        ws = wb[sheet_name]

        if _is_na_semanal_sheet(ws):
            df = _extract_na_semanal_ws(ws)
            if df is None or df.empty:
                skipped.append(
                    SheetSkipInfo(
                        sheet=sheet_name,
                        reason="Falha ao interpretar planilha NA semanal.",
                        has_charts=_sheet_has_charts(ws),
                        non_empty_cells_sample=_count_non_empty_cells_sample(ws),
                    )
                )
            else:
                df_dict[sheet_name] = df
            continue

        has_charts = _sheet_has_charts(ws)
        non_empty_sample = _count_non_empty_cells_sample(ws)

        # 1) ignora "Gr?fico ..." se existe a aba irm? tabular
        if is_chart_sheet_name(sheet_name):
            canon = canonical_sheet_name(sheet_name)
            if canon and canon in canonical_to_data_sheet:
                skipped.append(
                    SheetSkipInfo(
                        sheet=sheet_name,
                        reason=f"Aba de gr?fico ignorada (existe aba tabular correspondente: '{canonical_to_data_sheet[canon]}')",
                        has_charts=has_charts,
                        non_empty_cells_sample=non_empty_sample,
                    )
                )
                continue

        # 2) fallback: tem gr?fico e pouco conte?do tabular
        if has_charts and non_empty_sample < 40:
            skipped.append(
                SheetSkipInfo(
                    sheet=sheet_name,
                    reason="Aba cont?m gr?fico e Não parece ter tabela (amostra com pouco conte?do).",
                    has_charts=True,
                    non_empty_cells_sample=non_empty_sample,
                )
            )
            continue

        # 3) extrai e limpa tabela
        try:
            df = _extract_table_from_ws(ws)
            if df is None:
                skipped.append(
                    SheetSkipInfo(
                        sheet=sheet_name,
                        reason="Não foi poss?vel identificar uma tabela na aba.",
                        has_charts=has_charts,
                        non_empty_cells_sample=non_empty_sample,
                    )
                )
                continue

            df = clean_basic(df)
            if df.empty:
                skipped.append(
                    SheetSkipInfo(
                        sheet=sheet_name,
                        reason="Tabela extra?da ficou vazia ap?s limpeza.",
                        has_charts=has_charts,
                        non_empty_cells_sample=non_empty_sample,
                    )
                )
                continue

            df_dict[sheet_name] = df

        except Exception as e:
            warnings.append(f"Erro ao processar a aba '{sheet_name}': {e}")
            skipped.append(
                SheetSkipInfo(
                    sheet=sheet_name,
                    reason=f"Exce??o ao processar: {e}",
                    has_charts=has_charts,
                    non_empty_cells_sample=non_empty_sample,
                )
            )

    if not df_dict:
        errors.append("Nenhuma aba tabular foi encontrada (ou todas foram ignoradas).")
        return DatasetResult(df_dict=None, errors=errors, warnings=warnings, skipped=skipped)

    if skipped:
        warnings.append(f"Foram ignoradas {len(skipped)} abas que Não pareciam tabulares (ex.: gr?ficos).")

    return DatasetResult(df_dict=df_dict, errors=errors, warnings=warnings, skipped=skipped)

def prep_vol_bombeado(vol_bombeado: pd.DataFrame) -> pd.DataFrame:
    df = normalize_dates(vol_bombeado, "Data")
    for c in ["Hidrômetro Manhã", "Hidrômetro Tarde", "Volume Bombeado (m³)"]:
        if c in df.columns:
            df[c] = df[c].map(parse_ptbr_number)
    df["prefix"] = df["Poço"].map(add_prefix)
    return df

def prep_vol_infiltrado(vol_infiltrado: pd.DataFrame) -> pd.DataFrame:
    df = normalize_dates(vol_infiltrado, "Data")
    for c in ["Hidrômetro Manhã", "Hidrômetro Tarde", "Volume Infiltrado"]:
        if c in df.columns:
            df[c] = df[c].map(parse_ptbr_number)
    df["prefix"] = df["Ponto"].map(add_prefix)
    return df

def prep_na_semanal(na_semanal: pd.DataFrame) -> pd.DataFrame:
    df = normalize_dates(na_semanal, "Data")
    for c in ["NA (m)", "NO (m)", "FL (m)"]:
        if c in df.columns:
            df[c] = df[c].map(parse_ptbr_number)
    df["prefix"] = df["Poco"].map(add_prefix)
    return df

def build_entity_df(entity: str, vb: pd.DataFrame, vi: pd.DataFrame, na: pd.DataFrame) -> pd.DataFrame:
    """
    Junta no mesmo DF as colunas possíveis para a entidade selecionada.
    Nem toda entidade existe nos 3 datasets, então vai entrando só o que tiver.
    """
    parts = []

    # vol bombeado por Poço
    df_vb = vb[(vb["Poço"] == entity) & (vb["Poço"] != "Acumulado")].copy()
    if not df_vb.empty:
        df_vb = df_vb.rename(columns={
            "Hidrômetro Manhã": "bombeado_hm",
            "Hidrômetro Tarde": "bombeado_ht",
            "Volume Bombeado (m³)": "bombeado_vol",
        })
        parts.append(df_vb[["Data", "bombeado_hm", "bombeado_ht", "bombeado_vol"]])

    # vol infiltrado por Ponto
    df_vi = vi[(vi["Ponto"] == entity) & (vi["Ponto"] != "Acumulado")].copy()
    if not df_vi.empty:
        df_vi = df_vi.rename(columns={
            "Hidrômetro Manhã": "infiltrado_hm",
            "Hidrômetro Tarde": "infiltrado_ht",
            "Volume Infiltrado": "infiltrado_vol",
        })
        parts.append(df_vi[["Data", "infiltrado_hm", "infiltrado_ht", "infiltrado_vol"]])

    # NA semanal por Poco
    df_na = na[na["Poco"] == entity].copy()
    if not df_na.empty:
        df_na = df_na.rename(columns={"NA (m)": "na_m"})
        parts.append(df_na[["Data", "na_m"]])

    if not parts:
        return pd.DataFrame(columns=["Data"])

    # merge progressivo por Data
    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on="Data", how="outer")

    out = out.sort_values("Data")
    return out
