# services/dataset_service.py
from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd
from openpyxl import load_workbook

from data.transformer import combine_sheets


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

def build_master_dataset(sheets_dict):
    master = combine_sheets(sheets_dict)
    # aqui você pode chamar parse_result e validators também
    return master

def build_dataset_from_excel(uploaded_file) -> DatasetResult:
    if "Histórico" in uploaded_file.name:
        all_sheets = pd.read_excel(uploaded_file, sheet_name=None)
        df_dict = {}

        # Dividindo as páginas do excel (sheets) em múltiplos dataframes
        df_dict["VOC"] = all_sheets["Resultados A. - SQIS VOC"]
        df_dict["SVOC"] = all_sheets["Resultados A. - SQIS SVOC"]
        df_dict["TPHFP"] = all_sheets["Resultados A. - SQIS TPH FP"]
        df_dict["TPHFR"] = all_sheets["Resultados A. - SQIS TPH FRACIO"]
        df_dict["MNA"] = all_sheets["Resultados Analíticos - MNA"]

        master = build_master_dataset(df_dict)

        return DatasetResult(df_dict=master, errors=[], warnings=[], skipped=[])
    else:
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

        # cria mapa "assunto -> aba tabular" para ignorar "Gráfico X" quando existe a irmã "X"
        canonical_to_data_sheet: dict[str, str] = {}
        for s in sheet_names:
            if not is_chart_sheet_name(s):
                canon = canonical_sheet_name(s)
                if canon:
                    canonical_to_data_sheet[canon] = s

        df_dict: dict[str, pd.DataFrame] = {}

        for sheet_name in sheet_names:
            ws = wb[sheet_name]

            has_charts = _sheet_has_charts(ws)
            non_empty_sample = _count_non_empty_cells_sample(ws)

            # 1) ignora "Gráfico ..." se existe a aba irmã tabular
            if is_chart_sheet_name(sheet_name):
                canon = canonical_sheet_name(sheet_name)
                if canon and canon in canonical_to_data_sheet:
                    skipped.append(
                        SheetSkipInfo(
                            sheet=sheet_name,
                            reason=f"Aba de gráfico ignorada (existe aba tabular correspondente: '{canonical_to_data_sheet[canon]}')",
                            has_charts=has_charts,
                            non_empty_cells_sample=non_empty_sample,
                        )
                    )
                    continue

            # 2) fallback: tem gráfico e pouco conteúdo tabular
            if has_charts and non_empty_sample < 40:
                skipped.append(
                    SheetSkipInfo(
                        sheet=sheet_name,
                        reason="Aba contém gráfico e não parece ter tabela (amostra com pouco conteúdo).",
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
                            reason="Não foi possível identificar uma tabela na aba.",
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
                            reason="Tabela extraída ficou vazia após limpeza.",
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
                        reason=f"Exceção ao processar: {e}",
                        has_charts=has_charts,
                        non_empty_cells_sample=non_empty_sample,
                    )
                )

        if not df_dict:
            errors.append("Nenhuma aba tabular foi encontrada (ou todas foram ignoradas).")
            return DatasetResult(df_dict=None, errors=errors, warnings=warnings, skipped=skipped)

        if skipped:
            warnings.append(f"Foram ignoradas {len(skipped)} abas que não pareciam tabulares (ex.: gráficos).")

        return DatasetResult(df_dict=df_dict, errors=errors, warnings=warnings, skipped=skipped)