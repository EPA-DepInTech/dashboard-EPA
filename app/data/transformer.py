import pandas as pd
import numpy as np
import re

def normalize_poco(poco: str) -> str:
    """
    Padroniza 'PM 27', 'PM-27', 'pm27' -> 'PM-27'
    (com zero à esquerda opcional; aqui deixei sem zero pra manter PM-27)
    """
    if poco is None:
        return ""
    s = str(poco).strip().upper()
    m = re.match(r"^PM\s*[-]?\s*(\d+)\s*$", s)
    if not m:
        return s
    num = int(m.group(1))
    return f"PM-{num}"  # se quiser zero à esquerda: f"PM-{num:02d}"


def wide_sheet_to_sample_rows(
    df_wide: pd.DataFrame,
    sheet_name: str | None = None,
    prefix_sheet: bool = False,
) -> pd.DataFrame:
    """
    Converte 1 sheet no formato "matriz" do Excel para:
      1 linha = 1 amostra (Poço, Data, Amostra + colunas de parâmetros)

    - Usa as duas primeiras linhas como metadados:
        linha 0 -> datas de coleta por coluna
        linha 1 -> id/identificação da amostra por coluna
    - Linhas 2+ -> parâmetros (coluna 'Ponto') e valores por coluna
    - Resolve colunas Unnamed herdando o último poço visto (efeito do merge do Excel)
    """

    df = df_wide.copy()


    while "Ponto" not in df.columns:
        df.columns = df.iloc[0]
        df = df.iloc[1:] #pula a primeira linha por que é vazia
        if df.size == 0:
            print("\n>ERRO NO DATAFRAME:\n      Não foi encontrado 'Ponto' como referência na tabela.")
            return -1
        # print(f'\n\nPROCURA PONTO: {df}\n\n')


    # 1) Metadados por coluna (datas e ids de amostra)
    date_row = df.iloc[0]     # "Data" por coluna
    sample_row = df.iloc[1]   # "Identificação da Amostra" por coluna
    data = df.iloc[2:].copy() # tabela de parâmetros/resultados

    # print(f'\n\nDate: {date_row}\nSample: {sample_row}\nDados: {data}\n\n')

    # 2) Define colunas fixas (metadados do parâmetro) e colunas de amostra
    fixed_cols = ["Ponto", "CAS", "Unid.", "Listas Orientadoras", "Unnamed: 4"]
    fixed_cols = [c for c in fixed_cols if c in data.columns]

    # O resto das colunas são "amostras" (poços + Unnamed vindos de merge)
        # sample_cols = [c for c in data.columns if c not in fixed_cols]
    # -> troca por índices (funciona mesmo com nomes duplicados)
    sample_idx = [i for i, col in enumerate(data.columns) if col not in fixed_cols]

    # 3) Resolve o merge: Unnamed herda o último poço válido (usando nome da coluna)
    def _is_blank_header(x) -> bool:
        if x is None:
            return True
        # x pode ser float nan
        try:
            if pd.isna(x):
                return True
        except Exception:
            pass
        s = str(x).strip().lower()
        return s in ("", "nan", "none")

    def _looks_like_pm(x) -> bool:
        if _is_blank_header(x):
            return False
        s = str(x).strip().upper()
        return re.match(r"^PM\s*[-]?\s*\d+\s*$", s) is not None

    col_to_poco = {}
    current = None
    for i in sample_idx:
        raw = data.columns[i]

        # 1) Se a coluna tem PM explícito, atualiza current
        if _looks_like_pm(raw):
            current = str(raw).strip()

        # 2) Se for Unnamed ou vazio/nan, NÃO atualiza current (herda)
        else:
            name = str(raw).strip()
            if name.startswith("Unnamed:") or _is_blank_header(raw):
                pass
            else:
                # se vier algo estranho mas não vazio, também não atualiza (mais seguro)
                pass

        col_to_poco[i] = current


        # Filtra apenas as linhas que são parâmetros válidos (evita lixo/vazios)
    pontos_col = data["Ponto"]
    if isinstance(pontos_col, pd.DataFrame):
        pontos_col = pontos_col.iloc[:, 0]  # se 'Ponto' duplicou, pega a primeira

    pontos_col = pontos_col.astype(str).str.strip()
    params_mask = pontos_col.notna() & pontos_col.ne("") & (pontos_col.str.lower() != "nan")

    data_params = data.loc[params_mask].copy()

    # garante que data_params["Ponto"] é Series limpa
    p = data_params["Ponto"]
    if isinstance(p, pd.DataFrame):
        p = p.iloc[:, 0]
    data_params["Ponto"] = p.astype(str).str.strip()

    records = []
    for i in sample_idx:
        poco_raw = col_to_poco.get(i)
        if poco_raw is None:
            continue

        poco_id = normalize_poco(poco_raw)

        # pega sempre ESCALAR por posição (nunca vira Series)
        coleta = pd.to_datetime(date_row.iloc[i], errors="coerce")
        amostra_id = sample_row.iloc[i]

        rec = {"Poço": poco_id, "Data": coleta, "Amostra": amostra_id}
        if sheet_name:
            rec["Sheet"] = sheet_name

        # valores da coluna i (por posição)
        pontos = data_params["Ponto"]
        if isinstance(pontos, pd.DataFrame):
            pontos = pontos.iloc[:, 0]

        vals = data_params.iloc[:, i]  # <- por posição, sempre Series
        values = dict(zip(pontos.tolist(), vals.tolist()))

        # (opcional) proteger nomes reservados sem perder dado:
        reserved = {"Poço", "Data", "Amostra", "Sheet"}
        values = { (f"Param|{k}" if str(k).strip() in reserved else k): v for k, v in values.items() }

        rec.update(values)
        records.append(rec)

    out = pd.DataFrame(records)

    return out

def _first_notna(s: pd.Series):
    s2 = s.dropna()
    return s2.iloc[0] if len(s2) else np.nan

def combine_sheets(sheets: dict[str, pd.DataFrame], prefix_sheet: bool = False) -> pd.DataFrame:
    parts = []
    for name, df in sheets.items():
        parts.append(wide_sheet_to_sample_rows(df, sheet_name=name, prefix_sheet=prefix_sheet))

    out = pd.concat(parts, ignore_index=True)

    keys = ["Poço", "Data", "Amostra"]
    keys = [k for k in keys if k in out.columns]
    non_keys = [c for c in out.columns if c not in keys]

    out = out.groupby(keys, as_index=False).agg({c: _first_notna for c in non_keys})
    return out