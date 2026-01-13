# services/dataset_service.py
from dataclasses import dataclass
import pandas as pd
from data.transformer import combine_sheets

@dataclass
class DatasetResult:
    df_dict: dict[pd.DataFrame] | None
    errors: list[str]
    warnings: list[str]

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
        print(f'\n\n{master}\n\n')

    return DatasetResult(df_dict=master, errors=[], warnings=[])
    # return DatasetResult(df=None, errors=["Estrutura do Excel ainda não definida."], warnings=[])
