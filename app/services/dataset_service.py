# services/dataset_service.py
from dataclasses import dataclass
import pandas as pd

@dataclass
class DatasetResult:
    df_dict: dict[pd.DataFrame] | None
    errors: list[str]
    warnings: list[str]

def build_dataset_from_excel(uploaded_file) -> DatasetResult:
    if "Histórico" in uploaded_file.name:
        all_sheets = pd.read_excel(uploaded_file, sheet_name=None)
        df_dict = {}
        print(f'\n\n{all_sheets.keys()}\n\n')
        df_dict["VOC"] = all_sheets["Resultados A. - SQIS VOC"]
        df_dict["SVOC"] = all_sheets["Resultados A. - SQIS SVOC"]
        df_dict["TPH"] = all_sheets["Resultados A. - SQIS TPH FP"]
    return DatasetResult(df_dict=df_dict, errors=[], warnings=[])
    # return DatasetResult(df=None, errors=["Estrutura do Excel ainda não definida."], warnings=[])
