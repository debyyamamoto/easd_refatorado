import os
import json
import pandas as pd


class ParquetDataset:
    def __init__(self, dataset_name: str) -> None:
        self.name = dataset_name

    def generate(self, p_dataset_path: str, p_separator: str, p_json_schema_path: str):
        df = self._read_dataset(p_dataset_path, p_separator)
        json_schema = self._read_schema(p_json_schema_path)
        df = self._set_dtypes(df, json_schema)
        df = df.dropna()

        self._export_parquet(df)

    def _read_dataset(self, p_path: str, p_separator: str):
        ext = os.path.splitext(p_path)[-1].lower()

        if ext == ".csv":
            df = pd.read_csv(p_path, sep=p_separator)
        elif ext in [".xls", ".xlsx"]:
            df = pd.read_excel(p_path)
        elif ext == ".json":
            df = pd.read_json(p_path)
        elif ext == ".html":
            df = pd.read_html(p_path)[0]
        elif ext == ".parquet":
            df = pd.read_parquet(p_path)
        else:
            raise ValueError(f"Error: Unsupported file format '{ext}'")

        return df

    def _read_schema(self, p_path: str) -> dict:
        with open(p_path, "r") as file:
            dtype_schema = json.load(file)

        return dtype_schema

    def _set_dtypes(self, p_df: pd.DataFrame, p_dict_schema: dict):
        return p_df.astype(p_dict_schema)

    def _export_parquet(self, p_df: pd.DataFrame):
        p_df.to_parquet(f"{self.name}.parquet", engine="pyarrow", index=False)
