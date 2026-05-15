from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _bootstrap import ensure_src_on_path


DEFAULT_INPUT_PATH = "/Volumes/ml/fraudshield/data"
DEFAULT_OUTPUT_PATH = "/Volumes/ml/fraudshield/data/credit_card_transactions_fe.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FraudShield feature engineering in Databricks.")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="Ruta del CSV raw o directorio de entrada.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Ruta de salida parquet dentro del Volume.")
    return parser.parse_args()


def _is_databricks_path(path_value: str) -> bool:
    return path_value.startswith("/Volumes/") or path_value.startswith("dbfs:/")


def _load_input_dataframe(input_path: str):
    if _is_databricks_path(input_path):
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()
        return spark.read.option("header", True).option("inferSchema", True).csv(input_path)

    if Path(input_path).is_dir():
        input_path = str(Path(input_path) / "credit_card_transactions.csv")

    if input_path.lower().endswith(".parquet"):
        return pd.read_parquet(input_path)

    return pd.read_csv(input_path)


def _save_output_dataframe(df_fe, output_path: str) -> None:
    if hasattr(df_fe, "write"):
        df_fe.write.mode("overwrite").parquet(output_path)
        return

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_fe.to_parquet(output_file, index=False)


def main() -> None:
    ensure_src_on_path()

    from fraudshield_databricks import build_feature_engineered_dataset, build_feature_engineered_dataset_spark

    args = parse_args()

    df_raw = _load_input_dataframe(args.input)

    if hasattr(df_raw, "count") and hasattr(df_raw, "write"):
        df_fe = build_feature_engineered_dataset_spark(df_raw)
        row_count = df_fe.count()
        column_count = len(df_fe.columns)
        _save_output_dataframe(df_fe, args.output)
    else:
        df_fe = build_feature_engineered_dataset(df_raw)
        row_count = len(df_fe)
        column_count = len(df_fe.columns)
        _save_output_dataframe(df_fe, args.output)

    print(f"Filas: {row_count:,}")
    print(f"Columnas totales: {column_count:,}")
    print(f"Archivo guardado en: {args.output}")


if __name__ == "__main__":
    main()