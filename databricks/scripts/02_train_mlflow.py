from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _bootstrap import ensure_src_on_path


DEFAULT_INPUT_PATH = "/Volumes/ml/fraudshield/data/credit_card_transactions_fe.parquet"
DEFAULT_EXPERIMENT_NAME = "/Shared/FraudShield"
DEFAULT_REGISTERED_MODEL_NAME = "FraudShield_LightGBM"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FraudShield model with MLflow in Databricks.")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="Ruta parquet del dataset engineered.")
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME, help="Nombre del experimento MLflow.")
    parser.add_argument(
        "--registered-model-name",
        default=DEFAULT_REGISTERED_MODEL_NAME,
        help="Nombre opcional para registrar el modelo en MLflow.",
    )
    return parser.parse_args()


def _is_databricks_path(path_value: str) -> bool:
    return path_value.startswith("/Volumes/") or path_value.startswith("dbfs:/")


def _load_input_dataframe(input_path: str):
    if _is_databricks_path(input_path):
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()
        df_spark = spark.read.parquet(input_path)
        print(f"Spark rows: {df_spark.count():,}")
        print(f"Spark columns: {len(df_spark.columns):,}")
        return df_spark.toPandas()

    return pd.read_parquet(input_path)


def main() -> None:
    ensure_src_on_path()

    from fraudshield_databricks import train_model_with_mlflow

    args = parse_args()
    df = _load_input_dataframe(args.input)
    result = train_model_with_mlflow(
        df,
        experiment_name=args.experiment_name,
        run_name="lightgbm_baseline_databricks",
        registered_model_name=args.registered_model_name,
    )

    print(f"Run ID: {result.run_id}")
    print(f"Model URI: {result.model_uri}")
    print(f"Metrics: {result.metrics}")


if __name__ == "__main__":
    main()