from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from feature_engineering_script import build_feature_engineered_dataset  # noqa: E402


RAW_COLUMNS = [
    "trans_date_trans_time",
    "cc_num",
    "merchant",
    "category",
    "amt",
    "first",
    "last",
    "gender",
    "street",
    "city",
    "state",
    "zip",
    "lat",
    "long",
    "city_pop",
    "job",
    "dob",
    "trans_num",
    "unix_time",
    "merch_lat",
    "merch_long",
    "merch_zipcode",
]


def transactions_to_dataframe(transactions: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(transactions)
    for col in RAW_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Falta la columna requerida en input: {col}")
    return df[RAW_COLUMNS].copy()


def build_model_ready_features(raw_df: pd.DataFrame, expected_columns: list[str]) -> pd.DataFrame:
    df_fe = build_feature_engineered_dataset(raw_df)

    missing_cols = [c for c in expected_columns if c not in df_fe.columns]
    if missing_cols:
        raise ValueError(
            "El post-procesamiento no pudo reproducir el schema del entrenamiento. "
            f"Columnas faltantes: {missing_cols}"
        )

    return df_fe[expected_columns].copy()
