from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from pandas.api.types import is_datetime64_any_dtype


RANDOM_STATE = 42
TARGET_COL = "is_fraud"


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    if len(categorical_cols) == 0:
        raise ValueError("No se detectaron columnas categoricas en el dataset de entrada.")

    for col in [c for c in X.columns if is_datetime64_any_dtype(X[c])]:
        raise ValueError(
            f"La columna {col!r} es datetime. Convierte el dataset a formato de modelado antes de entrenar."
        )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )

    model = LGBMClassifier(
        n_estimators=380,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena y guarda el modelo LightGBM final.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("credit_card_transactions_model_input.csv"),
        help="CSV con las variables finales de modelado",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts") / "lightgbm_pipeline.joblib",
        help="Ruta de salida del artefacto",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("artifacts") / "lightgbm_pipeline_metadata.json",
        help="Ruta de salida del metadata JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"No se encontro el dataset de entrada: {args.input}")

    df = pd.read_csv(args.input)
    if TARGET_COL not in df.columns:
        raise ValueError(f"La columna objetivo {TARGET_COL!r} no existe en el dataset.")

    y = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).astype(int)
    X = df.drop(columns=[TARGET_COL]).copy()

    pipeline = build_pipeline(X)
    pipeline.fit(X, y)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.output)

    metadata = {
        "model": "LightGBM",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input),
        "output": str(args.output),
        "rows": int(len(df)),
        "features": int(X.shape[1]),
        "target_rate": float(y.mean()),
        "random_state": RANDOM_STATE,
        "notes": "Pipeline de inferencia: imputacion + ordinal encoding + LightGBM entrenado sobre el dataset final de modelado.",
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Modelo guardado en: {args.output.resolve()}")
    print(f"Metadata guardado en: {args.metadata.resolve()}")
    print(f"Filas: {len(df):,} | Features: {X.shape[1]:,} | Fraud rate: {y.mean():.6f}")


if __name__ == "__main__":
    main()
