from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from mlflow.models.signature import infer_signature
from pandas.api.types import is_datetime64_any_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


RANDOM_STATE = 42
TARGET_COL = "is_fraud"


@dataclass
class TrainingResult:
    run_id: str
    experiment_name: str
    metrics: dict[str, float]
    row_count: int
    feature_count: int
    model_uri: str


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    if len(categorical_cols) == 0:
        raise ValueError("No se detectaron columnas categoricas en el dataset de entrada.")

    for col in [c for c in X.columns if is_datetime64_any_dtype(X[c])]:
        raise ValueError(
            f"La columna {col!r} sigue siendo datetime. Debe convertirse o eliminarse antes de entrenar."
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


def _build_training_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COL not in df.columns:
        raise ValueError(f"La columna objetivo {TARGET_COL!r} no existe en el dataset.")

    y = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).astype(int)
    X = df.drop(columns=[TARGET_COL]).copy()

    datetime_cols = X.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()
    if datetime_cols:
        X = X.drop(columns=datetime_cols)

    return X, y


def _evaluate(y_true: pd.Series, y_proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def train_model_with_mlflow(
    df: pd.DataFrame,
    *,
    experiment_name: str,
    run_name: str = "lightgbm_baseline_databricks",
    registered_model_name: str | None = None,
    test_size: float = 0.2,
    stratify: bool = True,
) -> TrainingResult:
    X, y = _build_training_frame(df)

    split_kwargs: dict[str, Any] = {
        "test_size": test_size,
        "random_state": RANDOM_STATE,
    }
    if stratify:
        split_kwargs["stratify"] = y

    X_train, X_test, y_train, y_test = train_test_split(X, y, **split_kwargs)

    mlflow.set_experiment(experiment_name)

    pipeline = build_pipeline(X_train)
    pipeline.fit(X_train, y_train)

    y_test_proba = pipeline.predict_proba(X_test)[:, 1]
    y_train_proba = pipeline.predict_proba(X_train)[:, 1]
    metrics = _evaluate(y_test, y_test_proba)
    train_metrics = _evaluate(y_train, y_train_proba)

    params = {
        "random_state": RANDOM_STATE,
        "test_size": test_size,
        "stratify": stratify,
        "n_estimators": 380,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "class_weight": "balanced",
    }

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metrics(metrics)
        mlflow.set_tag("dataset_rows", int(len(df)))
        mlflow.set_tag("feature_count", int(X.shape[1]))
        mlflow.set_tag("target_column", TARGET_COL)

        input_example = X_train.head(5)
        signature = infer_signature(X_train, pipeline.predict_proba(X_train))
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=registered_model_name,
        )

        mlflow.log_dict(
            {
                "experiment_name": experiment_name,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "row_count": int(len(df)),
                "feature_count": int(X.shape[1]),
                "metrics": metrics,
                "train_metrics": train_metrics,
            },
            artifact_file="training_summary.json",
        )

        return TrainingResult(
            run_id=run.info.run_id,
            experiment_name=experiment_name,
            metrics=metrics,
            row_count=int(len(df)),
            feature_count=int(X.shape[1]),
            model_uri=f"runs:/{run.info.run_id}/model",
        )
