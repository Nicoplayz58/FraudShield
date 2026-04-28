from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib


@dataclass
class ModelBundle:
    pipeline: Any
    expected_columns: list[str]
    model_name: str
    metadata: dict[str, Any]


def _extract_expected_columns(pipeline: Any) -> list[str]:
    if hasattr(pipeline, "feature_names_in_"):
        return list(pipeline.feature_names_in_)

    if hasattr(pipeline, "named_steps") and "preprocess" in pipeline.named_steps:
        preprocess = pipeline.named_steps["preprocess"]
    else:
        preprocess = None

    if preprocess is not None and hasattr(preprocess, "feature_names_in_"):
        return list(preprocess.feature_names_in_)

    raise ValueError("No se pudo inferir el schema esperado del modelo (feature_names_in_).")


def load_model_bundle(model_path: Path, metadata_path: Path | None = None) -> ModelBundle:
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontro el modelo en: {model_path}")

    pipeline = joblib.load(model_path)
    expected_columns = _extract_expected_columns(pipeline)

    metadata: dict[str, Any] = {}
    if metadata_path is not None and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    return ModelBundle(
        pipeline=pipeline,
        expected_columns=expected_columns,
        model_name=str(metadata.get("model", "LightGBM")),
        metadata=metadata,
    )
