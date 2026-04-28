from __future__ import annotations

from typing import Any

from sqlalchemy.engine import Engine

from db.store import persist_prediction_batch as _persist_prediction_batch


def persist_prediction_results(
    engine: Engine,
    *,
    model_name: str,
    threshold: float,
    transactions: list[dict[str, Any]],
    risk_scores: list[float],
    predictions: list[int],
) -> dict[str, Any]:
    return _persist_prediction_batch(
        engine,
        model_name=model_name,
        threshold=threshold,
        transactions=transactions,
        risk_scores=risk_scores,
        predictions=predictions,
    )
