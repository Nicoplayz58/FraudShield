from __future__ import annotations

import hashlib
import json
from typing import Any
from uuid import uuid4

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine

from db.schema import PREDICTION_EVENT_FIELDS, ensure_prediction_schema, prediction_batches, predictions as predictions_table


def _canonicalize_transactions(transactions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(item: dict[str, Any]) -> tuple[int, str]:
        trans_num = item.get("trans_num")
        if trans_num is None:
            return (1, json.dumps(item, sort_keys=True, default=str, ensure_ascii=True))
        return (0, str(trans_num))

    return [dict(row) for row in sorted(transactions, key=sort_key)]


def compute_batch_hash(model_name: str, threshold: float, transactions: list[dict[str, Any]]) -> str:
    canonical = {
        "model": str(model_name),
        "threshold": round(float(threshold), 6),
        "transactions": _canonicalize_transactions(transactions),
    }
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def persist_prediction_batch(
    engine: Engine,
    *,
    model_name: str,
    threshold: float,
    transactions: list[dict[str, Any]],
    risk_scores: list[float],
    predictions: list[int],
) -> dict[str, Any]:
    ensure_prediction_schema(engine)

    if len(transactions) != len(risk_scores) or len(transactions) != len(predictions):
        raise ValueError("transactions, risk_scores y predictions deben tener la misma longitud.")

    batch_hash = compute_batch_hash(model_name=model_name, threshold=threshold, transactions=transactions)
    batch_values = {
        "id": uuid4(),
        "batch_hash": batch_hash,
        "model": str(model_name),
        "threshold": float(threshold),
        "total_transactions": int(len(transactions)),
    }

    prediction_rows: list[dict[str, Any]] = []
    for transaction, score, prediction in zip(transactions, risk_scores, predictions):
        row = {
            "id": uuid4(),
            "batch_id": None,
            "trans_num": str(transaction["trans_num"]),
            "risk_score": float(score),
            "prediction": bool(int(prediction)),
        }
        for field in PREDICTION_EVENT_FIELDS:
            row[field] = transaction.get(field)
        prediction_rows.append(row)

    with engine.begin() as connection:
        batch_stmt = pg_insert(prediction_batches).values(batch_values)
        batch_stmt = batch_stmt.on_conflict_do_update(
            index_elements=[prediction_batches.c.batch_hash],
            set_={
                "model": batch_stmt.excluded.model,
                "threshold": batch_stmt.excluded.threshold,
                "total_transactions": batch_stmt.excluded.total_transactions,
            },
        ).returning(prediction_batches.c.id)
        persisted_batch_id = connection.execute(batch_stmt).scalar_one()

        if prediction_rows:
            for row in prediction_rows:
                row["batch_id"] = persisted_batch_id

            prediction_stmt = pg_insert(predictions_table).values(prediction_rows)
            prediction_stmt = prediction_stmt.on_conflict_do_nothing(index_elements=[predictions_table.c.batch_id, predictions_table.c.trans_num])
            connection.execute(prediction_stmt)

    return {
        "batch_id": str(persisted_batch_id),
        "batch_hash": batch_hash,
        "total_transactions": len(prediction_rows),
    }
