from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import auc, f1_score, precision_recall_curve, precision_score, recall_score
from sqlalchemy import text

from db.connection import create_database_engine, get_database_url


def fetch_evaluation_frame(database_url: str | None = None, batch_id: str | None = None) -> pd.DataFrame:
    engine = create_database_engine(get_database_url(database_url))

    where_clause = ""
    params: dict[str, Any] = {}
    if batch_id:
        where_clause = "WHERE p.batch_id = :batch_id"
        params["batch_id"] = batch_id

    sql = text(
        f"""
        WITH ranked_predictions AS (
            SELECT
                p.trans_num,
                p.prediction,
                p.risk_score,
                p.created_at,
                ROW_NUMBER() OVER (PARTITION BY p.trans_num ORDER BY p.created_at DESC, p.id DESC) AS rn
            FROM predictions p
            {where_clause}
        )
        SELECT
            t.trans_num,
            t.is_fraud,
            rp.prediction,
            rp.risk_score,
            rp.created_at
        FROM transactions t
        INNER JOIN ranked_predictions rp
            ON rp.trans_num = t.trans_num
           AND rp.rn = 1
        """
    )
    return pd.read_sql_query(sql, engine, params=params)


def compute_metrics(frame: pd.DataFrame) -> dict[str, float | None]:
    if frame.empty:
        return {"precision": None, "recall": None, "f1": None, "pr_auc": None}

    y_true = frame["is_fraud"].astype(int).to_numpy()
    y_pred = frame["prediction"].astype(int).to_numpy()
    y_score = frame["risk_score"].astype(float).to_numpy()

    metrics: dict[str, float | None] = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": None,
    }

    if np.unique(y_true).size > 1:
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        metrics["pr_auc"] = float(auc(recall, precision))

    return metrics


def print_report(frame: pd.DataFrame, metrics: dict[str, float | None]) -> None:
    total_rows = len(frame)
    fraud_rate = float(frame["is_fraud"].astype(int).mean()) if total_rows else 0.0

    print("\nFraudShield evaluation report")
    print("=" * 32)
    print(f"Rows joined: {total_rows}")
    print(f"Observed fraud rate: {fraud_rate:.4f}")
    print(f"Precision: {metrics['precision']:.4f}" if metrics["precision"] is not None else "Precision: n/a")
    print(f"Recall: {metrics['recall']:.4f}" if metrics["recall"] is not None else "Recall: n/a")
    print(f"F1-score: {metrics['f1']:.4f}" if metrics["f1"] is not None else "F1-score: n/a")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}" if metrics["pr_auc"] is not None else "PR-AUC: n/a")


def run_report(database_url: str | None = None, batch_id: str | None = None) -> dict[str, float | None]:
    frame = fetch_evaluation_frame(database_url=database_url, batch_id=batch_id)
    metrics = compute_metrics(frame)
    print_report(frame, metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate stored fraud predictions against ground truth")
    parser.add_argument("--database-url", type=str, default=None)
    parser.add_argument("--batch-id", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    run_report(database_url=args.database_url, batch_id=args.batch_id)


if __name__ == "__main__":
    main()
