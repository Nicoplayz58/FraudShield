from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

from api.client import post_prediction_batch
from db.connection import create_database_engine, get_database_url


API_FIELDS = [
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

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(slots=True)
class SimulationConfig:
    source: Literal["db", "csv"] = "db"
    batch_size: int = 500
    iterations: int = 1
    continuous: bool = False
    pause_seconds: float = 0.0
    endpoint_url: str = "http://127.0.0.1:8000"
    timeout_seconds: float = 30.0
    retries: int = 3
    csv_path: Path | None = None
    database_url: str | None = None
    table_name: str = "transactions"
    order_by: str = "trans_num"


def _validate_identifier(value: str, label: str) -> str:
    if not _IDENTIFIER_RE.match(value):
        raise ValueError(f"{label} invalido: {value}")
    return value


def iter_csv_batches(csv_path: Path, batch_size: int) -> Iterator[list[dict[str, Any]]]:
    for chunk in pd.read_csv(csv_path, usecols=API_FIELDS, chunksize=batch_size):
        yield chunk[API_FIELDS].to_dict(orient="records")


def iter_db_batches(database_url: str, table_name: str, batch_size: int, order_by: str = "trans_num") -> Iterator[list[dict[str, Any]]]:
    engine = create_database_engine(database_url)
    table_name = _validate_identifier(table_name, "table_name")
    order_by = _validate_identifier(order_by, "order_by")

    sql = text(f"SELECT {', '.join(API_FIELDS)} FROM {table_name} ORDER BY {order_by}")
    for chunk in pd.read_sql_query(sql, engine, chunksize=batch_size):
        yield chunk[API_FIELDS].to_dict(orient="records")


def _cycle_batches(batches: list[list[dict[str, Any]]]) -> Iterator[list[dict[str, Any]]]:
    while True:
        for batch in batches:
            yield batch


def _batch_iterator(config: SimulationConfig) -> Iterator[list[dict[str, Any]]]:
    if config.source == "csv":
        if config.csv_path is None:
            raise ValueError("csv_path es obligatorio cuando source='csv'.")
        return iter_csv_batches(config.csv_path, config.batch_size)

    database_url = get_database_url(config.database_url)
    return iter_db_batches(database_url, config.table_name, config.batch_size, config.order_by)


def run_realtime_simulation(config: SimulationConfig) -> dict[str, Any]:
    load_dotenv()
    logging.info(
        "Iniciando simulacion realtime: source=%s batch_size=%s iterations=%s continuous=%s",
        config.source,
        config.batch_size,
        config.iterations,
        config.continuous,
    )

    iterator: Iterator[list[dict[str, Any]]] = _batch_iterator(config)
    batches_sent = 0
    transactions_sent = 0
    failures = 0
    total_latency_seconds = 0.0
    target_batches = config.iterations if config.iterations > 0 else None

    while target_batches is None or batches_sent < target_batches:
        try:
            batch = next(iterator)
        except StopIteration:
            if config.continuous:
                iterator = _batch_iterator(config)
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
            else:
                break

        if not batch:
            continue

        try:
            response = post_prediction_batch(
                config.endpoint_url,
                batch,
                timeout_seconds=config.timeout_seconds,
                retries=config.retries,
            )
            batches_sent += 1
            transactions_sent += len(batch)
            total_latency_seconds += response.latency_seconds

            response_payload = response.payload if isinstance(response.payload, dict) else {}
            total_predictions = response_payload.get("total_transactions", len(batch))
            logging.info(
                "batch=%s size=%s status=%s latency_ms=%.2f total_predictions=%s",
                batches_sent,
                len(batch),
                response.status_code,
                response.latency_seconds * 1000.0,
                total_predictions,
            )
        except Exception as exc:
            failures += 1
            logging.exception("Fallo batch=%s: %s", batches_sent + 1, exc)

        if config.pause_seconds > 0:
            time.sleep(config.pause_seconds)

    return {
        "batches_sent": batches_sent,
        "transactions_sent": transactions_sent,
        "failures": failures,
        "total_latency_seconds": total_latency_seconds,
    }
