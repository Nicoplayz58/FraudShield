from __future__ import annotations

from sqlalchemy import BigInteger, Boolean, Column, Date, DateTime, ForeignKey, Index, Integer, MetaData, Numeric, Table, Text, UniqueConstraint, func
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.engine import Engine


metadata = MetaData()

PREDICTION_EVENT_FIELDS = [
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
    "unix_time",
    "merch_lat",
    "merch_long",
    "merch_zipcode",
]

PREDICTION_EVENT_COLUMN_TYPES = {
    "trans_date_trans_time": "TIMESTAMP WITHOUT TIME ZONE",
    "cc_num": "BIGINT",
    "merchant": "TEXT",
    "category": "TEXT",
    "amt": "DOUBLE PRECISION",
    "first": "TEXT",
    "last": "TEXT",
    "gender": "TEXT",
    "street": "TEXT",
    "city": "TEXT",
    "state": "TEXT",
    "zip": "INTEGER",
    "lat": "DOUBLE PRECISION",
    "long": "DOUBLE PRECISION",
    "city_pop": "INTEGER",
    "job": "TEXT",
    "dob": "DATE",
    "unix_time": "BIGINT",
    "merch_lat": "DOUBLE PRECISION",
    "merch_long": "DOUBLE PRECISION",
    "merch_zipcode": "DOUBLE PRECISION",
}

PREDICTION_EVENT_BACKFILL_SOURCE_COLUMNS = {
    "long": "longitude",
}

PREDICTION_EVENT_BACKFILL_SOURCE_EXPRESSIONS = {
    "trans_date_trans_time": "CAST(t.trans_date_trans_time AS TIMESTAMP WITHOUT TIME ZONE)",
    "cc_num": "CAST(t.cc_num AS BIGINT)",
    "merchant": "t.merchant",
    "category": "t.category",
    "amt": "CAST(t.amt AS DOUBLE PRECISION)",
    "first": "t.first",
    "last": "t.last",
    "gender": "t.gender",
    "street": "t.street",
    "city": "t.city",
    "state": "t.state",
    "zip": "CAST(t.zip AS INTEGER)",
    "lat": "CAST(t.lat AS DOUBLE PRECISION)",
    "long": "CAST(t.longitude AS DOUBLE PRECISION)",
    "city_pop": "CAST(t.city_pop AS INTEGER)",
    "job": "t.job",
    "dob": "CAST(t.dob AS DATE)",
    "unix_time": "CAST(t.unix_time AS BIGINT)",
    "merch_lat": "CAST(t.merch_lat AS DOUBLE PRECISION)",
    "merch_long": "CAST(t.merch_long AS DOUBLE PRECISION)",
    "merch_zipcode": "CAST(t.merch_zipcode AS DOUBLE PRECISION)",
}

prediction_batches = Table(
    "prediction_batches",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column("batch_hash", Text, nullable=False, unique=True),
    Column("model", Text, nullable=False),
    Column("threshold", Numeric(10, 4), nullable=False),
    Column("total_transactions", Integer, nullable=False),
    Column("created_at", DateTime(timezone=False), nullable=False, server_default=func.now()),
)

predictions = Table(
    "predictions",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column("batch_id", UUID(as_uuid=True), ForeignKey("prediction_batches.id", ondelete="CASCADE"), nullable=False),
    Column("trans_num", Text, nullable=False),
    Column("trans_date_trans_time", DateTime(timezone=False), nullable=True),
    Column("cc_num", BigInteger, nullable=True),
    Column("merchant", Text, nullable=True),
    Column("category", Text, nullable=True),
    Column("amt", Numeric(12, 6), nullable=True),
    Column("first", Text, nullable=True),
    Column("last", Text, nullable=True),
    Column("gender", Text, nullable=True),
    Column("street", Text, nullable=True),
    Column("city", Text, nullable=True),
    Column("state", Text, nullable=True),
    Column("zip", Integer, nullable=True),
    Column("lat", Numeric(12, 6), nullable=True),
    Column("long", Numeric(12, 6), nullable=True),
    Column("city_pop", Integer, nullable=True),
    Column("job", Text, nullable=True),
    Column("dob", Date, nullable=True),
    Column("unix_time", BigInteger, nullable=True),
    Column("merch_lat", Numeric(12, 6), nullable=True),
    Column("merch_long", Numeric(12, 6), nullable=True),
    Column("merch_zipcode", Numeric(12, 6), nullable=True),
    Column("risk_score", Numeric(12, 6), nullable=False),
    Column("prediction", Boolean, nullable=False),
    Column("created_at", DateTime(timezone=False), nullable=False, server_default=func.now()),
    UniqueConstraint("batch_id", "trans_num", name="uq_predictions_batch_trans_num"),
)

Index("ix_predictions_trans_num", predictions.c.trans_num)
Index("ix_predictions_created_at", predictions.c.created_at)
Index("ix_prediction_batches_created_at", prediction_batches.c.created_at)


def ensure_prediction_schema(engine: Engine) -> None:
    metadata.create_all(engine, tables=[prediction_batches, predictions])
    _ensure_prediction_payload_columns(engine)


def _ensure_prediction_payload_columns(engine: Engine) -> None:
    alter_statements = [
        f"ALTER TABLE predictions ADD COLUMN IF NOT EXISTS {column_name} {PREDICTION_EVENT_COLUMN_TYPES[column_name]}"
        for column_name in PREDICTION_EVENT_FIELDS
    ]
    assignment_sql = ", ".join(
        f"{column_name} = {PREDICTION_EVENT_BACKFILL_SOURCE_EXPRESSIONS[column_name]}"
        for column_name in PREDICTION_EVENT_FIELDS
    )

    update_statement = f"""
        UPDATE predictions p
        SET {assignment_sql}
        FROM transactions t
        WHERE p.trans_num = t.trans_num
    """

    with engine.begin() as connection:
        for statement in alter_statements:
            connection.execute(text(statement))
        connection.execute(text(update_statement))


def ensure_dashboard_indexes(engine: Engine) -> None:
    statements = [
        "CREATE INDEX IF NOT EXISTS ix_transactions_merchant ON transactions (merchant)",
        "CREATE INDEX IF NOT EXISTS ix_transactions_category ON transactions (category)",
        "CREATE INDEX IF NOT EXISTS ix_transactions_trans_num ON transactions (trans_num)",
        "CREATE INDEX IF NOT EXISTS ix_predictions_trans_num_created_at ON predictions (trans_num, created_at DESC)",
        "CREATE INDEX IF NOT EXISTS ix_predictions_trans_date_trans_time ON predictions (trans_date_trans_time)",
        "CREATE INDEX IF NOT EXISTS ix_predictions_merchant ON predictions (merchant)",
        "CREATE INDEX IF NOT EXISTS ix_predictions_category ON predictions (category)",
        "CREATE INDEX IF NOT EXISTS ix_predictions_amt ON predictions (amt)",
    ]

    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))
