from __future__ import annotations

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Index, Integer, MetaData, Numeric, Table, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.engine import Engine


metadata = MetaData()

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
