"""Save batch summaries and quality results to the database."""
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from sqlalchemy import (Boolean, Column, DateTime, Float, ForeignKey, Integer, MetaData,
                        String, Table, create_engine, delete, insert)

metadata = MetaData()

batches = Table(
    "batches", metadata,
    Column("batch_id", String(12), primary_key=True),
    Column("source_file", String, nullable=False),
    Column("ingested_at", DateTime(timezone=True), nullable=False),
    Column("row_count", Integer, nullable=False),
    Column("valid_count", Integer, nullable=False),
    Column("rejected_count", Integer, nullable=False),
    Column("status", String, nullable=False),
)

quality_results = Table(
    "quality_results", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("batch_id", String(12), ForeignKey("batches.batch_id"), nullable=False),
    Column("rule", String, nullable=False),
    Column("column_name", String, nullable=False),
    Column("failed_count", Integer, nullable=False),
    Column("failure_rate", Float, nullable=False),
    Column("severity", String, nullable=False),
    Column("message", String),
)

monitoring_results = Table(
    "monitoring_results", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("batch_id", String(12), ForeignKey("batches.batch_id"), nullable=False),
    Column("feature", String, nullable=False),
    Column("test", String, nullable=False),
    Column("statistic", Float, nullable=False),
    Column("p_value", Float, nullable=False),
    Column("threshold", Float, nullable=False),
    Column("drift_detected", Boolean, nullable=False),
    Column("n_reference", Integer, nullable=False),
    Column("n_current", Integer, nullable=False),
)

alerts = Table(
    "alerts", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("batch_id", String(12), ForeignKey("batches.batch_id"), nullable=False),
    Column("category", String, nullable=False),    # "quality" or "drift"
    Column("severity", String, nullable=False),    # "warning" or "critical"
    Column("message", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

model_evaluations = Table(
    "model_evaluations", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("batch_id", String(12), ForeignKey("batches.batch_id"), nullable=False),
    Column("model_version", String, nullable=False),
    Column("n_labelled", Integer, nullable=False),
    Column("accuracy", Float),
    Column("precision", Float),
    Column("recall", Float),
    Column("f1", Float),
    Column("roc_auc", Float),     # empty if the batch had only one class
    Column("pr_auc", Float),
)


def get_engine(url: str | None = None):
    """Connect using DATABASE_URL from your .env file (or a URL passed in, for tests)."""
    if url is None:
        load_dotenv()
        url = os.environ["DATABASE_URL"]
    return create_engine(url)


def create_tables(engine) -> None:
    metadata.create_all(engine)   # only creates tables that don't exist yet


def save_batch(engine, batch: dict, results: list, drift_results=(), alert_list=(),
               evaluation=None) -> None:
    """Replace any earlier record of this batch, so reruns never duplicate it."""
    batch_id = batch["batch_id"]
    now = datetime.now(timezone.utc)
    with engine.begin() as conn:   # one transaction: all of it is saved, or none
        # Delete children first: they point at the batch row (foreign keys)
        for table in (model_evaluations, alerts, monitoring_results, quality_results):
            conn.execute(delete(table).where(table.c.batch_id == batch_id))
        conn.execute(delete(batches).where(batches.c.batch_id == batch_id))

        conn.execute(insert(batches).values(**batch, ingested_at=now))
        rows = [{"batch_id": batch_id, "rule": r.rule, "column_name": r.column,
                 "failed_count": r.failed_count, "failure_rate": r.failure_rate,
                 "severity": r.severity, "message": r.message}
                for r in results if r.failed_count > 0]   # store failures only
        if rows:
            conn.execute(insert(quality_results), rows)
        rows = [{"batch_id": batch_id, **vars(d)} for d in drift_results]   # store every test
        if rows:
            conn.execute(insert(monitoring_results), rows)
        rows = [{"batch_id": batch_id, **a, "created_at": now} for a in alert_list]
        if rows:
            conn.execute(insert(alerts), rows)
        if evaluation:   # only when the batch came with labels
            conn.execute(insert(model_evaluations).values(batch_id=batch_id, **evaluation))