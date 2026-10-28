"""Save batch summaries and quality results to the database."""
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from sqlalchemy import (Column, DateTime, Float, ForeignKey, Integer, MetaData,
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


def get_engine(url: str | None = None):
    """Connect using DATABASE_URL from your .env file (or a URL passed in, for tests)."""
    if url is None:
        load_dotenv()
        url = os.environ["DATABASE_URL"]
    return create_engine(url)


def create_tables(engine) -> None:
    metadata.create_all(engine)   # only creates tables that don't exist yet


def save_batch(engine, batch: dict, results: list) -> None:
    """Replace any earlier record of this batch, so reruns never duplicate it."""
    with engine.begin() as conn:   # one transaction: all of it is saved, or none
        conn.execute(delete(quality_results).where(quality_results.c.batch_id == batch["batch_id"]))
        conn.execute(delete(batches).where(batches.c.batch_id == batch["batch_id"]))
        conn.execute(insert(batches).values(**batch, ingested_at=datetime.now(timezone.utc)))
        rows = [{"batch_id": batch["batch_id"], "rule": r.rule, "column_name": r.column,
                 "failed_count": r.failed_count, "failure_rate": r.failure_rate,
                 "severity": r.severity, "message": r.message}
                for r in results if r.failed_count > 0]   # store failures only
        if rows:
            conn.execute(insert(quality_results), rows)