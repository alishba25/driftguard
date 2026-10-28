from sqlalchemy import func, select

from driftguard.storage import batches, create_tables, get_engine, quality_results, save_batch
from driftguard.validation import RuleResult


def test_rerunning_a_batch_does_not_duplicate_it(tmp_path):
    # SQLite in a temporary file: tests run without PostgreSQL
    engine = get_engine(f"sqlite:///{tmp_path / 'test.db'}")
    create_tables(engine)
    batch = {"batch_id": "abc123", "source_file": "x.csv", "row_count": 10,
             "valid_count": 9, "rejected_count": 1, "status": "passed_with_rejections"}
    results = [RuleResult("range", "age", 1, 0.1, "error", "age out of range")]

    save_batch(engine, batch, results)
    save_batch(engine, batch, results)   # run the same batch again

    with engine.connect() as conn:
        assert conn.execute(select(func.count()).select_from(batches)).scalar() == 1
        assert conn.execute(select(func.count()).select_from(quality_results)).scalar() == 1