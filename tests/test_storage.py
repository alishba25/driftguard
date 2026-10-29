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



def test_drift_results_and_alerts_are_saved(tmp_path):
    from driftguard.monitoring import DriftResult
    from driftguard.storage import alerts, monitoring_results

    engine = get_engine(f"sqlite:///{tmp_path / 'test.db'}")
    create_tables(engine)
    batch = {"batch_id": "def456", "source_file": "y.csv", "row_count": 10,
             "valid_count": 10, "rejected_count": 0, "status": "passed"}
    drift = [DriftResult("age", "ks", 0.2, 0.0001, 0.004, True, 100, 10)]
    alert = [{"category": "drift", "severity": "warning", "message": "age changed"}]

    save_batch(engine, batch, [], drift, alert)
    save_batch(engine, batch, [], drift, alert)   # rerun

    with engine.connect() as conn:
        assert conn.execute(select(func.count()).select_from(monitoring_results)).scalar() == 1
        assert conn.execute(select(func.count()).select_from(alerts)).scalar() == 1       