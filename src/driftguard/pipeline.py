"""Run one batch through the pipeline:  python -m driftguard.pipeline <file.csv>"""
import logging
import sys
from pathlib import Path

from driftguard import config
from driftguard.alerts import make_alerts
from driftguard.evaluation import evaluate, load_model
from driftguard.ingestion import make_batch_id, read_batch
from driftguard.monitoring import detect_drift, load_reference
from driftguard.schema import load_schema
from driftguard.storage import create_tables, get_engine, save_batch
from driftguard.validation import validate_batch

log = logging.getLogger("driftguard")


def setup_logging() -> None:
    config.LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(config.LOG_PATH), logging.StreamHandler()],
    )


def run_batch(path: Path, engine, schema: dict, reference, model) -> dict:
    batch_id = make_batch_id(path)
    log.info("Batch %s: reading %s", batch_id, path)
    df = read_batch(path)

    valid, rejected, results = validate_batch(df, schema)

    if any(r.severity == "critical" for r in results):
        status = "failed"
    elif len(rejected) > 0:
        status = "passed_with_rejections"
    else:
        status = "passed"

    # Keep rejected rows in a file for inspection (not in the database)
    if len(rejected) > 0:
        config.REJECTED_DIR.mkdir(parents=True, exist_ok=True)
        rejected.to_csv(config.REJECTED_DIR / f"{batch_id}.csv", index=False)

    batch = {"batch_id": batch_id, "source_file": str(path), "row_count": len(df),
             "valid_count": len(valid), "rejected_count": len(rejected), "status": status}

    # Drift is checked on valid rows only: broken rows were already caught above
    drift_results = detect_drift(reference, valid) if len(valid) > 0 else []
    alert_list = make_alerts(batch, results, drift_results)

        # Model performance, only if this batch has labels (monitors above never used them)
    evaluation = None
    if config.TARGET in valid.columns and len(valid) > 0:
        evaluation = {"model_version": config.MODEL_VERSION, **evaluate(model, valid)}
        log.info("Batch %s: model ROC-AUC %s on %d labelled rows", batch_id,
                 None if evaluation["roc_auc"] is None else round(evaluation["roc_auc"], 3),
                 evaluation["n_labelled"])

    save_batch(engine, batch, results, drift_results, alert_list, evaluation)

    for r in results:
        if r.failed_count > 0:
            log.warning("Batch %s: %s on %s failed for %d rows (%.1f%%)",
                        batch_id, r.rule, r.column, r.failed_count, 100 * r.failure_rate)
    log.info("Batch %s: %d rows, %d valid, %d rejected -> %s",
             batch_id, len(df), len(valid), len(rejected), status)
    for a in alert_list:
        log.warning("Batch %s: ALERT [%s/%s] %s", batch_id, a["category"], a["severity"], a["message"])
    return batch


def main():
    setup_logging()
    if len(sys.argv) < 2:
        print("Usage: python -m driftguard.pipeline <batch.csv> [more.csv ...]")
        sys.exit(1)
    engine = get_engine()
    create_tables(engine)
    schema = load_schema()
    reference = load_reference()   # loaded once, reused for every batch
    model = load_model()
    for path in sys.argv[1:]:
        run_batch(Path(path), engine, schema, reference, model)


if __name__ == "__main__":
    main()