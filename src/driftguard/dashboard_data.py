"""Read everything the dashboard shows. No Streamlit here, so it can be tested."""
import json
from pathlib import Path

import pandas as pd

from driftguard import config

RESULTS = Path("experiments/results")


def short_name(path: str) -> str:
    """'C:\\...\\data\\batches\\batch_001.csv' -> 'batch_001' (works for / and \\)."""
    return Path(path.replace("\\", "/")).stem


def load_batches(engine) -> pd.DataFrame:
    """One row per batch, oldest first, with its model score and alert count."""
    df = pd.read_sql("""
        SELECT b.batch_id, b.source_file, b.ingested_at, b.row_count,
               b.valid_count, b.rejected_count, b.status,
               m.roc_auc, m.pr_auc,
               (SELECT count(*) FROM alerts a WHERE a.batch_id = b.batch_id) AS n_alerts
        FROM batches b
        LEFT JOIN model_evaluations m ON m.batch_id = b.batch_id
        ORDER BY b.ingested_at
    """, engine)
    df["batch"] = df["source_file"].map(short_name)
    df["rejection_rate"] = df["rejected_count"] / df["row_count"]
    return df


def load_quality(engine) -> pd.DataFrame:
    df = pd.read_sql("""
        SELECT b.source_file, q.rule, q.column_name, q.failed_count, q.failure_rate, q.message
        FROM quality_results q JOIN batches b USING (batch_id)
        ORDER BY b.ingested_at
    """, engine)
    df["batch"] = df["source_file"].map(short_name)
    return df


def load_drift(engine) -> pd.DataFrame:
    df = pd.read_sql("""
        SELECT b.source_file, b.ingested_at, d.feature, d.test, d.statistic,
               d.p_value, d.threshold, d.drift_detected
        FROM monitoring_results d JOIN batches b USING (batch_id)
        ORDER BY b.ingested_at
    """, engine)
    df["batch"] = df["source_file"].map(short_name)
    return df


def load_alerts(engine) -> pd.DataFrame:
    df = pd.read_sql("""
        SELECT a.created_at, b.source_file, a.category, a.severity, a.message
        FROM alerts a JOIN batches b USING (batch_id)
        ORDER BY a.created_at DESC
    """, engine)
    df["batch"] = df["source_file"].map(short_name)
    df["next_step"] = df["category"].map(NEXT_STEPS)
    return df


# Advisory only: the dashboard suggests, a person decides (proposal, section 7)
NEXT_STEPS = {
    "quality": "Open data/rejected/ for this batch and check the upstream source",
    "drift": "Compare the drifted features with the reference; check model performance once labels arrive",
}


def load_baseline_auc() -> float:
    with open(config.BASELINE_METRICS_PATH) as f:
        return json.load(f)["roc_auc"]


def load_experiment_results():
    """The Phase 4 summary tables, or None if the experiment hasn't been run."""
    files = ["summary_overall.csv", "summary_by_scenario.csv", "corruption_blame.csv"]
    if not all((RESULTS / f).exists() for f in files):
        return None
    return [pd.read_csv(RESULTS / f) for f in files]