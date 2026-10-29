"""Run every scenario x seed through all four monitoring strategies.

    python experiments/run_experiments.py

Writes one row per batch to experiments/results/batch_results.csv.
"""
import io
import time
from pathlib import Path

import pandas as pd
import yaml

from driftguard import config
from driftguard.evaluation import evaluate, load_baseline, load_model, to_model_frame
from driftguard.ingestion import read_batch
from driftguard.monitoring import detect_drift, load_reference
from driftguard.scenarios import make_run
from driftguard.schema import load_schema
from driftguard.validation import validate_batch

CONFIG = Path("experiments/configs/main.yaml")
OUT = Path("experiments/results/batch_results.csv")


def as_received(batch: pd.DataFrame) -> pd.DataFrame:
    """Round-trip through CSV text, so monitors see exactly what a real file would give."""
    buffer = io.StringIO()
    batch.to_csv(buffer, index=False)
    buffer.seek(0)
    return read_batch(buffer)


def main():
    settings = yaml.safe_load(CONFIG.read_text())
    pool = pd.read_csv("data/splits/monitor_pool.csv").drop(columns=["label"])
    reference = load_reference()
    schema = load_schema()
    model = load_model()
    baseline_auc = load_baseline()["roc_auc"]

    rows = []
    for name, scenario in settings["scenarios"].items():
        for seed in settings["seeds"]:
            print(f"{name:14} seed {seed}")
            for batch, truth in make_run(pool, scenario, seed, settings):
                # ---- Ground truth: labels and model performance (monitors never see these) ----
                received = as_received(batch)
                perf = evaluate(model, received)
                harmful = perf["roc_auc"] is not None and \
                    perf["roc_auc"] < baseline_auc - settings["harm_auc_drop"]

                # ---- What the monitors see: the batch WITHOUT its label column ----
                inputs = received.drop(columns=[config.TARGET])
                t0 = time.perf_counter()
                valid, rejected, quality = validate_batch(inputs, schema)
                failed = any(r.severity == "critical" for r in quality)
                rejection_rate = len(rejected) / len(inputs)
                quality_alert = failed or rejection_rate > config.QUALITY_ALERT_RATE

                drift_raw = detect_drift(reference, to_model_frame(inputs))   # no quality gate
                drift_valid = detect_drift(reference, valid) if len(valid) else []
                runtime_ms = 1000 * (time.perf_counter() - t0)

                drifted_raw = [d.feature for d in drift_raw if d.drift_detected]
                drifted_valid = [d.feature for d in drift_valid if d.drift_detected]

                rows.append({
                    "scenario": name, "seed": seed, **truth,
                    "harmful": harmful,
                    "roc_auc": perf["roc_auc"], "pr_auc": perf["pr_auc"],
                    "rejection_rate": round(rejection_rate, 4),
                    "drifted_raw": ";".join(drifted_raw),
                    "drifted_valid": ";".join(drifted_valid),
                    # ---- The four strategies' decisions ----
                    "alert_none": False,
                    "alert_quality": quality_alert,
                    "alert_drift": len(drifted_raw) > 0,
                    "alert_combined": quality_alert or len(drifted_valid) > 0,
                    "monitor_ms": round(runtime_ms, 1),
                })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nWrote {len(rows)} batch results to {OUT}")


if __name__ == "__main__":
    main()