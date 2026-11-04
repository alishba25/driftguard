"""Run every scenario x seed through all four monitoring strategies.

    python experiments/run_experiments.py                                   # the original Phase 4 run
    python experiments/run_experiments.py experiments/configs/adult_20seeds.yaml

Writes one row per batch to <results_dir>/batch_results.csv. If the config lists
several models, each batch is scored by every model in the same pass: the
monitors don't depend on the model, so only the harm columns differ.
"""
import argparse
import io
import json
import os
import time
from pathlib import Path

import pandas as pd
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("config", nargs="?", default="experiments/configs/main.yaml")
args = parser.parse_args()
settings = yaml.safe_load(Path(args.config).read_text())
os.environ["DRIFTGUARD_DATASET"] = settings.get("dataset", "adult")   # before importing driftguard

from driftguard import config  # noqa: E402
from driftguard.evaluation import evaluate, load_baseline, load_model, to_model_frame  # noqa: E402
from driftguard.ingestion import read_batch  # noqa: E402
from driftguard.monitoring import detect_drift, load_reference  # noqa: E402
from driftguard.scenarios import make_run  # noqa: E402
from driftguard.schema import load_schema  # noqa: E402
from driftguard.validation import validate_batch  # noqa: E402

MODELS = settings.get("models", ["logreg"])
MULTI = "models" in settings          # original Phase 4 config keeps its original columns
RESULTS = Path(settings.get("results_dir", "experiments/results"))


def as_received(batch: pd.DataFrame) -> pd.DataFrame:
    """Round-trip through CSV text, so monitors see exactly what a real file would give."""
    buffer = io.StringIO()
    batch.to_csv(buffer, index=False)
    buffer.seek(0)
    return read_batch(buffer)


def main():
    pool = pd.read_csv(config.MONITOR_POOL_PATH).drop(columns=["label"], errors="ignore")
    reference = load_reference()
    schema = load_schema()
    models = {name: load_model(name) for name in MODELS}
    baselines = {name: load_baseline(name)["roc_auc"] for name in MODELS}

    rows = []
    for name, scenario in settings["scenarios"].items():
        for seed in settings["seeds"]:
            print(f"{name:14} seed {seed}")
            for batch, truth in make_run(pool, scenario, seed, settings):
                received = as_received(batch)

                # ---- Ground truth: model performance, using labels (monitors never see these) ----
                perf = {}
                for m in MODELS:
                    p = evaluate(models[m], received)
                    perf[m] = {"roc_auc": p["roc_auc"], "pr_auc": p["pr_auc"],
                               "harmful": p["roc_auc"] is not None and
                               p["roc_auc"] < baselines[m] - settings["harm_auc_drop"]}

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

                primary = perf[MODELS[0]]
                row = {
                    "scenario": name, "seed": seed, **truth,
                    "harmful": primary["harmful"],
                    "roc_auc": primary["roc_auc"], "pr_auc": primary["pr_auc"],
                    "rejection_rate": round(rejection_rate, 4),
                    "drifted_raw": ";".join(drifted_raw),
                    "drifted_valid": ";".join(drifted_valid),
                    # ---- The four strategies' decisions ----
                    "alert_none": False,
                    "alert_quality": quality_alert,
                    "alert_drift": len(drifted_raw) > 0,
                    "alert_combined": quality_alert or len(drifted_valid) > 0,
                    "monitor_ms": round(runtime_ms, 1),
                }
                if MULTI:   # one set of harm columns per model
                    for m in MODELS:
                        row[f"roc_auc_{m}"] = perf[m]["roc_auc"]
                        row[f"pr_auc_{m}"] = perf[m]["pr_auc"]
                        row[f"harmful_{m}"] = perf[m]["harmful"]
                rows.append(row)

    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS / "batch_results.csv", index=False)
    (RESULTS / "baselines.json").write_text(json.dumps(baselines, indent=2))
    print(f"\nWrote {len(rows)} batch results to {RESULTS / 'batch_results.csv'}")


if __name__ == "__main__":
    main()