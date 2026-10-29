"""Compare DriftGuard's drift decisions with Evidently's on the same batches.

Run this from the separate Evidently environment (.venv-evidently).
"""
import sys
import warnings
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from driftguard import config
from driftguard.monitoring import detect_drift, load_reference

warnings.filterwarnings("ignore")
FEATURES = config.NUMERIC + config.CATEGORICAL
OUT = Path("reports/evidently")
OUT.mkdir(parents=True, exist_ok=True)

reference = load_reference()[FEATURES]
definition = DataDefinition(numerical_columns=config.NUMERIC,
                            categorical_columns=config.CATEGORICAL)

for path in sys.argv[1:]:
    current = pd.read_csv(path)[FEATURES]

    # ---- Evidently ----
    result = Report([DataDriftPreset()]).run(
        current_data=Dataset.from_pandas(current, data_definition=definition),
        reference_data=Dataset.from_pandas(reference, data_definition=definition))
    result.save_html(str(OUT / f"{Path(path).stem}.html"))
    evidently = {}
    for metric in result.dict()["metrics"]:
        cfg = metric["config"]
        if "column" in cfg:
            evidently[cfg["column"]] = (cfg["method"], metric["value"], cfg["threshold"])

    # ---- DriftGuard ----
    ours = {r.feature: r for r in detect_drift(reference, current)}

    print(f"\n=== {Path(path).name} ===")
    print(f"{'feature':16}{'DriftGuard':>12}   {'Evidently method':32}{'value':>8}  {'Evidently':>9}")
    for col in FEATURES:
        method, value, threshold = evidently[col]
        ev_drift = "DRIFT" if value >= threshold else "-"
        dg_drift = "DRIFT" if ours[col].drift_detected else "-"
        print(f"{col:16}{dg_drift:>12}   {method:32}{value:8.3f}  {ev_drift:>9}")

print(f"\nHTML reports saved in {OUT}/")