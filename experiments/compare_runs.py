"""Compare two runs of the experiment, row by row.

    python experiments/compare_runs.py <first.csv> <second.csv>

Reports (1) any batch where a strategy made a different alert decision, and
(2) the largest difference in each numeric column. Timings are ignored.
"""
import sys

import pandas as pd

KEY = ["scenario", "seed", "batch_index"]
DECISIONS = ["alert_quality", "alert_drift", "alert_combined", "harmful"]

a = pd.read_csv(sys.argv[1]).drop(columns="monitor_ms").sort_values(KEY).reset_index(drop=True)
b = pd.read_csv(sys.argv[2]).drop(columns="monitor_ms").sort_values(KEY).reset_index(drop=True)

if len(a) != len(b) or not a[KEY].equals(b[KEY]):
    sys.exit(f"Different batches: {len(a)} rows vs {len(b)} rows. Same config in both runs?")

if a.equals(b):
    print(f"IDENTICAL: all {len(a)} batches match exactly (timings excluded).")
    sys.exit(0)

# ---- 1. Decisions: these are what the paper's results are built from ----
changed = a[DECISIONS] != b[DECISIONS]
n_changed = int(changed.any(axis=1).sum())
print(f"Batches with a different decision: {n_changed} of {len(a)}")
for col in DECISIONS:
    if changed[col].any():
        rows = a.loc[changed[col], KEY]
        print(f"  {col}: {len(rows)} batch(es), e.g. {rows.iloc[0].to_dict()}")

# ---- 2. Numbers: tiny differences usually come from library versions ----
print("\nLargest difference per numeric column:")
for col in a.select_dtypes("number").columns.difference(KEY):
    diff = (a[col] - b[col]).abs().max()
    print(f"  {col:16} {diff:.2e}")