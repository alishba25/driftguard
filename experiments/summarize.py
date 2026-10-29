"""Turn batch_results.csv into the results tables for the paper.

    python experiments/summarize.py
"""
from pathlib import Path

import pandas as pd
import yaml

from driftguard.scoring import alert_scores, detection_delay, false_alerts_per_100

RESULTS = Path("experiments/results")
STRATEGIES = ["none", "quality", "drift", "combined"]
settings = yaml.safe_load(Path("experiments/configs/main.yaml").read_text())
start = settings["shift_start"]
df = pd.read_csv(RESULTS / "batch_results.csv")

# ---- Table 1: each strategy against both kinds of ground truth ----
overall = []
for s in STRATEGIES:
    alerts = df[f"alert_{s}"]
    changed = alert_scores(alerts, df["input_changed"])
    harm = alert_scores(alerts, df["harmful"])
    overall.append({
        "strategy": s,
        "precision_changed": changed["precision"], "recall_changed": changed["recall"],
        "f1_changed": changed["f1"],
        "false_alerts_per_100": false_alerts_per_100(alerts, df["input_changed"]),
        "precision_harmful": harm["precision"], "recall_harmful": harm["recall"],
    })
overall = pd.DataFrame(overall)

# ---- Table 2: per scenario, share of changed batches alerted + model impact ----
by_scenario = []
for name, g in df.groupby("scenario", sort=False):
    after = g[g["batch_index"] >= start]
    inputs_change = bool(after["input_changed"].any())   # False for A (stable) and F (labels only)
    row = {"scenario": name,
           "mean_roc_auc_after": after["roc_auc"].mean(),
           "harmful_share_after": after["harmful"].mean()}
    for s in STRATEGIES:
        row[f"{s}_alert_rate_after"] = after[f"alert_{s}"].mean()
        delays = [detection_delay(r[f"alert_{s}"], r["batch_index"], start)
                  for _, r in g.groupby("seed")]
        found = [d for d in delays if d is not None]
        # Delay only means something when the inputs actually changed
        row[f"{s}_delay"] = (sum(found) / len(found) if found else None) if inputs_change else None
        row[f"{s}_missed_runs"] = delays.count(None) if inputs_change else None
    by_scenario.append(row)
by_scenario = pd.DataFrame(by_scenario)

# ---- Table 3: what each strategy blamed, on corrupted batches ----
corrupted = df[(df["scenario"] == "B_corruption") & (df["batch_index"] >= start)]
blame = pd.DataFrame([
    {"strategy": "drift-only", "said": "distribution changed in: " + f,
     "batches": n} for f, n in corrupted["drifted_raw"].fillna("(none)").value_counts().items()
] + [
    {"strategy": "combined", "said": "quality problem", "batches": int(corrupted["alert_quality"].sum())},
    {"strategy": "combined", "said": "also drift in valid rows",
     "batches": int((corrupted["drifted_valid"].fillna("") != "").sum())},
])

overall.to_csv(RESULTS / "summary_overall.csv", index=False)
blame.to_csv(RESULTS / "corruption_blame.csv", index=False)
by_scenario.to_csv(RESULTS / "summary_by_scenario.csv", index=False)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
print("=== Overall (all scenarios, all seeds) ===")
print(overall.round(3).to_string(index=False))
print(f"\nBaseline ROC-AUC rule: harmful = more than {settings['harm_auc_drop']} below baseline")
print("\n=== Alert rate on changed batches (batches 8-15) ===")
cols = ["scenario"] + [f"{s}_alert_rate_after" for s in STRATEGIES[1:]]
print(by_scenario[cols].round(2).to_string(index=False))
print("\n=== Detection delay in batches (missed runs out of 5) ===")
for _, r in by_scenario.iterrows():
    if pd.isna(r["drift_missed_runs"]):
        print(f"{r['scenario']:14} n/a (inputs never change)")
        continue
    parts = []
    for s in STRATEGIES[1:]:
        d = "-" if pd.isna(r[f"{s}_delay"]) else f"{r[f'{s}_delay']:.1f}"
        parts.append(f"{s}: {d} ({int(r[f'{s}_missed_runs'])} missed)")
    print(f"{r['scenario']:14} " + " | ".join(parts))
print("\n=== Model impact after the change ===")
print(by_scenario[["scenario", "mean_roc_auc_after", "harmful_share_after"]].round(3).to_string(index=False))
print(f"\n=== What each strategy blamed, on the {len(corrupted)} corrupted batches ===")
print(blame.to_string(index=False))