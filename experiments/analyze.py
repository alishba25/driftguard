"""Confidence intervals and harm-threshold sensitivity for one experiment run.

    python experiments/analyze.py experiments/configs/adult_20seeds.yaml

95% confidence intervals come from a bootstrap over seeds: each seed is one
whole run (16 batches), so runs, not single batches, are resampled. Batches
inside a run share rows with other scenarios (the paired design), so treating
them as independent would make the intervals too narrow.

Writes ci_overall.csv, ci_by_scenario.csv and harm_sensitivity.csv to the
config's results_dir, and one alerts-vs-harm figure per model to research/figures/.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

N_BOOT = 2000
THRESHOLDS = [0.02, 0.03, 0.05]   # 0.03 is the pre-registered rule; the others are sensitivity checks
STRATEGIES = ["quality", "drift", "combined"]

parser = argparse.ArgumentParser()
parser.add_argument("config", nargs="?", default="experiments/configs/main.yaml")
args = parser.parse_args()
settings = yaml.safe_load(Path(args.config).read_text())
RESULTS = Path(settings.get("results_dir", "experiments/results"))
MODELS = settings.get("models", ["logreg"])
START = settings["shift_start"]
MAIN_T = settings["harm_auc_drop"]

df = pd.read_csv(RESULTS / "batch_results.csv")
if (RESULTS / "baselines.json").exists():
    baselines = json.loads((RESULTS / "baselines.json").read_text())
else:   # the original Phase 4 run, before baselines.json existed
    baselines = {"logreg": json.loads(Path("reports/baseline_metrics.json").read_text())["roc_auc"]}
SCENARIOS = list(dict.fromkeys(df["scenario"]))


def roc_col(m):
    return f"roc_auc_{m}" if f"roc_auc_{m}" in df.columns else "roc_auc"


# ---- 1. Count everything per seed (a seed = one independent run of every scenario) ----
def seed_counts(g: pd.DataFrame) -> dict:
    c = {}
    changed = g["input_changed"].astype(bool)
    after = g["batch_index"] >= START
    c["changed_n"], c["quiet_n"] = changed.sum(), (~changed).sum()
    harm = {}
    for m in MODELS:
        roc = g[roc_col(m)]
        for t in THRESHOLDS:
            harm[m, t] = (roc < baselines[m] - t).fillna(False)
            c[f"harmful_n|{m}|{t}"] = harm[m, t].sum()
    for s in STRATEGIES:
        a = g[f"alert_{s}"].astype(bool)
        c[f"alerts_n|{s}"] = a.sum()
        c[f"tp_changed|{s}"] = (a & changed).sum()
        c[f"fp_quiet|{s}"] = (a & ~changed).sum()
        for (m, t), h in harm.items():
            c[f"tp_harm|{s}|{m}|{t}"] = (a & h).sum()
    for sc in SCENARIOS:
        rows = (g["scenario"] == sc) & after
        c[f"after_n|{sc}"] = rows.sum()
        for s in STRATEGIES:
            c[f"alerts_after|{sc}|{s}"] = (g.loc[rows, f"alert_{s}"].astype(bool)).sum()
        for (m, t), h in harm.items():
            c[f"harm_after|{sc}|{m}|{t}"] = h[rows].sum()
        for m in MODELS:
            c[f"roc_sum_after|{sc}|{m}"] = g.loc[rows, roc_col(m)].sum()
    return c


per_seed = pd.DataFrame([seed_counts(g) for _, g in df.groupby("seed")]).astype(float)
keys = list(per_seed.columns)
counts = per_seed.to_numpy()

# ---- 2. Bootstrap: resample whole seeds with replacement, add their counts up ----
rng = np.random.default_rng(0)
n_seeds = len(counts)
samples = rng.integers(0, n_seeds, size=(N_BOOT, n_seeds))
boot = counts[samples].sum(axis=1)                 # N_BOOT x number of counts
point = counts.sum(axis=0)
idx = {k: i for i, k in enumerate(keys)}


def ratio(num, den, scale=1.0):
    """Estimate and 95% interval of num/den. Undefined (NaN) when den is 0."""
    with np.errstate(invalid="ignore", divide="ignore"):
        est = scale * point[idx[num]] / point[idx[den]] if point[idx[den]] else np.nan
        dist = scale * boot[:, idx[num]] / boot[:, idx[den]]
    dist = dist[np.isfinite(dist)]
    low, high = (np.percentile(dist, [2.5, 97.5]) if len(dist) else (np.nan, np.nan))
    return round(float(est), 4), round(float(low), 4), round(float(high), 4)


# ---- 3. Overall results per strategy, with intervals ----
overall = []
for s in STRATEGIES:
    def add(metric, num, den, scale=1.0, model="-"):
        est, low, high = ratio(num, den, scale)
        overall.append({"strategy": s, "model": model, "metric": metric,
                        "estimate": est, "ci_low": low, "ci_high": high})
    add("recall_changed", f"tp_changed|{s}", "changed_n")
    add("precision_changed", f"tp_changed|{s}", f"alerts_n|{s}")
    add("false_alerts_per_100", f"fp_quiet|{s}", "quiet_n", 100)
    for m in MODELS:
        add("precision_harmful", f"tp_harm|{s}|{m}|{MAIN_T}", f"alerts_n|{s}", model=m)
        add("recall_harmful", f"tp_harm|{s}|{m}|{MAIN_T}", f"harmful_n|{m}|{MAIN_T}", model=m)
overall = pd.DataFrame(overall)

# ---- 4. Per scenario: alert rate, harm share and mean ROC-AUC after the change ----
by_scenario = []
for sc in SCENARIOS:
    def add(metric, num, den, model="-"):
        est, low, high = ratio(num, den)
        by_scenario.append({"scenario": sc, "model": model, "metric": metric,
                            "estimate": est, "ci_low": low, "ci_high": high})
    for s in STRATEGIES:
        add(f"{s}_alert_rate", f"alerts_after|{sc}|{s}", f"after_n|{sc}")
    for m in MODELS:
        add("harmful_share", f"harm_after|{sc}|{m}|{MAIN_T}", f"after_n|{sc}", model=m)
        add("mean_roc_auc", f"roc_sum_after|{sc}|{m}", f"after_n|{sc}", model=m)
by_scenario = pd.DataFrame(by_scenario)

# ---- 5. Sensitivity: does the conclusion survive a stricter or looser harm rule? ----
sensitivity = []
for m in MODELS:
    for t in THRESHOLDS:
        row = {"model": m, "harm_drop": t,
               "harmful_batches": int(point[idx[f"harmful_n|{m}|{t}"]]),
               "drift_alerts_on_harmful": ratio(f"tp_harm|drift|{m}|{t}", "alerts_n|drift")[0],
               "harmful_batches_alerted": ratio(f"tp_harm|drift|{m}|{t}", f"harmful_n|{m}|{t}")[0]}
        for sc in SCENARIOS:
            row[f"harm_share_{sc}"] = ratio(f"harm_after|{sc}|{m}|{t}", f"after_n|{sc}")[0]
        sensitivity.append(row)
sensitivity = pd.DataFrame(sensitivity)

overall.to_csv(RESULTS / "ci_overall.csv", index=False)
by_scenario.to_csv(RESULTS / "ci_by_scenario.csv", index=False)
sensitivity.to_csv(RESULTS / "harm_sensitivity.csv", index=False)


def show(frame):
    f = frame.copy()
    if "estimate" in f:
        f["95% CI"] = "[" + f["ci_low"].map("{:.3f}".format) + ", " + f["ci_high"].map("{:.3f}".format) + "]"
        f = f.drop(columns=["ci_low", "ci_high"])
    print(f.to_string(index=False))


pd.set_option("display.width", 220)
print(f"=== {RESULTS.name}: {n_seeds} seeds, {len(df)} batches; bootstrap over seeds, {N_BOOT} resamples ===\n")
print("--- Overall, per strategy (harm rule: ROC-AUC more than "
      f"{MAIN_T} below each model's own baseline) ---")
show(overall[overall["strategy"] != "quality"])
print("\n--- Per scenario, after the change ---")
show(by_scenario[by_scenario["metric"].isin(["drift_alert_rate", "harmful_share", "mean_roc_auc"])])
print("\n--- Harm-threshold sensitivity (drift-only strategy) ---")
print(sensitivity.round(3).to_string(index=False))

# ---- 6. Figure: alerts vs harm per scenario, with 95% intervals, one per model ----
FIGS = Path("research/figures")
FIGS.mkdir(parents=True, exist_ok=True)
ORANGE, INK, GREY = "#eb6834", "#3d3d3a", "#b4b2a9"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": GREY, "axes.grid": True, "grid.color": "#e8e7e2",
                     "axes.axisbelow": True, "legend.frameon": False,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})
for m in MODELS:
    fig, ax = plt.subplots(figsize=(6.4, 0.45 * len(SCENARIOS) + 1.4))
    for k, sc in enumerate(SCENARIOS):
        y = len(SCENARIOS) - 1 - k
        a = by_scenario[(by_scenario.scenario == sc) & (by_scenario.metric == "drift_alert_rate")].iloc[0]
        h = by_scenario[(by_scenario.scenario == sc) & (by_scenario.metric == "harmful_share")
                        & (by_scenario.model == m)].iloc[0]
        ax.plot([a.estimate, h.estimate], [y, y], color=GREY, linewidth=2, zorder=1)
        for r, colour, dy in ((a, ORANGE, 0.12), (h, INK, -0.12)):
            ax.errorbar(r.estimate, y + dy, xerr=[[r.estimate - r.ci_low], [r.ci_high - r.estimate]],
                        fmt="o", color=colour, ecolor=colour, elinewidth=1.2, capsize=3,
                        markersize=7, markeredgecolor="white", zorder=3)
    ax.errorbar([], [], fmt="o", color=ORANGE, label="Drift alert raised")
    ax.errorbar([], [], fmt="o", color=INK, label=f"Model actually harmed ({m})")
    ax.set_yticks(range(len(SCENARIOS)), SCENARIOS[::-1])
    ax.set_xlim(-0.03, 1.03)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel(f"Share of batches after the change point, with 95% intervals ({n_seeds} seeds)")
    ax.grid(axis="y", visible=False)
    ax.legend(ncols=2, loc="upper left", bbox_to_anchor=(0, 1.0 + 0.6 / len(SCENARIOS)))
    out = FIGS / f"{RESULTS.name}_alerts_vs_harm_{m}.png"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Wrote {out} and .pdf")