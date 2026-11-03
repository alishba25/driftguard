"""Make the paper's figures from the experiment results.

    python experiments/make_figures.py

Reads experiments/results/*.csv and writes PNG + PDF files to research/figures/.
Every number in every figure comes from the CSVs, so figures can always be rebuilt.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # draw to files, no window
import matplotlib.pyplot as plt
import pandas as pd
import yaml

RESULTS = Path("experiments/results")
OUT = Path("research/figures")
OUT.mkdir(parents=True, exist_ok=True)

settings = yaml.safe_load(Path("experiments/configs/main.yaml").read_text())
START = settings["shift_start"]
baseline = json.loads(Path("reports/baseline_metrics.json").read_text())["roc_auc"]
harm_line = baseline - settings["harm_auc_drop"]

batches = pd.read_csv(RESULTS / "batch_results.csv")
by_scenario = pd.read_csv(RESULTS / "summary_by_scenario.csv")
SCENARIOS = list(by_scenario["scenario"])
LABELS = [s.replace("_", "\n", 1) for s in SCENARIOS]       # "C_shift_low" -> "C\nshift_low"

# Colour-blind-checked palette; one colour per strategy, used the same in every figure
BLUE, ORANGE, AQUA, GREY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#b4b2a9", "#3d3d3a"
STRATEGY_COLOURS = {"quality": BLUE, "drift": ORANGE, "combined": AQUA}

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": GREY, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.grid": True, "grid.color": "#e8e7e2", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"Wrote {OUT / name}.png and .pdf")


# ---- Figure 1: alert rate after the change, by strategy and scenario ----
fig, ax = plt.subplots(figsize=(8, 3.6))
width = 0.26
for k, (strategy, colour) in enumerate(STRATEGY_COLOURS.items()):
    x = [i + (k - 1) * width for i in range(len(SCENARIOS))]
    ax.bar(x, by_scenario[f"{strategy}_alert_rate_after"], width=width - 0.03,
           color=colour, label=strategy)
ax.set_xticks(range(len(SCENARIOS)), LABELS)
ax.set_ylabel("Share of changed batches\nthat raised an alert")
ax.set_ylim(0, 1.05)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.grid(axis="x", visible=False)
ax.legend(title="Strategy", ncols=3, loc="upper left", bbox_to_anchor=(0, 1.18))
save(fig, "fig1_alert_rate_by_strategy")

# ---- Figure 2: alerts vs actual harm, per scenario (H4) ----
fig, ax = plt.subplots(figsize=(6.4, 3.8))
y = range(len(SCENARIOS))[::-1]
alert = by_scenario["drift_alert_rate_after"]
harm = by_scenario["harmful_share_after"]
for yi, a, h in zip(y, alert, harm):
    ax.plot([a, h], [yi, yi], color=GREY, linewidth=2, zorder=1)
ax.scatter(alert, list(y), s=60, color=ORANGE, edgecolor="white", linewidth=1.5,
           label="Drift alert raised", zorder=2)
ax.scatter(harm, list(y), s=60, color=INK, edgecolor="white", linewidth=1.5,
           label="Model actually harmed", zorder=3)
ax.set_yticks(list(y), SCENARIOS)
ax.set_xlim(-0.03, 1.03)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.set_xlabel("Share of changed batches (batches 8-15, 5 seeds)")
ax.grid(axis="y", visible=False)
ax.legend(ncols=2, loc="upper left", bbox_to_anchor=(0, 1.13))
save(fig, "fig2_alerts_vs_harm")

# ---- Figure 3: model ROC-AUC across the run, per scenario ----
fig, ax = plt.subplots(figsize=(7.2, 3.8))
mean_auc = batches.groupby(["scenario", "batch_index"])["roc_auc"].mean().unstack(0)
highlight = {"D_sudden": ORANGE, "F_concept": BLUE}
for name in SCENARIOS:   # every scenario in grey; batches 0-7 are identical across them
    ax.plot(mean_auc.index, mean_auc[name], color=GREY, linewidth=1.2, zorder=2)
for name, colour in highlight.items():   # the two harmful scenarios, from the change on
    after = mean_auc.loc[START - 1:, name]
    ax.plot(after.index, after, color=colour, linewidth=2, zorder=3)
    ax.annotate(name, (mean_auc.index[-1], mean_auc[name].iloc[-1]),
                xytext=(6, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
ax.plot([], [], color=GREY, linewidth=1.2, label="other scenarios (A, B, C, E)")
ax.axhline(baseline, color=INK, linewidth=1, linestyle="--")
ax.axhline(harm_line, color=INK, linewidth=1, linestyle=":")
ax.annotate(f"baseline {baseline:.3f}", (0, baseline), xytext=(0, 4),
            textcoords="offset points", fontsize=8, color=INK)
ax.annotate(f"harm threshold {harm_line:.3f}", (0, harm_line), xytext=(0, 4),
            textcoords="offset points", fontsize=8, color=INK)
ax.axvline(START - 0.5, color=GREY, linewidth=1)
ax.annotate("change starts", (START - 0.5, ax.get_ylim()[0]), xytext=(4, 4),
            textcoords="offset points", fontsize=8, color=INK)
ax.set_xlabel("Batch (in processing order)")
ax.set_ylabel("ROC-AUC (mean of 5 seeds)")
ax.set_xticks(range(0, settings["n_batches"], 2))
ax.legend(loc="lower left")
save(fig, "fig3_roc_auc_over_batches")

# ---- Figure 4: how quickly drift alerts appear after the change ----
fig, ax = plt.subplots(figsize=(7.2, 3.4))
rate = batches.groupby(["scenario", "batch_index"])["alert_drift"].mean().unstack(0)
# In these results B and every C give exactly the same line as D, so D stands in for them
shown = {"D_sudden": ORANGE, "E_gradual": BLUE, "F_concept": INK}
for name, colour in shown.items():
    ax.plot(rate.index, rate[name], color=colour, linewidth=2, marker="o", markersize=4,
            label=name)
ax.axvline(START - 0.5, color=GREY, linewidth=1)
ax.set_xlabel("Batch (in processing order)")
ax.set_ylabel("Share of the 5 runs\nwith a drift alert")
ax.set_ylim(-0.03, 1.05)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.set_xticks(range(0, settings["n_batches"], 2))
ax.legend(ncols=3, loc="upper left", bbox_to_anchor=(0, 1.16))
save(fig, "fig4_detection_over_batches")