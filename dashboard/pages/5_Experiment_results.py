import plotly.express as px
import streamlit as st

from common import page_header
from driftguard.dashboard_data import load_experiment_results

page_header("Experiment results (Phase 4)",
            "Four monitoring strategies on scenarios A-F, 5 seeds each. "
            "Read from experiments/results/, not the database.")

results = load_experiment_results()
if results is None:
    st.info("No results yet. Run `python experiments/run_experiments.py` "
            "then `python experiments/summarize.py`.")
    st.stop()
overall, by_scenario, blame = results

st.subheader("Each strategy against both kinds of truth")
st.dataframe(overall.round(3), hide_index=True, use_container_width=True)

st.subheader("Share of changed batches that raised an alert")
strategies = ["quality", "drift", "combined"]
long = by_scenario.melt(id_vars="scenario",
                        value_vars=[f"{s}_alert_rate_after" for s in strategies],
                        var_name="strategy", value_name="alert_rate")
long["strategy"] = long["strategy"].str.replace("_alert_rate_after", "")
colours = {"quality": "#9AA5B1", "drift": "#6C9BE8", "combined": "#1F3A93"}
fig = px.bar(long, x="scenario", y="alert_rate", color="strategy", barmode="group",
             color_discrete_map=colours,
             labels={"alert_rate": "Alert rate after the change", "scenario": "Scenario"})
fig.update_yaxes(tickformat=".0%")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Did the change actually hurt the model?")
fig = px.bar(by_scenario, x="scenario", y="harmful_share_after",
             labels={"harmful_share_after": "Share of changed batches that were harmful",
                     "scenario": "Scenario"})
fig.update_yaxes(tickformat=".0%", range=[0, 1])
st.plotly_chart(fig, use_container_width=True)
st.caption("Compare with the chart above: alerts and harm don't line up (hypothesis H4).")

st.subheader("What each strategy blamed, on corrupted batches")
st.dataframe(blame, hide_index=True, use_container_width=True)