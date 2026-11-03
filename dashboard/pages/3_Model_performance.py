import plotly.express as px
import streamlit as st

from common import cached, no_data_message, page_header
from driftguard.dashboard_data import load_baseline_auc, load_batches

page_header("Model performance",
            "Only for batches that arrived with labels. Drift alerts alone don't tell you this.")

batches = cached(load_batches).dropna(subset=["roc_auc"])
if batches.empty:
    no_data_message()
    st.stop()

baseline = load_baseline_auc()
harm_line = baseline - 0.03

# ROC-AUC decides harm (Phase 4 rule), so it gets the chart and the threshold lines
fig = px.line(batches, x="batch", y="roc_auc", markers=True,
              labels={"batch": "Batch", "roc_auc": "ROC-AUC"})
fig.add_hline(y=baseline, line_dash="dash", annotation_text=f"baseline {baseline:.3f}")
fig.add_hline(y=harm_line, line_dash="dot", annotation_text=f"harm threshold {harm_line:.3f}")
st.plotly_chart(fig, use_container_width=True)

harmful = batches[batches["roc_auc"] < harm_line]
if harmful.empty:
    st.success("No batch fell more than 0.03 below the baseline ROC-AUC.")
else:
    st.warning(f"{len(harmful)} batch(es) below the harm threshold: " + ", ".join(harmful["batch"]))

# PR-AUC moves with the share of >50K earners, so it's shown but not used for the harm rule
st.subheader("All scores")
st.dataframe(batches[["batch", "roc_auc", "pr_auc"]].round(3).iloc[::-1],
             hide_index=True, use_container_width=True)
st.caption("PR-AUC also changes when the share of >50K earners changes, "
           "so it is shown for reference only.")