"""DriftGuard dashboard.   Run:  streamlit run dashboard/Overview.py"""
import pandas as pd
import plotly.express as px
import streamlit as st

from common import cached, no_data_message, page_header
from driftguard.dashboard_data import load_alerts, load_baseline_auc, load_batches

page_header("DriftGuard overview",
            "Data quality, distribution drift and model performance for each incoming batch.")

batches = cached(load_batches)
if batches.empty:
    no_data_message()
    st.stop()

alerts = cached(load_alerts)
baseline = load_baseline_auc()
latest = batches.iloc[-1]

# ---- Headline numbers ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("Batches processed", len(batches))
c2.metric("Alerts raised", len(alerts))
c3.metric("Rows rejected", f"{batches['rejected_count'].sum() / batches['row_count'].sum():.1%}")
if pd.notna(latest["roc_auc"]):   # the batch had labels
    c4.metric("Latest ROC-AUC", f"{latest['roc_auc']:.3f}",
              delta=f"{latest['roc_auc'] - baseline:+.3f} vs baseline")
else:
    c4.metric("Latest ROC-AUC", "no labels")

# ---- Model performance over batches ----
st.subheader("Model performance by batch")
fig = px.line(batches, x="batch", y="roc_auc", markers=True,
              labels={"batch": "Batch (in processing order)", "roc_auc": "ROC-AUC"})
fig.add_hline(y=baseline, line_dash="dash", annotation_text=f"baseline {baseline:.3f}")
st.plotly_chart(fig, use_container_width=True)

# ---- Batch history ----
st.subheader("Batch history")
st.dataframe(
    batches[["batch", "ingested_at", "row_count", "rejected_count", "status", "n_alerts", "roc_auc"]]
    .iloc[::-1],   # newest first
    hide_index=True, use_container_width=True)