import plotly.express as px
import streamlit as st

from common import cached, no_data_message, page_header
from driftguard.dashboard_data import load_batches, load_quality

page_header("Data quality",
            "Rows that failed validation, by batch and by rule. Rejected rows never reach the model.")

batches = cached(load_batches)
if batches.empty:
    no_data_message()
    st.stop()

fig = px.bar(batches, x="batch", y="rejection_rate",
             labels={"batch": "Batch", "rejection_rate": "Share of rows rejected"})
fig.update_yaxes(tickformat=".0%")
fig.add_hline(y=0.01, line_dash="dash", annotation_text="alert above 1%")
st.plotly_chart(fig, use_container_width=True)

quality = cached(load_quality)
st.subheader("Failures by rule")
if quality.empty:
    st.success("No quality failures recorded.")
else:
    st.dataframe(quality[["batch", "rule", "column_name", "failed_count", "failure_rate", "message"]],
                 hide_index=True, use_container_width=True)