import streamlit as st

from common import cached, no_data_message, page_header
from driftguard.dashboard_data import load_alerts

page_header("Alert history",
            "Newest first. Suggested next steps are advice for a person, not automatic actions.")

alerts = cached(load_alerts)
if alerts.empty:
    no_data_message()
    st.stop()

categories = st.multiselect("Category", sorted(alerts["category"].unique()),
                            default=sorted(alerts["category"].unique()))
shown = alerts[alerts["category"].isin(categories)]
st.dataframe(shown[["created_at", "batch", "category", "severity", "message", "next_step"]],
             hide_index=True, use_container_width=True)