import numpy as np
import plotly.express as px
import streamlit as st

from common import cached, no_data_message, page_header
from driftguard.dashboard_data import load_drift

page_header("Drift monitoring",
            "Each batch's valid rows compared with the training data, feature by feature.")

drift = cached(load_drift)
if drift.empty:
    no_data_message()
    st.stop()

threshold = drift["threshold"].iloc[0]
st.write(f"A feature has drifted when its p-value is below **{threshold:.4f}** "
         "(0.05 split across 12 features, Bonferroni).")

# ---- Heatmap: how surprising each feature's difference is, per batch ----
drift["surprise"] = -np.log10(drift["p_value"].clip(lower=1e-300))
order = drift.drop_duplicates("batch")["batch"].tolist()
grid = drift.pivot(index="feature", columns="batch", values="surprise")[order]
# Capped at 10 so one enormous value doesn't wash out every other cell
fig = px.imshow(grid, aspect="auto", color_continuous_scale="Blues", range_color=[0, 10],
                labels={"color": "-log10(p)", "x": "Batch", "y": "Feature"})
st.plotly_chart(fig, use_container_width=True)
st.caption(f"Darker = less likely by chance. Drift is flagged above -log10(p) = "
           f"{-np.log10(threshold):.1f}; the colour scale stops at 10.")

# ---- One batch in detail ----
st.subheader("One batch in detail")
chosen = st.selectbox("Batch", order[::-1])
detail = drift[drift["batch"] == chosen][["feature", "test", "statistic", "p_value", "drift_detected"]]
detail = detail.sort_values("p_value")
detail["p_value"] = detail["p_value"].map(lambda p: f"{p:.1e}")   # e.g. 3.2e-12
st.dataframe(detail, hide_index=True, use_container_width=True)