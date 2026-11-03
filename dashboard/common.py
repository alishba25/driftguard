"""Shared helpers for every dashboard page."""
import streamlit as st

from driftguard import dashboard_data
from driftguard.storage import get_engine


@st.cache_resource          # one database connection pool for the whole app
def engine():
    return get_engine()


@st.cache_data(ttl=60)      # re-read from the database at most once a minute
def _run(loader_name: str):
    # The cache is keyed by loader_name, so each table gets its own cache entry
    return getattr(dashboard_data, loader_name)(engine())


def cached(loader):
    return _run(loader.__name__)


def page_header(title: str, caption: str):
    """Same wide layout and heading style on every page."""
    st.set_page_config(page_title=f"DriftGuard: {title}", layout="wide")
    st.title(title)
    st.caption(caption)


def no_data_message():
    st.info("No batches in the database yet. Run the pipeline first, for example:\n\n"
            "`python -m driftguard.pipeline data/batches/batch_001.csv`")