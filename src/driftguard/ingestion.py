"""Read a batch file exactly as it arrives, plus a stable ID for it."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def make_batch_id(path: Path) -> str:
    """Same file contents -> same ID, so rerunning a batch never duplicates it."""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return digest[:12]


def read_batch(path: Path) -> pd.DataFrame:
    # Read everything as text first. Turning text into numbers is a quality
    # check (validation), so it must not happen silently here.
    df = pd.read_csv(path, dtype=str)
    for col in df.columns:
        df[col] = df[col].str.strip()
    return df.replace({"?": np.nan, "": np.nan})
    