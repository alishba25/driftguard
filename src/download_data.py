"""Step 3: download the UCI Adult dataset and save a raw copy and a cleaned copy."""
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path("data/raw/adult_original.csv")
CLEAN_PATH = Path("data/processed/adult_clean.csv")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Tidy the raw data without changing what it means."""
    df = df.copy()
    # 1. Remove stray spaces around text values (" Private" -> "Private")
    for col in df.select_dtypes(exclude="number").columns:
        df[col] = df[col].str.strip()
    # 2. The dataset marks unknown values with "?" -> turn them into real missing values
    df = df.replace("?", np.nan)
    # 3. The test part of the file writes ">50K." with a full stop -> remove it
    df["income"] = df["income"].str.rstrip(".")
    return df


def main():
    from ucimlrepo import fetch_ucirepo  # imported here so clean() works without internet

    print("Downloading Adult dataset from UCI...")
    adult = fetch_ucirepo(id=2)
    raw = adult.data.original  # features + target in one table

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(RAW_PATH, index=False)
    print(f"Saved raw copy: {RAW_PATH} ({len(raw)} rows) on {date.today()}")

    cleaned = clean(raw)
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(CLEAN_PATH, index=False)
    print(f"Saved clean copy: {CLEAN_PATH}")
    print("Income values now:", cleaned["income"].value_counts().to_dict())


if __name__ == "__main__":
    main()