"""Download Folktables ACS income data and build splits with REAL distribution shift.

    python src/prepare_acs.py

Source population: California, 2018. Its people are split into train / test /
monitor pool, exactly like Adult. Target populations: other US states in the
same year, used as real shifted batches. Downloads about 1 GB the first time
(cached in data/acs/raw/, which git ignores).
"""
import os
from pathlib import Path

os.environ["DRIFTGUARD_DATASET"] = "acs"   # so the schema below is built for ACS

import numpy as np
import pandas as pd
from folktables import ACSDataSource, ACSIncome
from sklearn.model_selection import train_test_split

from driftguard.schema import build_schema, save_schema

SEED = 42
YEAR = "2018"
SOURCE_STATE = "CA"
TARGET_STATES = ["NY", "TX", "FL", "PR"]   # PR = Puerto Rico, expected to differ most
SOURCE_ROWS = 50_000     # 60% train, 20% test, 20% monitor pool, as for Adult
TARGET_ROWS = 10_000     # per target state; each run uses 8 x 600 = 4,800 of them

NUMERIC = ["AGEP", "SCHL", "WKHP"]
CATEGORICAL = ["COW", "MAR", "OCCP", "POBP", "RELP", "SEX", "RAC1P"]
OUT = Path("data/acs")
SPLITS = OUT / "splits"
SPLITS.mkdir(parents=True, exist_ok=True)

source = ACSDataSource(survey_year=YEAR, horizon="1-Year", survey="person",
                       root_dir=str(OUT / "raw"))


def load_state(state: str) -> pd.DataFrame:
    """One state's ACSIncome rows in DriftGuard's format."""
    raw = source.get_data(states=[state], download=True)
    features, label, _ = ACSIncome.df_to_pandas(raw)
    df = pd.DataFrame(index=features.index)
    for col in NUMERIC:
        df[col] = features[col].astype(int)
    for col in CATEGORICAL:
        codes = features[col].astype(int)
        # Folktables writes -1 for "missing"; store codes as text ("c4") so they
        # are always read back as categories, never as numbers
        df[col] = np.where(codes == -1, None, "c" + codes.astype(str))
    df["income"] = np.where(label.to_numpy().ravel(), ">50K", "<=50K")
    print(f"{state}: {len(df):,} rows, {np.mean(df['income'] == '>50K'):.1%} earn >50K")
    return df


# ---- Source: California, split three ways like Adult ----
ca = load_state(SOURCE_STATE).sample(SOURCE_ROWS, random_state=SEED)
train, rest = train_test_split(ca, test_size=0.4, random_state=SEED, stratify=ca["income"])
test, pool = train_test_split(rest, test_size=0.5, random_state=SEED, stratify=rest["income"])
train.to_csv(SPLITS / "train.csv", index=False)
test.to_csv(SPLITS / "test.csv", index=False)
pool.to_csv(SPLITS / "monitor_pool.csv", index=False)
print(f"Splits -> train {len(train):,}, test {len(test):,}, monitor pool {len(pool):,}")

# ---- Targets: other states, kept whole up to TARGET_ROWS ----
for state in TARGET_STATES:
    df = load_state(state)
    df = df.sample(min(len(df), TARGET_ROWS), random_state=SEED)
    df.to_csv(OUT / f"target_{state}.csv", index=False)
    print(f"Saved {OUT / f'target_{state}.csv'} ({len(df):,} rows)")

# ---- Allowed categories for the quality checks, learned from the training split ----
save_schema(build_schema(train))
print("Saved reports/acs_reference_schema.json")