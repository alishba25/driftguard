"""Make two demo batches with a deliberate, known distribution shift."""
from pathlib import Path

import pandas as pd

SEED = 7
OUT = Path("data/batches")
pool = pd.read_csv("data/splits/monitor_pool.csv").drop(columns=["label"])

# ---- Numeric shift: everyone is 8 years older ----
older = pool.sample(1000, random_state=SEED).copy()
older["age"] = (older["age"] + 8).clip(upper=90)
older.to_csv(OUT / "batch_shift_age.csv", index=False)

# ---- Categorical shift: over-sample married people ----
married = pool[pool["marital-status"] == "Married-civ-spouse"].sample(700, random_state=SEED)
others = pool[pool["marital-status"] != "Married-civ-spouse"].sample(300, random_state=SEED)
pd.concat([married, others]).sample(frac=1, random_state=SEED).to_csv(
    OUT / "batch_shift_marital.csv", index=False)

print("Wrote batch_shift_age.csv (age +8 years) and batch_shift_marital.csv (70% married)")