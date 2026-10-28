"""Cut the monitor pool into small batches, plus one deliberately broken demo batch."""
from pathlib import Path

import pandas as pd

SEED = 42
BATCH_SIZE = 1000
OUT = Path("data/batches")
OUT.mkdir(parents=True, exist_ok=True)

pool = pd.read_csv("data/splits/monitor_pool.csv")
pool = pool.drop(columns=["label"])   # batches arrive without our helper column
pool = pool.sample(frac=1, random_state=SEED).reset_index(drop=True)   # shuffle

# ---- 5 clean batches ----
for i in range(5):
    batch = pool.iloc[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
    batch.to_csv(OUT / f"batch_{i + 1:03}.csv", index=False)

# ---- 1 broken batch, with problems you know the exact count of ----
bad = pool.iloc[5000:5200].copy().reset_index(drop=True)
bad["age"] = bad["age"].astype("object")
bad["hours-per-week"] = bad["hours-per-week"].astype("object")
bad.loc[0:9, "age"] = None                  # 10 missing ages
bad.loc[10:19, "sex"] = "Unknown"           # 10 invalid categories
bad.loc[20:24, "age"] = 250                 # 5 impossible ages
bad.loc[25:29, "hours-per-week"] = "forty"  # 5 text values in a number column
bad = pd.concat([bad, bad.iloc[100:110]])   # 10 duplicate rows
bad.to_csv(OUT / "batch_bad_demo.csv", index=False)

print(f"Wrote 5 clean batches and 1 broken batch to {OUT}/")
print("Expected rejections in the broken batch: 40 (10 + 10 + 5 + 5 + 10)")