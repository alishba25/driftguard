"""How often does the detector cry wolf on batches where nothing changed?

Draws 100 random 'stable' batches from the monitor pool (same population as
training) and counts how many raise a drift alert, with and without correction.
"""
import pandas as pd

from driftguard import config
from driftguard.monitoring import detect_drift, load_reference

N_BATCHES = 100
BATCH_SIZE = 1000

reference = load_reference()
pool = pd.read_csv("data/splits/monitor_pool.csv")

for correction in ["none", "bonferroni"]:
    config.DRIFT_CORRECTION = correction   # change the setting for this run only
    alerted = 0
    for seed in range(N_BATCHES):
        batch = pool.sample(BATCH_SIZE, random_state=seed)
        results = detect_drift(reference, batch)
        if any(r.drift_detected for r in results):
            alerted += 1
    print(f"correction={correction:10}  false alerts: {alerted} per {N_BATCHES} stable batches")