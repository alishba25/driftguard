"""Build sequences of batches with a known, recorded change (the ground truth)."""
import numpy as np
import pandas as pd

from driftguard import config


def make_run(pool: pd.DataFrame, scenario: dict, seed: int, settings: dict):
    """Return a list of (batch, truth) pairs for one scenario and one seed.

    batch = the data a monitor would receive (including the label column,
            which the experiment removes before monitoring)
    truth = what we did to the batch; never shown to the monitors
    """
    n, size, start = settings["n_batches"], settings["batch_size"], settings["shift_start"]
    rng = np.random.default_rng(seed)
    shuffled = pool.sample(frac=1, random_state=seed).reset_index(drop=True)

    run = []
    for i in range(n):
        batch = shuffled.iloc[i * size:(i + 1) * size].copy().reset_index(drop=True)
        changed = scenario["type"] != "stable" and i >= start
        severity = 0.0
        if changed:
            batch, severity = apply_change(batch, scenario, i, settings, pool, rng)
        truth = {
            "batch_index": i,
            # Did the INPUT data change? (label_flip changes only the labels)
            "input_changed": changed and scenario["type"] != "label_flip",
            "severity": severity,
        }
        run.append((batch, truth))
    return run


def apply_change(batch, scenario, i, settings, pool, rng):
    kind = scenario["type"]

    if kind == "corruption":
        return corrupt(batch, scenario["rate"], rng), scenario["rate"]

    if kind == "numeric_shift":
        return shift_column(batch, scenario["column"], scenario["amount"]), scenario["amount"]

    if kind == "gradual_shift":
        steps = settings["n_batches"] - settings["shift_start"]
        amount = scenario["max_amount"] * (i - settings["shift_start"] + 1) / steps
        return shift_column(batch, scenario["column"], amount), round(amount, 2)

    if kind == "resample":
        col, value, share = scenario["column"], scenario["value"], scenario["share"]
        n_match = round(len(batch) * share)
        match = pool[pool[col] == value]
        other = pool[pool[col] != value]
        picked = pd.concat([
            match.sample(n_match, random_state=int(rng.integers(1e9))),
            other.sample(len(batch) - n_match, random_state=int(rng.integers(1e9))),
        ])
        return picked.sample(frac=1, random_state=int(rng.integers(1e9))).reset_index(drop=True), share

    if kind == "label_flip":
        batch = batch.copy()
        who = batch[scenario["column"]] == scenario["value"]
        flip = who & (rng.random(len(batch)) < scenario["flip_rate"])
        batch.loc[flip, config.TARGET] = np.where(
            batch.loc[flip, config.TARGET] == ">50K", "<=50K", ">50K")
        return batch, scenario["flip_rate"]

    raise ValueError(f"Unknown scenario type: {kind}")


def shift_column(batch, column, amount):
    """Add a fixed amount, then keep values inside the valid range (so rows stay valid)."""
    low, high = config.NUMERIC_RANGES[column]
    batch = batch.copy()
    batch[column] = (batch[column] + amount).clip(low, high).round()
    return batch


def corrupt(batch, rate, rng):
    """Break `rate` of the rows, split evenly over five kinds of problem."""
    batch = batch.copy()
    batch["age"] = batch["age"].astype("object")
    batch["hours-per-week"] = batch["hours-per-week"].astype("object")
    rows = rng.choice(len(batch), size=round(len(batch) * rate), replace=False)
    for k, row in enumerate(rows):
        problem = k % 5
        if problem == 0:
            batch.loc[row, "age"] = None                 # missing required value
        elif problem == 1:
            batch.loc[row, "sex"] = "Unknown"            # invalid category
        elif problem == 2:
            batch.loc[row, "age"] = 250                  # out of range
        elif problem == 3:
            batch.loc[row, "hours-per-week"] = "forty"   # wrong type
        else:
            batch.iloc[row] = batch.iloc[(row + 1) % len(batch)]   # duplicate row
    return batch