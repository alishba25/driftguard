import numpy as np
import pandas as pd
import pytest

from driftguard import config
from driftguard.scenarios import make_run

SETTINGS = {"n_batches": 4, "batch_size": 50, "shift_start": 2}


@pytest.fixture
def pool():
    """A small fake pool with every column the scenarios touch."""
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({col: rng.integers(20, 60, n) for col in config.NUMERIC})
    for col in config.CATEGORICAL:
        df[col] = rng.choice(["A", "B"], n)
    df["sex"] = rng.choice(["Male", "Female"], n)
    df["marital-status"] = rng.choice(["Married-civ-spouse", "Never-married"], n)
    df[config.TARGET] = rng.choice(["<=50K", ">50K"], n)
    return df


def test_same_seed_gives_identical_batches(pool):
    scenario = {"type": "numeric_shift", "column": "age", "amount": 5}
    run1 = make_run(pool, scenario, seed=3, settings=SETTINGS)
    run2 = make_run(pool, scenario, seed=3, settings=SETTINGS)
    for (b1, _), (b2, _) in zip(run1, run2):
        pd.testing.assert_frame_equal(b1, b2)


def test_change_starts_at_shift_start(pool):
    run = make_run(pool, {"type": "numeric_shift", "column": "age", "amount": 5}, 0, SETTINGS)
    assert [t["input_changed"] for _, t in run] == [False, False, True, True]


def test_stable_never_changes(pool):
    run = make_run(pool, {"type": "stable"}, 0, SETTINGS)
    assert not any(t["input_changed"] for _, t in run)


def test_gradual_shift_grows(pool):
    run = make_run(pool, {"type": "gradual_shift", "column": "age", "max_amount": 10}, 0, SETTINGS)
    assert [t["severity"] for _, t in run] == [0.0, 0.0, 5.0, 10.0]


def test_label_flip_changes_labels_but_not_inputs(pool):
    stable = make_run(pool, {"type": "stable"}, 0, SETTINGS)
    flipped = make_run(pool, {"type": "label_flip", "column": "marital-status",
                              "value": "Married-civ-spouse", "flip_rate": 1.0}, 0, SETTINGS)
    before, after = stable[3][0], flipped[3][0]
    inputs = config.NUMERIC + config.CATEGORICAL
    pd.testing.assert_frame_equal(before[inputs], after[inputs])          # inputs identical
    married = before["marital-status"] == "Married-civ-spouse"
    assert (before.loc[married, config.TARGET] != after.loc[married, config.TARGET]).all()
    assert flipped[3][1]["input_changed"] is False                        # truth: inputs unchanged


def test_corruption_breaks_the_right_number_of_rows(pool):
    run = make_run(pool, {"type": "corruption", "rate": 0.2}, 0, SETTINGS)
    batch = run[2][0]
    assert (batch["sex"] == "Unknown").sum() == 2          # 10 broken rows / 5 kinds
    assert (batch["hours-per-week"] == "forty").sum() == 2