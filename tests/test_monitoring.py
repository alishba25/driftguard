import numpy as np
import pandas as pd

from driftguard import config
from driftguard.alerts import make_alerts
from driftguard.monitoring import chi2_test, detect_drift, ks_test, per_feature_threshold


def fake_data(n, seed, age_shift=0, married_share=0.5):
    """Simple synthetic data with every column the detector expects."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({col: rng.normal(40, 10, n) for col in config.NUMERIC})
    for col in config.CATEGORICAL:
        df[col] = rng.choice(["A", "B", "C"], n)
    df["age"] = df["age"] + age_shift
    df["marital-status"] = np.where(rng.random(n) < married_share, "Married", "Single")
    return df


def test_bonferroni_divides_alpha():
    assert per_feature_threshold(12) == config.DRIFT_ALPHA / 12


def test_same_distribution_is_not_drift():
    stat, p = ks_test(pd.Series(np.random.default_rng(0).normal(0, 1, 2000)),
                      pd.Series(np.random.default_rng(1).normal(0, 1, 2000)))
    assert p > 0.01


def test_numeric_shift_is_detected():
    results = detect_drift(fake_data(5000, 0), fake_data(1000, 1, age_shift=8))
    drifted = {r.feature for r in results if r.drift_detected}
    assert drifted == {"age"}


def test_categorical_shift_is_detected():
    results = detect_drift(fake_data(5000, 0), fake_data(1000, 1, married_share=0.8))
    drifted = {r.feature for r in results if r.drift_detected}
    assert drifted == {"marital-status"}


def test_missing_values_count_as_a_category():
    ref = pd.Series(["A", "B"] * 500)
    cur = pd.Series(["A", "B"] * 250 + [None] * 500)   # half suddenly missing
    _, p = chi2_test(ref, cur)
    assert p < 0.001


def test_drift_alert_names_the_features():
    batch = {"status": "passed", "row_count": 1000, "rejected_count": 0}
    results = detect_drift(fake_data(5000, 0), fake_data(1000, 1, age_shift=8))
    alerts = make_alerts(batch, [], results)
    assert len(alerts) == 1
    assert alerts[0]["category"] == "drift"
    assert "age" in alerts[0]["message"]


def test_small_rejection_rate_does_not_alert():
    batch = {"status": "passed_with_rejections", "row_count": 1000, "rejected_count": 5}
    assert make_alerts(batch, [], []) == []