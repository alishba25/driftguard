import pandas as pd

from driftguard.scoring import alert_scores, detection_delay, false_alerts_per_100


def test_alert_scores_by_hand():
    alerts = pd.Series([True, True, False, False, True])
    truth = pd.Series([True, False, True, False, True])
    s = alert_scores(alerts, truth)
    assert (s["tp"], s["fp"], s["fn"]) == (2, 1, 1)
    assert s["precision"] == 2 / 3
    assert s["recall"] == 2 / 3


def test_no_alerts_means_precision_is_undefined():
    s = alert_scores(pd.Series([False, False]), pd.Series([True, False]))
    assert s["precision"] is None
    assert s["recall"] == 0.0


def test_false_alerts_only_count_quiet_batches():
    alerts = pd.Series([True, False, True, True])
    truth = pd.Series([False, False, True, True])   # first two should be quiet
    assert false_alerts_per_100(alerts, truth) == 50.0


def test_detection_delay():
    idx = pd.Series(range(8))
    alerts = pd.Series([True, False, False, False, False, False, True, True])
    assert detection_delay(alerts, idx, start=4) == 2      # alert at batch 0 is before the change
    assert detection_delay(pd.Series([False] * 8), idx, start=4) is None