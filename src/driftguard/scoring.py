"""Score alerts against the known ground truth."""
import pandas as pd


def alert_scores(alerts: pd.Series, truth: pd.Series) -> dict:
    """Precision, recall and F1 of alerts (True/False per batch) against truth (True/False)."""
    alerts, truth = alerts.astype(bool), truth.astype(bool)
    tp = int((alerts & truth).sum())     # alerted, and should have
    fp = int((alerts & ~truth).sum())    # alerted, but shouldn't have
    fn = int((~alerts & truth).sum())    # stayed quiet, but shouldn't have
    precision = tp / (tp + fp) if tp + fp else None   # no alerts at all: undefined
    recall = tp / (tp + fn) if tp + fn else None
    if precision and recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None if precision is None or recall is None else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def false_alerts_per_100(alerts: pd.Series, truth: pd.Series) -> float:
    """Alerts on batches where nothing changed, scaled to 100 such batches."""
    quiet = ~truth.astype(bool)
    return 100 * alerts[quiet].astype(bool).sum() / quiet.sum() if quiet.sum() else None


def detection_delay(alerts: pd.Series, batch_index: pd.Series, start: int):
    """Batches from the start of the change to the first alert (None = never detected)."""
    hits = batch_index[alerts.astype(bool) & (batch_index >= start)]
    return int(hits.min() - start) if len(hits) else None