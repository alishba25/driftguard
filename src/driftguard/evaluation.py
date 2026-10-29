"""Model predictions and performance metrics for one batch (needs labels)."""
import json

import joblib
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

from driftguard import config


def load_model():
    return joblib.load(config.MODEL_PATH)


def load_baseline() -> dict:
    with open(config.BASELINE_METRICS_PATH) as f:
        return json.load(f)


def to_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """What an unmonitored pipeline would feed the model: numbers coerced, bad values blank."""
    X = df[config.NUMERIC + config.CATEGORICAL].copy()
    for col in config.NUMERIC:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X


def evaluate(model, df: pd.DataFrame) -> dict:
    """Predict on the batch and compare with its true labels."""
    y = (df[config.TARGET] == ">50K").astype(int)
    X = to_model_frame(df)
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]
    both_classes = y.nunique() == 2   # AUC needs at least one of each class
    return {
        "n_labelled": int(len(y)),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, prob) if both_classes else None,
        "pr_auc": average_precision_score(y, prob) if both_classes else None,
    }