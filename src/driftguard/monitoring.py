"""Drift detection: compare a batch's distributions with the reference data."""
from dataclasses import dataclass

import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

from driftguard import config


@dataclass
class DriftResult:
    feature: str
    test: str            # "ks" or "chi2"
    statistic: float     # how different (KS: 0 = same, 1 = completely different)
    p_value: float       # how surprising the difference is if nothing changed
    threshold: float     # p-value below this = drift
    drift_detected: bool
    n_reference: int
    n_current: int


def per_feature_threshold(n_features: int) -> float:
    """Bonferroni: divide the false-alarm budget across all the features we test."""
    if config.DRIFT_CORRECTION == "bonferroni":
        return config.DRIFT_ALPHA / n_features
    return config.DRIFT_ALPHA


def ks_test(reference: pd.Series, current: pd.Series):
    """Kolmogorov-Smirnov: largest gap between the two cumulative distributions."""
    result = ks_2samp(reference.dropna(), current.dropna())
    return float(result.statistic), float(result.pvalue)


def prepare_categories(reference: pd.Series, current: pd.Series):
    """Treat missing as its own category and group rare categories together."""
    ref = reference.fillna("__missing__")
    cur = current.fillna("__missing__")
    shares = ref.value_counts(normalize=True)
    common = set(shares[shares >= config.RARE_CATEGORY_SHARE].index)
    ref = ref.where(ref.isin(common), "__other__")
    cur = cur.where(cur.isin(common), "__other__")
    return ref, cur


def chi2_test(reference: pd.Series, current: pd.Series):
    """Chi-square: do the category proportions differ more than chance allows?"""
    ref, cur = prepare_categories(reference, current)
    table = pd.DataFrame({"reference": ref.value_counts(), "current": cur.value_counts()}).fillna(0)
    table = table[table.sum(axis=1) > 0]
    if len(table) < 2:   # only one category in both: nothing to compare
        return 0.0, 1.0
    statistic, p_value, _, _ = chi2_contingency(table.T)
    return float(statistic), float(p_value)


def detect_drift(reference: pd.DataFrame, current: pd.DataFrame) -> list[DriftResult]:
    features = config.NUMERIC + config.CATEGORICAL
    threshold = per_feature_threshold(len(features))
    results = []
    for col in features:
        if col in config.NUMERIC:
            test = "ks"
            statistic, p_value = ks_test(reference[col], current[col])
        else:
            test = "chi2"
            statistic, p_value = chi2_test(reference[col], current[col])
        results.append(DriftResult(col, test, round(statistic, 4), p_value, threshold,
                                   p_value < threshold, len(reference), len(current)))
    return results


def load_reference() -> pd.DataFrame:
    return pd.read_csv(config.REFERENCE_PATH)