"""Data-quality checks: decide which rows are valid and record why others are not."""
from dataclasses import dataclass

import pandas as pd

from driftguard import config


@dataclass
class RuleResult:
    rule: str
    column: str
    failed_count: int
    failure_rate: float
    severity: str        # "critical" = whole batch unusable, "error" = row rejected
    message: str


def validate_batch(df: pd.DataFrame, schema: dict):
    """Return (valid_rows, rejected_rows, list of RuleResult)."""
    n = len(df)
    results = []

    # ---- Check 0: is there any data at all? (whole-batch check) ----
    if n == 0:
        results.append(RuleResult("not_empty", "*", 0, 1.0, "critical", "Batch has no rows"))
        return df, df, results

    # ---- Check 1: are all required columns present? (whole-batch check) ----
    missing_cols = [c for c in config.REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        results.append(RuleResult("required_columns", ",".join(missing_cols), n, 1.0,
                                  "critical", f"Batch is missing columns: {missing_cols}"))
        rejected = df.assign(rejection_reasons="missing_columns")
        return df.iloc[0:0], rejected, results

    df = df.reset_index(drop=True)   # a fresh 0..n-1 row numbering (also makes a copy)
    # One list of reasons per row; a row is rejected if its list is not empty
    reasons = pd.Series([[] for _ in range(n)])

    def record(rule, column, failed_mask, message):
        count = int(failed_mask.sum())
        results.append(RuleResult(rule, column, count, round(count / n, 4) if n else 0.0,
                                  "error", message))
        for idx in df.index[failed_mask]:
            reasons[idx].append(f"{rule}:{column}")

    # ---- Check 2: numeric columns must contain numbers ----
    was_missing = df.isna()   # remember what was truly empty before converting
    for col in config.NUMERIC:
        as_number = pd.to_numeric(df[col], errors="coerce")
        bad_type = df[col].notna() & as_number.isna()   # had a value, but not a number
        record("numeric_type", col, bad_type, f"{col} has non-numeric values")
        df[col] = as_number

    # ---- Check 3: no missing values where they shouldn't be ----
    for col in config.REQUIRED_COLUMNS:
        if col in config.MISSING_ALLOWED:
            continue
        record("not_missing", col, was_missing[col], f"{col} is missing")

    # ---- Check 4: categories must be ones seen in training ----
    for col in config.CATEGORICAL:
        allowed = set(schema["allowed_categories"][col])
        unknown = df[col].notna() & ~df[col].isin(allowed)
        record("allowed_category", col, unknown, f"{col} has values not seen in training")

    # ---- Check 5: numbers must be within plausible ranges ----
    for col, (low, high) in schema["numeric_ranges"].items():
        out = df[col].notna() & ((df[col] < low) | (df[col] > high))
        record("range", col, out, f"{col} outside [{low}, {high}]")

    # ---- Check 6: exact duplicate rows (keep the first copy) ----
    dup = df.duplicated(keep="first")
    record("duplicate_row", "*", dup, "exact duplicate of an earlier row")

    is_rejected = reasons.apply(len) > 0
    rejected = df[is_rejected].assign(rejection_reasons=reasons[is_rejected].apply(";".join))
    valid = df[~is_rejected]
    return valid, rejected, results