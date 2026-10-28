import pandas as pd
import pytest

from driftguard import config
from driftguard.validation import validate_batch


@pytest.fixture
def schema():
    """A tiny hand-written schema, so tests don't depend on the real data."""
    return {
        "allowed_categories": {
            "workclass": ["Private", "State-gov"],
            "marital-status": ["Never-married", "Divorced"],
            "occupation": ["Sales", "Tech-support"],
            "relationship": ["Husband", "Wife"],
            "race": ["White", "Black"],
            "sex": ["Male", "Female"],
            "native-country": ["India", "United-States"],
        },
        "numeric_ranges": {col: list(r) for col, r in config.NUMERIC_RANGES.items()},
    }


def good_row(**changes):
    """One valid row as text (the way ingestion reads it); override fields to break it."""
    row = {"age": "39", "education-num": "13", "capital-gain": "0", "capital-loss": "0",
           "hours-per-week": "40", "workclass": "Private", "marital-status": "Never-married",
           "occupation": "Sales", "relationship": "Husband", "race": "White",
           "sex": "Male", "native-country": "India"}
    row.update(changes)
    return row


def failures(results, rule):
    return sum(r.failed_count for r in results if r.rule == rule)


def test_clean_batch_passes(schema):
    df = pd.DataFrame([good_row(), good_row(age="50")])
    valid, rejected, results = validate_batch(df, schema)
    assert len(valid) == 2
    assert len(rejected) == 0


def test_missing_column_fails_whole_batch(schema):
    df = pd.DataFrame([good_row()]).drop(columns=["age"])
    valid, rejected, results = validate_batch(df, schema)
    assert len(valid) == 0
    assert results[0].severity == "critical"


def test_text_in_numeric_column_is_rejected(schema):
    df = pd.DataFrame([good_row(), good_row(**{"hours-per-week": "forty"})])
    valid, rejected, results = validate_batch(df, schema)
    assert len(rejected) == 1
    assert failures(results, "numeric_type") == 1


def test_unknown_category_is_rejected(schema):
    df = pd.DataFrame([good_row(sex="Unknown")])
    _, rejected, results = validate_batch(df, schema)
    assert len(rejected) == 1
    assert "allowed_category:sex" in rejected["rejection_reasons"].iloc[0]


def test_out_of_range_is_rejected(schema):
    df = pd.DataFrame([good_row(age="250"), good_row(age="-3")])
    _, rejected, results = validate_batch(df, schema)
    assert failures(results, "range") == 2


def test_missing_value_allowed_only_in_some_columns(schema):
    df = pd.DataFrame([good_row(occupation=None), good_row(sex=None)])
    valid, rejected, _ = validate_batch(df, schema)
    assert len(valid) == 1          # missing occupation is normal
    assert len(rejected) == 1       # missing sex is not


def test_duplicates_keep_first_copy(schema):
    df = pd.DataFrame([good_row(), good_row(), good_row()])
    valid, rejected, results = validate_batch(df, schema)
    assert len(valid) == 1
    assert failures(results, "duplicate_row") == 2


def test_empty_batch_fails(schema):
    df = pd.DataFrame(columns=list(good_row().keys()))
    valid, rejected, results = validate_batch(df, schema)
    assert results[0].rule == "not_empty"
    assert results[0].severity == "critical"