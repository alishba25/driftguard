# DriftGuard

A data-quality and ML-monitoring platform. It checks incoming data batches,
detects distribution drift, and records whether drift affects model performance.

**Research question:** How effectively can data-quality checks combined with
drift detection distinguish harmful distribution changes from harmless shifts
in machine-learning input data?

**Status:** Phase 2 - Complete

## How to run (Phase 1)

    python -m venv .venv
    .venv\Scripts\Activate.ps1        # Mac: source .venv/bin/activate
    pip install -r requirements.txt
    python src/download_data.py
    python src/explore.py
    python src/train_baseline.py

## Baseline results

Logistic regression, seed 42, evaluated on a 20% stratified test split.

| Metric | Value |
| --- | --- |
| Accuracy | 0.8524 |
| Precision | 0.7366 |
| Recall | 0.5961 |
| F1 | 0.6589 |
| ROC-AUC | 0.9056 |
| PR-AUC | 0.7612 |

## Known limitations so far

- 1994 US census data; not representative of today or of other countries.
- Includes sensitive attributes (race, sex); this project studies monitoring, not fairness.
- Duplicate rows kept as-is for now.

## How to run (Phase 2)

    python -m venv .venv
    .venv\Scripts\Activate.ps1        # Mac: source .venv/bin/activate
    pip install -r requirements.txt
    pip install -e .
    copy .env.example .env             # then put your PostgreSQL password in .env

    python src/download_data.py        # Phase 1: data
    python src/train_baseline.py       # Phase 1: model + splits
    python -m driftguard.schema        # Phase 2: allowed categories
    python scripts/make_batches.py     # Phase 2: test batches
    python -m driftguard.pipeline data/batches/batch_001.csv
    pytest                             # run all tests