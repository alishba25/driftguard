"""One place for every setting the project shares."""
from pathlib import Path

# ---- Columns (from your Day 4 findings) ----
NUMERIC = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL = ["workclass", "marital-status", "occupation", "relationship",
               "race", "sex", "native-country"]
REQUIRED_COLUMNS = NUMERIC + CATEGORICAL   # a batch must contain all of these
TARGET = "income"

# Missing values are normal in these columns (Day 4, finding 3).
# A missing value anywhere else is a quality failure.
MISSING_ALLOWED = ["workclass", "occupation", "native-country"]

# Plausible ranges, set by hand from what makes sense in the real world,
# not just from the min/max seen in training data. Document why for each.
NUMERIC_RANGES = {
    "age": (17, 100),             # dataset covers working-age adults; >100 is implausible
    "education-num": (1, 16),     # fixed coding: 1 = Preschool ... 16 = Doctorate
    "capital-gain": (0, 99999),   # 99999 is the dataset's cap
    "capital-loss": (0, 99999),
    "hours-per-week": (1, 99),    # 99 is the dataset's cap
}

# ---- Paths ----
TRAIN_PATH = Path("data/splits/train.csv")
SCHEMA_PATH = Path("reports/reference_schema.json")
REJECTED_DIR = Path("data/rejected")
LOG_PATH = Path("logs/pipeline.log")


# ---- Drift detection (Phase 3) ----
REFERENCE_PATH = TRAIN_PATH      # the model's training data is the "normal" we compare against
DRIFT_ALPHA = 0.05               # overall false-alarm budget per batch
DRIFT_CORRECTION = "bonferroni"  # split that budget across features ("none" to switch off)
RARE_CATEGORY_SHARE = 0.01       # categories under 1% of reference are grouped as "__other__"
QUALITY_ALERT_RATE = 0.01        # alert if more than 1% of a batch's rows are rejected


# ---- Model (Phase 4) ----
MODEL_PATH = Path("models/baseline_logreg.joblib")
MODEL_VERSION = "baseline_logreg_v1"
BASELINE_METRICS_PATH = Path("reports/baseline_metrics.json")

