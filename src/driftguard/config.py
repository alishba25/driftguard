"""One place for every setting the project shares.

Two datasets are supported. Adult is the default; to use the Folktables
(ACS income) data instead, set the environment variable DRIFTGUARD_DATASET=acs.
The experiment runner and train_model.py set it for you from their arguments.
"""
import os
from pathlib import Path

DATASET = os.environ.get("DRIFTGUARD_DATASET", "adult")
TARGET = "income"            # values ">50K" / "<=50K" in both datasets

if DATASET == "adult":
    # ---- Columns (from your Day 4 findings) ----
    NUMERIC = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    CATEGORICAL = ["workclass", "marital-status", "occupation", "relationship",
                   "race", "sex", "native-country"]
    # Missing values are normal in these columns (Day 4, finding 3)
    MISSING_ALLOWED = ["workclass", "occupation", "native-country"]
    # Plausible ranges, set by hand from what makes sense in the real world
    NUMERIC_RANGES = {
        "age": (17, 100),             # working-age adults; >100 is implausible
        "education-num": (1, 16),     # fixed coding: 1 = Preschool ... 16 = Doctorate
        "capital-gain": (0, 99999),   # 99999 is the dataset's cap
        "capital-loss": (0, 99999),
        "hours-per-week": (1, 99),    # 99 is the dataset's cap
    }
    # Columns the corruption scenario breaks (age, sex, hours worked)
    CORRUPT_COLUMNS = {"age": "age", "sex": "sex", "hours": "hours-per-week"}
    SPLIT_DIR = Path("data/splits")
    SCHEMA_PATH = Path("reports/reference_schema.json")

elif DATASET == "acs":
    # Folktables ACSIncome features (Ding et al. 2021). Codes are stored as
    # text like "c1" so they are always read back as categories, never numbers.
    NUMERIC = ["AGEP", "SCHL", "WKHP"]   # age, education level (1-24), hours per week
    CATEGORICAL = ["COW", "MAR", "OCCP", "POBP", "RELP", "SEX", "RAC1P"]
    MISSING_ALLOWED = ["COW"]            # class of worker can be blank in the census
    NUMERIC_RANGES = {
        "AGEP": (17, 99),     # ACSIncome keeps people over 16; the census caps age at 99
        "SCHL": (1, 24),      # fixed coding: 1 = no schooling ... 24 = doctorate
        "WKHP": (1, 99),      # ACSIncome keeps people working >0 hours; 99 is the cap
    }
    CORRUPT_COLUMNS = {"age": "AGEP", "sex": "SEX", "hours": "WKHP"}
    SPLIT_DIR = Path("data/acs/splits")
    SCHEMA_PATH = Path("reports/acs_reference_schema.json")

else:
    raise ValueError(f"Unknown DRIFTGUARD_DATASET: {DATASET!r} (use 'adult' or 'acs')")

REQUIRED_COLUMNS = NUMERIC + CATEGORICAL   # a batch must contain all of these

# ---- Paths ----
TRAIN_PATH = SPLIT_DIR / "train.csv"
TEST_PATH = SPLIT_DIR / "test.csv"
MONITOR_POOL_PATH = SPLIT_DIR / "monitor_pool.csv"
REJECTED_DIR = Path("data/rejected")
LOG_PATH = Path("logs/pipeline.log")

# ---- Drift detection (Phase 3) ----
REFERENCE_PATH = TRAIN_PATH      # the model's training data is the "normal" we compare against
DRIFT_ALPHA = 0.05               # overall false-alarm budget per batch
DRIFT_CORRECTION = "bonferroni"  # split that budget across features ("none" to switch off)
RARE_CATEGORY_SHARE = 0.01       # categories under 1% of reference are grouped as "__other__"
QUALITY_ALERT_RATE = 0.01        # alert if more than 1% of a batch's rows are rejected


# ---- Models (Phases 4 and 7) ----
def model_path(model_name: str = "logreg") -> Path:
    if DATASET == "adult" and model_name == "logreg":
        return Path("models/baseline_logreg.joblib")       # the Phase 1 file, unchanged
    return Path(f"models/{DATASET}_{model_name}.joblib")


def baseline_path(model_name: str = "logreg") -> Path:
    if DATASET == "adult" and model_name == "logreg":
        return Path("reports/baseline_metrics.json")       # the Phase 1 file, unchanged
    return Path(f"reports/baseline_metrics_{DATASET}_{model_name}.json")


MODEL_PATH = model_path("logreg")
MODEL_VERSION = "baseline_logreg_v1"
BASELINE_METRICS_PATH = baseline_path("logreg")