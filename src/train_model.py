"""Train one model on one dataset's training split and save its baseline metrics.

    python src/train_model.py --dataset adult --model gbm
    python src/train_model.py --dataset acs --model logreg

Models: logreg (logistic regression, as in Phase 1) and gbm (gradient-boosted
trees, scikit-learn's HistGradientBoostingClassifier, similar to XGBoost).
The splits must already exist (src/train_baseline.py for Adult,
src/prepare_acs.py for ACS).
"""
import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["adult", "acs"], required=True)
parser.add_argument("--model", choices=["logreg", "gbm"], required=True)
args = parser.parse_args()
os.environ["DRIFTGUARD_DATASET"] = args.dataset   # must be set before importing driftguard

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,  # noqa: E402
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler  # noqa: E402

from driftguard import config  # noqa: E402

SEED = 42
FEATURES = config.NUMERIC + config.CATEGORICAL

if args.model == "logreg":   # identical to the Phase 1 baseline
    preprocess = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), config.NUMERIC),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), config.CATEGORICAL),
    ])
    classifier = LogisticRegression(max_iter=1000, random_state=SEED)
else:   # trees need no scaling; categories become integer codes, unseen ones -> -1
    preprocess = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), config.NUMERIC),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("codes", OrdinalEncoder(handle_unknown="use_encoded_value",
                                                   unknown_value=-1))]), config.CATEGORICAL),
    ])
    classifier = HistGradientBoostingClassifier(random_state=SEED)

model = Pipeline([("preprocess", preprocess), ("classifier", classifier)])

train = pd.read_csv(config.TRAIN_PATH)
test = pd.read_csv(config.TEST_PATH)
y_train = (train[config.TARGET] == ">50K").astype(int)
y_test = (test[config.TARGET] == ">50K").astype(int)
model.fit(train[FEATURES], y_train)

pred = model.predict(test[FEATURES])
prob = model.predict_proba(test[FEATURES])[:, 1]
metrics = {
    "accuracy": accuracy_score(y_test, pred),
    "precision": precision_score(y_test, pred),
    "recall": recall_score(y_test, pred),
    "f1": f1_score(y_test, pred),
    "roc_auc": roc_auc_score(y_test, prob),
    "pr_auc": average_precision_score(y_test, prob),
    "positive_rate_in_test": float(y_test.mean()),
    "n_test": int(len(y_test)),
    "seed": SEED,
}
metrics = {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}

config.model_path(args.model).parent.mkdir(exist_ok=True)
joblib.dump(model, config.model_path(args.model))
with open(config.baseline_path(args.model), "w") as f:
    json.dump(metrics, f, indent=2)
print(f"{args.dataset} / {args.model}: {json.dumps(metrics)}")
print(f"Saved {config.model_path(args.model)} and {config.baseline_path(args.model)}")