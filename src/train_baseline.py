"""Steps 5-6: split the data, train a logistic regression baseline, save metrics."""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42  # fixed so every run gives the same split and the same numbers

NUMERIC = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL = ["workclass", "marital-status", "occupation", "relationship",
               "race", "sex", "native-country"]
TARGET = "income"

# ---- 1. Load data and make the target a 0/1 number ----
df = pd.read_csv("data/processed/adult_clean.csv")
df["label"] = (df[TARGET] == ">50K").astype(int)

# ---- 2. Split three ways: train 60%, test 20%, monitoring pool 20% ----
train, rest = train_test_split(df, test_size=0.4, random_state=SEED, stratify=df["label"])
test, monitor_pool = train_test_split(rest, test_size=0.5, random_state=SEED, stratify=rest["label"])

Path("data/splits").mkdir(parents=True, exist_ok=True)
train.to_csv("data/splits/train.csv", index=False)
test.to_csv("data/splits/test.csv", index=False)
monitor_pool.to_csv("data/splits/monitor_pool.csv", index=False)
print(f"Split sizes -> train: {len(train)}, test: {len(test)}, monitor pool: {len(monitor_pool)}")

# ---- 3. Build the preprocessing + model pipeline ----
numeric_steps = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])
categorical_steps = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
preprocess = ColumnTransformer([
    ("num", numeric_steps, NUMERIC),
    ("cat", categorical_steps, CATEGORICAL),
])
model = Pipeline([
    ("preprocess", preprocess),
    ("classifier", LogisticRegression(max_iter=1000, random_state=SEED)),
])

# ---- 4. Train on the training set only ----
model.fit(train[NUMERIC + CATEGORICAL], train["label"])

# ---- 5. Evaluate on the test set ----
X_test, y_test = test[NUMERIC + CATEGORICAL], test["label"]
pred = model.predict(X_test)            # 0 or 1
prob = model.predict_proba(X_test)[:, 1]  # probability of >50K

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

# ---- 6. Save the model and the metrics ----
Path("models").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)
joblib.dump(model, "models/baseline_logreg.joblib")
with open("reports/baseline_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(json.dumps(metrics, indent=2))
print("Saved model to models/baseline_logreg.joblib and metrics to reports/baseline_metrics.json")