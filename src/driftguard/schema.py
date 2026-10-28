"""Learn the allowed categories from the training data and save them."""
import json

import pandas as pd

from driftguard import config


def build_schema(train: pd.DataFrame) -> dict:
    """Every category the model saw in training becomes an allowed value."""
    allowed = {}
    for col in config.CATEGORICAL:
        allowed[col] = sorted(train[col].dropna().unique().tolist())
    return {
        "built_from": str(config.TRAIN_PATH),
        "n_rows": int(len(train)),
        "allowed_categories": allowed,
        "numeric_ranges": {col: list(r) for col, r in config.NUMERIC_RANGES.items()},
    }


def save_schema(schema: dict) -> None:
    config.SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.SCHEMA_PATH, "w") as f:
        json.dump(schema, f, indent=2)


def load_schema() -> dict:
    with open(config.SCHEMA_PATH) as f:
        return json.load(f)


if __name__ == "__main__":
    train = pd.read_csv(config.TRAIN_PATH)
    schema = build_schema(train)
    save_schema(schema)
    for col, values in schema["allowed_categories"].items():
        print(f"{col:15} {len(values):3} allowed values")
    print(f"Saved {config.SCHEMA_PATH}")