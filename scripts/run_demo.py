"""Build everything from scratch and fill the database, in one command.

    python scripts/run_demo.py

Skips the download if the data is already there. Safe to run again.
"""
import subprocess
import sys
from pathlib import Path

STEPS = [
    ("Download the dataset", ["src/download_data.py"], Path("data/processed/adult_clean.csv")),
    ("Train the baseline model", ["src/train_baseline.py"], None),
    ("Learn allowed categories", ["-m", "driftguard.schema"], None),
    ("Make demo batches", ["scripts/make_batches.py"], None),
    ("Make shifted batches", ["scripts/make_shifted_batches.py"], None),
]


def run(args):
    # sys.executable = the Python running this script, so it works inside any environment
    subprocess.run([sys.executable, *args], check=True)


for title, args, skip_if_exists in STEPS:
    if skip_if_exists and skip_if_exists.exists():
        print(f"--- {title}: already done, skipping")
        continue
    print(f"--- {title}")
    run(args)

print("--- Run every batch through the pipeline")
batches = sorted(str(p) for p in Path("data/batches").glob("*.csv"))
run(["-m", "driftguard.pipeline", *batches])

print("\nDone. Start the dashboard with:  streamlit run dashboard/Overview.py")