"""Quick look: compare one batch with the reference (no database needed)."""
import sys

from driftguard.ingestion import read_batch
from driftguard.monitoring import detect_drift, load_reference
from driftguard.schema import load_schema
from driftguard.validation import validate_batch

if len(sys.argv) < 2:
    sys.exit("Usage: python scripts/try_drift.py <batch.csv>")

valid, _, _ = validate_batch(read_batch(sys.argv[1]), load_schema())
results = detect_drift(load_reference(), valid)

print(f"Threshold per feature: p < {results[0].threshold:.4f}\n")
print(f"{'feature':16}{'test':6}{'statistic':>10}{'p-value':>12}  drift?")
for r in results:
    flag = "DRIFT" if r.drift_detected else ""
    print(f"{r.feature:16}{r.test:6}{r.statistic:10.4f}{r.p_value:12.2e}  {flag}")