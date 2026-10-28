"""Quick look: validate one batch file and print the results (no database needed)."""
import sys

from driftguard.ingestion import read_batch
from driftguard.schema import load_schema
from driftguard.validation import validate_batch

df = read_batch(sys.argv[1])
valid, rejected, results = validate_batch(df, load_schema())

print(f"{len(df)} rows -> {len(valid)} valid, {len(rejected)} rejected\n")
for r in results:
    if r.failed_count > 0:
        print(f"  {r.rule:17} {r.column:15} {r.failed_count:4} rows  ({r.message})")