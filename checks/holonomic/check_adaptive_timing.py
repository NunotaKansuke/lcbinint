"""Check that adaptive benchmark stage timings are finite and correctly scaled."""
import csv
import json
import math
import sys


path = sys.argv[1] if len(sys.argv) > 1 else "paired.csv"
stage_names = ["topology_ms", "setup_ms", "physics_ms", "estimator_ms", "scheduler_ms"]
bad = []
rows = 0
finite_rows = 0
with open(path, newline="") as f:
    for line_no, row in enumerate(csv.DictReader(f), 2):
        rows += 1
        try:
            whole = float(row["whole_ms"])
            values = {name: float(row[name]) for name in stage_names}
        except (KeyError, TypeError, ValueError):
            bad.append({"line": line_no, "reason": "parse"})
            continue
        if not math.isfinite(whole) or any(not math.isfinite(v) for v in values.values()):
            bad.append({"line": line_no, "reason": "nonfinite"})
            continue
        if whole < 0 or any(v < 0 for v in values.values()):
            bad.append({"line": line_no, "reason": "negative"})
            continue
        finite_rows += 1
        # The stage clocks are nested inside the timed epoch call. A small
        # clock/readout mismatch is allowed, while the old printf type error
        # (billions of ms for a ~1 ms epoch) must fail loudly.
        stage_sum = sum(values.values())
        if any(v > 10.0 * max(whole, 1e-6) + 0.1 for v in values.values()):
            bad.append({"line": line_no, "reason": "stage_scale", "whole_ms": whole, **values})
        elif stage_sum > 2.0 * max(whole, 1e-6) + 0.1:
            bad.append({"line": line_no, "reason": "stage_sum", "whole_ms": whole,
                        "stage_sum_ms": stage_sum, **values})

result = {
    "file": path,
    "rows": rows,
    "finite_rows": finite_rows,
    "bad_rows": len(bad),
    "passed": not bad and rows > 0,
    "stages": stage_names,
    "rule": "finite nonnegative stages; each <= 10*whole+0.1 ms; sum <= 2*whole+0.1 ms",
    "bad": bad[:50],
}
print(json.dumps(result, indent=2, allow_nan=False))
if bad or rows == 0:
    raise SystemExit(1)
