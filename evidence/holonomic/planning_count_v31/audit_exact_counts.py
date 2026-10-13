#!/usr/bin/env python3
"""Exact rational Sturm counts for all physical-probe disagreements."""
import json
import re
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent


def trim(poly):
    while poly and poly[-1] == 0:
        poly.pop()
    return poly


def exact_count(coefficients):
    p = trim([Fraction.from_float(x) for x in coefficients])
    if len(p) <= 1:
        return 0
    chain = [p, [k*p[k] for k in range(1, len(p))]]
    while len(chain[-1]) > 1:
        remainder, divisor = chain[-2][:], chain[-1]
        while len(remainder) >= len(divisor):
            factor = remainder[-1] / divisor[-1]
            offset = len(remainder) - len(divisor)
            for j, coefficient in enumerate(divisor):
                remainder[offset+j] -= factor*coefficient
            trim(remainder)
        if not remainder:
            break
        chain.append([-x for x in remainder])
    signs = [1 if p[-1] > 0 else -1 for p in chain]
    negative = [s*(-1 if (len(p)-1) % 2 else 1) for s, p in zip(signs, chain)]
    variations = lambda values: sum(a != b for a, b in zip(values, values[1:]))
    return variations(negative) - variations(signs)


records = []
for line in (HERE / "physical_audit.log").read_text().splitlines():
    if not line.startswith(("QUARTIC_COUNT_DISAGREEMENT", "QUARTIC_COUNT_RECOVERY")):
        continue
    values = tuple(float(x) for x in line.split("coefficients=")[1].split())
    fast = int(re.search(r"(?:direct|count)=(\d+)", line).group(1))
    old = re.search(r"sturm=(\d+)", line)
    records.append((values, fast, int(old.group(1)) if old else None))
cache = {}
results = []
for coefficients, fast, old in records:
    if coefficients not in cache:
        cache[coefficients] = exact_count(coefficients)
    exact = cache[coefficients]
    results.append({"coefficients": coefficients, "fast": fast, "legacy": old, "exact": exact})
summary = {
    "physical_records": len(records), "unique_polynomials": len(cache),
    "fast_mismatches": sum(r["fast"] != r["exact"] for r in results),
    "legacy_certified_disagreements": sum(r["legacy"] is not None for r in results),
    "legacy_certified_wrong": sum(r["legacy"] is not None and r["legacy"] != r["exact"] for r in results),
    "legacy_unresolved": sum(r["legacy"] is None for r in results),
}
(HERE / "exact_count_audit.json").write_text(json.dumps({"summary": summary, "records": results}, indent=2)+"\n")
print(json.dumps(summary))
if summary["fast_mismatches"]:
    raise SystemExit(1)
