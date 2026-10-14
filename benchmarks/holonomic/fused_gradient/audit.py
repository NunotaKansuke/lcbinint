"""Compare the isolated tangent experiment against the original value lane."""
import csv
import json
import math
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
def read(name):
    with (root / name).open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))
def key(row):
    return tuple(row[k] for k in ("case", "profile", "dbin", "epoch"))

baseline = {key(r): r for r in read("full_baseline.tsv")}
rows = read("full_fused.tsv")
value = {key(r): r for r in rows if r["mode"] == "0"}
gradient = {key(r): r for r in rows if r["mode"] == "3"}
assert baseline.keys() == value.keys() == gradient.keys()
fields = ("mu", "value_converged", "stop", "nodes", "support")
def changed(other):
    return [k for k in baseline if any(baseline[k][f] != other[k][f] for f in fields)]
ratios = [float(gradient[k]["total_ms"]) / float(value[k]["total_ms"]) for k in baseline]
ordered = sorted(ratios)
differences = [abs(float(gradient[k]["mu"]) - float(baseline[k]["mu"])) for k in baseline]
violations = [k for k, d in zip(baseline, differences)
              if d > 1e-8 + 1e-4 * abs(float(baseline[k]["mu"]))]
status_changes = [k for k in baseline if any(baseline[k][f] != gradient[k][f]
                  for f in ("value_converged", "stop", "support"))]
report = {
    "inputs": len(baseline),
    "original_vs_overlay_value_changes": len(changed(value)),
    "original_vs_fused_value_changes": len(changed(gradient)),
    "fused_primal_differences_exceeding_value_tolerance": len(violations),
    "fused_primal_status_changes": len(status_changes),
    "fused_primal_max_scaled_difference": max(d / max(1., abs(float(baseline[k]["mu"])))
                                              for k, d in zip(baseline, differences)),
    "paired_total_median_ratio": statistics.median(ratios),
    "paired_total_sum_ratio": sum(float(r["total_ms"]) for r in gradient.values()) / sum(float(r["total_ms"]) for r in value.values()),
    "paired_frontier_sum_ratio": sum(float(r["frontier_ms"]) for r in gradient.values()) / sum(float(r["frontier_ms"]) for r in value.values()),
    "paired_total_p95_ratio": ordered[int(.95 * (len(ordered) - 1))],
    "paired_total_max_ratio": max(ratios),
    "inputs_over_1_3": sum(r > 1.3 for r in ratios),
    "fused_vs_original_total_sum_ratio": sum(float(r["total_ms"]) for r in gradient.values()) / sum(float(r["total_ms"]) for r in baseline.values()),
    "overlay_vs_original_total_sum_ratio": sum(float(r["total_ms"]) for r in value.values()) / sum(float(r["total_ms"]) for r in baseline.values()),
    "all_five_derivatives_finite": sum(all(math.isfinite(float(r[f"g{j}"])) for j in range(5)) for r in gradient.values()),
    "gradient_precision_checked": False,
    "gradient_precision_certified": False,
}
report["profiles"] = {}
for profile in ("uniform", "linear"):
    keys = [k for k in baseline if k[1] == profile]
    pr = [float(gradient[k]["total_ms"]) / float(value[k]["total_ms"]) for k in keys]
    report["profiles"][profile] = {
        "inputs": len(keys), "median_ratio": statistics.median(pr),
        "sum_ratio": sum(float(gradient[k]["total_ms"]) for k in keys) /
                     sum(float(value[k]["total_ms"]) for k in keys),
        "inputs_over_1_3": sum(r > 1.3 for r in pr),
    }
(root / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
assert not changed(value), "experimental overlay changed the value lane"
assert not violations, "fused primal difference exceeds the unchanged value tolerance"
assert not status_changes, "fused gradients changed the primal status"
