#!/usr/bin/env python3
"""Summarize independent direct-integral FD data for rand035."""
import csv
import json
import math
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read_records(path):
    with path.open() as f:
        return [line.rstrip("\n").split("\t") for line in f if line.strip()]


raw = []
for filename in ("reference.tsv", "reference_x_high.tsv"):
    raw.extend(read_records(HERE / filename))
topo = [r for r in raw if r[0] == "TOPO"]
integrals = [r for r in raw if r[0] == "REF"]
fds = [r for r in raw if r[0] == "FD5"]
if len(topo) != 32 or any(int(r[7]) != 0 for r in topo):
    raise SystemExit(f"fresh cold topology audit failed/incomplete: {len(topo)} rows")
if len(integrals) != 144 or any(int(r[12]) != 0 for r in integrals):
    raise SystemExit(f"direct reference integrals failed/incomplete: {len(integrals)} rows")

by_fd = {(r[3], int(r[4]), int(r[6])): (float(r[7]), float(r[8])) for r in fds}
comparisons = []
for axis in ("X", "Y"):
    final_res = 6 if axis == "X" else 5
    for u, quantity_index in ((0.0, 0), (0.5, 1)):
        selected = by_fd[(axis, 2, final_res)][quantity_index]
        middle = by_fd[(axis, 1, final_res)][quantity_index]
        larger = by_fd[(axis, 0, 4)][quantity_index]
        previous = by_fd[(axis, 2, final_res - 1)][quantity_index]
        previous_middle = by_fd[(axis, 1, final_res - 1)][quantity_index]
        spread = max(abs(selected - middle), abs(selected - previous),
                     abs(middle - previous_middle))
        comparisons.append({
            "case": "rand035", "u": u, "axis": axis,
            "reference": {
                "value_gradient": selected,
                "observed_stability_spread": spread,
                "estimator": "5-point central FD of direct physical-value integral",
                "chosen_resolution": {
                    "radial_order": 160 if final_res == 6 else 128,
                    "radial_subdivisions": 32 if final_res == 6 else 16,
                    "angular_nodes": 1024,
                },
                "chosen_step": 0.00025,
            },
            "cross_checks": {
                "same_resolution_mid_step": middle,
                "coarser_resolution_same_step": previous,
                "coarser_resolution_mid_step": previous_middle,
                "large_step_resolution4": larger,
            },
        })

summary = {
    "phase": "Phase72 moving-map independent rand035 FD reference",
    "reference_method": {
        "fresh_cold_topology_rows": len(topo),
        "all_topology_status_ok": True,
        "direct_value_integrals": len(integrals),
        "all_direct_integrals_ok": True,
        "finite_difference": "5-point central difference with fresh topology at each +/-h and +/-2h",
        "observable": "direct phi and independently reconstructed angular boundary; no production analytic gradient or K-rule",
        "note": "Observed stability spread is diagnostic, not a formal interval enclosure.",
    },
    "comparisons": comparisons,
}
(HERE / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps({"summary": str(HERE / "summary.json"), "comparisons": len(comparisons),
                  "topology_rows": len(topo), "direct_integrals": len(integrals)}, indent=2))
