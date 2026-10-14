#!/usr/bin/env python3
"""Sanity checks for the diagnostic shadow and its completeness labels."""
import json
from pathlib import Path

here = Path(__file__).resolve().parent
d = json.loads((here / "summary.json").read_text())
assert d["base_commit"] == "dfe09dee62a99bbd18f3ce01246c84c6e130464b"
assert d["production_path_modified"] is False and d["hermite_used"] is False
assert len(d["results"]) == 5
for r in d["results"]:
    assert r["topology"]["status"] == 0
    assert r["topology"]["endpoint_sensitivity_map_valid"] is True
    fine_checks = [x for x in r["pointwise_fixed_R_fd_checks"] if x["step"] <= 1.0000001e-7]
    assert fine_checks
    assert all(x["all_samples_reliable"] for x in fine_checks)
    assert max(x["relative_difference"] for x in fine_checks) < 1e-6
    q8 = r["levels"][-1]
    assert q8["level"] == 8 and q8["nodes_per_cell"] == 255
    assert "ledger_diagnostics_level8" in r
    if r["case"] == "caustic-cross":
        assert q8["all_cells_complete"] is True
    else:
        assert q8["all_cells_complete"] is False
        assert r["invalid_sample_reason_counts_level8"] == {"DegenerateChart": 48}
print(f"validated {len(d['results'])} shadow cases; topology/pointwise checks pass")
