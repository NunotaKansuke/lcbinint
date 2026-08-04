#!/usr/bin/env python3
"""Which inverse-ray grid, decided on a quantity the pipeline already holds.

``speed_analysis`` reports the polar-over-Cartesian time ratio and finds its
median sitting on 1.00 in every cell.  That median is not the answer; it says
the two grids tie on most geometries, which leaves open the only question that
matters -- what to do on the geometries where they do not tie.

The cost of a switching rule is not its median ratio.  It is the total time the
rule spends over the corpus against the total an oracle would spend, and by that
measure always-Cartesian runs 1.29-1.45x oracle and always-polar 1.51-2.72x.
Splitting that residual by rho, by d/rho and by mass ratio gets nowhere: the
per-bucket penalties wander without order.  Splitting it by magnification lands
immediately.  At every accuracy and both profiles, the entire cost of
always-Cartesian sits in the top magnification quartile, and the three quartiles
below it are within 4% of oracle.

So the rule is a magnification threshold, and it has to be evaluated on the
magnification the pipeline knows *before* it picks a grid, not on the
finite-source answer it is trying to compute.  This scans both: the
finite-source magnification, which bounds what any such rule could achieve, and
the point-source magnification the multipole stage has already produced by the
time the grid is chosen, which is what an implementation can actually test.

Run against the speed sweep directory:

    python -m tests.diagnostics.recal2026.grid_switch \\
        tests/diagnostics/results/recal2026/speed_discovery \\
        --output tests/diagnostics/results/recal2026/grid_switch_rule.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .speed_analysis import ACCURACY_TARGETS, cheapest_meeting, load, _usable

# The scan is deliberately coarse and logarithmic.  A threshold read off this
# corpus to two significant figures would be reading its noise; what the corpus
# supports is the decade.
THRESHOLDS = (10.0, 20.0, 30.0, 50.0, 100.0, 200.0, 500.0, 1000.0)

PROFILES = ("uniform", "linear")


def _point_source_magnifications(rows):
    """The multipole stage's magnification for each row, keyed by row index.

    Recomputed rather than stored: the sweep records the converged
    finite-source answer, which is the wrong quantity to switch on.  A
    point-source evaluation is cheap next to the grids being compared.
    """
    import lcbinint

    native = lcbinint.LightCurve(lens="binary", options=lcbinint.Options(
        coordinates="vbm", nbin="auto", caustic_bins=1400, reltol=1.0e-3))
    cache = {}
    out = []
    for row in rows:
        key = (row["s"], row["q"], row["x"], row["y"], row["rho"],
               row["limb_darkening_c"])
        if key not in cache:
            info = native.info(row["x"], t0=0.0, tE=1.0, u0=row["y"],
                               alpha=0.0, s=row["s"], q=row["q"],
                               rho=row["rho"],
                               limb_darkening_c=row["limb_darkening_c"])
            cache[key] = float(
                np.asarray(info.point_source_magnifications).ravel()[0])
        out.append(cache[key])
    return out


def _pairs(rows, point, profile, target):
    """Blocks where both grids reached ``target``, with both magnifications."""
    out = []
    for row, point_magnification in zip(rows, point):
        if row.get("profile") != profile or not _usable(row, target):
            continue
        cartesian = cheapest_meeting(row, "lcbinint_cartesian", target)
        polar = cheapest_meeting(row, "lcbinint_polar", target)
        if cartesian is None or polar is None:
            continue
        out.append((point_magnification, float(row["magnification"]),
                    cartesian[0], polar[0]))
    return out


def scan(rows, point, profile, target):
    """Every threshold's corpus cost, against the oracle and the two constants."""
    pairs = _pairs(rows, point, profile, target)
    if not pairs:
        return None
    point_magnification = np.array([entry[0] for entry in pairs])
    finite_magnification = np.array([entry[1] for entry in pairs])
    cartesian = np.array([entry[2] for entry in pairs])
    polar = np.array([entry[3] for entry in pairs])
    oracle = float(np.minimum(cartesian, polar).sum())

    def cost(mask):
        return float(np.where(mask, polar, cartesian).sum()) / oracle

    report = {
        "blocks": len(pairs),
        "oracle_seconds": oracle,
        "always_cartesian": float(cartesian.sum()) / oracle,
        "always_polar": float(polar.sum()) / oracle,
        "thresholds": [],
    }
    for threshold in THRESHOLDS:
        report["thresholds"].append({
            "magnification": threshold,
            "point_source": cost(point_magnification > threshold),
            "point_source_share": float((point_magnification > threshold).mean()),
            "finite_source": cost(finite_magnification > threshold),
            "finite_source_share": float(
                (finite_magnification > threshold).mean()),
        })
    best = min(report["thresholds"], key=lambda entry: entry["point_source"])
    report["best_point_source_threshold"] = best["magnification"]
    report["best_point_source_cost"] = best["point_source"]
    return report


def summarise(directory):
    rows = load(directory)
    point = _point_source_magnifications(rows)
    result = {"directory": str(directory), "thresholds": list(THRESHOLDS)}
    for profile in PROFILES:
        per_target = {}
        for target in ACCURACY_TARGETS:
            report = scan(rows, point, profile, target)
            if report is not None:
                per_target[repr(target)] = report
        if per_target:
            result[profile] = per_target
    return result


def _print(result):
    for profile in PROFILES:
        if profile not in result:
            continue
        print(f"\n===== {profile} =====")
        for target, report in result[profile].items():
            print(f"\n-- delivered accuracy {float(target):g}   "
                  f"{report['blocks']} blocks")
            print(f"   always-Cartesian {report['always_cartesian']:.3f}x oracle"
                  f"   always-polar {report['always_polar']:.3f}x")
            print("   polar when A >     point-source        finite-source")
            for entry in report["thresholds"]:
                print(f"     {entry['magnification']:8g}"
                      f"     {entry['point_source']:.3f}x "
                      f"({entry['point_source_share']:5.1%})"
                      f"     {entry['finite_source']:.3f}x "
                      f"({entry['finite_source_share']:5.1%})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--output", help="write the scan as json")
    arguments = parser.parse_args()
    result = summarise(arguments.directory)
    _print(result)
    if arguments.output:
        Path(arguments.output).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
