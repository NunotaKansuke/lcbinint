#!/usr/bin/env python3
"""Fast native regression checks for the three tangency correctness defects.

This consumes the already-computed independent arbiter values; it never treats
VBMicrolensing alone as ground truth.  The expensive source-plane integration
does not need to be repeated to check a native implementation change.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "tests/diagnostics/results/recal2026"
MODULE = "tests.diagnostics.recal2026.tangency_correctness"
os.environ.setdefault("LCBININT_PROBE_BUILD", str(ROOT / "build"))
from . import probe_build  # noqa: E402

probe_build.activate()
from .engines import lcbinint_fixed  # noqa: E402
C_CASE = {
    "s": 1.3,
    "q": 1.0e-4,
    "rho": 0.0015515122255040925,
    "x": 0.0014470002051158422,
    "y": 0.0006812879524093549,
    "limb_darkening_c": 0.0,
}
C_ORDER_CASE = {
    "s": 1.05,
    "q": 0.001,
    "rho": 0.012087561335986492,
    "x": 0.010495776906606516,
    "y": 0.004729643382853728,
    "limb_darkening_c": 0.0,
}


def relative_gap(value, reference):
    return abs(value - reference) / max(abs(reference), 1.0)


def evaluate(case, grid, nbin):
    engine = lcbinint_fixed(grid, nbin, case.get("limb_darkening_c", 0.0))
    result = engine(case["s"], case["q"], case["rho"], case["x"], case["y"])
    return {
        "value": result.magnification,
        "estimated_error": result.error_estimate,
        "certified": result.support_proven,
        "converged": result.converged,
        "method": result.method,
        "image_count": result.extra["image_count"],
        "seconds": result.seconds,
    }


def arbiter_corpus(path, expected):
    rows = json.loads(path.read_text())
    selected = [row for row in rows if row["verdict"]["status"] == "decided"]
    if expected == "vbm":
        selected = [
            row for row in selected
            if row["verdict"]["closest"] == "vbm_contour"
            and row["arbiter"]["self_gap"]
            <= 0.1 * row["verdict"]["party_spread"]
        ]
    output = []
    for index, row in enumerate(selected, 1):
        case = row["case"]
        arbiter = row["arbiter"]["value"]
        measured = {grid: evaluate(case, grid, 400)
                    for grid in ("cartesian", "polar")}
        for entry in measured.values():
            entry["relative_gap_to_arbiter"] = relative_gap(entry["value"], arbiter)
        output.append({
            "index": index,
            "case": case,
            "arbiter": {
                "value": arbiter,
                "self_gap": row["arbiter"]["self_gap"],
                "party_spread": row["verdict"]["party_spread"],
                "original_closest": row["verdict"]["closest"],
            },
            "measured": measured,
        })
        print(
            f"[{index}/{len(selected)}] {path.stem}: "
            f"cart={measured['cartesian']['relative_gap_to_arbiter']:.3e} "
            f"polar={measured['polar']['relative_gap_to_arbiter']:.3e}",
            flush=True,
        )
    return output


def _subprocess_c(policy, order=None, case=C_CASE, legacy=False):
    env = os.environ.copy()
    env["LCBININT_PROBE_POLICY"] = policy
    if order:
        env["LCBININT_DIAGNOSTIC_SEED_ORDER"] = order
    else:
        env.pop("LCBININT_DIAGNOSTIC_SEED_ORDER", None)
    env["LCBININT_DIAGNOSTIC_CASE"] = json.dumps(case)
    if legacy:
        env["LCBININT_DIAGNOSTIC_UNSORTED_SEEDS"] = "1"
    else:
        env.pop("LCBININT_DIAGNOSTIC_UNSORTED_SEEDS", None)
    command = [sys.executable, "-m", MODULE, "--single-c"]
    completed = subprocess.run(
        command, cwd=ROOT, env=env, check=True, text=True,
        stdout=subprocess.PIPE)
    return json.loads(completed.stdout)


def defect_c():
    # Offset depth changes only how many certified probe seeds are appended.
    shallow = _subprocess_c("rings=0,offsets=1")
    deeper = _subprocess_c("rings=0,offsets=2")
    normal = _subprocess_c(
        "rings=0,offsets=7", order="jacobian-ascending", case=C_ORDER_CASE)
    reverse = _subprocess_c(
        "rings=0,offsets=7", order="jacobian-descending", case=C_ORDER_CASE)
    legacy_normal = _subprocess_c(
        "rings=0,offsets=7", order="jacobian-ascending",
        case=C_ORDER_CASE, legacy=True)
    legacy_reverse = _subprocess_c(
        "rings=0,offsets=7", order="jacobian-descending",
        case=C_ORDER_CASE, legacy=True)
    return {
        "case": C_CASE,
        "seed_addition": {
            "shallow": shallow,
            "deeper": deeper,
            "delta": deeper["value"] - shallow["value"],
        },
        "seed_order": {
            "case": C_ORDER_CASE,
            "forward": normal,
            "reverse": reverse,
            "delta": reverse["value"] - normal["value"],
        },
        "legacy_seed_order_reproduction": {
            "case": C_ORDER_CASE,
            "forward": legacy_normal,
            "reverse": legacy_reverse,
            "delta": legacy_reverse["value"] - legacy_normal["value"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-c", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--seed-order-scan", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-corpus", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.single_c:
        case = json.loads(os.environ.get("LCBININT_DIAGNOSTIC_CASE", "null")) or C_CASE
        nbin = 16 if case == C_ORDER_CASE else 128
        print(json.dumps(evaluate(case, "cartesian", nbin)))
        return
    if args.seed_order_scan:
        policy = os.environ.get("LCBININT_PROBE_POLICY", "")
        offset = next(
            (item.split("=", 1)[1] for item in policy.split(",")
             if item.startswith("offsets=")), None)
        scan_source = f"cert_{offset}.json" if offset else "full.json"
        stored = json.loads((RESULTS / "probe" / scan_source).read_text())
        changed = []
        for row in stored["rows"]:
            prior = (row.get("measured") or {}).get("16") or {}
            if prior.get("method") != "inverse_ray_cartesian":
                continue
            current = evaluate(row, "cartesian", 16)
            delta = current["value"] - prior["magnification"]
            if abs(delta) > 1.0e-12 * max(abs(current["value"]), 1.0):
                changed.append({
                    "row_id": row["row_id"],
                    "case": {
                        key: row.get(key, 0.0)
                        for key in C_CASE
                    },
                    "forward": prior["magnification"],
                    "reverse": current["value"],
                    "delta": delta,
                })
        print(json.dumps(changed))
        return

    started = time.perf_counter()
    payload = {"defect_c": defect_c()}
    if not args.skip_corpus:
        payload["defect_a"] = arbiter_corpus(
            RESULTS / "tangency_arbitration.json", "cartesian")
        payload["defect_b_all"] = arbiter_corpus(
            RESULTS / "tangency_arbitration_vbm.json", "all")
        payload["defect_b"] = [
            row for row in payload["defect_b_all"]
            if row["arbiter"]["original_closest"] == "vbm_contour"
        ]
    if args.check:
        c = payload["defect_c"]
        assert abs(c["seed_addition"]["delta"]) <= 1.0e-12
        assert abs(c["seed_order"]["delta"]) <= 1.0e-12
        assert abs(c["legacy_seed_order_reproduction"]["delta"]) > 1.0e-3
        if not args.skip_corpus:
            assert len(payload["defect_a"]) == 11
            assert len(payload["defect_b"]) == 9
            for row in payload["defect_a"]:
                measured = row["measured"]["cartesian"]
                assert measured["certified"]
                assert measured["method"] == "inverse_ray_cartesian"
                assert measured["relative_gap_to_arbiter"] <= max(
                    row["arbiter"]["self_gap"], 3.0e-4)
            for row in payload["defect_b"]:
                for grid in ("cartesian", "polar"):
                    measured = row["measured"][grid]
                    assert measured["certified"]
                    assert measured["method"] == "source_plane_quadrature"
                    assert measured["relative_gap_to_arbiter"] <= 1.0e-4
    payload["seconds"] = time.perf_counter() - started
    rendered = json.dumps(payload, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n")
        print(f"wrote {args.output}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
