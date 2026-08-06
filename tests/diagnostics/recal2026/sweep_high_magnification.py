#!/usr/bin/env python3
"""More blocks where the magnification is high, because that is where it matters.

The main sweep samples geometry uniformly in its own parameters, which makes it
a good corpus for the nbin rule and a misleading one for the speed comparison:
about half its blocks land below A=3, where the contour is unbeatable and every
engine agrees, so the headline win rate is mostly a statement about the sampling
rather than about the method.  Split by magnification the picture is different,
and the highest bin -- A above 1000 -- is where lcbinint actually wins, on 42
blocks.  Forty-two is not enough to decide anything, and the bin below it
(300-1000, 50 blocks) dips in a way that is either real or noise.

This fills in those bins.  It reuses ``sweep_speed.evaluate_block`` verbatim so
the rows are the same shape, measured the same way, gated by the same reference
ladder, and can be concatenated with the existing corpus rather than compared
against it.  What changes is only where the positions are drawn from: small
sources sitting on or just inside the caustic, which is what produces high
magnification.  Linear limb darkening only -- the uniform comparison is already
settled, and settled negatively.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from .geometry import ANCHOR_MASS_RATIOS, ANCHOR_SEPARATIONS, LensCase, caustic_branches
from .sweep_speed import PROFILES, _initialise, evaluate_block

# The radii that dominate the existing corpus's high-magnification blocks: of
# the 214 blocks above A=1000, most sit at 1e-4.5 to 1e-4.
SOURCE_RADII = (3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3)

# Kept wide on purpose.  In the existing corpus the blocks above A=1000 are
# spread across factors from 0 to 1.7 rather than concentrated at zero, so
# magnification is not a function of approach distance alone -- it depends on
# which part of which caustic the sample landed on.  Choosing the geometry and
# hoping for the magnification does not work; a first attempt at radii of 3e-5
# and 1e-4 sitting exactly on the caustic produced A of 55 and 10.
DISTANCE_FACTORS = (0.0, 0.15, 0.35, 0.6, 0.8, 0.95, 1.1, 1.35)

# So the position is screened instead of predicted.  One loose evaluation costs
# milliseconds against roughly a hundred seconds for the block it gates, which
# makes rejection almost free and lets the sweep aim at a magnification band
# directly rather than at a geometry that might produce one.
SCREEN_TOLERANCE = 1.0e-3
MAX_SCREEN_ATTEMPTS = 600


def _candidate(case, branches, rng):
    """One position on a caustic normal, drawn the way the main sweep draws."""
    rho = case.source_radius
    factor = DISTANCE_FACTORS[rng.integers(len(DISTANCE_FACTORS))]
    branch = branches[rng.integers(len(branches))]
    index = int(rng.integers(len(branch)))
    point = branch[index]
    following = branch[(index + 1) % len(branch)]
    tangent = following - point
    norm = math.hypot(tangent[0], tangent[1])
    if norm <= 0.0:
        return None
    normal = np.array([-tangent[1] / norm, tangent[0] / norm])
    sign = 1.0 if rng.random() < 0.5 else -1.0
    jitter = (rng.random() - 0.5) * 0.5 * rho
    offset = (point + sign * factor * rho * normal + jitter * (tangent / norm))
    return {"x": float(offset[0]), "y": float(offset[1]),
            "intended_distance_factor": float(factor)}


def sample_positions(case, branches, rng, per_case, minimum_magnification):
    """Positions whose magnification is actually in the band being filled.

    Screened rather than constructed: see the note on DISTANCE_FACTORS.  The
    screen uses the shipping automatic path at a loose tolerance, which is not
    the quantity the block will later report but is far closer to it than any
    geometric proxy, and is wrong only in the direction of admitting a block
    whose measured magnification lands slightly below the cut.
    """
    from .engines import lcbinint_auto

    if not branches:
        return []
    screen = lcbinint_auto(SCREEN_TOLERANCE, PROFILES["linear"])
    positions, attempts = [], 0
    while len(positions) < per_case and attempts < MAX_SCREEN_ATTEMPTS:
        attempts += 1
        position = _candidate(case, branches, rng)
        if position is None:
            continue
        try:
            evaluation = screen(case.separation, case.mass_ratio,
                                case.source_radius, position["x"],
                                position["y"], time_it=False)
        except Exception:  # noqa: BLE001
            continue
        value = evaluation.magnification
        if not (math.isfinite(value) and value >= minimum_magnification):
            continue
        position["screened_magnification"] = float(value)
        positions.append(position)
    return positions


def make_cases(count, seed):
    """Lens geometries paired with the small radii this sweep is about."""
    rng = np.random.default_rng(seed)
    anchors = [(s, q) for s in ANCHOR_SEPARATIONS for q in ANCHOR_MASS_RATIOS]
    rng.shuffle(anchors)
    cases = []
    for index in range(count):
        s, q = anchors[index % len(anchors)]
        cases.append(LensCase(index, float(s), float(q),
                              float(SOURCE_RADII[index % len(SOURCE_RADII)])))
    return cases


def run_case(case, seed, per_case, budget, repeat, minimum_magnification):
    rng = np.random.default_rng(seed + 104729 * case.case_id)
    try:
        branches = caustic_branches(case.separation, case.mass_ratio)
    except Exception as error:  # noqa: BLE001
        return {"case": case.as_dict(),
                "error": f"caustics: {type(error).__name__}: {error}",
                "rows": []}
    rows = []
    for position in sample_positions(case, branches, rng, per_case,
                                     minimum_magnification):
        try:
            rows.append(evaluate_block(
                case, position, "linear", PROFILES["linear"],
                budget=budget, repeat=repeat))
        except Exception as error:  # noqa: BLE001
            rows.append({"case_id": case.case_id, "x": position["x"],
                         "y": position["y"], "profile": "linear",
                         "error": f"{type(error).__name__}: {error}"})
    return {"case": case.as_dict(), "rows": rows}


def _worker(payload):
    case, seed, per_case, budget, repeat, minimum, output = payload
    target = Path(output) / f"block-{case.case_id:05d}.json"
    if target.exists():
        return case.case_id, "skipped", 0.0
    started = time.perf_counter()
    result = run_case(case, seed, per_case, budget, repeat, minimum)
    result["seconds"] = time.perf_counter() - started
    result["core"] = sorted(os.sched_getaffinity(0))
    temporary = target.with_suffix(".partial")
    temporary.write_text(json.dumps(result))
    temporary.replace(target)
    return case.case_id, "done", result["seconds"]


def main():
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cases", type=int, default=60)
    parser.add_argument("--per-case", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--reference-budget", type=float, default=20.0)
    parser.add_argument("--minimum-magnification", type=float, default=300.0)
    parser.add_argument("--cores", required=True,
                        help="comma-separated core list, one worker per core")
    arguments = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    cores = [int(item) for item in arguments.cores.split(",")]
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)

    cases = make_cases(arguments.cases, arguments.seed)
    (output / "manifest.json").write_text(json.dumps({
        "seed": arguments.seed, "cases": arguments.cases,
        "per_case": arguments.per_case, "profiles": ["linear"],
        "source_radii": list(SOURCE_RADII),
        "distance_factors": list(DISTANCE_FACTORS),
        "reference_budget": arguments.reference_budget,
        "repeat": arguments.repeat, "cores": cores,
        "minimum_magnification": arguments.minimum_magnification,
        "purpose": "fill the high-magnification bins of the speed comparison",
        "lens_cases": [c.as_dict() for c in cases],
    }, indent=2))

    payloads = [(case, arguments.seed, arguments.per_case,
                 arguments.reference_budget, arguments.repeat,
                 arguments.minimum_magnification, str(output))
                for case in cases]

    print(f"{len(cases)} cases x {arguments.per_case} positions on "
          f"{len(cores)} cores, load before start: {os.getloadavg()[0]:.2f}",
          flush=True)
    started = time.perf_counter()
    counter = mp.Value("i", 0)
    lock = mp.Lock()
    done = 0
    with mp.Pool(len(cores), initializer=_initialise,
                 initargs=(cores, counter, lock)) as pool:
        for case_id, status, seconds in pool.imap_unordered(_worker, payloads):
            done += 1
            print(f"  [{done}/{len(cases)}] case {case_id} {status} "
                  f"{seconds:.0f}s", flush=True)
    print(f"done in {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
