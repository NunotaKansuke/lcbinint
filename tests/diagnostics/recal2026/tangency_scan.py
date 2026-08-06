"""How far the two grids and the contour integrator fall apart near tangency.

The arbiter in ``tangency_arbiter`` settles individual disputed epochs.  This
module answers the prior question: how wide is the band, and how deep does the
disagreement go below the 1e-3 threshold that made those epochs visible in the
first place?

Three independent estimates are taken at every sampled position -- lcbinint's
Cartesian inverse-ray grid at the top of the ladder, lcbinint's polar grid at
the top of the ladder, and VBMicrolensing's contour integrator at a relative
tolerance far below either.  Two of the three share lcbinint's certificate and
flood fill but not each other's discretisation; the third shares nothing.  A
position where all three agree tells us little, and is the common case.  A
position where exactly one dissents names the party that is wrong, which is the
measurement this module produces.

Positions are placed on the local outward normal of the caustic at controlled
multiples of the source radius, and the *achieved* distance is read back from
lcbinint rather than assumed: the normal step overshoots wherever the caustic
curves, and near a cusp it overshoots badly.  Binning on the intended factor
rather than the achieved one would smear the band being measured.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

# Distance factors, dense across the band the disputed epochs occupy and sparse
# outside it.  The outside points are not padding: without them there is no
# baseline to say the band is special, only an assertion that it is.
DISTANCE_FACTORS = (
    0.30, 0.50, 0.65, 0.75, 0.80, 0.85, 0.90, 0.94, 0.97, 0.99,
    1.00, 1.01, 1.03, 1.06, 1.10, 1.15, 1.20, 1.28, 1.35, 1.45,
    1.60, 1.80, 2.10, 2.60, 3.50, 5.00,
)

# The contour witness's relative tolerance.  Two decades below the tightest
# accuracy this campaign calibrates, so a dissent at 1e-4 cannot be the
# witness's own truncation.
CONTOUR_RELATIVE = 1.0e-8
CONTOUR_ABSOLUTE_FLOOR = 1.0e-12


def _pin(cores):
    try:
        index = int(os.environ.get("RECAL_WORKER_INDEX", "0"))
    except ValueError:
        index = 0
    if not cores:
        return None
    core = cores[index % len(cores)]
    try:
        os.sched_setaffinity(0, {core})
    except OSError:
        return None
    return core


def _initialise(cores, counter, lock):
    with lock:
        os.environ["RECAL_WORKER_INDEX"] = str(counter.value)
        counter.value += 1
    _pin(cores)


def positions_for(case, factors=DISTANCE_FACTORS, per_factor=3, seed=0):
    """Normal offsets from sampled caustic points, both signs offered.

    Which sign leaves the caustic depends on where on the branch the sample
    landed, so both are emitted and the achieved distance decides which one was
    the outward step; a position that ended up closer than intended is kept and
    recorded at its achieved distance rather than discarded.
    """
    from .geometry import caustic_branches

    branches = caustic_branches(case["s"], case["q"])
    if not branches:
        return []
    rng = np.random.default_rng(seed)
    rho = case["rho"]
    out = []
    for factor in factors:
        for _ in range(per_factor):
            branch = branches[rng.integers(len(branches))]
            index = int(rng.integers(len(branch)))
            point = branch[index]
            following = branch[(index + 1) % len(branch)]
            tangent = following - point
            norm = math.hypot(tangent[0], tangent[1])
            if norm <= 0.0:
                continue
            normal = np.array([-tangent[1] / norm, tangent[0] / norm])
            for sign in (1.0, -1.0):
                offset = point + sign * factor * rho * normal
                out.append({"x": float(offset[0]), "y": float(offset[1]),
                            "intended_distance_factor": float(factor)})
    return out


def three_witnesses(s, q, rho, x, y, profile_c, *, bucket=400,
                    contour_relative=CONTOUR_RELATIVE):
    """Cartesian grid, polar grid, contour -- values plus what they claim."""
    from .engines import VbmEngine, lcbinint_fixed

    entry = {}
    for name, grid in (("cartesian", "cartesian"), ("polar", "polar")):
        engine = lcbinint_fixed(grid, bucket, profile_c)
        started = time.perf_counter()
        result = engine(s, q, rho, x, y)
        entry[name] = {
            "value": result.magnification,
            "error_estimate": result.error_estimate,
            "certified": bool(result.support_proven),
            "converged": bool(result.converged),
            "method": result.method,
            "seconds": time.perf_counter() - started,
        }
        if name == "cartesian":
            entry["caustic_distance"] = result.extra["caustic_distance"]
            entry["point_magnification"] = result.extra["point_magnification"]
            entry["image_count"] = result.extra["image_count"]

    contour = VbmEngine(tol=CONTOUR_ABSOLUTE_FLOOR, profile_c=profile_c,
                        reltol=contour_relative)
    started = time.perf_counter()
    try:
        value = contour(s, q, rho, x, y).magnification
    except Exception as error:  # noqa: BLE001
        value = float("nan")
        entry["contour_error"] = f"{type(error).__name__}: {error}"
    entry["contour"] = {"value": value,
                        "seconds": time.perf_counter() - started}
    return entry


def dissent(entry):
    """Which of the three is the outlier, and by how much.

    Reported as each party's distance from the *median* of the three.  The
    median is used rather than a nominated truth because that is the whole
    point: no party is privileged here, and taking any one of them as the
    baseline would decide the question the measurement is asking.
    """
    values = {
        name: entry[name]["value"]
        for name in ("cartesian", "polar", "contour")
        if math.isfinite(entry[name]["value"])
    }
    if len(values) < 3:
        return {"status": f"only {len(values)} finite witnesses"}
    middle = float(np.median(list(values.values())))
    scale = max(abs(middle), 1.0)
    gaps = {name: abs(value - middle) / scale for name, value in values.items()}
    spread = (max(values.values()) - min(values.values())) / scale
    outlier = max(gaps, key=gaps.get)
    # A dissent is only meaningful if the other two actually agree; three
    # values scattered evenly name no outlier, they name a hard position.
    others = sorted(gap for name, gap in gaps.items() if name != outlier)
    return {
        "status": "ok",
        "median": middle,
        "spread": spread,
        "gaps": gaps,
        "outlier": outlier,
        "outlier_gap": gaps[outlier],
        "agreement_of_others": others[-1] if others else float("nan"),
    }


def _worker(payload):
    case, profiles, per_factor, seed, output, bucket = payload
    target = Path(output) / f"case-{case['case_id']:05d}.json"
    if target.exists():
        return case["case_id"], "skipped", 0.0

    started = time.perf_counter()
    rows = []
    for position in positions_for(case, per_factor=per_factor,
                                  seed=seed + case["case_id"]):
        for profile_name, profile_c in profiles:
            try:
                entry = three_witnesses(
                    case["s"], case["q"], case["rho"],
                    position["x"], position["y"], profile_c, bucket=bucket)
            except Exception as error:  # noqa: BLE001
                rows.append({**position, "profile": profile_name,
                             "error": f"{type(error).__name__}: {error}"})
                continue
            rows.append({
                **position,
                "profile": profile_name,
                "limb_darkening_c": profile_c,
                "achieved_distance_factor":
                    entry["caustic_distance"] / case["rho"],
                "witnesses": entry,
                "dissent": dissent(entry),
            })
    result = {"case": case, "rows": rows,
              "seconds": time.perf_counter() - started,
              "core": sorted(os.sched_getaffinity(0))}
    temporary = target.with_suffix(".partial")
    temporary.write_text(json.dumps(result))
    temporary.replace(target)
    return case["case_id"], "done", result["seconds"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cases", type=int, default=60)
    parser.add_argument("--per-factor", type=int, default=2)
    parser.add_argument("--bucket", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--cores", default="")
    parser.add_argument("--profiles", default="uniform,linear")
    arguments = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")

    from .engines import PROFILES
    from .geometry import make_lens_cases

    profiles = [(name.strip(), PROFILES[name.strip()])
                for name in arguments.profiles.split(",") if name.strip()]
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)

    if arguments.cores:
        cores = [int(part) for part in arguments.cores.split(",")]
    else:
        cores = sorted(os.sched_getaffinity(0))[-arguments.workers:]

    cases = [
        {"case_id": case.case_id, "s": case.separation,
         "q": case.mass_ratio, "rho": case.source_radius}
        for case in make_lens_cases(arguments.cases, arguments.seed)
    ]
    (output / "manifest.json").write_text(json.dumps({
        "seed": arguments.seed, "cases": arguments.cases,
        "per_factor": arguments.per_factor, "bucket": arguments.bucket,
        "distance_factors": list(DISTANCE_FACTORS),
        "contour_relative": CONTOUR_RELATIVE,
        "profiles": [name for name, _ in profiles],
    }, indent=1))

    payloads = [
        (case, profiles, arguments.per_factor, arguments.seed,
         str(output), arguments.bucket)
        for case in cases
    ]
    counter = mp.Value("i", 0)
    lock = mp.Lock()
    started = time.perf_counter()
    with mp.Pool(min(arguments.workers, len(cores)), initializer=_initialise,
                 initargs=(cores, counter, lock)) as pool:
        for done, (case_id, status, seconds) in enumerate(
                pool.imap_unordered(_worker, payloads), 1):
            print(f"[{done}/{len(payloads)}] case {case_id} {status} "
                  f"{seconds:.1f}s  (elapsed {time.perf_counter()-started:.0f}s)",
                  flush=True)


if __name__ == "__main__":
    main()
