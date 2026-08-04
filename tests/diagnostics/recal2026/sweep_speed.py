#!/usr/bin/env python3
"""Accuracy-versus-time sweep: the campaign's headline measurement.

The question is not "which engine is faster", which has no answer, but "at a
given achieved accuracy, which engine is faster, and where does that change".
So no engine is run at a nominal tolerance chosen to match another's.  Each is
swept across its own accuracy knob and lands somewhere on the (achieved error,
seconds per epoch) plane; the comparison is between the resulting Pareto
fronts.

lcbinint is timed through ``LightCurve.info``, which returns the caustic
distances, image counts, per-epoch error estimates and method names alongside
the magnifications.  That is measured to cost the same as ``magnification``
alone to within 0.2% at every resolution, so reporting ``info`` timings does not
handicap lcbinint against engines that return a bare array.

The unit of measurement is a contiguous block of epochs, not a single position.
Entering lcbinint costs a flat per-call setup that is larger than a whole
16-bin evaluation, so per-call timing reports coarse and fine grids as nearly
equally fast -- a harness artefact that would erase the resolution axis this
campaign exists to measure.  Blocks are short enough that magnification varies
little across one, which is what lets the results be plotted against
magnification, and every engine compared here is a light-curve API anyway.

Achieved error is measured against a reference built on a few epochs of the
block by ``reference.build``, so a block whose reference is not sharp enough
for a given error floor is reported as such rather than being given a number.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from . import reference
from .engines import (
    BUCKETS,
    DEFAULT_BLOCK,
    PROFILES,
    lcbinint_fixed,
    time_light_curve_block,
    time_vbm_block,
)
from .geometry import caustic_branches, make_lens_cases, sample_positions

# Resolution buckets timed on the speed sweep.  The full ladder is measured for
# accuracy by ``sweep_resolution``; here a spread that spans the range at
# roughly constant log spacing is enough to place the cost curve, and the saved
# time buys more geometries instead.
TIMED_BUCKETS = (16, 24, 32, 50, 64, 100, 160, 256, 400)

# lcbinint's automatic path is timed at the tolerances the new rule is for.
TIMED_TOLERANCES = (1.0e-2, 1.0e-3, 1.0e-4)

# VBMicrolensing is swept on RelTol with Tol pinned below anything reachable,
# so the relative criterion is the one that stops it.  Sweeping the absolute
# Tol instead would penalise VBM enormously on exactly the high-magnification
# rows this campaign cares most about, for a precision nobody asked for.
TIMED_VBM_RELATIVE = (1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7)

# Epochs of the block that carry a reference.  Four is enough to see whether
# the error is uniform across the block or concentrated at one end, which is
# the failure mode that would make a block-median error misleading.
REFERENCE_EPOCHS = 4

# The block spans this many source radii, centred on the sampled position.  Wide
# enough that the epochs are genuinely distinct evaluations, narrow enough that
# the magnification stays within a factor of a few, so the block has a
# well-defined place on the magnification axis.
BLOCK_SPAN_IN_RADII = 0.4


def _relative(value, target):
    return abs(value - target) / max(abs(target), 1.0)


def block_reference(s, q, rho, times, u0, alpha, profile_c, *, budget):
    """References on a few epochs of the block, with their uncertainties.

    Uses only the top of the resolution ladder: the speed sweep needs to know
    what the right answer is, not how coarse a grid could have found it.  That
    is ``sweep_resolution``'s job, and duplicating it here would double the cost
    of the more expensive of the two sweeps.
    """
    from .frames import position_from_trajectory

    indices = np.unique(np.linspace(
        0, len(times) - 1, REFERENCE_EPOCHS).astype(int))
    entries = {}
    for index in indices:
        x, y = position_from_trajectory(u0, alpha, float(times[index]))
        cartesian = reference.evaluate_ladder(
            s, q, rho, x, y, profile_c, grid="cartesian",
            buckets=reference.LADDER_TOP, time_budget=budget)
        polar = reference.evaluate_ladder(
            s, q, rho, x, y, profile_c, grid="polar",
            buckets=reference.LADDER_TOP, time_budget=budget)
        contour = reference.contour_reference(
            s, q, rho, x, y, profile_c, time_budget=budget)
        entries[int(index)] = reference.build(
            s, q, rho, x, y, profile_c, cartesian=cartesian, polar=polar,
            contour=contour["value"], contour_self_gap=contour["self_gap"])
    return entries


def achieved_error(values, references):
    """How far one engine's block landed from the reference epochs.

    Reported as the worst and the median of the sampled epochs.  The worst is
    what a tolerance claim has to honour; the median is what a plot of typical
    behaviour should show, and the two differing is itself a result.
    """
    gaps = []
    for index, entry in references.items():
        if entry.get("status") != "ok":
            continue
        value = float(values[index])
        if not math.isfinite(value):
            gaps.append(float("inf"))
            continue
        gaps.append(_relative(value, entry["value"]))
    if not gaps:
        return {"worst": float("nan"), "median": float("nan"), "epochs": 0}
    return {
        "worst": float(max(gaps)),
        "median": float(np.median(gaps)),
        "epochs": len(gaps),
    }


def reference_floor(references):
    """The smallest error this block's references can distinguish from zero."""
    uncertainties = [
        entry["uncertainty"] for entry in references.values()
        if entry.get("status") == "ok"
        and math.isfinite(entry.get("uncertainty", math.inf))
    ]
    return float(max(uncertainties)) if uncertainties else float("inf")


def evaluate_block(case, position, profile_name, profile_c, *, budget, repeat,
                   seconds_cap=0.25):
    """One geometry, every engine, every knob setting."""
    s, q, rho = case.separation, case.mass_ratio, case.source_radius
    x0, y0 = position["x"], position["y"]
    # alpha=0 makes the abscissa the time axis, so a block is a span in x at
    # fixed y; frames.verify() confirms both engines read it the same way.
    u0, alpha = y0, 0.0
    span = BLOCK_SPAN_IN_RADII * rho
    times = np.linspace(x0 - 0.5 * span, x0 + 0.5 * span, DEFAULT_BLOCK)

    started = time.perf_counter()
    row = {
        "case_id": case.case_id,
        "s": s, "q": q, "rho": rho,
        "x": x0, "y": y0, "u0": u0, "alpha": alpha,
        "profile": profile_name,
        "limb_darkening_c": profile_c,
        "intended_distance_factor": position["intended_distance_factor"],
        "block_epochs": int(DEFAULT_BLOCK),
        "engines": [],
    }

    references = block_reference(
        s, q, rho, times, u0, alpha, profile_c, budget=budget)
    row["reference_floor"] = reference_floor(references)
    row["references"] = {
        str(index): {
            "value": entry.get("value"),
            "uncertainty": entry.get("uncertainty"),
            "status": entry.get("status"),
        }
        for index, entry in references.items()
    }

    def record(kind, knob, seconds, values, extra=None):
        entry = {
            "engine": kind,
            "knob": knob,
            "seconds_per_epoch": seconds,
            "error": achieved_error(values, references),
        }
        if extra:
            entry.update(extra)
        row["engines"].append(entry)

    for grid in ("cartesian", "polar"):
        for bucket in TIMED_BUCKETS:
            # Cost grows roughly with the square of the bin count, so once a
            # bucket is this slow the finer ones are minutes of wall time for a
            # point that is already far off the useful end of the Pareto front.
            # Stopping is recorded, because a missing point must not be read as
            # a fast one.
            if (row["engines"] and row["engines"][-1].get("engine")
                    == f"lcbinint_{grid}"
                    and row["engines"][-1].get("seconds_per_epoch", 0.0)
                    > seconds_cap):
                row["engines"].append({
                    "engine": f"lcbinint_{grid}", "knob": bucket,
                    "censored_after_seconds_per_epoch":
                        row["engines"][-1]["seconds_per_epoch"]})
                break
            try:
                seconds, info = time_light_curve_block(
                    lambda g=grid, b=bucket: lcbinint_fixed(
                        g, b, profile_c)._ensure(),
                    s, q, rho, times, u0, alpha, profile_c, repeat=repeat)
            except Exception as error:  # noqa: BLE001
                row["engines"].append({
                    "engine": f"lcbinint_{grid}", "knob": bucket,
                    "failed": f"{type(error).__name__}: {error}"})
                continue
            values = np.asarray(info.finite_source_magnifications, dtype=float)
            errors = np.asarray(
                info.finite_source_error_estimates, dtype=float)
            methods = sorted({str(name) for name in np.asarray(
                info.finite_source_method_names).ravel()})
            record(f"lcbinint_{grid}", bucket, seconds, values, {
                # Recorded even though lcbinint_fixed forces the grid, because
                # a bucket that escaped to a cheaper route would otherwise look
                # like a spectacularly fast grid rather than a bypassed one.
                "methods": methods,
                # A block is only usable as evidence if every epoch in it was
                # certified; one unproven epoch makes the whole block's timing
                # a measurement of an answer the library would refuse to give.
                "certified_epochs": int(np.isfinite(errors).sum()),
                "self_reported_error": float(np.nanmax(errors))
                if np.isfinite(errors).any() else float("inf"),
            })

    for tolerance in TIMED_TOLERANCES:
        import lcbinint

        def factory(t=tolerance):
            return lcbinint.LightCurve(
                lens="binary",
                options=lcbinint.Options(
                    coordinates="vbm", nbin="auto", caustic_bins=1400,
                    max_source_bins=400, reltol=t))

        try:
            seconds, info = time_light_curve_block(
                factory, s, q, rho, times, u0, alpha, profile_c, repeat=repeat)
        except Exception as error:  # noqa: BLE001
            row["engines"].append({
                "engine": "lcbinint_auto", "knob": tolerance,
                "failed": f"{type(error).__name__}: {error}"})
            continue
        values = np.asarray(info.finite_source_magnifications, dtype=float)
        errors = np.asarray(info.finite_source_error_estimates, dtype=float)
        methods = [str(name) for name in
                   np.asarray(info.finite_source_method_names).ravel()]
        bins = np.asarray(info.finite_source_refinement_levels, dtype=float)
        record("lcbinint_auto", tolerance, seconds, values, {
            "certified_epochs": int(np.isfinite(errors).sum()),
            "self_reported_error": float(np.nanmax(errors))
            if np.isfinite(errors).any() else float("inf"),
            "methods": sorted(set(methods)),
            "refinement_max": float(np.nanmax(bins)) if bins.size else float("nan"),
        })

    for reltol in TIMED_VBM_RELATIVE:
        try:
            seconds, values = time_vbm_block(
                reference.CONTOUR_ABSOLUTE_FLOOR, reltol,
                s, q, rho, times, u0, alpha, profile_c, repeat=repeat)
        except Exception as error:  # noqa: BLE001
            row["engines"].append({
                "engine": "vbm", "knob": reltol,
                "failed": f"{type(error).__name__}: {error}"})
            continue
        record("vbm", reltol, seconds, values)

    finite = [
        float(entry.get("value", float("nan")))
        for entry in references.values() if entry.get("status") == "ok"
    ]
    row["magnification"] = float(np.median(finite)) if finite else float("nan")
    if finite:
        row["magnification_spread"] = float(max(finite) / max(min(finite), 1e-30))
    row["row_seconds"] = time.perf_counter() - started
    return row


def run_case(case, seed, per_case, budget, repeat, profiles, seconds_cap=0.25):
    rng = np.random.default_rng(seed + 104729 * case.case_id)
    try:
        branches = caustic_branches(case.separation, case.mass_ratio)
    except Exception as error:  # noqa: BLE001
        return {"case": case.as_dict(),
                "error": f"caustics: {type(error).__name__}: {error}",
                "rows": []}
    rows = []
    for position in sample_positions(case, branches, rng, per_case):
        for profile_name in profiles:
            try:
                rows.append(evaluate_block(
                    case, position, profile_name, PROFILES[profile_name],
                    budget=budget, repeat=repeat, seconds_cap=seconds_cap))
            except Exception as error:  # noqa: BLE001
                rows.append({
                    "case_id": case.case_id,
                    "x": position["x"], "y": position["y"],
                    "profile": profile_name,
                    "error": f"{type(error).__name__}: {error}",
                })
    return {"case": case.as_dict(), "rows": rows}


def _pin_to_own_core(cores):
    """Give this worker one core to itself.

    Timing is the measurement here, so workers must not migrate between cores
    or share one.  Without pinning the per-epoch numbers pick up the scheduler's
    decisions, which vary with what else the machine is doing and would make the
    Pareto fronts unreproducible.
    """
    if not cores:
        return None
    try:
        index = int(os.environ.get("RECAL_WORKER_INDEX", "0"))
    except ValueError:
        index = 0
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
    _pin_to_own_core(cores)


def _worker(payload):
    case, seed, per_case, budget, repeat, profiles, output = payload
    target = Path(output) / f"block-{case.case_id:05d}.json"
    if target.exists():
        return case.case_id, "skipped", 0.0
    started = time.perf_counter()
    result = run_case(case, seed, per_case, budget, repeat, profiles)
    result["seconds"] = time.perf_counter() - started
    result["core"] = sorted(os.sched_getaffinity(0))
    temporary = target.with_suffix(".partial")
    temporary.write_text(json.dumps(result))
    temporary.replace(target)
    return case.case_id, "done", result["seconds"]


def available_cores(requested):
    """Cores this sweep may use, leaving the rest of the machine alone."""
    cores = sorted(os.sched_getaffinity(0))
    if requested and requested < len(cores):
        # Take from the top, so a sweep started while other work already holds
        # the low-numbered cores is less likely to land on top of it.
        cores = cores[-requested:]
    return cores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cases", type=int, default=120)
    parser.add_argument("--per-case", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--reference-budget", type=float, default=20.0)
    parser.add_argument("--profiles", default="uniform,linear")
    arguments = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    profiles = [p.strip() for p in arguments.profiles.split(",") if p.strip()]
    cores = available_cores(arguments.workers)

    cases = make_lens_cases(arguments.cases, arguments.seed)
    (output / "manifest.json").write_text(json.dumps({
        "seed": arguments.seed,
        "cases": arguments.cases,
        "per_case": arguments.per_case,
        "profiles": profiles,
        "block_epochs": DEFAULT_BLOCK,
        "block_span_in_radii": BLOCK_SPAN_IN_RADII,
        "repeat": arguments.repeat,
        "timed_buckets": list(TIMED_BUCKETS),
        "timed_tolerances": list(TIMED_TOLERANCES),
        "timed_vbm_relative": list(TIMED_VBM_RELATIVE),
        "all_buckets": list(BUCKETS),
        "cores": cores,
        "lens_cases": [c.as_dict() for c in cases],
    }, indent=2))

    payloads = [
        (case, arguments.seed, arguments.per_case, arguments.reference_budget,
         arguments.repeat, profiles, str(output))
        for case in cases
    ]

    import multiprocessing

    counter = multiprocessing.Value("i", 0)
    lock = multiprocessing.Lock()
    started = time.perf_counter()
    done = 0
    with multiprocessing.Pool(
            arguments.workers, initializer=_initialise,
            initargs=(cores, counter, lock)) as pool:
        for case_id, status, elapsed in pool.imap_unordered(_worker, payloads):
            done += 1
            rate = done / max(time.perf_counter() - started, 1e-9)
            remaining = (len(payloads) - done) / max(rate, 1e-9)
            print(f"[{done}/{len(payloads)}] case {case_id} {status} "
                  f"{elapsed:.1f}s  eta {remaining / 60:.1f} min", flush=True)
    print(f"finished in {(time.perf_counter() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
