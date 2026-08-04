#!/usr/bin/env python3
"""A small, blunt speed comparison against VBMicrolensing and microlux.

This is deliberately not ``sweep_speed``.  That sweep exists to derive the
library's own rules -- when to switch grids, how many bins a tolerance needs --
and a rule that will be applied to every geometry a user has has to be measured
over every geometry we can afford.  A cross-tool comparison answers a different
and much smaller question: on the axes where the answer is expected to change
sign, does it, and by how much.  Measuring that to three significant figures
over three thousand geometries would be precision nobody reads.

The axes are the ones where the sign is expected to move:

* limb darkening, because the contour methods integrate a darkened profile by
  stacking annuli while a grid method pays nothing extra for it;
* the source radius, because it sets how much of the grid is source and how
  much of the contour is arc;
* the magnification, because a large magnification means a source near or on a
  caustic, which is where a contour has the most work to do.

The magnification axis is the *point-source* magnification, not the finite one.
A finite source saturates: at ``rho = 0.1`` no geometry of any lens reaches a
magnification of a thousand, so a finite-source axis would stop three of the
radii at different places and the rows could not be compared across them.  The
point-source value is a property of where the source is rather than of how big
it is, so the same seven positions exist at every radius, and it is also the
number lcbinint's own multipole stage holds at the moment it picks a method --
the same quantity the grid-switching rule is written against.

Everything else is held fixed, and the fixed values are written down here
rather than swept.  Held-fixed is not the same as irrelevant: ``s`` and ``q``
below place a central caustic in the source plane, and a different pair would
move the numbers.  The claim this benchmark supports is about the shape of the
comparison across the three axes above, at one representative lens.

Two lcbinint columns are reported, and the distinction is the one thing here
that must not be collapsed.  The shipping path answers a distant epoch with a
hexadecapole or a point source and never builds a grid, so timing it and
calling the result "inverse-ray speed" would be timing the absence of an
inverse ray.  The second column disables those exits and forces the integrator
at every epoch.  What a user pays and what our quadrature costs against theirs
are different questions; the ratios are taken against the forced column,
because that is the one comparing an integration to an integration.

For the same reason the timed block is narrow -- see ``BLOCK_SPAN_IN_RADII`` --
and the route the shipping path actually took is printed on every row.  A block
that used more than one method is marked ``MIXED``, because its per-epoch
median is then an average over two different computations.

Accuracy is reported next to every timing.  A speed comparison in which the
engines did not agree to the tolerance they were asked for is a comparison of
two different computations, and the accuracy column is what lets a reader see
that they were not.

Run it with::

    taskset -c 40 python -m tests.diagnostics.recal2026.bench_tools \\
        --output tests/diagnostics/results/recal2026/bench_tools.json
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from .engines import lcbinint_auto, time_light_curve_block, time_vbm_block
# The reference machinery is the speed sweep's, not a second one written here.
# It climbs a resolution ladder on both grids and cross-checks the result
# against a contour integration, and it returns an uncertainty -- so a row
# whose reference cannot resolve the tolerance under test says so instead of
# reporting a confident wrong error.  A benchmark that graded four engines
# against its own privately-built truth would be grading them against lcbinint.
from .sweep_speed import achieved_error, block_reference, reference_floor

# One planetary binary with a central caustic.  Nothing here sweeps the lens:
# see the module docstring for what that does and does not licence.
LENS_S = 1.1
LENS_Q = 1.0e-3

# Uniform, and the linear coefficient the campaign uses everywhere else.
PROFILES = (("uniform", 0.0), ("linear", 0.5))

SOURCE_RADII = (1.0e-1, 1.0e-2, 1.0e-3)

# Point-source magnification at the block's centre.  Targeted rather than
# stumbled upon: the impact parameter is chosen to land near each of these.
# The top of the range is where a small source is expected to win, so it is
# worth reaching even though it puts the trajectory inside the central caustic.
MAGNIFICATION_TARGETS = (1.5, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
TARGET_TOLERANCE = 0.30

# Two block lengths, because two of the four engines are traced and vectorised:
# their per-epoch cost falls as the block grows, and a single block length
# would report one point on that curve as if it were the cost.
#
# The long block is ``(SHORT - 1) * STRIDE + 1`` rather than a round number so
# that its epochs land exactly on the short block's, five to one.  That is what
# lets one reference serve both: the reference is by far the most expensive
# thing here -- a converged ladder per epoch -- and building it twice for the
# same geometry would double the run to compare each length against a slightly
# different set of positions, which is worse as well as slower.
SHORT_BLOCK = 30
BLOCK_STRIDE = 5
LONG_BLOCK = (SHORT_BLOCK - 1) * BLOCK_STRIDE + 1

# Repeats of the whole block, per block length.  The short block is repeated
# because thirty epochs is few enough that one slow epoch moves the median; the
# long block is not, because a hundred and forty-six epochs have already
# averaged over that, and repeating it would cost as much again as the entire
# rest of the run for a digit that does not move.
REPEATS = {SHORT_BLOCK: 2, LONG_BLOCK: 1}

# Width of a block, in source radii.  This has to be narrow, and the reason is
# the whole point of the benchmark.  A block four radii wide crosses out of the
# regime it was placed in: some of its epochs are close enough to the caustic
# to need the integrator and some are far enough to be answered by a
# hexadecapole or by a point source, and the median per-epoch time over that
# mixture is not the speed of anything.  It would also be a mixture in a
# different proportion for each engine, because the engines do not take their
# shortcuts at the same places.  Narrow keeps one block in one regime, which is
# what makes a row comparable across the four engines.  Same value, for the
# same reason, as the speed sweep's.
BLOCK_SPAN_IN_RADII = 0.4

TOLERANCE = 1.0e-4

# Seconds each reference epoch may spend climbing its resolution ladder.  At
# the sweep's own twenty the floor came out around 1e-5 against a tolerance of
# 1e-4, which is close enough that an error at the tolerance could not be told
# from the reference's own uncertainty.  It is a cap, not a cost: an easy
# geometry converges long before it.
REFERENCE_BUDGET = 45.0


def _impact_parameter_for(target, rho, profile_c):
    """The impact parameter whose central point-source magnification is ``target``.

    Solved by scanning rather than inverted.  A single lens would give
    ``u = 1/A`` to start from, but this is a binary and the whole reason the
    large targets are interesting is that they put the trajectory near the
    central caustic, which is exactly where that approximation stops holding.
    A scan costs a second and is right everywhere.

    The magnification is read at the block's centre rather than at its peak.
    The peak of a block that crosses a caustic is a number set by where the
    sampling happened to land relative to the divergence, so two radii asked
    for the same peak would not be at comparable positions; the centre is a
    property of the trajectory alone.
    """
    curve = lcbinint_auto(1.0e-3, profile_c)._ensure()
    centre = np.array([0.0])

    best = None
    for u0 in np.geomspace(3.0e-5, 2.0, 160):
        info = curve.info(centre, t0=0.0, tE=1.0, u0=float(u0), alpha=0.0,
                          s=LENS_S, q=LENS_Q, rho=rho,
                          limb_darkening_c=profile_c)
        point = float(np.asarray(info.point_source_magnifications).ravel()[0])
        gap = abs(point - target) / target
        if best is None or gap < best[0]:
            best = (gap, float(u0), point)
    return best


def _error(values, references):
    """Worst relative gap to the reference epochs, or ``None`` if unmeasured."""
    result = achieved_error(np.asarray(values, dtype=float).ravel(), references)
    return result["worst"] if result["epochs"] else None


def _time_lcbinint(s, q, rho, times, u0, profile_c, references, repeat):
    seconds, info = time_light_curve_block(
        lambda: lcbinint_auto(TOLERANCE, profile_c)._ensure(),
        s, q, rho, times, u0, 0.0, profile_c, repeat=repeat)
    values = np.asarray(info.finite_source_magnifications, dtype=float)
    return {
        "seconds_per_epoch": seconds,
        "error": _error(values, references),
        # Which routes the automatic path took.  A row where it fell to the
        # multipole is a row where lcbinint is fast for a reason that has
        # nothing to do with its grid, and reading it as a grid result would be
        # the same mistake as comparing a point-source shortcut against a
        # contour integration.
        "methods": sorted({str(name) for name in np.asarray(
            info.finite_source_method_names).ravel()}),
    }


def _time_vbm(s, q, rho, times, u0, profile_c, references, repeat):
    # RelTol is the criterion; the absolute Tol is pinned below anything
    # reachable so it never stops the integration first.  Leaving Tol at a
    # usable value would make VBM chase an absolute accuracy nobody asked for
    # exactly on the high-magnification rows this benchmark is built around,
    # and would report that as VBM being slow.  Same convention as sweep_speed.
    from . import reference

    seconds, values = time_vbm_block(
        reference.CONTOUR_ABSOLUTE_FLOOR, TOLERANCE,
        s, q, rho, times, u0, 0.0, profile_c, repeat=repeat)
    return {"seconds_per_epoch": seconds, "error": _error(values, references)}


def _time_microlux(s, q, rho, times, u0, profile_c, references, repeat):
    from .engines_ext import MicroluxEngine

    engine = MicroluxEngine(tol=TOLERANCE, strategy=(60, 60, 120, 240, 480),
                            profile_c=profile_c)
    started = time.perf_counter()
    result = engine.time_block(s, q, rho, times, u0, 0.0, repeat=repeat)
    return {
        "seconds_per_epoch": result["seconds_per_epoch"],
        "error": _error(result["values"], references),
        # microlux's sampler has a fixed budget; when it runs out it warns and
        # returns what it has.  A point that hit the ceiling is not a point on
        # the accuracy/speed front and has to be marked as such.
        "budget_exhausted": result["budget_exhausted"],
        "first_call_seconds": result["first_call_seconds"],
        "total_seconds": time.perf_counter() - started,
    }


def _time_lcbinint_jax(s, q, rho, times, u0, profile_c, references, repeat):
    from .engines_ext import LcbinintJaxEngine

    engine = LcbinintJaxEngine(reltol=TOLERANCE, profile_c=profile_c)
    result = engine.time_block(s, q, rho, times, u0, 0.0, repeat=repeat)
    return {
        "seconds_per_epoch": result["seconds_per_epoch"],
        "error": _error(result["values"], references),
        "first_call_seconds": result["first_call_seconds"],
    }


def _time_lcbinint_grid(s, q, rho, times, u0, profile_c, references, repeat):
    """The integrator alone, with every cheap exit disabled.

    ``lcbinint`` above is the shipping path and will answer a distant epoch
    with a hexadecapole or a point source, which is the right thing for it to
    do and the wrong thing to put in a column headed "inverse-ray speed".  This
    column forces the grid at every epoch so the integrator is what is being
    timed.  The two columns answer different questions -- what a user pays, and
    what our quadrature costs against theirs -- and neither substitutes for the
    other, so both are reported.
    """
    from .engines import LcbinintEngine

    seconds, info = time_light_curve_block(
        lambda: LcbinintEngine(grid=None, nbin="auto", profile_c=profile_c,
                               reltol=TOLERANCE, max_source_bins=400,
                               force_grid=True)._ensure(),
        s, q, rho, times, u0, 0.0, profile_c, repeat=repeat)
    values = np.asarray(info.finite_source_magnifications, dtype=float)
    return {
        "seconds_per_epoch": seconds,
        "error": _error(values, references),
        "methods": sorted({str(name) for name in np.asarray(
            info.finite_source_method_names).ravel()}),
    }


ENGINES = (
    ("lcbinint", _time_lcbinint),
    ("lcbinint_grid", _time_lcbinint_grid),
    ("vbm", _time_vbm),
    ("microlux", _time_microlux),
    ("lcbinint_jax", _time_lcbinint_jax),
)


def geometries():
    """Every (profile, rho, magnification) position this benchmark times.

    The impact parameter is solved here rather than inside the timing loop, so
    the scan -- which is itself a few seconds of magnification evaluations --
    cannot land in a measurement.
    """
    out = []
    for profile, profile_c in PROFILES:
        for rho in SOURCE_RADII:
            for target in MAGNIFICATION_TARGETS:
                gap, u0, point = _impact_parameter_for(target, rho, profile_c)
                entry = {"profile": profile, "limb_darkening_c": profile_c,
                         "rho": rho, "target": target}
                if gap > TARGET_TOLERANCE:
                    entry["skipped"] = f"unreachable: nearest {point:.4g}"
                else:
                    entry["u0"] = u0
                    entry["point_magnification"] = point
                out.append(entry)
    return out


def _block(rho, epochs):
    span = BLOCK_SPAN_IN_RADII * rho
    return np.linspace(-0.5 * span, 0.5 * span, epochs)


def run_geometry(geometry):
    """Both block lengths at one position, against one shared reference."""
    if "skipped" in geometry:
        return [geometry]
    rho, u0, profile_c = (geometry["rho"], geometry["u0"],
                          geometry["limb_darkening_c"])

    short = _block(rho, SHORT_BLOCK)
    references = block_reference(LENS_S, LENS_Q, rho, short, u0, 0.0,
                                 profile_c, budget=REFERENCE_BUDGET)
    resolved = [entry["value"] for entry in references.values()
                if entry.get("status") == "ok"]
    floor = reference_floor(references)

    out = []
    for epochs in (SHORT_BLOCK, LONG_BLOCK):
        times = _block(rho, epochs)
        # The long block's epochs are a superset of the short one's, so the
        # reference for short-block epoch ``k`` is the reference for long-block
        # epoch ``k * BLOCK_STRIDE``; ``_block`` producing that superset is
        # exactly what ``LONG_BLOCK``'s odd value buys.
        stride = BLOCK_STRIDE if epochs == LONG_BLOCK else 1
        mapped = {index * stride: entry for index, entry in references.items()}

        cell = dict(geometry, epochs=epochs)
        cell["reference_floor"] = floor
        cell["reference_epochs"] = len(resolved)
        # The finite-source magnification the block actually reaches, kept
        # alongside the point-source value it was placed by: the gap between
        # the two is the finite-source suppression, which is itself part of the
        # story at the large radii.
        cell["magnification"] = (float(np.median(resolved)) if resolved
                                 else float("nan"))
        cell["engines"] = {}
        for name, function in ENGINES:
            try:
                cell["engines"][name] = function(
                    LENS_S, LENS_Q, rho, times, u0, profile_c, mapped,
                    REPEATS[epochs])
            except Exception as error:  # noqa: BLE001
                cell["engines"][name] = {
                    "failed": f"{type(error).__name__}: {error}"}
            entry = cell["engines"][name]
            # Per-epoch is the comparable number, but a reader budgeting a fit
            # wants the wall time of a curve, so the block total is carried too.
            if entry.get("seconds_per_epoch"):
                entry["seconds_per_block"] = (entry["seconds_per_epoch"]
                                              * epochs)
        out.append(cell)
    return out


def _print(results):
    """Absolute milliseconds per epoch first; ratios are a convenience column.

    The absolute numbers are the ones a reader can act on -- they say what a
    light curve costs -- and they are also the ones that need the machine
    stated next to them, which the JSON carries.  The ratios are printed second
    because they are what the absolute numbers imply, not a separate result.
    """
    print(f"lens s={LENS_S} q={LENS_Q}, reltol={TOLERANCE:g}, "
          f"cores={sorted(os.sched_getaffinity(0))}")
    print("A_pt = point-source magnification at block centre (the axis); "
          "A_fs = finite-source magnification reached")
    print("route = what the shipping path did; 'grid' column forces the "
          "integrator, and the ratios are taken against it\n")
    header = (f"{'prof':<7}{'rho':>7}{'A_pt':>8}{'A_fs':>8}{'ep':>5}  "
              f"{'lcbinint':>9}{'grid':>9}{'vbm':>9}{'microlux':>9}{'jax':>9}  "
              f"{'vbm/gr':>7}{'mlx/gr':>7}  {'route':<14}errors (worst rel.)")
    print(header)
    print("-" * len(header))
    for cell in results:
        if "skipped" in cell:
            print(f"{cell['profile']:<7}{cell['rho']:>7.3g}"
                  f"{cell['target']:>8.3g}{'-':>8}{'-':>5}  {cell['skipped']}")
            continue
        engines = cell["engines"]

        def ms(name):
            value = engines.get(name, {}).get("seconds_per_epoch")
            return f"{value * 1e3:9.3f}" if value else f"{'-':>9}"

        def ratio(name):
            # Against the forced-grid column, not the shipping one: a ratio to
            # a block that took the point-source exit would say how much faster
            # a contour integration is than not integrating.
            ours = engines.get("lcbinint_grid", {}).get("seconds_per_epoch")
            theirs = engines.get(name, {}).get("seconds_per_epoch")
            return f"{theirs / ours:7.2f}" if ours and theirs else f"{'-':>7}"

        # One block, one regime -- so a block that used more than one method is
        # a block whose per-epoch median is an average over two different
        # computations, and the span needs narrowing rather than the number
        # reporting.  Printed rather than silently averaged.
        taken = engines.get("lcbinint", {}).get("methods") or ["?"]
        route = "+".join(name.split("_")[0][:6] for name in taken)
        if len(taken) > 1:
            route = "MIXED:" + route

        short = {"lcbinint": "lcbi", "lcbinint_grid": "grid", "vbm": "vbm",
                 "microlux": "mlux", "lcbinint_jax": "jax"}
        errors = " ".join(
            f"{short[name]}={entry['error']:.0e}"
            # A microlux point that ran out of sampling budget is not a point
            # on the accuracy/speed front, and has to be legible as such.
            + ("!" if entry.get("budget_exhausted") else "")
            for name, entry in engines.items()
            if entry.get("error") is not None)
        # An error smaller than the reference's own uncertainty is not a
        # measurement of the engine, so the row is flagged rather than read.
        if not cell["reference_floor"] <= 0.1 * TOLERANCE:
            errors += f"  [REF FLOOR {cell['reference_floor']:.0e}]"
        print(f"{cell['profile']:<7}{cell['rho']:>7.3g}"
              f"{cell['point_magnification']:>8.4g}"
              f"{cell['magnification']:>8.4g}{cell['epochs']:>5}  "
              f"{ms('lcbinint')}{ms('lcbinint_grid')}{ms('vbm')}"
              f"{ms('microlux')}{ms('lcbinint_jax')}"
              f"  {ratio('vbm')}{ratio('microlux')}  {route:<14}{errors}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--jax-cache", default=None,
                        help="directory for JAX's persistent compilation cache")
    parser.add_argument("--cores", default="40",
                        help="comma-separated cores to pin to")
    arguments = parser.parse_args()

    # Pinned, and pinned before anything imports a threaded library.  The
    # numbers here are absolute milliseconds, and an absolute millisecond that
    # was measured while the scheduler moved the process between two sockets is
    # not a measurement of the engine.  How many cores is a choice; that it is
    # a fixed set is not.
    cores = {int(item) for item in arguments.cores.split(",") if item.strip()}
    os.sched_setaffinity(0, cores)

    from .engines_ext import configure_jax
    configure_jax(arguments.jax_cache)

    plan = geometries()
    live = [item for item in plan if "skipped" not in item]
    print(f"{len(plan)} positions, {len(live)} reachable, "
          f"{2 * len(live)} timed cells", flush=True)
    results = []
    started = time.perf_counter()
    for index, geometry in enumerate(plan, 1):
        results.extend(run_geometry(geometry))
        print(f"  [{index}/{len(plan)}] {geometry['profile']} "
              f"rho={geometry['rho']:g} A*={geometry['target']:g}  "
              f"{time.perf_counter() - started:7.1f}s", flush=True)

    print()
    _print(results)
    if arguments.output:
        with open(arguments.output, "w") as stream:
            json.dump({"lens": {"s": LENS_S, "q": LENS_Q},
                       "tolerance": TOLERANCE,
                       # Absolute milliseconds mean nothing without saying
                       # what they were measured on, and every engine here is
                       # pinned to one core so none of them is quietly using
                       # sixty-four.
                       "cores": sorted(os.sched_getaffinity(0)),
                       "cells": results}, stream,
                      indent=1)
        print(f"\nwrote {arguments.output}")


if __name__ == "__main__":
    main()
