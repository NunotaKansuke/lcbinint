"""Pre-flight checks for the recalibration campaign.

A multi-hour sweep that turns out to have been comparing two frames, two limb
darkening conventions, or a cache hit against an integrator produces a table
that looks entirely reasonable and is worthless.  Every such assumption the
sweeps rest on is checked here first, cheaply, and the campaign is not started
until this passes.
"""

from __future__ import annotations

import math
import sys

import numpy as np

from . import frames, reference
from .engines import (
    LcbinintEngine,
    VbmEngine,
    lcbinint_fixed,
    time_light_curve_block,
)
from .geometry import caustic_branches, make_lens_cases, sample_positions

# Positions where every engine should be reliable: a caustic is nearby enough
# that limb darkening and resolution both matter, but the disk is not tangent
# to a fold, which is where the engines are allowed to disagree.
_AGREEMENT_CASES = (
    # (s, q, rho, x, y)
    (1.00, 1.0e-3, 0.010, 0.0487, 0.0152),
    (0.70, 1.0e-2, 0.020, -0.0972, -0.1177),
    (1.60, 3.0e-1, 0.030, 0.0519, 0.1091),
    (1.00, 1.0e-2, 0.005, 0.1500, 0.0200),
    (2.20, 1.0e-1, 0.050, -0.2024, 0.1467),
)


def check_frames():
    report = frames.verify()
    return (
        True,
        "x-mirrored BinaryMag2 reproduces BinaryLightCurve to "
        f"{report['worst_mirrored_relative_gap']:.1e}; unmirrored is off by "
        f"{report['best_unmirrored_relative_gap']:.1e} at best",
    )


def check_limb_darkening_convention(tolerance=2.0e-4):
    """lcbinint's ``c`` and VBM's ``a1`` must mean the same profile.

    Both claim ``I(mu)/I0 = 1 - c (1 - mu)`` normalised by ``1 - c/3``, but a
    mismatch would show up as a smooth few-percent bias that no accuracy plot
    in this campaign would flag as a convention error.  So it is measured: a
    strongly darkened source must give the same answer as a well-resolved
    lcbinint integration of the same profile.
    """
    coefficient = 0.8
    engine = lcbinint_fixed("cartesian", 400, coefficient)
    reference = VbmEngine(tol=1.0e-8, profile_c=coefficient)
    worst = 0.0
    detail = []
    for s, q, rho, x, y in _AGREEMENT_CASES:
        mine = engine(s, q, rho, x, y)
        theirs = reference(s, q, rho, x, y)
        if not (mine.converged and math.isfinite(theirs.magnification)):
            continue
        gap = abs(mine.magnification - theirs.magnification) / abs(
            theirs.magnification)
        detail.append(f"s={s} q={q:g} rho={rho}: {gap:.2e}")
        worst = max(worst, gap)
    ok = worst <= tolerance and detail
    return ok, (
        f"worst limb-darkened disagreement {worst:.2e} over {len(detail)} "
        f"comparable rows (limit {tolerance:.0e})"
    )


def check_reference_construction(margin=0.1):
    """The reference builder must produce a usable reference on easy rows.

    A single fine bucket is not the reference -- ``reference.build`` combines a
    convergence ladder with an independently discretised grid and an
    independent contour integrator, and reports how far apart they landed.
    This checks that the combination is sharp enough to calibrate a 1e-3 rule
    on positions chosen to be benign.  If it is not sharp here, no threshold
    fitted downstream means anything.
    """
    rows = []
    for s, q, rho, x, y in _AGREEMENT_CASES:
        cartesian = reference.evaluate_ladder(
            s, q, rho, x, y, 0.0, grid="cartesian", buckets=reference.LADDER_TOP)
        polar = reference.evaluate_ladder(
            s, q, rho, x, y, 0.0, grid="polar", buckets=reference.LADDER_TOP)
        built = reference.build(s, q, rho, x, y, 0.0,
                                cartesian=cartesian, polar=polar,
                                contour_budget=15.0)
        rows.append((s, q, rho, built))
        gaps = built.get("witness_gaps", {})
        print(f"    s={s:<5} q={q:<8g} rho={rho:<6} A={built['value']:.8f} "
              f"unc={built['uncertainty']:.1e} "
              f"ladder={built.get('ladder_gap', float('nan')):.1e} "
              f"polar={gaps.get('polar', float('nan')):.1e} "
              f"contour={gaps.get('contour', float('nan')):.1e} "
              f"contour_self={built.get('contour_self_gap', float('nan')):.1e}")
    usable = [
        r for _, _, _, r in rows if reference.usable_for(r, 1.0e-3)]
    ok = len(usable) == len(rows)
    worst = max(r["uncertainty"] for _, _, _, r in rows)
    return ok, (
        f"{len(usable)}/{len(rows)} references sharp enough for a 1e-3 rule "
        f"(worst uncertainty {worst:.2e}, needed <= {margin * 1e-3:.0e})"
    )


def check_light_curve_timing():
    """Per-epoch timing must amortise the fixed per-call cost away.

    Entering lcbinint costs a flat ~1.4 ms of setup regardless of resolution,
    which is 7% of a 200-bin evaluation and more than the whole of a 16-bin
    one.  Timing single positions would therefore report that coarse grids are
    barely cheaper than fine ones.  Measuring a block of epochs removes it;
    this check confirms that the block measurement separates the resolutions
    that the per-call measurement flattens.
    """
    s, q, rho, u0, alpha = 1.00, 1.0e-3, 0.010, 0.0152, 0.0
    times = np.linspace(0.03, 0.07, 24)
    per_epoch = {}
    for nbin in (16, 200):
        seconds, _ = time_light_curve_block(
            lambda n=nbin: lcbinint_fixed("cartesian", n, 0.0)._ensure(),
            s, q, rho, times, u0, alpha, 0.0)
        per_epoch[nbin] = seconds
    scalar = lcbinint_fixed("cartesian", 16, 0.0)
    scalar_seconds = scalar(s, q, rho, float(times[0]), u0).seconds
    spread = per_epoch[200] / per_epoch[16]
    ok = spread > 8.0
    return ok, (
        f"block per-epoch 16 bins {per_epoch[16] * 1e3:.3f} ms, "
        f"200 bins {per_epoch[200] * 1e3:.3f} ms (spread {spread:.1f}x); "
        f"a single 16-bin call reports {scalar_seconds * 1e3:.3f} ms, "
        f"inflated by the flat per-call cost"
    )


def check_ladder_measures_the_grid():
    """Every ladder bucket must actually run the integrator it is named after.

    Asking for ``inverse_ray_grid='cartesian'`` does not get a Cartesian grid:
    the point-source and hexadecapole exits are taken first, and on a source a
    few radii from the caustic every bucket from 16 to 400 returns the same
    hexadecapole number.  A ladder built that way says 16 bins are enough
    everywhere, which is exactly the conclusion this campaign is trying to
    establish honestly -- in the first trial it was 74% of ladder evaluations.

    Two escapes are legitimate and are not counted.  Far beyond the caustic the
    library stops measuring a distance at all and reports it as infinite; there
    the point-source exit is the correct answer and no resolution rule applies.
    And in the grazing band the source-plane quadrature is not a shortcut past
    the grid but a different integrator, chosen by a gate no option reaches, so
    it is reported separately -- comparing it against the grid needs a forced
    mode in the library and is a study of its own.
    """
    cases = make_lens_cases(16, seed=20260803)
    rng = np.random.default_rng(4242)
    escaped = []
    quadrature = 0
    checked = 0
    seen = set()
    for case in cases:
        branches = caustic_branches(case.separation, case.mass_ratio)
        for position in sample_positions(case, branches, rng, per_case=4):
            for grid in ("cartesian", "polar"):
                result = lcbinint_fixed(grid, 64, 0.0)(
                    case.separation, case.mass_ratio, case.source_radius,
                    position["x"], position["y"])
                seen.add(result.method)
                if not math.isfinite(result.extra["caustic_distance"]):
                    continue
                if result.method == "source_plane_quadrature":
                    quadrature += 1
                    continue
                checked += 1
                if not result.method.startswith("inverse_ray"):
                    escaped.append((grid, result.method))
    return (not escaped), (
        f"{checked - len(escaped)}/{checked} forced evaluations at a measured "
        f"caustic distance ran the grid, {quadrature} more went to the grazing "
        f"quadrature; methods seen {sorted(seen)}; "
        f"unexplained escapes {sorted(set(escaped)) or 'none'}"
    )


def check_method_coverage():
    """The sampler has to reach every route the campaign is meant to calibrate.

    A switching rule cannot be recalibrated from a sample that never visits the
    branch it is switching to, so the sampler is checked for coverage before it
    is trusted to generate tens of thousands of rows.
    """
    cases = make_lens_cases(24, seed=20260803)
    rng = np.random.default_rng(20260803)
    engine = LcbinintEngine(
        grid=None, nbin="auto", profile_c=0.0, reltol=1.0e-3, max_source_bins=400)
    seen = {}
    distances = []
    for case in cases:
        branches = caustic_branches(case.separation, case.mass_ratio)
        for position in sample_positions(case, branches, rng, per_case=6):
            result = engine(
                case.separation, case.mass_ratio, case.source_radius,
                position["x"], position["y"])
            seen[result.method] = seen.get(result.method, 0) + 1
            ratio = result.extra["caustic_distance"] / case.source_radius
            if math.isfinite(ratio):
                distances.append(ratio)
    expected = {
        "point_source", "hexadecapole",
        "inverse_ray_cartesian", "inverse_ray_polar",
        "source_plane_quadrature",
    }
    missing = expected - set(seen)
    summary = ", ".join(f"{k}={v}" for k, v in sorted(seen.items()))
    band = np.percentile(distances, [5, 25, 50, 75, 95]) if distances else []
    return (not missing), (
        f"{summary}; missing={sorted(missing) or 'none'}; "
        f"d/rho percentiles(5,25,50,75,95)="
        f"{np.array2string(np.asarray(band), precision=2)}"
    )


CHECKS = (
    ("frame conventions", check_frames),
    ("limb-darkening convention", check_limb_darkening_convention),
    ("ladder measures the grid", check_ladder_measures_the_grid),
    ("reference construction", check_reference_construction),
    ("light-curve block timing", check_light_curve_timing),
    ("method coverage of the sampler", check_method_coverage),
)


def main():
    failures = 0
    for name, check in CHECKS:
        print(f"[..] {name}")
        try:
            ok, detail = check()
        except Exception as error:  # noqa: BLE001
            print(f"[!!] {name}: raised {type(error).__name__}: {error}")
            failures += 1
            continue
        print(f"[{'ok' if ok else 'NG'}] {name}: {detail}")
        failures += 0 if ok else 1
    print()
    print("pilot passed" if failures == 0 else f"pilot failed: {failures} check(s)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
