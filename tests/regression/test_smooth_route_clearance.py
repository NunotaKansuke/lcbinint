"""The smooth-expansion routes must never be chosen inside a caustic.

`point_source` and `hexadecapole` both expand the
magnification about the disk centre.  Every one of them is wrong by an amount
that has nothing to do with its own error estimate if a caustic passes through
the disk, because an image pair is created or destroyed inside the integration
domain.  The router keeps them out of that regime with calibrated local proxies
(ghost and planetary indicators, and a release distance in units of rho).

Those constants are calibrations and this test does not pin any of them.  What
it pins is the property they exist to deliver, stated in terms the component
certificate can prove: no smooth route may be chosen for a disk that the
caustic provably enters.  "Provably" is the certificate's own bound -- the
distance from the centre to the caustic *polyline*, minus the local sagitta,
which is a lower bound on the distance to the true caustic because a chord can
only lie on the near side of the arc it subtends.

The sample is deliberately concentrated where the decision is made: every trial
starts at a random caustic vertex and steps a random zero-to-thirty rho away.
Uniform sampling over a box lands there almost never.
"""

import numpy as np
import pytest


CAUSTIC_POINTS = 1400


def _options(lcbinint, lens, **extra):
    if lens == "binary":
        extra["coordinates"] = "center_of_mass"
    return lcbinint.Options(caustic_bins=CAUSTIC_POINTS, **extra)


def _polyline(lcbinint, params, lens="binary"):
    curve = lcbinint.LightCurve(lens=lens, options=_options(lcbinint, lens))
    branches = curve.caustics(
        {"t0": 0.0, "tE": 1.0, "u0": 0.0, "alpha": 0.0, "rho": 0.0, **params},
        n_points=CAUSTIC_POINTS,
    )
    return [
        (np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        for x, y in zip(branches.x, branches.y)
        if len(x) >= 3
    ]


def _clearance(branches, px, py):
    """Lower bound on the distance from (px, py) to the true caustic."""
    best = np.inf
    for xs, ys in branches:
        x2, y2 = np.roll(xs, -1), np.roll(ys, -1)
        dx, dy = x2 - xs, y2 - ys
        length2 = dx * dx + dy * dy
        length2[length2 == 0.0] = 1.0e-300
        t = np.clip(((px - xs) * dx + (py - ys) * dy) / length2, 0.0, 1.0)
        segment = np.hypot(px - (xs + t * dx), py - (ys + t * dy))
        # Sagitta of the arc each chord subtends, valid at cusps too.
        sagitta = np.hypot(
            np.roll(xs, 1) - 2.0 * xs + x2, np.roll(ys, 1) - 2.0 * ys + y2)
        margin = np.maximum(sagitta, np.roll(sagitta, -1))
        best = min(best, float(np.min(segment - margin)))
    return best


SMOOTH_ROUTES = frozenset(
    {"point_source", "hexadecapole"})


def _sweep(lcbinint, rng, geometries, per_geometry, draw_lens, lens="binary"):
    curve = lcbinint.LightCurve(
        lens=lens, options=_options(lcbinint, lens, nbin="auto"))
    violations = []
    smooth = 0
    for _ in range(geometries):
        params = draw_lens(rng)
        branches = _polyline(lcbinint, params, lens=lens)
        if not branches:
            continue
        vertices_x = np.concatenate([xs for xs, _ in branches])
        vertices_y = np.concatenate([ys for _, ys in branches])
        for _ in range(per_geometry):
            rho = float(np.exp(rng.uniform(np.log(3.0e-4), np.log(3.0e-2))))
            index = int(rng.integers(vertices_x.size))
            angle = float(rng.uniform(0.0, 2.0 * np.pi))
            step = float(rng.uniform(0.0, 30.0)) * rho
            x = float(vertices_x[index]) + step * np.cos(angle)
            y = float(vertices_y[index]) + step * np.sin(angle)
            info = curve.info(
                [x], t0=0.0, tE=1.0, u0=y, alpha=0.0, rho=rho, **params)
            method = info.finite_source_method_names[0]
            if method not in SMOOTH_ROUTES:
                continue
            smooth += 1
            bound = _clearance(branches, x, y)
            if bound < rho:
                violations.append(
                    f"{method} params={params} x={x!r} y={y!r} rho={rho!r} "
                    f"clearance/rho={bound / rho:+.4f}"
                )
    return smooth, violations


def _draw_binary(rng):
    return {
        "s": float(np.exp(rng.uniform(np.log(0.4), np.log(2.5)))),
        "q": float(np.exp(rng.uniform(np.log(1.0e-4), np.log(1.0)))),
    }


def _draw_triple(rng):
    return {
        "s": float(np.exp(rng.uniform(np.log(0.6), np.log(1.6)))),
        "q": float(np.exp(rng.uniform(np.log(1.0e-4), np.log(0.3)))),
        "q2": float(np.exp(rng.uniform(np.log(1.0e-5), np.log(0.1)))),
        "sep2": float(rng.uniform(0.3, 1.4)),
        "ang": float(rng.uniform(0.0, 2.0 * np.pi)),
    }


def test_binary_smooth_routes_stay_clear_of_the_caustic():
    lcbinint = pytest.importorskip("lcbinint")
    rng = np.random.default_rng(20260802)
    smooth, violations = _sweep(lcbinint, rng, 80, 16, _draw_binary)
    assert smooth > 500, f"sample degenerated: only {smooth} smooth routes"
    assert not violations, "\n".join(violations[:10])


def test_triple_smooth_routes_stay_clear_of_the_caustic():
    lcbinint = pytest.importorskip("lcbinint")
    rng = np.random.default_rng(20260803)
    smooth, violations = _sweep(
        lcbinint, rng, 24, 10, _draw_triple, lens="triple")
    assert smooth > 60, f"sample degenerated: only {smooth} smooth routes"
    assert not violations, "\n".join(violations[:10])
