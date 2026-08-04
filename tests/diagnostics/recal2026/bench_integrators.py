#!/usr/bin/env python3
"""Integration speed, and nothing else.

Every engine here is pinned to its integrator.  That is the whole design.  A
library's shipping entry point answers an easy position with a hexadecapole or
a point source and never builds a grid or a contour, so timing it and calling
the result "inverse-ray speed" times the absence of an inverse ray -- and the
shortcuts do not fire at the same places for different libraries, so a mixture
is not even a consistent mixture.  What is compared here is quadrature against
quadrature:

* ``VBMicrolensing.BinaryMag`` and ``BinaryMagDark`` -- the contour, without the
  quadrupole test that ``BinaryMag2`` applies first.
* ``microlux.contour_integral`` -- the adaptive contour.  ``binary_mag`` is not
  used: its own docstring says it "will dynamically choose full contour
  integration or point source approximation based on the quadrupole test".
* lcbinint's inverse ray, Cartesian and polar, with the point-source,
  hexadecapole and adaptive-hexadecapole exits all zeroed.
* the same inverse ray through the JAX backend, called as the integration
  primitive rather than through the light-curve API, which routes.

Positions are placed on the caustic's own normal at controlled multiples of
``rho``, because that is where an integrator is actually exercised and because
it makes the distance to the caustic an axis rather than an accident.

Three conventions are established by measurement rather than by reading, in
``--verify``, because each of them fails silently:

* VBMicrolensing's limb darkening.  ``BinaryMagDark`` takes ``a1`` as a
  positional argument; setting ``vbm.a1`` and calling ``BinaryMag`` instead
  returns a uniform-source answer with no error and no warning.
* microlux's quadrupole test.  ``contour_integral`` does not apply it, but the
  limb-darkened path is only reachable through ``extended_light_curve``, which
  does.  The test does not depend on the brightness profile, so checking that
  the uniform contour and the uniform light-curve entry agree at every sampled
  position establishes that it never fires for either profile.
* the JAX Cartesian backend's ``cell_size``, which is a length and not a bin
  count, so it has to be tied to ``nbin`` by matching an answer.

Bin counts come from the campaign's own ``nbin_rule.json``: the 99%-coverage
constant for each grid and tolerance.  Picking them here would be inventing a
second rule alongside the one this campaign just established.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

from .engines import Evaluation, LcbinintEngine

LENS_S = 1.1
LENS_Q = 1.0e-3

PROFILES = (("uniform", 0.0), ("linear", 0.5))

SOURCE_RADII = (1.0e-1, 3.0e-2, 1.0e-2, 3.0e-3, 1.0e-3)

# Distance from the caustic, in source radii.  Zero would be a tangency and is
# excluded deliberately: it is a measure-zero configuration whose cost is set
# by how close the sampler happened to land, not by the geometry.
DISTANCE_FACTORS = (0.1, 0.3, 1.0, 3.0)

TOLERANCES = (1.0e-2, 1.0e-3, 1.0e-4)

# The 99%-coverage constants from nbin_rule.json, keyed (grid, tolerance).
# Loaded rather than hard-coded when the file is present; these are the values
# it held when this was written, kept as the fallback so the benchmark still
# runs on a checkout without the results directory.
NBIN_FALLBACK = {
    ("cartesian", 1.0e-2): 16, ("cartesian", 1.0e-3): 50,
    ("cartesian", 1.0e-4): 128,
    ("polar", 1.0e-2): 24, ("polar", 1.0e-3): 100, ("polar", 1.0e-4): 320,
}

NBIN_RULE_PATH = "tests/diagnostics/results/recal2026/nbin_rule.json"

# microlux's sampling budget, paired with the tolerance so the knob measures
# accuracy rather than how many points it was allowed.  Same reasoning, and the
# same ladder, as engines_ext.MICROLUX_SETTINGS.
MICROLUX_STRATEGY = {
    1.0e-2: (30, 30, 60, 120, 240),
    1.0e-3: (30, 30, 60, 120, 240),
    1.0e-4: (60, 60, 120, 240, 480),
}

# VBM and microlux both take the source position mirrored in x relative to the
# frame lcbinint reports.  frames.verify() and engines_ext establish this for
# the light-curve entry points; --verify re-establishes it for these.
X_SIGN = -1.0

# Absolute accuracy for VBM's contour, set below anything reachable so the
# relative criterion is what stops it -- the same convention the speed sweep
# uses, and the one that does not penalise VBM at high magnification.
VBM_ABSOLUTE_FLOOR = 1.0e-12


def nbin_rule(path=NBIN_RULE_PATH):
    """Bins per grid and tolerance, from the campaign's own fitted rule."""
    try:
        with open(path) as stream:
            payload = json.load(stream)
    except OSError:
        return dict(NBIN_FALLBACK)
    out = {}
    for grid, entry in payload["grids"].items():
        for tolerance, values in entry["tolerances"].items():
            out[(grid, float(tolerance))] = int(values["constant_bins"])
    return out or dict(NBIN_FALLBACK)


# --------------------------------------------------------------------------
# positions
# --------------------------------------------------------------------------

def caustic_positions(rho, factors=DISTANCE_FACTORS, per_factor=12, seed=20260804):
    """Points at ``factor * rho`` along the caustic's outward normal.

    Sampled around the whole caustic rather than at a chosen fold or cusp: the
    cost of an integration varies along the curve -- a cusp has more images
    arriving and leaving than a fold does -- and picking one feature by hand
    would make that variation invisible while looking like precision.
    """
    from .geometry import caustic_branches

    branches = caustic_branches(LENS_S, LENS_Q)
    rng = np.random.default_rng(seed)
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
            # Both signs of the normal are offered because which one leaves the
            # caustic depends on where on the branch the sample landed, and
            # guessing would bias the set toward disk interiors.
            sign = 1.0 if rng.random() < 0.5 else -1.0
            position = point + sign * factor * rho * normal
            out.append({"x": float(position[0]), "y": float(position[1]),
                        "rho": rho, "distance_factor": factor})
    return out


# --------------------------------------------------------------------------
# engines, each pinned to its integrator
# --------------------------------------------------------------------------

def _timed(call, repeat):
    """Median of ``repeat`` warm calls, and the value.

    One untimed call first: every engine here either builds a caustic cache, or
    compiles, or both, and the first call would otherwise report that instead
    of the integration.
    """
    value = call()
    samples = []
    for _ in range(max(repeat, 1)):
        started = time.perf_counter()
        value = call()
        samples.append(time.perf_counter() - started)
    return float(np.median(samples)), value


class LcbinintIntegrator:
    """Native inverse ray on one grid, with every cheap exit disabled."""

    def __init__(self, grid, nbin, profile_c, tolerance):
        self.grid = grid
        self.nbin = nbin
        self._engine = LcbinintEngine(
            grid=grid, nbin=nbin, profile_c=profile_c, reltol=tolerance,
            max_source_bins=nbin, force_grid=True)

    def __call__(self, rho, x, y, repeat):
        seconds, evaluation = _timed(
            lambda: self._engine(LENS_S, LENS_Q, rho, x, y, time_it=False),
            repeat)
        return {
            "seconds": seconds,
            "value": evaluation.magnification,
            "method": evaluation.method,
            "converged": evaluation.converged,
            "self_reported_error": evaluation.error_estimate,
            "point_magnification": evaluation.extra["point_magnification"],
            "caustic_distance": evaluation.extra["caustic_distance"],
        }


class VbmContour:
    """VBMicrolensing's contour, without the quadrupole test.

    ``BinaryMag`` and ``BinaryMagDark`` are the raw contour entry points;
    ``BinaryMag2`` is the one that tries a multipole first.

    The two do not take the same arguments, and the difference is a trap.
    ``BinaryMag(s, q, y1, y2, rho, accuracy)`` ends in an absolute accuracy;
    ``BinaryMagDark(s, q, y1, y2, rho, a1)`` ends in the limb-darkening
    coefficient and reads its accuracy from ``Tol``/``RelTol`` instead.  Six
    arguments either way, so passing an accuracy in the last slot out of habit
    sets ``a1`` to it -- an accuracy of 1e-12 becomes a limb-darkening
    coefficient of 1e-12, which is a uniform source, and nothing complains.
    The installed docstring documents a seventh ``accuracy`` argument that the
    binding does not accept, and describes ``a1`` as "source angular radius",
    so it cannot be used to settle this; ``--verify`` settles it by measurement.
    """

    def __init__(self, profile_c, tolerance):
        import VBMicrolensing

        self.profile_c = profile_c
        self._vbm = VBMicrolensing.VBMicrolensing()
        self._vbm.Tol = VBM_ABSOLUTE_FLOOR
        self._vbm.RelTol = tolerance

    def __call__(self, rho, x, y, repeat):
        y1 = X_SIGN * x
        if self.profile_c:
            call = lambda: self._vbm.BinaryMagDark(
                LENS_S, LENS_Q, y1, y, rho, self.profile_c)
        else:
            call = lambda: self._vbm.BinaryMag(
                LENS_S, LENS_Q, y1, y, rho, VBM_ABSOLUTE_FLOOR)
        seconds, value = _timed(call, repeat)
        return {"seconds": seconds, "value": float(value)}


class MicroluxContour:
    """microlux's adaptive contour, reached without the quadrupole test.

    Uniform sources go through ``contour_integral``, which does not apply the
    test at all.  Limb darkening is only reachable through
    ``extended_light_curve``, which does apply it before choosing a path -- so
    ``--verify`` establishes that it never fires at these positions, and since
    the test reads only ``rho, s, q`` and the position, that settles it for
    both profiles.
    """

    def __init__(self, profile_c, tolerance):
        import microlux
        from microlux.model import to_lowmass

        self._microlux = microlux
        self._to_lowmass = to_lowmass
        self.profile_c = profile_c
        self.tolerance = tolerance
        self.strategy = MICROLUX_STRATEGY[tolerance]

    def _trajectory(self, x, y):
        # microlux works in the MulensModel centre-of-mass frame and then
        # converts to its own low-mass frame; the x mirror is the same one VBM
        # needs.  A single complex position, shaped as the length-one
        # trajectory the entry points expect.
        import jax.numpy as jnp

        centre = jnp.asarray([complex(X_SIGN * x, y)])
        return self._to_lowmass(LENS_S, LENS_Q, centre)

    def __call__(self, rho, x, y, repeat):
        import jax

        trajectory = self._trajectory(x, y)
        if self.profile_c:
            from microlux.limb_darkening import LinearLimbDarkening

            profile = LinearLimbDarkening(self.profile_c)
            call = lambda: np.asarray(jax.block_until_ready(
                self._microlux.extended_light_curve(
                    trajectory, LENS_S, LENS_Q, rho, self.tolerance,
                    self.tolerance, default_strategy=self.strategy,
                    limb_darkening=profile))).ravel()[0]
        else:
            call = lambda: float(np.asarray(jax.block_until_ready(
                self._microlux.contour_integral(
                    trajectory, self.tolerance, self.tolerance, rho,
                    LENS_S, LENS_Q, self.strategy, True)[0])).ravel()[0])
        seconds, value = _timed(call, repeat)
        return {"seconds": seconds, "value": float(value)}


class JaxIntegrator:
    """lcbinint's inverse ray through JAX, called as the integration primitive.

    Not ``binary_magnification_trajectory``: that is the routed path, and it
    would put the hexadecapole and point-source exits back into a column headed
    "inverse ray".  These are the grid kernels themselves.
    """

    def __init__(self, grid, nbin, profile_c, cell_size_factor):
        import jax

        self._jax = jax
        self.grid = grid
        self.nbin = nbin
        self.profile_c = profile_c
        self.cell_size_factor = cell_size_factor
        self._compiled = None

    def __call__(self, rho, x, y, repeat):
        import lcbinint_jax as lj

        if self.grid == "cartesian":
            call = lambda: lj.binary_inverse_ray_cartesian_ffi(
                np.float64(X_SIGN * x), np.float64(y), LENS_S, LENS_Q, rho,
                self.profile_c, 0.0,
                cell_size=self.cell_size_factor * rho / self.nbin)
        else:
            call = lambda: lj.binary_inverse_ray_polar_ffi(
                np.float64(X_SIGN * x), np.float64(y), LENS_S, LENS_Q, rho,
                self.profile_c, 0.0, resolution=self.nbin)

        def run():
            result = call()
            return float(np.asarray(
                self._jax.block_until_ready(result.magnification)).ravel()[0])

        seconds, value = _timed(run, repeat)
        return {"seconds": seconds, "value": value}


# --------------------------------------------------------------------------
# verification of the three silent conventions
# --------------------------------------------------------------------------

def verify(tolerance=1.0e-4):
    """Pin the conventions that fail silently.  Returns a report and a verdict."""
    report = {}
    probe_rho = 1.0e-2
    probes = caustic_positions(probe_rho, factors=(0.3, 1.0), per_factor=3)

    # -- a converged answer to check the others against ---------------------
    truth = {}
    for profile, profile_c in PROFILES:
        engine = LcbinintEngine(grid="cartesian", nbin=400, profile_c=profile_c,
                                reltol=1.0e-7, max_source_bins=400,
                                force_grid=True)
        truth[profile] = [engine(LENS_S, LENS_Q, probe_rho, p["x"], p["y"],
                                 time_it=False).magnification for p in probes]

    # -- VBM: is a1 actually reaching the integrator? -----------------------
    # The failure this catches is the quiet one: BinaryMagDark called with the
    # wrong argument order, or a1 set on the object and never read, both give a
    # uniform-source answer that looks perfectly reasonable on its own.
    vbm_dark = VbmContour(0.5, tolerance)
    vbm_uniform = VbmContour(0.0, tolerance)
    dark_gap = max(
        abs(vbm_dark(probe_rho, p["x"], p["y"], 1)["value"] - expected)
        / max(expected, 1.0)
        for p, expected in zip(probes, truth["linear"]))
    uniform_gap = max(
        abs(vbm_uniform(probe_rho, p["x"], p["y"], 1)["value"] - expected)
        / max(expected, 1.0)
        for p, expected in zip(probes, truth["uniform"]))
    # If limb darkening were being dropped, the darkened call would land on the
    # uniform answer instead; that separation is what makes the first number
    # mean something.
    confusion = min(
        abs(vbm_dark(probe_rho, p["x"], p["y"], 1)["value"] - expected)
        / max(expected, 1.0)
        for p, expected in zip(probes, truth["uniform"]))
    report["vbm"] = {
        "linear_gap": dark_gap, "uniform_gap": uniform_gap,
        "gap_if_darkening_were_dropped": confusion,
        "ok": dark_gap < 1.0e-3 and uniform_gap < 1.0e-3
              and confusion > 100.0 * dark_gap,
    }

    # -- microlux: does the quadrupole test ever fire at these positions? ---
    # contour_integral does not apply the test; extended_light_curve does.  If
    # the two agree at every probe then the test never fired, and because it
    # reads only rho, s, q and the position it cannot fire for the limb-darkened
    # profile either.
    micro = MicroluxContour(0.0, tolerance)
    import jax
    import microlux
    worst_route = 0.0
    worst_truth = 0.0
    for probe, expected in zip(probes, truth["uniform"]):
        trajectory = micro._trajectory(probe["x"], probe["y"])
        raw = float(np.asarray(microlux.contour_integral(
            trajectory, tolerance, tolerance, probe_rho, LENS_S, LENS_Q,
            micro.strategy, True)[0]).ravel()[0])
        routed = float(np.asarray(jax.block_until_ready(
            microlux.extended_light_curve(
                trajectory, LENS_S, LENS_Q, probe_rho, tolerance, tolerance,
                default_strategy=micro.strategy))).ravel()[0])
        worst_route = max(worst_route, abs(raw - routed) / max(raw, 1.0))
        worst_truth = max(worst_truth, abs(raw - expected) / max(expected, 1.0))
    report["microlux"] = {
        "contour_vs_routed": worst_route, "contour_vs_truth": worst_truth,
        "ok": worst_route < 1.0e-9 and worst_truth < 1.0e-3,
    }

    # -- JAX: what is cell_size in units of rho and nbin? -------------------
    # Tried as a factor on rho/nbin.  The right one reproduces the native grid's
    # answer; the wrong ones are a different grid, so they miss by far more than
    # the tolerance and the choice is not a close call.
    import lcbinint_jax as lj
    native = LcbinintEngine(grid="cartesian", nbin=64, profile_c=0.0,
                            reltol=tolerance, max_source_bins=64,
                            force_grid=True)
    candidates = {}
    for factor in (0.5, 1.0, 2.0):
        gaps = []
        for probe in probes:
            expected = native(LENS_S, LENS_Q, probe_rho, probe["x"],
                              probe["y"], time_it=False).magnification
            got = float(np.asarray(lj.binary_inverse_ray_cartesian_ffi(
                np.float64(X_SIGN * probe["x"]), np.float64(probe["y"]),
                LENS_S, LENS_Q, probe_rho, 0.0, 0.0,
                cell_size=factor * probe_rho / 64).magnification).ravel()[0])
            gaps.append(abs(got - expected) / max(expected, 1.0))
        candidates[factor] = float(np.median(gaps))
    best = min(candidates, key=candidates.get)
    others = [value for factor, value in candidates.items() if factor != best]
    report["jax_cell_size"] = {
        "candidates": candidates, "best_factor": best,
        "ok": candidates[best] < 1.0e-2 and min(others) > 10.0 * candidates[best],
    }

    report["ok"] = all(section["ok"] for section in report.values()
                       if isinstance(section, dict))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--cores", default="40")
    parser.add_argument("--output")
    arguments = parser.parse_args()

    os.sched_setaffinity(0, {int(item) for item in arguments.cores.split(",")})
    from .engines_ext import configure_jax
    configure_jax(os.environ.get("RECAL_JAX_CACHE"))

    if arguments.verify:
        report = verify()
        print(json.dumps(report, indent=1, default=float))
        raise SystemExit(0 if report["ok"] else 1)

    raise SystemExit("sweep not implemented yet; run --verify first")


if __name__ == "__main__":
    main()
