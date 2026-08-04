"""Uniform evaluation interface over the engines this campaign compares.

Every engine here answers the same question -- the finite-source magnification
of a binary lens at one source position -- and every one of them exposes a
different accuracy knob.  Comparing them at matched *nominal* tolerance would
compare their labelling conventions rather than their numerics, so the sweeps
drive each knob over a range and compare achieved error against measured time.
That is what :func:`accuracy_knobs` is for.

Positions are always given in lcbinint's ``coordinates='vbm'`` frame.  Each
wrapper is responsible for translating into whatever its engine wants; see
``frames.py`` for why that translation is measured rather than remembered.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

import lcbinint

from .frames import position_from_trajectory

# The resolution buckets the ladder measures.  From 16 upwards these are the
# native selector's own; below 16 they are an extension this campaign adds.
#
# The extension is not cosmetic.  The shipping selector cannot return fewer than
# 16 bins -- the floor is hard-coded in three places -- and the first pass of
# this campaign found that 99.4% of sampled geometries already meet a 1e-2
# target at 16.  A ladder whose coarsest rung is 16 can only report "16 was
# enough", which is censoring, not a measurement: it cannot distinguish a row
# that needed 16 from one that would have been fine with 4.  Since the component
# certificate made correctness independent of grid density, that censored region
# is precisely where the remaining cost is, so the ladder is extended into it.
SHIPPING_BUCKET_FLOOR = 16
BUCKETS = (4, 6, 8, 10, 12,
           16, 24, 32, 40, 50, 64, 80, 100, 128, 160, 200, 256, 320, 400)

# Source profiles.  ``c`` is the linear coefficient in
# ``I(mu)/I0 = 1 - c (1 - mu)``, normalised by ``1 - c/3``.  lcbinint, VBM's
# ``a1`` and microlux's ``LinearLimbDarkening.a`` all use this same convention;
# ``pilot.py`` checks that empirically rather than trusting the reading.
PROFILES = {
    "uniform": 0.0,
    "linear": 0.5,
}


@dataclass
class Evaluation:
    """One engine's answer at one position."""

    magnification: float = float("nan")
    error_estimate: float = float("inf")
    converged: bool = False
    method: str = ""
    seconds: float = float("nan")
    # Support was proven for image-plane routes iff the fail-closed error is
    # finite; the certificate reports an unproven disk as an infinite error.
    support_proven: bool = False
    extra: dict = field(default_factory=dict)


def _scalar(value):
    return float(np.asarray(value).ravel()[0])


# --------------------------------------------------------------------------
# lcbinint (native)
# --------------------------------------------------------------------------

class LcbinintEngine:
    """Native lcbinint through the public light-curve API.

    A ``LightCurve`` carries the per-lens caustic cache, so one instance is
    reused across the positions of a single lens.  That is what a light curve
    does in production, and it is the timing that matters; the cold cost of
    building the cache is measured separately by ``cold_seconds``.
    """

    def __init__(self, *, grid, nbin, profile_c, reltol=0.0, tol=0.0,
                 max_source_bins=None, polar_grid_ratio=0.0, caustic_bins=1400,
                 force_grid=False):
        self.grid = grid
        self.nbin = nbin
        self.profile_c = profile_c
        self.reltol = reltol
        self.tol = tol
        self.force_grid = force_grid
        options = {
            "coordinates": "vbm",
            "nbin": nbin,
            "caustic_bins": caustic_bins,
        }
        if grid is not None:
            options["inverse_ray_grid"] = grid
        if force_grid:
            # Naming a grid is not enough to get one.  The point-source and
            # hexadecapole exits are taken before the grid is ever chosen, so
            # ``inverse_ray_grid='cartesian'`` on a source three radii from the
            # caustic silently returns a hexadecapole value at every bucket --
            # in the first trial that was 74% of all ladder evaluations, and it
            # makes a resolution ladder report that 16 bins are always enough.
            # Zeroing all three exits is what actually forces the integrator.
            options["point_source_threshold"] = 0.0
            options["hexadecapole_threshold"] = 0.0
            options["adaptive_hex_threshold"] = 0.0
        if max_source_bins is not None:
            options["max_source_bins"] = max_source_bins
        if polar_grid_ratio:
            options["polar_grid_ratio"] = polar_grid_ratio
        if reltol:
            options["reltol"] = reltol
        if tol:
            options["tol"] = tol
        self._options = options
        self._curve = None

    @property
    def name(self):
        grid = self.grid or "auto"
        return f"lcbinint:{grid}:{self.nbin}"

    def _ensure(self):
        if self._curve is None:
            self._curve = lcbinint.LightCurve(
                lens="binary", options=lcbinint.Options(**self._options))
        return self._curve

    def reset(self):
        """Drop all caches, so the next call pays the cold cost."""
        self._curve = None

    def __call__(self, s, q, rho, x, y, *, time_it=True):
        curve = self._ensure()
        # alpha=0 makes the time argument the abscissa and u0 the ordinate;
        # frames.verify() checks that this lands exactly where it claims.
        started = time.perf_counter()
        info = curve.info(
            x, t0=0.0, tE=1.0, u0=y, alpha=0.0,
            s=s, q=q, rho=rho,
            limb_darkening_c=self.profile_c,
        )
        elapsed = time.perf_counter() - started
        error = _scalar(info.finite_source_error_estimates)
        return Evaluation(
            magnification=_scalar(info.finite_source_magnifications),
            error_estimate=error,
            converged=bool(_scalar(info.finite_source_converged)),
            method=str(np.asarray(info.finite_source_method_names).ravel()[0]),
            seconds=elapsed if time_it else float("nan"),
            support_proven=math.isfinite(error),
            extra={
                "caustic_distance": _scalar(info.caustic_distances),
                "point_magnification": _scalar(info.point_source_magnifications),
                "refinement_level": int(_scalar(info.finite_source_refinement_levels)),
                "image_count": int(_scalar(info.image_counts)),
                "source_x": _scalar(info.source_x),
                "source_y": _scalar(info.source_y),
            },
        )


def lcbinint_fixed(grid, nbin, profile_c, *, force_grid=True, **kwargs):
    """A single fixed resolution on one grid, with no tolerance feedback.

    ``max_source_bins`` is pinned to ``nbin`` so the runtime cannot quietly
    escalate, and the cheaper routes are disabled so the bucket is what is
    actually measured: the ladder needs to know what each grid resolution alone
    can do, which is a different question from what the library would choose.
    """
    return LcbinintEngine(
        grid=grid, nbin=nbin, profile_c=profile_c,
        max_source_bins=nbin, force_grid=force_grid, **kwargs)


def lcbinint_auto(reltol, profile_c, max_source_bins=400, **kwargs):
    """The shipping automatic path: its own resolution and method choice."""
    return LcbinintEngine(
        grid=None, nbin="auto", profile_c=profile_c, reltol=reltol,
        max_source_bins=max_source_bins, **kwargs)


# --------------------------------------------------------------------------
# VBMicrolensing
# --------------------------------------------------------------------------

class VbmEngine:
    """VBMicrolensing contour integration.

    ``BinaryMag2`` takes the source position mirrored in x relative to the
    frame lcbinint reports; ``frames.verify()`` measures that and refuses to
    run if it ever stops being true.
    """

    def __init__(self, *, tol, profile_c, reltol=0.0):
        import VBMicrolensing

        self.tol = tol
        self.reltol = reltol
        self.profile_c = profile_c
        self._vbm = VBMicrolensing.VBMicrolensing()
        self._vbm.Tol = tol
        self._vbm.RelTol = reltol
        if profile_c:
            self._vbm.a1 = profile_c
            self._vbm.SetLDprofile(self._vbm.LDlinear)

    @property
    def name(self):
        return f"vbm:{self.tol:g}"

    def reset(self):
        pass

    def __call__(self, s, q, rho, x, y, *, time_it=True):
        started = time.perf_counter()
        if self.profile_c:
            # In VBMicrolensing 5.x the sixth argument of BinaryMagDark is the
            # accuracy, not the limb-darkening coefficient; the coefficient is
            # taken from the object.  Passing the coefficient positionally is
            # accepted silently and reads as a tolerance, which for c=0.8 means
            # asking for 80% accuracy -- it produced a 2.6% disagreement that
            # looked exactly like a convention mismatch and was not one.
            value = self._vbm.BinaryMagDark(s, q, -x, y, rho, self.tol)
        else:
            value = self._vbm.BinaryMag2(s, q, -x, y, rho)
        elapsed = time.perf_counter() - started
        value = float(value)
        return Evaluation(
            magnification=value,
            # VBM does not return a usable per-call error estimate through this
            # entry point; the sweeps measure its error against the reference
            # instead of asking it.
            error_estimate=float("nan"),
            converged=math.isfinite(value) and value > 0.0,
            method="vbm_contour",
            seconds=elapsed if time_it else float("nan"),
            support_proven=math.isfinite(value),
        )


# --------------------------------------------------------------------------
# Accuracy knobs, for the achieved-error-versus-time comparison
# --------------------------------------------------------------------------

def accuracy_knobs(engine_kind):
    """The knob values each engine is swept over on the Pareto plots.

    Matched nominal tolerance is not a fair comparison because the engines do
    not mean the same thing by it, so each one is swept across its own range
    and judged on where it lands in (achieved error, time).
    """
    if engine_kind == "vbm":
        return (1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7)
    if engine_kind == "lcbinint_auto":
        return (1.0e-2, 1.0e-3, 1.0e-4)
    if engine_kind in ("lcbinint_cartesian", "lcbinint_polar"):
        return BUCKETS
    if engine_kind == "microlux":
        return (1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5)
    raise ValueError(f"unknown engine kind {engine_kind!r}")


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------
#
# Timing one source position at a time does not measure what a light curve
# costs.  Each entry into lcbinint carries a fixed ~1.4 ms of per-call setup,
# dominated by rebuilding the caustic polyline for the lens; measured across
# resolutions it is flat, so it is 7% of a 200-bin evaluation and 180% of a
# 16-bin one.  Timing per call would therefore report that coarse grids are
# barely cheaper than fine ones, which is an artefact of the harness rather
# than of the integrator.
#
# Every engine compared here -- lcbinint, VBM, microLUX, the JAX path -- is
# really a light-curve API, so the measurement unit is a contiguous block of
# epochs.  Blocks are kept short enough that the magnification varies little
# across one, which is what lets the results be plotted against magnification,
# and long enough that the fixed cost is amortised to insignificance.

DEFAULT_BLOCK = 24


def time_light_curve_block(curve_factory, s, q, rho, times, u0, alpha,
                           profile_c, *, repeat=3):
    """Per-epoch wall time of one contiguous block of epochs.

    The engine is rebuilt for the warm-up pass and then reused, so the reported
    time is the marginal cost of an epoch inside a curve -- what a light-curve
    fit actually pays -- rather than the cost of a cold start.
    """
    curve = curve_factory()
    values = curve.info(
        times, t0=0.0, tE=1.0, u0=u0, alpha=alpha, s=s, q=q, rho=rho,
        limb_darkening_c=profile_c)
    samples = []
    for _ in range(max(repeat, 1)):
        started = time.perf_counter()
        values = curve.info(
            times, t0=0.0, tE=1.0, u0=u0, alpha=alpha, s=s, q=q, rho=rho,
            limb_darkening_c=profile_c)
        samples.append((time.perf_counter() - started) / len(times))
    return float(np.median(samples)), values


def time_vbm_block(tol, reltol, s, q, rho, times, u0, alpha, profile_c,
                   *, repeat=3):
    """The same block through VBMicrolensing's own light-curve entry point.

    ``BinaryLightCurve`` takes trajectory parameters rather than positions, so
    it is immune to the frame question and is the like-for-like counterpart of
    ``LightCurve.info``.
    """
    import VBMicrolensing

    vbm = VBMicrolensing.VBMicrolensing()
    vbm.Tol = tol
    vbm.RelTol = reltol
    if profile_c:
        vbm.a1 = profile_c
        vbm.SetLDprofile(vbm.LDlinear)
    parameters = [
        math.log(s), math.log(q), u0, alpha, math.log(rho), 0.0, 0.0]
    times = list(times)
    values = vbm.BinaryLightCurve(parameters, times)
    samples = []
    for _ in range(max(repeat, 1)):
        started = time.perf_counter()
        values = vbm.BinaryLightCurve(parameters, times)
        samples.append((time.perf_counter() - started) / len(times))
    return float(np.median(samples)), np.asarray(values[0], dtype=float)
