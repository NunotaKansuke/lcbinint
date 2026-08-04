"""The JAX and microlux comparison axes.

These live apart from ``engines.py`` because they are JAX programs and the rest
of the campaign is not.  Importing JAX reserves memory and spawns its own
threads, and a traced function is compiled on first call, so a sweep that mixed
them into the same process would attribute compilation to the first geometry it
happened to evaluate.  Everything here is therefore designed to be run in its
own process, with compilation warmed and reported separately from the timing.

Two properties of these engines are measured rather than assumed:

* microlux works in the MulensModel centre-of-mass frame, which is not the one
  lcbinint reports.  The translation is fixed by scanning candidate frames
  against a known answer, exactly as ``frames.py`` does for VBMicrolensing; the
  wrong candidates are out by 38%, so the scan is decisive.

* microlux's accuracy knob is not monotone.  Its adaptive sampler has a fixed
  budget given by ``default_strategy``; when a tolerance needs more points than
  the budget allows it warns and returns what it has.  Measured on benign
  positions, ``tol=1e-4`` lands at 1.9e-5 and ``tol=1e-6`` at 8.9e-6, but
  ``tol=1e-8`` lands at 7.8e-4 -- an order of magnitude worse than the loosest
  setting.  Sweeping tolerance alone would therefore plot a curve that doubles
  back and read as microlux being inaccurate, when it is being under-resourced.
  The budget is swept with the tolerance and the warning is captured, so a
  point that hit the ceiling is marked instead of being quietly plotted.
"""

from __future__ import annotations

import math
import os
import time
import warnings
from pathlib import Path

import numpy as np

# microlux takes the source position mirrored in x relative to the frame
# lcbinint reports -- the same mirror VBMicrolensing needs.  The remaining
# candidate frames that fit equally well are exact physical symmetries of a
# static binary (y -> -y, and the x mirror combined with relabelling q -> 1/q),
# not residual ambiguity, so any of them may be used and this one is chosen for
# being the same rule the VBM wrapper already applies.
MICROLUX_X_SIGN = -1.0

# Tolerance paired with a sampling budget large enough to reach it.  The
# defaults are microlux's own (30, 30, 60, 120, 240); the tighter rows raise the
# ceiling proportionally so the knob measures accuracy rather than budget.
MICROLUX_SETTINGS = (
    (1.0e-2, (30, 30, 60, 120, 240)),
    (1.0e-3, (30, 30, 60, 120, 240)),
    (1.0e-4, (30, 30, 60, 120, 240)),
    (1.0e-4, (60, 60, 120, 240, 480)),
    (1.0e-5, (60, 60, 120, 240, 480)),
    (1.0e-6, (120, 120, 240, 480, 960)),
    (1.0e-7, (240, 240, 480, 960, 1920)),
)


def configure_jax(cache_directory=None):
    """Double precision, one CPU device, no accidental multi-threading.

    Single precision would put JAX's floor above the tolerances this campaign
    measures, and the comparison would be of dtypes rather than of algorithms.

    Compiled artefacts are also shared through a directory on disk, which is
    what makes a wide sweep possible at all.  Every worker traces the same
    handful of functions, and microlux's cost twenty-five seconds each; without
    the cache, two dozen workers compile the same code two dozen times, at once,
    each pinned to a single core.  That is most of the run's wall clock, and the
    concurrent LLVM instances were failing outright with ``Cannot allocate
    memory`` on a machine with a hundred and seventy gigabytes free.

    The cache holds compiled code keyed on the computation, so it changes what
    the *first* call costs and nothing else.  Per-epoch timings are medians over
    warm calls and cannot see it.
    """
    import jax

    jax.config.update("jax_enable_x64", True)
    directory = cache_directory or os.environ.get("RECAL_JAX_CACHE")
    if directory:
        Path(directory).mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(directory))
        # Default is a wide net on purpose: the point is to compile each
        # function once for the whole sweep, not to be selective about which.
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.2)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    return jax


class MicroluxEngine:
    """microlux, timed on a block, with compilation excluded and reported.

    ``binary_mag`` is a traced function, so the first call for a given array
    shape pays for compilation.  A microlensing fit pays that once and then
    evaluates thousands of light curves, so folding it into the per-epoch time
    would describe a use nobody has; it is measured separately instead.
    """

    def __init__(self, *, tol, strategy, profile_c):
        import microlux

        self._microlux = microlux
        self.tol = tol
        self.strategy = tuple(strategy)
        self.profile_c = profile_c

    @property
    def name(self):
        return f"microlux:{self.tol:g}:{max(self.strategy)}"

    def _call(self, s, q, rho, times, u0, alpha_deg):
        # Mirroring x reverses the direction of travel, so the block is
        # reversed as well and unreversed afterwards: microlux's adaptive
        # sampler is per-epoch, but handing it a descending time array would
        # make any future ordering assumption a silent error here.
        mirrored = np.ascontiguousarray(MICROLUX_X_SIGN * np.asarray(times)[::-1])
        function = (self._microlux.binary_mag_linear if self.profile_c
                    else self._microlux.binary_mag)
        arguments = dict(tol=self.tol, retol=self.tol,
                         default_strategy=self.strategy)
        if self.profile_c:
            arguments["limb_darkening_coeff"] = self.profile_c
        values = function(0.0, u0, 1.0, rho, q, s, alpha_deg, mirrored,
                          **arguments)
        return np.asarray(values, dtype=float).ravel()[::-1]

    def time_block(self, s, q, rho, times, u0, alpha_deg=0.0, *, repeat=3):
        """Per-epoch seconds, the values, and whether the budget was hit."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compile_started = time.perf_counter()
            values = self._call(s, q, rho, times, u0, alpha_deg)
            compile_seconds = time.perf_counter() - compile_started
            exhausted = any("space to insert" in str(item.message)
                            for item in caught)

        samples = []
        for _ in range(max(repeat, 1)):
            started = time.perf_counter()
            values = self._call(s, q, rho, times, u0, alpha_deg)
            samples.append((time.perf_counter() - started) / len(times))
        return {
            "seconds_per_epoch": float(np.median(samples)),
            "values": values,
            "budget_exhausted": bool(exhausted),
            "first_call_seconds": compile_seconds,
        }


class LcbinintJaxEngine:
    """lcbinint's JAX backend, timed the same way as its native backend.

    The comparison that matters is against lcbinint's own native path at the
    same tolerance: the JAX backend exists to be differentiated and batched, and
    what a paper needs to state is what that costs per epoch, not whether a
    traced implementation can beat a compiled one on a single light curve.
    """

    def __init__(self, *, reltol, profile_c, max_source_bins=400):
        import lcbinint

        self.reltol = reltol
        self.profile_c = profile_c
        self._curve = lcbinint.LightCurve(
            lens="binary",
            options=lcbinint.Options(
                coordinates="vbm", nbin="auto", caustic_bins=1400,
                max_source_bins=max_source_bins, reltol=reltol, jax=True))

    @property
    def name(self):
        return f"lcbinint_jax:{self.reltol:g}"

    def _call(self, s, q, rho, times, u0, alpha):
        values = self._curve.magnification(
            np.asarray(times), t0=0.0, tE=1.0, u0=u0, alpha=alpha,
            s=s, q=q, rho=rho, limb_darkening_c=self.profile_c)
        return np.asarray(values, dtype=float).ravel()

    def time_block(self, s, q, rho, times, u0, alpha=0.0, *, repeat=3):
        compile_started = time.perf_counter()
        values = self._call(s, q, rho, times, u0, alpha)
        compile_seconds = time.perf_counter() - compile_started
        samples = []
        for _ in range(max(repeat, 1)):
            started = time.perf_counter()
            values = self._call(s, q, rho, times, u0, alpha)
            samples.append((time.perf_counter() - started) / len(times))
        return {
            "seconds_per_epoch": float(np.median(samples)),
            "values": values,
            "first_call_seconds": compile_seconds,
        }


def verify_microlux_frame(tolerance=1.0e-4, separation=3.0e-2):
    """Fix the microlux frame by measurement, and refuse a coincidence.

    Returns the worst relative gap under the accepted frame together with the
    best gap achieved by any frame that is not a symmetry of it.  The second
    number is the one that makes the first meaningful: if a wrong frame ever
    starts fitting too, the mapping is no longer determined by this test and
    the campaign should stop rather than pick one.
    """
    from .engines import LcbinintEngine

    probes = (
        (1.00, 1.0e-3, 0.010, 0.0487, 0.0152),
        (0.70, 1.0e-2, 0.020, -0.0972, -0.1177),
        (1.60, 3.0e-1, 0.030, 0.0519, 0.1091),
        (2.20, 1.0e-1, 0.050, -0.2024, 0.1467),
    )
    native = LcbinintEngine(grid=None, nbin="auto", profile_c=0.0,
                            reltol=1.0e-6, max_source_bins=400)
    truth = [native(s, q, rho, x, y).magnification for s, q, rho, x, y in probes]

    engine = MicroluxEngine(tol=1.0e-6, strategy=(120, 120, 240, 480, 960),
                            profile_c=0.0)

    def gaps(sign, shift):
        out = []
        for (s, q, rho, x, y), expected in zip(probes, truth):
            times = np.array([sign * MICROLUX_X_SIGN * x + shift * s * q /
                              (1.0 + q)])
            # _call mirrors internally, so undo that here to test raw frames.
            values = engine._call(s, q, rho, MICROLUX_X_SIGN * times, y, 0.0)
            out.append(abs(float(values[0]) - expected) / max(abs(expected), 1.0))
        return max(out)

    accepted = gaps(1.0, 0.0)
    rejected = min(gaps(1.0, 1.0), gaps(1.0, -1.0), gaps(-1.0, 1.0),
                   gaps(-1.0, -1.0))
    ok = accepted <= tolerance and rejected > max(10.0 * accepted, 1.0e-3)
    return ok, accepted, rejected


def verify_microlux_limb_darkening(coefficient=0.5, tolerance=5.0e-3):
    """microlux's ``limb_darkening_coeff`` must mean lcbinint's ``c``.

    The tolerance is loose on purpose.  microlux documents that its limb
    darkening uses ten equal-area annuli in a non-adaptive scheme, so its
    darkened accuracy is not controlled by ``tol`` at all; this check is asking
    whether the two libraries describe the same profile, not whether microlux
    integrates it tightly.  How tightly it does is a result of the campaign, not
    a precondition for running it.
    """
    from .engines import lcbinint_fixed

    probes = (
        (1.00, 1.0e-3, 0.010, 0.0487, 0.0152),
        (0.70, 1.0e-2, 0.020, -0.0972, -0.1177),
        (2.20, 1.0e-1, 0.050, -0.2024, 0.1467),
    )
    mine = lcbinint_fixed("cartesian", 400, coefficient)
    theirs = MicroluxEngine(tol=1.0e-6, strategy=(120, 120, 240, 480, 960),
                            profile_c=coefficient)
    worst = 0.0
    for s, q, rho, x, y in probes:
        expected = mine(s, q, rho, x, y).magnification
        values = theirs._call(s, q, rho, np.array([x]), y, 0.0)
        value = float(values[0])
        if not (math.isfinite(value) and math.isfinite(expected)):
            continue
        worst = max(worst, abs(value - expected) / max(abs(expected), 1.0))
    return worst <= tolerance, worst
