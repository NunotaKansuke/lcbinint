"""Source-position frame conventions, discovered rather than assumed.

Every engine in this campaign takes a source position, and they do not all
take it in the same frame.  A sign error here does not crash: it silently
compares two different geometries, and the resulting table looks like an
accuracy result.  That already happened once during the resolvability-guard
investigation, where an unmirrored ``BinaryMag2`` reference made lcbinint look
1.8--12.4% wrong and made a useless guard look justified.

So the conventions are not written down from memory here.  They are measured
at import time against a reference that no frame question can affect --
``BinaryLightCurve``, which takes trajectory parameters rather than a position
-- and the campaign refuses to start if any of them has moved.
"""

from __future__ import annotations

import math

import numpy as np

import lcbinint

# Geometries used to pin the conventions.  They deliberately straddle the
# close/resonant/wide topologies and both mass-ratio extremes, so a convention
# that happens to be symmetric for one lens cannot pass.
_PROBES = (
    # (s, q, u0, alpha, rho, t)
    (1.00, 1.0e-3, -0.01, 0.50, 0.010, 0.05),
    (0.70, 1.0e-2, 0.13, 1.90, 0.020, -0.08),
    (1.60, 3.0e-1, 0.05, 0.70, 0.030, 0.11),
    (0.95, 1.0e-5, -0.03, 2.60, 0.005, 0.02),
    (2.20, 1.0e-1, 0.20, 0.30, 0.050, -0.15),
)


def source_position(s, q, u0, alpha, rho, t):
    """Where lcbinint says the source is, in its own ``coordinates='vbm'`` frame."""
    curve = lcbinint.LightCurve(
        lens="binary",
        options=lcbinint.Options(coordinates="vbm", nbin="auto"),
    )
    info = curve.info(
        t, t0=0.0, tE=1.0, u0=u0, alpha=alpha, s=s, q=q, rho=rho,
        limb_darkening_c=0.0,
    )
    return (
        float(np.asarray(info.source_x).ravel()[0]),
        float(np.asarray(info.source_y).ravel()[0]),
    )


def position_from_trajectory(u0, alpha, t):
    """The position an ``alpha=0`` trajectory places the source at.

    The sweeps drive lcbinint positionally by abusing the trajectory: with
    ``alpha=0``, ``t0=0`` and ``tE=1`` the time argument becomes the abscissa
    and ``u0`` the ordinate.  This helper exists so that assumption is stated
    in one place and checked in :func:`verify`, rather than being retyped into
    every sweep.
    """
    cos_a = math.cos(alpha)
    sin_a = math.sin(alpha)
    return (t * cos_a - u0 * sin_a, t * sin_a + u0 * cos_a)


def verify(tolerance=3.0e-13):
    """Measure the conventions and report them.

    Returns a dict describing what was found.  Raises if a convention that the
    sweeps depend on is not the one they were written against.
    """
    import VBMicrolensing

    vbm = VBMicrolensing.VBMicrolensing()
    vbm.Tol = 1.0e-10
    vbm.RelTol = 0.0

    findings = []
    for s, q, u0, alpha, rho, t in _PROBES:
        # BinaryLightCurve takes no position at all, so it cannot be affected
        # by any frame convention.  It is the anchor for everything else.
        anchor = float(vbm.BinaryLightCurve(
            [math.log(s), math.log(q), u0, alpha, math.log(rho), 0.0, 0.0],
            [t],
        )[0][0])

        x, y = source_position(s, q, u0, alpha, rho, t)
        direct = float(vbm.BinaryMag2(s, q, x, y, rho))
        mirrored = float(vbm.BinaryMag2(s, q, -x, y, rho))

        findings.append({
            "s": s, "q": q, "rho": rho,
            "source_x": x, "source_y": y,
            "anchor": anchor,
            "direct_rel": abs(direct - anchor) / anchor,
            "mirrored_rel": abs(mirrored - anchor) / anchor,
        })

    worst_mirrored = max(f["mirrored_rel"] for f in findings)
    best_direct = min(f["direct_rel"] for f in findings)
    if worst_mirrored > tolerance:
        raise RuntimeError(
            "VBM BinaryMag2 no longer reproduces BinaryLightCurve under the "
            f"x-mirrored convention (worst relative gap {worst_mirrored:.3e}). "
            "Re-derive the frame before trusting any reference in this campaign."
        )
    if best_direct <= tolerance:
        raise RuntimeError(
            "BinaryMag2 now agrees without mirroring as well; the convention "
            "is ambiguous and the mirror in vbm_uniform() must be re-derived."
        )

    # The positional driving convention used by the sweeps.
    position_errors = []
    for s, q, u0, alpha, rho, t in _PROBES:
        got = source_position(s, q, u0, alpha, rho, t)
        want = position_from_trajectory(u0, alpha, t)
        position_errors.append(math.hypot(got[0] - want[0], got[1] - want[1]))
    worst_position = max(position_errors)
    if worst_position > 1.0e-12:
        raise RuntimeError(
            "An alpha-rotated trajectory no longer lands where "
            f"position_from_trajectory() says (worst offset {worst_position:.3e}). "
            "The sweeps place sources by trajectory and would be sampling "
            "somewhere other than the caustic features they target."
        )

    return {
        "vbm_binary_mag2_mirrors_x": True,
        "worst_mirrored_relative_gap": worst_mirrored,
        "best_unmirrored_relative_gap": best_direct,
        "worst_trajectory_position_offset": worst_position,
        "probes": findings,
    }


if __name__ == "__main__":
    import json

    report = verify()
    print(json.dumps(
        {k: v for k, v in report.items() if k != "probes"}, indent=2))
    print()
    print(f"{'s':>6} {'q':>9} {'rho':>7} {'x':>12} {'y':>12} "
          f"{'mirrored':>10} {'direct':>10}")
    for f in report["probes"]:
        print(f"{f['s']:6.2f} {f['q']:9.1e} {f['rho']:7.3f} "
              f"{f['source_x']:12.6f} {f['source_y']:12.6f} "
              f"{f['mirrored_rel']:10.2e} {f['direct_rel']:10.2e}")
