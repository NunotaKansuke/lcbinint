"""The band the old resolvability guard vetoed is answerable.

``fixed_inverse_ray_binary`` used to carry a second veto beside the support
certificate: when a caustic touched the disk and the mass-ratio scale was
thinner than four cells (``q_small < 4 rho / source_bins``) it floored the error
at ``3e-3 * max(|A|, 1)`` and refused convergence.  The floor is three times any
tolerance a caller is likely to ask for, so inside that band every answer was a
NaN -- including answers that were already correct to 1e-5.

The guard predates the support certificate and the per-component refinement, and
a 7000-trial sweep across 50, 100 and 400 bins found no correlation at all
between its predicate and the true error, and no case where removing it let a
wrong answer through: the area indicator, which counts boundary cells and so
scales like ``1/source_bins``, rejects the under-resolved rows on its own.

These tests pin the property that replaced it -- not the calibration.  A large
source over a planetary caustic must either report a value inside the budget it
claims, or refuse; what it must not do is refuse a value it computed correctly.
"""

import pytest


SEPARATION = 1.0
MASS_RATIO = 1.0e-3
U0 = -0.01
ALPHA = 0.5
TIME = 0.05

# VBMicrolensing BinaryLightCurve at Tol=1e-9.  These radii all sit inside the
# old guard's band at every bin count it could reach: q = 1e-3 against
# 4 rho / 400 = 1e-3 rho, so the predicate fires for every rho above 1.0.
# Beyond that the disk swallows the whole planetary caustic, which is the
# regime the guard made unreachable.
REFERENCES = {
    0.1: 18.731285456,
    0.2: 9.909054417,
    0.3: 6.703656119,
    0.5: 4.116591492,
    0.8: 2.691526204,
}

RELTOL = 1.0e-3


def _info(lcbinint, source_radius, **options):
    curve = lcbinint.LightCurve(
        lens="binary",
        options=lcbinint.Options(
            coordinates="vbm",
            inverse_ray_grid="cartesian",
            **options,
        ),
    )
    return curve.info(
        TIME,
        t0=0.0,
        tE=1.0,
        u0=U0,
        alpha=ALPHA,
        s=SEPARATION,
        q=MASS_RATIO,
        rho=source_radius,
        limb_darkening_c=0.0,
    )


def _scalars(info):
    import numpy as np

    return (
        float(np.asarray(info.finite_source_magnifications).ravel()[0]),
        float(np.asarray(info.finite_source_error_estimates).ravel()[0]),
        bool(np.asarray(info.finite_source_converged).ravel()[0]),
    )


@pytest.mark.parametrize("source_radius", sorted(REFERENCES))
def test_large_source_over_a_planetary_caustic_is_answered(source_radius):
    """The band is reachable: these all used to be NaN."""
    lcbinint = pytest.importorskip("lcbinint")
    info = _info(
        lcbinint, source_radius,
        nbin="auto", max_source_bins=400, reltol=RELTOL,
    )
    value, estimate, converged = _scalars(info)
    reference = REFERENCES[source_radius]

    assert converged, "a certified, resolved disk must not be refused"
    assert value == pytest.approx(reference, rel=1.0e-3)
    # The reported budget has to be honest about the value it accompanies.
    assert estimate <= RELTOL * max(abs(value), 1.0)


@pytest.mark.parametrize("source_radius", sorted(REFERENCES))
def test_the_reported_budget_bounds_the_true_error(source_radius):
    """Convergence is a claim about accuracy, so check it against VBM.

    The area indicator is a boundary-cell count, so it is an upper bound rather
    than an estimate -- the true error here runs one to two orders of magnitude
    below what is reported.  The contract is only the inequality.
    """
    lcbinint = pytest.importorskip("lcbinint")
    info = _info(
        lcbinint, source_radius,
        nbin="auto", max_source_bins=400, reltol=RELTOL,
    )
    value, estimate, converged = _scalars(info)
    if not converged:
        pytest.skip("refused, so it makes no accuracy claim")
    assert abs(value - REFERENCES[source_radius]) <= estimate


def test_coarse_grids_are_still_refused():
    """Removing the guard did not remove the floor under the indicator.

    At 16 bins a rho=0.3 disk is four cells across the planetary caustic, and
    the boundary count that replaced the guard scales like 1/source_bins -- so
    it grows into exactly the regime the guard was trying to name, and refuses
    on its own without a hand-set constant.
    """
    lcbinint = pytest.importorskip("lcbinint")
    info = _info(
        lcbinint, 0.3,
        source_bins=16, max_source_bins=16, finite_source_reltol=1.0e-5,
    )
    _, estimate, converged = _scalars(info)
    assert not converged
    assert estimate > 1.0e-5
