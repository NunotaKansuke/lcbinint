"""Acceptance tests for per-component grid refinement.

The Cartesian image-plane lattice is sized from the *source* (``rho / bins``),
which says nothing about the images it has to resolve.  Near a tangency the two
fold images form a sliver -- 55:1 on the reference cusp geometry -- so at 64
bins it is 2.1 cells across and 115 along, and the row scan is integrating a
width it barely samples.  That, not a missing component, is the residual
``h^1.7`` order there.

Refining the whole grid to fix it costs ``k^2`` over the whole disk.  Refining
the one component costs ``k^2`` over the sliver.  These tests pin the two
halves of that trade: the thin component gets measurably better, and every
geometry that has no thin component remains numerically unchanged.

The failure mode the guards exist for -- a coarse fill whose extent was decided
by a neighbour's claims, refined alone on an empty lattice, running away into
the neighbour and counting its area a second time -- is pinned by
``test_point_source_safety.test_forced_cartesian_high_magnification_does_not_truncate_image_area``,
where it doubled a 9000x magnification.
"""

import pytest


# VBMicrolensing BinaryMag2 at Tol=1e-9/1e-10.
CUSP_REFERENCE = 3.960888498085
CUSP_PARAMS = dict(s=1.2, q=0.1)
CUSP_X, CUSP_U0, CUSP_RHO = 0.653, 0.020, 0.020

# The same ladder on the uniform grid, i.e. with refinement disabled.  These are
# the numbers per-component refinement has to beat; they are recorded rather
# than recomputed because the point of the test is the comparison.
CUSP_UNIFORM_LADDER = {
    32: 3.942493231,
    64: 3.952459074,
    128: 3.958808689,
    256: 3.960254926,
    512: 3.960698849,
}

TRIPLE_CAP_REFERENCE = 17.500498641
TRIPLE_CAP_PARAMS = dict(s=1.0, q=1.0e-3, q2=1.0e-4, sep2=0.5, ang=1.2)
TRIPLE_CAP_X, TRIPLE_CAP_U0 = -0.05, 0.02
TRIPLE_CAP_RHO = 6.497855561e-03 / 0.99

# Geometries with no thin component: refinement must not fire.  The scanline
# multi-interval boundary correction may change the last few parts in 1e9.
CLEAR_GEOMETRIES = (
    ("wide_equal_mass", dict(s=1.8, q=1.0), 0.30, 0.10, 0.02, 128,
     1.644285791543),
    ("close_binary", dict(s=0.6, q=0.5), 0.05, 0.02, 0.01, 128,
     12.302093936985),
    ("planetary", dict(s=1.05, q=1.0e-3), 0.05, 0.02, 0.005, 128,
     17.450204716764),
    ("caustic_crossing", dict(s=1.0, q=0.2), 0.10, 0.05, 0.01, 128,
     5.366034638431),
    ("far_from_caustic", dict(s=1.0, q=1.0e-3), 0.40, 0.30, 0.01, 128,
     2.177131051261),
)


def _magnification(lcbinint, lens, params, x, u0, rho, bins):
    options = dict(inverse_ray_grid="cartesian", source_bins=bins,
                   max_source_bins=bins)
    if lens == "binary":
        options["coordinates"] = "center_of_mass"
    curve = lcbinint.LightCurve(lens=lens, options=lcbinint.Options(**options))
    return curve.magnification(
        x, t0=0.0, tE=1.0, u0=u0, alpha=0.0, rho=rho, **params).item()


def _cusp(lcbinint, bins):
    return _magnification(
        lcbinint, "binary", CUSP_PARAMS, CUSP_X, CUSP_U0, CUSP_RHO, bins)


@pytest.mark.parametrize("bins", sorted(CUSP_UNIFORM_LADDER))
def test_thin_component_refinement_beats_the_uniform_grid(bins):
    """At every resolution the sliver is thinner than the trigger, so it fires.

    A factor of two is the interesting threshold: below it the refinement is
    not paying for the extra fill, and the measured factor is 2.1 to 3.1.
    """
    lcbinint = pytest.importorskip("lcbinint")
    value = _cusp(lcbinint, bins)
    error = abs(value - CUSP_REFERENCE)
    uniform_error = abs(CUSP_UNIFORM_LADDER[bins] - CUSP_REFERENCE)
    assert error < 0.5 * uniform_error, (
        f"source_bins={bins}: {error:.3e} vs uniform {uniform_error:.3e}"
    )
    # Refining a component can only add area the coarse lattice could not see,
    # and the sliver is entirely inside the disk, so the approach stays from
    # below.  Overshooting means a component was counted twice.
    assert value < CUSP_REFERENCE


def test_thin_component_refinement_keeps_the_ladder_monotone():
    """The refinement factor is chosen per component, so it changes with bins.

    A scheme whose accuracy jumped around as ``k`` stepped would be useless to
    the adaptive loop, which reads its error estimate off the difference
    between two resolutions.
    """
    lcbinint = pytest.importorskip("lcbinint")
    values = [_cusp(lcbinint, bins) for bins in sorted(CUSP_UNIFORM_LADDER)]
    for coarse, fine in zip(values, values[1:]):
        assert fine > coarse
    deltas = [fine - coarse for coarse, fine in zip(values, values[1:])]
    assert deltas == sorted(deltas, reverse=True)


def test_thin_component_refinement_holds_on_the_triple_cap():
    """The same sliver appears in the triple five-image cap.

    The triple lens reaches the Cartesian core by a different route (its own
    seed set and image map), so it is the check that the refinement is a
    property of the fill and not of the binary caller.
    """
    lcbinint = pytest.importorskip("lcbinint")
    bins_ladder = (32, 64, 128, 256)
    values = [
        _magnification(lcbinint, "triple", TRIPLE_CAP_PARAMS, TRIPLE_CAP_X,
                       TRIPLE_CAP_U0, TRIPLE_CAP_RHO, bins)
        for bins in bins_ladder
    ]
    for coarse, fine in zip(values, values[1:]):
        assert fine > coarse
    for bins, value in zip(bins_ladder, values):
        assert value < TRIPLE_CAP_REFERENCE
    assert values[-1] == pytest.approx(TRIPLE_CAP_REFERENCE, rel=1.0e-5)


@pytest.mark.parametrize(
    ("name", "params", "x", "u0", "rho", "bins", "expected"),
    CLEAR_GEOMETRIES,
    ids=[case[0] for case in CLEAR_GEOMETRIES],
)
def test_refinement_leaves_geometries_without_a_thin_component_alone(
    name, params, x, u0, rho, bins, expected
):
    lcbinint = pytest.importorskip("lcbinint")
    value = _magnification(lcbinint, "binary", params, x, u0, rho, bins)
    assert value == pytest.approx(expected, rel=1.0e-8)
