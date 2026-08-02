"""Acceptance tests for the certified triple component support.

The completeness argument behind ``certify_disk_support`` never mentions the
number of lenses: every component of ``D \\ K`` either is all of ``D`` or has a
caustic arc on its boundary, and on such an arc the distance to the disk centre
attains a local extremum inside ``D``.  So the same certificate that fixed the
binary tangency (see ``test_binary_cusp_component``) ports to the triple lens,
where it matters more -- a triple caustic has many more arcs that can graze a
disk, and the nested caustics rule out any fixed image-count floor as a proxy.

The geometry here is built the way the binary one was: take the caustic
polyline, measure the true minimum distance ``d`` from the chosen source centre
to it, and set ``rho = d / 0.99`` so the extra-image cap is ``0.01 * rho`` deep
by construction.  Without the certificate the seeding misses that cap and the
adaptive integrator converges -- and reports convergence -- on a value short by
a whole image component.
"""

import pytest


PARAMS = {
    "t0": 0.0,
    "tE": 1.0,
    "alpha": 0.0,
    "s": 1.0,
    "q": 1.0e-3,
    "q2": 1.0e-4,
    "sep2": 0.5,
    "ang": 1.2,
}
SOURCE_X = -0.05
SOURCE_Y = 0.02
# Minimum distance from (SOURCE_X, SOURCE_Y) to the caustic is 6.497855561e-03,
# measured on a 3000-point-per-branch polyline; the disk overlaps it by 1%.
SOURCE_RADIUS = 6.497855561e-03 / 0.99

# lcbinint's own converged value: adaptive precision at finite_source_reltol
# 1e-4 and 1e-5 agree to all printed digits, and the fixed-grid ladder is
# approaching it from below (2048 bins gives 17.500497508).  VBMicrolensing is
# not used as an external reference for triple lenses here.
UNIFORM_REFERENCE = 17.500498640914

# Same geometry with a linear limb-darkening coefficient of 0.5, from the
# 2048-bin fixed grid.
LINEAR_LIMB_REFERENCE = 17.490127816121

# What the magnification collapses to when the capped component is lost: the
# value the adaptive integrator converged on, and certified as converged, before
# the certificate existed.  The gap is 3.1e-2, i.e. 1.8e-3 relative.
UNIFORM_MISSING_COMPONENT = 17.469486455187


def _curve(lcbinint, **options):
    return lcbinint.LightCurve(
        lens="triple",
        options=lcbinint.Options(inverse_ray_grid="cartesian", **options),
    )


def _magnification(
    curve, source_x=SOURCE_X, source_y=SOURCE_Y, source_radius=SOURCE_RADIUS,
    limb_darkening_c=0.0,
):
    return curve.magnification(
        source_x,
        u0=source_y,
        rho=source_radius,
        limb_darkening_c=limb_darkening_c,
        **PARAMS,
    ).item()


def _ladder(lcbinint, bins_ladder, limb_darkening_c=0.0):
    values = []
    for bins in bins_ladder:
        curve = _curve(lcbinint, source_bins=bins, max_source_bins=bins)
        values.append(_magnification(curve, limb_darkening_c=limb_darkening_c))
    return values


@pytest.mark.parametrize(
    ("limb_darkening_c", "reference"),
    ((0.0, UNIFORM_REFERENCE), (0.5, LINEAR_LIMB_REFERENCE)),
    ids=("uniform", "linear_limb"),
)
def test_certified_triple_component_refines_to_reference(
    limb_darkening_c, reference
):
    lcbinint = pytest.importorskip("lcbinint")
    bins_ladder = (64, 128, 256, 512)
    values = _ladder(lcbinint, bins_ladder, limb_darkening_c)

    assert values[-1] == pytest.approx(reference, abs=1.0e-3)
    deltas = [abs(b - a) for a, b in zip(values, values[1:])]
    assert deltas == sorted(deltas, reverse=True)
    assert all(later < 0.6 * earlier for earlier, later in zip(deltas, deltas[1:]))


def test_certified_triple_component_is_not_a_resolution_artefact():
    """The certificate is geometric, so the component appears at every bin count.

    This is the assertion the probe rings cannot satisfy: their offsets are a
    fixed fraction of rho, so whether they land inside a cap this shallow is a
    property of the bin count rather than of the lens.  A support derived from
    the caustic alone makes the sequence strictly monotone instead.
    """
    lcbinint = pytest.importorskip("lcbinint")
    bins_ladder = (32, 48, 64, 96, 128, 192, 256)
    values = _ladder(lcbinint, bins_ladder)
    for (coarse_bins, coarse), (fine_bins, fine) in zip(
        zip(bins_ladder, values), zip(bins_ladder[1:], values[1:])
    ):
        assert fine > coarse, (
            f"source_bins={fine_bins} lost magnification relative to "
            f"{coarse_bins}: {fine} < {coarse}"
        )

    gap = UNIFORM_REFERENCE - UNIFORM_MISSING_COMPONENT
    for bins, value in zip(bins_ladder, values):
        assert value > UNIFORM_MISSING_COMPONENT + 0.5 * gap, (
            f"capped component missing at source_bins={bins}: {value}"
        )


def test_triple_adaptive_precision_reaches_the_reference():
    """The failure this replaces was silent: it converged on the wrong value.

    Both tolerances used to settle on ``UNIFORM_MISSING_COMPONENT`` with
    ``all_converged`` true, so tightening the tolerance did not expose it.
    """
    lcbinint = pytest.importorskip("lcbinint")
    for reltol in (1.0e-4, 1.0e-5):
        curve = _curve(
            lcbinint, finite_source_reltol=reltol, max_source_bins=4096
        )
        value = _magnification(curve)
        assert value == pytest.approx(UNIFORM_REFERENCE, rel=1.0e-5), (
            f"finite_source_reltol={reltol:g}"
        )


@pytest.mark.parametrize(
    ("source_x", "source_y", "source_radius", "expected"),
    (
        (-0.05, 0.02, 0.00325, 17.471844735930),
        (-0.12, -0.06, 0.02, 7.363833436941),
        (0.06, 0.06, 0.02, 12.119997681164),
    ),
    ids=("near_tangency_clear", "outer_fold_clear", "cusp_clear"),
)
def test_certified_triple_support_leaves_clear_geometries_alone(
    source_x, source_y, source_radius, expected
):
    """With the caustic outside the disk the certificate finds no extremum.

    Each of these centres sits two to three source radii clear of the nearest
    caustic point, so the certified stage emits no probe and solves no lens
    equation.  The pinned values are lcbinint's own 512-bin results; what is
    being guarded is that they neither moved when the certificate was added nor
    depend on the bin count.
    """
    lcbinint = pytest.importorskip("lcbinint")
    values = [
        _magnification(
            _curve(lcbinint, source_bins=bins, max_source_bins=bins),
            source_x=source_x,
            source_y=source_y,
            source_radius=source_radius,
        )
        for bins in (128, 512)
    ]
    assert values[1] == pytest.approx(expected, rel=1.0e-9)
    assert values[0] == pytest.approx(values[1], rel=1.0e-4)
