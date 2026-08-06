"""Acceptance tests for the certified binary component support.

When the caustic is nearly tangent to the source limb, the five-image region is
a cap only ``4e-4 * rho`` deep.  Every limb raster the seeding used to rely on
either hit that cap or missed it depending on how many limb samples it drew, so
the magnification silently converged to a value that was short by the whole
extra image pair (3.9485 instead of 3.9609, i.e. -3.1e-3 relative) while the
error indicator still reported convergence.

``certify_disk_support`` replaces the raster with a geometric
completeness certificate: it probes both normals of every local extremum of the
caustic-to-centre distance, which reaches every component of ``D \\ K`` for
reasons that do not mention any grid or sample count.  These tests pin both
halves of that contract -- the component is found, and it is found independently
of the integration resolution.
"""

import pytest


SEPARATION = 1.2
MASS_RATIO = 0.1
SOURCE_X = 0.653
SOURCE_Y = 0.020
SOURCE_RADIUS = 0.020

# VBMicrolensing BinaryMag2 at Tol=1e-9 and Tol=1e-10 (both agree to 1.1e-10).
# Note that Tol=1e-7 returns 3.960889170843, which is 6.7e-7 high; the tangency
# is where VBM's own accuracy goal stops being met.
UNIFORM_REFERENCE = 3.960888498085

# lcbinint's own converged value (adaptive precision, finite_source_reltol
# 1e-4 and 1e-5 agree to all printed digits).  VBM is not usable as an external
# reference for this one: BinaryMagDark returns 3.9675 for a1=0.25 but 4.5751
# for a1=0.5 on this geometry, so its limb-darkened annulus scheme has broken
# down before it reaches the coefficient under test.
LINEAR_LIMB_REFERENCE = 3.836158013

# What the magnification collapses to when the extra image pair is lost.  Any
# tolerance well inside this gap distinguishes "integrated coarsely" from
# "integrated a different image set".
UNIFORM_MISSING_COMPONENT = 3.948480


def _curve(lcbinint, **options):
    return lcbinint.LightCurve(
        lens="binary",
        options=lcbinint.Options(
            coordinates="center_of_mass",
            inverse_ray_grid="cartesian",
            **options,
        ),
    )


def _magnification(curve, source_y=SOURCE_Y, limb_darkening_c=0.0):
    return curve.magnification(
        SOURCE_X,
        t0=0.0,
        tE=1.0,
        u0=source_y,
        alpha=0.0,
        s=SEPARATION,
        q=MASS_RATIO,
        rho=SOURCE_RADIUS,
        limb_darkening_c=limb_darkening_c,
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
def test_certified_component_refines_to_reference(limb_darkening_c, reference):
    lcbinint = pytest.importorskip("lcbinint")
    bins_ladder = (64, 128, 256, 512)
    values = _ladder(lcbinint, bins_ladder, limb_darkening_c)

    # The cap is present at every resolution, so the sequence approaches the
    # reference from below instead of settling on the three-image value.
    assert values[-1] == pytest.approx(reference, abs=1.0e-3)
    deltas = [abs(b - a) for a, b in zip(values, values[1:])]
    assert deltas == sorted(deltas, reverse=True)
    # The hybrid chooses the larger of the legacy and multi-run component
    # footprints, so the first handoff can sit just above a factor of two.  The
    # sequence must still show clear convergence, and becomes much faster once
    # the cap is a few cells deep.
    assert all(later < 0.55 * earlier for earlier, later in zip(deltas, deltas[1:]))


def test_certified_component_is_not_a_resolution_artefact():
    """The certificate is geometric, so the component appears at every bin count.

    The historical failure mode was resolution dependent: some bin counts drew a
    limb sample inside the 0.088 rad cap and some did not, so the sequence
    dipped wherever the cap was missed.  A support derived from the caustic
    alone cannot do that, and the sequence is strictly monotone instead.
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

    # Once the grid itself is fine enough to be worth judging, the value must
    # sit in the upper half of the gap between the three-image answer and the
    # reference.  Losing the pair would put it at or below the lower end.
    gap = UNIFORM_REFERENCE - UNIFORM_MISSING_COMPONENT
    for bins, value in zip(bins_ladder, values):
        if bins < 128:
            continue
        assert value > UNIFORM_MISSING_COMPONENT + 0.5 * gap, (
            f"extra image pair missing at source_bins={bins}: {value}"
        )


@pytest.mark.parametrize("reltol", (1.0e-3, 1.0e-4, 1.0e-5))
def test_adaptive_precision_meets_the_tolerance_it_is_given(reltol):
    """The requested tolerance is the contract, at every tolerance.

    This used to assert 1e-5 at ``finite_source_reltol=1e-4``, which the
    adaptive loop delivered by accident: its error estimate is the raw
    difference between the calibrated grid and half of it, so on this geometry
    it exceeded the 1e-4 budget at 400 bins and escalated straight to the 4096
    cap -- 1.2 s of work for a 1e-4 request.  Per-component refinement cuts the
    error at a fixed resolution by ~2.3x here, the difference the estimate is
    built from falls with it, and the loop stops at 400 bins with a 3.2e-5
    error in 22 ms.  Both answers honour the tolerance they were asked for;
    only the second one is worth its cost, so the tolerance is what is pinned.
    """
    lcbinint = pytest.importorskip("lcbinint")
    curve = _curve(
        lcbinint,
        finite_source_reltol=reltol,
        max_source_bins=4096,
    )
    value = _magnification(curve)
    assert value == pytest.approx(UNIFORM_REFERENCE, rel=reltol)
    # Meeting a loose tolerance by losing the extra image pair and landing near
    # 3.9485 would satisfy 1e-3 on its own; the certificate has to hold too.
    gap = UNIFORM_REFERENCE - UNIFORM_MISSING_COMPONENT
    assert value > UNIFORM_MISSING_COMPONENT + 0.5 * gap


def test_tangency_sweep_stays_accurate_on_both_sides():
    """Sweeping the source through the tangency must not spike.

    ``u0`` between 0.01999 and 0.02001 moves the caustic from just inside the
    limb to just outside it.  Without the certificate the error jumps to
    -3.1e-3 exactly where the cap becomes thin; with it the error stays inside
    the ordinary discretisation budget of a 256-bin Cartesian grid.
    """
    lcbinint = pytest.importorskip("lcbinint")
    vbm = pytest.importorskip("VBMicrolensing")
    reference_engine = vbm.VBMicrolensing()
    reference_engine.RelTol = 0.0
    reference_engine.Tol = 1.0e-9

    curve = _curve(lcbinint, source_bins=256, max_source_bins=256)
    for source_y in (0.01990, 0.01995, 0.019990, 0.020000, 0.020001,
                     0.020005, 0.020010, 0.02005):
        reference = reference_engine.BinaryMag2(
            SEPARATION, MASS_RATIO, SOURCE_X, source_y, SOURCE_RADIUS)
        value = _magnification(curve, source_y=source_y)
        assert value == pytest.approx(reference, rel=5.0e-4), (
            f"u0={source_y}: {value} vs {reference}"
        )


@pytest.mark.parametrize(
    ("separation", "mass_ratio", "y1", "y2", "source_radius"),
    (
        (1.5, 1.0, 0.4, -0.3, 0.02),
        (0.7, 0.3, -0.6, 0.4, 0.03),
        (1.0, 1.0e-3, 0.3, 0.4, 0.01),
    ),
    ids=("wide_equal_mass", "close_binary", "planetary"),
)
def test_certified_support_leaves_clear_geometries_alone(
    separation, mass_ratio, y1, y2, source_radius
):
    """With the caustic clear of the disk the certificate finds no extremum.

    It then emits no probe and solves no lens equation, so these cases must keep
    the accuracy they had before the certified stage existed.
    """
    lcbinint = pytest.importorskip("lcbinint")
    vbm = pytest.importorskip("VBMicrolensing")
    reference_engine = vbm.VBMicrolensing()
    reference_engine.RelTol = 0.0
    reference_engine.Tol = 1.0e-9
    reference = reference_engine.BinaryMag2(
        separation, mass_ratio, y1, y2, source_radius)

    value = lcbinint.LightCurve(
        lens="binary",
        options=lcbinint.Options(
            coordinates="center_of_mass",
            inverse_ray_grid="cartesian",
            source_bins=128,
            max_source_bins=128,
        ),
    ).magnification(
        y1, t0=0.0, tE=1.0, u0=y2, alpha=0.0,
        s=separation, q=mass_ratio, rho=source_radius,
    ).item()
    assert value == pytest.approx(reference, rel=1.0e-4)
