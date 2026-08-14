"""Hand-picked geometries pinning the calibrated point-source safety margins.

Each test here fixes one decision at one place: a specific cusp, ghost or
planetary indicator must veto (or release) a specific fast path.  They pin the
calibration, not the property it exists to deliver -- for that, see
`test_smooth_route_clearance.py`, which asserts over a randomised near-caustic
sample that no smooth expansion is ever chosen for a disk the caustic provably
enters.
"""

import math

import numpy as np
import pytest


def _point_magnification(lcbinint, s, q, x, y):
    curve = lcbinint.LightCurve(options=lcbinint.Options(coordinates="center_of_mass"))
    return curve([0.0], t0=-x, tE=1.0, u0=y, alpha=0.0,
                 s=s, q=q, rho=0.0)[0]


def test_binary_trajectory_uses_validated_root_continuation():
    lcbinint = pytest.importorskip("lcbinint")
    curve = lcbinint.LightCurve(
        options=lcbinint.Options(coordinates="center_of_mass")
    )
    times = np.linspace(-0.8, 0.8, 64)
    params = {
        "t0": 0.0,
        "tE": 1.0,
        "u0": 0.05,
        "alpha": 0.2,
        "s": 1.2,
        "q": 1.0e-2,
        "rho": 0.0,
    }

    info = curve.info(times, params)

    assert sum(info.root_used_warm_start) >= len(times) - 1
    assert sum(info.root_used_cold_retry) < len(times) // 4
    assert max(info.root_max_residuals) < 1.0e-8
    assert set(info.image_counts) <= {3, 5}


def test_binary_safety_diagnostic_reuses_point_source_solution():
    lcbinint = pytest.importorskip("lcbinint")

    diagnostic = lcbinint._binary_safety_diagnostic(1.35, 0.32, 0.9, 0.0)

    assert diagnostic.magnification == pytest.approx(
        _point_magnification(lcbinint, 1.35, 0.32, 0.9, 0.0)
    )
    assert diagnostic.image_count == 3
    assert diagnostic.ghost_count == 2
    assert diagnostic.quadrupole_indicator > 0.0
    assert diagnostic.ghost_indicator > 0.0


def test_binary_safety_diagnostic_has_no_ghosts_inside_caustic():
    lcbinint = pytest.importorskip("lcbinint")

    diagnostic = lcbinint._binary_safety_diagnostic(1.35, 0.32, 0.0, 0.0)

    assert diagnostic.image_count == 5
    assert diagnostic.ghost_count == 0
    assert diagnostic.ghost_indicator == 0.0
    assert diagnostic.cusp_indicator > 0.0


def test_planetary_caustic_safety_vetoes_point_and_hex_fast_paths():
    lcbinint = pytest.importorskip("lcbinint")
    light_curve = lcbinint.LightCurve(
        options=lcbinint.Options(
            coordinates="center_of_mass",
            caustic_bins=400,
            nbin="auto",
            tol=1.0e-2,
            hex_tol=1.0e-2,
            inverse_ray_grid="cartesian",
        )
    )

    # Planetary-caustic configuration used in MNRAS 479, 5157, Fig. 5.
    info = light_curve.info(
        [3.397671299],
        t0=0.0,
        tE=1.0,
        u0=0.01,
        alpha=0.0,
        s=3.67,
        q=1.0e-6,
        rho=1.0e-2,
    )

    assert info.finite_source_method_names[0] not in {
        "point_source",
        "hexadecapole",
    }
    assert info.point_source_ghost_counts == [2]
    assert info.point_source_planetary_distances2[0] < 2.0e-4
    assert info.point_source_safety_flags[0] & 4 == 0


def test_cusp_safety_can_fall_through_to_accurate_hexadecapole():
    lcbinint = pytest.importorskip("lcbinint")
    vbm_module = pytest.importorskip("VBMicrolensing")
    light_curve = lcbinint.LightCurve(
        options=lcbinint.Options(
            coordinates="center_of_mass",
            caustic_bins=400,
            nbin="auto",
            tol=1.0e-2,
            hex_tol=1.0e-2,
        )
    )
    x = 0.8155011021
    info = light_curve.info(
        [x],
        t0=0.0,
        tE=1.0,
        u0=0.0,
        alpha=0.0,
        s=1.35,
        q=0.32,
        rho=1.0e-2,
    )
    reference = vbm_module.VBMicrolensing().BinaryMagDark(
        1.35, 0.32, x, 0.0, 1.0e-2, 1.0e-5
    )

    assert info.finite_source_method_names == ["hexadecapole"]
    assert info.magnifications[0] == pytest.approx(reference, abs=1.0e-4)


def test_ghost_safety_margin_blocks_fold_outside_false_point_source():
    lcbinint = pytest.importorskip("lcbinint")
    s = 1.5
    q = 0.1
    rho = 1.0e-3
    x = 0.7821993300517519
    y = 0.1628548323146869
    diagnostic = lcbinint._binary_safety_diagnostic(s, q, x, y)
    safety_radius = rho + 1.0e-3

    # A coefficient of one would accept this point, despite a finite-source
    # correction of order unity.  The calibrated factor of two rejects it.
    assert safety_radius * diagnostic.ghost_indicator < 1.0
    assert 2.0 * safety_radius * diagnostic.ghost_indicator > 1.0

    light_curve = lcbinint.LightCurve(
        options=lcbinint.Options(
            coordinates="center_of_mass",
            caustic_bins=400,
            nbin="auto",
            tol=1.0e-2,
            hex_tol=1.0e-2,
            inverse_ray_grid="cartesian",
        )
    )
    info = light_curve.info(
        [x],
        t0=0.0,
        tE=1.0,
        u0=y,
        alpha=0.0,
        s=s,
        q=q,
        rho=rho,
    )

    assert info.finite_source_method_names[0] not in {
        "point_source",
        "hexadecapole",
    }
    assert info.point_source_safety_flags[0] & 2 == 0


def test_ghost_safety_margin_is_independently_safe_on_broad_sweep_case():
    lcbinint = pytest.importorskip("lcbinint")
    s = 0.9807009185049715
    q = 0.04812606069889609
    rho = 0.004349576458703525
    x = 0.12423799972033645
    y = 0.11965557805931838
    diagnostic = lcbinint._binary_safety_diagnostic(s, q, x, y)
    safety_radius = rho + 1.0e-3

    # The broad coefficient sweep found that a factor of two still admits this
    # point, whose point-source error is about 129 times the requested tolerance.
    assert 2.0 * safety_radius * diagnostic.ghost_indicator < 1.0
    assert 3.0 * safety_radius * diagnostic.ghost_indicator > 1.0

    light_curve = lcbinint.LightCurve(
        options=lcbinint.Options(
            coordinates="center_of_mass",
            caustic_bins=400,
            nbin="auto",
            tol=0.007556554954751993,
            hex_tol=0.007556554954751993,
            inverse_ray_grid="cartesian",
        )
    )
    info = light_curve.info(
        [x],
        t0=0.0,
        tE=1.0,
        u0=y,
        alpha=0.0,
        rho=rho,
        s=s,
        q=q,
    )

    assert info.point_source_safety_flags[0] & 2 == 0
    assert info.finite_source_method_names[0] != "point_source"


def test_refined_distance_releases_remote_planetary_topology_proxy():
    """Disconnected caustic proxies must not force remote Roman epochs to IR."""
    t0 = 206.741143614
    rho = math.exp(-7.08071184808)
    lcbinint = pytest.importorskip("lcbinint")
    light_curve = lcbinint.LightCurve(
        options=lcbinint.Options(coordinates="vbm")
    )
    info = light_curve.info(
        [t0 - 5.0, t0 - 3.0],
        t0=t0,
        tE=60.0,
        u0=0.025,
        alpha=1.22173047640,
        s=math.exp(0.05666264897),
        q=0.001,
        rho=rho,
        limb_darkening_c=0.407200474,
    )

    # The raw planetary proxy fails at both positions, but the measured
    # caustic distances are 69 and 34 source radii.  The tolerance-aware local
    # check chooses point and hex respectively instead of inverse rays.
    assert all(flags & 4 == 0 for flags in info.point_source_safety_flags)
    assert min(distance / rho for distance in info.caustic_distances) > 20.0
    assert info.finite_source_method_names == ["point_source", "hexadecapole"]
    assert info.magnifications == pytest.approx(
        [11.39063613233733, 17.449601693756392], rel=1.0e-3
    )


def test_forced_cartesian_high_magnification_does_not_truncate_image_area():
    lcbinint = pytest.importorskip("lcbinint")
    light_curve = lcbinint.LightCurve(
        options=lcbinint.Options(
            coordinates="center_of_mass",
            caustic_bins=600,
            source_bins=40,
            max_source_bins=40,
            tol=5.0e-4,
            hex_tol=5.0e-4,
            inverse_ray_grid="cartesian",
        )
    )
    info = light_curve.info(
        [-5.98297901390347e-5],
        t0=0.0,
        tE=1.0,
        u0=1.2570751933400963e-4,
        alpha=0.0,
        rho=1.3791586509913708e-4,
        s=0.36217490830115934,
        q=2.3676600414698552e-5,
    )

    # Dense integration of the mutually consistent lcbinint and
    # VBMicrolensing point-source solutions converges near 9,000 here.  The
    # finite-source VBMicrolensing API is not used as the oracle because it is
    # pathological in this extreme-magnification case.
    assert info.finite_source_method_names == ["inverse_ray_cartesian"]
    assert info.finite_source_magnifications[0] == pytest.approx(9.0e3, rel=6.0e-3)


def test_forced_cartesian_fold_walk_uses_global_magnification_guard():
    lcbinint = pytest.importorskip("lcbinint")
    light_curve = lcbinint.LightCurve(
        options=lcbinint.Options(
            coordinates="center_of_mass",
            caustic_bins=1200,
            source_bins=16,
            max_source_bins=16,
            point_source_threshold=0.0,
            hexadecapole_threshold=0.0,
            adaptive_hex_threshold=0.0,
            inverse_ray_grid="cartesian",
        )
    )
    info = light_curve.info(
        [-1.6433642987581792e-4],
        t0=0.0,
        tE=1.0,
        u0=-2.804330245953445e-4,
        alpha=0.0,
        rho=3.11831703999747e-4,
        s=0.9665580853710293,
        q=3.9768857748147137e-5,
    )

    # A seed near the end of this long fold image has local magnification only
    # ~18, while the connected finite-source image has magnification ~3954.
    # A seed-local walk guard used to stop normally progressing rows and turn
    # every tested Cartesian resolution into a numerical error.
    assert info.finite_source_method_names == ["inverse_ray_cartesian"]
    assert info.finite_source_magnifications[0] == pytest.approx(3954.0, rel=2.0e-3)
