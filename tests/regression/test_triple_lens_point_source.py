import numpy as np
import pytest

import lcbinint


TRIPLE_POINT_REFERENCE_CASES = [
    pytest.param(
        -0.09263782795758546,
        -0.03908195790173323,
        1.0,
        1.0e-3,
        1.0e-4,
        0.5,
        1.2,
        10.529790084883288,
        5.0e-4,
        id="planetary_subsystem_left",
    ),
    pytest.param(
        -0.00479425538604203,
        0.008775825618903728,
        1.0,
        1.0e-3,
        1.0e-4,
        0.5,
        1.2,
        118.58394756835955,
        5.0e-4,
        id="planetary_subsystem_high_magnification",
    ),
    pytest.param(
        0.17067435180044185,
        0.10449139266017765,
        1.0,
        1.0e-3,
        1.0e-4,
        0.5,
        1.2,
        5.0081788428186362,
        5.0e-4,
        id="planetary_subsystem_right",
    ),
    pytest.param(
        0.35,
        -0.22,
        0.8,
        0.03,
        0.02,
        0.35,
        -0.7,
        2.3663298774361103,
        1.0e-9,
        id="moderate_inner_pair",
    ),
    pytest.param(
        -0.45,
        0.18,
        1.4,
        0.2,
        0.05,
        0.7,
        2.1,
        2.5951753373288202,
        1.0e-9,
        id="wide_primary",
    ),
]


@pytest.mark.parametrize(
    "source_x,source_y,s,q,q2,sep2,ang,reference,tolerance",
    TRIPLE_POINT_REFERENCE_CASES,
)
def test_triple_lens_point_source_matches_legacy_amp_point3(
    source_x,
    source_y,
    s,
    q,
    q2,
    sep2,
    ang,
    reference,
    tolerance,
):
    # References are hard-coded values generated from the legacy lcbinint.c
    # amp_point3 path.  The legacy code uses the original (non-VBM) geometry
    # convention, so we match it here with param_type='lcbinint'.  With alpha=0
    # both conventions give the same trajectory, so the test verifies geometry.
    light_curve = lcbinint.LightCurve(
        lens="triple", options=lcbinint.Options(param_type="lcbinint")
    )
    actual = light_curve(
        np.array([source_x]),
        {
            "t0": 0.0,
            "tE": 1.0,
            "u0": source_y,
            "alpha": 0.0,
            "s": s,
            "q": q,
            "q2": q2,
            "sep2": sep2,
            "ang": ang,
            "rho": 0.0,
        },
    )

    assert float(actual[0]) == pytest.approx(reference, rel=tolerance)


def test_triple_lens_light_curve_smoke():
    light_curve = lcbinint.LightCurve(lens="triple")
    times = np.array([-0.1, 0.0, 0.2])
    params = {
        "t0": 0.0,
        "tE": 1.0,
        "u0": 0.01,
        "alpha": 0.5,
        "s": 1.0,
        "q": 1.0e-3,
        "q2": 1.0e-4,
        "sep2": 0.5,
        "ang": 1.2,
        "rho": 0.0,
    }

    info = light_curve.info(times, params)

    assert np.all(np.isfinite(info.magnifications))
    assert info.magnifications == pytest.approx(info.point_source_magnifications)
    assert min(info.image_counts) > 0
    assert len(info.root_candidate_counts) == len(times)
    assert min(info.root_candidate_counts) >= min(info.image_counts)
    assert len(info.root_max_residuals) == len(times)
    assert max(info.root_max_residuals) < 1.0e-7
    assert set(info.root_needs_high_precision) <= {0, 1}
    assert set(info.root_used_high_precision) <= {0, 1}
    assert not any(info.root_used_high_precision)


def test_triple_lens_finite_source_cartesian_inverse_ray():
    light_curve = lcbinint.LightCurve(
        lens="triple",
        options=lcbinint.Options(mode=1),
    )
    times = np.array([0.0])
    params = {
        "s": 1.0,
        "q": 1.0e-3,
        "q2": 1.0e-4,
        "sep2": 0.5,
        "ang": 1.2,
        "rho": 0.0,
    }

    info = light_curve.info(times, {**params, "rho": 1.0e-3})

    assert np.isfinite(info.magnifications[0])
    assert info.finite_source_method_names == ["inverse_ray_cartesian"]
    assert info.finite_source_magnifications == pytest.approx(info.magnifications)


def test_triple_lens_auto_nbin_uses_calibrated_cartesian_resolution():
    """Triple ``nbin='auto'`` selects the validated fixed-grid bucket."""
    params = {
        "t0": 0.0,
        "tE": 1.0,
        "u0": 0.0,
        "alpha": 0.0,
        "s": 1.0,
        "q": 1.0e-3,
        "q2": 1.0e-4,
        "sep2": 0.5,
        "ang": 1.2,
        "rho": 1.0e-3,
    }
    common_options = dict(
        mode=1,
        point_source_threshold=20.0,
        hexadecapole_threshold=0.0,
        adaptive_hex_threshold=0.0,
        max_source_bins=400,
    )
    automatic = lcbinint.LightCurve(
        lens="triple", options=lcbinint.Options(nbin="auto", **common_options)
    ).info(np.array([0.0]), params)
    # The triple distance proxy is not used for runtime extrapolation.
    fixed = lcbinint.LightCurve(
        lens="triple", options=lcbinint.Options(nbin=256, **common_options)
    ).info(np.array([0.0]), params)

    assert automatic.finite_source_method_names == ["inverse_ray_cartesian"]
    assert automatic.magnifications == pytest.approx(fixed.magnifications, rel=0.0, abs=0.0)


def test_triple_lens_finite_source_uses_hexadecapole_between_point_and_ir():
    # Source at (-0.075, -0.025): derivative check fails but hexadecapole
    # self-consistency passes, so the mid-tier method is chosen.
    light_curve = lcbinint.LightCurve(
        lens="triple",
        options=lcbinint.Options(source_bins=10, caustic_bins=96),
    )
    times = np.array([-0.075])
    params = {
        "t0": 0.0,
        "tE": 1.0,
        "u0": -0.025,
        "alpha": 0.0,
        "s": 1.0,
        "q": 1.0e-3,
        "q2": 1.0e-4,
        "sep2": 0.5,
        "ang": 1.2,
        "rho": 3.0e-3,
    }

    info = light_curve.info(times, params)

    assert np.isfinite(info.magnifications[0])
    assert info.finite_source_method_names == ["hexadecapole"]
    assert info.finite_source_error_estimates[0] < 1.0e-3


def test_triple_lens_auto_blocks_fast_paths_inside_five_source_radii():
    """A calibrated near-caustic case must reach Cartesian integration.

    This was a false point-source result with a 4.87-rho caustic distance;
    its point result missed the 512-bin Cartesian reference by 1.3e-3.
    """
    light_curve = lcbinint.LightCurve(
        lens="triple",
        options=lcbinint.Options(param_type="lcbinint", caustic_bins=1400),
    )
    params = {
        "t0": 0.0,
        "tE": 1.0,
        "u0": -0.03650441437746342,
        "alpha": 0.0,
        "s": 4.0,
        "q": 1.0e-3,
        "q2": 1.0e-6,
        "sep2": 0.08,
        "ang": 2.6179938779914944,
        "rho": 1.0e-3,
    }

    info = light_curve.info(np.array([3.808742130329254]), params)

    assert info.finite_source_method_names == ["inverse_ray_cartesian"]
    assert info.magnifications[0] == pytest.approx(1.030906607613782, rel=1.0e-3)


def test_triple_lens_auto_uses_topology_safe_grazing_quadrature():
    """Both image-plane grids miss this outside-limb image finger."""
    light_curve = lcbinint.LightCurve(
        lens="triple",
        options=lcbinint.Options(param_type="lcbinint", caustic_bins=1400),
    )
    params = {
        "t0": 0.0,
        "tE": 1.0,
        "u0": 1.828067703049959,
        "alpha": 0.0,
        "s": 0.3,
        "q": 0.1,
        "q2": 1.0e-5,
        "sep2": 1.0,
        "ang": 0.0,
        "rho": 1.0e-4,
    }

    info = light_curve.info(np.array([-2.482023432364708]), params)

    assert info.finite_source_method_names == ["source_plane_quadrature"]
    assert info.finite_source_converged == [True]
    assert info.finite_source_refinement_levels == [1]
    assert info.magnifications[0] == pytest.approx(5.622472944839292, rel=1.0e-3)


def test_triple_lens_finite_source_approaches_point_source_for_small_rho():
    light_curve = lcbinint.LightCurve(
        lens="triple",
        options=lcbinint.Options(source_bins=12),
    )
    params = {
        "t0": 0.0,
        "tE": 1.0,
        "u0": -0.22,
        "alpha": 0.0,
        "s": 0.8,
        "q": 0.03,
        "q2": 0.02,
        "sep2": 0.35,
        "ang": -0.7,
    }

    point = light_curve(np.array([0.35]), {**params, "rho": 0.0})
    finite = light_curve(np.array([0.35]), {**params, "rho": 1.0e-5})

    assert float(finite[0]) == pytest.approx(float(point[0]), rel=2.0e-3)


def test_triple_lens_finite_source_auto_uses_seed_complete_polar():
    light_curve = lcbinint.LightCurve(
        lens="triple",
        options=lcbinint.Options(mode=4),
    )
    times = np.array([0.0])
    params = {
        "t0": 0.0,
        "tE": 1.0,
        "u0": 0.008,
        "alpha": 0.0,
        "s": 1.2,
        "q": 1.0e-3,
        "q2": 1.0e-3,
        "sep2": 0.1,
        "ang": 0.0,
        "rho": 5.0e-4,
    }
    info = light_curve.info(times, params)
    assert info.finite_source_method_names == ["inverse_ray_polar"]
    assert np.isfinite(info.magnifications[0])
    assert info.point_source_magnifications[0] >= 100.0


def test_triple_lens_polar_seeds_finite_source_only_fold_component():
    """Centre-image-only polar converged low by 1.2e-3 on this frozen row."""
    light_curve = lcbinint.LightCurve(
        lens="triple",
        options=lcbinint.Options(
            param_type="lcbinint",
            caustic_bins=1200,
            inverse_ray_grid="polar",
            nbin="auto",
            max_source_bins=512,
            polar_grid_ratio=12.0,
            point_source_threshold=1.0e9,
            hexadecapole_threshold=0.0,
            adaptive_hex_threshold=0.0,
        ),
        limb_darkening=lcbinint.LimbDarkening.linear(0.5),
    )
    params = {
        "t0": 0.0,
        "tE": 1.0,
        "u0": 0.025585304221408988,
        "alpha": 0.0,
        "s": 1.0,
        "q": 0.1,
        "q2": 1.0e-5,
        "sep2": 1.0,
        "ang": 1.5707963267948966,
        "rho": 1.0e-4,
    }

    info = light_curve.info(np.array([-0.046038516588439035]), params)

    assert info.finite_source_method_names == ["inverse_ray_polar"]
    assert info.magnifications[0] == pytest.approx(152.7134857937064, rel=1.0e-3)


def test_triple_lens_accepts_keyword_parameters():
    light_curve = lcbinint.LightCurve(lens="triple")
    times = np.array([0.0])

    magnifications = light_curve(
        times,
        t0=0.0,
        tE=1.0,
        u0=0.01,
        alpha=0.5,
        s=1.0,
        q=1.0e-3,
        q2=1.0e-4,
        sep2=0.5,
        ang=1.2,
        rho=0.0,
    )
    assert np.isfinite(magnifications[0])
