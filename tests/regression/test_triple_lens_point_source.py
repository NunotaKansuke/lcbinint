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
    # amp_point3 path. With alpha=0, time=source_x and u0=source_y produce the
    # requested source coordinate in the static VBM trajectory convention.
    light_curve = lcbinint.LightCurve(lens="triple_lens")
    actual = light_curve.magnification(
        source_x,
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

    assert actual == pytest.approx(reference, rel=tolerance)


def test_triple_lens_light_curve_info_smoke():
    light_curve = lcbinint.LightCurve(lens="triple_lens")
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


def test_triple_lens_finite_source_cartesian_inverse_ray():
    light_curve = lcbinint.LightCurve(lens="triple_lens")
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


def test_triple_lens_finite_source_uses_hexadecapole_between_point_and_ir():
    light_curve = lcbinint.LightCurve(
        lens="triple_lens",
        options=lcbinint.Options(source_bins=10, caustic_bins=96),
    )
    times = np.array([-0.2])
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
        "rho": 5.0e-3,
    }

    info = light_curve.info(times, params)

    assert np.isfinite(info.magnifications[0])
    assert info.finite_source_method_names == ["hexadecapole"]
    assert info.finite_source_error_estimates[0] < 1.0e-3


def test_triple_lens_finite_source_approaches_point_source_for_small_rho():
    light_curve = lcbinint.LightCurve(
        lens="triple_lens",
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

    point = light_curve.magnification(0.35, {**params, "rho": 0.0})
    finite = light_curve.magnification(0.35, {**params, "rho": 1.0e-5})

    assert finite == pytest.approx(point, rel=2.0e-3)


def test_triple_lens_finite_source_auto_mode_uses_polar_for_high_magnification_small_source():
    # High point-source magnification + small source (rho=0.0005) means the
    # source is off-caustic relative to the source size, so the auto mode
    # (finite_mode=4) should pick polar inverse-ray for speed.
    light_curve = lcbinint.LightCurve(
        lens="triple_lens",
        options=lcbinint.Options(inverse_ray_grid="auto"),
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


def test_triple_lens_rejects_unsupported_combinations():
    light_curve = lcbinint.LightCurve(lens="triple_lens")
    times = np.array([0.0])
    params = {
        "s": 1.0,
        "q": 1.0e-3,
        "q2": 1.0e-4,
        "sep2": 0.5,
        "ang": 1.2,
        "rho": 0.0,
    }

    with pytest.raises(ValueError, match="q2 > 0"):
        light_curve(times, {**params, "q2": 0.0})

    with pytest.raises(ValueError, match="parameter dictionary"):
        light_curve.light_curve(times, s=1.0, q=1.0e-3, rho=0.0)
