import math
import sys
from pathlib import Path

import pytest


BINARY_POINT_CASES = [
    pytest.param(1.0, 0.1, 0.2, 0.1, 5.871444912771214, id="resonant_low_q"),
    pytest.param(0.7, 0.3, -0.4, 0.2, 2.116643550532278, id="close_binary"),
    pytest.param(1.5, 1.0, 0.05, -0.2, 1.5493462433112466, id="wide_equal_mass"),
    pytest.param(1.0, 1.0e-3, 0.3, 0.4, 2.1789388609029046, id="planetary"),
]

BINARY_FINITE_CASES = [
    pytest.param(1.0, 0.1, 0.4, 0.3, 0.02, id="low_q_small_source"),
    pytest.param(0.7, 0.3, -0.6, 0.4, 0.03, id="close_binary_small_source"),
    pytest.param(1.5, 1.0, 0.4, -0.3, 0.02, id="wide_equal_mass_small_source"),
]

BINARY_LIMB_DARKENING_CASES = [
    pytest.param(1.0, 0.1, 0.4, 0.3, 0.02, id="low_q_small_source"),
    pytest.param(0.7, 0.3, -0.6, 0.4, 0.03, id="close_binary_small_source"),
    pytest.param(1.4, 0.4, 0.2, -0.15, 0.025, id="wide_hard_small_source"),
]


def _vbm_binary_mag0(separation, mass_ratio, y1, y2):
    module = pytest.importorskip("VBBinaryLensing")
    vbbinary_lensing = module.VBBinaryLensing()
    return vbbinary_lensing.BinaryMag0(separation, mass_ratio, y1, y2)


def _vbm_binary_mag2(separation, mass_ratio, y1, y2, rho):
    module = pytest.importorskip("VBBinaryLensing")
    vbbinary_lensing = module.VBBinaryLensing()
    return vbbinary_lensing.BinaryMag2(separation, mass_ratio, y1, y2, rho)


def _vbm_binary_mag_dark(separation, mass_ratio, y1, y2, rho, limb_darkening_c):
    module = pytest.importorskip("VBBinaryLensing")
    vbbinary_lensing = module.VBBinaryLensing()
    vbbinary_lensing.Tol = 1.0e-3
    vbbinary_lensing.a1 = limb_darkening_c
    return vbbinary_lensing.BinaryMagDark(
        separation, mass_ratio, y1, y2, rho, vbbinary_lensing.Tol
    )


def _lcbinint_binary_mag0(separation, mass_ratio, y1, y2):
    lcbinint = pytest.importorskip("lcbinint")

    if hasattr(lcbinint, "binary_mag0"):
        return lcbinint.binary_mag0(separation, mass_ratio, y1, y2)

    raise NotImplementedError(
        "lcbinint binary point-source Python API is not implemented yet"
    )


def _lcbinint_lens_model_mag0(separation, mass_ratio, y1, y2):
    lcbinint = pytest.importorskip("lcbinint")

    params = lcbinint.LensParams(
        t0=0.0,
        tE=1.0,
        umin=y2,
        theta=0.0,
        q=mass_ratio,
        sep=separation,
    )
    options = lcbinint.Options(center_of_mass=1)

    return lcbinint.LensModel(params, options).magnification(y1)


@pytest.mark.parametrize("separation,mass_ratio,y1,y2,expected", BINARY_POINT_CASES)
def test_vbm_binary_reference_values_are_stable(
    separation, mass_ratio, y1, y2, expected
):
    actual = _vbm_binary_mag0(separation, mass_ratio, y1, y2)

    assert math.isfinite(actual)
    assert math.isclose(actual, expected, rel_tol=5.0e-13, abs_tol=5.0e-13)


@pytest.mark.parametrize("separation,mass_ratio,y1,y2,expected", BINARY_POINT_CASES)
def test_lcbinint_binary_point_source_matches_vbm(
    separation, mass_ratio, y1, y2, expected
):
    del expected

    reference = _vbm_binary_mag0(separation, mass_ratio, y1, y2)
    actual = _lcbinint_binary_mag0(separation, mass_ratio, y1, y2)

    assert math.isfinite(actual)
    assert math.isclose(actual, reference, rel_tol=1.0e-10, abs_tol=1.0e-11)


@pytest.mark.parametrize("separation,mass_ratio,y1,y2,expected", BINARY_POINT_CASES)
def test_lcbinint_lens_model_binary_point_source_matches_vbm(
    separation, mass_ratio, y1, y2, expected
):
    del expected

    reference = _vbm_binary_mag0(separation, mass_ratio, y1, y2)
    actual = _lcbinint_lens_model_mag0(separation, mass_ratio, y1, y2)

    assert math.isfinite(actual)
    assert math.isclose(actual, reference, rel_tol=1.0e-10, abs_tol=1.0e-11)


def test_lcbinint_lens_model_wide_binary_legacy_offset_matches_vbm():
    separation = 1.5
    mass_ratio = 1.0
    y1 = 0.2
    y2 = 0.1
    m2 = mass_ratio / (1.0 + mass_ratio)
    legacy_offset = m2 * separation - m2 / separation

    reference = _vbm_binary_mag0(separation, mass_ratio, y1 - legacy_offset, y2)

    lcbinint = pytest.importorskip("lcbinint")
    params = lcbinint.LensParams(
        t0=0.0,
        tE=1.0,
        umin=y2,
        theta=0.0,
        q=mass_ratio,
        sep=separation,
    )
    options = lcbinint.Options(center_of_mass=0)
    actual = lcbinint.LensModel(params, options).magnification(y1)

    assert math.isfinite(actual)
    assert math.isclose(actual, reference, rel_tol=1.0e-10, abs_tol=1.0e-11)


@pytest.mark.parametrize("separation,mass_ratio,y1,y2,rho", BINARY_FINITE_CASES)
def test_lcbinint_lens_model_binary_finite_source_matches_vbm(
    separation, mass_ratio, y1, y2, rho
):
    reference = _vbm_binary_mag2(separation, mass_ratio, y1, y2, rho)

    lcbinint = pytest.importorskip("lcbinint")
    params = lcbinint.LensParams(
        t0=0.0,
        tE=1.0,
        umin=y2,
        theta=0.0,
        q=mass_ratio,
        sep=separation,
        rho=rho,
    )
    options = lcbinint.Options(center_of_mass=1, source_bins=80)
    actual = lcbinint.LensModel(params, options).magnification(y1)

    assert math.isfinite(actual)
    assert math.isclose(actual, reference, rel_tol=1.0e-2, abs_tol=1.0e-2)


def test_lcbinint_lens_model_polar_finite_source_matches_vbm():
    separation = 1.4
    mass_ratio = 0.4
    y1 = 0.2
    y2 = -0.15
    rho = 0.025
    reference = _vbm_binary_mag2(separation, mass_ratio, y1, y2, rho)

    lcbinint = pytest.importorskip("lcbinint")
    params = lcbinint.LensParams(
        t0=0.0,
        tE=1.0,
        umin=y2,
        theta=0.0,
        q=mass_ratio,
        sep=separation,
        rho=rho,
    )
    options = lcbinint.Options(center_of_mass=1, mode=2, source_bins=80)
    actual = lcbinint.LensModel(params, options).magnification(y1)

    assert math.isfinite(actual)
    assert math.isclose(actual, reference, rel_tol=5.0e-3, abs_tol=5.0e-3)


@pytest.mark.parametrize("separation,mass_ratio,y1,y2,rho", BINARY_FINITE_CASES)
def test_lcbinint_lens_model_linear_limb_darkening_matches_vbm(
    separation, mass_ratio, y1, y2, rho
):
    limb_darkening_c = 0.5
    reference = _vbm_binary_mag_dark(
        separation, mass_ratio, y1, y2, rho, limb_darkening_c
    )

    lcbinint = pytest.importorskip("lcbinint")
    params = lcbinint.LensParams(
        t0=0.0,
        tE=1.0,
        umin=y2,
        theta=0.0,
        q=mass_ratio,
        sep=separation,
        rho=rho,
        limb_darkening_c=limb_darkening_c,
    )
    options = lcbinint.Options(center_of_mass=1, source_bins=80)
    actual = lcbinint.LensModel(params, options).magnification(y1)

    assert math.isfinite(actual)
    assert math.isclose(actual, reference, rel_tol=1.0e-2, abs_tol=1.0e-2)


@pytest.mark.parametrize("separation,mass_ratio,y1,y2,rho", BINARY_LIMB_DARKENING_CASES)
@pytest.mark.parametrize("mode", [1, 2])
def test_lcbinint_lens_model_limb_darkening_matches_vbm(
    separation, mass_ratio, y1, y2, rho, mode
):
    limb_darkening_c = 0.5
    reference = _vbm_binary_mag_dark(
        separation, mass_ratio, y1, y2, rho, limb_darkening_c
    )

    lcbinint = pytest.importorskip("lcbinint")
    params = lcbinint.LensParams(
        t0=0.0,
        tE=1.0,
        umin=y2,
        theta=0.0,
        q=mass_ratio,
        sep=separation,
        rho=rho,
        limb_darkening_c=limb_darkening_c,
    )
    options = lcbinint.Options(center_of_mass=1, mode=mode, source_bins=80)
    actual = lcbinint.LensModel(params, options).magnification(y1)

    assert math.isfinite(actual)
    assert math.isclose(actual, reference, rel_tol=5.0e-3, abs_tol=5.0e-3)


@pytest.mark.parametrize("mode", [1, 2])
@pytest.mark.parametrize("limb_darkened", [False, True])
def test_lcbinint_caustic_light_curve_points_match_vbm(
    mode, limb_darkened
):
    lcbinint = pytest.importorskip("lcbinint")
    module = pytest.importorskip("VBBinaryLensing")

    separation = 1.4
    mass_ratio = 0.4
    y2 = -0.15
    rho = 0.025
    limb_darkening_c = 0.5
    y1_values = [-0.25854879065888237, -0.2485404503753128, 0.5821517931609672, 0.6021684737281067]

    vbbinary_lensing = module.VBBinaryLensing()
    vbbinary_lensing.Tol = 1.0e-3
    if limb_darkened:
        vbbinary_lensing.a1 = limb_darkening_c
    reference = []
    for y1 in y1_values:
        if limb_darkened:
            reference.append(
                vbbinary_lensing.BinaryMagDark(
                    separation, mass_ratio, y1, y2, rho, vbbinary_lensing.Tol
                )
            )
        else:
            reference.append(
                vbbinary_lensing.BinaryMag2(separation, mass_ratio, y1, y2, rho)
            )

    params = lcbinint.LensParams(
        t0=0.0,
        tE=1.0,
        umin=y2,
        theta=0.0,
        q=mass_ratio,
        sep=separation,
        rho=rho,
        limb_darkening_c=limb_darkening_c if limb_darkened else 0.0,
    )
    options = lcbinint.Options(center_of_mass=1, mode=mode, source_bins=80)
    actual = lcbinint.LensModel(params, options).light_curve(y1_values).magnifications

    for actual_value, reference_value in zip(actual, reference):
        assert math.isfinite(actual_value)
        assert math.isclose(actual_value, reference_value, rel_tol=1.5e-3, abs_tol=1.5e-3)


@pytest.mark.parametrize("mode", [1, 2])
def test_lcbinint_high_magnification_light_curve_matches_vbm(
    mode,
):
    lcbinint = pytest.importorskip("lcbinint")
    module = pytest.importorskip("VBBinaryLensing")

    separation = 1.0
    mass_ratio = 0.1
    umin = 0.01
    rho = 0.003
    times = [-0.08, -0.06, -0.04, -0.03, -0.02, -0.01, 0.0, 0.02, 0.04]

    vbbinary_lensing = module.VBBinaryLensing()
    vbbinary_lensing.Tol = 1.0e-3
    reference = vbbinary_lensing.BinaryLightCurve(
        [math.log(separation), math.log(mass_ratio), umin, math.pi, math.log(rho), 0.0, 0.0],
        times,
    )[0]

    params = lcbinint.LensParams(
        t0=0.0,
        tE=1.0,
        umin=umin,
        theta=0.0,
        q=mass_ratio,
        sep=separation,
        rho=rho,
    )
    options = lcbinint.Options(
        center_of_mass=1,
        mode=mode,
        point_source_threshold=1.0e9,
        hexadecapole_threshold=1.0e9,
        source_bins=80,
    )
    actual = lcbinint.LensModel(params, options).light_curve(times).magnifications

    for actual_value, reference_value in zip(actual, reference):
        assert math.isfinite(actual_value)
        assert math.isclose(actual_value, reference_value, rel_tol=1.5e-3, abs_tol=1.5e-3)


@pytest.mark.parametrize("alpha,u0,separation,mass_ratio", [
    pytest.param(0.5, -0.01, 1.0, 0.001, id="nonzero_alpha_planetary"),
    pytest.param(0.0, 0.1, 1.0, 0.1, id="zero_alpha_low_q"),
    pytest.param(1.2, -0.05, 0.7, 0.3, id="nonzero_alpha_close_binary"),
    pytest.param(0.5, -0.01, 1.5, 0.001, id="nonzero_alpha_wide_planetary"),
])
def test_lcbinint_vbbl_compatible_mode_matches_binary_light_curve(
    alpha, u0, separation, mass_ratio
):
    lcbinint = pytest.importorskip("lcbinint")
    module = pytest.importorskip("VBBinaryLensing")

    rho = 1e-4
    times = [-0.4, -0.2, 0.0, 0.2, 0.4]

    vbb = module.VBBinaryLensing()
    vbb.Tol = 1.0e-3
    reference = vbb.BinaryLightCurve(
        [math.log(separation), math.log(mass_ratio), u0, alpha, math.log(rho), 0.0, 0.0],
        times,
    )[0]

    params = lcbinint.LensParams(
        t0=0.0, tE=1.0,
        u0=u0,
        alpha=alpha,
        q=mass_ratio,
        sep=separation,
        rho=rho,
    )
    options = lcbinint.Options(center_of_mass=1, source_bins=80, vbbl_compatible=1)
    actual = lcbinint.LensModel(params, options).light_curve(times).magnifications

    for actual_value, reference_value in zip(actual, reference):
        assert math.isfinite(actual_value)
        assert math.isclose(actual_value, reference_value, rel_tol=1.5e-3, abs_tol=1.5e-3)


@pytest.mark.parametrize("mode", [1, 2, 3])
def test_lcbinint_vbbl_compatible_large_source_planetary_caustic_crossing(mode):
    lcbinint = pytest.importorskip("lcbinint")
    module = pytest.importorskip("VBBinaryLensing")

    separation = 1.0
    mass_ratio = 0.001
    u0 = -0.01
    alpha = 0.5
    rho = 0.01
    times = [0.014035087719298178, 0.018045112781954864]

    vbb = module.VBBinaryLensing()
    vbb.Tol = 1.0e-3
    reference = vbb.BinaryLightCurve(
        [math.log(separation), math.log(mass_ratio), u0, alpha, math.log(rho), 0.0, 0.0],
        times,
    )[0]

    params = lcbinint.LensParams(
        t0=0.0,
        tE=1.0,
        u0=u0,
        alpha=alpha,
        q=mass_ratio,
        sep=separation,
        rho=rho,
    )
    options = lcbinint.Options(
        center_of_mass=1,
        source_bins=200,
        vbbl_compatible=1,
        mode=mode,
    )
    actual = lcbinint.LensModel(params, options).light_curve(times).magnifications

    for actual_value, reference_value in zip(actual, reference):
        assert math.isfinite(actual_value)
        assert math.isclose(actual_value, reference_value, rel_tol=2.5e-3, abs_tol=2.5e-3)


def test_lcbinint_spine_mode_falls_back_for_oversampled_planetary_fold_arc():
    lcbinint = pytest.importorskip("lcbinint")
    module = pytest.importorskip("VBBinaryLensing")

    separation = 1.0
    mass_ratio = 0.001
    u0 = -0.01
    alpha = 0.5
    rho = 1.0e-4
    times = [-0.03007518796992481, 0.04611528822055133]

    vbb = module.VBBinaryLensing()
    vbb.Tol = 1.0e-3
    reference = vbb.BinaryLightCurve(
        [math.log(separation), math.log(mass_ratio), u0, alpha, math.log(rho), 0.0, 0.0],
        times,
    )[0]

    params = lcbinint.LensParams(
        t0=0.0,
        tE=1.0,
        u0=u0,
        alpha=alpha,
        q=mass_ratio,
        sep=separation,
        rho=rho,
    )
    options = lcbinint.Options(
        center_of_mass=1,
        source_bins=200,
        vbbl_compatible=1,
        mode=3,
    )
    actual = lcbinint.LensModel(params, options).light_curve(times).magnifications

    for actual_value, reference_value in zip(actual, reference):
        assert math.isfinite(actual_value)
        assert math.isclose(actual_value, reference_value, rel_tol=5.0e-5, abs_tol=5.0e-5)


def test_lcbinint_adaptive_source_bins_refines_cartesian_grid_from_diagnostics():
    lcbinint = pytest.importorskip("lcbinint")
    module = pytest.importorskip("VBBinaryLensing")

    separation = 1.0
    mass_ratio = 0.001
    u0 = -0.01
    alpha = 0.5
    rho = 1.0e-4
    times = [-0.03007518796992481, 0.04611528822055133]

    vbb = module.VBBinaryLensing()
    vbb.Tol = 1.0e-3
    reference = vbb.BinaryLightCurve(
        [math.log(separation), math.log(mass_ratio), u0, alpha, math.log(rho), 0.0, 0.0],
        times,
    )[0]

    params = lcbinint.LensParams(
        t0=0.0,
        tE=1.0,
        u0=u0,
        alpha=alpha,
        q=mass_ratio,
        sep=separation,
        rho=rho,
    )
    fixed_options = lcbinint.Options(
        center_of_mass=1,
        source_bins=50,
        vbbl_compatible=1,
    )
    adaptive_options = lcbinint.Options(
        center_of_mass=1,
        source_bins=50,
        adaptive_source_bins=1,
        max_source_bins=200,
        vbbl_compatible=1,
        reltol=1.0e-4,
    )

    fixed = lcbinint.LensModel(params, fixed_options).light_curve(times).magnifications
    adaptive = lcbinint.LensModel(params, adaptive_options).light_curve(times)
    fixed_rel = max(abs(a / b - 1.0) for a, b in zip(fixed, reference))
    adaptive_rel = max(abs(a / b - 1.0) for a, b in zip(adaptive.magnifications, reference))

    assert max(adaptive.finite_source_refinement_levels) > 0
    assert max(adaptive.finite_source_error_estimates) > 0.0
    assert adaptive_rel < fixed_rel
    assert adaptive_rel < 1.0e-4


def test_lcbinint_cartesian_ir_seeds_grazing_caustic_limb_images():
    lcbinint = pytest.importorskip("lcbinint")
    module = pytest.importorskip("VBBinaryLensing")

    separation = 0.8
    mass_ratio = 0.01
    u0 = -0.01
    alpha = 0.3
    rho = 5.0e-3
    time = 0.006015037593984918

    vbb = module.VBBinaryLensing()
    vbb.Tol = 1.0e-6
    reference = vbb.BinaryLightCurve(
        [math.log(separation), math.log(mass_ratio), u0, alpha, math.log(rho), 0.0, 0.0],
        [time],
    )[0][0]

    params = lcbinint.LensParams(
        t0=0.0,
        tE=1.0,
        u0=u0,
        alpha=alpha,
        q=mass_ratio,
        sep=separation,
        rho=rho,
    )
    fixed = lcbinint.LensModel(
        params,
        lcbinint.Options(source_bins=50, vbbl_compatible=1, adaptive_source_bins=0),
    ).magnification(time)
    adaptive = lcbinint.LensModel(
        params,
        lcbinint.Options(source_bins=50, vbbl_compatible=1, reltol=1.0e-3, max_source_bins=400),
    ).magnification(time)

    assert math.isfinite(fixed)
    assert math.isfinite(adaptive)
    assert abs(fixed / reference - 1.0) < 3.0e-3
    assert abs(adaptive / reference - 1.0) < 1.0e-3


def test_lcbinint_cartesian_ir_keeps_same_parity_fold_branch_seed():
    lcbinint = pytest.importorskip("lcbinint")
    module = pytest.importorskip("VBBinaryLensing")

    separation = 0.95
    mass_ratio = 0.01
    u0 = -0.001
    alpha = 0.5
    rho = 5.0e-3
    time = 0.006015037593984918

    vbb = module.VBBinaryLensing()
    vbb.Tol = 1.0e-3
    reference = vbb.BinaryLightCurve(
        [math.log(separation), math.log(mass_ratio), u0, alpha, math.log(rho), 0.0, 0.0],
        [time],
    )[0][0]

    params = lcbinint.LensParams(
        t0=0.0,
        tE=1.0,
        u0=u0,
        alpha=alpha,
        q=mass_ratio,
        sep=separation,
        rho=rho,
    )
    actual = lcbinint.LensModel(
        params,
        lcbinint.Options(source_bins=50, vbbl_compatible=1, adaptive_source_bins=0),
    ).magnification(time)

    assert abs(actual / reference - 1.0) < 1.0e-3


def test_lcbinint_local_boundary_estimate_avoids_ld_over_refinement():
    lcbinint = pytest.importorskip("lcbinint")
    module = pytest.importorskip("VBBinaryLensing")

    separation = 0.95
    mass_ratio = 0.01
    u0 = -0.001
    alpha = 0.5
    rho = 5.0e-3
    limb_darkening_c = 0.5
    time = 0.010025062656641603

    vbb = module.VBBinaryLensing()
    vbb.Tol = 1.0e-3
    vbb.a1 = limb_darkening_c
    reference = vbb.BinaryLightCurve(
        [math.log(separation), math.log(mass_ratio), u0, alpha, math.log(rho), 0.0, 0.0],
        [time],
    )[0][0]

    params = lcbinint.LensParams(
        t0=0.0,
        tE=1.0,
        u0=u0,
        alpha=alpha,
        q=mass_ratio,
        sep=separation,
        rho=rho,
        limb_darkening_c=limb_darkening_c,
    )
    curve = lcbinint.LensModel(
        params,
        lcbinint.Options(source_bins=50, max_source_bins=400, reltol=1.0e-3, vbbl_compatible=1),
    ).light_curve([time])

    assert curve.finite_source_refinement_levels[0] <= 2
    assert curve.finite_source_converged[0]
    assert abs(curve.magnifications[0] / reference - 1.0) < 1.0e-3


def test_lcbinint_cartesian_ir_does_not_clip_moderate_fold_image_area():
    lcbinint = pytest.importorskip("lcbinint")
    module = pytest.importorskip("VBBinaryLensing")

    separation = 1.160935533582098
    mass_ratio = 0.003086323166308305
    u0 = -0.001382074492745227
    alpha = 0.7396109342207111
    rho = 0.018442211657959038
    time = 0.037042588756702244

    vbb = module.VBBinaryLensing()
    vbb.Tol = 1.0e-6
    reference = vbb.BinaryLightCurve(
        [math.log(separation), math.log(mass_ratio), u0, alpha, math.log(rho), 0.0, 0.0],
        [time],
    )[0][0]

    params = lcbinint.LensParams(
        t0=0.0,
        tE=1.0,
        u0=u0,
        alpha=alpha,
        q=mass_ratio,
        sep=separation,
        rho=rho,
    )
    options = lcbinint.Options(
        source_bins=50,
        mode=1,
        vbbl_compatible=1,
    )
    actual = lcbinint.LensModel(params, options).light_curve([time]).magnifications[0]

    # Regression for the old |J| < 0.5 fold guard, which clipped a valid image
    # component and converged to a biased value near 32.065 (rel. error ~1.8e-3).
    assert math.isclose(actual, reference, rel_tol=1.0e-4, abs_tol=1.0e-4)


def test_lcbinint_cartesian_ir_does_not_double_subtract_wide_caustic_fold_overlap():
    lcbinint = pytest.importorskip("lcbinint")
    module = pytest.importorskip("VBBinaryLensing")

    separation = 0.95
    mass_ratio = 0.01
    u0 = -0.01
    alpha = 0.5
    rho = 1.0e-2
    time = 0.006015037593984918

    vbb = module.VBBinaryLensing()
    vbb.Tol = 1.0e-5
    reference = vbb.BinaryLightCurve(
        [math.log(separation), math.log(mass_ratio), u0, alpha, math.log(rho), 0.0, 0.0],
        [time],
    )[0][0]

    params = lcbinint.LensParams(
        t0=0.0,
        tE=1.0,
        u0=u0,
        alpha=alpha,
        q=mass_ratio,
        sep=separation,
        rho=rho,
    )
    actual = lcbinint.LensModel(
        params,
        lcbinint.Options(source_bins=50, mode=1, vbbl_compatible=1),
    ).light_curve([time]).magnifications[0]

    # At this grid phase the old overlap bookkeeping processed two equivalent
    # fold components and subtracted the same previous component twice, causing
    # a deterministic ~9.6% underestimate.
    assert math.isclose(actual, reference, rel_tol=1.0e-3, abs_tol=1.0e-3)


@pytest.mark.parametrize(
    ("rho", "time", "relative_tolerance"),
    [
        (3.0e-2, 0.05012531328320802, 1.0e-3),
        (3.0e-1, 0.30275689223057634, 1.0e-3),
    ],
)
def test_lcbinint_adaptive_large_source_seed_refinement_regressions(
    rho,
    time,
    relative_tolerance,
):
    lcbinint = pytest.importorskip("lcbinint")
    module = pytest.importorskip("VBBinaryLensing")

    separation = 1.0
    mass_ratio = 0.001
    u0 = -0.01
    alpha = 0.5

    vbb = module.VBBinaryLensing()
    vbb.Tol = 1.0e-5
    reference = vbb.BinaryLightCurve(
        [math.log(separation), math.log(mass_ratio), u0, alpha, math.log(rho), 0.0, 0.0],
        [time],
    )[0][0]

    params = lcbinint.LensParams(
        t0=0.0,
        tE=1.0,
        u0=u0,
        alpha=alpha,
        q=mass_ratio,
        sep=separation,
        rho=rho,
    )
    options = lcbinint.Options(
        center_of_mass=1,
        source_bins=50,
        max_source_bins=400,
        reltol=1.0e-3,
        vbbl_compatible=1,
    )
    curve = lcbinint.LensModel(params, options).light_curve([time])
    actual = curve.magnifications[0]

    assert curve.all_converged
    assert math.isclose(actual, reference, rel_tol=relative_tolerance, abs_tol=relative_tolerance)


def test_lcbinint_options_exposes_fields():
    lcbinint = pytest.importorskip("lcbinint")

    options = lcbinint.Options(
        caustic_bins=128,
        source_bins=40,
        mode=2,
        point_source_threshold=8.0,
        hexadecapole_threshold=2.5,
        adaptive_source_bins=1,
        max_source_bins=160,
        tol=1.0e-4,
        reltol=2.0e-4,
    )

    assert options.caustic_bins == 128
    assert options.source_bins == 40
    assert options.mode == 2
    assert options.point_source_threshold == 8.0
    assert options.hexadecapole_threshold == 2.5
    assert options.adaptive_source_bins == 1
    assert options.max_source_bins == 160
    assert options.finite_source_tol == 1.0e-4
    assert options.finite_source_reltol == 2.0e-4
    assert options.tol == 1.0e-4
    assert options.reltol == 2.0e-4

    auto_options = lcbinint.Options(reltol=1.0e-3)
    assert auto_options.adaptive_source_bins == 1
    assert auto_options.finite_source_reltol == 1.0e-3

    fixed_options = lcbinint.Options(reltol=1.0e-3, adaptive_source_bins=0)
    assert fixed_options.adaptive_source_bins == 0
    assert fixed_options.finite_source_reltol == 1.0e-3


def test_lcbinint_finite_source_smoke():
    lcbinint = pytest.importorskip("lcbinint")
    params = lcbinint.LensParams(
        t0=0.0,
        tE=1.0,
        umin=0.2,
        theta=0.0,
        q=0.1,
        sep=1.0,
        rho=0.02,
    )
    options = lcbinint.Options(center_of_mass=1, caustic_bins=128, source_bins=20)

    actual = lcbinint.LensModel(params, options).magnification(0.2)

    assert math.isfinite(actual)


def test_lcbinint_spine_mode_wide_caustic_fold_pair():
    """mode=3 spine on a caustic-born fold pair should match mode=1 within 2e-4."""
    lcbinint = pytest.importorskip("lcbinint")
    separation, mass_ratio = 1.4, 0.4
    y1, y2 = -0.24854045037531268, -0.15
    rho = 1e-4
    source_bins = 300
    params = lcbinint.LensParams(
        t0=0.0, tE=1.0, umin=y2, theta=0.0, q=mass_ratio, sep=separation, rho=rho
    )
    mag1 = lcbinint.LensModel(params, lcbinint.Options(center_of_mass=1, mode=1, source_bins=source_bins)).magnification(y1)
    mag3 = lcbinint.LensModel(params, lcbinint.Options(center_of_mass=1, mode=3, source_bins=source_bins)).magnification(y1)
    assert math.isfinite(mag1)
    assert math.isfinite(mag3)
    assert math.isclose(mag3, mag1, rel_tol=2e-4, abs_tol=2e-4)


def test_lcbinint_spine_mode_non_caustic_guard():
    """mode=3 should fall back to cartesian for a non-caustic high-mag source."""
    lcbinint = pytest.importorskip("lcbinint")
    separation, mass_ratio = 0.6, 1.0
    y1, y2 = -0.09201927708355606, 0.029966615534330332
    rho = 0.003
    source_bins = 60
    params = lcbinint.LensParams(
        t0=0.0, tE=1.0, umin=y2, theta=0.0, q=mass_ratio, sep=separation, rho=rho
    )
    mag1 = lcbinint.LensModel(params, lcbinint.Options(center_of_mass=1, mode=1, source_bins=source_bins)).magnification(y1)
    mag3 = lcbinint.LensModel(params, lcbinint.Options(center_of_mass=1, mode=3, source_bins=source_bins)).magnification(y1)
    assert math.isfinite(mag1)
    assert math.isfinite(mag3)
    assert math.isclose(mag3, mag1, rel_tol=1e-10, abs_tol=1e-10)


def test_lcbinint_lens_params_exposes_limb_darkening_coefficients():
    lcbinint = pytest.importorskip("lcbinint")

    params = lcbinint.LensParams(limb_darkening_c=0.5, limb_darkening_d=0.2)

    assert params.limb_darkening_c == 0.5
    assert params.limb_darkening_d == 0.2


def test_lcbinint_lens_model_light_curve_exposes_source_trajectory():
    lcbinint = pytest.importorskip("lcbinint")
    params = lcbinint.LensParams(
        t0=10.0,
        tE=2.0,
        umin=0.1,
        theta=0.0,
        q=0.1,
        sep=1.0,
    )
    options = lcbinint.Options(center_of_mass=1)
    model = lcbinint.LensModel(params, options)
    times = [8.0, 10.0, 12.0]

    curve = model.light_curve(times)

    assert curve.times == times
    assert curve.source_x == [-1.0, 0.0, 1.0]
    assert curve.source_y == [0.1, 0.1, 0.1]
    assert curve.magnifications == model.magnifications(times)
    assert model.source_position(10.0) == (0.0, 0.1)
    assert model.source_positions(times) == [(-1.0, 0.1), (0.0, 0.1), (1.0, 0.1)]
    assert len(curve.point_source_magnifications) == len(times)
    assert len(curve.finite_source_magnifications) == len(times)
    assert len(curve.image_counts) == len(times)


def test_lcbinint_lens_model_light_curve_accepts_empty_times():
    lcbinint = pytest.importorskip("lcbinint")
    model = lcbinint.LensModel(lcbinint.LensParams(), lcbinint.Options())

    curve = model.light_curve([])

    assert curve.times == []
    assert curve.magnifications == []
    assert model.magnifications([]) == []
    assert model.source_positions([]) == []


def test_lcbinint_lens_model_estimates_source_bins_from_self_convergence():
    lcbinint = pytest.importorskip("lcbinint")

    params = lcbinint.LensParams(
        t0=0.0,
        tE=1.0,
        umin=-0.15,
        theta=0.0,
        q=0.4,
        sep=1.4,
        rho=0.025,
    )
    options = lcbinint.Options(center_of_mass=1, source_bins=80)
    model = lcbinint.LensModel(params, options)
    times = [-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0]

    estimate = model.estimate_source_bins(
        times,
        candidate_bins=[20, 40, 60, 80],
        max_sample_points=9,
    )

    assert estimate.reference_source_bins == 80
    assert estimate.sampled_times == times
    assert [candidate.source_bins for candidate in estimate.candidates] == [20, 40, 60, 80]
    assert estimate.candidates[-1].accepted
    assert estimate.candidates[-1].max_relative_difference == pytest.approx(0.0)


def test_lcbinint_annual_parallax_source_trajectory_matches_jacscanomaly():
    jacscanomaly_src = Path("/rogue1_8/nunota/jacscanomaly/src")
    if not jacscanomaly_src.exists():
        pytest.skip("local jacscanomaly checkout is not available")
    sys.path.insert(0, str(jacscanomaly_src))
    jnp = pytest.importorskip("jax.numpy")
    trajectory = pytest.importorskip("jacscanomaly.trajectory")

    lcbinint = pytest.importorskip("lcbinint")
    ra_deg = 267.623337808
    dec_deg = -29.1164180355
    tref = 2459000.0
    t0 = 2459001.0
    tE = 80.0
    u0 = 0.12
    piN = 0.02
    piE = 0.03
    times = [2458990.0, 2459000.0, 2459010.0]

    projector = trajectory.make_parallax_projector(ra_deg, dec_deg, tref)
    tau, beta = trajectory.u_parallax_tau_beta(
        jnp.asarray(times), t0, tE, u0, piN, piE, projector
    )

    params = lcbinint.LensParams(
        t0=t0,
        tE=tE,
        umin=u0,
        q=0.1,
        sep=1.0,
        theta=0.0,
        piEN=piN,
        piEE=piE,
        ra=ra_deg,
        dec=dec_deg,
        tfix=tref,
    )
    actual = lcbinint.LensModel(params, lcbinint.Options(center_of_mass=1)).source_positions(times)

    for (x, y), expected_tau, expected_beta in zip(actual, tau, beta):
        assert math.isclose(x, float(expected_tau), rel_tol=1.0e-10, abs_tol=1.0e-10)
        assert math.isclose(y, float(expected_beta), rel_tol=1.0e-10, abs_tol=1.0e-10)


def _microjax_lom():
    microlux_src = Path("/rogue1_8/nunota/microlux/src")
    if not microlux_src.exists():
        pytest.skip("local microlux checkout is not available")
    sys.path.insert(0, str(microlux_src))
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    return pytest.importorskip("microlux._vendor.microjax.trajectory.lom")


def _assert_angle_close(actual, expected, rel_tol=1.0e-11, abs_tol=1.0e-11):
    delta = math.atan2(math.sin(actual - expected), math.cos(actual - expected))
    assert math.isclose(delta, 0.0, rel_tol=rel_tol, abs_tol=abs_tol)


@pytest.mark.parametrize("time", [2450000.0, 2450017.25, 2450060.5])
def test_lcbinint_circular_orbital_motion_matches_microjax_vbm_formula(time):
    lom = _microjax_lom()
    lcbinint = pytest.importorskip("lcbinint")
    args = dict(
        separation=1.25,
        angle=0.4,
        g1=0.006,
        g2=-0.004,
        g3=0.009,
        reference_time=2450000.0,
    )

    actual_s, actual_alpha, actual_sz = lcbinint.circular_orbital_motion(time, **args)
    expected_s, expected_alpha, expected_sz = lom.circular_orbital_motion_3d(
        time,
        s0=args["separation"],
        alpha0=args["angle"],
        w1=args["g1"],
        w2=args["g2"],
        w3=args["g3"],
        tref=args["reference_time"],
    )

    assert math.isclose(actual_s, float(expected_s), rel_tol=1.0e-11, abs_tol=1.0e-11)
    _assert_angle_close(actual_alpha, float(expected_alpha))
    assert math.isclose(actual_sz, float(expected_sz), rel_tol=1.0e-11, abs_tol=1.0e-11)


@pytest.mark.parametrize("time", [2450000.0, 2450013.0, 2450041.5])
def test_lcbinint_kepler_orbital_motion_matches_microjax_vbm_formula(time):
    lom = _microjax_lom()
    lcbinint = pytest.importorskip("lcbinint")
    args = dict(
        separation=0.92,
        angle=-0.3,
        g1=0.004,
        g2=0.011,
        g3=0.006,
        szs=0.35,
        ar=1.4,
        reference_time=2450000.0,
    )

    actual_s, actual_alpha, actual_sz = lcbinint.kepler_orbital_motion(time, **args)
    expected_s, expected_alpha, expected_sz = lom.elliptic_orbital_motion_3d(
        time,
        s0=args["separation"],
        alpha0=args["angle"],
        w1=args["g1"],
        w2=args["g2"],
        w3=args["g3"],
        szs=args["szs"],
        ar=args["ar"],
        tref=args["reference_time"],
    )

    assert math.isclose(actual_s, float(expected_s), rel_tol=1.0e-11, abs_tol=1.0e-11)
    _assert_angle_close(actual_alpha, float(expected_alpha))
    assert math.isclose(actual_sz, float(expected_sz), rel_tol=1.0e-11, abs_tol=1.0e-11)


def test_lcbinint_lens_model_circular_orbital_motion_uses_instantaneous_state():
    lcbinint = pytest.importorskip("lcbinint")
    time = 2.0
    params = lcbinint.LensParams(
        t0=1.0,
        tE=1.0,
        umin=0.15,
        q=0.2,
        sep=1.1,
        theta=0.0,
        tfix=0.5,
        orbital_motion_mode=lcbinint.OrbitalMotionMode.CIRCULAR,
        g2=0.08,
    )
    options = lcbinint.Options(center_of_mass=1)

    separation, alpha, _ = lcbinint.circular_orbital_motion(
        time, params.sep, params.theta, params.g1, params.g2, params.g3, params.tfix
    )
    tau = (time - params.t0) / params.tE
    x = tau * math.cos(alpha) + params.umin * math.sin(alpha)
    y = params.umin * math.cos(alpha) - tau * math.sin(alpha)
    expected = lcbinint.binary_mag0(separation, params.q, x, y)
    actual = lcbinint.LensModel(params, options).magnification(time)

    assert math.isfinite(actual)
    assert math.isclose(actual, expected, rel_tol=1.0e-11, abs_tol=1.0e-11)
