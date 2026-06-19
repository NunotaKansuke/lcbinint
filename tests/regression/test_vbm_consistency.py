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
    return vbbinary_lensing.BinaryMagDark(
        separation, mass_ratio, y1, y2, rho, limb_darkening_c
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
    options = lcbinint.Options(
        center_of_mass=1,
        tolerance=1.0e-3,
        relative_tolerance=1.0e-3,
        source_bins=80,
    )
    actual = lcbinint.LensModel(params, options).magnification(y1)

    assert math.isfinite(actual)
    assert math.isclose(actual, reference, rel_tol=1.0e-2, abs_tol=1.0e-2)


def test_lcbinint_lens_model_binary_finite_source_polar_matches_vbm():
    separation = 0.7
    mass_ratio = 0.3
    y1 = -0.6
    y2 = 0.4
    rho = 0.03
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
    options = lcbinint.Options(
        center_of_mass=1,
        inverse_ray_method=lcbinint.InverseRayMethod.POLAR,
        tolerance=1.0e-3,
        relative_tolerance=1.0e-2,
        source_bins=80,
    )
    actual = lcbinint.LensModel(params, options).magnification(y1)

    assert math.isfinite(actual)
    assert math.isclose(actual, reference, rel_tol=1.0e-2, abs_tol=1.0e-2)


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
    options = lcbinint.Options(
        center_of_mass=1,
        tolerance=1.0e-3,
        relative_tolerance=1.0e-3,
        source_bins=80,
    )
    actual = lcbinint.LensModel(params, options).magnification(y1)

    assert math.isfinite(actual)
    assert math.isclose(actual, reference, rel_tol=1.0e-2, abs_tol=1.0e-2)


def test_lcbinint_lens_model_reports_finite_source_non_convergence():
    lcbinint = pytest.importorskip("lcbinint")
    params = lcbinint.LensParams(
        t0=0.0,
        tE=1.0,
        umin=0.0,
        theta=0.0,
        q=0.1,
        sep=1.0,
        rho=0.02,
    )
    options = lcbinint.Options(
        center_of_mass=1,
        tolerance=1.0e-14,
        relative_tolerance=0.0,
        source_bins=4,
    )

    with pytest.raises(RuntimeError, match="numerical error"):
        lcbinint.LensModel(params, options).magnification(0.45)


def test_lcbinint_lens_model_relative_tolerance_can_accept_refinement():
    lcbinint = pytest.importorskip("lcbinint")
    params = lcbinint.LensParams(
        t0=0.0,
        tE=1.0,
        umin=0.0,
        theta=0.0,
        q=0.1,
        sep=1.0,
        rho=0.02,
    )
    options = lcbinint.Options(
        center_of_mass=1,
        tolerance=1.0e-14,
        relative_tolerance=1.0,
        source_bins=4,
    )

    actual = lcbinint.LensModel(params, options).magnification(0.45)

    assert math.isfinite(actual)


def test_lcbinint_options_exposes_inverse_ray_method():
    lcbinint = pytest.importorskip("lcbinint")

    options = lcbinint.Options(inverse_ray_method=lcbinint.InverseRayMethod.POLAR)

    assert options.inverse_ray_method == lcbinint.InverseRayMethod.POLAR


def test_lcbinint_options_exposes_legacy_finite_source_mode():
    lcbinint = pytest.importorskip("lcbinint")

    options = lcbinint.Options(
        finite_source_mode=lcbinint.FiniteSourceMode.LEGACY,
        caustic_bins=128,
        source_bins=12,
        legacy_finite_mode=5,
        legacy_kinji=8.0,
        legacy_hex=2.5,
    )

    assert options.finite_source_mode == lcbinint.FiniteSourceMode.LEGACY
    assert options.caustic_bins == 128
    assert options.source_bins == 12
    assert options.legacy_finite_mode == 5
    assert options.legacy_kinji == 8.0
    assert options.legacy_hex == 2.5


def test_lcbinint_legacy_mode_smoke():
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
    options = lcbinint.Options(
        finite_source_mode=lcbinint.FiniteSourceMode.LEGACY,
        center_of_mass=1,
        caustic_bins=128,
        source_bins=20,
        relative_tolerance=1.0,
    )

    actual = lcbinint.LensModel(params, options).magnification(0.2)

    assert math.isfinite(actual)


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
