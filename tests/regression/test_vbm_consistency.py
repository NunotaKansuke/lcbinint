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
    module = pytest.importorskip("VBMicrolensing")
    vbbinary_lensing = module.VBMicrolensing()
    return vbbinary_lensing.BinaryMag0(separation, mass_ratio, y1, y2)


def _vbm_binary_mag2(separation, mass_ratio, y1, y2, rho):
    module = pytest.importorskip("VBMicrolensing")
    vbbinary_lensing = module.VBMicrolensing()
    return vbbinary_lensing.BinaryMag2(separation, mass_ratio, y1, y2, rho)


def _vbm_binary_mag_dark(separation, mass_ratio, y1, y2, rho, limb_darkening_c):
    module = pytest.importorskip("VBMicrolensing")
    vbbinary_lensing = module.VBMicrolensing()
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


def _lcbinint_function_api_mag0(separation, mass_ratio, y1, y2):
    lcbinint = pytest.importorskip("lcbinint")

    params = lcbinint.LensParams(
        t0=0.0,
        tE=1.0,
        umin=y2,
        theta=0.0,
        q=mass_ratio,
        sep=separation,
    )
    options = lcbinint.Options(coordinates="center_of_mass")

    return _model(lcbinint, params, options).magnification(y1)


def _copy_options(lcbinint, options, *, source_bins=None):
    return lcbinint.Options(
        source_bins=options.source_bins if source_bins is None else source_bins,
        mode=options.mode,
        caustic_bins=options.caustic_bins,
        grid_ratio=options.grid_ratio,
        point_source_threshold=options.point_source_threshold,
        hexadecapole_threshold=options.hexadecapole_threshold,
        adaptive_hex_threshold=options.adaptive_hex_threshold,
        adaptive_source_bins=options.adaptive_source_bins,
        max_source_bins=options.max_source_bins,
        tol=options.tol,
        reltol=options.reltol,
        coordinates=options.coordinates,
    )


def _api_kwargs(lcbinint, params, options):
    return dict(
        t0=params.t0,
        tE=params.tE,
        u0=params.umin,
        alpha=params.theta,
        s=params.sep,
        q=params.q,
        rho=params.rho,
        limb_darkening=lcbinint.LimbDarkening(
            c=params.limb_darkening_c,
            d=params.limb_darkening_d,
        ),
        event=lcbinint.EventCoordinates(
            ra=params.ra,
            dec=params.dec,
            tfix=params.tfix,
        ),
        options=options,
        piEN=params.piEN,
        piEE=params.piEE,
        g1=params.g1,
        g2=params.g2,
        g3=params.g3,
        orbital_motion_mode=params.orbital_motion_mode,
        lom_szs=params.lom_szs,
        lom_ar=params.lom_ar,
    )


class _ApiModel:
    def __init__(self, lcbinint, params, options):
        self._lcbinint = lcbinint
        self._params = params
        self._options = options

    def magnification(self, time):
        return self._lcbinint.magnification(
            time,
            **_api_kwargs(self._lcbinint, self._params, self._options),
        )

    def magnifications(self, times):
        return list(self._lcbinint.binary_light_curve(
            times,
            **_api_kwargs(self._lcbinint, self._params, self._options),
        ))

    def light_curve(self, times):
        return self._lcbinint.light_curve_info(
            times,
            **_api_kwargs(self._lcbinint, self._params, self._options),
        )

    def source_position(self, time):
        curve = self.light_curve([time])
        return (curve.source_x[0], curve.source_y[0])

    def source_positions(self, times):
        curve = self.light_curve(times)
        return list(zip(curve.source_x, curve.source_y))

    def estimate_source_bins(self, times, candidate_bins, max_sample_points):
        if max_sample_points <= 0:
            raise ValueError("max_sample_points must be positive")

        if not times:
            sampled_times = []
        elif max_sample_points >= len(times):
            sampled_times = list(times)
        elif max_sample_points == 1:
            sampled_times = [times[len(times) // 2]]
        else:
            sampled_times = [
                times[round(i * (len(times) - 1) / (max_sample_points - 1))]
                for i in range(max_sample_points)
            ]

        candidate_bins = sorted({bins for bins in candidate_bins if bins > 0})
        reference_bins = candidate_bins[-1]
        reference = _ApiModel(
            self._lcbinint,
            self._params,
            _copy_options(self._lcbinint, self._options, source_bins=reference_bins),
        ).magnifications(sampled_times)

        class Candidate:
            pass

        candidates = []
        for bins in candidate_bins:
            values = reference if bins == reference_bins else _ApiModel(
                self._lcbinint,
                self._params,
                _copy_options(self._lcbinint, self._options, source_bins=bins),
            ).magnifications(sampled_times)
            candidate = Candidate()
            candidate.source_bins = bins
            candidate.max_absolute_difference = 0.0
            candidate.max_relative_difference = 0.0
            squared_relative = 0.0
            for value, ref in zip(values, reference):
                diff = abs(value - ref)
                rel = diff / max(abs(ref), 1.0e-300)
                candidate.max_absolute_difference = max(candidate.max_absolute_difference, diff)
                candidate.max_relative_difference = max(candidate.max_relative_difference, rel)
                squared_relative += rel * rel
            candidate.rms_relative_difference = (
                math.sqrt(squared_relative / len(values)) if values else 0.0
            )
            candidate.accepted = bins == reference_bins
            candidates.append(candidate)

        class Estimate:
            pass

        estimate = Estimate()
        estimate.reference_source_bins = reference_bins
        estimate.recommended_source_bins = reference_bins
        estimate.sampled_times = sampled_times
        estimate.candidates = candidates
        return estimate


def _model(lcbinint, params, options):
    return _ApiModel(lcbinint, params, options)


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
def test_lcbinint_function_api_binary_point_source_matches_vbm(
    separation, mass_ratio, y1, y2, expected
):
    del expected

    reference = _vbm_binary_mag0(separation, mass_ratio, y1, y2)
    actual = _lcbinint_function_api_mag0(separation, mass_ratio, y1, y2)

    assert math.isfinite(actual)
    assert math.isclose(actual, reference, rel_tol=1.0e-10, abs_tol=1.0e-11)


def test_lcbinint_function_api_wide_binary_original_offset_matches_vbm():
    separation = 1.5
    mass_ratio = 1.0
    y1 = 0.2
    y2 = 0.1
    m2 = mass_ratio / (1.0 + mass_ratio)
    original_offset = m2 * separation - m2 / separation

    reference = _vbm_binary_mag0(separation, mass_ratio, y1 - original_offset, y2)

    lcbinint = pytest.importorskip("lcbinint")
    params = lcbinint.LensParams(
        t0=0.0,
        tE=1.0,
        umin=y2,
        theta=0.0,
        q=mass_ratio,
        sep=separation,
    )
    options = lcbinint.Options(coordinates="lcbinint")
    actual = _model(lcbinint, params, options).magnification(y1)

    assert math.isfinite(actual)
    assert math.isclose(actual, reference, rel_tol=1.0e-10, abs_tol=1.0e-11)


@pytest.mark.parametrize("separation,mass_ratio,y1,y2,rho", BINARY_FINITE_CASES)
def test_lcbinint_function_api_binary_finite_source_matches_vbm(
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
    options = lcbinint.Options(coordinates="center_of_mass", source_bins=80)
    actual = _model(lcbinint, params, options).magnification(y1)

    assert math.isfinite(actual)
    assert math.isclose(actual, reference, rel_tol=1.0e-2, abs_tol=1.0e-2)


def test_lcbinint_function_api_polar_finite_source_matches_vbm():
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
    options = lcbinint.Options(coordinates="center_of_mass", mode=2, source_bins=80)
    actual = _model(lcbinint, params, options).magnification(y1)

    assert math.isfinite(actual)
    assert math.isclose(actual, reference, rel_tol=5.0e-3, abs_tol=5.0e-3)


def test_lcbinint_polar_high_magnification_curve_matches_vbm_without_cartesian_fallback():
    lcbinint = pytest.importorskip("lcbinint")
    module = pytest.importorskip("VBMicrolensing")
    np = pytest.importorskip("numpy")

    separation = 0.95
    mass_ratio = 1.0e-2
    u0 = -1.0e-3
    alpha = 0.5
    rho = 5.0e-3
    times = np.linspace(-0.04, 0.04, 41)

    vbb = module.VBMicrolensing()
    vbb.Tol = 1.0e-3
    reference = np.asarray(
        vbb.BinaryLightCurve(
            [math.log(separation), math.log(mass_ratio), u0, alpha, math.log(rho), 0.0, 0.0],
            times.tolist(),
        )[0],
        dtype=float,
    )

    func = lcbinint.LightCurve(
        options=lcbinint.Options(
            coordinates="vbm",
            mode=2,
            source_bins=50,
            point_source_threshold=1.0e9,
            hexadecapole_threshold=1.0e9,
        )
    )
    curve = func.info(
        times.tolist(),
        t0=0.0,
        tE=1.0,
        u0=u0,
        alpha=alpha,
        s=separation,
        q=mass_ratio,
        rho=rho,
    )
    actual = np.asarray(curve.magnifications, dtype=float)
    relative_error = np.abs(actual / reference - 1.0)

    assert set(curve.finite_source_method_names) == {"inverse_ray_polar"}
    assert float(relative_error.max()) < 1.5e-3


@pytest.mark.parametrize("separation,mass_ratio,y1,y2,rho", BINARY_FINITE_CASES)
def test_lcbinint_function_api_linear_limb_darkening_matches_vbm(
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
    options = lcbinint.Options(coordinates="center_of_mass", source_bins=80)
    actual = _model(lcbinint, params, options).magnification(y1)

    assert math.isfinite(actual)
    assert math.isclose(actual, reference, rel_tol=1.0e-2, abs_tol=1.0e-2)


@pytest.mark.parametrize("separation,mass_ratio,y1,y2,rho", BINARY_LIMB_DARKENING_CASES)
@pytest.mark.parametrize("mode", [1, 2])
def test_lcbinint_function_api_limb_darkening_matches_vbm(
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
    options = lcbinint.Options(coordinates="center_of_mass", mode=mode, source_bins=80)
    actual = _model(lcbinint, params, options).magnification(y1)

    assert math.isfinite(actual)
    assert math.isclose(actual, reference, rel_tol=5.0e-3, abs_tol=5.0e-3)


@pytest.mark.parametrize("mode", [1, 2])
@pytest.mark.parametrize("limb_darkened", [False, True])
def test_lcbinint_caustic_light_curve_points_match_vbm(
    mode, limb_darkened
):
    lcbinint = pytest.importorskip("lcbinint")
    module = pytest.importorskip("VBMicrolensing")

    separation = 1.4
    mass_ratio = 0.4
    y2 = -0.15
    rho = 0.025
    limb_darkening_c = 0.5
    y1_values = [-0.25854879065888237, -0.2485404503753128, 0.5821517931609672, 0.6021684737281067]

    vbbinary_lensing = module.VBMicrolensing()
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
    options = lcbinint.Options(coordinates="center_of_mass", mode=mode, source_bins=80)
    actual = _model(lcbinint, params, options).light_curve(y1_values).magnifications

    for actual_value, reference_value in zip(actual, reference):
        assert math.isfinite(actual_value)
        assert math.isclose(actual_value, reference_value, rel_tol=1.5e-3, abs_tol=1.5e-3)


@pytest.mark.parametrize("mode", [1, 2])
def test_lcbinint_high_magnification_light_curve_matches_vbm(
    mode,
):
    lcbinint = pytest.importorskip("lcbinint")
    module = pytest.importorskip("VBMicrolensing")

    separation = 1.0
    mass_ratio = 0.1
    umin = 0.01
    rho = 0.003
    times = [-0.08, -0.06, -0.04, -0.03, -0.02, -0.01, 0.0, 0.02, 0.04]

    vbbinary_lensing = module.VBMicrolensing()
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
        coordinates="center_of_mass",
        mode=mode,
        point_source_threshold=1.0e9,
        hexadecapole_threshold=1.0e9,
        source_bins=80,
    )
    actual = _model(lcbinint, params, options).light_curve(times).magnifications

    for actual_value, reference_value in zip(actual, reference):
        assert math.isfinite(actual_value)
        assert math.isclose(actual_value, reference_value, rel_tol=1.5e-3, abs_tol=1.5e-3)


@pytest.mark.parametrize("alpha,u0,separation,mass_ratio", [
    pytest.param(0.5, -0.01, 1.0, 0.001, id="nonzero_alpha_planetary"),
    pytest.param(0.0, 0.1, 1.0, 0.1, id="zero_alpha_low_q"),
    pytest.param(1.2, -0.05, 0.7, 0.3, id="nonzero_alpha_close_binary"),
    pytest.param(0.5, -0.01, 1.5, 0.001, id="nonzero_alpha_wide_planetary"),
])
def test_lcbinint_coordinates_mode_matches_binary_light_curve(
    alpha, u0, separation, mass_ratio
):
    lcbinint = pytest.importorskip("lcbinint")
    module = pytest.importorskip("VBMicrolensing")

    rho = 1e-4
    times = [-0.4, -0.2, 0.0, 0.2, 0.4]

    vbb = module.VBMicrolensing()
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
    options = lcbinint.Options(coordinates="vbm", source_bins=80)
    actual = _model(lcbinint, params, options).light_curve(times).magnifications

    for actual_value, reference_value in zip(actual, reference):
        assert math.isfinite(actual_value)
        assert math.isclose(actual_value, reference_value, rel_tol=1.5e-3, abs_tol=1.5e-3)


@pytest.mark.parametrize("mode", [1, 2, 3])
def test_lcbinint_coordinates_large_source_planetary_caustic_crossing(mode):
    lcbinint = pytest.importorskip("lcbinint")
    module = pytest.importorskip("VBMicrolensing")

    separation = 1.0
    mass_ratio = 0.001
    u0 = -0.01
    alpha = 0.5
    rho = 0.01
    times = [0.014035087719298178, 0.018045112781954864]

    vbb = module.VBMicrolensing()
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
        coordinates="vbm",
        source_bins=200,
        mode=mode,
    )
    actual = _model(lcbinint, params, options).light_curve(times).magnifications

    for actual_value, reference_value in zip(actual, reference):
        assert math.isfinite(actual_value)
        assert math.isclose(actual_value, reference_value, rel_tol=2.5e-3, abs_tol=2.5e-3)


def test_lcbinint_spine_mode_falls_back_for_oversampled_planetary_fold_arc():
    lcbinint = pytest.importorskip("lcbinint")
    module = pytest.importorskip("VBMicrolensing")

    separation = 1.0
    mass_ratio = 0.001
    u0 = -0.01
    alpha = 0.5
    rho = 1.0e-4
    times = [-0.03007518796992481, 0.04611528822055133]

    vbb = module.VBMicrolensing()
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
        coordinates="vbm",
        source_bins=200,        mode=3,
    )
    actual = _model(lcbinint, params, options).light_curve(times).magnifications

    for actual_value, reference_value in zip(actual, reference):
        assert math.isfinite(actual_value)
        assert math.isclose(actual_value, reference_value, rel_tol=5.0e-5, abs_tol=5.0e-5)


def test_lcbinint_adaptive_source_bins_refines_cartesian_grid_from_diagnostics():
    lcbinint = pytest.importorskip("lcbinint")
    module = pytest.importorskip("VBMicrolensing")

    separation = 1.0
    mass_ratio = 0.001
    u0 = -0.01
    alpha = 0.5
    rho = 1.0e-4
    times = [-0.03007518796992481, 0.04611528822055133]

    vbb = module.VBMicrolensing()
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
        coordinates="vbm",
        source_bins=50,
    )
    adaptive_options = lcbinint.Options(
        coordinates="vbm",
        source_bins=50,
        adaptive_source_bins=1,
        max_source_bins=200,
        reltol=1.0e-4,
    )

    fixed = _model(lcbinint, params, fixed_options).light_curve(times).magnifications
    adaptive = _model(lcbinint, params, adaptive_options).light_curve(times)
    fixed_rel = max(abs(a / b - 1.0) for a, b in zip(fixed, reference))
    adaptive_rel = max(abs(a / b - 1.0) for a, b in zip(adaptive.magnifications, reference))

    assert max(adaptive.finite_source_refinement_levels) > 0
    assert max(adaptive.finite_source_error_estimates) > 0.0
    assert adaptive_rel < 5.0e-4
    assert adaptive_rel <= max(1.05 * fixed_rel, 5.0e-4)
    assert (not adaptive.all_converged) or max(adaptive.finite_source_error_estimates) < 5.0e-4


def test_lcbinint_cartesian_ir_seeds_grazing_caustic_limb_images():
    lcbinint = pytest.importorskip("lcbinint")
    module = pytest.importorskip("VBMicrolensing")

    separation = 0.8
    mass_ratio = 0.01
    u0 = -0.01
    alpha = 0.3
    rho = 5.0e-3
    time = 0.006015037593984918

    vbb = module.VBMicrolensing()
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
    fixed = _model(lcbinint, 
        params,
        lcbinint.Options(source_bins=50, adaptive_source_bins=0),
    ).magnification(time)
    adaptive = _model(lcbinint, 
        params,
        lcbinint.Options(source_bins=50, reltol=1.0e-3, max_source_bins=400),
    ).magnification(time)

    assert math.isfinite(fixed)
    assert math.isfinite(adaptive)
    assert abs(fixed / reference - 1.0) < 3.0e-3
    assert abs(adaptive / reference - 1.0) < 1.0e-3


def test_lcbinint_cartesian_ir_keeps_same_parity_fold_branch_seed():
    lcbinint = pytest.importorskip("lcbinint")
    module = pytest.importorskip("VBMicrolensing")

    separation = 0.95
    mass_ratio = 0.01
    u0 = -0.001
    alpha = 0.5
    rho = 5.0e-3
    time = 0.006015037593984918

    vbb = module.VBMicrolensing()
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
    actual = _model(lcbinint, 
        params,
        lcbinint.Options(source_bins=50, adaptive_source_bins=0),
    ).magnification(time)

    assert abs(actual / reference - 1.0) < 1.0e-3


def test_lcbinint_local_boundary_estimate_avoids_ld_over_refinement():
    lcbinint = pytest.importorskip("lcbinint")
    module = pytest.importorskip("VBMicrolensing")

    separation = 0.95
    mass_ratio = 0.01
    u0 = -0.001
    alpha = 0.5
    rho = 5.0e-3
    limb_darkening_c = 0.5
    time = 0.010025062656641603

    vbb = module.VBMicrolensing()
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
    curve = _model(lcbinint, 
        params,
        lcbinint.Options(source_bins=50, adaptive_source_bins=1, max_source_bins=400, reltol=1.0e-3),
    ).light_curve([time])

    assert curve.finite_source_refinement_levels[0] <= 2
    assert curve.finite_source_converged[0]
    assert abs(curve.magnifications[0] / reference - 1.0) < 1.0e-3


def test_lcbinint_cartesian_ir_does_not_clip_moderate_fold_image_area():
    lcbinint = pytest.importorskip("lcbinint")
    module = pytest.importorskip("VBMicrolensing")

    separation = 1.160935533582098
    mass_ratio = 0.003086323166308305
    u0 = -0.001382074492745227
    alpha = 0.7396109342207111
    rho = 0.018442211657959038
    time = 0.037042588756702244

    vbb = module.VBMicrolensing()
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
        mode=1,    )
    actual = _model(lcbinint, params, options).light_curve([time]).magnifications[0]

    # Regression for the old |J| < 0.5 fold guard, which clipped a valid image
    # component and converged to a biased value near 32.065 (rel. error ~1.8e-3).
    assert math.isclose(actual, reference, rel_tol=1.0e-4, abs_tol=1.0e-4)


def test_lcbinint_cartesian_ir_does_not_double_subtract_wide_caustic_fold_overlap():
    lcbinint = pytest.importorskip("lcbinint")
    module = pytest.importorskip("VBMicrolensing")

    separation = 0.95
    mass_ratio = 0.01
    u0 = -0.01
    alpha = 0.5
    rho = 1.0e-2
    time = 0.006015037593984918

    vbb = module.VBMicrolensing()
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
    actual = _model(lcbinint, 
        params,
        lcbinint.Options(source_bins=50, mode=1),
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
    module = pytest.importorskip("VBMicrolensing")

    separation = 1.0
    mass_ratio = 0.001
    u0 = -0.01
    alpha = 0.5

    vbb = module.VBMicrolensing()
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
        coordinates="vbm",
        source_bins=50,
        max_source_bins=400,
        reltol=1.0e-3,
    )
    curve = _model(lcbinint, params, options).light_curve([time])
    actual = curve.magnifications[0]

    target = relative_tolerance * max(abs(actual), 1.0)
    abs_error = abs(actual - reference)
    assert (not curve.all_converged) or abs_error <= 1.05 * target
    assert math.isclose(actual, reference, rel_tol=relative_tolerance, abs_tol=relative_tolerance)


@pytest.mark.parametrize(
    (
        "separation",
        "mass_ratio",
        "u0",
        "alpha",
        "rho",
        "limb_darkening_c",
        "time",
        "source_bins",
        "reltol",
    ),
    [
        (
            0.5230965983889266,
            0.8995994635360866,
            -0.1557048648048206,
            0.660230880975817,
            0.005574278492276441,
            0.0,
            -0.11634042842617024,
            32,
            3.0e-4,
        ),
        (
            1.1713076898489538,
            0.0007844185165287579,
            0.004505401872662171,
            0.06427213952313962,
            0.010115357413313333,
            0.0,
            -0.2159711139387346,
            50,
            1.0e-3,
        ),
        (
            1.196479798624462,
            0.11491292602165815,
            -0.006051090180408796,
            2.513807433617138,
            0.0006101933225879976,
            0.5,
            -0.08225990756232886,
            32,
            1.0e-3,
        ),
        (
            1.381119461729901,
            0.29349056783334654,
            -0.005897478800685341,
            2.4416231208743646,
            0.0013658814448766453,
            0.0,
            -0.2569251621520807,
            32,
            1.0e-3,
        ),
    ],
)
def test_lcbinint_adaptive_does_not_accept_known_local_error_underestimates(
    separation,
    mass_ratio,
    u0,
    alpha,
    rho,
    limb_darkening_c,
    time,
    source_bins,
    reltol,
):
    lcbinint = pytest.importorskip("lcbinint")
    module = pytest.importorskip("VBMicrolensing")

    vbb = module.VBMicrolensing()
    vbb.Tol = 1.0e-5
    if limb_darkening_c != 0.0:
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
    curve = _model(lcbinint, 
        params,
        lcbinint.Options(
            source_bins=source_bins,
            max_source_bins=400,
            reltol=reltol,        ),
    ).light_curve([time])
    actual = curve.magnifications[0]
    target = reltol * max(abs(actual), 1.0)
    abs_error = abs(actual - reference)

    assert math.isfinite(actual)
    assert (not curve.finite_source_converged[0]) or abs_error <= 1.05 * target


def test_lcbinint_options_exposes_fields():
    lcbinint = pytest.importorskip("lcbinint")

    default_options = lcbinint.Options()
    assert default_options.source_bins == 50
    assert default_options.adaptive_source_bins == 0
    assert default_options.max_source_bins == 400
    assert default_options.finite_source_tol == 0.0
    assert default_options.finite_source_reltol == 0.0
    assert default_options.reltol == 0.0
    assert default_options.hex_tol == 1.0e-3

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
        hex_tol=3.0e-4,
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
    assert options.hex_tol == 3.0e-4

    auto_options = lcbinint.Options(reltol=1.0e-3)
    assert auto_options.adaptive_source_bins == 0
    assert auto_options.finite_source_reltol == 1.0e-3

    adaptive_options = lcbinint.Options(reltol=1.0e-3, adaptive_source_bins=1)
    assert adaptive_options.adaptive_source_bins == 1
    assert adaptive_options.finite_source_reltol == 1.0e-3


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
    options = lcbinint.Options(coordinates="vbm", caustic_bins=128, source_bins=20)

    actual = _model(lcbinint, params, options).magnification(0.2)

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
    mag1 = _model(lcbinint, params, lcbinint.Options(coordinates="vbm", mode=1, source_bins=source_bins)).magnification(y1)
    mag3 = _model(lcbinint, params, lcbinint.Options(coordinates="vbm", mode=3, source_bins=source_bins)).magnification(y1)
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
    mag1 = _model(lcbinint, params, lcbinint.Options(coordinates="vbm", mode=1, source_bins=source_bins)).magnification(y1)
    mag3 = _model(lcbinint, params, lcbinint.Options(coordinates="vbm", mode=3, source_bins=source_bins)).magnification(y1)
    assert math.isfinite(mag1)
    assert math.isfinite(mag3)
    assert math.isclose(mag3, mag1, rel_tol=1e-10, abs_tol=1e-10)


def test_lcbinint_lens_params_exposes_limb_darkening_coefficients():
    lcbinint = pytest.importorskip("lcbinint")

    params = lcbinint.LensParams(limb_darkening_c=0.5, limb_darkening_d=0.2)

    assert params.limb_darkening_c == 0.5
    assert params.limb_darkening_d == 0.2


def test_lcbinint_function_api_light_curve_exposes_source_trajectory():
    lcbinint = pytest.importorskip("lcbinint")
    params = lcbinint.LensParams(
        t0=10.0,
        tE=2.0,
        umin=0.1,
        theta=0.0,
        q=0.1,
        sep=1.0,
    )
    options = lcbinint.Options(coordinates="vbm")
    model = _model(lcbinint, params, options)
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
    assert len(curve.finite_source_methods) == len(times)
    assert len(curve.finite_source_method_names) == len(times)
    assert set(curve.finite_source_method_names) <= {
        "point_source",
        "hexadecapole",
        "inverse_ray_cartesian",
        "inverse_ray_polar",
        "inverse_ray_spine",
    }


def test_lcbinint_function_api_light_curve_accepts_empty_times():
    lcbinint = pytest.importorskip("lcbinint")
    model = _model(lcbinint, lcbinint.LensParams(), lcbinint.Options())

    curve = model.light_curve([])

    assert curve.times == []
    assert curve.magnifications == []
    assert model.magnifications([]) == []
    assert model.source_positions([]) == []


def test_lcbinint_callable_geometry_source_trajectory_matches_info():
    lcbinint = pytest.importorskip("lcbinint")

    func = lcbinint.LightCurve(options=lcbinint.Options(coordinates="vbm"))
    times = [-1.0, 0.0, 1.0]

    trajectory = func.source_trajectory(
        times,
        t0=0.0,
        tE=1.0,
        u0=0.1,
        alpha=0.0,
        s=1.0,
        q=0.1,
    )
    info = func.info(
        times,
        t0=0.0,
        tE=1.0,
        u0=0.1,
        alpha=0.0,
        s=1.0,
        q=0.1,
        rho=0.0,
    )

    assert trajectory.times == times
    assert trajectory.x == pytest.approx(info.source_x)
    assert trajectory.y == pytest.approx(info.source_y)


def test_lcbinint_callable_geometry_caustics_and_critical_curves_have_branches():
    lcbinint = pytest.importorskip("lcbinint")

    func = lcbinint.LightCurve(options=lcbinint.Options(coordinates="vbm"))

    caustics = func.caustics(s=1.0, q=1.0e-3, n_points=64)
    critical_curves = func.critical_curves(s=1.0, q=1.0e-3, n_points=64)

    assert len(caustics.x) == 4
    assert len(caustics.y) == 4
    assert len(critical_curves.x) == 4
    assert len(critical_curves.y) == 4
    assert [len(branch) for branch in caustics.x] == [64, 64, 64, 64]
    assert [len(branch) for branch in caustics.y] == [64, 64, 64, 64]
    assert [len(branch) for branch in critical_curves.x] == [64, 64, 64, 64]
    assert [len(branch) for branch in critical_curves.y] == [64, 64, 64, 64]


def test_lcbinint_parallax_callable_geometry_source_trajectory_is_available():
    lcbinint = pytest.importorskip("lcbinint")

    func = lcbinint.LightCurve(
        event=lcbinint.EventCoordinates(ra=267.6, dec=-29.1, tfix=2459000.0),
        parallax=True,
    )

    trajectory = func.source_trajectory(
        [2458990.0, 2459000.0],
        t0=2459000.0,
        tE=80.0,
        u0=0.1,
        alpha=0.2,
        s=1.0,
        q=0.1,
        piEN=0.01,
        piEE=0.02,
    )

    assert len(trajectory.x) == 2
    assert len(trajectory.y) == 2
    assert all(math.isfinite(value) for value in trajectory.x)
    assert all(math.isfinite(value) for value in trajectory.y)


def test_lcbinint_high_level_binary_light_curve_matches_function_api():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    times = [-0.1, -0.02, 0.0, 0.03, 0.1]
    options = lcbinint.Options(coordinates="vbm", source_bins=50)
    limb_darkening = lcbinint.LimbDarkening.linear(0.5)
    params = lcbinint.LensParams(
        t0=0.0,
        tE=1.0,
        u0=-0.01,
        alpha=0.5,
        sep=1.0,
        q=1.0e-3,
        rho=1.0e-4,
        limb_darkening_c=0.5,
    )

    expected = _model(lcbinint, params, options).light_curve(times).magnifications
    actual = lcbinint.binary_light_curve(
        times,
        t0=0.0,
        tE=1.0,
        u0=-0.01,
        alpha=0.5,
        s=1.0,
        q=1.0e-3,
        rho=1.0e-4,
        limb_darkening=limb_darkening,
        options=options,
    )

    assert actual == pytest.approx(expected)
    actual_array = lcbinint.light_curve(
        np.asarray(times),
        t0=0.0,
        tE=1.0,
        u0=-0.01,
        alpha=0.5,
        s=1.0,
        q=1.0e-3,
        rho=1.0e-4,
        limb_darkening=limb_darkening,
        options=options,
    )
    assert actual_array.tolist() == pytest.approx(expected)
    assert lcbinint.Options().coordinates == "vbm"
    assert lcbinint.binary_magnification(
        times[2],
        t0=0.0,
        tE=1.0,
        u0=-0.01,
        alpha=0.5,
        s=1.0,
        q=1.0e-3,
        rho=1.0e-4,
        limb_darkening=limb_darkening,
        options=options,
    ) == pytest.approx(expected[2])


def test_lcbinint_light_curve_func_matches_high_level_api():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    times = np.asarray([-0.1, -0.02, 0.0, 0.03, 0.1])
    options = lcbinint.Options(coordinates="vbm", source_bins=50, reltol=1.0e-3)
    event = lcbinint.EventCoordinates(ra=267.6, dec=-29.1, tfix=2459000.0)
    limb_darkening = lcbinint.LimbDarkening.linear(0.5)

    func = lcbinint.LightCurve(
        lens="binary_lens",
        event=event,
        options=options,
        limb_darkening=limb_darkening,
        orbital_motion_mode=lcbinint.OrbitalMotionMode.STATIC,
    )
    kwargs = dict(
        t0=0.0,
        tE=1.0,
        u0=-0.01,
        alpha=0.5,
        s=1.0,
        q=1.0e-3,
        rho=1.0e-3,
    )

    expected = lcbinint.light_curve(
        times,
        event=event,
        options=options,
        limb_darkening=limb_darkening,
        **kwargs,
    )
    actual = func(times, **kwargs)

    assert func.lens == "binary_lens"
    assert func.event.ra == pytest.approx(event.ra)
    assert func.options.source_bins == options.source_bins
    assert func.limb_darkening.c == pytest.approx(0.5)
    assert not func.parallax
    assert actual.tolist() == pytest.approx(expected.tolist())
    assert func.light_curve(times, **kwargs).tolist() == pytest.approx(expected.tolist())
    assert func.list(times.tolist(), **kwargs) == pytest.approx(expected.tolist())
    assert func.magnification(times[2], **kwargs) == pytest.approx(expected[2])

    info = func.info(times.tolist(), **kwargs)
    assert info.magnifications == pytest.approx(expected.tolist())
    assert len(info.finite_source_method_names) == len(times)


def test_lcbinint_parallax_light_curve_func_matches_high_level_api():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    times = np.asarray([2458990.0, 2459000.0, 2459010.0])
    options = lcbinint.Options(coordinates="vbm", source_bins=50, reltol=1.0e-3)
    event = lcbinint.EventCoordinates(ra=267.623337808, dec=-29.1164180355, tfix=2459000.0)
    limb_darkening = lcbinint.LimbDarkening.none()

    func = lcbinint.LightCurve(
        lens="binary_lens",
        event=event,
        options=options,
        limb_darkening=limb_darkening,
        parallax=True,
    )
    kwargs = dict(
        t0=2459001.0,
        tE=80.0,
        u0=0.12,
        alpha=0.4,
        s=1.1,
        q=1.0e-3,
        rho=1.0e-3,
        piEN=0.02,
        piEE=0.03,
    )

    expected = lcbinint.light_curve(
        times,
        event=event,
        options=options,
        limb_darkening=limb_darkening,
        **kwargs,
    )
    actual = func(times, **kwargs)

    assert type(func).__name__ == "LightCurve"
    assert lcbinint.ParallaxLightCurve is lcbinint.LightCurve
    assert lcbinint.ParallaxLightCurveFunc is lcbinint.LightCurve
    assert func.lens == "binary_lens"
    assert func.parallax
    assert actual.tolist() == pytest.approx(expected.tolist())
    assert func.info(times.tolist(), **kwargs).magnifications == pytest.approx(expected.tolist())
    assert func.magnification(times[1], **kwargs) == pytest.approx(expected[1])


def test_lcbinint_orbital_motion_light_curve_func_matches_high_level_api():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    times = np.asarray([-0.2, -0.05, 0.0, 0.08, 0.2])
    options = lcbinint.Options(coordinates="vbm", source_bins=50, reltol=1.0e-3)
    event = lcbinint.EventCoordinates(ra=267.6, dec=-29.1, tfix=0.0)
    limb_darkening = lcbinint.LimbDarkening.none()

    func = lcbinint.LightCurve(
        lens="binary_lens",
        event=event,
        options=options,
        limb_darkening=limb_darkening,
        orbital_motion_mode=lcbinint.OrbitalMotionMode.CIRCULAR,
    )
    kwargs = dict(
        t0=0.0,
        tE=1.0,
        u0=0.12,
        alpha=0.4,
        s=1.1,
        q=1.0e-3,
        rho=0.0,
        g1=0.01,
        g2=0.02,
        g3=0.03,
    )

    expected = lcbinint.light_curve(
        times,
        event=event,
        options=options,
        limb_darkening=limb_darkening,
        orbital_motion_mode=lcbinint.OrbitalMotionMode.CIRCULAR,
        **kwargs,
    )
    actual = func(times, **kwargs)

    assert actual.tolist() == pytest.approx(expected.tolist())
    assert func.info(times.tolist(), **kwargs).magnifications == pytest.approx(expected.tolist())
    assert func.magnification(times[2], **kwargs) == pytest.approx(expected[2])


def test_lcbinint_circular_lom_light_curve_func_matches_vbm():
    lcbinint = pytest.importorskip("lcbinint")
    module = pytest.importorskip("VBMicrolensing")
    np = pytest.importorskip("numpy")

    vbb = module.VBMicrolensing()
    vbb.Tol = 1.0e-5
    vbb.SetObjectCoordinates("17:45:40.04 -29:00:28.1")
    vbb.t0_par_fixed = 1

    separation = 0.97
    mass_ratio = 10.0 ** -1.5
    u0 = 0.01
    alpha = 0.1
    tE = math.exp(1.5)
    t0 = 10.0
    rho = 1.0e-8
    g1 = 0.01
    g2 = 0.02
    g3 = 0.03
    times = np.linspace(2.0, 69.0, 41)
    vbb.t0_par = t0

    reference = vbb.BinaryLightCurveOrbital(
        [
            math.log(separation),
            math.log(mass_ratio),
            u0,
            alpha,
            math.log(rho),
            math.log(tE),
            t0,
            0.0,
            0.0,
            g1,
            g2,
            g3,
        ],
        times.tolist(),
    )[0]

    func = lcbinint.LightCurve(
        lens="binary_lens",
        event=lcbinint.EventCoordinates(tfix=0.0),
        options=lcbinint.Options(coordinates="vbm", source_bins=50),
        limb_darkening=lcbinint.LimbDarkening.none(),
        orbital_motion_mode=lcbinint.OrbitalMotionMode.CIRCULAR,
    )
    actual = func(
        times,
        t0=t0,
        tE=tE,
        u0=u0,
        alpha=alpha,
        s=separation,
        q=mass_ratio,
        rho=0.0,
        g1=g1,
        g2=g2,
        g3=g3,
    )

    assert actual.tolist() == pytest.approx(reference, rel=1.0e-7, abs=1.0e-7)


def test_lcbinint_kepler_lom_light_curve_func_matches_vbm_when_reference_is_t0():
    lcbinint = pytest.importorskip("lcbinint")
    module = pytest.importorskip("VBMicrolensing")
    np = pytest.importorskip("numpy")

    vbb = module.VBMicrolensing()
    vbb.Tol = 1.0e-5
    vbb.SetObjectCoordinates("17:45:40.04 -29:00:28.1")
    vbb.t0_par_fixed = 1

    separation = 0.97
    mass_ratio = 10.0 ** -1.5
    u0 = 0.01
    alpha = 0.1
    tE = math.exp(1.5)
    t0 = 10.0
    rho = 1.0e-8
    g1 = 0.004
    g2 = 0.011
    g3 = 0.006
    szs = 0.2
    ar = 1.4
    vbm_reference_time = t0
    times = np.linspace(2.0, 69.0, 41)
    vbb.t0_par = vbm_reference_time

    reference = vbb.BinaryLightCurveKepler(
        [
            math.log(separation),
            math.log(mass_ratio),
            u0,
            alpha,
            math.log(rho),
            math.log(tE),
            t0,
            0.0,
            0.0,
            g1,
            g2,
            g3,
            szs,
            ar,
        ],
        times.tolist(),
    )[0]

    func = lcbinint.LightCurve(
        lens="binary_lens",
        event=lcbinint.EventCoordinates(tfix=vbm_reference_time),
        options=lcbinint.Options(coordinates="vbm", source_bins=50),
        limb_darkening=lcbinint.LimbDarkening.none(),
        orbital_motion_mode=lcbinint.OrbitalMotionMode.KEPLER,
    )
    actual = func(
        times,
        t0=t0,
        tE=tE,
        u0=u0,
        alpha=alpha,
        s=separation,
        q=mass_ratio,
        rho=0.0,
        g1=g1,
        g2=g2,
        g3=g3,
        lom_szs=szs,
        lom_ar=ar,
    )

    assert actual.tolist() == pytest.approx(reference, rel=1.0e-7, abs=1.0e-7)


def test_lcbinint_kepler_lom_reference_time_is_fixed_by_tfix():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    times = np.linspace(2.0, 69.0, 21)
    common = dict(
        tE=math.exp(1.5),
        u0=0.01,
        alpha=0.1,
        s=0.97,
        q=10.0 ** -1.5,
        rho=0.0,
        g1=0.004,
        g2=0.011,
        g3=0.006,
        lom_szs=0.2,
        lom_ar=1.4,
    )
    moving_reference = lcbinint.LightCurve(
        lens="binary_lens",
        event=lcbinint.EventCoordinates(tfix=0.0),
        options=lcbinint.Options(coordinates="vbm", source_bins=50),
        limb_darkening=lcbinint.LimbDarkening.none(),
        orbital_motion_mode=lcbinint.OrbitalMotionMode.KEPLER,
    )
    fixed_reference = lcbinint.LightCurve(
        lens="binary_lens",
        event=lcbinint.EventCoordinates(tfix=7000.0),
        options=lcbinint.Options(coordinates="vbm", source_bins=50),
        limb_darkening=lcbinint.LimbDarkening.none(),
        orbital_motion_mode=lcbinint.OrbitalMotionMode.KEPLER,
    )

    moving_reference_values = moving_reference(times, t0=10.0, **common)
    fixed_reference_values = fixed_reference(times, t0=10.0, **common)

    assert np.max(np.abs(moving_reference_values - fixed_reference_values)) > 1.0e-2


def test_lcbinint_limb_darkening_and_event_coordinate_helpers():
    lcbinint = pytest.importorskip("lcbinint")

    none = lcbinint.LimbDarkening.none()
    linear = lcbinint.LimbDarkening.linear(0.4)
    quadratic = lcbinint.LimbDarkening.quadratic(0.4, 0.2)
    event = lcbinint.EventCoordinates(ra=1.0, dec=2.0, tfix=3.0)

    assert (none.c, none.d) == (0.0, 0.0)
    assert (linear.c, linear.d) == (0.4, 0.0)
    assert (quadratic.c, quadratic.d) == (0.4, 0.2)
    assert (event.ra, event.dec, event.tfix) == (1.0, 2.0, 3.0)


def test_lcbinint_function_api_estimates_source_bins_from_self_convergence():
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
    options = lcbinint.Options(coordinates="vbm", source_bins=80)
    model = _model(lcbinint, params, options)
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
    actual = _model(lcbinint, params, lcbinint.Options(coordinates="vbm")).source_positions(times)

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


def test_lcbinint_function_api_circular_orbital_motion_uses_instantaneous_state():
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
    options = lcbinint.Options(coordinates="center_of_mass")

    separation, alpha, _ = lcbinint.circular_orbital_motion(
        time, params.sep, params.theta, params.g1, params.g2, params.g3, params.tfix
    )
    tau = (time - params.t0) / params.tE
    x = tau * math.cos(alpha) + params.umin * math.sin(alpha)
    y = params.umin * math.cos(alpha) - tau * math.sin(alpha)
    expected = lcbinint.binary_mag0(separation, params.q, x, y)
    actual = _model(lcbinint, params, options).magnification(time)

    assert math.isfinite(actual)
    assert math.isclose(actual, expected, rel_tol=1.0e-11, abs_tol=1.0e-11)
