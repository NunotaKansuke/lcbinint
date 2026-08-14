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
    curve = lcbinint.LightCurve(
        lens="binary", options=lcbinint.Options(coordinates="center_of_mass")
    )
    return curve.magnification(
        y1, t0=0.0, tE=1.0, u0=y2, alpha=0.0,
        s=separation, q=mass_ratio, rho=0.0,
    ).item()


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
        inverse_ray_grid=options.inverse_ray_grid,
        caustic_bins=options.caustic_bins,
        grid_ratio=options.grid_ratio,
        point_source_threshold=options.point_source_threshold,
        hexadecapole_threshold=options.hexadecapole_threshold,
        adaptive_hex_threshold=options.adaptive_hex_threshold,
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
        ra=params.ra,
        dec=params.dec,
        tfix=params.tfix,
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
        return self._curve().magnification(time, self._params).item()

    def magnifications(self, times):
        return self._curve()(times, self._params).tolist()

    def light_curve(self, times):
        return self._curve().info(times, self._params)

    def _curve(self):
        params = self._params
        return self._lcbinint.LightCurve(
            lens="binary",
            options=self._options,
            limb_darkening=self._lcbinint.LimbDarkening(
                c=params.limb_darkening_c, d=params.limb_darkening_d,
            ),
            parallax=bool(params.piEN or params.piEE),
            orbital_motion_mode=params.orbital_motion_mode,
            sky=self._lcbinint.obs.SkyCoord(params.ra, params.dec),
            t_ref=params.tfix,
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
    options = lcbinint.Options(coordinates="center_of_mass", inverse_ray_grid="polar", source_bins=80)
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
            inverse_ray_grid="polar",
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

    assert "inverse_ray_cartesian" not in set(curve.finite_source_method_names)
    assert set(curve.finite_source_method_names) <= {
        "inverse_ray_polar",
    }
    assert float(relative_error.max()) < 1.5e-3


def test_lcbinint_auto_inverse_ray_uses_polar_only_for_high_magnification():
    lcbinint = pytest.importorskip("lcbinint")

    options = lcbinint.Options(
        coordinates="vbm",
        inverse_ray_grid="auto",
        nbin="auto",
        point_source_threshold=1.0e9,
        hexadecapole_threshold=1.0e9,
    )
    func = lcbinint.LightCurve(options=options)
    common = dict(
        t0=0.0,
        tE=1.0,
        u0=-1.0e-3,
        alpha=0.5,
        s=0.95,
        q=1.0e-2,
        rho=5.0e-3,
    )

    high = func.info([0.004], **common)
    low = func.info([-0.2], **common)

    assert high.finite_source_method_names == ["inverse_ray_polar"]
    assert low.finite_source_method_names == ["inverse_ray_cartesian"]
    assert high.finite_source_error_estimates[0] > 0.0
    assert high.finite_source_converged == [True]
    high_budget = 1.0e-4 + 1.0e-3 * max(abs(high.magnifications[0]), 1.0)
    assert high.finite_source_error_estimates[0] <= high_budget

    explicit_default = lcbinint.LightCurve(options=lcbinint.Options(
        coordinates="vbm",
        inverse_ray_grid="auto",
        nbin="auto",
        reltol=1.0e-3,
        point_source_threshold=1.0e9,
        hexadecapole_threshold=1.0e9,
    )).info([0.004], **common)
    assert explicit_default.magnifications == high.magnifications
    assert explicit_default.finite_source_error_estimates == high.finite_source_error_estimates
    assert explicit_default.finite_source_refinement_levels == high.finite_source_refinement_levels

    dark_limb = lcbinint.LightCurve(
        options=options,
        limb_darkening=lcbinint.LimbDarkening.linear(1.0),
    ).info([0.004], **common)
    assert dark_limb.finite_source_method_names == ["inverse_ray_polar"]
    assert dark_limb.finite_source_error_estimates[0] > 0.0


def test_lcbinint_auto_nbin_reproduces_independent_validation_row():
    """Auto accepts an independently accurate calibrated preselection."""
    lcbinint = pytest.importorskip("lcbinint")
    source_x = 0.008845738870878478
    source_y = 0.001678002323850983
    common_options = dict(
        coordinates="center_of_mass",
        caustic_bins=1200,
        inverse_ray_grid="auto",
        point_source_threshold=0.0,
        hexadecapole_threshold=0.0,
        adaptive_hex_threshold=0.0,
    )
    params = dict(
        t0=0.0, tE=1.0, u0=source_y, alpha=0.0,
        s=0.3, q=0.001, rho=0.003,
    )
    limb = lcbinint.LimbDarkening.linear(0.5)
    auto = lcbinint.LightCurve(
        options=lcbinint.Options(nbin="auto", **common_options),
        limb_darkening=limb,
    ).info([source_x], **params)
    fixed = lcbinint.LightCurve(
        options=lcbinint.Options(nbin=40, **common_options),
        limb_darkening=limb,
    ).info([source_x], **params)

    assert auto.finite_source_method_names == ["inverse_ray_cartesian"]
    assert auto.finite_source_refinement_levels == [0]
    assert auto.finite_source_converged == [True]
    assert fixed.finite_source_refinement_levels == [0]
    assert fixed.finite_source_converged == [False]
    assert auto.magnifications != fixed.magnifications
    assert abs(auto.magnifications[0] / 113.42113550323353 - 1.0) < 1.0e-3


def test_lcbinint_auto_nbin_accepts_second_order_smooth_resonant_boundary():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    times = np.linspace(-0.15, 0.07, 40)
    params = lcbinint.LensParams(
        t0=0.0,
        tE=1.0,
        u0=0.03,
        alpha=0.0,
        sep=1.05,
        q=1.0e-3,
        rho=1.8e-3,
    )
    limb = lcbinint.LimbDarkening.linear(0.407200474)

    def evaluate(nbin, max_source_bins=400):
        return lcbinint.LightCurve(
            options=lcbinint.Options(
                coordinates="vbm",
                nbin=nbin,
                max_source_bins=max_source_bins,
            ),
            limb_darkening=limb,
        ).info(times, params)

    auto = evaluate("auto")
    capped = evaluate("auto", max_source_bins=40)
    fixed = evaluate(40)
    reference = evaluate(200)

    # Epochs that clear the caustic by at least one source radius have a smooth
    # image boundary, the second-order edge correction applies, and none of them
    # may spend a refinement.  One epoch is not of that kind: at index 8
    # (t = -0.1049) the caustic passes 0.016 rho from the disk centre, so the
    # disk straddles it and the boundary is genuinely not smooth there.  The
    # certified support seeds the fold pair, the scan reports a fold seed, and
    # the error indicator correctly stops applying its smooth-boundary discount.
    # That epoch is excluded below and checked on its own terms.
    smooth = [
        index
        for index, distance in enumerate(auto.caustic_distances)
        if distance >= params.rho
    ]
    crossing = [
        index
        for index, distance in enumerate(auto.caustic_distances)
        if distance < 0.1 * params.rho
    ]
    assert len(smooth) == 38
    assert crossing == [8]

    for info in (auto, capped, fixed):
        assert all(info.finite_source_converged[index] for index in smooth)
        assert all(
            info.finite_source_refinement_levels[index] == 0 for index in smooth
        )
    assert any(
        method == "inverse_ray_cartesian" and level == 0
        for method, level in zip(
            auto.finite_source_method_names,
            auto.finite_source_refinement_levels,
        )
    )

    # The multi-interval scanline fill resolves the crossing at both the
    # automatic bucket and the 40-bin cap.  Their measured values remain inside
    # the default part-in-a-thousand budget checked below.
    assert auto.all_converged
    assert max(auto.finite_source_refinement_levels) == 0
    assert capped.all_converged
    assert fixed.all_converged
    assert max(capped.finite_source_refinement_levels) == 0
    assert max(fixed.finite_source_refinement_levels) == 0
    assert max(
        abs(actual / expected - 1.0)
        for actual, expected in zip(auto.magnifications, reference.magnifications)
    ) < 1.0e-3
    assert max(
        abs(actual / expected - 1.0)
        for actual, expected in zip(fixed.magnifications, reference.magnifications)
    ) < 1.0e-3


def test_measured_far_caustic_releases_local_topology_veto_to_hexadecapole():
    """A smooth high-mag point must not be sent straight to inverse rays.

    This geometry is taken from a Roman parallax+Kepler trajectory.  The
    local ghost/planetary proxy is conservative here, but the nearest measured
    caustic segment is more than fifteen source radii away.  The independently
    checked hexadecapole result agrees with VBM while avoiding the expensive
    inverse-ray path.
    """
    lcbinint = pytest.importorskip("lcbinint")

    source_x = 0.0009490340842049525
    source_y = 0.02998953302951774
    rho = 0.0018
    curve = lcbinint.LightCurve(
        coordinates="vbm",
        limb_darkening=lcbinint.LimbDarkening.linear(0.407200474),
    ).info(
        [source_x],
        t0=0.0,
        tE=1.0,
        u0=source_y,
        alpha=0.0,
        s=1.300000973571089,
        q=1.0e-3,
        rho=rho,
    )

    assert curve.caustic_distances[0] / rho > 15.0
    assert curve.finite_source_method_names == ["hexadecapole"]
    assert abs(curve.magnifications[0] / 33.368159744967215 - 1.0) < 1.0e-5


def test_lcbinint_auto_inverse_ray_avoids_tiny_source_cartesian_aliasing():
    lcbinint = pytest.importorskip("lcbinint")
    module = pytest.importorskip("VBMicrolensing")

    separation = 1.0
    mass_ratio = 1.0e-3
    u0 = -1.0e-4
    alpha = 0.5
    rho = 1.0e-4
    time = 0.0006

    vbb = module.VBMicrolensing()
    vbb.Tol = 1.0e-6
    reference = vbb.BinaryLightCurve(
        [math.log(separation), math.log(mass_ratio), u0, alpha, math.log(rho), 0.0, 0.0],
        [time],
    )[0][0]

    func = lcbinint.LightCurve(
        options=lcbinint.Options(
            coordinates="vbm",
            inverse_ray_grid="auto",
            source_bins=50,
            point_source_threshold=1.0e9,
            hexadecapole_threshold=1.0e9,
        )
    )
    actual = func.info(
        [time],
        t0=0.0,
        tE=1.0,
        u0=u0,
        alpha=alpha,
        s=separation,
        q=mass_ratio,
        rho=rho,
    )

    assert actual.finite_source_method_names == ["inverse_ray_polar"]
    assert abs(actual.magnifications[0] / reference - 1.0) < 1.0e-3


def test_lcbinint_polar_uses_radius_aware_angular_resolution_for_low_magnification():
    lcbinint = pytest.importorskip("lcbinint")
    module = pytest.importorskip("VBMicrolensing")

    separation = 1.251936920212136
    mass_ratio = 0.010229080749960234
    u0 = -0.045915477051696046
    alpha = 1.008883116714675
    rho = 0.03791189085132994
    limb_darkening_c = 0.5
    time = 0.2526222052684984

    vbb = module.VBMicrolensing()
    vbb.Tol = 1.0e-6
    vbb.a1 = limb_darkening_c
    reference = vbb.BinaryLightCurve(
        [math.log(separation), math.log(mass_ratio), u0, alpha, math.log(rho), 0.0, 0.0],
        [time],
    )[0][0]

    func = lcbinint.LightCurve(
        options=lcbinint.Options(
            coordinates="vbm",
            inverse_ray_grid="polar",
            source_bins=50,
            point_source_threshold=1.0e9,
            hexadecapole_threshold=1.0e9,
        ),
        limb_darkening=lcbinint.LimbDarkening.linear(limb_darkening_c),
    )
    actual = func.info(
        [time],
        t0=0.0,
        tE=1.0,
        u0=u0,
        alpha=alpha,
        s=separation,
        q=mass_ratio,
        rho=rho,
    )

    assert actual.finite_source_method_names == ["inverse_ray_polar"]
    assert abs(actual.magnifications[0] / reference - 1.0) < 1.0e-3


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
@pytest.mark.parametrize("inverse_ray_grid", ["cartesian", "polar"])
def test_lcbinint_function_api_limb_darkening_matches_vbm(
    separation, mass_ratio, y1, y2, rho, inverse_ray_grid
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
        coordinates="center_of_mass",
        inverse_ray_grid=inverse_ray_grid,
        source_bins=80,
    )
    actual = _model(lcbinint, params, options).magnification(y1)

    assert math.isfinite(actual)
    assert math.isclose(actual, reference, rel_tol=5.0e-3, abs_tol=5.0e-3)


@pytest.mark.parametrize("inverse_ray_grid", ["cartesian", "polar"])
@pytest.mark.parametrize("limb_darkened", [False, True])
def test_lcbinint_caustic_light_curve_points_match_vbm(
    inverse_ray_grid, limb_darkened
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
    options = lcbinint.Options(
        coordinates="center_of_mass",
        inverse_ray_grid=inverse_ray_grid,
        source_bins=80,
    )
    actual = _model(lcbinint, params, options).light_curve(y1_values).magnifications

    for actual_value, reference_value in zip(actual, reference):
        assert math.isfinite(actual_value)
        assert math.isclose(actual_value, reference_value, rel_tol=1.5e-3, abs_tol=1.5e-3)


@pytest.mark.parametrize("inverse_ray_grid", ["cartesian", "polar"])
def test_lcbinint_high_magnification_light_curve_matches_vbm(
    inverse_ray_grid,
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
        inverse_ray_grid=inverse_ray_grid,
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


@pytest.mark.parametrize("inverse_ray_grid", ["cartesian", "polar", "auto"])
def test_lcbinint_coordinates_large_source_planetary_caustic_crossing(inverse_ray_grid):
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
        inverse_ray_grid=inverse_ray_grid,
    )
    actual = _model(lcbinint, params, options).light_curve(times).magnifications

    for actual_value, reference_value in zip(actual, reference):
        assert math.isfinite(actual_value)
        assert math.isclose(actual_value, reference_value, rel_tol=2.5e-3, abs_tol=2.5e-3)


def test_lcbinint_auto_nbin_selects_resolution_and_meets_runtime_budget():
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
        nbin="auto",
        max_source_bins=200,
        reltol=1.0e-4,
    )

    fixed = _model(lcbinint, params, fixed_options).light_curve(times).magnifications
    adaptive = _model(lcbinint, params, adaptive_options).light_curve(times)
    fixed_rel = max(abs(a / b - 1.0) for a, b in zip(fixed, reference))
    adaptive_rel = max(abs(a / b - 1.0) for a, b in zip(adaptive.magnifications, reference))

    assert max(adaptive.finite_source_refinement_levels) <= 13
    assert max(adaptive.finite_source_error_estimates) > 0.0
    assert adaptive_rel < 5.0e-4
    assert adaptive_rel <= max(1.05 * fixed_rel, 5.0e-4)
    target_errors = [
        adaptive_options.finite_source_tol
        + adaptive_options.finite_source_reltol * max(abs(mag), 1.0)
        for mag in adaptive.magnifications
    ]
    for converged, error, target in zip(
        adaptive.finite_source_converged,
        adaptive.finite_source_error_estimates,
        target_errors,
    ):
        assert (not converged) or error <= target


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
        lcbinint.Options(source_bins=50),
    ).magnification(time)
    adaptive = _model(lcbinint, 
        params,
        lcbinint.Options(nbin="auto", reltol=1.0e-3, max_source_bins=400),
    ).magnification(time)

    assert math.isfinite(fixed)
    assert math.isfinite(adaptive)
    assert abs(fixed / reference - 1.0) < 3.0e-3
    assert abs(adaptive / reference - 1.0) < 1.0e-3


@pytest.mark.parametrize("grid", ["cartesian", "polar"])
def test_explicit_binary_grid_does_not_fall_back_to_source_plane(grid):
    """An explicit image-plane request must actually execute that grid.

    An explicit image-plane request must not be intercepted by any automatic
    fallback: otherwise a fixed-grid accuracy or speed test silently measures
    a different integrator.  This is a small, high-magnification grazing case
    where the old route was reproducibly selected.
    """
    lcbinint = pytest.importorskip("lcbinint")
    curve = lcbinint.LightCurve(options=lcbinint.Options(
        coordinates="vbm",
        inverse_ray_grid=grid,
        nbin=64,
        max_source_bins=64,
        point_source_threshold=0.0,
        hexadecapole_threshold=0.0,
        adaptive_hex_threshold=0.0,
    ))
    info = curve.info(
        -0.019735930969284804,
        t0=0.0,
        tE=1.0,
        u0=-0.01132640105806183,
        alpha=0.0,
        s=0.3,
        q=0.01,
        rho=0.020133555236025995,
    )
    assert info.finite_source_method_names == [f"inverse_ray_{grid}"]


def test_binary_auto_uses_inverse_ray_for_grazing_source():
    """Automatic binary routing must use the calibrated IR path.

    Source-plane quadrature remains an explicit diagnostic/preplanned method,
    but it is not a production binary fallback.  Keeping this assertion on a
    formerly source-plane-routed grazing point prevents a speed comparison
    from silently switching integrators again.
    """
    lcbinint = pytest.importorskip("lcbinint")
    curve = lcbinint.LightCurve(options=lcbinint.Options(
        coordinates="vbm",
        inverse_ray_grid="auto",
        nbin="auto",
        reltol=1.0e-3,
        max_source_bins=400,
        caustic_bins=1400,
    ))
    info = curve.info(
        [0.006015037593984918],
        t0=0.0,
        tE=1.0,
        u0=-0.01,
        alpha=0.3,
        s=0.8,
        q=0.01,
        rho=5.0e-3,
    )
    assert info.finite_source_method_names[0] in {
        "inverse_ray_cartesian",
        "inverse_ray_polar",
    }


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
        lcbinint.Options(source_bins=50),
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
        lcbinint.Options(nbin="auto", max_source_bins=400, reltol=1.0e-3),
    ).light_curve([time])

    assert curve.finite_source_refinement_levels[0] <= 1
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
        inverse_ray_grid="cartesian",    )
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
        lcbinint.Options(source_bins=50, inverse_ray_grid="cartesian"),
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
        nbin="auto",
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
            reltol=reltol,
            inverse_ray_grid="cartesian",
        ),
    ).light_curve([time])
    actual = curve.finite_source_magnifications[0]
    target = reltol * max(abs(actual), 1.0)
    abs_error = abs(actual - reference)

    assert math.isfinite(actual)
    assert math.isfinite(curve.magnifications[0]) == curve.finite_source_converged[0]
    assert (not curve.finite_source_converged[0]) or abs_error <= 1.05 * target


def test_absolute_only_tolerance_does_not_inherit_default_relative_budget():
    """An uncalibrated absolute-only auto request fails with its own status."""
    lcbinint = pytest.importorskip("lcbinint")

    params = lcbinint.LensParams(
        t0=0.0,
        tE=1.0,
        u0=-0.1557048648048206,
        alpha=0.660230880975817,
        q=0.8995994635360866,
        sep=0.5230965983889266,
        rho=0.005574278492276441,
    )
    curve = _model(
        lcbinint,
        params,
        lcbinint.Options(
            nbin="auto",
            max_source_bins=400,
            tol=1.0e-5,
            reltol=0.0,
            inverse_ray_grid="cartesian",
        ),
    ).light_curve([-0.11634042842617024])

    assert curve.statuses == ["unsupported_tolerance"]
    assert math.isnan(curve.magnifications[0])
    assert curve.finite_source_converged == [False]
    assert not curve.all_converged

    fixed = _model(
        lcbinint,
        params,
        lcbinint.Options(
            nbin=400,
            tol=1.0e-5,
            reltol=0.0,
            inverse_ray_grid="cartesian",
        ),
    ).light_curve([-0.11634042842617024])
    assert fixed.statuses != ["unsupported_tolerance"]


def test_tightening_absolute_tolerance_is_monotonic_and_order_independent():
    lcbinint = pytest.importorskip("lcbinint")
    params = dict(
        t0=0.0,
        tE=1.0,
        u0=-0.1557048648048206,
        alpha=0.660230880975817,
        q=0.8995994635360866,
        s=0.5230965983889266,
        rho=0.005574278492276441,
    )
    time = -0.11634042842617024

    def evaluate(tol, times):
        return lcbinint.LightCurve(options=lcbinint.Options(
            nbin=80,
            tol=tol,
            reltol=0.0,
            inverse_ray_grid="cartesian",
        )).info(times, **params)

    loose = evaluate(1.0e-2, [time])
    tight = evaluate(1.0e-5, [time])
    assert loose.finite_source_magnifications == tight.finite_source_magnifications
    assert loose.finite_source_error_estimates == tight.finite_source_error_estimates
    assert loose.finite_source_converged == [True]
    assert tight.finite_source_converged == [False]

    times = [time, -0.11]
    forward = evaluate(1.0e-5, times)
    reverse = evaluate(1.0e-5, list(reversed(times)))
    assert forward.finite_source_magnifications == list(
        reversed(reverse.finite_source_magnifications)
    )
    assert forward.finite_source_error_estimates == list(
        reversed(reverse.finite_source_error_estimates)
    )
    assert forward.finite_source_converged == list(
        reversed(reverse.finite_source_converged)
    )


def test_explicit_polar_tolerance_has_measured_convergence_state():
    """An explicit tolerance cannot be certified by a synthetic zero error."""
    lcbinint = pytest.importorskip("lcbinint")

    options = lcbinint.Options(
        coordinates="vbm",
        inverse_ray_grid="polar",
        nbin=50,
        tol=1.0e-5,
        reltol=0.0,
        point_source_threshold=1.0e9,
        hexadecapole_threshold=1.0e9,
    )
    curve = lcbinint.LightCurve(options=options).info(
        [0.004],
        t0=0.0,
        tE=1.0,
        u0=-1.0e-3,
        alpha=0.5,
        s=0.95,
        q=1.0e-2,
        rho=5.0e-3,
    )

    assert curve.finite_source_method_names == ["inverse_ray_polar"]
    assert curve.finite_source_error_estimates[0] > 0.0
    assert (
        not curve.finite_source_converged[0]
        or curve.finite_source_error_estimates[0] <= 1.0e-5
    )


def test_explicit_cartesian_tolerance_cross_checks_optimistic_area_estimate():
    """A coarse/fine check catches a known low-amplitude lattice alias."""
    lcbinint = pytest.importorskip("lcbinint")
    light_curve = lcbinint.LightCurve(options=lcbinint.Options(
        coordinates="center_of_mass",
        caustic_bins=600,
        nbin=240,
        tol=1.0e-5,
        reltol=0.0,
        hex_tol=1.0e-5,
        inverse_ray_grid="cartesian",
    ))
    params = dict(
        t0=0.0,
        tE=1.0,
        u0=0.008912280938420293,
        alpha=0.0,
        s=0.1,
        q=1.0e-6,
        rho=0.01,
    )
    curve = light_curve.info([-9.872063992894269], **params)

    assert curve.finite_source_method_names == ["inverse_ray_cartesian"]
    assert curve.finite_source_error_estimates[0] > 1.0e-5
    assert curve.finite_source_converged == [False]
    with pytest.raises(RuntimeError, match="numerical error"):
        light_curve.magnification([-9.872063992894269], **params)


def test_lcbinint_options_exposes_fields():
    lcbinint = pytest.importorskip("lcbinint")

    default_options = lcbinint.Options()
    assert default_options.source_bins == 50
    assert default_options.nbin == "auto"
    assert default_options.max_source_bins == 400
    assert default_options.finite_source_tol == 0.0
    assert default_options.finite_source_reltol == 0.0
    assert default_options.reltol == 0.0
    assert default_options.hex_tol == 1.0e-3
    assert default_options.inverse_ray_grid == "auto"
    assert default_options.polar_nbin is None
    assert default_options.polar_source_bins is None
    assert default_options.polar_grid_ratio is None

    options = lcbinint.Options(
        caustic_bins=128,
        nbin=40,
        inverse_ray_grid="polar",
        polar_nbin=36,
        polar_grid_ratio=5.0,
        point_source_threshold=8.0,
        hexadecapole_threshold=2.5,
        max_source_bins=160,
        tol=1.0e-4,
        reltol=2.0e-4,
        hex_tol=3.0e-4,
    )

    assert options.caustic_bins == 128
    assert options.source_bins == 40
    assert options.nbin == 40
    assert options.inverse_ray_grid == "polar"
    assert options._mode == 2
    assert options.polar_nbin == 36
    assert options.polar_source_bins == 36
    assert options.polar_grid_ratio == 5.0
    assert options.point_source_threshold == 8.0
    assert options.hexadecapole_threshold == 2.5
    assert options.max_source_bins == 160
    assert options.finite_source_tol == 1.0e-4
    assert options.finite_source_reltol == 2.0e-4
    assert options.tol == 1.0e-4
    assert options.reltol == 2.0e-4
    assert options.hex_tol == 3.0e-4

    options.polar_nbin = None
    options.polar_grid_ratio = None
    assert options.polar_nbin is None
    assert options.polar_source_bins is None
    assert options.polar_grid_ratio is None

    auto_options = lcbinint.Options(reltol=1.0e-3)
    assert auto_options.nbin == "auto"
    assert auto_options.finite_source_reltol == 1.0e-3

    adaptive_options = lcbinint.Options(nbin="auto", reltol=1.0e-3)
    assert adaptive_options.nbin == "auto"
    assert adaptive_options.finite_source_reltol == 1.0e-3

    adaptive_options.nbin = 64
    assert adaptive_options.nbin == 64
    adaptive_options.nbin = "auto"
    assert adaptive_options.nbin == "auto"
    with pytest.raises(ValueError, match="positive integer or 'auto'"):
        lcbinint.Options(nbin="adaptive")
    with pytest.raises(ValueError, match="positive integer or 'auto'"):
        lcbinint.Options(nbin=0)


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

    # The resonant topology is one physical closed curve.  All four polynomial
    # roots participate in that curve rather than being exposed as four
    # root-index branches.
    assert len(caustics.x) == 1
    assert len(caustics.y) == 1
    assert len(critical_curves.x) == 1
    assert len(critical_curves.y) == 1
    assert [len(branch) for branch in caustics.x] == [4 * 64]
    assert [len(branch) for branch in caustics.y] == [4 * 64]
    assert [len(branch) for branch in critical_curves.x] == [4 * 64]
    assert [len(branch) for branch in critical_curves.y] == [4 * 64]


def test_lcbinint_parallax_callable_geometry_source_trajectory_is_available():
    lcbinint = pytest.importorskip("lcbinint")

    func = lcbinint.LightCurve(
        sky=lcbinint.obs.SkyCoord(267.6, -29.1),
        t_ref=2459000.0,
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


def test_light_curve_is_the_only_high_level_evaluation_api():
    lcbinint = pytest.importorskip("lcbinint")
    np = pytest.importorskip("numpy")

    times = np.asarray([-0.1, -0.02, 0.0, 0.03, 0.1])
    curve = lcbinint.LightCurve(
        lens="binary",
        options=lcbinint.Options(coordinates="vbm", source_bins=50),
        limb_darkening=lcbinint.LimbDarkening.linear(0.5),
        sky=lcbinint.obs.SkyCoord(267.6, -29.1),
        t_ref=2459000.0,
    )
    params = dict(t0=0.0, tE=1.0, u0=-0.01, alpha=0.5,
                  s=1.0, q=1.0e-3, rho=1.0e-3)

    actual = curve(times, **params)
    assert curve(times, params).tolist() == pytest.approx(actual.tolist())
    assert curve.magnification(times[2], **params) == pytest.approx(actual[2])
    assert curve.info(times, **params).magnifications == pytest.approx(actual.tolist())
    for removed in ("light_curve", "binary_light_curve", "magnification", "binary_magnification"):
        assert not hasattr(lcbinint, removed)


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
        lens="binary",
        t_ref=0.0,
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
        lens="binary",
        t_ref=vbm_reference_time,
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
        lens="binary",
        t_ref=0.0,
        options=lcbinint.Options(coordinates="vbm", source_bins=50),
        limb_darkening=lcbinint.LimbDarkening.none(),
        orbital_motion_mode=lcbinint.OrbitalMotionMode.KEPLER,
    )
    fixed_reference = lcbinint.LightCurve(
        lens="binary",
        t_ref=7000.0,
        options=lcbinint.Options(coordinates="vbm", source_bins=50),
        limb_darkening=lcbinint.LimbDarkening.none(),
        orbital_motion_mode=lcbinint.OrbitalMotionMode.KEPLER,
    )

    moving_reference_values = moving_reference(times, t0=10.0, **common)
    fixed_reference_values = fixed_reference(times, t0=10.0, **common)

    assert np.max(np.abs(moving_reference_values - fixed_reference_values)) > 1.0e-2


def test_lcbinint_limb_darkening_and_obs_helpers():
    lcbinint = pytest.importorskip("lcbinint")

    none = lcbinint.LimbDarkening.none()
    linear = lcbinint.LimbDarkening.linear(0.4)
    quadratic = lcbinint.LimbDarkening.quadratic(0.4, 0.2)
    sky = lcbinint.obs.SkyCoord(1.0, 2.0)
    site = lcbinint.obs.Site("ground", 3.0, 4.0)

    assert (none.c, none.d) == (0.0, 0.0)
    assert (linear.c, linear.d) == (0.4, 0.0)
    assert (quadratic.c, quadratic.d) == (0.4, 0.2)
    assert (sky.ra_deg, sky.dec_deg) == (1.0, 2.0)
    assert (site.lat_deg, site.lon_deg) == (3.0, 4.0)


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
def test_lcbinint_circular_lom_separation_matches_microjax_vbm_formula(time):
    lom = _microjax_lom()
    lcbinint = pytest.importorskip("lcbinint")
    args = dict(
        s=1.25,
        alpha=0.4,
        g1=0.006,
        g2=-0.004,
        g3=0.009,
        tfix=2450000.0,
    )

    curve = lcbinint.LightCurve(
        lens="binary",
        orbital_motion="circular",
        t_ref=args["tfix"],
    )
    actual_s = curve.separation(time, args)
    expected_s, _, _ = lom.circular_orbital_motion_3d(
        time,
        s0=args["s"],
        alpha0=args["alpha"],
        w1=args["g1"],
        w2=args["g2"],
        w3=args["g3"],
        tref=args["tfix"],
    )

    assert math.isclose(actual_s, float(expected_s), rel_tol=1.0e-11, abs_tol=1.0e-11)


@pytest.mark.parametrize("time", [2450000.0, 2450013.0, 2450041.5])
def test_lcbinint_kepler_lom_separation_matches_microjax_vbm_formula(time):
    lom = _microjax_lom()
    lcbinint = pytest.importorskip("lcbinint")
    args = dict(
        s=0.92,
        alpha=-0.3,
        g1=0.004,
        g2=0.011,
        g3=0.006,
        lom_szs=0.35,
        lom_ar=1.4,
        tfix=2450000.0,
    )

    curve = lcbinint.LightCurve(
        lens="binary",
        orbital_motion="kepler",
        t_ref=args["tfix"],
    )
    actual_s = curve.separation(time, args)
    expected_s, _, _ = lom.elliptic_orbital_motion_3d(
        time,
        s0=args["s"],
        alpha0=args["alpha"],
        w1=args["g1"],
        w2=args["g2"],
        w3=args["g3"],
        szs=args["lom_szs"],
        ar=args["lom_ar"],
        tref=args["tfix"],
    )

    assert math.isclose(actual_s, float(expected_s), rel_tol=1.0e-11, abs_tol=1.0e-11)


def test_lcbinint_light_curve_separation_uses_instantaneous_lom_state():
    lcbinint = pytest.importorskip("lcbinint")
    time = 2.0
    params = dict(
        t0=1.0,
        tE=1.0,
        u0=0.15,
        q=0.2,
        s=1.1,
        alpha=0.0,
        g1=0.02,
        g2=0.08,
        g3=0.03,
    )
    curve = lcbinint.LightCurve(
        lens="binary",
        options=lcbinint.Options(coordinates="center_of_mass"),
        orbital_motion="circular",
        t_ref=0.5,
    )

    actual = curve.separation(time, params)
    static = lcbinint.LightCurve(lens="binary").separation(params)

    assert math.isfinite(actual)
    assert actual != pytest.approx(static)
