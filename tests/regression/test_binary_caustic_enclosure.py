"""Independent critical quartics check the continuous-caustic routing bound."""

import math

import numpy as np
import pytest


def _independent_caustic(s, q):
    a = -abs(s) if q < 1 else abs(s)
    q = min(q, 1/q)
    m1 = 1/(1+q)
    m2 = q*m1
    points = []
    # Offset phases deliberately avoid the centres/edges of native intervals.
    for angle in 2*np.pi*(np.arange(257)+.371)/257:
        phase = np.exp(1j*angle)
        coefficients = [m2*a*a, -2*m2*a, 1-phase*a*a, 2*phase*a, -phase]
        for root in np.roots(coefficients[::-1]):
            points.append(root-m1/(np.conj(root)-a)-m2/np.conj(root)-a*m1)
    return np.asarray(points)


@pytest.mark.parametrize("s,q", [
    (.45, .3), (.7, .3), (1., .1), (1.2, .01), (2.4, 1e-4),
    (.7, 1/.3), (1.2, 100.), (2.4, 1e4),
    (.1, 1e-4), (.3, 1.), (.7, 1.), (.2, 1e3),
    # Exact topology boundaries can have colliding critical roots. An
    # unproved enclosure must return zero and preserve the ordinary fallback.
    ((1+.1**(1/3))**1.5/math.sqrt(1.1), .1),
])
def test_continuous_enclosure_contains_independent_critical_phases(s, q):
    lcbinint = pytest.importorskip("lcbinint")
    bound = lcbinint._native._binary_caustic_clearance_lower_bound
    caustic = _independent_caustic(s, q)
    # A caustic point cannot have a strictly positive distance lower bound.
    for z in caustic[::7]:
        assert bound(s, q, float(z.real), float(z.imag)) <= 2e-10
    # For remote sources the lower bound must never exceed even the nearest
    # independently computed caustic sample (whose distance is an upper bound
    # on true continuous clearance).
    for source in [.2+.3j, -.23+.3j, 2.+.7j, -.8-.5j]:
        lower = bound(s, q, source.real, source.imag)
        assert 0 <= lower <= np.min(abs(caustic-source)) + 2e-10


def test_remote_ghost_veto_is_resolved_without_detailed_caustic():
    lcbinint = pytest.importorskip("lcbinint")
    vbm_module = pytest.importorskip("VBMicrolensing")
    params = dict(t0=0., u0=.3, tE=1., alpha=0., rho=.003, q=.01, s=1.2)
    curve = lcbinint.LightCurve(lens="binary", options=lcbinint.Options(
        param_type="vbm", nbin="auto", finite_source_tol=1e-3, finite_source_reltol=1e-3))
    times = np.linspace(-1, 1, 201)
    info = curve.info(times, params)
    assert set(info.finite_source_method_names) == {"point_source"}
    # Preserve the calibrated ghost veto. The alternate topology witness,
    # rather than a weakened constant, permits this remote point.
    assert info.point_source_safety_flags[77] & 2 == 0
    assert math.isinf(info.caustic_distances[77])
    assert lcbinint._native._binary_caustic_clearance_lower_bound(
        1.2, 100., -.23, .3) > 10*params["rho"]
    reference = vbm_module.VBMicrolensing()
    reference.Tol = reference.RelTol = 1e-3
    vp = [math.log(1.2), math.log(.01), .3, 0., math.log(.003), 0., 0.]
    expected = reference.BinaryLightCurve(vp, times.tolist())[0]
    np.testing.assert_allclose(info.magnifications, expected, rtol=2e-12, atol=2e-12)


def test_lightcurve_api_shares_selector_and_reports_uncertified_support():
    lcbinint = pytest.importorskip("lcbinint")

    curve = lcbinint.LightCurve(lens="binary", options=lcbinint.Options(
        param_type="vbm", nbin="auto", finite_source_tol=1e-3, finite_source_reltol=1e-3))
    params = dict(t0=0., u0=.3, tE=1., alpha=0., rho=.003, q=.01, s=1.2)
    times = np.linspace(-1, 1, 201)
    result = curve(times, params)
    diagnostic = result.diagnostics
    np.testing.assert_allclose(
        result.magnifications, diagnostic["mu"], rtol=2e-12, atol=2e-12
    )
    scalar_result = curve(0., params)
    assert scalar_result.magnifications.shape == (1,)
    # Changing the source/lens parameters must refresh the root and geometry
    # state. Unproven support keeps a finite estimate and is reported by status.
    crossing = dict(params, u0=.03, alpha=-.08)
    crossing_times = np.linspace(-.2, .8, 201)
    crossing_result = curve(crossing_times, crossing)
    assert np.any(crossing_result.diagnostics["integrator_used"])
    assert not np.all(crossing_result.value_converged)
    assert np.all(np.isfinite(crossing_result.magnifications))
    assert np.any(crossing_result.status != lcbinint.LightCurveInfo.OK)


def test_lightcurve_api_exposes_algebraic_boundary_values_and_support_diagnostics():
    lcbinint = pytest.importorskip("lcbinint")
    options = lcbinint.Options(
        param_type="vbm",
        nbin="auto",
        finite_source_tol=1e-3,
        finite_source_reltol=1e-3,
        finite_source_backend="algebraic_boundary",
    )
    curve = lcbinint.LightCurve(lens="binary", options=options)
    assert curve.options.finite_source_backend == "algebraic_boundary"
    assert not curve.supports_fused_light_curve_likelihood

    params = dict(t0=0., u0=.03, tE=1., alpha=-.08, rho=.003, q=.01, s=1.2)
    times = np.linspace(-.2, .8, 201)
    info = curve(times, params)

    assert np.any(info.diagnostics["integrator_used"])
    assert "algebraic_boundary" in set(info.finite_source_method_name)
    assert np.any(~info.value_converged)
    assert np.any(~info.support_certified)
    assert np.all(np.isfinite(info.magnifications))
    np.testing.assert_allclose(
        info.magnifications, info.diagnostics["mu"], rtol=2e-12, atol=2e-12
    )
    with pytest.raises(NotImplementedError, match="magnification_batch"):
        curve.magnification_batch(times[:2], [params])


def test_algebraic_boundary_backend_integrates_with_public_jax_grad():
    lcbinint = pytest.importorskip("lcbinint")
    jax = pytest.importorskip("jax")
    lcbinint_jax = pytest.importorskip("lcbinint_jax")
    if not lcbinint_jax.cpp_algebraic_boundary_batch_ffi_available():
        pytest.skip("lcbinint was built without the algebraic-boundary JAX FFI")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    with pytest.raises(ValueError, match="finite_source_backend"):
        lcbinint.Options(finite_source_backend="holonomic")
    options = lcbinint.Options(
        jax=True,
        param_type="vbm",
        finite_source_tol=1e-3,
        finite_source_reltol=1e-3,
        finite_source_backend="algebraic_boundary",
    )
    curve = lcbinint.LightCurve(lens="binary", options=options)
    assert curve.options.jax

    params = dict(t0=0., u0=.3, tE=10., alpha=.5, rho=.003, q=.01, s=1.2)
    times = jnp.asarray([-.6], dtype=jnp.float64)
    values = np.asarray(curve(times, params).magnifications)
    gradient = float(jax.grad(
        lambda u0: jnp.sum(curve(times, dict(params, u0=u0)))
    )(jnp.asarray(params["u0"], dtype=jnp.float64)))

    native_curve = lcbinint.LightCurve(
        lens="binary",
        options=lcbinint.Options(
            param_type="vbm",
            nbin="auto",
            finite_source_tol=1e-3,
            finite_source_reltol=1e-3,
            finite_source_backend="algebraic_boundary",
        ),
    )
    native_result = native_curve(np.asarray(times), params)
    native = native_result.diagnostics
    assert native["finite_source_method_name"] == ["point_source"]
    np.testing.assert_allclose(values, native["mu"], rtol=2e-12, atol=2e-12)
    step = 1e-4
    plus = native_curve(
        np.asarray(times), dict(params, u0=params["u0"] + step)
    ).magnifications[0]
    minus = native_curve(
        np.asarray(times), dict(params, u0=params["u0"] - step)
    ).magnifications[0]
    finite_difference = (plus - minus) / (2 * step)
    assert np.isfinite(gradient)
    np.testing.assert_allclose(gradient, finite_difference, rtol=2e-5, atol=2e-7)

    deep_options = dict(
        param_type="center_of_mass",
        nbin="auto",
        finite_source_tol=1e-4,
        finite_source_reltol=1e-4,
        finite_source_backend="algebraic_boundary",
    )
    deep_params = dict(t0=0., u0=.15, tE=1., alpha=0., rho=.05, q=.28, s=1.35)
    deep_times = jnp.asarray([.22], dtype=jnp.float64)
    deep_ld = lcbinint.LimbDarkening.linear(.4)
    deep_curve = lcbinint.LightCurve(
        lens="binary",
        options=lcbinint.Options(jax=True, **deep_options),
        limb_darkening=deep_ld,
    )
    deep_value = np.asarray(deep_curve(deep_times, deep_params))
    q_gradient = float(jax.grad(
        lambda q: jnp.sum(deep_curve(deep_times, dict(deep_params, q=q)))
    )(jnp.asarray(deep_params["q"], dtype=jnp.float64)))
    deep_native = lcbinint.LightCurve(
        lens="binary",
        options=lcbinint.Options(**deep_options),
        limb_darkening=deep_ld,
    )
    deep_result = deep_native(np.asarray(deep_times), deep_params)
    deep_info = deep_result.diagnostics
    assert deep_info["finite_source_method_name"] == ["algebraic_boundary"]
    assert deep_info["value_converged"].tolist() == [True]
    assert deep_info["support_certified"].tolist() == [True]
    np.testing.assert_allclose(
        deep_value, deep_result.magnifications, rtol=1e-8, atol=2e-8
    )
    q_step = 1e-5
    q_plus = deep_native(
        np.asarray(deep_times), dict(deep_params, q=deep_params["q"] + q_step)
    ).magnifications[0]
    q_minus = deep_native(
        np.asarray(deep_times), dict(deep_params, q=deep_params["q"] - q_step)
    ).magnifications[0]
    q_finite_difference = (q_plus - q_minus) / (2 * q_step)
    assert np.isfinite(q_gradient)
    np.testing.assert_allclose(q_gradient, q_finite_difference, rtol=2e-5, atol=2e-7)


def test_native_algebraic_boundary_lightcurve_gradient_contract():
    lcbinint = pytest.importorskip("lcbinint")

    point_curve = lcbinint.LightCurve(
        lens="binary",
        options=lcbinint.Options(
            param_type="vbm",
            nbin="auto",
            finite_source_tol=1e-3,
            finite_source_reltol=1e-3,
        ),
    )
    assert point_curve.options.finite_source_backend == "algebraic_boundary"
    point_params = dict(
        t0=0., u0=.3, tE=10., alpha=.5, rho=.003, q=.01, s=1.2
    )
    point_time = -10.
    point_value = point_curve(point_time, point_params)
    point_gradient = point_curve(
        point_time, point_params, with_gradient=True
    )
    np.testing.assert_array_equal(
        point_gradient.magnifications, point_value.magnifications
    )
    np.testing.assert_array_equal(point_gradient.status, point_value.status)
    np.testing.assert_array_equal(
        point_gradient.value_converged, point_value.value_converged
    )
    np.testing.assert_array_equal(
        point_gradient.support_certified, point_value.support_certified
    )
    assert point_gradient.gradient_parameters == (
        "t0", "u0", "tE", "alpha", "rho", "q", "s"
    )
    assert point_gradient.gradient_valid.tolist() == [True]
    steps = {
        "t0": 1e-5, "u0": 1e-5, "tE": 1e-5, "alpha": 1e-5,
        "rho": 1e-6, "q": 1e-6, "s": 1e-5,
    }
    for column, name in enumerate(point_gradient.gradient_parameters):
        step = steps[name]
        plus = point_curve(
            point_time, dict(point_params, **{name: point_params[name] + step})
        ).magnifications[0]
        minus = point_curve(
            point_time, dict(point_params, **{name: point_params[name] - step})
        ).magnifications[0]
        finite_difference = (plus - minus) / (2 * step)
        np.testing.assert_allclose(
            point_gradient.gradients[0, column], finite_difference,
            rtol=2e-5, atol=2e-7,
            err_msg=f"native algebraic-boundary derivative for {name}",
        )

    deep_curve = lcbinint.LightCurve(
        lens="binary",
        options=lcbinint.Options(
            param_type="center_of_mass",
            nbin="auto",
            finite_source_tol=1e-4,
            finite_source_reltol=1e-4,
        ),
        limb_darkening=lcbinint.LimbDarkening.linear(.4),
    )
    deep_params = dict(
        t0=0., u0=.15, tE=1., alpha=0., rho=.05, q=.28, s=1.35
    )
    deep_time = .22
    deep_value = deep_curve(deep_time, deep_params)
    deep_gradient = deep_curve(
        deep_time, deep_params, with_gradient=True
    )
    np.testing.assert_array_equal(
        deep_gradient.magnifications, deep_value.magnifications
    )
    np.testing.assert_array_equal(deep_gradient.status, deep_value.status)
    assert deep_gradient.gradient_valid.tolist() == [True]
    step = 1e-5
    plus = deep_curve(
        deep_time, dict(deep_params, q=deep_params["q"] + step)
    ).magnifications[0]
    minus = deep_curve(
        deep_time, dict(deep_params, q=deep_params["q"] - step)
    ).magnifications[0]
    finite_difference = (plus - minus) / (2 * step)
    np.testing.assert_allclose(
        deep_gradient.gradients[0, 5], finite_difference,
        rtol=2e-5, atol=2e-7,
    )

    wide_curve = lcbinint.LightCurve(
        lens="binary",
        options=lcbinint.Options(
            param_type="lcbinint",
            nbin="auto",
            finite_source_tol=1e-4,
            finite_source_reltol=1e-4,
        ),
        limb_darkening=lcbinint.LimbDarkening.linear(.4),
    )
    wide_params = dict(deep_params, alpha=.2)
    wide_gradient = wide_curve(.22, wide_params, with_gradient=True)
    assert wide_gradient.gradient_valid.tolist() == [True]
    for name, column in (("q", 5), ("s", 6)):
        step = 1e-5
        plus = wide_curve(
            .22, dict(wide_params, **{name: wide_params[name] + step})
        ).magnifications[0]
        minus = wide_curve(
            .22, dict(wide_params, **{name: wide_params[name] - step})
        ).magnifications[0]
        finite_difference = (plus - minus) / (2 * step)
        np.testing.assert_allclose(
            wide_gradient.gradients[0, column], finite_difference,
            rtol=2e-5, atol=2e-7,
            err_msg=f"wide-binary coordinate derivative for {name}",
        )

    unconverged_curve = lcbinint.LightCurve(
        lens="binary",
        options=lcbinint.Options(
            param_type="vbm",
            nbin="auto",
            finite_source_tol=1e-3,
            finite_source_reltol=1e-3,
        ),
        limb_darkening=lcbinint.LimbDarkening.linear(.5),
    )
    unconverged_params = dict(
        t0=0., u0=-1e-5, tE=10., alpha=.5, rho=.5, q=.001, s=.6
    )
    failed_time = -0.7000000000000001
    failed_value = unconverged_curve(failed_time, unconverged_params)
    failed_gradient = unconverged_curve(
        failed_time, unconverged_params, with_gradient=True
    )
    np.testing.assert_array_equal(
        failed_gradient.magnifications, failed_value.magnifications
    )
    np.testing.assert_array_equal(failed_gradient.status, failed_value.status)
    np.testing.assert_array_equal(
        failed_gradient.value_converged, failed_value.value_converged
    )
    np.testing.assert_array_equal(
        failed_gradient.support_certified, failed_value.support_certified
    )
    assert failed_gradient.status.tolist() == [lcbinint.LightCurveInfo.UNCONVERGED]
    assert np.all(np.isfinite(failed_gradient.magnifications))
    assert failed_gradient.gradient_valid.tolist() == [False]
    assert np.all(failed_gradient.gradients == 0.)


@pytest.mark.parametrize("ordering", ["forward", "reverse", "interleaved"])
def test_point_strip_route_handles_reversed_and_jumping_trajectories(ordering):
    lcbinint = pytest.importorskip("lcbinint")
    vbm_module = pytest.importorskip("VBMicrolensing")
    curve = lcbinint.LightCurve(lens="binary", options=lcbinint.Options(
        param_type="vbm", nbin="auto", finite_source_tol=1e-3, finite_source_reltol=1e-3))
    times = np.linspace(-1., 1., 201)
    if ordering == "reverse":
        times = times[::-1].copy()
    elif ordering == "interleaved":
        times = np.concatenate([times[::2], times[1::2][::-1]])
    params = dict(t0=0., u0=.3, tE=1., alpha=0., rho=.003, q=.01, s=1.2)
    diagnostic = curve._native._evaluate_algebraic_boundary(
        times, params, 1e-3, 1e-3
    )
    assert set(diagnostic["finite_source_method_name"]) == {"point_source"}
    reference = vbm_module.VBMicrolensing()
    reference.Tol = reference.RelTol = 1e-3
    vp = [math.log(1.2), math.log(.01), .3, 0., math.log(.003), 0., 0.]
    expected = reference.BinaryLightCurve(vp, times.tolist())[0]
    result = curve(times, params)
    np.testing.assert_allclose(
        result.magnifications, expected, rtol=2e-12, atol=2e-12
    )
