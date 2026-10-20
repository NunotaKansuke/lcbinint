"""Public value/tangent routing, archived counterexamples, and independent checks."""
import json
from pathlib import Path

import numpy as np
import pytest

import lcbinint as lc

ROOT = Path(__file__).resolve().parents[2]
ORDER = ("t0", "u0", "tE", "alpha", "rho", "q", "s")


def test_unresolved_outer_edge_returns_computed_gradient_candidate():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    model = lc.LightCurve(lens="binary", options=lc.Options(
        param_type="vbm", source_bins=80, max_source_bins=400,
        finite_source_tol=1e-4, finite_source_reltol=1e-3,
        finite_source_backend="algebraic_boundary"),
        limb_darkening=lc.LimbDarkening.linear(0.6584))
    params = dict(s=0.5860467981775465, q=2.9795910469372738e-5,
                  u0=4.2345762431841034e-7, alpha=2.22602246134627,
                  rho=2.6295524980988746e-6, tE=2033.082239820582,
                  t0=10068.977085148737)
    epochs = np.array([10068.966367])
    result = model(epochs, params, with_gradient=True, with_diagnostics=True)
    assert not result.gradient_valid[0]
    assert np.isfinite(result.gradients).all()
    assert np.any(result.gradients != 0)
    assert "unresolved_outer_edge_pair" in result.diagnostics["stop"][0]
    local = result.diagnostics["local_gradients"][0]
    np.testing.assert_allclose(local, [42552153555.271843, -9190385671.3383465,
        19319360648.881779, 11.177209980384188, -1148496.3863479905], rtol=1e-6)
    value, gradient = jax.jit(jax.value_and_grad(lambda rho:
        model(jnp.asarray(epochs), dict(params, rho=rho)).magnifications.sum()))(
            jnp.asarray(params["rho"]))
    assert np.isfinite(float(value)) and np.isfinite(float(gradient))
    np.testing.assert_allclose(value, result.magnifications[0], rtol=1e-3)
    np.testing.assert_allclose(gradient, result.gradients[0, ORDER.index("rho")], rtol=1e-3)


def curve(limb=0., tol=1e-3, coordinates="vbm"):
    return lc.LightCurve(lens="binary", options=lc.Options(
        param_type=coordinates, nbin="auto", finite_source_tol=tol,
        finite_source_reltol=tol, finite_source_backend="algebraic_boundary"),
        limb_darkening=lc.LimbDarkening.linear(limb))


def saved_case(number):
    return json.loads((ROOT / "tests/data/algebraic_boundary" /
                       f"case_{number:03}.json").read_text())


@pytest.mark.parametrize("limb", [0., .5, .999999])
def test_same_selector_and_value_budget_with_or_without_tangents(limb):
    model = curve(limb)
    params = dict(t0=0., u0=.03, tE=1., alpha=-.08, rho=.003, q=.01, s=1.2)
    times = np.linspace(-.2, .8, 41)
    value = model(times, params, with_diagnostics=True)
    tangent = model(times, params, with_gradient=True, with_diagnostics=True)
    np.testing.assert_array_equal(value.diagnostics["finite_source_method"],
                                  tangent.diagnostics["finite_source_method"])
    assert np.all(np.isfinite(tangent.gradients))
    # Two independently stopped quadratures each have the requested allowance.
    budget = 2e-3 * np.maximum(1., np.abs(value.magnifications))
    assert np.all(np.abs(value.magnifications-tangent.magnifications) <= budget)
    backends = value.diagnostics["integrator_backend"]
    assert "seeded_radial_frontier_nested_fejer" in backends
    assert not any("legacy" in stop for stop in tangent.diagnostics["stop"])
    assert tangent.gradient_parameters == ORDER


def test_saved_case107_and_parameter_dictionary_order():
    case = saved_case(107)
    times = np.linspace(-case["span"], case["span"], 33)*case["params"]["tE"]
    model = curve(case["limb"])
    a = model(times, case["params"], with_gradient=True, with_diagnostics=True)
    b = model(times, dict(reversed(list(case["params"].items()))),
              with_gradient=True, with_diagnostics=True)
    old = np.load(ROOT / "tests/data/algebraic_boundary/case_107_reference.npz")
    np.testing.assert_allclose(a.magnifications, old["mu"], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(a.gradients, b.gradients, rtol=1e-12, atol=1e-12)
    assert np.all(a.gradient_valid)
    assert np.all(np.isfinite(a.gradients))


def test_case117_exact_limb_and_handoff_epoch_return_candidates():
    case = saved_case(117)
    p = case["params"]
    times = -p["rho"]*p["tE"] + np.array([-1e-5, 0., 1e-5])
    result = curve(case["limb"])(times, p, with_gradient=True, with_diagnostics=True)
    assert np.all(np.isfinite(result.magnifications))
    assert np.all(np.isfinite(result.gradients))
    assert result.gradients.shape == (3, 7)
    # Geometry validity and precision certification must remain separate.
    assert np.asarray(result.gradient_valid).dtype == bool
    assert np.all(result.value_converged)
    # This was a factor-of-two mismatch in the legacy handoff. The q signal
    # has a stable stricter-value finite difference at the off-contact epoch.
    h = p["q"]*.003
    strict = curve(case["limb"], tol=1e-7)
    plus = strict(times[:1], dict(p, q=p["q"]+h))
    minus = strict(times[:1], dict(p, q=p["q"]-h))
    reference = (plus.magnifications[0]-minus.magnifications[0])/(2*h)
    assert result.gradients[0, ORDER.index("q")] == pytest.approx(reference, rel=1e-3)


def test_unproven_seed_support_keeps_finite_candidate_and_false_reliability():
    case = saved_case(10)
    times = np.linspace(-case["span"], case["span"], 33)*case["params"]["tE"]
    result = curve(case["limb"])(times, case["params"],
                                with_gradient=True, with_diagnostics=True)
    assert np.all(np.isfinite(result.gradients))
    assert np.any(~result.gradient_valid)
    assert "seed_support_unproven" in result.diagnostics["gradient_stop"]


def test_case081_preserves_smooth_disk_cancellation_repair():
    case = saved_case(81)
    times = np.linspace(-case["span"], case["span"], 33)*case["params"]["tE"]
    result = curve(case["limb"])(times[[12]], case["params"],
                                with_gradient=True, with_diagnostics=True)
    g = dict(zip(result.gradient_parameters, result.gradients[0]))
    # Independently recorded stable central-difference sequences from the handoff.
    assert g["u0"] == pytest.approx(-365.305655, rel=3e-3)
    assert g["rho"] == pytest.approx(17.967796, rel=3e-3)


def test_point_value_matches_independent_vbm_lightcurve():
    vb = pytest.importorskip("VBMicrolensing").VBMicrolensing()
    vb.Tol = vb.RelTol = 1e-3
    p = dict(t0=0., u0=.3, tE=1., alpha=0., rho=.003, q=.01, s=1.2)
    times = np.linspace(-1, 1, 41)
    result = curve()(times, p)
    vp = [np.log(p["s"]), np.log(p["q"]), p["u0"], p["alpha"],
          np.log(p["rho"]), np.log(p["tE"]), p["t0"]]
    reference = np.asarray(vb.BinaryLightCurve(vp, times.tolist())[0])
    np.testing.assert_allclose(result.magnifications, reference, rtol=2e-12, atol=2e-12)


def test_quadratic_ld_is_explicitly_unsupported():
    p = dict(t0=0., u0=.03, tE=1., alpha=-.08, rho=.003, q=.01, s=1.2,
             limb_darkening_d=.2)
    with pytest.raises(ValueError, match="d must be zero"):
        curve()([0.], p)


@pytest.mark.parametrize("jax_option", [False, True])
def test_concrete_gradient_never_rescans_or_fills_with_finite_differences(jax_option):
    model = curve(.5)
    model.options.jax = jax_option
    def unexpected(*args, **kwargs):
        pytest.fail("a concrete fused call must reuse its own gradient")
    model._evaluate_gradients = unexpected
    model._fill_missing_gradient_candidates = unexpected
    p = dict(s=1.2, q=.01, rho=.003, alpha=-.08, tE=1., u0=.03, t0=0.)
    result = model([0., .1], p, with_gradient=True)
    assert result.gradients.shape == (2, 7)
    assert np.all(np.isfinite(result.gradients))


def test_scalar_helper_uses_the_shared_controller():
    from lcbinint.algebraic_boundary import lightcurve_value, lightcurve_magnification
    model = curve(.5)
    p = dict(t0=0., u0=.03, tE=1., alpha=-.08, rho=.003, q=.01, s=1.2)
    direct = model([0.], p, with_gradient=True)
    value = lightcurve_value(model, 0., p, with_gradient=True)
    np.testing.assert_array_equal(value["gradients"], direct.gradients[0])
    assert value["gradient_precision_checked"] is False
    assert np.isfinite(lightcurve_magnification(model, 0., p))


def test_parameters_container_reuses_the_same_seven_parameter_fused_pass():
    model = curve(.5)
    def unexpected(*args, **kwargs):
        pytest.fail("Parameters defaults must not send a fused call through JAX or FD")
    model._evaluate_gradients = unexpected
    model._fill_missing_gradient_candidates = unexpected
    params = lc.Parameters(t0=0., u0=.3, tE=1., alpha=0., rho=.003, q=.01, s=1.2)
    result = model([0.], params, with_gradient=True, with_diagnostics=True)
    assert result.gradient_parameters == ORDER
    assert result.gradients.shape == (1, 7)
    assert np.all(np.isfinite(result.gradients))
    assert "point_source_safety_flags" in result.diagnostics


@pytest.mark.parametrize("coordinates", ["vbm", "center_of_mass", "lcbinint"])
@pytest.mark.parametrize("rho", [0., .003])
def test_jax_and_native_use_same_value_and_derivative_conventions(coordinates, rho):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from lcbinint_jax import cpp_algebraic_boundary_batch_ffi_available
    if not cpp_algebraic_boundary_batch_ffi_available():
        pytest.skip("JAX CPU FFI is disabled")
    jax.config.update("jax_enable_x64", True)
    p = dict(t0=0., u0=.03, tE=1., alpha=-.08, rho=rho, q=.01, s=1.2)
    times = jnp.array([-.05, 0., .1])
    model = curve(.5, coordinates=coordinates)
    native = model(np.asarray(times), p, with_gradient=True)
    def evaluate(x):
        return model(times, dict(p, u0=x)).magnifications
    values, gradients = jax.jvp(evaluate, (jnp.asarray(p["u0"]),), (jnp.asarray(1.),))
    np.testing.assert_allclose(values, native.magnifications, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(gradients, native.gradients[:, 1], rtol=2e-7, atol=2e-7)


@pytest.mark.parametrize("key", ["mu", "gradients"])
def test_nonfinite_candidates_raise_instead_of_returning_nan(monkeypatch, key):
    from types import SimpleNamespace
    from lcbinint import algebraic_boundary as backend
    fake = SimpleNamespace(_native=SimpleNamespace(
        _evaluate_algebraic_boundary=lambda *args: {key: np.array([np.nan])}))
    monkeypatch.setattr(backend, "_validated_value_request",
        lambda *args: (np.array([0.]), {}, 1e-3, 1e-3, False))
    with pytest.raises(RuntimeError, match="no finite " + key + " candidate"):
        backend.lightcurve_value(fake, [0.], with_gradient=key == "gradients")


def test_smooth_contact_repair_preserves_q_and_radius_derivatives():
    params = dict(t0=0., tE=10., u0=-.01, alpha=180., s=1.5, q=.01, rho=.001)
    result = curve(.5, tol=1e-4)([.076], params, with_gradient=True)
    strict = curve(.5, tol=1e-7)
    for name, step in (("q", 1e-6), ("rho", 1e-7)):
        plus = strict([.076], dict(params, **{name: params[name] + step}))
        minus = strict([.076], dict(params, **{name: params[name] - step}))
        reference = (plus.magnifications[0] - minus.magnifications[0]) / (2*step)
        assert result.gradients[0, ORDER.index(name)] == pytest.approx(reference, rel=3e-5)


@pytest.mark.parametrize("with_gradient", [False, True])
def test_ordinary_api_selects_the_algebraic_boundary_controller(with_gradient):
    params = dict(t0=0.0, u0=0.03, tE=1.0, alpha=-0.08, rho=0.003, q=0.01, s=1.2)
    times = np.array([0.0, 0.1])
    automatic = lc.LightCurve(options=lc.Options(nbin="auto"))
    explicit = curve()
    assert automatic.options.finite_source_backend == "algebraic_boundary"
    actual = automatic(
        times, params, with_gradient=with_gradient, with_diagnostics=True
    )
    expected = explicit(
        times, params, with_gradient=with_gradient, with_diagnostics=True
    )
    np.testing.assert_array_equal(actual.magnifications, expected.magnifications)
    assert (
        "seeded_radial_frontier_nested_fejer"
        in actual.diagnostics["integrator_backend"]
    )
    if with_gradient:
        np.testing.assert_array_equal(actual.gradients, expected.gradients)
        np.testing.assert_array_equal(actual.gradient_valid, expected.gradient_valid)


@pytest.mark.parametrize(
    "parameters,t0",
    [
        ({"t0": -2.0, "t_0": 1.0}, 1.0),
        ({"t_0": 1.0, "t0": -2.0}, -2.0),
    ],
)
def test_default_backend_preserves_parameter_alias_order(parameters, t0):
    model = lc.LightCurve()
    params = dict(u0=0.03, tE=1.0, alpha=-0.08, rho=0.003, q=0.01, s=1.2)
    actual = model([-0.1, 0.1], dict(params, **parameters), with_gradient=True)
    expected = model([-0.1, 0.1], dict(params, t0=t0), with_gradient=True)
    np.testing.assert_array_equal(actual.magnifications, expected.magnifications)
    np.testing.assert_array_equal(actual.gradients, expected.gradients)


def test_default_backend_jit_matches_native_values_and_tangents():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    model = lc.LightCurve()
    params = dict(t0=0.0, u0=0.03, tE=1.0, alpha=-0.08, rho=0.003, q=0.01, s=1.2)
    times = jnp.array([-0.05, 0.0, 0.1])
    native = model(np.asarray(times), params, with_gradient=True)

    def values(u0):
        return model(times, dict(params, u0=u0)).magnifications

    actual, derivative = jax.jit(
        lambda u0: jax.jvp(values, (u0,), (jnp.ones_like(u0),))
    )(params["u0"])
    np.testing.assert_allclose(actual, native.magnifications, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(derivative, native.gradients[:, 1], rtol=2e-7, atol=2e-7)


@pytest.mark.parametrize("with_gradient", [False, True])
def test_converged_value_without_coverage_is_not_a_seed_failure(with_gradient):
    case = saved_case(117)
    result = curve(.5)([0.], case["params"], with_gradient=with_gradient)
    assert result.status_names.tolist() == ["coverage_uncertified"]
    assert result.all_values_converged
    assert not result.all_converged
    assert np.all(result.seed_support_proven)
    assert np.all(result.support_required)
    assert not np.any(result.support_certified)
    assert result.diagnostics is None


def test_value_failure_takes_priority_over_uncertified_coverage():
    params = dict(s=.6, q=1e-4, rho=.05, t0=0., tE=10., u0=-1e-5, alpha=.5)
    result = curve(.5, tol=1e-7)([0.], params, with_gradient=True,
                                with_diagnostics=True)
    assert result.status_names.tolist() == ["unconverged"]
    assert not result.all_values_converged
    assert not result.all_converged
    assert not np.any(result.support_certified)
    assert "InnerAccuracyLimited" in result.diagnostics["stop"][0]


def test_seed_failure_remains_distinct_from_missing_coverage_certificate():
    case = saved_case(10)
    times = np.linspace(-case["span"], case["span"], 33)*case["params"]["tE"]
    result = curve(case["limb"])(times, case["params"], with_gradient=True)
    missing = np.asarray(result.support_required) & ~np.asarray(result.seed_support_proven)
    assert np.any(missing)
    np.testing.assert_array_equal(result.status[missing], lc.LightCurveInfo.SUPPORT_UNPROVEN)


def test_support_metadata_survives_jax_pytree_roundtrip():
    jax = pytest.importorskip("jax")
    case = saved_case(117)
    result = curve(.5)([0.], case["params"])
    # The public gradient path registers this record as a pytree.
    lc._register_lightcurve_info_pytree()
    restored = jax.jit(lambda record: record)(result)
    for name in ("status", "value_converged", "support_certified",
                 "seed_support_proven", "frontier_discovery_completed", "support_required"):
        np.testing.assert_array_equal(getattr(restored, name), getattr(result, name))


@pytest.mark.parametrize("limb", [0., .5])
@pytest.mark.parametrize("with_gradient", [False, True])
def test_default_call_preserves_reliability_without_detailed_diagnostics(limb, with_gradient):
    model = curve(limb)
    params = dict(t0=0., u0=.03, tE=1., alpha=-.08, rho=.003, q=.01, s=1.2)
    times = np.linspace(-.2, .8, 41)
    ordinary = model(times, params, with_gradient=with_gradient)
    detailed = model(times, params, with_gradient=with_gradient, with_diagnostics=True)
    assert ordinary.diagnostics is None
    assert "seconds" in detailed.diagnostics
    assert np.any(detailed.support_required)
    assert np.any(~detailed.support_certified[detailed.support_required])
    for field in ("magnifications", "status", "value_converged", "support_certified",
                  "support_required", "seed_support_proven", "frontier_discovery_completed"):
        np.testing.assert_array_equal(getattr(ordinary, field), getattr(detailed, field))
    if with_gradient:
        np.testing.assert_array_equal(ordinary.gradients, detailed.gradients)
        np.testing.assert_array_equal(ordinary.gradient_valid, detailed.gradient_valid)


def test_value_only_native_result_omits_optional_diagnostic_arrays():
    from lcbinint.algebraic_boundary import lightcurve_value
    model = curve(.5)
    params = dict(t0=0., u0=.03, tE=1., alpha=-.08, rho=.003, q=.01, s=1.2)
    compact = lightcurve_value(model, 0., params, with_diagnostics=False)
    detailed = lightcurve_value(model, 0., params)
    for field in ("mu", "value_converged", "fully_converged", "support_certified",
                  "seed_support_proven", "frontier_discovery_completed", "integrator_used"):
        np.testing.assert_array_equal(compact[field], detailed[field])
    assert not {"seconds", "stop", "node_evaluations", "error_estimate"} & compact.keys()
    assert {"seconds", "stop", "node_evaluations", "error_estimate"} <= detailed.keys()
