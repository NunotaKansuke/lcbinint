"""Public value/tangent routing, archived counterexamples, and independent checks."""
import json
from pathlib import Path

import numpy as np
import pytest

import lcbinint as lc

ROOT = Path(__file__).resolve().parents[2]
ORDER = ("t0", "u0", "tE", "alpha", "rho", "q", "s")


@pytest.mark.parametrize("relative_tolerance", [1e-3, 1e-7, 1e-9])
@pytest.mark.parametrize("limb", [0., .6101, .6584])
@pytest.mark.parametrize("radius_factor", [.999, 1., 1.001])
@pytest.mark.parametrize("source_sign", [-1., 1.])
def test_edge_pair_transverse_gradient_matches_value_difference(
        relative_tolerance, limb, radius_factor, source_sign):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from lcbinint_jax import binary_algebraic_boundary_batch_ffi

    jax.config.update("jax_enable_x64", True)
    point = jnp.asarray([-6.3359522138426355e-05, source_sign*1.9809194818738246e-09,
                        .63244064975758063, 704.16939274402819,
                        .00031754130897469404*radius_factor])

    def value(parameters, tolerance):
        return binary_algebraic_boundary_batch_ffi(
            parameters[:1], parameters[1:2], parameters[2], parameters[3], parameters[4],
            limb_c=limb, absolute_tolerance=1e-10,
            relative_tolerance=tolerance).sum()

    magnitude, gradient = jax.jit(jax.value_and_grad(
        lambda parameters: value(parameters, relative_tolerance)))(point)
    scalar = jax.jit(lambda parameters: value(parameters, 1e-9))
    assert np.isfinite(float(magnitude)) and np.isfinite(np.asarray(gradient)).all()
    for step in [2e-5, 1e-5]:
        delta = jnp.asarray([0., step, 0., 0., 0.])
        stencil = [float(scalar(point+offset*delta)) for offset in [-2, -1, 1, 2]]
        reference = (stencil[0]-8*stencil[1]+8*stencil[2]-stencil[3])/(12*step)
        np.testing.assert_allclose(gradient[1], reference, rtol=.02, atol=0)


@pytest.mark.parametrize("radius_factor", [.999, 1., 1.001])
@pytest.mark.parametrize("limb", [0., .6584])
@pytest.mark.parametrize("relative_tolerance", [1e-3, 1e-7])
def test_spurious_arc_topology_recovery_matches_value_and_tangent(
        radius_factor, limb, relative_tolerance):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from lcbinint_jax import binary_algebraic_boundary_batch_ffi

    jax.config.update("jax_enable_x64", True)
    point = jnp.asarray([-6.3359522138426355e-05, 1.9809194818738246e-09,
                        .63244064975758063, 704.16939274402819,
                        .00031754130897469404*radius_factor])

    def value(parameters):
        return binary_algebraic_boundary_batch_ffi(
            parameters[:1], parameters[1:2], parameters[2], parameters[3], parameters[4],
            limb_c=limb, absolute_tolerance=1e-4,
            relative_tolerance=relative_tolerance).sum()

    scalar = jax.jit(value)(point)
    magnitude, gradient = jax.jit(jax.value_and_grad(value))(point)
    assert np.isfinite(float(scalar)) and float(scalar) > 0
    assert np.isfinite(np.asarray(gradient)).all()
    np.testing.assert_allclose(scalar, magnitude, rtol=2*relative_tolerance, atol=2e-4)


@pytest.mark.parametrize("radius_factor", [.999, 1., 1.001])
@pytest.mark.parametrize("limb", [0., .6584])
@pytest.mark.parametrize("relative_tolerance", [1e-7, 1e-9])
def test_grazing_arc_endpoint_precision_matches_physical_integrand(
        radius_factor, limb, relative_tolerance):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from lcbinint_jax import binary_algebraic_boundary_batch_ffi

    jax.config.update("jax_enable_x64", True)
    point = jnp.asarray([.0001763733600786604, .00025841859275296266,
                        .63860976638688738, 685.48439331981217,
                        .00030760727910230425*radius_factor])

    def value(parameters):
        return binary_algebraic_boundary_batch_ffi(
            parameters[:1], parameters[1:2], parameters[2], parameters[3], parameters[4],
            limb_c=limb, absolute_tolerance=1e-10,
            relative_tolerance=relative_tolerance).sum()

    magnitude, gradient = jax.jit(jax.value_and_grad(value))(point)
    scalar = jax.jit(value)(point)
    assert np.isfinite(float(magnitude)) and float(magnitude) > 0
    assert np.isfinite(np.asarray(gradient)).all()
    np.testing.assert_allclose(scalar, magnitude, rtol=2*relative_tolerance, atol=2e-10)


def test_uncertified_grazing_reseed_preserves_better_integrated_candidate():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from lcbinint_jax import binary_algebraic_boundary_batch_ffi

    jax.config.update("jax_enable_x64", True)
    point = jnp.asarray([.0001763733600786604, .00025841859275296266,
                        .63860976638688738, 685.48439331981217, .00030760727910230425])

    def value(parameters):
        return binary_algebraic_boundary_batch_ffi(
            parameters[:1], parameters[1:2], parameters[2], parameters[3], parameters[4],
            limb_c=.6584, absolute_tolerance=1e-10, relative_tolerance=1e-9).sum()

    scalar = jax.jit(value)
    magnitude, gradient = jax.jit(jax.value_and_grad(value))(point)
    np.testing.assert_allclose(magnitude, scalar(point), rtol=2e-9, atol=2e-10)
    for step in [1e-6, 5e-7]:
        delta = jnp.zeros(5).at[2].set(step)
        stencil = [float(scalar(point+offset*delta)) for offset in [-2, -1, 1, 2]]
        reference = (stencil[0]-8*stencil[1]+8*stencil[2]-stencil[3])/(12*step)
        np.testing.assert_allclose(gradient[2], reference, rtol=.02, atol=0)


@pytest.mark.parametrize("radius_factor", [.5, 1., 2.])
@pytest.mark.parametrize("relative_tolerance", [1e-3, 1e-7])
def test_value_only_precise_retry_matches_tangent_without_requesting_gradient(
        radius_factor, relative_tolerance):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from lcbinint_jax import binary_algebraic_boundary_batch_ffi

    jax.config.update("jax_enable_x64", True)
    point = jnp.asarray([9.3455853692711944e-8, -1.3751070916121869e-7,
                         .53460147483028242, 15499133.519658864,
                         6.6117332425490384e-9*radius_factor])

    def value(parameters):
        return binary_algebraic_boundary_batch_ffi(
            parameters[:1], parameters[1:2], parameters[2], parameters[3], parameters[4],
            limb_c=.6101, absolute_tolerance=1e-4,
            relative_tolerance=relative_tolerance).sum()

    primal = float(jax.jit(value)(point))
    tangent, gradient = jax.jit(jax.value_and_grad(value))(point)
    assert np.isfinite(primal) and primal > 0
    assert np.isfinite(np.asarray(gradient)).all()
    np.testing.assert_allclose(primal, tangent, rtol=2*relative_tolerance, atol=2e-4)


@pytest.mark.parametrize("radius_factor", [.99, 1., 1.01])
def test_precise_frontier_count_retains_extreme_image_candidate(radius_factor):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from lcbinint_jax import binary_algebraic_boundary_batch_ffi

    jax.config.update("jax_enable_x64", True)
    point = jnp.asarray([9.3455853692711944e-8, -1.3751070916121869e-7,
                         .53460147483028242, 15499133.519658864,
                         6.6117332425490384e-9*radius_factor])

    def value(parameters):
        return binary_algebraic_boundary_batch_ffi(
            parameters[:1], parameters[1:2], parameters[2], parameters[3], parameters[4],
            limb_c=.6101, absolute_tolerance=1e-4, relative_tolerance=1e-3).sum()

    magnitude, gradient = jax.jit(jax.value_and_grad(value))(point)
    assert float(magnitude) > 0
    assert np.isfinite(float(magnitude)) and np.isfinite(np.asarray(gradient)).all()


@pytest.mark.parametrize("radius_factor", [.99, 1., 1.01])
def test_regular_retry_carries_detected_edge_fold_into_rho_derivative(radius_factor):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from lcbinint_jax import binary_algebraic_boundary_batch_ffi

    jax.config.update("jax_enable_x64", True)
    radius = 1.9744542129953203e-6*radius_factor

    def value(source_radius):
        return binary_algebraic_boundary_batch_ffi(
            jnp.asarray([-4.4264421197973816e-7]), jnp.asarray([-9.248813492409009e-9]),
            .57918659571672504, 44218.353678937485, source_radius,
            limb_c=.6584, absolute_tolerance=1e-4, relative_tolerance=1e-7).sum()

    evaluate = jax.jit(jax.value_and_grad(value))
    magnitude, derivative = evaluate(jnp.asarray(radius))
    assert np.isfinite(float(magnitude)) and np.isfinite(float(derivative))
    step = radius*.03
    magnitudes = []
    for offset in [-2, -1, 1, 2]:
        candidate, candidate_derivative = evaluate(jnp.asarray(radius+offset*step))
        assert np.isfinite(float(candidate_derivative))
        magnitudes.append(float(candidate))
    reference = (magnitudes[0]-8*magnitudes[1]+8*magnitudes[2]-magnitudes[3])/(12*step)
    np.testing.assert_allclose(derivative, reference, rtol=2e-3, atol=0)


@pytest.mark.parametrize("radius_factor", [.999, 1., 1.001])
@pytest.mark.parametrize("point,expected", [
    ([-4.4264421197973816e-7, -9.248813492409009e-9, .57918659571672504,
      44218.353678937485, 1.9744542129953203e-6], 138742.),
    ([2.1818342880858221e-6, -3.2406334809690517e-6, .55324440915935458,
      40688.138518897671, 1.5846510113090483e-6], 209685.),
])
def test_refined_frontier_endpoints_preserve_integral_candidate(radius_factor, point, expected):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from lcbinint_jax import binary_algebraic_boundary_batch_ffi

    jax.config.update("jax_enable_x64", True)
    parameters = jnp.asarray(point).at[4].multiply(radius_factor)

    def value(point):
        return binary_algebraic_boundary_batch_ffi(point[:1], point[1:2],
            point[2], point[3], point[4], limb_c=.6584,
            absolute_tolerance=1e-4, relative_tolerance=1e-3).sum()

    magnitude, gradient = jax.jit(jax.value_and_grad(value))(parameters)
    np.testing.assert_allclose(magnitude, expected, rtol=1e-3)
    assert np.isfinite(np.asarray(gradient)).all()
    assert np.all(np.asarray(gradient) != 0)


def test_failed_gaussian_replay_returns_full_integrated_tangent():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    model = lc.LightCurve(lens="binary", options=lc.Options(
        param_type="vbm", source_bins=80, max_source_bins=400,
        finite_source_tol=1e-4, finite_source_reltol=1e-3,
        finite_source_backend="algebraic_boundary"),
        limb_darkening=lc.LimbDarkening.linear(.6584))
    params = dict(s=.5871802887641477, q=3.0467643042352905e-5,
                  u0=3.816215217738804e-7, alpha=2.2231984683456436,
                  rho=2.74595032535453e-6, tE=1948.7502783769364,
                  t0=10068.977111052587)
    epochs = np.array([10068.966367])
    result = model(epochs, params, with_gradient=True, with_diagnostics=True)
    np.testing.assert_allclose(result.magnifications, [163624.77605788625], rtol=1e-6)
    assert np.isfinite(result.gradients).all()
    assert np.all(np.asarray(result.diagnostics["local_gradients"]) != 0)
    assert not result.support_certified[0]
    value, gradient = jax.jit(jax.value_and_grad(lambda rho:
        model(jnp.asarray(epochs), dict(params, rho=rho)).magnifications.sum()))(
            jnp.asarray(params["rho"]))
    np.testing.assert_allclose(value, result.magnifications[0], rtol=1e-3)
    np.testing.assert_allclose(gradient, result.gradients[0, ORDER.index("rho")], rtol=1e-3)


@pytest.mark.parametrize("position", [
    [-0.5343556322225544, 0.060577536394565853, 0.0008609241753183762,
     2.22602246134627, 0.005346096482560667, 7.617308265255118, 10068.977085148737],
    [-0.5343554660567305, 0.060577508429898036, 0.0008609162347723387,
     2.226021650697565, 0.005346084369817453, 7.617305091849514, 10068.977085155946],
])
@pytest.mark.parametrize("epoch,limb,expected", [
    (10068.978004, .6584, 102885.),
    (10068.936543, .6101, 28784.5),
    (10068.927908, .6101, 27264.9),
])
def test_failed_panel_does_not_discard_remaining_integral(position, epoch, limb, expected):
    timescale = np.exp(position[5])
    params = dict(s=np.exp(position[0]), q=position[1]/timescale,
                  u0=position[2]/timescale, alpha=position[3],
                  rho=position[4]/timescale, tE=timescale, t0=position[6])
    model = lc.LightCurve(lens="binary", options=lc.Options(
        param_type="vbm", source_bins=80, max_source_bins=400,
        finite_source_tol=1e-4, finite_source_reltol=1e-3,
        finite_source_backend="algebraic_boundary"),
        limb_darkening=lc.LimbDarkening.linear(limb))
    value = model([epoch], params, with_diagnostics=True)
    tangent = model([epoch], params, with_gradient=True, with_diagnostics=True)
    np.testing.assert_allclose(value.magnifications, [expected], rtol=1e-3)
    np.testing.assert_allclose(value.magnifications, tangent.magnifications, rtol=1e-3)
    assert np.isfinite(tangent.gradients).all()
    assert not value.support_certified[0]


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
    np.testing.assert_allclose(local[2], result.gradients[0, ORDER.index("rho")], rtol=1e-12)
    vector = jnp.asarray([params[name] for name in ORDER])
    value, gradient = jax.jit(jax.value_and_grad(lambda values:
        model(jnp.asarray(epochs), dict(zip(ORDER, values))).magnifications.sum()))(vector)
    assert np.isfinite(float(value)) and np.isfinite(np.asarray(gradient)).all()
    np.testing.assert_allclose(value, result.magnifications[0], rtol=1e-3)
    np.testing.assert_allclose(gradient, result.gradients[0], rtol=1e-3)


@pytest.mark.parametrize("prime_cache", [False, True])
def test_thin_image_separation_gradient_matches_lcbinint_difference(prime_cache):
    model = lc.LightCurve(lens="binary", options=lc.Options(
        param_type="vbm", source_bins=80, max_source_bins=400,
        finite_source_tol=1e-4, finite_source_reltol=1e-3,
        finite_source_backend="algebraic_boundary"),
        limb_darkening=lc.LimbDarkening.linear(0.6584))
    params = dict(s=0.5860467981775465, q=2.9795910469372738e-5,
                  u0=4.2345762431841034e-7, alpha=2.22602246134627,
                  rho=2.6295524980988746e-6, tE=2033.082239820582,
                  t0=10068.977085148737)
    if prime_cache:
        model([10068.955876], params)
    result = model([10068.978004], params, with_gradient=True)
    reference = lc.LightCurve(lens="binary", options=lc.Options(
        param_type="vbm", source_bins=80, max_source_bins=400,
        finite_source_tol=1e-6, finite_source_reltol=1e-3,
        finite_source_backend="algebraic_boundary"),
        limb_darkening=lc.LimbDarkening.linear(0.6584))
    estimates = []
    for step in (.002, .0012, .0006, .0004):
        lower = reference([10068.978004], dict(params, s=params["s"]-step))
        upper = reference([10068.978004], dict(params, s=params["s"]+step))
        assert lower.value_converged[0] and upper.value_converged[0]
        estimates.append((upper.magnifications[0]-lower.magnifications[0])/(2*step))
    np.testing.assert_allclose(estimates, np.full(len(estimates), estimates[0]), rtol=2e-3)
    np.testing.assert_allclose(result.gradients[0, ORDER.index("s")], estimates, rtol=2e-3)


@pytest.mark.parametrize("epoch", [10068.90835, 10068.902049])
def test_thin_image_radius_gradient_matches_lcbinint_difference(epoch):
    model = lc.LightCurve(lens="binary", options=lc.Options(
        param_type="vbm", source_bins=80, max_source_bins=400,
        finite_source_tol=1e-4, finite_source_reltol=1e-3,
        finite_source_backend="algebraic_boundary"),
        limb_darkening=lc.LimbDarkening.linear(0.6101))
    params = dict(s=0.5860467981775465, q=2.9795910469372738e-5,
                  u0=4.2345762431841034e-7, alpha=2.22602246134627,
                  rho=2.6295524980988746e-6, tE=2033.082239820582,
                  t0=10068.977085148737)
    result = model([epoch], params, with_gradient=True)
    reference = lc.LightCurve(lens="binary", options=lc.Options(
        param_type="vbm", source_bins=80, max_source_bins=400,
        finite_source_tol=1e-6, finite_source_reltol=1e-6,
        finite_source_backend="algebraic_boundary"),
        limb_darkening=lc.LimbDarkening.linear(0.6101))
    estimates = []
    for fraction in (.02, .01, .005):
        step = fraction*params["rho"]
        lower = reference([epoch], dict(params, rho=params["rho"]-step))
        upper = reference([epoch], dict(params, rho=params["rho"]+step))
        assert lower.value_converged[0] and upper.value_converged[0]
        estimates.append((upper.magnifications[0]-lower.magnifications[0])/(2*step))
    np.testing.assert_allclose(estimates, np.full(len(estimates), estimates[0]), rtol=2e-3)
    np.testing.assert_allclose(result.gradients[0, ORDER.index("rho")], estimates, rtol=2e-3)


@pytest.mark.parametrize("geometry,steps", [
    ([-6.223668442569351e-7, 9.431949650091298e-8,
      .5869060472959381, 33743.839035992525, 2.593160843075081e-6], (.008, .004, .002)),
    ([-6.1545940024148303e-7, 1.0262576006676361e-7,
      .58581547234577547, 32847.071943362222, 2.656870730615185e-6], (.006, .003, .002, .0012)),
])
def test_physical_fold_roots_rescue_retains_com_tangent(geometry, steps):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from lcbinint_jax import binary_algebraic_boundary_batch_ffi

    jax.config.update("jax_enable_x64", True)
    point = jnp.asarray(geometry)

    def info(geometry):
        return binary_algebraic_boundary_batch_ffi(
            geometry[:1], geometry[1:2], geometry[2], geometry[3], geometry[4],
            limb_c=.6584, return_info=True)

    def value(geometry):
        return info(geometry).magnification.sum()

    magnitude, gradient = jax.jit(jax.value_and_grad(value))(point)
    assert float(magnitude)>0 and np.isfinite(np.asarray(gradient)).all()
    scalar = jax.jit(info)
    estimates = []
    for step in steps:
        offset = jnp.asarray([0., 0., step, 0., 0.])
        lower, upper = scalar(point-offset), scalar(point+offset)
        assert bool(lower.value_converged[0]) and bool(upper.value_converged[0])
        estimates.append(float((upper.magnification[0]-lower.magnification[0])/(2*step)))
    np.testing.assert_allclose(estimates, np.full(len(estimates), estimates[0]), rtol=2e-3)
    np.testing.assert_allclose(float(gradient[2]), estimates, rtol=2e-3)


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


def test_case010_exact_plan_resolves_seed_support_without_discarding_candidates():
    case = saved_case(10)
    times = np.linspace(-case["span"], case["span"], 33)*case["params"]["tE"]
    result = curve(case["limb"])(times, case["params"],
                                with_gradient=True, with_diagnostics=True)
    assert np.all(np.isfinite(result.gradients))
    assert np.all(result.seed_support_proven)
    assert np.any(result.support_required)


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
    assert "gradient_precision_checked" not in value
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


def test_seed_failure_remains_distinct_from_missing_coverage_certificate(monkeypatch):
    import lcbinint.algebraic_boundary as boundary

    case = saved_case(117)
    model = curve(case["limb"])
    times = np.array([0.0])
    raw = boundary.lightcurve_value(model, times, case["params"],
                                    with_gradient=True, with_diagnostics=True)
    raw["seed_support_proven"] = np.zeros_like(raw["seed_support_proven"], dtype=bool)
    raw["gradient_valid"] = np.zeros_like(raw["gradient_valid"], dtype=bool)
    raw["fully_converged"] = np.zeros_like(raw["fully_converged"], dtype=bool)
    monkeypatch.setattr(boundary, "lightcurve_value", lambda *args, **kwargs: dict(raw))
    result = model(times, case["params"], with_gradient=True)
    missing = np.asarray(result.support_required) & ~np.asarray(result.seed_support_proven)
    assert np.any(missing)
    np.testing.assert_array_equal(result.status[missing], lc.LightCurveInfo.SUPPORT_UNPROVEN)
    np.testing.assert_array_equal(result.magnifications, raw["mu"])
    np.testing.assert_array_equal(result.gradients, raw["gradients"])
    assert not np.any(result.gradient_valid)


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


@pytest.mark.parametrize("with_gradient", [False, True])
def test_missing_integral_candidate_never_returns_initialized_zero(with_gradient):
    model = lc.LightCurve(options=lc.Options(
        caustic_bins=1400, coordinates="center_of_mass", nbin="auto"))
    params = dict(t0=0., tE=1., alpha=0., s=1.0969276945030313,
                  q=0.00010573716122179744, rho=0.015630246022285084,
                  u0=0.0002088807679604192)
    try:
        result = model([0.00037639405005819387], params,
                       with_gradient=with_gradient, with_diagnostics=True)
    except RuntimeError as error:
        expected = ("no finite integral/derivative candidate" if with_gradient
                    else "no finite value candidate")
        assert expected in str(error)
        return
    # Future solvers may recover this integral; an uncomputed zero is never
    # an acceptable replacement for either a candidate or an explicit error.
    assert np.all(np.isfinite(result.magnifications))
    assert np.all(result.magnifications > 0.)
    assert np.all(result.diagnostics["node_evaluations"] > 0)
    if with_gradient:
        assert np.all(np.isfinite(result.gradients))
        assert np.any(result.gradients != 0.)
