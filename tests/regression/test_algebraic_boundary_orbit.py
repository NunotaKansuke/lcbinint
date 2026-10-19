"""Dynamic geometry, fused trajectory derivatives, and the vector separation ABI."""
import json
from pathlib import Path

import numpy as np
import pytest
import lcbinint as lc

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

CORE = ("t0", "u0", "tE", "alpha", "rho", "q", "s")


def parameters(orbit):
    p = dict(t0=0.2, u0=.12, tE=3., alpha=.4, rho=.003, q=.01, s=1.2,
             g1=.011, g2=-.005, g3=.009)
    if orbit == "kepler":
        p.update(lom_szs=.7, lom_ar=1.3)
    return p


def curve(orbit, coordinates="vbm", limb=0., tol=1e-6, **kwargs):
    return lc.LightCurve(orbital_motion=orbit, t_ref=.1,
        options=lc.Options(coordinates=coordinates, tol=tol, reltol=tol),
        limb_darkening=lc.LimbDarkening.linear(limb), **kwargs)


@pytest.mark.parametrize("orbit", ["circular", "kepler"])
@pytest.mark.parametrize("coordinates", ["vbm", "center_of_mass", "lcbinint"])
@pytest.mark.parametrize("limb", [0., .5])
def test_orbital_values_and_fused_derivatives_match_jax(orbit, coordinates, limb):
    model = curve(orbit, coordinates, limb)
    p = parameters(orbit)
    times = np.array([-1., -.3, .15, .4, 1.2])
    assert model.options.finite_source_backend == "algebraic_boundary"
    ordinary = model(times, p)
    fused = model(times, p, with_gradient=True, with_diagnostics=True)
    assert fused.gradient_parameters == CORE + tuple(k for k in p if k not in CORE)
    np.testing.assert_allclose(fused.magnifications, ordinary.magnifications, rtol=2e-6, atol=2e-6)
    assert np.all(np.isfinite(fused.gradients))
    from lcbinint.jax_backend import magnification
    vector = jnp.array([p[k] for k in fused.gradient_parameters])
    def evaluate(v):
        return magnification(model._native, model.options, jnp.array(times),
            dict(zip(fused.gradient_parameters, v)), finite_source_backend="algebraic_boundary")
    actual = jax.jit(evaluate)(vector)
    np.testing.assert_allclose(actual, ordinary.magnifications, rtol=2e-8, atol=2e-8)
    reverse = jax.jit(jax.grad(lambda v: jnp.sum(evaluate(v))))(vector)
    np.testing.assert_allclose(reverse, np.sum(fused.gradients, axis=0), rtol=2e-7, atol=2e-7)
    for name in ("s", "q", "g1", "g2", "g3") + (("lom_szs", "lom_ar") if orbit == "kepler" else ()):
        step = 1e-4 * max(abs(p[name]), .01)
        plus = model(times, dict(p, **{name: p[name] + step})).magnifications
        minus = model(times, dict(p, **{name: p[name] - step})).magnifications
        fd = (plus-minus)/(2*step)
        np.testing.assert_allclose(fused.gradients[:, fused.gradient_parameters.index(name)], fd,
                                   rtol=2e-3, atol=2e-5)
    # Changing the supplied parameter values must not freeze the cached transform.
    changed = dict(p, g1=p["g1"]*1.2)
    second = model(times, changed, with_gradient=True)
    other = curve(orbit, coordinates, limb)(times, changed, with_gradient=True)
    np.testing.assert_allclose(second.gradients, other.gradients, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("orbit", ["circular", "kepler"])
@pytest.mark.parametrize("limb", [0., .5])
def test_orbit_integral_and_unsorted_batch_match_individual_geometry(orbit, limb):
    model = curve(orbit, limb=limb)
    p = dict(parameters(orbit), u0=.03, alpha=-.08, t0=0., tE=1.)
    times = np.array([.35, -.2, .1, .35, .6])
    result = model(times, p, with_diagnostics=True)
    geometry = model.finite_source_geometry(times, p)
    from lcbinint_jax import binary_algebraic_boundary_batch_ffi
    got = binary_algebraic_boundary_batch_ffi(jnp.array(geometry.source_x),
        jnp.array(geometry.source_y), jnp.array(geometry.separation),
        float(np.asarray(geometry.mass_ratio)[0]), p["rho"], limb_c=limb,
        absolute_tolerance=1e-6, relative_tolerance=1e-6)
    np.testing.assert_allclose(got, result.magnifications, rtol=1e-8, atol=1e-8)
    assert np.any(result.support_required)
    fused = model(times, p, with_gradient=True)
    assert np.all(np.isfinite(fused.gradients))
    np.testing.assert_allclose(fused.magnifications, result.magnifications, rtol=2e-6, atol=2e-6)
    individual = np.vstack([model(np.array([t]), p, with_gradient=True).gradients[0]
                            for t in times])
    np.testing.assert_allclose(fused.gradients, individual, rtol=2e-3, atol=2e-4)
    np.testing.assert_allclose(model(times, p, with_gradient=True,
        with_diagnostics=True).diagnostics["source_x"], geometry.source_x)


def test_vector_separation_shape_and_scalar_reverse_reduction():
    from lcbinint_jax import binary_algebraic_boundary_batch_ffi as call
    x, y = jnp.array([-.4, .2, .7]), jnp.array([.3, .2, .1])
    def evaluate(s):
        return call(x, y, s, .01, .001)
    scalar = evaluate(1.2)
    vector = evaluate(jnp.full_like(x, 1.2))
    np.testing.assert_array_equal(scalar, vector)
    gs = jax.grad(lambda s: jnp.sum(evaluate(s)))(1.2)
    gv = jax.grad(lambda s: jnp.sum(evaluate(s)))(jnp.full_like(x, 1.2))
    np.testing.assert_allclose(gs, jnp.sum(gv), rtol=1e-12)
    with pytest.raises(ValueError, match="separation"):
        evaluate(jnp.ones(2))


@pytest.mark.parametrize("coordinates", ["vbm", "center_of_mass", "lcbinint"])
def test_parallax_and_kepler_geometry_chain(coordinates):
    p = dict(parameters("kepler"), t0=10063.8, piEN=.018, piEE=.042)
    times = np.array([10062., 10063.5, 10064., 10065.])
    model = lc.LightCurve(orbital_motion="kepler", parallax=True, t_ref=10063.7,
        sky=lc.obs.SkyCoord(269.4084583, -28.6412583),
        options=lc.Options(coordinates=coordinates, tol=1e-6, reltol=1e-6))
    ordinary = model(times, p)
    fused = model(times, p, with_gradient=True)
    from lcbinint.jax_backend import magnification
    jax_values = magnification(model._native, model.options, jnp.array(times), p,
                              finite_source_backend="algebraic_boundary")
    np.testing.assert_allclose(jax_values, ordinary.magnifications, rtol=2e-8, atol=2e-8)
    for name in ("piEN", "piEE", "g1"):
        step = 1e-5
        plus = model(times, dict(p, **{name: p[name]+step})).magnifications
        minus = model(times, dict(p, **{name: p[name]-step})).magnifications
        np.testing.assert_allclose(fused.gradients[:, fused.gradient_parameters.index(name)],
                                   (plus-minus)/(2*step), rtol=2e-3, atol=2e-5)


@pytest.mark.parametrize("limb", [0., .5])
def test_kepler_values_against_independent_vbm(limb):
    vb = pytest.importorskip("VBMicrolensing").VBMicrolensing()
    vb.SetObjectCoordinates("17:57:38.03 -28:38:28.53")
    vb.Tol = vb.RelTol = 1e-7
    vb.a1 = limb
    p = dict(parameters("kepler"), u0=.03, alpha=-.08, t0=.1, tE=1.)
    times = np.linspace(-.2, .6, 13)
    vb.t0_par = .1
    vector = [np.log(p["s"]), np.log(p["q"]), p["u0"], p["alpha"],
              np.log(p["rho"]), np.log(p["tE"]), p["t0"], 0., 0.,
              p["g1"], p["g2"], p["g3"], p["lom_szs"], p["lom_ar"]]
    reference = np.asarray(vb.BinaryLightCurveKepler(vector, times.tolist())[0])
    if limb:
        # Direct VBM LD retains a strict-tolerance discrepancy at two epochs.
        # Use independently converged uniform-disc annuli, retaining all times.
        data = json.loads((Path(__file__).parents[1] /
            "data/algebraic_boundary/orbit_ld_reference.json").read_text())
        assert data["params"] == p
        np.testing.assert_array_equal(data["times"], times)
        reference = np.asarray(data["values"])
    result = curve("kepler", limb=limb, tol=1e-7)(times, p)
    np.testing.assert_allclose(result.magnifications, reference, rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize("g2", [0., .01])
def test_face_on_circular_branch_has_finite_reverse_derivatives(g2):
    model = curve("circular")
    p = dict(parameters("circular"), g1=0., g2=g2, g3=0.)
    times = np.array([-.4, .1, .7])
    result = model(times, p, with_gradient=True)
    assert np.all(np.isfinite(result.gradients))
    from lcbinint.jax_backend import magnification
    def loss(g):
        return jnp.sum(magnification(model._native, model.options, jnp.array(times),
            dict(p, g1=g[0], g2=g[1], g3=g[2]), finite_source_backend="algebraic_boundary"))
    actual = jax.grad(loss)(jnp.array([0., g2, 0.]))
    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual, result.gradients[:, -3:].sum(axis=0), rtol=1e-8, atol=1e-8)


def test_native_orbital_derivative_mapping_requires_float64():
    with jax.experimental.disable_x64():
        with pytest.raises(RuntimeError, match="64-bit JAX"):
            curve("circular")(np.array([0.1]), parameters("circular"), with_gradient=True)


@pytest.mark.parametrize("coordinates", ["vbm", "center_of_mass", "lcbinint"])
def test_orbit_crosses_unit_separation_without_reusing_stale_geometry(coordinates):
    model = curve("circular", coordinates)
    p = dict(parameters("circular"), s=1., g1=.2, g2=.03, g3=.04)
    times = np.array([-.7, .1, .8, -.7])
    geometry = model.finite_source_geometry(times, p)
    assert np.min(geometry.separation) < 1. < np.max(geometry.separation)
    actual = model(times, p).magnifications
    independent = np.array([model(np.array([t]), p).magnifications[0] for t in times])
    np.testing.assert_allclose(actual, independent, rtol=2e-6, atol=2e-6)
    from lcbinint.jax_backend import magnification
    expected = magnification(model._native, model.options, jnp.array(times), p,
                             finite_source_backend="algebraic_boundary")
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("orbit", ["circular", "kepler"])
def test_zero_reference_matches_native_t0_sentinel(orbit):
    p = parameters(orbit)
    times = np.array([-.4, .1, .7])
    model = lc.LightCurve(orbital_motion=orbit, t_ref=0.,
        options=lc.Options(coordinates="vbm", tol=1e-6, reltol=1e-6))
    native = model(times, p, with_gradient=True)
    from lcbinint.jax_backend import magnification
    actual = magnification(model._native, model.options, jnp.array(times), p,
                           finite_source_backend="algebraic_boundary")
    np.testing.assert_allclose(actual, native.magnifications, rtol=2e-8, atol=2e-8)
    step=1e-5
    fd=(model(times, dict(p,t0=p["t0"]+step)).magnifications-
        model(times, dict(p,t0=p["t0"]-step)).magnifications)/(2*step)
    np.testing.assert_allclose(native.gradients[:,0],fd,rtol=2e-3,atol=2e-5)
