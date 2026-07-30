"""Regression tests for the public JAX LightCurve contract."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lcbinint
from lcbinint.jax_backend import _normalize_parameters


PARAMETERS = {
    "t0": 0.0,
    "tE": 24.0,
    "u0": 0.08,
    "alpha": 0.7,
    "s": 0.9,
    "q": 0.1,
    "rho": 0.008,
}


def test_options_keep_positional_api_and_jax_selector_outside_c_options():
    options = lcbinint.Options("center_of_mass")
    assert options.param_type == "center_of_mass"
    assert not options.jax

    options.jax = True
    assert options.jax
    assert "backend='jax'" in repr(options)

    curve = lcbinint.LightCurve(options=options)
    assert isinstance(curve(jnp.asarray(0.0), PARAMETERS), jax.Array)
    curve.options.jax = False
    assert isinstance(curve(np.asarray(0.0), PARAMETERS), np.ndarray)

    disabled = lcbinint.LightCurve(
        options=lcbinint.Options(jax=True), jax=False
    )
    assert isinstance(disabled(np.asarray(0.0), PARAMETERS), np.ndarray)


def test_options_repr_is_safe_when_native_initialization_fails():
    options = lcbinint.Options.__new__(lcbinint.Options)
    assert repr(options) == "<lc.Options (uninitialized)>"
    with pytest.raises(AttributeError):
        getattr(options, "param_type")
    with pytest.raises(TypeError) as error:
        lcbinint.Options(object())
    assert "RecursionError" not in repr(error.value)


def test_public_options_are_accepted_by_binary_ray_shooting():
    value = lcbinint.binary_ray_shooting(
        0.3,
        0.2,
        s=1.2,
        q=0.1,
        rho=0.01,
        options=lcbinint.Options("center_of_mass", source_bins=24),
    )
    assert np.isfinite(value)


def test_jax_parameter_aliases_match_native_light_curve():
    aliases = {
        "t_0": PARAMETERS["t0"],
        "t_E": PARAMETERS["tE"],
        "umin": PARAMETERS["u0"],
        "theta": PARAMETERS["alpha"],
        "sep": PARAMETERS["s"],
        "q": PARAMETERS["q"],
        "rho1": PARAMETERS["rho"],
    }
    native = lcbinint.LightCurve(options=lcbinint.Options())
    jax_curve = lcbinint.LightCurve(options=lcbinint.Options(jax=True))
    times = jnp.asarray((-1.0, 0.0, 1.0))

    np.testing.assert_allclose(
        jax_curve(times, aliases),
        native(np.asarray(times), aliases),
        rtol=5.0e-5,
        atol=5.0e-4,
    )


@pytest.mark.parametrize(
    "parameters, expected_t0",
    (
        ({**PARAMETERS, "t0": -2.0, "t_0": 1.0}, 1.0),
        ({"t_0": 1.0, **PARAMETERS, "t0": -2.0}, -2.0),
    ),
)
def test_jax_alias_precedence_matches_native_mapping(parameters, expected_t0):
    normalized = _normalize_parameters(parameters)
    assert normalized["t0"] == expected_t0

    native = lcbinint.LightCurve(options=lcbinint.Options())
    jax_curve = lcbinint.LightCurve(options=lcbinint.Options(jax=True))
    times = jnp.asarray((-1.0, 0.0, 1.0))
    canonical = {**PARAMETERS, "t0": expected_t0}
    np.testing.assert_allclose(
        jax_curve(times, parameters),
        jax_curve(times, canonical),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        jax_curve(times, parameters),
        native(np.asarray(times), parameters),
        rtol=2.0e-4,
        atol=2.0e-3,
    )


def test_xallarap_velocity_aliases_share_native_storage_and_precedence():
    normalized = _normalize_parameters(
        {"omega_xa": 0.1, "w1": 0.2, "inc_xa": 0.3, "w2": 0.4,
         "phi_xa": 0.5, "w3": 0.6}
    )
    assert normalized["w1"] == 0.2
    assert normalized["w2"] == normalized["inc_xa"] == 0.4
    assert normalized["w3"] == 0.6

    reversed_aliases = _normalize_parameters(
        {"w1": 0.2, "omega_xa": 0.1, "w2": 0.4, "inc_xa": 0.3,
         "w3": 0.6, "phi_xa": 0.5}
    )
    assert reversed_aliases["w1"] == 0.1
    assert reversed_aliases["w2"] == reversed_aliases["inc_xa"] == 0.3
    assert reversed_aliases["w3"] == 0.5


def test_jax_rejects_unknown_mapping_keys_like_native():
    curve = lcbinint.LightCurve(options=lcbinint.Options(jax=True))
    with pytest.raises(KeyError, match="unknown parameter 'not_a_parameter'"):
        curve(jnp.asarray((0.0,)), {**PARAMETERS, "not_a_parameter": 1.0})


@pytest.mark.parametrize("q2", (None, 0.0, -0.01))
def test_jax_triple_requires_positive_q2(q2):
    parameters = dict(PARAMETERS)
    if q2 is not None:
        parameters["q2"] = q2
    curve = lcbinint.LightCurve(lens="triple", options=lcbinint.Options(jax=True))

    with pytest.raises(RuntimeError, match="requires a positive q2"):
        curve(jnp.asarray((0.0,)), parameters)


def test_jax_binary_rejects_positive_q2():
    curve = lcbinint.LightCurve(options=lcbinint.Options(jax=True))
    with pytest.raises(RuntimeError, match="cannot be used with a positive q2"):
        curve(jnp.asarray((0.0,)), {**PARAMETERS, "q2": 0.01})


def test_jitted_triple_q2_remains_differentiable_while_checked_at_runtime():
    curve = lcbinint.LightCurve(lens="triple", options=lcbinint.Options(jax=True))
    parameters = {**PARAMETERS, "q2": 0.01, "sep2": 1.3, "ang": 0.5}

    def value(q2):
        return jnp.sum(curve(jnp.asarray((0.0,)), {**parameters, "q2": q2}))

    value, gradient = jax.jit(jax.value_and_grad(value))(parameters["q2"])
    assert jnp.isfinite(value)
    assert jnp.isfinite(gradient)

    invalid = jax.jit(lambda q2: curve(
        jnp.asarray((0.0,)), {**parameters, "q2": q2}
    ))(0.0)
    assert jnp.all(jnp.isnan(invalid))


def test_jax_higher_order_model_requires_t_ref():
    curve = lcbinint.LightCurve(
        parallax=True,
        sky=lcbinint.obs.SkyCoord(270.0, -30.0),
        options=lcbinint.Options(jax=True),
    )
    with pytest.raises(RuntimeError, match="t_ref must be set"):
        curve(jnp.asarray((0.0,)), PARAMETERS)


def test_jax_scalar_time_matches_native_single_epoch_shape_and_value():
    native = lcbinint.LightCurve(options=lcbinint.Options())
    jax_curve = lcbinint.LightCurve(options=lcbinint.Options(jax=True))

    actual = jax_curve(jnp.asarray(0.0), PARAMETERS)
    expected = native(np.asarray(0.0), PARAMETERS)

    assert actual.shape == (1,)
    assert expected.shape == (1,)
    np.testing.assert_allclose(actual, expected, rtol=5.0e-5, atol=5.0e-4)
