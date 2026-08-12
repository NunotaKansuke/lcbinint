"""Regression tests for the public JAX LightCurve contract."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lcbinint
import lcbinint_jax
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

    disabled = lcbinint.LightCurve(options=lcbinint.Options(jax=True), jax=False)
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


def test_binary_ray_shooting_jax_selector_matches_native_and_differentiates():
    options = lcbinint.Options(
        jax=True,
        coordinates="center_of_mass",
        tol=1.0e-4,
        reltol=1.0e-4,
    )
    arguments = {
        "s": 1.4,
        "q": 1.0e-3,
        "rho": 0.01,
        "limb_darkening": lcbinint.LimbDarkening.linear(0.4),
    }
    actual = lcbinint.binary_ray_shooting(0.3, 0.4, options=options, **arguments)
    expected = lcbinint.binary_ray_shooting(
        0.3,
        0.4,
        options=lcbinint.Options(
            coordinates="center_of_mass",
            tol=1.0e-4,
            reltol=1.0e-4,
        ),
        **arguments,
    )

    assert isinstance(actual, jax.Array)
    np.testing.assert_allclose(actual, expected, rtol=1.0e-4, atol=1.0e-4)
    gradient = jax.grad(
        lambda x: lcbinint.binary_ray_shooting(x, 0.4, options=options, **arguments)
    )(0.3)
    assert jnp.isfinite(gradient)


def test_binary_ray_shooting_explicit_jax_override_has_constructor_precedence():
    selected = lcbinint.Options(jax=True, nbin=32)
    native = lcbinint.binary_ray_shooting(
        0.3,
        0.4,
        s=1.4,
        q=1.0e-3,
        rho=0.01,
        options=selected,
        jax=False,
    )
    differentiable = lcbinint.binary_ray_shooting(
        0.3,
        0.4,
        s=1.4,
        q=1.0e-3,
        rho=0.01,
        options=lcbinint.Options(nbin=32),
        jax=True,
    )
    assert isinstance(native, float)
    assert isinstance(differentiable, jax.Array)
    np.testing.assert_allclose(differentiable, native, rtol=1.0e-3, atol=1.0e-3)


def test_binary_ray_shooting_jax_validation_matches_native_contract():
    with pytest.raises(ValueError, match="positive rho"):
        lcbinint.binary_ray_shooting(
            0.3,
            0.4,
            s=1.4,
            q=1.0e-3,
            rho=0.0,
            jax=True,
        )
    compiled = jax.jit(
        lambda rho: lcbinint.binary_ray_shooting(
            0.3,
            0.4,
            s=1.4,
            q=1.0e-3,
            rho=rho,
            jax=True,
        )
    )
    assert jnp.isnan(compiled(0.0))
    assert jnp.isfinite(compiled(0.01))


def test_jax_magnification_batch_preserves_native_row_major_contract():
    curve = lcbinint.LightCurve(options=lcbinint.Options(jax=True))
    times = jnp.asarray((-1.0, 0.0, 1.0))
    rows = (PARAMETERS, {**PARAMETERS, "u0": 0.12})
    actual = curve.magnification_batch(times, rows)
    expected = jnp.stack(tuple(curve(times, row) for row in rows))

    assert isinstance(actual, jax.Array)
    assert actual.shape == (2, 3)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    derivative = jax.grad(
        lambda u0: jnp.sum(
            curve.magnification_batch(times, (PARAMETERS, {**PARAMETERS, "u0": u0}))
        )
    )(0.12)
    assert jnp.isfinite(derivative)


def test_jax_info_reports_the_selected_backend_pipeline():
    curve = lcbinint.LightCurve(options=lcbinint.Options(jax=True))
    times = jnp.asarray((-1.0, 0.0, 1.0))
    diagnostics = curve.info(times, PARAMETERS)

    assert isinstance(diagnostics.magnifications, jax.Array)
    assert diagnostics.magnifications.shape == (3,)
    np.testing.assert_allclose(
        diagnostics.magnifications,
        curve(times, PARAMETERS),
        rtol=0.0,
        atol=0.0,
    )
    assert len(diagnostics.finite_source_method_names) == 3
    assert set(diagnostics.finite_source_method_names) <= {
        "point_source",
        "hexadecapole",
        "inverse_ray_cartesian",
        "inverse_ray_polar",
        "source_plane_quadrature",
    }
    assert diagnostics.all_converged
    assert diagnostics.unconverged_indices == []


def test_jax_auto_curve_fails_closed_when_requested_accuracy_is_unmet():
    parameters = {
        "t0": 0.0,
        "tE": 1.0,
        "u0": 0.653,
        "alpha": np.pi / 2.0,
        "s": 1.2,
        "q": 0.1,
        "rho": 0.02,
        "limb_darkening_c": 0.4,
    }
    # A resolution ladder on this cusp settles at 3.8649464 (2048 cells per
    # source radius), and the bounded frontier now reaches 3.8649587 -- 3.2e-6
    # relative -- so 1e-5 is genuinely met and only 1e-6 is out of reach.
    curve = lcbinint.LightCurve(
        options=lcbinint.Options(
            jax=True,
            coordinates="center_of_mass",
            tol=0.0,
            reltol=1.0e-6,
        )
    )
    times = jnp.asarray((-0.02,))

    magnification = curve(times, parameters)
    diagnostics = curve.info(times, parameters)

    assert bool(jnp.isnan(magnification[0]))
    assert bool(jnp.isnan(diagnostics.magnifications[0]))
    assert bool(jnp.isfinite(diagnostics.finite_source_magnifications[0]))
    assert not diagnostics.all_converged
    assert diagnostics.unconverged_indices == [0]
    assert bool(jnp.isfinite(diagnostics.finite_source_error_estimates[0]))


@pytest.mark.parametrize("grid", ("cartesian", "polar"))
def test_jax_light_curve_honors_fixed_nbin_and_grid_options(grid):
    native = lcbinint.LightCurve(
        options=lcbinint.Options(
            nbin=32,
            inverse_ray_grid=grid,
            coordinates="center_of_mass",
        )
    )
    differentiable = lcbinint.LightCurve(
        options=lcbinint.Options(
            jax=True,
            nbin=32,
            inverse_ray_grid=grid,
            coordinates="center_of_mass",
        )
    )
    times = jnp.asarray((-0.1, 0.0, 0.1))
    actual = differentiable(times, PARAMETERS)
    expected = native(np.asarray(times), PARAMETERS)
    diagnostics = differentiable.info(times, PARAMETERS)

    assert isinstance(actual, jax.Array)
    np.testing.assert_allclose(actual, expected, rtol=2.0e-3, atol=2.0e-3)
    np.testing.assert_allclose(diagnostics.magnifications, actual, rtol=0.0, atol=0.0)


def test_jax_geometry_helpers_share_the_public_trajectory_and_ad():
    native = lcbinint.LightCurve()
    curve = lcbinint.LightCurve(options=lcbinint.Options(jax=True))
    times = jnp.asarray((-1.0, 0.0, 1.0))
    trajectory = curve.source_trajectory(times, PARAMETERS)
    geometry = curve.finite_source_geometry(times, PARAMETERS)
    expected_trajectory = native.source_trajectory(np.asarray(times), PARAMETERS)

    assert isinstance(trajectory.x, jax.Array)
    assert isinstance(geometry.source_x, jax.Array)
    np.testing.assert_allclose(
        trajectory.x, expected_trajectory.x, rtol=1.0e-12, atol=1.0e-12
    )
    np.testing.assert_allclose(
        trajectory.y, expected_trajectory.y, rtol=1.0e-12, atol=1.0e-12
    )
    np.testing.assert_allclose(geometry.source_x, trajectory.x)
    np.testing.assert_allclose(geometry.source_y, trajectory.y)
    np.testing.assert_allclose(
        curve.separation(times, PARAMETERS),
        geometry.separation,
        rtol=0.0,
        atol=0.0,
    )
    derivative = jax.grad(
        lambda u0: jnp.sum(curve.source_trajectory(times, {**PARAMETERS, "u0": u0}).y)
    )(PARAMETERS["u0"])
    assert jnp.isfinite(derivative)


def test_jax_batch_likelihood_keeps_magnification_and_flux_fit_differentiable():
    curve = lcbinint.LightCurve(options=lcbinint.Options(jax=True))
    times = jnp.asarray((-1.0, -0.5, 0.0, 0.5, 1.0))
    rows = (PARAMETERS, {**PARAMETERS, "u0": 0.12})
    model = curve.magnification_batch(times, rows)
    flux = 100.0 * model[0] + 5.0
    error = jnp.full(times.shape, 0.2)
    result = curve.light_curve_log_likelihood_batch(
        times, flux, error, rows, "gaussian", "fit"
    )

    assert isinstance(result["log_likelihood"], jax.Array)
    assert result["log_likelihood"].shape == (2,)
    np.testing.assert_allclose(
        result["source_flux"][0], 100.0, rtol=1.0e-9, atol=1.0e-9
    )
    np.testing.assert_allclose(result["blend_flux"][0], 5.0, rtol=1.0e-9, atol=1.0e-9)
    gradient = jax.grad(
        lambda u0: jnp.sum(
            curve.light_curve_log_likelihood_batch(
                times,
                flux,
                error,
                (PARAMETERS, {**PARAMETERS, "u0": u0}),
                "gaussian",
                "fit",
            )["log_likelihood"]
        )
    )(0.12)
    assert jnp.isfinite(gradient)


def test_jax_batch_student_t_sample_flux_matches_closed_form():
    curve = lcbinint.LightCurve(options=lcbinint.Options(jax=True))
    times = jnp.asarray((-0.5, 0.0, 0.5))
    row = {**PARAMETERS, "Fs_survey": 100.0, "Fb_survey": 5.0}
    magnification = curve(times, PARAMETERS)
    flux = 100.0 * magnification + 5.0
    result = curve.light_curve_log_likelihood_batch(
        times,
        flux,
        jnp.ones(times.shape),
        (row,),
        "student_t",
        "sample",
        4.0,
        jnp.asarray((100.0,)),
        jnp.asarray((5.0,)),
    )
    expected = times.shape[0] * (
        jax.scipy.special.gammaln(2.5)
        - jax.scipy.special.gammaln(2.0)
        - 0.5 * jnp.log(4.0 * jnp.pi)
    )
    np.testing.assert_allclose(
        result["log_likelihood"][0], expected, rtol=1.0e-12, atol=1.0e-12
    )


def test_jax_batch_gaussian_marginalization_reports_native_flux_statistics():
    curve = lcbinint.LightCurve(options=lcbinint.Options(jax=True))
    times = jnp.asarray((-1.0, -0.4, 0.1, 0.6, 1.2))
    magnification = curve(times, PARAMETERS)
    flux = 80.0 * magnification + 3.0 + jnp.asarray((0.03, -0.02, 0.01, -0.01, 0.02))
    error = jnp.asarray((0.2, 0.3, 0.2, 0.25, 0.3))
    result = curve.light_curve_log_likelihood_batch(
        times,
        flux,
        error,
        (PARAMETERS,),
        "gaussian",
        "marginalize",
    )

    weights = 1.0 / np.square(np.asarray(error))
    design = np.column_stack((np.asarray(magnification), np.ones(times.shape)))
    normal = design.T @ (weights[:, None] * design)
    source, blend = np.linalg.solve(normal, design.T @ (weights * flux))
    residual = (np.asarray(flux) - source * magnification - blend) / error
    chi2 = float(residual @ residual)
    determinant = float(np.linalg.det(normal))
    degrees = times.shape[0] - 2
    expected_log_likelihood = -0.5 * degrees * np.log(chi2) - 0.5 * np.log(determinant)
    expected_scale = np.sqrt(chi2 / degrees * np.sum(weights) / determinant)

    np.testing.assert_allclose(result["source_flux"][0], source, rtol=1.0e-10)
    np.testing.assert_allclose(result["blend_flux"][0], blend, rtol=1.0e-10)
    np.testing.assert_allclose(
        result["log_likelihood"][0], expected_log_likelihood, rtol=1.0e-10
    )
    np.testing.assert_allclose(
        result["conditional_scale"][0], expected_scale, rtol=1.0e-10
    )
    np.testing.assert_allclose(result["conditional_df"][0], degrees)


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
        {"omega_xa": 0.1, "w1": 0.2, "inc_xa": 0.3, "w2": 0.4, "phi_xa": 0.5, "w3": 0.6}
    )
    assert normalized["w1"] == 0.2
    assert normalized["w2"] == normalized["inc_xa"] == 0.4
    assert normalized["w3"] == 0.6

    reversed_aliases = _normalize_parameters(
        {"w1": 0.2, "omega_xa": 0.1, "w2": 0.4, "inc_xa": 0.3, "w3": 0.6, "phi_xa": 0.5}
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

    invalid = jax.jit(lambda q2: curve(jnp.asarray((0.0,)), {**parameters, "q2": q2}))(
        0.0
    )
    assert jnp.all(jnp.isnan(invalid))


@pytest.mark.parametrize(
    "limb_c,limb_d,expected_mode",
    (
        (0.0, 0.0, "uniform"),
        (0.4, 0.0, "linear"),
        (0.3, 0.2, "two_coefficient"),
    ),
)
def test_public_jax_triple_uses_smallest_static_moment_kernel(
    monkeypatch, limb_c, limb_d, expected_mode
):
    selected_modes = []

    def fake_triple_magnification_batch(source_x, *args, **kwargs):
        selected_modes.append(kwargs["moment_mode"])
        return SimpleNamespace(
            magnification=jnp.full_like(jnp.asarray(source_x), 7.25)
        )

    monkeypatch.setattr(
        lcbinint_jax,
        "triple_magnification_batch",
        fake_triple_magnification_batch,
    )
    curve = lcbinint.LightCurve(
        lens="triple",
        limb_darkening=lcbinint.LimbDarkening(limb_c, limb_d),
        options=lcbinint.Options(jax=True),
    )
    parameters = {
        **PARAMETERS,
        "q2": 0.01,
        "sep2": 1.3,
        "ang": 0.5,
    }

    actual = curve(jnp.asarray((0.0, 0.5)), parameters)

    np.testing.assert_array_equal(actual, jnp.asarray((7.25, 7.25)))
    assert selected_modes == [expected_mode]


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
