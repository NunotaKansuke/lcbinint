import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lcbinint


TIMES = jnp.linspace(-3.0, 3.0, 12)
BINARY_PARAMETERS = {
    "t0": 0.0,
    "tE": 24.0,
    "u0": 0.08,
    "alpha": 0.7,
    "s": 0.9,
    "q": 0.1,
    "rho": 0.008,
}
TRIPLE_PARAMETERS = {
    **BINARY_PARAMETERS,
    "sep2": 1.3,
    "q2": 0.01,
    "ang": 0.5,
}


def curves(lens="binary", **kwargs):
    native_options = lcbinint.Options(tol=1.0e-4, reltol=1.0e-4, **kwargs)
    jax_options = lcbinint.Options(
        jax=True,
        tol=1.0e-4,
        reltol=1.0e-4,
        **kwargs,
    )
    return (
        lcbinint.LightCurve(lens=lens, options=native_options),
        lcbinint.LightCurve(lens=lens, options=jax_options),
    )


def test_options_select_jax_without_changing_native_default():
    native = lcbinint.Options()
    selected = lcbinint.Options(jax=True)

    assert not native.jax
    assert selected.jax
    assert "backend='jax'" in repr(selected)


@pytest.mark.parametrize(
    ("lens", "parameters"),
    (("binary", BINARY_PARAMETERS), ("triple", TRIPLE_PARAMETERS)),
)
def test_public_jax_light_curve_matches_native_and_differentiates(
    lens, parameters
):
    native, jax_curve = curves(lens)
    expected = native(np.asarray(TIMES), parameters)
    actual = jax_curve(TIMES, parameters)

    assert isinstance(actual, jax.Array)
    np.testing.assert_allclose(actual, expected, rtol=5.0e-5, atol=5.0e-4)

    def loss(impact_parameter):
        active = dict(parameters)
        active["u0"] = impact_parameter
        return jnp.sum(jax_curve(TIMES, active))

    value, gradient = jax.jit(jax.value_and_grad(loss))(parameters["u0"])
    assert jnp.isfinite(value)
    assert jnp.isfinite(gradient)
    assert gradient != 0.0


def test_public_jax_api_respects_center_of_mass_coordinates():
    parameters = dict(BINARY_PARAMETERS)
    parameters.update({"s": 1.2, "q": 0.2})
    native, jax_curve = curves("binary", coordinates="center_of_mass")

    np.testing.assert_allclose(
        jax_curve(TIMES, parameters),
        native(np.asarray(TIMES), parameters),
        rtol=5.0e-5,
        atol=5.0e-4,
    )


def test_public_jax_api_supports_static_binary_sources():
    parameters = {
        **{
            name: value
            for name, value in BINARY_PARAMETERS.items()
            if name != "rho"
        },
        "rho1": 0.008,
        "rho2": 0.006,
        "t0_2": 0.7,
        "u0_2": 0.2,
        "flux_ratio": 0.3,
    }
    native_options = lcbinint.Options(tol=1.0e-4, reltol=1.0e-4)
    jax_options = lcbinint.Options(
        jax=True, tol=1.0e-4, reltol=1.0e-4
    )
    native = lcbinint.LightCurve(source="binary", options=native_options)
    jax_curve = lcbinint.LightCurve(source="binary", options=jax_options)

    np.testing.assert_allclose(
        jax_curve(TIMES, parameters),
        native(np.asarray(TIMES), parameters),
        rtol=8.0e-5,
        atol=8.0e-4,
    )
    gradient = jax.grad(
        lambda flux_ratio: jnp.sum(
            jax_curve(TIMES, {**parameters, "flux_ratio": flux_ratio})
        )
    )(parameters["flux_ratio"])
    assert jnp.isfinite(gradient)

    components = jax_curve.binary_source_components(TIMES, parameters)
    native_components = native.binary_source_components(
        np.asarray(TIMES), parameters
    )
    assert isinstance(components.total, jax.Array)
    np.testing.assert_allclose(
        components.total, jax_curve(TIMES, parameters), rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        components.source1.magnification,
        native_components.source1.magnification,
        rtol=8.0e-5,
        atol=8.0e-4,
    )
    np.testing.assert_allclose(
        components.source2.magnification,
        native_components.source2.magnification,
        rtol=8.0e-5,
        atol=8.0e-4,
    )


def test_public_jax_api_composes_higher_order_effects_and_gradient():
    times = jnp.asarray((7497.0, 7500.0, 7504.0))
    parameters = {
        "t0": 7500.0,
        "tE": 25.0,
        "u0": 0.15,
        "alpha": 0.4,
        "s": 1.1,
        "q": 10.0,
        "rho": 0.008,
        "piEN": 0.12,
        "piEE": -0.05,
        "g1": 0.004,
        "g2": 0.011,
        "g3": 0.006,
        "lom_szs": 0.2,
        "lom_ar": 1.4,
        "xi_1": 0.006,
        "xi_2": -0.003,
        "w1": 0.004,
        "w2": 0.35,
        "w3": 0.08,
        "xa_szs": 0.2,
        "xa_ar": 1.4,
    }
    model = lcbinint.Model(
        parallax=True,
        terrestrial=True,
        orbital_motion="kepler",
        xallarap="kepler_velocity",
        sky=lcbinint.obs.SkyCoord(270.0, -30.0),
        t_ref=7500.0,
    )
    site = lcbinint.obs.Site("ground", -29.0, -70.7)
    native = lcbinint.LightCurve(
        model=model,
        site=site,
        options=lcbinint.Options(tol=1.0e-4, reltol=1.0e-4),
    )
    jax_curve = lcbinint.LightCurve(
        model=model,
        site=site,
        options=lcbinint.Options(
            jax=True, tol=1.0e-4, reltol=1.0e-4
        ),
    )

    np.testing.assert_allclose(
        jax_curve(times, parameters),
        native(np.asarray(times), parameters),
        rtol=8.0e-5,
        atol=8.0e-4,
    )
    gradient = jax.grad(
        lambda impact_parameter: jnp.sum(
            jax_curve(
                times,
                {**parameters, "u0": impact_parameter},
            )
        )
    )(parameters["u0"])
    assert jnp.isfinite(gradient)


def test_public_jax_api_supports_space_parallax_and_gradient():
    times = jnp.asarray((8998.2, 9000.0, 9001.6))
    table = np.asarray(
        (
            (2458998.0, 0.0, 0.0, 0.010),
            (2458999.0, 0.0, 0.0, 0.011),
            (2459000.0, 0.0, 0.0, 0.012),
            (2459001.0, 0.0, 0.0, 0.013),
            (2459002.0, 0.0, 0.0, 0.014),
        )
    )
    parameters = {
        "t0": 9000.0,
        "tE": 20.0,
        "u0": 0.1,
        "alpha": 0.0,
        "s": 1.1,
        "q": 10.0,
        "rho": 0.0,
        "piEN": 0.1,
        "piEE": 0.05,
    }
    model = lcbinint.Model(
        parallax=True,
        sky=lcbinint.obs.SkyCoord(270.0, -30.0),
        t_ref=9000.0,
    )
    site = lcbinint.obs.Site("space", table)
    native = lcbinint.LightCurve(
        model=model,
        site=site,
        options=lcbinint.Options(coordinates="vbm"),
    )
    jax_curve = lcbinint.LightCurve(
        model=model,
        site=site,
        options=lcbinint.Options(jax=True, coordinates="vbm"),
    )

    np.testing.assert_allclose(
        jax_curve(times, parameters),
        native(np.asarray(times), parameters),
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    gradient = jax.grad(
        lambda pi_en: jnp.sum(
            jax_curve(times, {**parameters, "piEN": pi_en})
        )
    )(parameters["piEN"])
    assert jnp.isfinite(gradient)
    assert gradient != 0.0


@pytest.mark.parametrize(
    ("xallarap", "source_coordinates", "extra"),
    (
        (
            "circular_elements",
            "none",
            {"period_xa": 35.0, "inc_xa": 1.1},
        ),
        (
            "orbital_elements",
            "none",
            {
                "period_xa": 35.0,
                "inc_xa": 1.1,
                "ecc_xa": 0.2,
                "peri_xa": 0.4,
            },
        ),
        (
            "circular_velocity",
            "xallarap",
            {"w1": 0.02, "w2": 1.1, "w3": 0.3},
        ),
        (
            "kepler_velocity",
            "xallarap",
            {
                "w1": 0.02,
                "w2": 1.1,
                "w3": 0.3,
                "xa_szs": 0.2,
                "xa_ar": 1.4,
            },
        ),
        (
            "circular_velocity",
            "trajectory_offset",
            {
                "t0": -0.4,
                "u0": 0.3,
                "t0_2": 0.8,
                "u0_2": -0.1,
                "w1": 0.02,
                "w2": 1.1,
                "w3": 0.3,
            },
        ),
        (
            "kepler_velocity",
            "trajectory_offset",
            {
                "t0": -0.4,
                "u0": 0.3,
                "t0_2": 0.8,
                "u0_2": -0.1,
                "w1": 0.02,
                "w2": 1.1,
                "w3": 0.3,
                "xa_szs": 0.2,
                "xa_ar": 1.4,
            },
        ),
    ),
)
def test_public_jax_api_supports_binary_source_xallarap(
    xallarap, source_coordinates, extra
):
    times = jnp.asarray((-1.0, 0.0, 1.0))
    parameters = {
        "t0": 0.0,
        "tE": 20.0,
        "u0": 0.3,
        "alpha": 0.5,
        "s": 1.0,
        "q": 10.0,
        "rho1": 0.0,
        "rho2": 0.0,
        "flux_ratio": 0.4,
        "source_mass_ratio": 0.5,
        "xi_1": 0.05,
        "xi_2": -0.02,
        **extra,
    }
    model_arguments = {
        "source": "binary",
        "xallarap": xallarap,
        "t_ref": 0.0,
    }
    if source_coordinates != "none":
        model_arguments["source_orbit_coordinates"] = source_coordinates
    model = lcbinint.Model(**model_arguments)
    native = lcbinint.LightCurve(
        model=model,
        options=lcbinint.Options(coordinates="vbm"),
    )
    jax_curve = lcbinint.LightCurve(
        model=model,
        options=lcbinint.Options(jax=True, coordinates="vbm"),
    )

    np.testing.assert_allclose(
        jax_curve(times, parameters),
        native(np.asarray(times), parameters),
        rtol=2.0e-10,
        atol=2.0e-10,
    )
    gradient = jax.grad(
        lambda source_mass_ratio: jnp.sum(
            jax_curve(
                times,
                {
                    **parameters,
                    "source_mass_ratio": source_mass_ratio,
                },
            )
        )
    )(parameters["source_mass_ratio"])
    assert jnp.isfinite(gradient)
    assert gradient != 0.0


def test_public_jax_api_composes_space_parallax_and_binary_source_xallarap():
    times = jnp.asarray((8999.7, 9000.0, 9000.3))
    table = np.asarray(
        (
            (2458999.0, 20.0, -10.0, 0.010),
            (2459000.0, 21.0, -10.5, 0.011),
            (2459001.0, 22.0, -11.0, 0.012),
        )
    )
    parameters = {
        "t0": 9000.0,
        "tE": 25.0,
        "u0": 0.3,
        "alpha": 0.4,
        "s": 1.1,
        "q": 10.0,
        "rho1": 0.0,
        "rho2": 0.0,
        "flux_ratio": 0.4,
        "source_mass_ratio": 0.7,
        "xi_1": 0.006,
        "xi_2": -0.003,
        "w1": 0.004,
        "w2": 0.35,
        "w3": 0.08,
        "piEN": 0.12,
        "piEE": -0.05,
    }
    model = lcbinint.Model(
        source="binary",
        parallax=True,
        xallarap="circular_velocity",
        source_orbit_coordinates="xallarap",
        sky=lcbinint.obs.SkyCoord(270.0, -30.0),
        t_ref=9000.0,
    )
    site = lcbinint.obs.Site("space", table)
    native = lcbinint.LightCurve(
        model=model,
        site=site,
        options=lcbinint.Options(coordinates="vbm"),
    )
    jax_curve = lcbinint.LightCurve(
        model=model,
        site=site,
        options=lcbinint.Options(jax=True, coordinates="vbm"),
    )

    np.testing.assert_allclose(
        jax_curve(times, parameters),
        native(np.asarray(times), parameters),
        rtol=2.0e-10,
        atol=2.0e-10,
    )
    gradient = jax.grad(
        lambda pi_en: jnp.sum(
            jax_curve(times, {**parameters, "piEN": pi_en})
        )
    )(parameters["piEN"])
    assert jnp.isfinite(gradient)
    assert gradient != 0.0


def test_public_jax_api_supports_triple_binary_source_xallarap():
    times = jnp.asarray((-1.0, 0.0, 1.0))
    parameters = {
        "t0": 0.0,
        "tE": 20.0,
        "u0": 0.3,
        "alpha": 0.5,
        "s": 1.0,
        "q": 10.0,
        "q2": 0.03,
        "sep2": 0.7,
        "ang": 0.8,
        "rho1": 0.0,
        "rho2": 0.0,
        "flux_ratio": 0.4,
        "source_mass_ratio": 0.5,
        "xi_1": 0.05,
        "xi_2": -0.02,
        "w1": 0.02,
        "w2": 1.1,
        "w3": 0.3,
    }
    model = lcbinint.Model(
        lens="triple",
        source="binary",
        xallarap="circular_velocity",
        source_orbit_coordinates="xallarap",
        t_ref=0.0,
    )
    native = lcbinint.LightCurve(
        model=model,
        options=lcbinint.Options(coordinates="vbm"),
    )
    jax_curve = lcbinint.LightCurve(
        model=model,
        options=lcbinint.Options(jax=True, coordinates="vbm"),
    )

    np.testing.assert_allclose(
        jax_curve(times, parameters),
        native(np.asarray(times), parameters),
        rtol=2.0e-10,
        atol=2.0e-10,
    )
    gradient = jax.grad(
        lambda source_mass_ratio: jnp.sum(
            jax_curve(
                times,
                {
                    **parameters,
                    "source_mass_ratio": source_mass_ratio,
                },
            )
        )
    )(parameters["source_mass_ratio"])
    assert jnp.isfinite(gradient)
    assert gradient != 0.0
