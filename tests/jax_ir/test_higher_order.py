import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lcbinint
import lcbinint_jax.higher_order as higher_order
from lcbinint_jax import (
    annual_parallax_offsets,
    binary_lens_trajectory,
    binary_source_magnification_light_curve,
    binary_source_trajectories,
    circular_orbital_motion,
    load_earth_ephemeris,
    space_parallax_offsets,
    terrestrial_parallax_offsets,
    xallarap_offsets,
)

jax.config.update("jax_enable_x64", True)


_TIMES = np.asarray((-0.5, 0.2, 1.0))
_PARAMETERS = {
    "t0": 0.0,
    "tE": 10.0,
    "u0": 0.1,
    "alpha": 0.3,
    "s": 1.2,
    # VBM geometry reports 1/q, so q=10 corresponds to the JAX q=0.1 lens.
    "q": 10.0,
    "rho": 0.01,
    "g1": 0.004,
    "g2": 0.011,
    "g3": 0.006,
    "lom_szs": 0.2,
    "lom_ar": 1.4,
}


def test_load_earth_ephemeris_reads_packaged_resource(monkeypatch, tmp_path):
    """Installed wheels load the table as package data, not via the checkout."""

    packaged_data = tmp_path / "data"
    packaged_data.mkdir()
    (packaged_data / "earth_orbital_parallax_table.txt").write_text(
        "$$SOE\n"
        "7000.0, ignored, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0\n"
        "7001.0, ignored, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0\n"
        "$$EOE\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(higher_order.resources, "files", lambda package: tmp_path)

    ephemeris = higher_order.load_earth_ephemeris()

    np.testing.assert_array_equal(ephemeris.time, (7000.0, 7001.0))
    np.testing.assert_array_equal(
        ephemeris.position, ((1.0, 2.0, 3.0), (7.0, 8.0, 9.0))
    )
    np.testing.assert_array_equal(
        ephemeris.velocity, ((4.0, 5.0, 6.0), (10.0, 11.0, 12.0))
    )


@pytest.mark.parametrize("mode", ("circular", "kepler"))
def test_lens_orbit_geometry_matches_native(mode):
    native = lcbinint.LightCurve(
        model=lcbinint.Model(orbital_motion=mode, t_ref=0.2),
        options=lcbinint.Options(coordinates="vbm"),
    ).finite_source_geometry(_TIMES, _PARAMETERS)
    actual = binary_lens_trajectory(
        _TIMES,
        t0=0.0,
        timescale=10.0,
        impact_parameter=0.1,
        separation=1.2,
        angle=0.3,
        reference_time=0.2,
        lens_orbit=mode,
        g1=0.004,
        g2=0.011,
        g3=0.006,
        line_of_sight_ratio=0.2,
        semimajor_axis_ratio=1.4,
    )
    np.testing.assert_allclose(actual.separation, native.separation, atol=5.0e-15)
    np.testing.assert_allclose(actual.source_x, native.source_x, atol=5.0e-15)
    np.testing.assert_allclose(actual.source_y, native.source_y, atol=5.0e-15)
    np.testing.assert_array_equal(actual.valid, True)


def test_annual_and_terrestrial_parallax_match_native():
    times = np.asarray((7499.7, 7500.0, 7500.3))
    parameters = {
        "t0": 7500.0,
        "tE": 30.0,
        "u0": 0.1,
        "alpha": 0.3,
        "s": 1.2,
        "q": 10.0,
        "rho": 0.01,
        "piEN": 0.12,
        "piEE": -0.05,
    }
    native = lcbinint.LightCurve(
        model=lcbinint.Model(
            parallax=True,
            terrestrial=True,
            sky=lcbinint.obs.SkyCoord(270.0, -30.0),
            t_ref=7500.0,
        ),
        site=lcbinint.obs.Site("ground", -29.0, -70.7),
        options=lcbinint.Options(coordinates="vbm"),
    ).finite_source_geometry(times, parameters)
    annual_tau, annual_beta = annual_parallax_offsets(
        times,
        0.12,
        -0.05,
        270.0,
        -30.0,
        7500.0,
        load_earth_ephemeris(),
    )
    site_tau, site_beta = terrestrial_parallax_offsets(
        times,
        0.12,
        -0.05,
        270.0,
        -30.0,
        -29.0,
        -70.7,
    )
    actual = binary_lens_trajectory(
        times,
        t0=7500.0,
        timescale=30.0,
        impact_parameter=0.1,
        separation=1.2,
        angle=0.3,
        tau_offset=annual_tau + site_tau,
        beta_offset=annual_beta + site_beta,
    )
    np.testing.assert_allclose(actual.source_x, native.source_x, atol=2.0e-15)
    np.testing.assert_allclose(actual.source_y, native.source_y, atol=2.0e-15)


@pytest.mark.parametrize(
    ("mode", "native_parameters", "jax_parameters"),
    (
        (
            "circular_elements",
            {"period_xa": 12.0, "inc_xa": 0.6},
            {"period": 12.0, "inclination": 0.6},
        ),
        (
            "orbital_elements",
            {
                "period_xa": 12.0,
                "inc_xa": 0.6,
                "ecc_xa": 0.2,
                "peri_xa": 0.4,
            },
            {
                "period": 12.0,
                "inclination": 0.6,
                "eccentricity": 0.2,
                "periapsis": 0.4,
            },
        ),
        (
            "circular_velocity",
            {"w1": 0.004, "w2": 0.35, "w3": 0.08},
            {"w1": 0.004, "w2": 0.35, "w3": 0.08},
        ),
        (
            "kepler_velocity",
            {
                "w1": 0.004,
                "w2": 0.35,
                "w3": 0.08,
                "xa_szs": 0.2,
                "xa_ar": 1.4,
            },
            {
                "w1": 0.004,
                "w2": 0.35,
                "w3": 0.08,
                "line_of_sight_ratio": 0.2,
                "semimajor_axis_ratio": 1.4,
            },
        ),
    ),
)
def test_xallarap_modes_match_native(mode, native_parameters, jax_parameters):
    times = np.asarray((7495.0, 7500.0, 7507.0))
    parameters = {
        "t0": 7500.0,
        "tE": 30.0,
        "u0": 0.2,
        "alpha": 0.7,
        "s": 0.9,
        "q": 10.0,
        "rho": 0.004,
        "xi_1": 0.006,
        "xi_2": -0.003,
        **native_parameters,
    }
    native = lcbinint.LightCurve(
        xallarap=mode,
        t_ref=7500.0,
        options=lcbinint.Options(coordinates="vbm"),
    ).finite_source_geometry(times, parameters)
    tau_offset, beta_offset = xallarap_offsets(
        times,
        mode=mode,
        xi_1=0.006,
        xi_2=-0.003,
        reference_time=7500.0,
        **jax_parameters,
    )
    actual = binary_lens_trajectory(
        times,
        t0=7500.0,
        timescale=30.0,
        impact_parameter=0.2,
        separation=0.9,
        angle=0.7,
        tau_offset=tau_offset,
        beta_offset=beta_offset,
    )
    np.testing.assert_allclose(actual.source_x, native.source_x, atol=3.0e-12)
    np.testing.assert_allclose(actual.source_y, native.source_y, atol=3.0e-12)


def test_space_parallax_matches_native_satellite_displacement():
    times = np.asarray((8998.2, 9000.0, 9001.6))
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
        "rho": 0.01,
        "piEN": 0.1,
        "piEE": 0.05,
    }
    model = lcbinint.Model(
        parallax=True,
        sky=lcbinint.obs.SkyCoord(270.0, -30.0),
        t_ref=9000.0,
    )
    earth = lcbinint.LightCurve(
        model=model,
        options=lcbinint.Options(coordinates="vbm"),
    ).finite_source_geometry(times, parameters)
    satellite = lcbinint.LightCurve(
        model=model,
        site=lcbinint.obs.Site("space", table),
        options=lcbinint.Options(coordinates="vbm"),
    ).finite_source_geometry(times, parameters)
    tau, beta = space_parallax_offsets(
        times, 0.1, 0.05, 270.0, -30.0, table
    )
    np.testing.assert_allclose(
        tau, np.asarray(satellite.source_x) - np.asarray(earth.source_x), atol=1.0e-15
    )
    np.testing.assert_allclose(
        beta, np.asarray(satellite.source_y) - np.asarray(earth.source_y), atol=1.0e-15
    )


def test_orbital_geometry_is_differentiable():
    times = jnp.asarray((-1.0, 0.0, 1.0))

    def loss(g1):
        state = circular_orbital_motion(
            times, 1.2, 0.3, g1, 0.011, 0.006, 0.2
        )
        return jnp.sum(state.separation + state.angle)

    gradient = jax.grad(loss)(0.004)
    step = 1.0e-6
    finite_difference = (loss(0.004 + step) - loss(0.004 - step)) / (2.0 * step)
    np.testing.assert_allclose(gradient, finite_difference, rtol=2.0e-7)


@pytest.mark.parametrize(
    ("curve_parameters", "native_parameters", "jax_parameters"),
    (
        (
            {"source": "binary"},
            {"t0_2": 7501.0, "u0_2": -0.05},
            {"t0_2": 7501.0, "impact_parameter_2": -0.05},
        ),
        (
            {
                "source": "binary",
                "xallarap": "circular_velocity",
                "source_orbit_coordinates": "xallarap",
                "t_ref": 7500.0,
            },
            {
                "source_mass_ratio": 0.7,
                "xi_1": 0.02,
                "xi_2": -0.01,
                "w1": 0.004,
                "w2": 0.35,
                "w3": 0.08,
            },
            {
                "source_mass_ratio": 0.7,
                "source_orbit_coordinates": "xallarap",
                "xallarap_mode": "circular_velocity",
                "xi_1": 0.02,
                "xi_2": -0.01,
                "reference_time": 7500.0,
                "xallarap_parameters": {
                    "w1": 0.004,
                    "w2": 0.35,
                    "w3": 0.08,
                },
            },
        ),
        (
            {
                "source": "binary",
                "xallarap": "circular_velocity",
                "source_orbit_coordinates": "trajectory_offset",
                "t_ref": 7500.0,
            },
            {
                "t0": 7499.4,
                "u0": 0.19,
                "t0_2": 7500.857142857,
                "u0_2": 0.214285714,
                "source_mass_ratio": 0.7,
                "w1": 0.004,
                "w2": 0.35,
                "w3": 0.08,
            },
            {
                "t0": 7499.4,
                "impact_parameter": 0.19,
                "t0_2": 7500.857142857,
                "impact_parameter_2": 0.214285714,
                "source_mass_ratio": 0.7,
                "source_orbit_coordinates": "trajectory_offset",
                "xallarap_mode": "circular_velocity",
                "reference_time": 7500.0,
                "xallarap_parameters": {
                    "w1": 0.004,
                    "w2": 0.35,
                    "w3": 0.08,
                },
            },
        ),
    ),
)
def test_binary_source_coordinate_modes_match_native(
    curve_parameters, native_parameters, jax_parameters
):
    times = np.asarray((7495.0, 7500.0, 7505.0))
    common_native = {
        "t0": 7500.0,
        "tE": 30.0,
        "u0": 0.2,
        "alpha": 0.7,
        "s": 0.9,
        "q": 10.0,
        "rho1": 0.004,
        "rho2": 0.002,
        "flux_ratio": 0.4,
        **native_parameters,
    }
    native = lcbinint.LightCurve(
        options=lcbinint.Options(coordinates="vbm"),
        **curve_parameters,
    ).binary_source_components(times, common_native)
    common_jax = {
        "t0": 7500.0,
        "timescale": 30.0,
        "impact_parameter": 0.2,
        "separation": 0.9,
        "angle": 0.7,
        **jax_parameters,
    }
    actual = binary_source_trajectories(times, **common_jax)
    for native_component, actual_component in (
        (native.source1, actual.source1),
        (native.source2, actual.source2),
    ):
        np.testing.assert_allclose(
            actual_component.source_x, native_component.trajectory.x, atol=2.0e-15
        )
        np.testing.assert_allclose(
            actual_component.source_y, native_component.trajectory.y, atol=2.0e-15
        )


def test_binary_source_fused_light_curve_gradient_is_finite():
    times = jnp.asarray((-3.0, 0.0, 3.0))

    def loss(source_mass_ratio):
        result = binary_source_magnification_light_curve(
            times,
            0.1,
            0.01,
            0.006,
            0.4,
            trajectory_parameters={
                "t0": 0.0,
                "timescale": 10.0,
                "impact_parameter": 0.4,
                "separation": 1.2,
                "angle": 0.3,
                "source_mass_ratio": source_mass_ratio,
                "source_orbit_coordinates": "xallarap",
                "xallarap_mode": "circular_velocity",
                "xi_1": 0.01,
                "xi_2": -0.005,
                "reference_time": 0.0,
                "xallarap_parameters": {
                    "w1": 0.004,
                    "w2": 0.35,
                    "w3": 0.08,
                },
            },
            integration_parameters={
                "resolution": 64,
                "tile_capacity": 1024,
                "limb_samples": 16,
                "source_plane_fallback": False,
                "moment_mode": "linear",
            },
        )
        return jnp.sum(result.total)

    value, gradient = jax.value_and_grad(loss)(0.7)
    assert bool(jnp.isfinite(value))
    assert bool(jnp.isfinite(gradient))


def test_all_single_source_higher_order_effects_compose_like_native():
    times = np.asarray((7497.0, 7500.0, 7504.0))
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
    native = lcbinint.LightCurve(
        model=lcbinint.Model(
            parallax=True,
            terrestrial=True,
            orbital_motion="kepler",
            xallarap="kepler_velocity",
            sky=lcbinint.obs.SkyCoord(270.0, -30.0),
            t_ref=7500.0,
        ),
        site=lcbinint.obs.Site("ground", -29.0, -70.7),
        options=lcbinint.Options(coordinates="vbm"),
    ).finite_source_geometry(times, parameters)
    parallax_tau, parallax_beta = annual_parallax_offsets(
        times,
        0.12,
        -0.05,
        270.0,
        -30.0,
        7500.0,
        load_earth_ephemeris(),
    )
    site_tau, site_beta = terrestrial_parallax_offsets(
        times, 0.12, -0.05, 270.0, -30.0, -29.0, -70.7
    )
    xa_tau, xa_beta = xallarap_offsets(
        times,
        mode="kepler_velocity",
        xi_1=0.006,
        xi_2=-0.003,
        reference_time=7500.0,
        w1=0.004,
        w2=0.35,
        w3=0.08,
        line_of_sight_ratio=0.2,
        semimajor_axis_ratio=1.4,
    )
    actual = binary_lens_trajectory(
        times,
        t0=7500.0,
        timescale=25.0,
        impact_parameter=0.15,
        separation=1.1,
        angle=0.4,
        reference_time=7500.0,
        lens_orbit="kepler",
        g1=0.004,
        g2=0.011,
        g3=0.006,
        line_of_sight_ratio=0.2,
        semimajor_axis_ratio=1.4,
        tau_offset=parallax_tau + site_tau + xa_tau,
        beta_offset=parallax_beta + site_beta + xa_beta,
    )
    np.testing.assert_allclose(actual.separation, native.separation, atol=5.0e-15)
    np.testing.assert_allclose(actual.source_x, native.source_x, atol=3.0e-15)
    np.testing.assert_allclose(actual.source_y, native.source_y, atol=3.0e-15)
