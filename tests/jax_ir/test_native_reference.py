import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lcbinint_jax import binary_inverse_ray


@pytest.mark.parametrize(
    "source_x,source_y,separation,mass_ratio,source_radius,limb_c,limb_d",
    (
        (0.2, 0.1, 1.2, 0.1, 0.2, 0.0, 0.0),
        (0.2, 0.1, 1.2, 0.1, 0.2, 0.4, 0.1),
        (0.653, 0.0, 1.2, 0.1, 0.02, 0.0, 0.0),
    ),
)
def test_jax_inverse_ray_matches_native_lcbinint(
    source_x,
    source_y,
    separation,
    mass_ratio,
    source_radius,
    limb_c,
    limb_d,
):
    lcbinint = pytest.importorskip("lcbinint", exc_type=ImportError)
    native = lcbinint.binary_ray_shooting(
        source_x,
        source_y,
        s=separation,
        q=mass_ratio,
        rho=source_radius,
        limb_darkening=lcbinint.LimbDarkening(c=limb_c, d=limb_d),
        options=lcbinint.Options(
            nbin=128,
            inverse_ray_grid="cartesian",
            coordinates="center_of_mass",
        ),
    )
    result = binary_inverse_ray(
        source_x,
        source_y,
        separation,
        mass_ratio,
        source_radius,
        limb_c,
        limb_d,
        resolution=128,
        tile_size=16,
        tile_capacity=4096,
        limb_samples=64,
    )

    assert bool(result.support_valid)
    np.testing.assert_allclose(
        result.magnification,
        native,
        rtol=2.0e-4,
        atol=5.0e-4,
    )


def test_jax_directional_derivative_matches_native_lcbinint_difference():
    lcbinint = pytest.importorskip("lcbinint", exc_type=ImportError)
    parameters = jnp.asarray([0.2, 0.1, 1.2, 0.1, 0.2, 0.4, 0.1])
    direction = jnp.asarray([0.2, -0.1, 0.05, 0.02, 0.0, 0.1, -0.05])

    def jax_value(active_parameters):
        return binary_inverse_ray(
            *active_parameters,
            resolution=128,
            tile_size=16,
            tile_capacity=4096,
            limb_samples=64,
        ).magnification

    _, jax_directional_derivative = jax.jvp(
        jax_value,
        (parameters,),
        (direction,),
    )

    native_options = lcbinint.Options(
        nbin=512,
        inverse_ray_grid="cartesian",
        coordinates="center_of_mass",
    )

    def native_value(active_parameters):
        x, y, separation, mass_ratio, source_radius, limb_c, limb_d = map(
            float, active_parameters
        )
        return lcbinint.binary_ray_shooting(
            x,
            y,
            s=separation,
            q=mass_ratio,
            rho=source_radius,
            limb_darkening=lcbinint.LimbDarkening(c=limb_c, d=limb_d),
            options=native_options,
        )

    step = 3.0e-3
    native_directional_difference = (
        native_value(parameters + step * direction)
        - native_value(parameters - step * direction)
    ) / (2.0 * step)
    np.testing.assert_allclose(
        jax_directional_derivative,
        native_directional_difference,
        rtol=5.0e-3,
        atol=5.0e-3,
    )
