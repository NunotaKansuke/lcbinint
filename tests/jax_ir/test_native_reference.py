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


@pytest.mark.parametrize("limb_c", (0.0, 0.5))
@pytest.mark.parametrize("normal_offset", (-0.98, 0.0, 0.98, 1.02))
def test_fold_image_birth_and_death_gradient_matches_native(
    limb_c,
    normal_offset,
):
    """Exercise both sides of an image-pair birth/death at a smooth fold."""

    lcbinint = pytest.importorskip("lcbinint", exc_type=ImportError)
    caustic = np.asarray([0.06611188225495068, 0.1549319030240759])
    normal = np.asarray([-0.64645695, -0.76295047])
    source_radius = 0.01
    source = caustic + normal_offset * source_radius * normal

    def jax_value(active_source):
        return binary_inverse_ray(
            active_source[0],
            active_source[1],
            0.9,
            0.1,
            source_radius,
            limb_c,
            0.0,
            resolution=128,
            tile_size=16,
            tile_capacity=16384,
            limb_samples=128,
        )

    result = jax_value(jnp.asarray(source))
    gradient = jax.grad(lambda active_source: jax_value(active_source).magnification)(
        jnp.asarray(source)
    )
    jax_normal_derivative = float(jnp.dot(gradient, normal))

    native_options = lcbinint.Options(
        nbin=256,
        inverse_ray_grid="cartesian",
        coordinates="center_of_mass",
    )

    def native_value(active_source):
        return lcbinint.binary_ray_shooting(
            float(active_source[0]),
            float(active_source[1]),
            s=0.9,
            q=0.1,
            rho=source_radius,
            limb_darkening=lcbinint.LimbDarkening(c=limb_c, d=0.0),
            options=native_options,
        )

    step = source_radius * 1.0e-3
    native_normal_derivative = (
        native_value(source + step * normal)
        - native_value(source - step * normal)
    ) / (2.0 * step)

    assert bool(result.support_valid)
    assert np.isfinite(float(result.magnification))
    assert np.isfinite(jax_normal_derivative)
    np.testing.assert_allclose(
        jax_normal_derivative,
        native_normal_derivative,
        rtol=5.0e-3,
        atol=1.0,
    )


def test_cusp_centre_gradient_matches_native():
    """A source-centre cusp crossing remains smooth for a finite source."""

    lcbinint = pytest.importorskip("lcbinint", exc_type=ImportError)
    source = jnp.asarray([0.356921074, 0.0])
    source_radius = 0.01

    def jax_value(source_x):
        return binary_inverse_ray(
            source_x,
            source[1],
            0.9,
            0.1,
            source_radius,
            0.0,
            0.0,
            resolution=192,
            tile_size=16,
            tile_capacity=16384,
            limb_samples=128,
        )

    result = jax_value(source[0])
    jax_derivative = float(
        jax.grad(lambda source_x: jax_value(source_x).magnification)(source[0])
    )
    native_options = lcbinint.Options(
        nbin=256,
        inverse_ray_grid="cartesian",
        coordinates="center_of_mass",
    )

    def native_value(source_x):
        return lcbinint.binary_ray_shooting(
            float(source_x),
            float(source[1]),
            s=0.9,
            q=0.1,
            rho=source_radius,
            limb_darkening=lcbinint.LimbDarkening(),
            options=native_options,
        )

    step = source_radius * 1.0e-3
    native_derivative = (
        native_value(source[0] + step) - native_value(source[0] - step)
    ) / (2.0 * step)

    assert bool(result.support_valid)
    np.testing.assert_allclose(
        jax_derivative,
        native_derivative,
        rtol=6.0e-3,
        atol=1.0,
    )
