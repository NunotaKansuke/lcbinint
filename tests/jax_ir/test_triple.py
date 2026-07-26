import jax
import jax.numpy as jnp
import numpy as np

import lcbinint
from lcbinint_jax import (
    triple_inverse_ray_dense,
    triple_lens_geometry,
    triple_lens_map_and_derivatives_real,
)
from lcbinint_jax.lens import binary_lens_map_and_derivatives_real

jax.config.update("jax_enable_x64", True)


def test_triple_geometry_is_center_of_mass_centered():
    geometry = triple_lens_geometry(1.0, 0.1, 0.03, 0.7, 0.8)
    np.testing.assert_allclose(jnp.sum(geometry.masses), 1.0, atol=1.0e-15)
    np.testing.assert_allclose(
        jnp.sum(geometry.positions * geometry.masses[:, None], axis=0),
        0.0,
        atol=1.0e-15,
    )


def test_zero_tertiary_mass_reduces_to_binary_lens_map():
    image_x = jnp.asarray((-0.8, 0.2, 1.1))
    image_y = jnp.asarray((0.4, -0.7, 0.3))
    triple_geometry = triple_lens_geometry(1.2, 0.1, 0.0, 0.7, 0.8)
    triple = triple_lens_map_and_derivatives_real(
        image_x, image_y, triple_geometry
    )
    binary = binary_lens_map_and_derivatives_real(
        image_x, image_y, 1.2, 0.1
    )
    for triple_value, binary_value in zip(triple, binary):
        np.testing.assert_allclose(triple_value, binary_value, atol=3.0e-15)


def test_dense_triple_inverse_ray_matches_native_uniform_source():
    parameters = {
        "t0": 0.0,
        "tE": 1.0,
        "u0": 0.3,
        "alpha": 0.0,
        "s": 1.0,
        "q": 0.1,
        "q2": 0.03,
        "sep2": 0.7,
        "ang": 0.8,
        "rho": 0.2,
    }
    native = lcbinint.LightCurve(
        model=lcbinint.Model(lens="triple"),
        options=lcbinint.Options(
            coordinates="original",
            nbin=200,
            inverse_ray_grid="cartesian",
        ),
    )([0.2], parameters)[0]
    actual = triple_inverse_ray_dense(
        0.2,
        0.3,
        1.0,
        0.1,
        0.03,
        0.7,
        0.8,
        0.2,
        resolution=512,
        image_extent=3.0,
        moment_mode="uniform",
    )
    assert bool(actual.support_valid)
    assert int(actual.contributing_cells) > 0
    assert int(actual.boundary_cells) > 0
    np.testing.assert_allclose(actual.magnification, native, rtol=3.0e-3)


def test_dense_triple_inverse_ray_has_parameter_gradients():
    def loss(source_x, tertiary_mass_ratio, tertiary_angle):
        return triple_inverse_ray_dense(
            source_x,
            0.3,
            1.0,
            0.1,
            tertiary_mass_ratio,
            0.7,
            tertiary_angle,
            0.2,
            resolution=512,
            image_extent=3.0,
            moment_mode="uniform",
        ).magnification

    value, gradients = jax.value_and_grad(loss, argnums=(0, 1, 2))(
        0.2, 0.03, 0.8
    )
    assert bool(jnp.isfinite(value))
    assert bool(jnp.all(jnp.isfinite(jnp.asarray(gradients))))
    step = 3.0e-5
    finite_difference = (
        loss(0.2 + step, 0.03, 0.8) - loss(0.2 - step, 0.03, 0.8)
    ) / (2.0 * step)
    np.testing.assert_allclose(gradients[0], finite_difference, rtol=1.0e-3)


def test_dense_triple_inverse_ray_rejects_insufficient_support():
    result = triple_inverse_ray_dense(
        0.2,
        0.3,
        1.0,
        0.1,
        0.03,
        0.7,
        0.8,
        0.2,
        resolution=128,
        image_extent=0.5,
        moment_mode="uniform",
    )
    assert not bool(result.support_valid)
