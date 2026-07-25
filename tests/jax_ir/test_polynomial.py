import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lcbinint_jax.images import binary_images
from lcbinint_jax.polynomial import (
    binary_lens_polynomial_coefficients,
    binary_lens_polynomial_roots,
)


@pytest.mark.parametrize(
    "source,separation,mass_ratio,physical_count",
    [
        (0.2 + 0.1j, 1.2, 0.1, 5),
        (0.01 + 0.01j, 0.8, 0.1, 5),
        (1.0 + 2.0j, 2.0, 0.001, 3),
        (0.5 + 0.0j, 1.0, 1.0, 3),
        (0.5 + 1.0e-12j, 1.0, 1.0, 3),
    ],
)
def test_binary_image_roots_and_physical_filter(
    source, separation, mass_ratio, physical_count
):
    images = binary_images(jnp.asarray(source), separation, mass_ratio)
    assert int(jnp.sum(images.physical)) == physical_count
    assert jnp.all(images.root_converged[images.physical])
    assert jnp.max(images.residuals[images.physical]) < 1.0e-9


def test_quintic_roots_satisfy_polynomial():
    source = jnp.asarray(0.2 + 0.1j)
    coefficients = binary_lens_polynomial_coefficients(source, 1.2, 0.1)
    roots = binary_lens_polynomial_roots(coefficients)
    residuals = jnp.abs(jnp.polyval(coefficients, roots.roots))
    scale = jnp.sum(jnp.abs(coefficients))
    assert jnp.all(roots.converged)
    assert jnp.max(residuals / scale) < 1.0e-11


def test_axis_source_uses_static_padded_root_slot():
    source = jnp.asarray(0.5 + 0.0j)
    coefficients = binary_lens_polynomial_coefficients(source, 1.0, 1.0)
    roots = binary_lens_polynomial_roots(coefficients)
    assert roots.roots.shape == (5,)
    assert int(jnp.sum(roots.converged)) == 4


def test_image_seed_roots_are_stopped_gradient():
    derivative = jax.grad(
        lambda source_x: jnp.sum(
            jnp.real(binary_images(source_x + 0.1j, 1.2, 0.1).roots)
        )
    )(jnp.asarray(0.2))
    np.testing.assert_array_equal(derivative, 0.0)


def test_source_limb_roots_vectorize_with_static_shape():
    angles = jnp.linspace(0.0, 2.0 * jnp.pi, 16, endpoint=False)
    sources = 0.2 + 0.1j + 0.02 * jnp.exp(1j * angles)
    images = jax.vmap(lambda source: binary_images(source, 1.2, 0.1))(sources)
    assert images.roots.shape == (16, 5)
    assert images.physical.shape == (16, 5)
    assert jnp.all(jnp.sum(images.physical, axis=1) >= 3)
