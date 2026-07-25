import jax
import jax.numpy as jnp
import numpy as np

from lcbinint_jax.lens import (
    binary_lens_map_and_derivatives_real,
    binary_lens_map_complex,
    binary_lens_map_real,
)


def test_real_and_complex_binary_lens_maps_agree():
    image = jnp.asarray([0.2 + 0.4j, 1.3 - 0.1j, -0.7 + 0.9j])
    mapped_complex = binary_lens_map_complex(image, 1.2, 0.07)
    mapped_x, mapped_y = binary_lens_map_real(image.real, image.imag, 1.2, 0.07)
    np.testing.assert_allclose(
        mapped_complex, mapped_x + 1j * mapped_y, rtol=2.0e-14, atol=2.0e-14
    )


def test_analytic_image_plane_jacobian_matches_jacfwd():
    image = jnp.asarray([0.37, -0.42])
    separation = 0.93
    mass_ratio = 0.013

    def mapped(real_xy):
        return jnp.stack(
            binary_lens_map_real(real_xy[0], real_xy[1], separation, mass_ratio)
        )

    expected = jax.jacfwd(mapped)(image)
    _, _, du_dx, du_dy, dv_dx, dv_dy = binary_lens_map_and_derivatives_real(
        image[0], image[1], separation, mass_ratio
    )
    actual = jnp.asarray([[du_dx, du_dy], [dv_dx, dv_dy]])
    np.testing.assert_allclose(actual, expected, rtol=2.0e-13, atol=2.0e-13)
