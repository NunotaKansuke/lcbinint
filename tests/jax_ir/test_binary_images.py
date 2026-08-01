import jax
import jax.numpy as jnp
import numpy as np

from lcbinint_jax import binary_images_ffi

jax.config.update("jax_enable_x64", True)


def test_polished_ghost_roots_are_not_counted_as_fold_images():
    """Nearby same-parity roots are classified by topology, not distance."""

    source_center = 0.653 + 0.02j
    source_radius = 0.02
    theta = 4.759407559293946
    source = source_center + source_radius * jnp.exp(1j * theta)

    images = binary_images_ffi(source, 1.2, 0.1)

    assert int(jnp.sum(images.physical)) == 3
    assert bool(jnp.all(images.root_converged[images.physical]))
    assert float(jnp.max(images.residuals[images.physical])) < 1.0e-9

    all_roots = np.asarray(images.roots)
    separations = np.abs(all_roots[:, None] - all_roots[None, :])
    separations += np.eye(all_roots.size)
    assert np.min(separations) < 1.0e-7


def test_binary_physical_image_count_never_becomes_four_across_fold():
    source_center = 0.653 + 0.02j
    source_radius = 0.02
    theta = jnp.linspace(4.61, 4.73, 49)

    counts = []
    for angle in theta:
        source = source_center + source_radius * jnp.exp(1j * angle)
        counts.append(int(jnp.sum(binary_images_ffi(source, 1.2, 0.1).physical)))

    assert set(counts).issubset({3, 5})
    assert set(counts) == {3, 5}
    assert 4 not in counts
