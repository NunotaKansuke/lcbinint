import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lcbinint_jax import binary_caustic_distance_batch_ffi

jax.config.update("jax_enable_x64", True)


def _segment_distance(point, start, end):
    delta = end - start
    scale = np.dot(delta, delta)
    fraction = 0.0 if scale == 0.0 else np.clip(
        np.dot(point - start, delta) / scale, 0.0, 1.0
    )
    return np.linalg.norm(point - (start + fraction * delta))


def _sampled_caustic_distance(lcbinint, source, separation, mass_ratio, bins):
    curve = lcbinint.LightCurve(
        options=lcbinint.Options(
            coordinates="center_of_mass",
            caustic_bins=bins,
        )
    )
    caustics = curve.caustics(s=separation, q=mass_ratio, n_points=bins)
    point = np.asarray(source)
    distances = []
    for branch_x, branch_y in zip(caustics.x, caustics.y):
        branch = np.column_stack((np.asarray(branch_x), np.asarray(branch_y)))
        for index, start in enumerate(branch):
            distances.append(
                _segment_distance(point, start, branch[(index + 1) % len(branch)])
            )
    return min(distances)


def test_binary_caustic_distance_matches_native_sampled_segments():
    lcbinint = pytest.importorskip("lcbinint", exc_type=ImportError)
    source_x = np.asarray((0.2, 0.653, -0.1))
    source_y = np.asarray((0.1, 0.0, 0.15))
    separation = 1.2
    mass_ratio = 0.1
    bins = 512
    result = binary_caustic_distance_batch_ffi(
        jnp.asarray(source_x),
        jnp.asarray(source_y),
        separation,
        mass_ratio,
        caustic_bins=bins,
    )
    reference = np.asarray(
        [
            _sampled_caustic_distance(
                lcbinint, source, separation, mass_ratio, bins
            )
            for source in zip(source_x, source_y)
        ]
    )
    np.testing.assert_allclose(result, reference, rtol=0.0, atol=2.0e-12)


def test_binary_caustic_distance_is_stopped_gradient():
    def total(source_x):
        return jnp.sum(
            binary_caustic_distance_batch_ffi(
                source_x,
                jnp.asarray((0.1, 0.0)),
                1.2,
                0.1,
                caustic_bins=256,
            )
        )

    gradient = jax.jit(jax.grad(total))(jnp.asarray((0.2, 0.653)))
    np.testing.assert_array_equal(gradient, np.zeros(2))
