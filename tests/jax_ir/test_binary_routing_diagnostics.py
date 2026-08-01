import jax
import jax.numpy as jnp

from lcbinint_jax import (
    binary_point_source_magnification,
    binary_routing_diagnostics_batch_ffi,
)

jax.config.update("jax_enable_x64", True)


def _diagnostics(source_x):
    source_y = jnp.asarray(
        (0.1, -0.0017214156233429664, 3.954526794514226e-05)
    )
    point = jax.vmap(
        lambda x, y: binary_point_source_magnification(
            x, y, 1.0, 1.0e-3
        ).magnification
    )(source_x, source_y)
    return binary_routing_diagnostics_batch_ffi(
        source_x,
        source_y,
        point,
        1.0,
        1.0e-3,
        1.0e-4,
    )


def test_binary_routing_diagnostics_classify_point_chord_and_crossing():
    source_x = jnp.asarray(
        (0.2, -0.0011744718711112381, -0.0007041237762303424)
    )
    result = _diagnostics(source_x)

    assert bool(result.point_preflight_safe[0])
    assert bool(result.point_safe[0])
    assert not bool(result.scan_performed[0])

    assert not bool(result.point_safe[1])
    assert bool(result.scan_performed[1])
    assert bool(result.chord_band[1])
    assert not bool(result.grazing_ring_band[1])

    assert bool(result.scan_performed[2])
    assert bool(
        result.any_vertex_inside[2] | result.has_crossing_probes[2]
    )
    assert not bool(result.chord_band[2])
    assert not bool(result.grazing_ring_band[2])


def test_binary_routing_diagnostics_are_stopped_gradient():
    source_x = jnp.asarray(
        (0.2, -0.0011744718711112381, -0.0007041237762303424)
    )
    gradient = jax.grad(
        lambda active: jnp.sum(_diagnostics(active).point_error_estimate)
    )(source_x)
    assert bool(jnp.all(gradient == 0.0))
