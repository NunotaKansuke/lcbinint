import jax.numpy as jnp
import numpy as np
import pytest

from lcbinint_jax import binary_inverse_ray_convergence

PARAMETERS = jnp.asarray([0.2, 0.1, 1.2, 0.1, 0.2, 0.4, 0.1])
DIRECTION = jnp.asarray([0.2, -0.1, 0.05, 0.02, 0.0, 0.1, -0.05])


def test_value_convergence_reports_all_normalized_observables():
    result = binary_inverse_ray_convergence(
        PARAMETERS,
        coarse_resolution=32,
        fine_resolution=64,
        coarse_tile_capacity=512,
        fine_tile_capacity=1024,
    )

    assert result.coarse_observables.shape == (4,)
    assert result.fine_observables.shape == (4,)
    assert bool(result.coarse_support_valid)
    assert bool(result.fine_support_valid)
    assert bool(result.value_converged)
    assert bool(result.moments_converged)
    assert not bool(result.gradient_checked)
    assert not bool(result.gradient_converged)
    assert np.all(np.isnan(result.derivative_errors))


def test_directional_gradient_convergence_improves_with_resolution():
    # The bounded frontier admits the tiles that hold the faint images hugging
    # each lens; the sampled one used to miss them entirely.  Those images are
    # a couple of cells across below resolution 128, so their area -- and far
    # more so its derivative -- is quantization noise there.  Reporting that as
    # unconverged is the honest answer, which is why the converged pair is
    # 128/256 and not 64/128.
    low = binary_inverse_ray_convergence(
        PARAMETERS,
        direction=DIRECTION,
        coarse_resolution=16,
        fine_resolution=32,
        coarse_tile_capacity=512,
        fine_tile_capacity=1024,
        gradient_atol=2.0e-3,
        gradient_rtol=2.0e-3,
    )
    high = binary_inverse_ray_convergence(
        PARAMETERS,
        direction=DIRECTION,
        coarse_resolution=128,
        fine_resolution=256,
        coarse_tile_capacity=4096,
        fine_tile_capacity=16384,
        gradient_atol=2.0e-3,
        gradient_rtol=2.0e-3,
    )

    assert bool(low.gradient_checked)
    assert not bool(low.gradient_converged)
    assert bool(high.gradient_converged)
    assert jnp.linalg.norm(high.derivative_errors) < jnp.linalg.norm(
        low.derivative_errors
    )


def test_convergence_rejects_invalid_shapes_and_resolution_order():
    with pytest.raises(ValueError, match=r"shape \(7,\)"):
        binary_inverse_ray_convergence(jnp.ones(6))
    with pytest.raises(ValueError, match="greater"):
        binary_inverse_ray_convergence(
            PARAMETERS,
            coarse_resolution=64,
            fine_resolution=32,
        )
