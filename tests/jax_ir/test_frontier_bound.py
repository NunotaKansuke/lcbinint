"""Binary/triple FFI multi-run support and the pure-JAX tile fallback.

Binary and triple Cartesian FFI trace the native horizontal multi-run topology
on a cell lattice. ``tile_size`` remains accepted for compatibility and work
budget sizing, but it does not partition FFI support. The optional pure-JAX
triple path still uses macro-tiles and retains its conservative Lipschitz
frontier.

The shallow cusp below checks that changing the legacy binary tile-size
argument cannot change the covered image support or the resulting magnification.
The resolution ladder separately checks that refinement improves the value.
"""

import numpy as np
import pytest

from lcbinint_jax import (
    binary_inverse_ray_uniform,
    cpp_triple_cartesian_epoch_ffi_available,
    triple_inverse_ray_adaptive,
)


# The tangency of tests/regression/test_component_refinement.py, where the two
# fold images form a 55:1 sliver.  VBMicrolensing BinaryMag2 at Tol=1e-9/1e-10.
CUSP_REFERENCE = 3.960888498085
CUSP_SEPARATION, CUSP_MASS_RATIO = 1.2, 0.1
CUSP_SOURCE = (0.653, 0.020)
CUSP_RADIUS = 0.020


def _cusp_magnification(resolution, legacy_tile_size, root_backend):
    return binary_inverse_ray_uniform(
        CUSP_SOURCE[0],
        CUSP_SOURCE[1],
        CUSP_SEPARATION,
        CUSP_MASS_RATIO,
        CUSP_RADIUS,
        resolution=resolution,
        tile_size=legacy_tile_size,
        tile_capacity=16384,
        limb_samples=128,
        root_backend=root_backend,
    )


@pytest.mark.parametrize("root_backend", ("auto", "jax"))
def test_binary_magnification_is_independent_of_legacy_tile_size(root_backend):
    """The compatibility argument may affect capacity, never support shape."""

    results = [
        _cusp_magnification(128, tile_size, root_backend)
        for tile_size in (8, 16, 32)
    ]
    for result in results:
        assert bool(result.support_valid)
    reference = float(results[0].magnification)
    for result in results[1:]:
        # The legacy capacity multiplier cannot alter run order when all three
        # settings stay within budget.
        np.testing.assert_allclose(
            float(result.magnification), reference, rtol=1.0e-12, atol=0.0
        )


def test_binary_resolution_ladder_is_monotone():
    """Refining the lattice may not make the tangency worse."""

    errors = [
        abs(float(_cusp_magnification(resolution, 16, "auto").magnification)
            / CUSP_REFERENCE - 1.0)
        for resolution in (64, 128, 256)
    ]
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 1.0e-4


# A wide triple with a low-mass tertiary, whose planetary caustic carries the
# thin component.
TRIPLE_PARAMETERS = (1.0, 1.0e-3, 1.0e-4, 0.5, 1.2)
TRIPLE_SOURCE = (-0.05, 0.02)
TRIPLE_RADIUS = 6.497855561e-03 / 0.99


@pytest.mark.parametrize("use_ffi", (True, False))
def test_triple_magnification_is_independent_of_tile_size(use_ffi):
    """Both FFI run fill and the pure-JAX tile path ignore tile granularity."""

    results = [
        triple_inverse_ray_adaptive(
            TRIPLE_SOURCE[0],
            TRIPLE_SOURCE[1],
            *TRIPLE_PARAMETERS,
            TRIPLE_RADIUS,
            resolution=128,
            tile_size=tile_size,
            tile_capacity=131072,
            limb_samples=64,
            moment_mode="uniform",
            use_ffi=use_ffi,
        )
        for tile_size in (8, 16, 32)
    ]
    for result in results:
        assert bool(result.support_valid)
    reference = float(results[0].magnification)
    for result in results[1:]:
        np.testing.assert_allclose(
            float(result.magnification), reference, rtol=1.0e-12, atol=0.0
        )


@pytest.mark.skipif(
    not cpp_triple_cartesian_epoch_ffi_available(),
    reason="triple Cartesian FFI is unavailable",
)
def test_triple_ffi_reports_run_support_count():
    result = triple_inverse_ray_adaptive(
        TRIPLE_SOURCE[0],
        TRIPLE_SOURCE[1],
        *TRIPLE_PARAMETERS,
        TRIPLE_RADIUS,
        resolution=32,
        tile_size=8,
        tile_capacity=4096,
        limb_samples=32,
        moment_mode="uniform",
        use_ffi=True,
    )
    assert bool(result.support_valid)
    assert int(result.support_count) > 0
