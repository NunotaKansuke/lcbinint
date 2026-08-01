"""The macro-tile frontier test must bound the lens map, not sample it.

``tile_size`` picks how coarsely the flood fill walks the image plane.  It is a
performance knob: it decides how many tiles the walk visits, and nothing about
which region of the image plane the images occupy.  The magnification must not
depend on it.

The previous frontier test sampled nine points per tile and admitted the tile if
one of them landed in the source disk.  That answers a different question, and a
thin image component that passes between the nine fails all nine -- the walk
stops on the component it was following, and still reports ``support_valid``.
On the tangency below it made the answer move with ``tile_size`` and made the
resolution ladder non-monotone: at 64/128/256/512 the sampled probe gave
-4.5e-4/-8.5e-4/-2.9e-4/-4.7e-5 relative to the reference, i.e. resolution 128
was *worse* than 64.

The bound in ``lcbinint_jax.discovery._tile_has_inside_probe`` replaces the nine
samples with a Lipschitz lower bound on ``|f(z) - zeta|`` over the whole tile,
which can over-admit but never under-admit.  These tests pin the two properties
that follow, not the calibration: the answer stops depending on ``tile_size``,
and the resolution ladder becomes monotone.
"""

import numpy as np
import pytest

from lcbinint_jax import (
    binary_inverse_ray_uniform,
    triple_inverse_ray_adaptive,
)


# The tangency of tests/regression/test_component_refinement.py, where the two
# fold images form a 55:1 sliver.  VBMicrolensing BinaryMag2 at Tol=1e-9/1e-10.
CUSP_REFERENCE = 3.960888498085
CUSP_SEPARATION, CUSP_MASS_RATIO = 1.2, 0.1
CUSP_SOURCE = (0.653, 0.020)
CUSP_RADIUS = 0.020


def _cusp_magnification(resolution, tile_size, root_backend):
    return binary_inverse_ray_uniform(
        CUSP_SOURCE[0],
        CUSP_SOURCE[1],
        CUSP_SEPARATION,
        CUSP_MASS_RATIO,
        CUSP_RADIUS,
        resolution=resolution,
        tile_size=tile_size,
        tile_capacity=16384,
        limb_samples=128,
        root_backend=root_backend,
    )


@pytest.mark.parametrize("root_backend", ("auto", "jax"))
def test_binary_magnification_is_independent_of_tile_size(root_backend):
    """A performance knob may cost time.  It may not move the answer."""

    results = [
        _cusp_magnification(128, tile_size, root_backend)
        for tile_size in (8, 16, 32)
    ]
    for result in results:
        assert bool(result.support_valid)
    reference = float(results[0].magnification)
    for result in results[1:]:
        # The walk visits the tiles in a different order, so the sum is not
        # bit-identical; the support it covers has to be the same one.
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
    """The triple frontier carries the same bound, so it owes the same property."""

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
