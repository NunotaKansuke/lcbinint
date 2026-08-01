"""Regression coverage for a finite-source cusp cap smaller than a limb raster."""

import math

import jax
import numpy as np
import pytest


jax.config.update("jax_enable_x64", True)

_ARGS = dict(
    source_x=0.653,
    source_y=0.020,
    separation=1.2,
    mass_ratio=0.1,
    source_radius=0.020,
)
_UNIFORM_REFERENCE = 3.960889170813
_LINEAR_C05_REFERENCE = 3.836256795465


def _native_value(lcbinint, bins, limb_c):
    curve = lcbinint.LightCurve(
        lens="binary",
        options=lcbinint.Options(
            coordinates="center_of_mass",
            source_bins=bins,
            max_source_bins=bins,
            inverse_ray_grid="cartesian",
        ),
    )
    return curve.magnification(
        _ARGS["source_x"],
        t0=0.0,
        tE=1.0,
        u0=_ARGS["source_y"],
        alpha=0.0,
        s=_ARGS["separation"],
        q=_ARGS["mass_ratio"],
        rho=_ARGS["source_radius"],
        limb_darkening_c=limb_c,
    ).item()


@pytest.mark.parametrize(
    ("limb_c", "reference"),
    ((0.0, _UNIFORM_REFERENCE), (0.5, _LINEAR_C05_REFERENCE)),
)
def test_native_cusp_component_is_resolved_and_refines(limb_c, reference):
    lcbinint = pytest.importorskip("lcbinint")
    values = [_native_value(lcbinint, bins, limb_c) for bins in (64, 128, 256, 512)]
    errors = [abs(value - reference) for value in values]
    assert errors[-1] < 1.0e-4
    assert errors[-1] < errors[0]
    assert abs(values[-1] - values[-2]) < abs(values[-2] - values[-3])


@pytest.mark.parametrize(
    ("limb_c", "reference"),
    ((0.0, _UNIFORM_REFERENCE), (0.5, _LINEAR_C05_REFERENCE)),
)
def test_jax_cusp_component_is_resolved_and_gradients_stabilize(limb_c, reference):
    from lcbinint_jax import binary_inverse_ray_linear, binary_inverse_ray_uniform

    def evaluate(resolution, *parameters):
        if limb_c == 0.0:
            return binary_inverse_ray_uniform(
                *parameters,
                resolution=resolution,
                tile_capacity=65536,
                limb_samples=16,
                cartesian_backend="jax",
            ).magnification
        return binary_inverse_ray_linear(
            *parameters,
            limb_c,
            resolution=resolution,
            tile_capacity=65536,
            limb_samples=16,
            cartesian_backend="jax",
        ).magnification

    parameters = tuple(_ARGS[name] for name in (
        "source_x", "source_y", "separation", "mass_ratio", "source_radius"
    ))
    values = [float(evaluate(resolution, *parameters)) for resolution in (64, 128, 256, 512)]
    assert abs(values[-1] - reference) < 1.0e-4
    assert abs(values[-1] - values[-2]) < abs(values[-2] - values[-3])

    gradients = []
    for resolution in (512, 1024):
        function = lambda *args: evaluate(resolution, *args)
        gradients.append(np.asarray(jax.grad(function, argnums=(0, 1, 2, 3, 4))(*parameters)))
    # The caustic-normal source coordinate and rho carry the singular local
    # response.  They must settle together under a final dyadic refinement.
    stiff = (1, 4)
    denominator = np.maximum(
        np.maximum(np.abs(gradients[0][list(stiff)]), np.abs(gradients[1][list(stiff)])),
        1.0,
    )
    assert np.max(
        np.abs(gradients[1][list(stiff)] - gradients[0][list(stiff)]) / denominator
    ) < 0.20
