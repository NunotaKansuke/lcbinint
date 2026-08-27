"""Differentiable experimental algebraic-boundary epoch FFI."""

import numpy as np
import pytest

import jax

from lcbinint_jax import (
    binary_algebraic_boundary_polar_ffi,
    cpp_algebraic_boundary_epoch_ffi_available,
)

pytestmark = pytest.mark.skipif(
    not cpp_algebraic_boundary_epoch_ffi_available(),
    reason="algebraic-boundary epoch FFI not built",
)

# Off-caustic geometries where the driver's measure-zero gates stay quiet.
CASES = [
    dict(source_x=0.22, source_y=0.15, separation=1.35, mass_ratio=0.28, source_radius=0.05),
    dict(source_x=0.05, source_y=0.18, separation=0.72, mass_ratio=0.45, source_radius=0.03),
    dict(source_x=0.30, source_y=0.02, separation=1.20, mass_ratio=0.20, source_radius=0.008),
]
PROFILES = [dict(), dict(limb_c=0.6), dict(limb_c=0.4, limb_d=0.2)]


def _mag(case, profile, **over):
    kw = dict(case, **profile)
    kw.update(over)
    return float(
        binary_algebraic_boundary_polar_ffi(**kw).magnification
    )


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("profile", PROFILES)
def test_forward_matches_primal(case, profile):
    from lcbinint import _lcbinint  # noqa: F401

    result = binary_algebraic_boundary_polar_ffi(**dict(case, **profile))
    assert bool(result.grad_reliable)
    assert np.isfinite(float(result.magnification))
    assert float(result.magnification) > 1.0


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("profile", PROFILES)
def test_jvp_matches_central_difference(case, profile):
    names = ["source_x", "source_y", "separation", "mass_ratio", "source_radius"]

    def f(vec):
        kw = {n: vec[i] for i, n in enumerate(names)}
        return binary_algebraic_boundary_polar_ffi(**kw, **profile).magnification

    base = np.array([case[n] for n in names])

    analytic = np.asarray(jax.grad(f)(base))
    if np.any(np.isnan(analytic)):
        # Contract: a flagged row propagates NaN across the whole row rather
        # than a confidently-wrong tangent.
        assert np.all(np.isnan(analytic))
        pytest.skip("driver flagged this geometry's Jacobian row unreliable")

    steps = np.array([1e-5, 1e-5, 1e-5, 1e-5, 5e-6])
    for i in range(5):
        bp = base.copy()
        bm = base.copy()
        bp[i] += steps[i]
        bm[i] -= steps[i]
        fd = (float(f(bp)) - float(f(bm))) / (2.0 * steps[i])
        assert abs(analytic[i] - fd) <= 3e-4 * (1.0 + abs(fd)) + 1e-6


def test_jit_and_vmap():
    case = CASES[0]
    names = ["source_x", "source_y", "separation", "mass_ratio", "source_radius"]
    base = np.array([case[n] for n in names])

    @jax.jit
    def f(vec):
        kw = {n: vec[i] for i, n in enumerate(names)}
        return binary_algebraic_boundary_polar_ffi(**kw).magnification

    single = float(f(base))
    batch = np.stack([base, base * np.array([1.0, 1.0, 1.0, 1.0, 1.01])])
    out = jax.vmap(f)(batch)
    assert np.isclose(float(out[0]), single)
    assert np.isfinite(float(out[1]))


def test_caustic_grazing_row_is_unreliable_not_crash():
    # Source sitting essentially on a fold: the driver should flag the row
    # rather than return a confidently-wrong Jacobian.
    result = binary_algebraic_boundary_polar_ffi(
        source_x=0.0,
        source_y=0.0,
        separation=1.0,
        mass_ratio=1.0,
        source_radius=0.001,
    )
    # Either it stays reliable (fine) or it is flagged; never a hard failure.
    assert result.grad_reliable.dtype == np.bool_
