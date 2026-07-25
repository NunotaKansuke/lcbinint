# Experimental JAX CPU inverse-ray MVP

The `lcbinint_jax` package is an experimental, standalone JAX implementation
of binary-lens finite-source inverse-ray integration. It does not import the
native `lcbinint` extension and does not require GSL.

Enable JAX 64-bit mode before evaluation:

```python
import jax

jax.config.update("jax_enable_x64", True)

from lcbinint_jax import binary_inverse_ray

result = binary_inverse_ray(
    source_x=0.2,
    source_y=0.1,
    separation=1.2,
    mass_ratio=0.1,
    source_radius=0.2,
    limb_c=0.4,
    limb_d=0.1,
    resolution=32,
    tile_size=16,
    tile_capacity=256,
    limb_samples=32,
)

print(result.magnification)
print(result.moments)
print(result.support_valid)
```

The calculation:

1. solves the binary-lens quintic, or its real-axis quartic limit, at the
   source centre and fixed source-limb angles;
2. filters physical images with the original lens equation;
3. discovers a fixed-capacity set of Cartesian macro-tiles;
4. integrates uniform and limb-darkening image-plane moments in one pass;
5. differentiates the ray integral while stopping gradients through discrete
   root and tile-support selection.

For a scalar loss:

```python
import jax.numpy as jnp

def model(parameters):
    x, y, s, q, rho, c, d = parameters
    return binary_inverse_ray(
        x, y, s, q, rho, c, d,
        resolution=32,
        tile_size=16,
        tile_capacity=256,
        limb_samples=32,
    ).magnification

parameters = jnp.array([0.2, 0.1, 1.2, 0.1, 0.2, 0.4, 0.1])
value, gradient = jax.value_and_grad(model)(parameters)
```

Use the coarse/fine diagnostic when a numerical acceptance decision is
required:

```python
from lcbinint_jax import binary_inverse_ray_convergence

diagnostic = binary_inverse_ray_convergence(
    parameters,
    direction=jnp.array([0.2, -0.1, 0.05, 0.02, 0.0, 0.1, -0.05]),
    coarse_resolution=64,
    fine_resolution=128,
    coarse_tile_capacity=1024,
    fine_tile_capacity=4096,
)

print(diagnostic.value_converged)
print(diagnostic.moments_converged)
print(diagnostic.gradient_converged)
```

The compared observables are magnification and the three image moments
normalized by the unlensed source area. Gradient convergence is checked only
when `direction` is supplied.

## Native reference validation

The focused test suite directly compares the JAX engine with
`lcbinint.binary_ray_shooting`, including quadratic limb darkening and a source
centred close to a resonant-caustic cusp. One 128-bin development comparison
gave:

| Case | JAX | native `lcbinint` | Absolute difference |
| --- | ---: | ---: | ---: |
| regular, uniform | 4.4812779 | 4.4809341 | 0.0003438 |
| regular, \(c=0.4,d=0.1\) | 4.4704993 | 4.4700322 | 0.0004671 |
| near cusp, uniform | 8.8565603 | 8.8563649 | 0.0001954 |

At the regular limb-darkened point, a JAX directional derivative of
approximately -1.7926 also agrees with a 512-bin native central difference to
within the development tolerance. These are regression checkpoints, not a
completed calibration over the full parameter domain.

## Current limitations

- Binary lenses only.
- The current `support_valid` flag detects root and tile-capacity failures; it
  is not a numerical-accuracy or gradient-convergence guarantee; use the
  coarse/fine diagnostic for that decision.
- Resolution and capacity buckets are not calibrated yet.
- Very small image components may still be missed by finite source-limb
  sampling; the planned halo and topology sweeps must validate this.
- The public API is experimental and may change.
- Root and support selection are deliberately stopped-gradient.

Use
[`tests/diagnostics/jax_ir/benchmark_cpu.py`](../tests/diagnostics/jax_ir/benchmark_cpu.py)
for compile-separated forward, JVP, and value-plus-gradient CPU timing.
The full implementation and validation plan is
[JAX CPU differentiable inverse-ray engine](design/jax-cpu-inverse-ray-plan.md).
