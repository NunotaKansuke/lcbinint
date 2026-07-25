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

For microLUX-compatible linear limb darkening, use the specialized path:

```python
from lcbinint_jax import binary_inverse_ray_linear

result = binary_inverse_ray_linear(
    source_x=0.2,
    source_y=0.1,
    separation=1.2,
    mass_ratio=0.1,
    source_radius=0.2,
    limb_c=0.4,
    resolution=64,
    tile_size=16,
    tile_capacity=512,
)
```

It accumulates only \(M_0\) and \(M_{1/2}\). The general
`binary_inverse_ray` path continues to support the two-coefficient
square-root law.

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

## Matched-accuracy CPU benchmark

The cross-engine harness
[`benchmark_engines.py`](../tests/diagnostics/jax_ir/benchmark_engines.py)
selects the lowest tested setting whose absolute error satisfies

\[
10^{-4}+10^{-4}\max(|A_{\rm ref}|,1).
\]

One warm scalar-epoch run on the development x86-64 CPU with JAX 0.6.2 gave:

| Case | native setting / time | JAX setting / forward | JAX JVP | JAX value + gradient |
| --- | ---: | ---: | ---: | ---: |
| regular, uniform | 32 bins / 2.30 ms | 64 / 30.48 ms | 72.70 ms | 132.49 ms |
| regular, square-root limb | 32 bins / 2.45 ms | 128 / 87.12 ms | 223.52 ms | 441.31 ms |
| resonant cusp, uniform | 48 bins / 2.72 ms | 64 / 51.63 ms | 131.55 ms | 247.97 ms |

For context, seven-parameter native central differences took 33.56, 34.53,
and 39.55 ms on the same three cases. The JAX reverse pass is therefore about
4.0, 12.8, and 6.3 times slower than fourteen native forward evaluations in
this initial implementation. JAX compilation was measured separately and is
not included in the warm times.

The JAX forward breakdown was:

| Case | Discovery | Fixed-support integration |
| --- | ---: | ---: |
| regular, uniform | 3.64 ms | 21.28 ms |
| regular, square-root limb | 11.13 ms | 70.39 ms |
| resonant cusp, uniform | 5.76 ms | 45.97 ms |

The current bottleneck is thus the ray integral, not polynomial roots or
macro-tile discovery. The native grid's reference error is non-monotonic in
some bins, so choosing the first oracle-passing setting is optimistic; the
full calibration arrays are retained in the JSON benchmark output. Even with
more conservative native settings, the present JAX kernel is not yet
competitive on scalar CPU latency.

VBMicrolensing uniform-source forward times were about 0.16--0.18 ms at the
same requested error budget. Its installed Python API did not reproduce
`lcbinint`'s two-coefficient square-root-law convention reliably, so no
limb-darkened VBMicrolensing timing is presented as a matched physical case.

## Optimized comparison with microLUX

The linear-limb-darkening path was optimized after the initial benchmark:

- tiles without boundary cells bypass the affine positive-part formula;
- cells just inside the limb on a boundary tile retain affine moment
  integration instead of reverting to midpoint brightness;
- the linear specialization removes the unused \(M_{1/4}\) graph;
- constant moment powers allow XLA to simplify the closed-form expressions;
- checkpointing the active-tile body reduces reverse-pass memory traffic.

The reproducible harness is
[`benchmark_microlux_linear.py`](../tests/diagnostics/jax_ir/benchmark_microlux_linear.py).
It compares against local microLUX commit
`a241b8c2f2198bc4846c0fa66e2bcdcf5cfa6428`, using JAX 0.6.2 on one CPU
device. Both engines use \(c=0.4\), and each reported setting satisfies

\[
10^{-4}+10^{-4}\max(|A_{\rm ref}|,1).
\]

Warm scalar timings were:

| Case | Engine setting | Forward | Directional JVP | Value + gradient |
| --- | --- | ---: | ---: | ---: |
| regular | inverse ray, resolution 64 | 8.48 ms | 16.77 ms | 31.06 ms |
| regular | microLUX, 10 annuli | 19.66 ms | 24.55 ms | 48.27 ms |
| resonant cusp | inverse ray, resolution 64 | 13.71 ms | 27.15 ms | 51.30 ms |
| resonant cusp | microLUX, 80 annuli | 343.68 ms | 418.33 ms | 595.52 ms |

The resulting inverse-ray speedups were:

| Case | Forward | Directional JVP | Value + gradient |
| --- | ---: | ---: | ---: |
| regular | 2.32x | 1.46x | 1.55x |
| resonant cusp | 25.06x | 15.41x | 11.61x |

For the regular case, the reference magnification was 4.47211877. The
inverse-ray value was 4.47262454 and the microLUX value was 4.47246599; both
passed the 0.00054721 budget.

At the cusp, VBMicrolensing's limb-darkened value disagreed with the
high-resolution native, JAX, and microLUX convergence sequence. The benchmark
therefore uses their consensus near 9.12500 rather than treating one solver as
an oracle. The inverse-ray value at resolution 64 was 9.12574905. microLUX's
default 10-annulus result missed the matched budget; 80 annuli produced
9.12483904 and passed.

These two cases meet the microLUX CPU target, including reverse mode. They do
not yet establish dominance over the full \((s,q,\rho,w,c)\) domain; a
stratified held-out sweep remains required.

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
