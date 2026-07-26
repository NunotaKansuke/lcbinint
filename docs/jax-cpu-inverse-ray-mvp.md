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

For a uniform source, use the still smaller \(M_0\)-only graph:

```python
from lcbinint_jax import binary_inverse_ray_uniform

result = binary_inverse_ray_uniform(
    source_x=0.2,
    source_y=0.1,
    separation=1.2,
    mass_ratio=0.1,
    source_radius=0.2,
    resolution=64,
    tile_size=16,
    tile_capacity=512,
)
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

## Four-engine positioning

The newer
[`benchmark_positioning.py`](../tests/diagnostics/jax_ir/benchmark_positioning.py)
compares the specialized uniform and linear paths against microLUX, native
`lcbinint`, and VBMicrolensing. It calibrates increasing JAX resolutions and
native source-bin counts against the same error budget before timing. The
microLUX configuration is also rejected when its achieved error misses that
budget. These are warm scalar CPU latencies; JIT compilation is excluded.

One development-machine run produced:

| Geometry and profile | JAX inverse ray | microLUX | native `lcbinint` | VBMicrolensing |
| --- | ---: | ---: | ---: | ---: |
| regular, uniform | 5.30 ms | 2.83 ms | 2.43 ms | 0.145 ms |
| resonant cusp, uniform | 8.17 ms | 5.60 ms | 2.94 ms | 0.160 ms |
| planetary far field, uniform | 3.60 ms | 1.70 ms | 1.25 ms | 0.052 ms |
| regular, linear \(c=0.4\) | 8.98 ms | 18.10 ms | 2.98 ms | 2.13 ms |
| resonant cusp, linear \(c=0.4\) | 13.64 ms | 347.87 ms | 3.24 ms | failed accuracy |
| planetary cusp, linear \(c=0.4\) | 8.45 ms | 288.88 ms | 2.22 ms | failed accuracy |
| planetary far field, linear \(c=0.4\) | 14.33 ms | 6.77 ms | 1.26 ms | 0.055 ms |

The corresponding differentiable timings were:

| Geometry and profile | JAX JVP / gradient | microLUX JVP / gradient |
| --- | ---: | ---: |
| regular, uniform | 7.18 / 23.47 ms | 4.13 / 4.85 ms |
| resonant cusp, uniform | 10.70 / 32.51 ms | 6.43 / 8.04 ms |
| regular, linear | 17.48 / 37.86 ms | 25.94 / 46.42 ms |
| resonant cusp, linear | 27.32 / 51.42 ms | 419.49 / 614.87 ms |
| planetary cusp, linear | 15.28 / 37.41 ms | 356.11 / 558.50 ms |
| planetary far field, linear | 25.04 / 71.33 ms | 12.86 / 35.99 ms |

This fixes the current position of the method:

- inverse rays do not win for uniform sources; one contour integration or a
  point/multipole shortcut is cheaper than rasterizing image area;
- inverse rays beat microLUX for linear limb darkening when a true
  finite-source calculation is required, by about 2x at the regular point and
  25--34x on the two tested caustic cusps;
- inverse rays lose away from caustics when microLUX can accept its
  point/quadrupole result;
- native `lcbinint` remains 3--4x faster than the JAX inverse-ray forward path;
- VBMicrolensing is the strongest forward-only baseline where its contour
  result meets the common accuracy check, but it is not differentiable in
  JAX and missed the selected cusp references.

The practical production architecture should therefore be hybrid: use a
differentiable point/multipole or contour path for uniform and safely
far-from-caustic epochs, and dispatch to the JAX inverse-ray moment kernel for
finite-source, limb-darkened caustic epochs. A broad held-out sweep is still
needed to calibrate that dispatch boundary. Native-bin errors are
non-monotonic, so the first-passing native setting remains an optimistic
latency comparison; the full calibration rows are retained by the harness.

## Experimental polar inverse rays and automatic dispatch

The JAX package now also contains an image-informed polar kernel. Physical
centre and source-limb roots are converted into stopped-gradient radial bands.
For each angular chunk, only bands near a sampled image boundary are activated.
The cells use the same affine positive-part moments as the Cartesian kernel,
with the local derivatives transformed from \((x,y)\) to \((r,\theta)\).
Forward, JVP, and reverse-mode differentiation all pass.

`binary_inverse_ray_auto` hides the coordinate choice:

- ordinary sources use the Cartesian macro-tile path;
- tiny sources above a stopped-gradient point-magnification threshold are sent
  directly to polar integration;
- Cartesian discovery overflow triggers a polar fallback.

The dispatcher defaults to the full two-coefficient square-root law. Callers
with a known uniform or linear profile can set `moment_mode="uniform"` or
`moment_mode="linear"`; the coordinate decision remains automatic while XLA
compiles the smaller one- or two-moment graph.

The focused benchmark
[`benchmark_polar.py`](../tests/diagnostics/jax_ir/benchmark_polar.py) uses
\(s=0.95,q=0.01,\rho=0.005\) at a high-magnification epoch with linear
limb darkening \(c=0.4\). A 512-bin native polar result, 95.42477782, defines
the same \(10^{-4}+10^{-4}\max(|A_{\rm ref}|,1)\) budget used elsewhere.

| Method / setting | Value | Absolute error | Warm forward |
| --- | ---: | ---: | ---: |
| JAX polar, 64 radial / 2048 angular | 95.44041365 | 0.01564, fails | 19.89 ms |
| JAX polar, 64 / 4096 | 95.43373464 | 0.00896, passes | 36.89 ms |
| JAX polar, 128 / 4096 | 95.43005217 | 0.00527, passes | 52.61 ms |
| JAX polar, 128 / 8192 | 95.42699341 | 0.00222, passes | 105.52 ms |
| microLUX, 80 annuli | 95.42525035 | 0.00047, passes | 88.06 ms |

The Cartesian JAX path overflowed even with 4096 tile slots on this point, so
polar integration materially expands the supported domain. The optimized
polar kernel packs only boundary cells into the expensive affine-moment path,
uses second-order interior moments, and subdivides boundary cells two by two.
This reduced the first passing forward configuration from 363.43 to 36.89 ms.
At that setting, JVP and reverse-gradient times were 43.59 and 117.17 ms,
versus 135.43 and 334.77 ms for microLUX. The matched-accuracy speedups are
therefore 2.39x forward, 3.11x JVP, and 2.86x reverse mode on this
high-magnification case.

## CPU speed optimization pass

The Cartesian reducer received the analogous sparse treatment. A boundary
tile now packs only its boundary cells into the affine positive-part formula.
Fully interior cells use a midpoint value plus the analytic second-order
correction

\[
E[\phi^p]\simeq \phi^p+
\frac{p(p-1)}{24}(\Delta_x^2+\Delta_y^2)\phi^{p-2}.
\]

Sixteen source-limb seeds replaced the previous default of 32. In a targeted
107-case sweep where the image topology changed somewhere across the source
disk, 8 samples missed one component by as much as 2.17%, while 16 reproduced
the 32-sample values in all tested cases. A frozen regression case protects
that boundary.

Re-running the four-engine matched-accuracy harness produced:

| Geometry/profile | JAX forward / JVP / grad | microLUX forward / JVP / grad |
| --- | ---: | ---: |
| regular, linear | 8.21 / 9.59 / 29.65 ms | 20.02 / 25.69 / 43.74 ms |
| resonant cusp, linear | 10.15 / 14.57 / 39.41 ms | 342.40 / 434.54 / 620.05 ms |
| planetary cusp, linear | 6.69 / 9.34 / 33.11 ms | 296.15 / 356.47 / 567.34 ms |
| planetary far field, linear | 13.48 / 15.39 / 57.38 ms | 7.91 / 10.73 / 36.16 ms |

The harness uses conservative fixed capacity 1024 and 32 limb seeds. Tight
validated buckets with 16 limb seeds further measured 5.12 / 7.75 / 19.40 ms
at the regular point and 8.82 / 12.46 / 33.89 ms at the resonant cusp.
Far-field and uniform epochs still favor contour or point/multipole methods;
that remaining gap belongs in the hybrid dispatcher rather than in additional
inverse-ray rasterization.

## Differentiable multipole fast path

The CPU dispatcher now has the missing far-field route.  The
`binary_hexadecapole` kernel evaluates the source centre and the same twelve
ring samples used by native `lcbinint`, then forms its quadrupole and
hexadecapole corrections including the linear and square-root limb-darkening
factors.  Image roots are found by the existing bounded solver in the forward
pass.  Their JVP is supplied by implicit differentiation of the two-dimensional
lens equation, so reverse mode does not differentiate the root iterations.

`binary_magnification_auto` first evaluates that expansion.  It accepts the
result only when all thirteen samples have the same physical-image count, all
roots are valid, the second- and fourth-order corrections remain hierarchically
ordered, and four times the absolute hexadecapole correction fits the requested
error budget.  Rejected epochs continue through the existing Cartesian/polar
dispatcher.  The discrete decision is stopped-gradient; the selected physical
magnification remains differentiable.

On the planetary far-field benchmark
\((x,y,s,q,\rho)=(0.3,0.4,1.4,10^{-3},0.01)\), \(c=0.4\), the expansion gives
2.17750899, agreeing with the high-accuracy VBMicrolensing value
2.1775089915.  One warm development-machine run measured:

| Kernel | Forward | Directional JVP | Value + gradient |
| --- | ---: | ---: | ---: |
| direct hexadecapole | 0.65 ms | 0.78 ms | 1.09 ms |
| full hybrid dispatcher | 1.34 ms | 1.18 ms | 2.38 ms |
| Cartesian inverse ray | 3.39 ms | 4.48 ms | 17.38 ms |

The earlier conservative four-engine run measured 7.91/10.73/36.16 ms for
microLUX on this case.  The hybrid route therefore removes the known far-field
loss without weakening the caustic inverse-ray path.  Its current
self-consistency and topology checks are empirical guards, not a proof that an
unsampled caustic cannot lie inside a source; the held-out dispatcher
calibration remains required.

## Stratified dispatcher pilot

The first broad JAX dispatcher sweep is documented in
[`jax-dispatcher-pilot-20260726`](../tests/diagnostics/results/jax-dispatcher-pilot-20260726/README.md).
It covers 144 close/resonant/wide rows over \(q=10^{-5}\ldots1\), three source
profiles, five caustic offsets, and a field control.  Native Cartesian
resolution convergence plus native Cartesian/polar agreement trusted 141
rows.

The original overflow fallback exposed a serious domain error: polar was
selected at 42 rows, including ordinary caustic crossings for which a
lens-centred radial representation is unsuitable.  Twelve support-valid
automatic results missed the common budget, eight of them polar.  Increasing
polar band capacity did not improve accuracy.

The calibrated hybrid policy is now fail-closed:

- the hexadecapole guard remains the first choice and had no false accept in
  the pilot;
- polar preselection requires stable sampled image topology and symmetric mass
  ratio at least \(5\times10^{-3}\), in addition to the existing tiny-source
  and high-magnification checks;
- Cartesian overflow no longer triggers an unconditional polar fallback in
  the production hybrid API;
- an unsupported hybrid result has `support_valid=False` and `NaN`
  magnification rather than returning a plausible partial integral;
- an extreme-planetary companion-matrix root fallback repairs physical limb
  roots missed by the bounded Ehrlich--Aberth solve.

The calibrated replay selected 69 multipole and 72 Cartesian rows.  Thirty-six
were explicitly unsupported at the current capacity, while six support-valid
rows exceeded the budget by only 1.01--1.16 times.  This is not full-domain
coverage yet: capacity/resolution buckets and coarse/fine gradient convergence
are the next required layer.  The separately validated elongated high-mag
case still selects polar and passes.

## Source-plane and expanded-capacity fallbacks

The first fallback layer now mirrors native `lcbinint` instead of assuming
that every rejected image-plane support can be repaired by a larger raster.
`binary_source_plane_quadrature` evaluates the differentiable point-source
solution over the source disk.  It provides both equal-area squared-radius
rings and the native tensor Gauss--Legendre chord mapping.  The chord rule is
the default because its nodes approach the limb efficiently.  Linear and
square-root limb darkening use the same surface-brightness law as the
image-plane moment kernels.  Coarse/fine disagreement is returned as an error
estimate and a failed rule is never silently accepted.

On the 36 unsupported rows in the pilot, chord orders 16/32 converged on 17
rows; every converged row met the independent native reference budget.  The
rule correctly rejected the sampled true caustic crossings, where point-source
singularities make low-order source-plane quadrature unreliable.  Replaying
the complete 141 trusted rows through the production dispatcher reduced
explicit invalid results from 36 to 21 with no new accuracy failure.  The
median warm time over that mixed replay remained 6.4 ms because the
source-plane branch runs only after image-support failure.

At the rescued equal-mass grazing point used by the regression test, the
16/32 chord pair took 119 ms forward, 112 ms for a directional JVP, and
118 ms for value plus reverse-mode gradient after compilation.  Native
`lcbinint` evaluated the same source-plane-routed reference in about 7 ms.
The JAX implementation is therefore a correctness fallback, not a competitor
to native source-plane batching; its value is that it preserves gradients and
runs only on the small rejected bucket.

The remaining complementary retry is exposed by
`expanded_cartesian_fallback=True`.  It compares 64/4096 and 128/16384
resolution/capacity buckets and accepts the fine value only when both supports
are valid and their magnifications agree within the requested budget.  Among
source-plane-rejected pilot rows this rescued seven additional rows with zero
false accepts.  Its warm scalar cost is hundreds of milliseconds, so it is
off by default: a trajectory fitter should collect failed epochs and evaluate
that sparse bucket together rather than putting every epoch through the large
executable.

## Conditional trajectory dispatcher

`binary_magnification_trajectory` evaluates a one-dimensional source path
with shared lens and source-profile parameters.  Its ordering follows the
native selector where the algorithms overlap:

1. differentiable hexadecapole with topology and error guards;
2. calibrated Cartesian or narrowly allowed polar inverse rays;
3. converged source-plane chord quadrature after image-support failure;
4. optional coarse/fine expanded Cartesian retry;
5. explicit `NaN` and `support_valid=False`.

Native `lcbinint` can cheaply reuse sampled caustic branches along a light
curve and uses refined caustic distance, local ghost diagnostics, and a
calibrated resolution model before this sequence.  Rebuilding that geometry
inside every JAX trace would erase the CPU advantage.  The JAX dispatcher
therefore uses the already-required 13-point hexadecapole topology and
self-consistency result as its first routing proxy.  The choice is
stopped-gradient; the selected physical calculation remains differentiable.

Two JAX execution layouts were measured before choosing the implementation.
A fixed-capacity layout first `vmap`ed all hexadecapole solves, packed rejected
indices, then scattered the expensive results.  A scalar-conditional layout
uses `lax.map`, retaining the scalar `lax.cond` at every epoch.  On a 64-epoch
half-hex/half-Cartesian path the packed layout took 200 ms versus 178 ms for
the scalar layout.  With only four rejected epochs it took 99 versus 71 ms;
at 256 epochs it regressed to 837 versus 180 ms.  Batched root solving and
pack/scatter overhead dominate on CPU, so the production trajectory API uses
the simpler scalar-conditional layout.

The trajectory default is the 96/4096 Cartesian bucket with 24 limb seeds.
The scalar API retains 64/1024 for low latency.  Its Cartesian execution
backend defaults to `auto`: the typed C++ FFI is used for the real kernel on
CPU when available, with pure JAX retained as the portable fallback.  Binary
image roots have the same independent backend choice.  The C++ root handler
constructs the quintic, polishes and deduplicates physical images, and returns
an analytic implicit root Jacobian through a JAX `custom_jvp`; it is shared by
hexadecapole, Cartesian/polar discovery, and source-plane quadrature.  On a
64-epoch resonant, linearly limb-darkened path, the trajectory default had
zero value-budget failures against a `Tol=1e-7` VBMicrolensing reference.  A
repeat after connecting both FFI backends to the dispatcher measured:

| Engine | 64-epoch forward |
| --- | ---: |
| native `lcbinint`, fixed Cartesian 64 | 130 ms |
| JAX conditional trajectory, pure JAX | 573 ms |
| JAX conditional trajectory, C++ FFI roots/discovery/cells | 158 ms |
| VBMicrolensing | 3.04 s |
| microLUX, 80 annuli | 14.37 s |

The FFI trajectory is 3.64 times faster than the pure-JAX trajectory,
91.1 times faster than microLUX, 19.3 times faster than VBMicrolensing, and
1.20 times slower than native on this pilot.  On a 16-epoch subset, FFI
separation-JVP/value-plus-gradient took 123/134 ms, versus 269 ms/1.06 s for
pure JAX and 13.25/28.07 s for microLUX.  Compilation is excluded from all
these warm timings.  FFI and JAX produced identical method counts
(32 hexadecapole and 32 Cartesian), no invalid epochs, and indistinguishable
VBMicrolensing error-budget ratios.  The reproducible harness is
`tests/diagnostics/jax_ir/benchmark_trajectory.py`.

Roots, Cartesian discovery, and fixed-support integration are now also
available as one fused FFI epoch handler.  It returns the same capacity/root
diagnostics and a seven-parameter analytic Jacobian, allowing the dispatcher
to preserve polar overflow routing.  All 84 support-valid rows in the
close/resonant/wide held-out sweep passed staged-versus-fused value, JVP, and
gradient budgets.

The measured speed gain from this fusion is small: a representative epoch
changed from 8.21 to 8.05 ms forward and from 24.26 to 23.96 ms for value plus
gradient.  The 64-epoch trajectory measured 156.83 ms fused versus 157.68 ms
before fusion, while native measured 129.77 ms.  This localizes the remaining
gap to C++ cell traversal and moment arithmetic rather than JAX intermediate
arrays or typed-FFI call overhead.

The subsequent portable CPU hot-loop pass eliminated most of that gap:

- fixed quarter/half powers use products and square roots instead of general
  `pow`;
- moment mode and boundary subdivision are compile-time specializations;
- subcell offsets are reused;
- tile-row lens classification is separated into SIMD-friendly arrays;
- the Jacobian path creates Jets only for contributing cells, and uniform
  interior cells require no lens-map Jet.

Representative uniform/linear/two-coefficient forward times improved from
4.72/6.09/30.29 ms to 3.13/3.83/14.06 ms.  Value-plus-gradient improved from
14.70/19.22/74.86 ms to 5.05/14.72/37.14 ms.

The trajectory dispatcher now sends all selected Cartesian epochs through one
masked FFI call. Hexadecapole selection remains vector-valued and stopped
gradient; polar-preselected or invalid-support rows retain the scalar
dispatcher, so method and fallback decisions are unchanged. The C++ handler
parallelizes independent active epochs with OpenMP, respects
`OMP_NUM_THREADS`, and caps its own team at 32 threads. The Skowron--Gould
scratch variables were made call-local before enabling this path.

On the 32-thread benchmark host, the 64-epoch linear trajectory measures
9.14 ms forward, 24.04 ms JVP, and 32.64 ms value-plus-gradient. The prior
scalar FFI path measured 97.94/93.39/103.12 ms. Native `lcbinint` forward is
134.84 ms, VBMicrolensing forward is 3.02 s, and microLUX measures
14.19/12.66/26.30 s. All 64 batched FFI values pass the common VBM error
budget. With `OMP_NUM_THREADS=1`, batching has essentially no forward
overhead (97.88 ms versus 98.01 ms scalar); the large throughput result is
therefore explicitly a multicore comparison.

With `--profile uniform`, batched FFI forward/JVP/value-plus-gradient measures
12.30/11.58/21.84 ms. Native forward is 130.91 ms and has two common-budget
misses; VBMicrolensing measures 46.08 ms, so the batched inverse-ray path is
3.75 times faster on this trajectory. The 80-annulus microLUX configuration
measures 14.07/12.80/25.15 s but is not an optimized uniform-source baseline.
FFI has zero uniform accuracy-budget failures.

The two hex/Cartesian switch boundaries in a 32-point replay were value-safe:
the four values adjacent to the boundaries used at most 0.43 of their error
budgets.

The initial source-coordinate gradient replay appeared to pass only 29/32
points.  The largest reported failure was a calibration error: its caustic
distance was \(0.984\rho\), while the reference step was \(0.03\rho\), so the
central difference straddled the narrow limb-contact layer.  The VBM estimate
changed from 160.24 at that step to 149.58--149.67 for local steps
\(10^{-5}\ldots3\times10^{-5}\); JAX AD gave 149.56.

The other two misses exposed a real integration defect.  Boundary cells used
only an affine reconstruction of
\(\phi=1-|u(z)-u_0|^2/\rho^2\), and fully interior limb-darkening cells omitted
the Laplacian term in their second-order moment.  Worse, a tile with no
boundary cells fell back to a different midpoint rule.  The corrected kernel:

- uses the harmonic lens-map identity
  \(\Delta\phi=-2\|J_u\|_F^2/\rho^2\), requiring no extra lens-map call;
- includes both \(f'(\phi)\Delta\phi\) and
  \(f''(\phi)|\nabla\phi|^2\) in every interior brightness moment;
- applies one consistent interior rule in boundary and boundary-free tiles;
- evaluates only detailed cells on a true curved lens map, using a calibrated
  3-by-3 subcell rule for uniform/linear profiles and retaining 4-by-4 when
  the singular square-root term is active;
- treats fully-inside cells with large relative variation as part of the
  detailed band for the singular \(\phi^{1/4}\) square-root profile;
- packs detailed cells in 8/16/32/64/128/256 static tiers, with capacity for
  every cell in a tile; there is no overflow retry or lower-order fallback.

With the corrected local reference, the same 32-point trajectory passes all
32 gradient budgets; the maximum budget ratio is 0.88 and the median is 0.02.
The 64-point value replay also has zero failures and a maximum budget ratio of
0.10.  Replaying all 141 trusted rows from the broader dispatcher pilot also
removed the six previous support-valid marginal misses; the remaining 21
invalid rows are still explicitly failed support cases.  No resolution or
source-plane fallback is involved in this correction.

A subsequent boundary-kernel profile separated quadrature order from support
resolution.  On the 96 trusted, support-valid Cartesian rows from the frozen
pilot, replacing 4-by-4 by 2-by-2 introduced no new forward-accuracy failure,
but the local caustic-gradient replay showed that linear profiles require
3-by-3 and the square-root term still requires 4-by-4.  The production default
therefore selects 3-by-3 for uniform/linear profiles and 4-by-4 only when
`limb_d != 0`.  This selection is static in the automatic Cartesian
dispatcher, so reverse mode compiles only the selected rule; the general
fixed-support default remains 4-by-4 for exact agreement with its specialized
entry points.  The fixed-support research API also exposes
`boundary_subdivision=0`: it partitions detailed cells by a stopped-gradient
curvature indicator and evaluates them directly at 2-by-2 or 4-by-4 order.
That adaptive rule is faster for forward/JVP workloads, but is not the default
because its extra packed branches made reverse-mode slower.

## Integrated polar, multipole, and trajectory FFI

The production CPU route now covers the three remaining dispatcher layers.
The polar kernel streams radial cells after discovering and merging the image
bands, the batched hexadecapole kernel evaluates its 13 samples in one call,
and the integrated trajectory kernel performs hexadecapole acceptance, polar
preselection, and Cartesian/polar execution behind one typed FFI boundary.
Each numerical kernel has an analytic seven-input Jacobian connected to JAX
with `custom_jvp`; selection and support construction remain deliberately
stopped-gradient.

The integrated call returns method, error, support, and fallback diagnostics
for every epoch. Ordinary rows stay entirely in the C++ batch. An explicit
invalid-support or capacity result sends only that row through the existing
scalar dispatcher, retaining its source-plane and enlarged-support behavior
without making retry work part of the fast path.

On the 32-thread benchmark host, the 64-epoch linear trajectory now measures
9.48 ms forward, 21.70 ms JVP, and 27.84 ms value-plus-gradient. Native
`lcbinint` forward is 135.20 ms, VBMicrolensing is 2.96 s, and microLUX
measures 14.05/13.07/27.00 s. The uniform trajectory measures
5.86/8.88/16.60 ms, versus 131.80 ms native, 45.76 ms VBMicrolensing, and
13.85/12.89/26.17 s microLUX. These figures include dispatcher selection and
all epochs, but exclude compilation; compile plus first execution is
0.39 s for linear and 0.30 s for uniform.

The focused polar case improves from 24.03/81.61 ms to 13.55/17.76 ms for
forward/value-plus-gradient. A 64-epoch hexadecapole batch improves from
2.085/3.246 ms to 0.094/0.295 ms. The eight-configuration held-out sweep
contains 120 rows and reports no value, JVP, gradient, discovery,
point-source, hexadecapole, or integrated-trajectory failures. The maximum
integrated value/JVP/gradient error consumes only
0.000011/0.0036/0.015 of the corresponding budget.

## Higher-order trajectories

The fused trajectory FFI now accepts an epoch-dependent separation array
instead of a single scalar separation. Its analytic Jacobian still has seven
columns per epoch, but the separation column now represents
\(\partial A_i/\partial s_i\). A scalar separation is broadcast by the public
API, preserving the static-lens interface. This makes lens orbital motion a
single C++ call rather than a `vmap` over scalar FFI calls.

`lcbinint_jax.higher_order` implements the native trajectory conventions in
pure `float64` JAX:

- circular and bound Kepler lens orbital motion;
- annual, terrestrial, and VBM-table satellite parallax;
- circular-elements, Kepler-elements, circular-velocity, and Kepler-velocity
  xallarap;
- independent binary-source tracks, source-CoM xallarap coordinates, and the
  tangent-trajectory-offset binary-source convention;
- flux-weighted binary-source magnification through two fused trajectory
  calls.

The geometry layer returns `source_x(t)`, `source_y(t)`, and `separation(t)`.
Consequently the custom-JVP inverse-ray kernel supplies derivatives with
respect to lens-plane geometry, while ordinary JAX differentiation propagates
them to \(t_0,t_E,u_0,\alpha,\boldsymbol{\pi}_E\), orbital-state, xallarap,
source-mass-ratio, and flux-ratio parameters.

Tests compare every lens-orbit and xallarap mode individually with the native
geometry. Circular/Kepler lens motion agrees to \(5\times10^{-15}\), all four
xallarap modes to \(3\times10^{-12}\), and annual, terrestrial, satellite, and
the combined annual+terrestrial+Kepler-orbit+Kepler-xallarap case to machine
precision. Both binary-source coordinate conventions also agree to machine
precision. A 64-epoch warm CPU check measured 5.78/20.54 ms
forward/value-plus-gradient for a static trajectory and 5.71/21.58 ms for the
same trajectory with circular lens motion. Geometry evolution was negligible
in forward mode and added about five percent to this reverse-mode measurement;
the dominant cost remains magnification integration.

## Triple-lens automatic magnification milestone

The differentiable triple-lens path now has a sparse production-oriented CPU
kernel in addition to the original dense reference. A native degree-10 solver
samples the source centre and limb, a four-neighbour BFS discovers connected
image-plane macro-tiles, and a fused C++ FFI integrates only those tiles.
Root selection, BFS topology, and grid size are stopped-gradient. Within the
fixed support, a forward-mode analytic jet propagates derivatives through the
source coordinates, both mass ratios, both separations, tertiary angle, source
radius, and both brightness coefficients. Both native center-of-mass and VBM
geometry conventions are supported.

The scalar public entry point is:

```python
import jax
import lcbinint_jax as lj

jax.config.update("jax_enable_x64", True)
result = lj.triple_inverse_ray_adaptive(
    0.2, 0.3,       # source x, y
    1.0, 0.1,       # s, q
    0.03, 0.7, 0.8, # q2, sep2, angle
    0.2,             # rho
    limb_c=0.3,
    limb_d=0.2,
)
value = result.magnification
gradient = jax.grad(
    lambda q2: lj.triple_inverse_ray_adaptive(
        0.2, 0.3, 1.0, 0.1, q2, 0.7, 0.8, 0.2,
        limb_c=0.3, limb_d=0.2,
    ).magnification
)(0.03)
```

The production entry points are now `triple_magnification_auto` for one epoch
and `triple_magnification_batch` for a trajectory. They dispatch each epoch
through a three-level differentiable hierarchy:

1. degree-10 point-source magnification with implicit root derivatives;
2. a 13-point hexadecapole expansion;
3. sparse Cartesian inverse rays.

Method codes are 0, 1, and 2 respectively. Point and hex batches use fused C++
FFIs with analytic Jacobians. The inverse-ray FFI receives an `active` mask, so
epochs accepted by either fast path do no cell traversal. All three branches
support `jax.jit`, forward AD, and reverse AD through custom JVPs; the discrete
method decision is stopped-gradient.

```python
result = lj.triple_magnification_batch(
    source_x, source_y,
    1.0, 0.1, 0.03, 0.7, 0.8, 0.02,
    limb_c=0.3, limb_d=0.2,
)
value = result.magnification
method = result.method
```

The default relative budget is \(10^{-4}\). Point acceptance uses the native
high-derivative indicator with a factor-four margin. Hex acceptance additionally
requires stable image topology, ordered quadrupole/hex corrections, and the
hex correction to fit the error budget. Multipoles are rejected when the point
magnification reaches 100: a frozen calibration found that a nearby caustic can
invalidate the local expansion while its last term still looks small. Those
epochs go directly to Cartesian inverse rays. The Cartesian discovery limit is
large enough for the calibrated high-magnification cases, but initially
reserves only 4096 tiles and grows on demand.

The zero-tertiary-mass map agrees with the binary map and all six image-plane
derivatives to \(3\times10^{-15}\). For
\((s,q,q_2,s_2,\psi,\rho)=(1,0.1,0.03,0.7,0.8,0.2)\), the default sparse
uniform-source result is 3.135506 versus native Cartesian 3.135455, a relative
difference of \(1.6\times10^{-5}\). All ten scalar derivatives agree with
local finite differences at roughly \(7\times10^{-5}\) to
\(8\times10^{-4}\) relative error in this case.

The final frozen fast-path calibration contains 510 triple-lens/profile rows
with forced Cartesian-512 references. At relative tolerance \(10^{-4}\), the
default rules accepted 30 point and 186 hex rows. None exceeded the requested
tolerance; the maximum accepted relative error was \(2.77\times10^{-5}\).
The three formerly bad hex rows all had point magnification about 152 and are
now rejected by the high-magnification guard.

On the current benchmark host, a warm jitted 64-epoch point batch took
0.038 ms and a 64-epoch hex batch 0.329 ms. A far trajectory accepted entirely
by the dispatcher took 0.300 ms. A 64-epoch caustic-crossing trajectory, for
which all epochs required resolution-96 inverse rays, took 13.57 ms through
the dispatcher versus 15.71 ms through the raw batch inverse-ray entry point.
Native forced Cartesian-96 took 177 ms and native auto took 1.25 s on that
same trajectory. Against forced native Cartesian-192, the JAX values had
median/max relative differences \(1.16\times10^{-5}\) and
\(1.10\times10^{-4}\). These are warm measurements on one host, not portable
guarantees.

## Current limitations

- The triple dispatcher currently has point, hex, and Cartesian branches, but
  no triple polar branch. Very high magnification with tiny sources is
  therefore accurate but can be substantially slower than a future polar or
  source-plane implementation.
- The 510-row calibration is broad enough to set the current default fast-path
  thresholds, but it does not prove a universal bound outside the sampled lens
  ratios, source radii, and limb profiles.
- The current `support_valid` flag detects root and tile-capacity failures; it
  is not a numerical-accuracy or gradient-convergence guarantee; use the
  coarse/fine diagnostic for that decision.
- Polar `support_valid` likewise checks band/root capacity, not quadrature
  convergence. The calibrated hybrid dispatcher therefore uses it only for a
  narrow elongated-arc regime and never as an unconditional overflow fallback.
- The first 64/4096 to 128/16384 Cartesian retry is calibrated only on the
  pilot sweep.  A larger held-out trajectory sweep is still required before
  enabling it by default.
- The close/resonant/wide held-out gradient sweep passes, but the calibrated
  parameter domain is finite; substantially more extreme mass ratios or
  source radii require a new accuracy sweep.
- Very small image components may still be missed by finite source-limb
  sampling; the planned halo and topology sweeps must validate this.
- The public API is experimental and may change.
- Root and support selection are deliberately stopped-gradient.

Use
[`tests/diagnostics/jax_ir/benchmark_cpu.py`](../tests/diagnostics/jax_ir/benchmark_cpu.py)
for compile-separated forward, JVP, and value-plus-gradient CPU timing.
The full implementation and validation plan is
[JAX CPU differentiable inverse-ray engine](design/jax-cpu-inverse-ray-plan.md).
