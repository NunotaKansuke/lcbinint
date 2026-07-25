# JAX CPU differentiable inverse-ray engine: implementation plan

Status: implementation in progress

Planning branch: `feature/jax-cpu-inverse-ray-plan`

Implementation branch: `feature/jax-cpu-inverse-ray-mvp`

Initial scope: static binary lens, finite circular source, CPU, JAX, differentiable

## 1. Mission

Build a finite-source microlensing engine whose defining properties are:

1. It evaluates finite-source magnification by inverse-ray integration in the
   image plane.
2. It is fast on ordinary multicore CPUs after JIT compilation.
3. It is implemented in JAX and supports `jit`, `vmap`, `jvp`, `grad`, and
   scalar-loss reverse-mode differentiation.
4. It evaluates limb darkening in the same inverse-ray pass as the uniform
   source, without concentric-source integrations.
5. It reports numerical convergence and topology/capacity failures instead of
   silently returning an under-resolved value.

The first deliverable is not a differentiable rewrite of every current
`lcbinint` feature. It is a focused binary-lens inverse-ray kernel and benchmark
suite. Point-source and multipole routing may be added later, but they must not
be used to hide the performance or accuracy of the inverse-ray kernel during
development.

## 2. Why inverse rays

For source centre \(w_c\), radius \(\rho\), lens parameters \(\theta\), and
image-plane position \(z\), define

\[
r^2(z) = \frac{|w(z;\theta)-w_c|^2}{\rho^2},
\qquad
\phi(z) = 1-r^2(z).
\]

Surface brightness is conserved by lensing, so the finite-source flux is an
image-plane integral. The two-coefficient limb-darkening law already used by
`lcbinint` can be written

\[
I(r) =
1-c(1-\mu)-d(1-\sqrt{\mu})
= (1-c-d)+c\phi^{1/2}+d\phi^{1/4},
\quad \mu=\sqrt{\phi},
\]

inside the source and zero outside. One inverse-ray traversal can therefore
accumulate three image-plane moments:

\[
M_0 = \int [\phi]_+^0\,d^2z,\qquad
M_{1/2} = \int [\phi]_+^{1/2}\,d^2z,\qquad
M_{1/4} = \int [\phi]_+^{1/4}\,d^2z.
\]

Here \([x]_+^0\) means the indicator \(1[x>0]\). The magnification is

\[
A(c,d) =
\frac{(1-c-d)M_0+cM_{1/2}+dM_{1/4}}
{\pi\rho^2(1-c/3-d/5)}.
\]

This decomposition is central to the project:

- uniform, linear, square-root, and the current two-coefficient profile share
  one ray pass;
- derivatives with respect to \(c\) and \(d\) are almost free;
- multiple passband coefficients can reuse the same three moments;
- the expensive operation is only the lens map at each image-plane sample.

The engine should expose the moments internally and form passband-specific
magnifications afterward.

## 3. What existing implementations teach us

### 3.1 `lcbinint`

The current Cartesian engine starts from physical and caustic-derived image
seeds, walks only connected image components, applies a sub-cell source-limb
correction, prevents fold fills from crossing the critical curve, and avoids
double counting through a claimed-cell registry. Its cost is tied to actual
image area instead of a global image-plane bounding box.

Useful ideas to retain:

- image-centred seed discovery;
- source-limb/caustic augmentation for newly created fold images;
- connected-component traversal rather than global ray shooting;
- Cartesian and polar alternatives for different image topology;
- boundary correction;
- coarse/fine or embedded error estimates;
- explicit convergence and overflow diagnostics.

Parts not suitable for direct JAX translation:

- dynamically growing C++ vectors, queues, hash tables, and row scratch space;
- data-dependent recursion or allocation;
- hard cell-membership decisions as the only representation of a moving
  source boundary;
- variable-sized outputs and retry loops that change array shapes.

### 3.2 microJAX

The current microJAX ICRS implementation maps sampled source-limb points to the
image plane, builds a fixed number of polar regions, and evaluates a dense
radial-by-angular grid in every region. It uses `lax.scan`, chunked `vmap`,
checkpointing, an Ehrlich-Aberth polynomial solver with implicit root JVPs, and
custom JVPs around source-boundary and limb-darkening functions.

That design is well suited to GPU throughput. Its CPU weakness is also clear:
large dense region grids perform many lens-map evaluations that do not
contribute to the image. Our design must reduce the number of evaluated rays,
not merely translate the same dense grid to a different device.

### 3.3 Design consequence

The proposed engine combines:

- microJAX's static-shape, pure-JAX root solving and recomputation strategy;
- `lcbinint`'s sparse image-centred traversal and explicit numerical safety;
- a new differentiable partial-cell moment rule for the source boundary.

It should be independently implemented. If implementation is later copied or
adapted from an external MIT-licensed project, attribution and license
requirements must be handled explicitly.

## 4. Proposed algorithm

The forward calculation has two logically separate stages.

### 4.1 Stage A: image discovery

Stage A finds a conservative, finite set of image-plane macro-tiles that may
map into the source. It determines computational support, not flux.

1. Solve the binary-lens polynomial for the source centre.
2. Solve it at a static bucket of source-limb angles.
3. Filter physical images by lens-equation residual.
4. Convert all physical centre and limb images to integration-lattice
   macro-tile IDs. Initial macro-tile discovery does not require root-branch
   matching.
5. Seed a bounded macro-tile flood fill from those IDs.
6. Keep a tile when its centre/corners map inside the source, straddle the
   source boundary, or are adjacent to an already active boundary tile.
7. Expand the final set by a configurable one- or two-tile halo.

All arrays have static capacities. Candidate capacities are bucketed, for
example:

| Bucket | Limb samples | Maximum active macro-tiles |
| --- | ---: | ---: |
| S | 16 | 32 |
| M | 32 | 128 |
| L | 64 | 512 |
| XL | 128 | 2048 |

These values are hypotheses, not frozen defaults. Calibration decides them.
Every discovery call returns:

- packed tile IDs and an active mask;
- physical-root residuals;
- queue overflow and tile-capacity overflow flags;
- a source-limb coverage diagnostic;
- estimated image count/topology metadata.

#### Differentiation policy for discovery

Tile IDs, root ordering, masks, and queue decisions are discrete. They are
treated as stopped-gradient support decisions. This is valid locally only if
the active tile set includes a halo on which the integrand is identically zero.
The value and derivative then do not depend on the exact outer support.

This is not permission to ignore support failures. A nonzero contribution in
the outer halo, a capacity overflow, or an unmatched limb branch sets
`converged=False`.

For the inverse-ray MVP, the root solve and all root-derived support are
wrapped in `stop_gradient`. The physical integral is differentiated through
the lens map at the rays, not through the algorithm that found its conservative
support. Implicit root derivatives are deferred until a differentiable
point-source or multipole API is added.

### 4.2 Stage B: differentiable tile integration

Use a globally aligned Cartesian ray lattice. A macro-tile contains a small
fixed array such as \(8\times8\) or \(16\times16\) cells. Active macro-tiles
are processed with `lax.scan`; cells within one tile are vectorized so XLA can
generate SIMD code.

For each cell:

1. Evaluate the binary lens map \(w(z;\theta)\).
2. Compute \(\phi=1-|w-w_c|^2/\rho^2\).
3. Evaluate the image-plane derivatives
   \(\partial\phi/\partial x\) and \(\partial\phi/\partial y\).
4. Accumulate \(M_0\), \(M_{1/2}\), and \(M_{1/4}\).

Cells are classified into three numerical cases:

- safely outside: zero contribution;
- safely inside: fixed low-order tensor quadrature of the three brightness
  bases;
- boundary cell: analytic or semi-analytic partial-cell integration under a
  locally affine approximation to \(\phi\).

The classification is a stopped-gradient numerical choice. Neighboring
formulae must agree to the requested discretization order at their switching
thresholds.

### 4.3 Boundary-cell primitive

A hard `phi > 0` mask gives a piecewise-constant area and an unusable gradient.
A sigmoid mask gives a tunable bias and couples accuracy to an artificial
smoothing width. Neither should be the primary production rule.

The preferred boundary model is

\[
\phi(x,y)\simeq a+bx+cy
\]

inside one ray cell. For each \(p\in\{0,1/2,1/4\}\), integrate

\[
\int_{\mathrm{cell}} [a+bx+cy]_+^p\,dx\,dy.
\]

For non-degenerate \(b,c\), this integral has an inclusion-exclusion form in
terms of positive powers at the four cell corners. Degenerate limits
\(b\rightarrow0\) or \(c\rightarrow0\) require stable divided-difference or
one-dimensional formulae. The implementation must not use the raw closed form
near cancellation without these limits.

For a cell parameterized by \(0\le x,y\le h\), the non-degenerate form is

\[
\frac{
[a+bh+ch]_+^{p+2}
-[a+bh]_+^{p+2}
-[a+ch]_+^{p+2}
+[a]_+^{p+2}}
{bc(p+1)(p+2)}.
\]

For \(p=0\), this is the area of the part of the cell on the positive side of
the affine boundary. For fractional \(p\), it integrates the limb-darkening
basis instead of sampling its singular pointwise derivative at the limb.

Prototype two implementations:

1. a stabilized analytic affine-cell moment primitive;
2. a fixed sub-cell Gauss rule used as an independent check and fallback.

Selection criteria are value convergence, directional-derivative convergence,
CPU throughput, and behavior when the source limb is tangent to a cell.

The analytic primitive has important expected advantages:

- no arbitrary sigmoid width;
- fractional area for the uniform moment;
- correct vanishing of the \(\phi^{1/2}\) and \(\phi^{1/4}\) terms at the limb;
- a smooth derivative as the boundary moves across a cell;
- vectorized scalar arithmetic suitable for CPU SIMD.

It remains an approximation because the lens map is not affine across a cell.
Coarse/fine convergence remains authoritative.

### 4.4 Avoiding double counting

Active macro-tiles are aligned to one global integer lattice and deduplicated
before integration. Every lattice cell therefore has a unique owner.

Do not integrate overlapping image-centred rectangles independently and then
attempt geometric overlap correction. Deduplicating integer macro-tile IDs is
cheaper and more reliable. Stage A should use fixed-capacity sorting/packing or
a bounded open-address table; both variants should be benchmarked because XLA
CPU performance is not obvious in advance.

### 4.5 Fold and critical-curve behavior

Unlike the current per-seed Cartesian flood fill, globally unique tiles do not
need a Jacobian-sign guard to prevent double counting. Both sides of a fold may
be evaluated; each distinct image-plane cell is counted once.

The Jacobian is still useful for:

- predicting local image scale and halo size;
- recognizing near-critical seeds;
- selecting resolution/capacity buckets;
- diagnostics and targeted calibration.

Near a lens pole or an unresolved critical image, non-finite mapping values are
masked as outside only when the cell can be proven irrelevant. Otherwise the
result is a numerical failure, not a partial magnification.

## 5. JAX differentiation design

### 5.1 Polynomial roots

Implement a pure-JAX binary polynomial coefficient builder and a batched
Ehrlich-Aberth solver in float64/complex128. The solver is a support-discovery
primitive and is stopped-gradient in the inverse-ray path.

Requirements:

- static degree five;
- fixed maximum iterations with a convergence mask;
- optional warm starts along source-limb angles and trajectories;
- residual-based physical-image filtering;
- no gradient through roots, iteration count, root ordering, physical masks,
  or initial-root selection in the inverse-ray MVP;
- explicit failure when roots are unresolved or near-multiple beyond the
  supported tolerance.

Branch matching is unnecessary when every physical limb root is only converted
to a macro-tile seed. If later scanline rasterization needs branch continuity,
matching is added as a separate stopped-gradient discovery operation.

An implicit root JVP,

\[
\dot z_i = -
\frac{\sum_k \dot a_k z_i^k}{P'(z_i)},
\]

is reserved for a future point-source or multipole API. It is deliberately not
on the MVP critical path.

### 5.2 Tile integral AD

Start with native JAX differentiation through the lens map and boundary
primitive. Then add a `custom_vjp` around the tile reducer if profiling shows
that reverse-mode stores too many ray intermediates.

The backward pass should recompute tile chunks rather than retain all
image-plane arrays. Use `jax.checkpoint`/`remat` at tile or chunk granularity.
Peak memory must scale with chunk size, not total light-curve length times ray
count.

### 5.3 Forward and reverse interfaces

Support both:

- directional JVP/Jacobian calculations for diagnostics and small parameter
  vectors;
- reverse-mode gradients of a scalar likelihood for optimization and HMC.

The benchmark must measure `value`, `jvp`, and `value_and_grad(loss)`
separately. A fast forward calculation with an unusably expensive reverse pass
does not meet the mission.

### 5.4 Differentiability contract

We promise a numerically converged local derivative of the discretized
finite-source integral when:

- root solving converged;
- discovery did not overflow;
- the outer tile halo has zero flux;
- coarse/fine value and derivative checks pass;
- parameters are not at a declared physical singularity.

We do not promise derivatives of:

- integer capacity selection;
- root permutations;
- retry count;
- diagnostic method labels;
- exact topology at a singular caustic contact.

Those are computational choices. The returned physical magnification and its
local derivative must remain stable under resolution changes.

## 6. CPU performance design

### 6.1 Static shapes, bounded work

JAX/XLA performs best with static shapes. Use a small set of compiled variants
instead of data-dependent array sizes:

- ray resolution buckets;
- macro-tile capacity buckets;
- source-limb sample buckets;
- epoch chunk sizes.

No Python callback or C++ `pure_callback` is allowed on the primary JIT path.
Such callbacks prevent whole-program XLA optimization and complicate AD.

### 6.2 SIMD inside tiles, scan across tiles

The expected CPU layout is:

```text
epoch chunk
  └─ active macro-tile scan
       └─ vectorized T×T ray cells
            └─ fused lens map + three moments
```

`T=8` and `T=16` are initial candidates. Too-small tiles increase loop/scatter
overhead; too-large tiles waste rays outside thin images.

Prefer structure-of-arrays real arithmetic in the hot lens-map kernel if
complex JAX lowering does not vectorize well. Benchmark complex128 against
explicit `(x, y)` float64 arrays before choosing.

The forward grid spacing may be selected from \(\rho/N\), but grid coordinates,
cell area, and tile bounds must be wrapped in `stop_gradient` for local
differentiation. The mathematical image-plane integration measure is fixed;
the derivative with respect to \(\rho\) must enter through \(\phi\) and the
source-flux normalization, not through artificial motion of the numerical
lattice. Resolution changes remain discrete convergence decisions.

### 6.3 Parallelism

Parallelize primarily over epochs and, secondarily, over tiles. Avoid one giant
`vmap` that materializes all rays for all epochs.

Benchmark:

- one CPU thread for algorithmic comparisons;
- default host thread count for user-facing throughput;
- scalar epoch latency;
- 100, 1,000, and 10,000 epoch throughput;
- compilation time reported separately from warm execution.

Thread-count and XLA environment settings must be recorded in every benchmark
artifact.

### 6.4 Reuse

Within one light curve:

- reuse polynomial roots from adjacent source-limb angles;
- optionally warm-start centre roots from adjacent epochs;
- cache static lattice coordinates for every resolution/tile shape;
- form several passband limb-darkening results from the same moment tuple;
- reuse discovery support for nearby evaluations only after proving the halo
  remains empty.

Parameter-dependent caches must never return stale support during gradient
evaluation.

## 7. Accuracy and convergence

### 7.1 Error budget

Use the existing convention:

\[
\mathrm{tol}+\mathrm{reltol}\max(|A|,1).
\]

Initial defaults may match `lcbinint`, but they are not considered calibrated
until the new engine passes its own sweep.

### 7.2 Coarse/fine estimator

Evaluate supported bucket pairs, initially \(N\) and \(2N\), with the same
discovery support expanded for the fine grid. Compare:

- all three raw moments;
- final magnification for representative \((c,d)\);
- directional derivatives with respect to \(w_x,w_y,s,q,\rho,c,d\).

Value convergence alone is insufficient. Return separate
`value_converged` and `gradient_converged` diagnostics during development.
The public API may later combine them according to whether gradients were
requested.

### 7.3 Validation references

Use multiple references:

- current `lcbinint` Cartesian and polar inverse-ray calculations;
- high-resolution self-convergence of the new engine;
- VBMicrolensing in ordinary validated regimes;
- source-plane quadrature as an independent integral where it converges;
- microJAX for matched ICRS test cases.

No single external solver is treated as an oracle in extreme
high-magnification cases.

The initial binary-lens calibration domain should match the existing
finite-source safety sweep where practical:

- \(0.1\le s\le4\);
- \(10^{-6}\le q\le1\);
- \(10^{-4}\le\rho\le0.1\);
- source positions stratified across far field, cusp approaches, fold
  approaches, caustic crossings, grazing/tangent cases, resonant caustics, and
  high-magnification tails;
- uniform, moderate, and strong limb-darkening coefficients, including
  nonzero values of both \(c\) and \(d\).

Calibration and held-out validation must use disjoint random seeds and retain
named regression cases for every discovered failure.

### 7.4 Gradient validation

For smooth test points:

- compare JVPs with symmetric finite differences over a step-size sweep;
- compare reverse-mode gradients with JVP dot-product identities;
- test translation, mass-exchange, and reflection symmetries;
- check convergence under ray-grid translation by half a cell;
- check continuity as the source limb crosses cell boundaries;
- check continuity across fold entry/exit trajectories.

Near caustics, a single finite-difference step is not a sufficient reference.
Require a plateau across several step sizes and agreement between independent
resolution buckets.

## 8. Scope and API

### 8.1 MVP scope

Included:

- binary lens;
- static \(s,q\);
- scalar source position and batched trajectories;
- finite \(\rho>0\);
- uniform, linear, square-root, and current two-coefficient limb darkening;
- Cartesian macro-tile inverse rays;
- float64/complex128;
- magnification, moments, diagnostics, JVP, and scalar-loss gradient;
- CPU benchmarks.

Deferred:

- point-source/hexadecapole production routing;
- polar inverse-ray backend;
- triple lenses;
- parallax and orbital trajectory models;
- binary sources;
- finite exposure integration;
- GPU optimization;
- a stable end-user API.

Higher-order trajectory effects can later remain ordinary differentiable JAX
transformations feeding source positions into the kernel.

### 8.2 Proposed package layout

```text
python/lcbinint_jax/
  __init__.py
  api.py                 # public experimental entry points
  types.py               # NamedTuple/PyTree result and static options
  lens.py                # real and complex binary lens maps
  polynomial.py          # coefficients and stopped-gradient EA roots
  images.py              # residuals and physical masks
  discovery.py           # fixed-capacity macro-tile discovery
  cell_moments.py        # stabilized partial-cell moment primitive
  integrate.py           # tile/chunk reducers
  convergence.py         # bucket selection and coarse/fine checks
  limb_darkening.py      # moment combination and normalization
python/lcbinint/jax_ir/
  __init__.py            # compatibility re-export
tests/jax_ir/
  test_roots.py
  test_cell_moments.py
  test_discovery.py
  test_magnification.py
  test_gradients.py
  test_overflow.py
tests/diagnostics/jax_ir/
  benchmark_cpu.py
  sweep_accuracy.py
  sweep_gradients.py
  inspect_tiles.py
```

JAX is an optional project dependency. The standalone `lcbinint_jax` namespace
does not import the native extension or require GSL. Importing ordinary
`lcbinint` must not require JAX.

### 8.3 Candidate API

```python
from lcbinint_jax import binary_inverse_ray, InverseRayOptions

result = binary_inverse_ray(
    source_x,
    source_y,
    separation=s,
    mass_ratio=q,
    source_radius=rho,
    limb_c=c,
    limb_d=d,
    options=InverseRayOptions(
        resolution=64,
        tile_size=8,
        tile_capacity=128,
        limb_samples=32,
    ),
)

result.magnification
result.moments
result.error_estimate
result.value_converged
result.discovery_overflow
```

Static options are compile-time arguments. Physical parameters are dynamic JAX
arrays. Diagnostic outputs are fixed-shape JAX values, not Python objects
created inside JIT.

## 9. Implementation phases and gates

### Phase 0: benchmark contract

Deliver:

- pinned CPU benchmark cases;
- benchmark harness separating compile and warm execution;
- current `lcbinint` forward and microJAX CPU baselines;
- recorded hardware, thread count, JAX version, precision, and tolerance.

Gate:

- benchmark runs reproducibly from one command and saves machine-readable
  results.

### Phase 1: lens map, moment algebra, and fixed-support hot kernel

Deliver:

- binary lens map;
- analytic combination of the three image moments into \(A(c,d)\);
- midpoint/subcell reference rule;
- stabilized affine boundary-cell prototype;
- fixed-support tile reducer with real and complex lens-map variants;
- forward, JVP, reverse-gradient, and CPU benchmark tests.

Gate:

- uniform and limb-darkened values converge on hand-selected regular, fold,
  cusp, and high-magnification cases;
- gradients converge with resolution;
- the affine primitive is retained only if it beats the subcell fallback in
  both accuracy per ray and CPU time;
- fixed-support CPU throughput justifies proceeding to discovery.

### Phase 2: root solver and support seeds

Deliver:

- binary polynomial coefficients;
- batched stopped-gradient EA root solver;
- cold start, warm start, phase-shift fallback, and Newton polishing;
- physical-root residual filtering;
- centre and source-limb tile seeds.

Gate:

- roots and physical images agree with current `lcbinint` over a stratified
  \((s,q,w)\) sweep;
- injected non-convergence is reported rather than silently filtered;
- all known centre and limb image components produce at least one tile seed.

### Phase 3: pure-JAX discovery

Deliver:

- centre and source-limb root sampling;
- bounded macro-tile flood fill;
- tile deduplication;
- halo, overflow, and coverage diagnostics.

Gate:

- zero missed-image cases in the initial stratified calibration set;
- every injected capacity failure is reported;
- final halo has zero flux within a calibrated threshold;
- discovery consumes a minority of warm forward time in typical cases.

### Phase 4: automatic resolution and trajectory batching

Deliver:

- static S/M/L/XL resolution-capacity buckets;
- coarse/fine retry through `lax.cond` or host-level compiled-function
  dispatch, chosen by benchmark;
- epoch chunking and warm starts;
- scalar-loss `value_and_grad`.

Gate:

- requested default value tolerance is met on the held-out sweep;
- gradient convergence failures are reported;
- memory remains bounded by configured epoch/tile chunk size.

### Phase 5: hardening

Deliver:

- broad random and topology-targeted sweeps;
- crash/NaN/overflow regressions;
- documentation of supported parameter domain;
- optional package dependency and experimental API;
- reproducible benchmark report.

Gate:

- all scientific and performance acceptance criteria below pass on at least
  two CPU architectures.

### Phase 6: optional extensions

Only after Phase 5:

- polar macro-tiles for extremely elongated high-magnification images;
- trajectory support and multiple passbands;
- point/hexadecapole routing;
- triple lens;
- GPU tuning.

## 10. Acceptance criteria

These are project gates, not promised benchmark results.

### 10.1 Scientific

At the calibrated default tolerance:

- no silent tile/queue/root overflow;
- no known missed image component in the held-out topology sweep;
- at least 99.9% of ordinary held-out points satisfy the requested value
  budget, with every miss marked unconverged;
- directional gradients meet the calibrated gradient budget at smooth points;
- fold-entry/exit gradient curves are stable under one resolution increase;
- results are finite or explicitly failed, never plausible partial areas.

The final numeric thresholds for gradient relative/absolute error must be
frozen from calibration before claiming production readiness.

### 10.2 Performance

Measure on pinned x86-64 and ARM64 CPUs after compilation:

- warm forward latency and throughput;
- warm JVP;
- warm scalar-loss value-and-gradient;
- peak resident memory;
- number of lens-map evaluations;
- discovery fraction of runtime.

Go/no-go targets for the first optimized prototype:

- at least 10x faster than microJAX CPU ICRS at matched achieved accuracy on
  the selected limb-darkened benchmark suite;
- warm finite-source forward no worse than 3x the current C++ `lcbinint`
  inverse-ray forward at comparable achieved accuracy;
- scalar-loss value-and-gradient no worse than 5x the new JAX forward time for
  the standard parameter vector;
- no memory growth proportional to total trajectory length when epoch
  chunking is enabled.

If the 10x microJAX target is missed because both engines evaluate a similar
number of rays, the sparse-support premise has failed and the design must be
reconsidered. If ray count is much lower but runtime is not, profile XLA
scatter/sort/loop overhead before changing the numerical method.

## 11. Principal risks and mitigations

| Risk | Consequence | Mitigation / decision point |
| --- | --- | --- |
| Fixed-capacity discovery misses a tiny fold image | Biased flux with plausible output | Limb-root seeds, halo checks, overflow flags, topology-targeted sweep |
| Macro-tile flood fill is slow in XLA CPU | Discovery erases integration savings | Compare sort/pack vs bounded hash; move work to branch-derived tile spans |
| Dense cells inside tiles still dominate | CPU target missed | Smaller tiles, scanline chunks, polar tiles for elongated images |
| Hard discovery changes produce gradient jumps | HMC instability | Stop-gradient support plus empty halo; test grid-shift and parameter perturbations |
| Affine boundary formula is unstable near zero slopes | NaNs or noisy derivatives | Stable divided differences, 1-D limits, subcell fallback |
| Fractional limb powers have singular pointwise derivatives | Gradient noise near limb | Integrate positive-part moments over cell analytically; never differentiate raw clipped centre sample at boundary |
| Root solver fails near multiple roots | Missing/incorrect seeds | Residual checks, warm starts, iteration cap diagnostics, boundary sampling redundancy |
| Reverse-mode stores all rays | Excessive memory | Tile/chunk `custom_vjp` or rematerialization |
| JAX complex lowering is slow on CPU | Poor SIMD | Benchmark explicit real arithmetic and retain faster kernel |
| Accuracy requires huge capacities | Static-shape waste | Bucketed compiled variants and explicit XL fallback |
| Performance only appears in loose tolerance | Limited scientific use | Publish accuracy-versus-runtime curves; set scope honestly |

## 12. Kill and pivot criteria

After Phase 3, reconsider the macro-tile design if any of the following holds:

- image completeness cannot be certified without source-limb sampling dense
  enough to dominate total runtime;
- fixed-capacity flood fill has frequent overflow in the target parameter
  domain;
- achieved ray reduction relative to microJAX dense polar regions is less
  than 4x on the limb-darkened benchmark suite;
- boundary-gradient convergence requires smoothing widths tuned separately for
  each event;
- JAX control-flow overhead makes the hot kernel slower than a dense region
  despite a large ray-count reduction.

Possible pivots, still respecting the inverse-ray mission:

1. replace tile flood fill with source-limb branch rasterization into
   non-overlapping scanline spans;
2. add polar tiles only for extreme elongated images;
3. implement the tile reducer as a JAX FFI CPU custom call with a manually
   supplied VJP, while keeping the public model and differentiation interface
   in JAX.

The FFI option is a last resort because it weakens portability and compiler
fusion. It is preferable to abandoning inverse rays if pure JAX control flow,
rather than lens-map arithmetic, is the demonstrated bottleneck.

## 13. Immediate experiments

Before committing to the full implementation:

1. Implement the three-moment formula and verify it against direct
   limb-brightness accumulation in the current C++ engine.
2. Prototype the affine positive-part cell moments for
   \(p=0,1/2,1/4\); test value and JVP as a line sweeps through a square.
3. Benchmark one pure-JAX \(8\times8\) and \(16\times16\) tile lens-map kernel
   in complex and real arithmetic.
4. Feed tile IDs exported from current `lcbinint` diagnostics into the JAX
   integrator to measure the best-case hot-kernel speed before implementing
   JAX discovery.
5. Compare dense microJAX region ray count against active `lcbinint` image-cell
   count on the same cases.
6. Prototype bounded tile discovery and measure overhead separately.

Experiment 4 is the most important early de-risking step. If a perfect
externally supplied support mask does not yield the required CPU speed, the
rest of the discovery work cannot rescue the design.

## 14. Initial implementation checkpoint

On 2026-07-26, the implementation branch contains the standalone
`lcbinint_jax` package with:

- real and complex binary-lens maps;
- stopped-gradient Ehrlich-Aberth quintic/quartic root solving and
  lens-equation residual filtering;
- centre-plus-source-limb macro-tile discovery with an explicit overflow flag;
- simultaneous \(M_0,M_{1/2},M_{1/4}\) accumulation;
- stabilized affine boundary-cell moments with axis-aligned limits;
- a JIT-compiled fixed-support macro-tile reducer;
- gradients with respect to \(w_x,w_y,s,q,\rho,c,d\);
- 42 focused value/JVP/reverse-gradient/discovery/reference tests;
- a machine-readable CPU benchmark harness.

One preliminary x86-64/JAX 0.6.2 CPU run at 40,000--43,264 rays measured:

| Tile | Lens arithmetic | Forward | Directional JVP | Value + gradient |
| --- | --- | ---: | ---: | ---: |
| 8 | real | 8.96 ms | 25.95 ms | 74.24 ms |
| 8 | complex | 9.65 ms | 21.16 ms | 51.82 ms |
| 16 | real | 7.17 ms | 21.15 ms | 41.04 ms |
| 16 | complex | 7.12 ms | 24.31 ms | 41.89 ms |

These are development measurements, not accepted performance claims. They do
show that a scalar `lax.cond` around each tile skips inactive work: on the same
625-slot, 8-by-8 configuration, warm forward time fell from about 8.85 ms for
625 active tiles to 1.95 ms for 128, 0.50 ms for 32, and 0.08 ms for none.
This supports continuing with packed sparse discovery.

The first automatic centre-plus-limb-root discovery prototype measured:

| Resolution | Tile slots / used | Forward | Directional JVP | Value + gradient |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 128 / 85 | 6.71 ms | 13.79 ms | 28.25 ms |
| 32 | 256 / 196 | 12.59 ms | 30.81 ms | 63.14 ms |
| 64 | 512 / 487 | 32.43 ms | 74.49 ms | 140.20 ms |

All three completed without root or discovery overflow on the benchmark case.
The value-plus-gradient path is about 4.2--5.0 times the forward path when the
capacity bucket is tight, meeting the initial ratio gate. Oversized capacities
substantially slow reverse mode even when inactive tiles are skipped, so
calibrated capacity buckets are a performance requirement rather than only a
memory optimization. A tile-reducer `custom_vjp` remains a possible later
optimization, not an immediate blocker.

The end-to-end result reports `support_valid`, not `converged`.
`support_valid=True` means only that root filtering and bounded tile discovery
did not report failure. A separate coarse/fine API now compares magnification,
all three source-area-normalized moments, and an optional directional JVP,
returning `value_converged`, `moments_converged`, and `gradient_converged`.

The native extension was rebuilt after installing the GSL development package
in the active Conda environment. Its CMake install destination was corrected
from the wheel root to `lcbinint/`, making editable imports use
`lcbinint._lcbinint` as intended. Direct 128-bin JAX/native regression cases
now cover uniform brightness, quadratic limb darkening, and a source near a
resonant-caustic cusp. Their absolute magnification differences in the initial
checkpoint are \(3.44\times10^{-4}\), \(4.67\times10^{-4}\), and
\(1.95\times10^{-4}\), respectively. A regular-case JAX directional JVP is
also checked against a 512-bin native central difference.

A first matched-accuracy scalar CPU benchmark then exposed the main performance
gap. At an error budget of
\(10^{-4}+10^{-4}\max(|A_{\rm ref}|,1)\), warm JAX/native forward times were
30.48/2.30 ms for a regular uniform source, 87.12/2.45 ms for the same source
with two-coefficient square-root limb darkening, and 51.63/2.72 ms near a
resonant cusp. JAX value-plus-gradient took 132.49, 441.31, and 247.97 ms,
versus 33.56, 34.53, and 39.55 ms for fourteen native evaluations forming a
seven-parameter central difference.

Separating the JAX forward path showed discovery at 3.64--11.13 ms and
fixed-support integration at 21.28--70.39 ms. The immediate optimization
priority is therefore accuracy per integrated ray and CPU execution of the
tile reducer. Root caching cannot by itself close the current scalar latency
gap. In particular, the limb-darkened case needed resolution 128 in JAX but
only 32 native source bins under the oracle-selected rule. Boundary treatment
and brightness-moment quadrature must be improved enough to lower that
resolution before lower-level `custom_vjp` work is likely to pay off.

## 15. References

- Miyazaki & Kawahara, “microJAX: A Differentiable Framework for Microlensing
  Modeling with GPU-Accelerated Image-Centered Ray Shooting,” 2025,
  <https://arxiv.org/abs/2510.02639>.
- microJAX source and documentation,
  <https://github.com/ShotaMiyazaki94/microjax>.
- Bennett, “Detection of Extrasolar Planets by Gravitational Microlensing,”
  image-centred ray-shooting discussion, 2010,
  <https://ui.adsabs.harvard.edu/abs/2010ApJ...716.1408B/abstract>.
- Ren & Zhu, “A Differentiable Binary Microlensing Model Using Adaptive
  Contour Integration Method,” 2025,
  <https://arxiv.org/abs/2501.07268>.
- Bozza, “Microlensing with an advanced contour integration algorithm,” 2010,
  <https://doi.org/10.1111/j.1365-2966.2010.17265.x>.
- `lcbinint` numerical-method documentation,
  [`../numerical-methods.md`](../numerical-methods.md).
