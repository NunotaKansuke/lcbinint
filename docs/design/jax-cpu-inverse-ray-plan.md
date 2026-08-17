# JAX CPU differentiable inverse-ray engine: implementation plan

Status: implementation complete; validation and calibration remain ongoing

Planning branch: `feature/jax-cpu-inverse-ray-plan`

Implementation branch: `feature/jax-cpu-inverse-ray-mvp`

Initial scope: static binary lens, finite circular source, CPU, JAX, differentiable

## 0. C++ execution-backend phase

The JAX implementation remains the algorithmic reference and public
composition layer.  The project will not call the pre-existing native
finite-source engine from a JAX differentiation rule.  Instead, selected
fixed-shape hot paths will gain an independently implemented C++ execution
backend with the same cell moments and stopped-support semantics as the JAX
kernel.

The work is split into gates:

1. **Raw fixed-support forward kernel.**  Keep discovery in JAX.  Pass
   `tile_origins`, `tile_mask`, and scalar physical parameters to a C++17
   kernel through the existing extension module.  This deliberately starts as
   a NumPy-callable benchmark/reference interface, outside `jit`, so raw
   kernel merit is measured before adding XLA integration.
2. **Semantic lockstep.**  Require agreement with
   `binary_inverse_ray_fixed_support` for all three moment modes, 3-by-3 and
   4-by-4 detailed-cell rules, masks, and diagnostics.  The C++ implementation
   follows the JAX equations; it does not reuse the native `lcbinint`
   integrator.
3. **FFI viability gate.**  Measure raw C++ time, Python dispatch time, and
   host/device transfer time separately.  Proceed to a JAX CPU FFI/custom-call
   only if fixed-support forward is at least 2 times faster on representative
   inverse-ray epochs.  The target is 2--4 times.
4. **Analytic tangent kernel.**  Add a C++ JVP for source position,
   separation, mass ratio, source radius, and limb coefficients.  Finite
   differences are not an accepted differentiation backend.  Validate each
   tangent against the JAX reference before exposing it through a custom JVP.
5. **Reverse mode.**  Add a VJP by contracting the same local analytic
   derivatives while scanning cells.  It must not materialize a
   rays-by-parameters Jacobian.
6. **Automatic backend choice.**  JAX retains discovery, multipole routing,
   trajectory transforms, and unsupported-platform fallback.  CPU calls use
   C++ only for calibrated shapes/modes; pure JAX remains the correctness
   fallback and the implementation on other platforms.

The forward gate reports both numerical and performance evidence.  Required
numerical evidence is moment agreement within `2e-11` relative/absolute on
ordinary cases and no change to boundary/active cell counts.  Required
performance evidence is compile-excluded median time over repeated calls,
including a representative production-discovered support rather than only a
small synthetic grid.

Initial non-goals are support discovery in C++, OpenMP across JAX trajectory
epochs, and wrapping the old native inverse-ray implementation.  Keeping
these out of the first gate makes any measured speedup attributable to the
new fixed-support kernel and leaves the differentiation boundary explicit.

### Whole-engine FFI phase

The fixed-support gate showed that leaving every other numerical kernel in
JAX is no longer the fastest CPU architecture.  The next phase moves the
expensive scalar-epoch kernels behind independent typed FFI boundaries while
retaining the dispatcher, trajectory transformations, static method choice,
and public API in JAX.  This is deliberately several calibrated handlers, not
one opaque native light-curve call:

1. **Macro-tile discovery FFI.**  Keep the current stopped-gradient image
   seeds, but replace the fixed-capacity JAX BFS and its linear duplicate
   searches with a bounded C++ queue and hash index.  Require exact support
   order, masks, capacity flags, and counts.
2. **Shared binary-image-root FFI.**  Implement the JAX binary quintic,
   bounded root solve, physical-image filtering, and polishing semantics in
   the new backend.  Expose stopped roots for discovery and implicit
   root-equation derivatives for point-source consumers.  This handler is
   shared by Cartesian, polar, hexadecapole, and source-plane quadrature.
3. **Fused Cartesian epoch.**  Fuse roots, discovery, and fixed-support
   integration after their independent handlers are calibrated.  Its custom
   JVP differentiates only the continuous cell integral; root/support choices
   remain stopped as in the JAX reference.
4. **Hexadecapole FFI.**  Solve the 13 point-source samples in one call,
   compute root tangents by implicit differentiation, and return the scalar
   expansion diagnostics plus a seven-parameter Jacobian.
5. **Polar FFI.**  Fuse stopped radial-band discovery with the polar cell
   traversal and analytic moments.  Preserve all band, radial, and boundary
   overflow diagnostics.
6. **Source-plane FFI.**  Reuse the common differentiable point-source kernel
   across fixed quadrature nodes.  This comes after hex because both use the
   same root/Jacobian primitive and hex is selected much more frequently.

Each step retains an explicit pure-JAX fallback and must independently pass
close/resonant/wide value and derivative sweeps.  Automatic dispatch promotes
a handler only after its gate passes.  Method selection itself stays in JAX,
so a failed or unsupported FFI branch cannot silently change the numerical
algorithm.

The first discovery handler is now implemented.  On the representative
1028-visited-tile support, JAX image seeds plus JAX BFS took 10.90 ms, while
the same JAX seeds plus C++ FFI BFS took 1.55 ms.  All tile indices, origins,
masks, overflow flags, and counts agreed exactly, both on that case and on all
40 source positions in the eight-configuration held-out sweep.  With the
already-calibrated cell FFI, a representative full Cartesian epoch improved
from 47.5 to 13.1 ms for forward, 103.6 to 36.6 ms for JVP, and 435 to
38.0 ms for value plus gradient.  The 64-epoch half-hex/half-Cartesian
trajectory improved from 573 to 239 ms.  Root solving remains JAX in these
measurements.

The shared binary-image-root handler is now implemented as well.  It mirrors
the JAX binary quintic, uses the core C++ polynomial solver, applies six
damped Newton polishing steps, and filters physical roots by lens-equation
residual.  Algebraic candidates that polish onto the same physical image are
deduplicated by scale-aware distance while retaining the lower-residual
candidate.  A second handler returns the analytic implicit root Jacobian with
respect to source x/y, separation, and mass ratio.  A JAX `custom_jvp` exposes
that Jacobian to forward and reverse AD without differentiating the iterative
root solver.

Median warm CPU timings were 45.2 versus 8.1 microseconds for one JAX/FFI
solve, 48.3 versus 7.3 microseconds for its JVP, and 1.40 ms versus
42.6 microseconds for 25 solves.  Switching the 13-point hexadecapole
calculation to the common handler improved forward/JVP/value-plus-gradient
from 1.084/1.187/1.381 ms to 0.055/0.069/0.152 ms.  The handler is shared by
hexadecapole, Cartesian discovery, polar discovery, and source-plane
quadrature; each public dispatcher retains explicit `"jax"`, `"ffi"`, and
portable `"auto"` root-backend choices.

The same eight-configuration, 120-row held-out sweep found no root-backend
failure in point-source values/four-parameter gradients or in
hexadecapole values/seven-parameter gradients for uniform, linear, and
two-coefficient profiles.  Difficult extreme-planetary caustic samples are
included.  With both root and Cartesian handlers enabled, the 64-epoch
half-hex/half-Cartesian trajectory improved from 573 to 158 ms.  It measured
131 ms in native `lcbinint`, 3.04 s in VBMicrolensing, and 14.37 s in
microLUX, with zero accuracy-budget failures in every row.

The next handler fuses centre/limb root solves, bounded macro-tile discovery,
and fixed-support integration into one typed CPU FFI call.  Its companion
handler repeats the stopped support construction and evaluates the continuous
cell integral with the same seven-parameter Jet Jacobian.  Overflow and root
failure remain explicit outputs, so the JAX dispatcher can retain its polar
fallback without materializing tile arrays.  Explicit `root_backend="jax"`
continues to select the staged path; `"auto"` and `"ffi"` use the fused
handler when present.

Across the 84 support-valid rows of the existing close/resonant/wide held-out
sweep, fused values, JVPs, and gradients all passed the same budgets as the
staged FFI.  Values differed at about \(10^{-15}\) or less.  A few
square-root-profile derivatives differed at \(10^{-10}\) in JVP and
\(10^{-9}\) in individual gradient components because internal root seeding
can permute the same support queue and therefore its floating-point reduction
order; these differences are well below the established JAX/FFI derivative
budgets.

Fusion is not the large remaining speed lever.  A representative Cartesian
epoch improved only from 8.21 to 8.05 ms forward and 24.26 to 23.96 ms for
value plus gradient.  The 64-epoch mixed trajectory changed from 157.68 to
156.83 ms forward; run-to-run noise dominated its JVP and reverse timings.
The native trajectory measured 129.77 ms in the same run.  Thus custom-call
and intermediate-support overhead account for only roughly one percent here;
the remaining native gap is primarily inside the cell traversal and moment
evaluation.

The first cell-hot-loop pass then removed that gap without requiring a
machine-specific build.  Fixed fractional powers in the affine boundary
formula are expanded into products and square roots, eliminating repeated
general `pow` calls.  Moment mode and boundary subdivision are compile-time
specializations.  Subcell offsets are precomputed, and each tile row first
computes source classification into structure-of-arrays buffers under an
OpenMP SIMD directive before the branchy integration pass.  The Jacobian path
uses the scalar classification first and constructs five-lane Jets only for
cells that contribute; fully interior uniform cells add constant area without
a lens-map Jet.

On a fixed CPU core, representative uniform/linear/two-coefficient epoch
forward times changed from 4.72/6.09/30.29 ms to 3.13/3.83/14.06 ms.
Value-plus-gradient changed from 14.70/19.22/74.86 ms to
5.05/14.72/37.14 ms.  A separate `-march=native` experiment showed additional
Jet headroom, but applying ISA target clones only to the hot template did not
reproduce it because the Jet arithmetic helpers remained in the default
translation-unit target.  The production implementation therefore remains a
portable build rather than imposing AVX2 or AVX-512.

The next trajectory-level pass added a masked Cartesian batch FFI. JAX still
computes the hexadecapole acceptance and polar-preselection masks; one C++
call evaluates only the selected Cartesian rows, while polar and invalid
support rows retain the complete scalar dispatcher. This preserves method and
fallback choices while removing one FFI boundary per active epoch. The
handler parallelizes independent epochs with OpenMP, honors
`OMP_NUM_THREADS`, and caps its team at 32. Before enabling this, mutable
Skowron--Gould scratch storage was changed from process-static to call-local;
the batch and scalar results then matched exactly under concurrent root
solves.

On the 32-thread benchmark host, the 64-epoch mixed linear trajectory now
measures 9.14/24.04/32.64 ms forward/JVP/value-plus-gradient. The scalar FFI
baseline was 97.94/93.39/103.12 ms, native forward is 134.84 ms,
VBMicrolensing forward is 3.02 s, and microLUX measures
14.19/12.66/26.30 s. All 64 rows retain zero common-budget failures.
At one OpenMP thread the batch and scalar forward times are 97.88 and
98.01 ms, respectively, making the distinction between boundary removal and
multicore throughput explicit.

The 32-thread uniform trajectory measures 12.30/11.58/21.84 ms for batched
FFI, 130.91 ms native, 46.08 ms VBMicrolensing, and
14.07/12.80/25.15 s microLUX. FFI again has zero accuracy-budget failures;
native has two under the common budget. The batched inverse ray is now
3.75 times faster than the appropriate VBMicrolensing uniform comparator on
this trajectory. The benchmark harness exposes `--profile uniform` and
`--profile linear` and records `OMP_NUM_THREADS`.

Repeating the eight-configuration held-out sweep after the arithmetic changes
again produced zero value, JVP, gradient, discovery, point-source,
hexadecapole, or fused/staged failures over all 84 support-valid rows.

### Whole-trajectory CPU FFI result

The remaining three whole-engine kernels are now implemented.

1. The polar handler performs centre/limb root discovery, radial-band merging,
   angular support selection, and streamed radial-cell integration in one
   typed FFI call. It returns the same moments and explicit root, band, and
   capacity diagnostics as the JAX implementation. Its analytic seven-input
   Jacobian is exposed through `custom_jvp`.
2. The batched hexadecapole handler evaluates all 13 point-source samples,
   topology checks, and the quadrupole/hexadecapole combination in C++. It
   parallelizes independent epochs and returns the full seven-input Jacobian.
3. The trajectory handler fuses hexadecapole acceptance, polar preselection,
   and Cartesian/polar execution into one OpenMP-parallel FFI call. Normal
   rows therefore cross a single JAX/C++ boundary. Only rows reporting an
   explicit unsupported or capacity condition re-enter the existing scalar
   dispatcher for its source-plane or expanded-support fallback.

For the representative high-magnification linear polar epoch at resolution
64 and angular resolution 4096, the streamed polar handler changed
forward/value-plus-gradient time from 24.03/81.61 ms to 13.55/17.76 ms.
Values and diagnostic counts agree with the JAX kernel at floating-point
roundoff; the largest component of the seven-input gradient differed by
\(3.2\times10^{-9}\).

For 64 hexadecapole epochs, the batched handler changed forward time from
2.085 to 0.094 ms and value-plus-gradient time from 3.246 to 0.295 ms.
The largest observed gradient difference from the staged root-FFI
implementation was \(2.1\times10^{-10}\).

With `OMP_NUM_THREADS=32`, the integrated 64-epoch linear trajectory measures
9.48/21.70/27.84 ms for forward/JVP/value-plus-gradient. Native `lcbinint`
forward measures 135.20 ms, VBMicrolensing 2.96 s, and microLUX
14.05/13.07/27.00 s in the same run. The uniform trajectory measures
5.86/8.88/16.60 ms, versus 131.80 ms native, 45.76 ms VBMicrolensing, and
13.85/12.89/26.17 s microLUX. Compile plus first execution is 0.39 s for
linear and 0.30 s for uniform. All integrated-FFI rows pass the shared
accuracy budgets.

The repeated eight-configuration held-out sweep contains 120 rows, of which
84 have valid inverse-ray support. It reports zero value, JVP, gradient,
discovery, point-source, hexadecapole, or fused-trajectory failures. For the
integrated trajectory specifically, the maximum value/JVP/gradient error
ratios are 0.000011/0.0036/0.015 of their respective budgets. Thus the
single-call path preserves the dispatcher semantics and derivative accuracy;
the remaining scalar path is a deliberate exceptional-row fallback rather
than a performance retry.

### First forward-gate result

The initial C++17/pybind prototype implements the real binary-lens map,
interior second-order moments, curved detailed-cell subdivision, all three
moment modes, and the JAX diagnostic counts.  It is available as
`binary_inverse_ray_fixed_support_cpp`; the name and docstring deliberately
state that this is an experimental host interface.

On the representative `s=1`, `q=0.1`, `rho=0.03`, `x=0.5` discovery at
resolution 96, the stopped support contained 1028 active 16-by-16 tiles,
4766 detailed cells, and 161056 contributing cells.  Median warm forward
times over 30 calls were 13.80 ms for the JAX fixed-support kernel and
6.52 ms for C++, a 2.11-times raw-kernel speedup including the Python call.
The magnification absolute difference was `6.5e-12`, the largest moment
difference was `2.7e-14`, and both diagnostic counts agreed exactly.

This passes the stated 2-times FFI viability gate.  It does **not** yet make
the production API faster: the prototype converts to host NumPy inputs and
cannot participate in `jit`, `jvp`, or `grad`.  The next implementation step
is therefore a CPU FFI/custom-call wrapper around this kernel, followed by
analytic JVP/VJP kernels.  The reproducible raw comparison is
`tests/diagnostics/jax_ir/benchmark_cpp_backend.py`.

### Typed CPU FFI result

The forward kernel is now registered as an XLA typed FFI target and exposed by
`binary_inverse_ray_fixed_support_ffi`.  It accepts JAX arrays, executes inside
`jax.jit`, returns the standard `FixedSupportResult`, and uses JAX's explicit
sequential batching rule.  The handler has strict rank/dtype/shape checks and
is registered only for CPU.  Builds without JAX headers retain all existing
native functionality and the raw pybind prototype, while the FFI entry point
raises a clear rebuild error.

On the same 1028-tile representative support, median warm times over 30 calls
were:

| path | fixed-support forward |
| --- | ---: |
| pure JAX | 13.64 ms |
| raw C++ through pybind | 6.75 ms |
| C++ through JAX FFI and `jit` | 6.76 ms |

The FFI path is 2.02 times faster than pure JAX and adds no measurable overhead
relative to the host prototype.  Its value, moments, and diagnostic counts
match the raw C++ path; the magnification difference from pure JAX remains
`6.5e-12`.

### Analytic differentiation result

The C++ cell algebra now uses a small forward-mode value type to accumulate
the Jacobian with respect to source x/y, separation, mass ratio, and source
radius during the same fixed-support traversal.  Limb-coefficient derivatives
are combined analytically from the completed moments, avoiding two inactive
derivative lanes in the hot cell loop.  Numerical support, cell classification,
tile origins, and cell size remain stopped-gradient.

A second typed FFI handler returns value plus the seven-parameter Jacobian.
`binary_inverse_ray_fixed_support_ffi` wraps it in `custom_jvp`: arbitrary
directional derivatives are Jacobian-vector products in JAX, and JAX
transposes that explicit linear map for reverse mode.  No finite differences
or derivative of the discrete support search are involved.

Uniform, linear, and two-coefficient modes pass value, moment-JVP, scalar JVP,
and full magnification-gradient comparisons against the pure-JAX reference.
On the representative 1028-tile linear case, the first calibrated run measured
36.92 ms versus 21.41 ms for directional JVP and 239.3 ms versus 21.42 ms for
value plus seven-parameter gradient.  Thus the analytic backend was 1.72 times
faster for JVP and 11.18 times faster for reverse mode while retaining the
roughly 2-times forward speedup.  The maximum seven-component gradient
difference was `4.3e-10`.

The backend replay now covers eight close/resonant/wide lens configurations,
three brightness profiles, field points, and source centres at 0.5, 1, 2, and
5 source radii from sampled caustics.  Of 120 attempted rows, 84 had valid
fixed support and all 84 passed value, directional-JVP, and full
seven-parameter-gradient budgets.  The other 36 were rejected by the existing
JAX discovery overflow/root checks before either cell backend was compared;
there were no C++-specific failures.  The reproducible harness is
`tests/diagnostics/jax_ir/sweep_cpp_backend.py`.

The typed FFI is therefore selected automatically for the real Cartesian
kernel on CPU when the extension exposes both FFI handlers.  Explicit
`cartesian_backend="jax"` and `"ffi"` choices remain available; `"auto"`
falls back to pure JAX for non-CPU platforms, complex kernels, and builds
without FFI support.  A one-direction tangent handler remains a possible
optimization if future trajectory profiles show that returning the full
Jacobian dominates forward JVP.

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
- 43 focused value/JVP/reverse-gradient/discovery/reference tests;
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

The first optimization pass changed that conclusion materially for linear limb
darkening. Tile-level lazy branching avoids evaluating affine boundary moments
on wholly interior or exterior tiles. Boundary-adjacent interior cells now
retain affine integration, improving \(M_{1/2}\) enough for resolution 64 to
meet the development error budget. A linear-only moment path removes
\(M_{1/4}\), constant powers expose more XLA simplification, and checkpointing
the active-tile body nearly halves reverse latency.

Against microLUX commit
`a241b8c2f2198bc4846c0fa66e2bcdcf5cfa6428`, the optimized regular
linear-limb case measured 8.48/16.77/31.06 ms for forward/JVP/value-plus-grad,
versus 19.66/24.55/48.27 ms for microLUX. Near a resonant cusp, matched
accuracy required 80 microLUX annuli; timings were 13.71/27.15/51.30 ms for
inverse rays and 343.68/418.33/595.52 ms for microLUX. Thus the current
development speedups are 1.46--2.32x on the regular case and 11.61--25.06x at
the cusp. A broad held-out sweep is still required before promoting those
case-level wins to a general performance claim.

A follow-up four-engine benchmark added an \(M_0\)-only uniform-source JAX
graph and fixed the method's current operating envelope. At matched achieved
accuracy, warm forward times for JAX/microLUX/native/VBMicrolensing were
5.30/2.83/2.43/0.145 ms at the regular uniform point and
8.17/5.60/2.94/0.160 ms at the uniform resonant cusp. Inverse rays therefore
do not have a scalar-CPU advantage without limb darkening.

With linear limb darkening, the same four-way comparison measured
8.98/18.10/2.98/2.13 ms at the regular point. At the resonant cusp it measured
13.64/347.87/3.24 ms for JAX/microLUX/native, while VBMicrolensing missed the
common error budget. A second cusp at \(q=10^{-3}\) measured
8.45/288.88/2.22 ms, again with the VBMicrolensing result outside the common
budget. Thus the microLUX advantage is reversed precisely where its concentric
contour evaluations multiply, and the result is not limited to \(q=0.1\).

The counterexample is equally important: at a planetary far-field point, the
linear-profile timings were 14.33/6.77/1.26/0.055 ms. microLUX and the native
solvers can take a point/quadrupole shortcut while the current JAX API always
rasterizes the finite source. The production design should now prioritize a
hybrid differentiable dispatcher:

1. point/multipole or one-contour evaluation for uniform sources and epochs
   certified far from caustics;
2. inverse-ray moments for limb-darkened epochs that fail the shortcut test;
3. coarse/fine and support diagnostics before accepting either numerical
   finite-source path.

This routing is not a retreat from the inverse-ray objective. It confines the
inverse-ray kernel to the regime where its single-pass limb moments provide a
measured CPU advantage. The immediate benchmark sweep should calibrate the
dispatcher boundary rather than attempt to make inverse rays beat contour
integration in the uniform far field.

The next implementation checkpoint added a differentiable polar inverse-ray
kernel and a Cartesian/polar dispatcher. Rather than port the native variable
length queue literally, the JAX path merges stopped-gradient centre/limb image
radii into fixed-capacity bands. It then constructs angle-local radial
intervals and evaluates them in fixed angular chunks. The lens-map derivatives
are transformed into radial and angular derivatives, allowing the existing
affine positive-part moment formula to integrate \(M_0,M_{1/2},M_{1/4}\).

This solves a real support problem: for a tested
\(s=0.95,q=0.01,\rho=0.005\) high-magnification epoch, Cartesian discovery
overflowed at 4096 macro tiles while the polar representation required three
radial bands. The automatic API preselects polar for tiny, high-magnification
sources and also uses it after Cartesian discovery overflow.

The initial polar implementation was not a speed win because every candidate
cell evaluated all affine fractional moments. Profiling showed roughly
254,000 contributing cells and 6,100 boundary cells, but about 16 million
fixed-shape candidates. Boundary packing, second-order interior moments, large
angular chunks, and two-by-two boundary subdivision changed the result. The
first common-budget passing setting is now 64 radial by 4096 angular cells:
36.89 ms forward, 43.59 ms JVP, and 117.17 ms reverse gradient. microLUX took
88.06, 135.43, and 334.77 ms on the same physical case.

The Cartesian kernel received the same boundary-only affine path and
second-order interior correction. A topology-changing source sweep established
16 rather than 8 source-limb samples as the lowest tested safe default. With
conservative benchmark capacities, regular linear limb darkening now measures
8.21/9.59/29.65 ms for forward/JVP/gradient versus
20.02/25.69/43.74 ms for microLUX. The resonant cusp measures
10.15/14.57/39.41 ms versus 342.40/434.54/620.05 ms. Tight validated capacity
buckets reduce the JAX rows further, but must not be selected without explicit
support and convergence checks.

The triple-lens CPU path is now connected end to end.  Its production
dispatcher uses method codes 0/1/2/3/4 for point source, hexadecapole,
Cartesian inverse ray, polar inverse ray, and source-plane quadrature.  A
native cached-caustic FFI supplies the stopped-gradient routing distance.
The frozen native guards are retained for the ordinary population:
point-source acceptance requires 20 source radii, hexadecapole requires five,
and caustic-clear high magnification uses polar beyond three radii.
Low-magnification outside-limb grazing rows use independent 64-order
ring/chord agreement with the calibrated factor-40 safety margin, followed by
bounded 160/256 and 400/512 chord escalation.

The triple polar implementation no longer scans a rectangular band graph.  A
C++ radial-run flood seeded by the source centre and 64 source-limb root sets
integrates the three limb moments directly.  Angular resolution is selected
from source radius, radial resolution, and maximum image radius.  On the
frozen \(A\simeq153,\rho=10^{-4}\) case, the 64-bin automatic grid gives
152.71070 versus native 152.71349, takes about 6--7 ms warm for the explicit
forward call, and reports no support failure.

The binary CPU FFI now uses the same connected-support principle instead of
retaining the original pure-JAX band rasterisation.  It reuses the Cartesian
centre/limb/component-certificate roots as seeds, discovers centre-inside
radial runs, and applies the existing affine moment rule only to those runs and
their one-cell boundary halo.  The pure-JAX fallback keeps fixed-capacity bands
because a variable-length queue remains a poor XLA representation.  Thus the
two backends share support semantics without forcing the CPU implementation to
pay for fixed-shape candidate regions.

Differentiating polar cell membership itself was not accurate: radial edge
terms omit the azimuthal shape derivative, while adding discrete cap terms
over-corrects the primal.  The accepted JVP instead uses the physical identity
valid on the caustic-clear polar route: integrate analytic triple
point-source derivatives over a fixed 64-by-256 equal-area source disk.  The
image-plane flood remains the fast primal.  On the frozen case all ten
derivatives \((x,y,s,q,q_2,s_2,\theta,\rho,c,d)\) agree with independent
high-order source-plane values within 0.3%; warm explicit value-plus-gradient
time is about 13--19 ms.

A 144-row contact-band audit over six held-out triple geometries exposed two
additional production constraints.  First, extreme mass hierarchies can
return only the converged, physically relevant root subset because
demagnified roots collapse at lens poles.  Discovery now follows the native
point path and accepts that subset rather than globally invalidating the
epoch.  Second, near-caustic Cartesian macro-tile support can exceed its
bounded capacity.  Deep crossings with \(A\ge100,d<0.8\rho\), very high
magnification \(A\ge10^4\), and the numerical \(A\ge10^8\) caustic limit use
calibrated seed-complete polar grids.  Other Cartesian failures are offered a
48/64 polar recovery only when the two values satisfy the requested error
budget; otherwise non-convergence remains explicit.  In the audit, all
remaining invalid rows were either deliberately reported source-plane
512-order non-convergence or two rejected recovery candidates—no overflowing
Cartesian value was silently accepted.

The subsequent public-trajectory gradient audit found that the fixed
source-plane JVP must not be reused for those deep-crossing polar rows.  Its
point-source derivative omits the distributional caustic/topology boundary
term even when the polar primal is accurate.  Crossing and recovery rows now
differentiate the image-plane level set instead: the Jet kernel integrates
affine boundary cells and a one-cell radial/azimuthal halo while retaining the
fast calibrated polar primal.  A derivative-only 256/128 grid is selected
when its estimated active-cell count fits the 30-million-cell budget.  The
five capacity-limited rows in the 37-geometry held-out near-polar set use the
valid primal-resolution Jacobian, with a 3:4 Richardson correction only for
the consistently first-order source-radius component.  The result reports
`gradient_resolution` and `gradient_extrapolated` so this rare regime remains
observable.  Native finite differences on all five capacity-limited rows
agree with the source-position derivative within 2.2%; the frozen extreme
multi-parameter row agrees within 3.7% over the six independently checked
parameters.

The binary topology audit additionally exercises a smooth fold on both sides
of image-pair birth/death and a resonant cusp crossing.  A source centre
exactly on a caustic can contain four *deduplicated* physical roots: the
critical pair is a repeated root.  Discovery now accepts three through five
physical roots and ignores convergence of unused nonphysical polynomial
slots, while still rejecting counts outside the binary-lens range.  This
removes the former false `root_failure`/NaN at a finite-source caustic
crossing.  At resolution 128, uniform and linear-limb fold normal derivatives
at offsets \(-0.98\rho,0,0.98\rho,1.02\rho\) agree with native Cartesian
finite differences within 0.3%; the cusp-centre derivative agrees within
0.5%.  A held-out close/resonant/wide/planetary sweep has 0.26% median
relative disagreement and below 1% disagreement after increasing the noisy
planetary native reference resolution.

There is one intentional mathematical boundary: when the *source limb* is
exactly tangent to a caustic, the finite-source magnification generally has
different one-sided derivatives.  No unique gradient exists at that exact
parameter value, so neither an AD rule nor a symmetric finite difference can
make it smooth without changing the physical model.  Centre crossings and
all offsets on either side remain differentiable; HMC encounters the exact
contact set with probability zero, although trajectories should not use an
exact-contact gradient as a validation reference.

### 14.13 End-to-end HMC audit

The public `LightCurve(options=Options(jax=True))` path is now exercised as a
reverse-mode scalar log density, through reversible leapfrog integration, and
inside NumPyro NUTS. The regression test uses the public parameter dictionary,
not a low-level integration shortcut. The heavier diagnostic is
`tests/diagnostics/jax_ir/benchmark_hmc.py`; it standardizes `t0` and `u0`
with local Fisher scales, records compilation separately from steady state,
checks Hamiltonian error and reversibility, audits dispatcher changes, and
reports divergences, acceptance, leapfrog counts, ESS, and split R-hat.

On the recorded CPU run with 24 epochs and linear limb darkening, the smooth
case (150 warmup plus 150 retained samples in each of two chains, dense mass)
had zero divergences, split R-hat 1.004/1.006, and ESS 216/184. The audited
fold-crossing case (100 plus 100 in each chain) also had zero divergences,
split R-hat 1.009/1.026, and ESS 160/180. In both cases the injected `t0` and
`u0` were inside one posterior standard deviation. Five-step leapfrog
reversibility was at machine precision; maximum absolute energy error was
0.00061 for the smooth curve and 0.0055 for the fold crossing. The exact
source-limb tangent remains excluded for the mathematical reason above.

Steady 24-epoch public likelihood/value-plus-gradient times were
9.8/40.2 ms for the smooth case and 17.4/61.2 ms for the fold crossing.
Native lcbinint forward-only times were 177 ms and 1.15 s respectively; VBM
forward-only times were 10.9 ms and 44.4 ms. Thus VBM remains slightly faster
for a forward-only caustic curve, while it does not supply the reverse-mode
gradient needed by NUTS.

The matched trajectory benchmark with 12 epochs measured the fused FFI at
6.06 ms forward and 25.4 ms value-plus-gradient, versus microlux at 10.86 s
and 24.80 s on this CPU (about 1790x and 977x respectively). Both met the same
accuracy budget. This extreme gap is mostly microlux's 80-annulus
limb-darkening integration rather than its uniform-source contour kernel. A
separate true uniform-source run (`limb_darkening=None`) measured 3.81 ms
versus 12.22 ms forward and 12.16 ms versus 34.99 ms value-plus-gradient:
the fused FFI remains 3.2x/2.9x faster, while the comparison is no longer
dominated by annulus quadrature.

A deliberately tiny four-epoch, two-warmup/two-sample NUTS probe took 3.56 s
with lcbinint and 68.3 s with microlux; neither run is long enough for
convergence and both diverged, so it is retained only as a compile/runtime
comparison, not a sampling-quality claim. Full JSON records are stored in
`docs/assets/jax_ir_hmc_benchmark.json`,
`docs/assets/jax_ir_hmc_smooth_dense.json`,
`docs/assets/lcbinint_hmc_constrained.json`, and
`docs/assets/microlux_hmc_constrained.json`; the matched per-evaluation record
is `docs/assets/microlux_trajectory_benchmark.json` and its uniform-source
counterpart is `docs/assets/microlux_uniform_trajectory_benchmark.json`.

### 14.14 Higher-dimensional and Triple HMC audit

`tests/diagnostics/jax_ir/benchmark_hmc_multidim.py` extends the inference
audit to the transformed Binary parameters
`(t0,u0,log_tE,alpha,log_s,log_q,log_rho)` and the corresponding ten Triple
parameters with `log_q2`, `log_sep2`, and `ang`. It builds a regularized local
Fisher whitening transform, but retains the public `LightCurve` call inside
NumPyro. This is a diagnostic parameterization, not a prescribed scientific
prior.

The 24-epoch seven-dimensional Binary run used 100 warmup and 100 retained
draws in each of two chains. It had zero divergences, maximum split R-hat
1.028, ESS 86--241, mean acceptance 0.945, and a maximum truth displacement
of 1.96 posterior standard deviations. Median/maximum leapfrog counts were
31/63. A value-plus-gradient took 41.4 ms versus 177 ms for a native
forward-only curve; the full run took 565 s. The sampler is therefore valid
but posterior geometry, rather than the FFI evaluation alone, controls total
runtime.

For Triple lenses, releasing all ten parameters immediately is deliberately
not presented as a converged scientific fit. A short 12-epoch run with only
30 warmup and 30 retained draws per chain and a 15-step tree ceiling had 15
divergences and poor R-hat. Separating the trajectory from the lens geometry
identifies this as a strongly coupled inference problem rather than an
inability to differentiate the Triple model:

- the smooth four-parameter Triple trajectory run had zero divergences,
  maximum R-hat 1.023, ESS 62--140, and maximum truth displacement 0.88
  posterior standard deviations;
- its gradient agreed with a stable discrete-primal difference within 0.77%,
  four-step energy error was 0.00030, and reversibility was at machine
  precision;
- at the independently audited Triple caustic crossing, all ten public-curve
  derivatives agree with high-accuracy native finite differences: nine are
  within 1% and the source-radius derivative is within 4.3%;
- a bounded caustic trajectory probe (10 warmup plus 10 retained draws) had
  zero divergences, mean acceptance 0.924, and 5--7 leapfrog steps. Its
  four trajectory derivatives agree with native differences within 0.67%.

At the caustic, an eight-epoch value-plus-gradient takes about 0.30 s versus
1.52 s for a native forward-only curve. The smooth 12-epoch figures are
0.138 s and 0.525 s. These results establish that the gradients are usable by
HMC; they do not claim that an uninitialized fully coupled ten-dimensional
Triple posterior is cheap or automatically well conditioned.

Finite differences of the discretized JAX primal require care. Numerical
support, cell classification, and cell size are intentionally
stopped-gradient. Differences smaller than the requested integration error
can therefore disagree badly with the analytic physical derivative. The
audit uses parameter-specific steps and an independent high-accuracy native
curve: for example, the apparent smooth-Triple `s` error falls from 6.7% to
0.12% at its stable step, while an excessively small `rho` difference is
dominated by native integration noise.

Recorded artifacts are
`docs/assets/jax_ir_hmc_binary_7d.json`,
`docs/assets/jax_ir_hmc_triple_10d_smooth.json`,
`docs/assets/jax_ir_hmc_triple_4d_smooth.json`,
`docs/assets/jax_ir_hmc_triple_6d_lens_gradient.json`,
`docs/assets/jax_ir_hmc_triple_10d_caustic_gradient.json`, and
`docs/assets/jax_ir_hmc_triple_4d_caustic_short.json`.

### 14.15 Public higher-order completion

The public `Options(jax=True)` adapter now carries the two remaining
native-supported higher-order connections. A space `Site` exposes its
validated, Cartesian AU ephemeris to the JAX geometry layer; spacecraft
parallax is linearly interpolated with the same reduced/full-JD convention as
the native engine and composes with annual parallax.

Binary-source xallarap now routes through the existing two-body trajectory
primitive instead of applying a single-source offset independently to both
components. The public path supports circular/orbital elements, circular/
Kepler velocity states, and both direct-xallarap and trajectory-offset
coordinates. Annual, terrestrial, or space parallax and binary-lens orbital
motion remain additive around the source orbit. The two component
magnifications are evaluated independently and flux-weighted only afterward.

Public regression coverage compares all six xallarap mode/coordinate
combinations against native values, differentiates the source mass ratio,
checks a composed space-parallax binary-source curve, and exercises the same
connection with a triple lens. Higher-order evaluation remains intentionally
limited to VBM-compatible coordinates; adding another coordinate convention
is outside this phase.

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
