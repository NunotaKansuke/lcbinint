# C++ algebraic-boundary integration: algorithm and validation

Date: 2026-08-07  
Branch: `algebraic-boundary-cpp`  
Implementation commits: `ae42f7f`, `2f65d98`, `2d8107c`

## Scope

This is an experimental, native C++ value kernel for a binary lens. It does
not change the public C API, FFI, JAX implementation, automatic dispatcher, or
the existing Cartesian and polar production kernels. A call measures a whole
single-source-position evaluation, including point-image seed generation,
component-support certification and completion, radial-band discovery, and
the integral. It is not an integration-only microbenchmark.

The polar algebraic kernel is the optimized path described below. The branch
also contains an algebraic Cartesian sextic implementation, but it continues
to use the shared Skowron--Gould solver with root continuation. The quartic
and radial-tangency optimizations are specific to the polar path.

## Boundary polynomial

Use native binary coordinates with lenses at `0` and `a`. For a fixed image
radius, write `z = R u`, where `|u| = 1`, and absorb the native source shift in

\[
\zeta_s = \zeta_c + a m_1.
\]

The source-boundary condition can be written as

\[
|A(u)|^2 - \rho^2 |D(u)|^2 = 0,
\]

with

\[
\begin{aligned}
A_0 &= -\zeta_s R^2,\\
A_1 &= R(R^2 - 1 + a\zeta_s),\\
A_2 &= a(m_2-R^2),\\
D_0 &= R^2,\\
D_1 &= -aR.
\end{aligned}
\]

On the unit circle, reverse-conjugating the factors gives the constant-first
quartic

\[
P(u)=A(u)\widetilde A(u)-\rho^2D(u)\widetilde D(u).
\]

Its coefficients are Hermitian palindromic:

\[
p_0=\overline{p_4},\qquad
p_1=\overline{p_3},\qquad
p_2\in\mathbb R.
\]

The hot path now constructs these five coefficients in a fixed-size array.
The public experimental coefficient helper still returns a vector for
compatibility with its tests.

## Real self-inversive quartic solver

Let

\[
p_4=\alpha+i\beta,\quad p_3=\gamma+i\delta,\quad p_2=\epsilon.
\]

Multiplying `P(exp(i theta))` by `exp(-2 i theta)` produces the real
trigonometric polynomial

\[
F(\theta)=\epsilon
+2(\gamma\cos\theta-\delta\sin\theta)
+2(\alpha\cos2\theta-\beta\sin2\theta).
\]

With `t = tan(theta/2)`, its real quartic coefficients are

\[
\begin{aligned}
q_0 &= \epsilon+2\gamma+2\alpha,\\
q_1 &= -4\delta-8\beta,\\
q_2 &= 2\epsilon-12\alpha,\\
q_3 &= -4\delta+8\beta,\\
q_4 &= \epsilon-2\gamma+2\alpha.
\end{aligned}
\]

The solver performs the following operations without a complex iterative
solve:

1. Detect degree drops and the root at `theta = pi`.
2. Solve the derivative cubic analytically.
3. Use the critical points to partition the real line into monotone pieces.
4. Find sign-changing roots with safeguarded Newton iterations.
5. Polish and map the real roots back to the unit circle.
6. Validate them against the original complex quartic and then against the
   original lens-map boundary residual.

Skowron--Gould remains the certification fallback. It is used when the
self-inversive structure, degree handling, polynomial residual, lens residual,
or even physical-root count is unsafe. This last check matters: a tiny-`q`
regression geometry produced an acceptable polynomial residual but lost one
root under the original lens-map residual. Treating the resulting odd count
as a fast-solver failure and retrying Skowron--Gould restored all 24 trajectory
positions. That geometry is now a focused regression test.

## Angular intervals and radial support

Certified unit-circle roots are converted to cyclic boundary angles. Adjacent
angles are classified by one midpoint test. With no roots, one direct test
classifies the full circle as full or empty. Duplicate tangency pairs cancel;
an odd physical-root count is never returned as a successful integral.

Each certified point-image seed starts a radial support search. Exponential
steps first bracket the active/empty transition. Three Boolean bisections then
identify the component and initialize the simultaneous tangency equations

\[
F(R,\theta)=0,\qquad \partial_\theta F(R,\theta)=0.
\]

A two-variable Newton solve uses analytic angular derivatives and a centered
finite-difference radial derivative. A proposed endpoint is accepted only
when a radius on the active side still contains an image and a radius on the
outside side is empty. Otherwise the code completes the original 16-step
bisection. Therefore the tangency solve is an acceleration, not a weakened
support test. Bands from all certified components are finally merged.

The smooth reference case changed from 157 boundary solves to 108. Of those,
106 used the real quartic path and two used Skowron--Gould. Five radial
endpoints used the direct tangency solve and one used the bisection fallback.

## Integration

For each fixed radius, the uniform angular moment is the exact sum of the
certified interval widths. Uniform brightness therefore requires no angular
brightness quadrature, although the remaining one-dimensional radial integral
is still numerical.

Linear and square-root limb-darkening moments use a sine-mapped embedded
Gauss--Kronrod 10/21 angular rule. An earlier optimization tried only the
7/15 value and was faster, but one independent-reference case moved to
`1.36e-4` relative error. The final 10/21 rule brought it to `5.55e-5` while
still replacing the old 15-point embedded pass plus a separate 16-point
Legendre pass.

The radial integral uses a sine-mapped embedded Gauss--Kronrod 7/15 rule and
best-first adaptive panel subdivision. The sine map regularizes the
square-root opening of an image interval at a support endpoint. The uniform,
`Phi^(1/2)`, and `Phi^(1/4)` moments are accumulated together.

## Benchmark method

The reported measurements used:

- Release build, GCC 11.5;
- Intel Xeon Gold 6530;
- `OPENBLAS_NUM_THREADS=1`;
- VBMicrolensing 5.5;
- direct scalar `BinaryMag` for uniform brightness;
- direct scalar `BinaryMagDark` for linear limb darkening, with `a1=0.5`,
  `LDlinear`, `minannuli=8`, and the sixth argument set to the absolute
  accuracy (`1e-12`), not the limb-darkening coefficient;
- warm-up before repeated timing, with the median of five repetitions for the
  expanded VBM comparison.

The small core reference uses the existing 640-bin polar raster. Because its
reference is not exact, requested tolerance and measured error are both shown.
The recal2026 corpus gate instead uses its stored independent reference epochs
and selects the cheapest setting whose measured relative error is at most
`1e-4`.

## Results

### Change relative to the previous algebraic kernel

This is a full single-position call on the smooth binary case, not an isolated
integral. Times are medians over 15 repetitions.

| profile | tolerance | previous | optimized | speedup |
|---|---:|---:|---:|---:|
| uniform | `1e-3` | 237.3 us | 162.4 us | 1.46x |
| linear LD | `1e-3` | 280.6 us | 191.9 us | 1.46x |
| uniform | `1e-4` | 237.2 us | 162.3 us | 1.46x |
| linear LD | `1e-4` | 280.5 us | 192.1 us | 1.46x |

### Expanded 17-geometry linear-limb-darkening comparison

The suite includes smooth, central, caustic-crossing, wide, resonant, tiny
source, and near-tangent cases. Every polar and Cartesian algebraic call
completed successfully at all three requested tolerances: 51/51 for each
coordinate system.

| requested tolerance | polar p90 error | VBM p90 error | median VBM / polar time | polar wins |
|---:|---:|---:|---:|---:|
| `1e-2` | `4.01e-3` | `1.15e-3` | 0.91x | 0/17 |
| `1e-3` | `4.50e-4` | `3.22e-4` | 1.57x | 13/17 |
| `1e-4` | `3.67e-4` | `6.66e-5` | 2.78x | 15/17 |

At `1e-3`, both p90 errors are below the requested tolerance, so the 1.57x
median advantage is a useful like-for-like summary. At requested `1e-4`, the
polar p90 error does not meet `1e-4`; the 2.78x timing ratio must not be
presented as a globally matched-accuracy result.

For the five-case uniform core at requested `1e-3`, all calls succeeded. The
polar p90 error was `2.07e-4`, the VBM p90 error was `1.85e-4`, and
`VBM/polar = 0.252`: VBM was about 3.96x faster. Thus the principal competitive
result remains linear limb darkening; uniform brightness still favors VBM.

At requested `1e-3`, the optimized polar kernel was 6.61x faster than the old
polar raster across the expanded linear suite and 24.1x faster across the
five-case uniform core.

### Independent-reference accuracy gate

On the stratified recal2026 subset at measured relative error `<=1e-4`:

- linear limb darkening: 16/16 rows found a successful qualifying setting;
- uniform brightness: 12/16 rows found a successful qualifying setting.

The four uniform misses are accuracy-gate misses, not evidence that the kernel
is faster than VBM at matched accuracy. They are another reason not to route
production uniform calculations to this experimental path yet.

## Verification commands

```text
cmake --build build --target test_core test_algebraic_boundary \
  benchmark_algebraic_boundary -j 8
OPENBLAS_NUM_THREADS=1 ctest --test-dir build --output-on-failure
OPENBLAS_NUM_THREADS=1 python \
  tests/diagnostics/benchmark_algebraic_vbm.py 5 8 limb-sweep
```

The final Release test run passed `unit_core` and `unit_algebraic_boundary`.
Focused coverage includes quartic identity, dense angular crossings,
wrap-around, full/empty circles, Cartesian sextic identity, profile agreement,
redundant certified seeds, tangency acceleration, and the ill-conditioned
real-quartic Skowron--Gould fallback.

## Current conclusion

The optimized polar algebraic method is a credible experimental linear
limb-darkening kernel: it is 1.46x faster than the first algebraic
implementation on a smooth full-call benchmark and, at requested `1e-3`, is
1.57x faster than VBM on the median expanded case while winning 13/17 cases.
It is not ready for automatic routing. Uniform performance remains behind
VBM, the independent-reference uniform gate is incomplete, and the requested
`1e-4` setting is not yet a reliable measured-`1e-4` guarantee across the
expanded geometry suite.

## 2026-08-09 pure-grid handoff reproduction

The handoff at
`tests/diagnostics/recal2026/HANDOFF_pure_grid_vbm_speed_comparison_20260809.md`
compares only finite-source work and defines

```text
R = t_VBM / t_LCB-in
```

The new comparison uses exactly the same important choices: the recorded pure
`lcbinint_auto` grid-route filter, `q >= 1e-4`, intended `d/rho < 2.01`, the
ten factors from `0` through `2`, reference epochs `(0, 7, 15, 23)`, VBM's
`BinaryMag`/`BinaryMagDark` calls, `a1=0.5` for linear LD, and the sixth
`BinaryMagDark` argument fixed at `1e-12`. The C++ batch executable measures a
whole experimental algebraic call, including seed generation, support
completion, radial bands, and integration. The first call per source warms the
object; two subsequent calls are timed and the median is kept.

The full stratified run used `case-count=160` per factor and produced the
following point counts. `measured` means both a qualifying algebraic result and
a qualifying VBM result were available; unresolved points remain visible in
the table instead of being silently counted as wins.

| profile | target | points | measured | algebraic wins | win rate / all | median R | p10--p90 R |
|---|---:|---:|---:|---:|---:|---:|---:|
| uniform | `1e-2` | 1132 | 1108 | 0 | 0.0% | 0.0598 | 0.0166--0.1317 |
| uniform | `1e-3` | 1132 | 1099 | 0 | 0.0% | 0.0919 | 0.0263--0.1932 |
| uniform | `1e-4` | 512 | 507 | 0 | 0.0% | 0.1986 | 0.0806--0.3852 |
| linear LD | `1e-2` | 1124 | 1117 | 58 | 5.2% | 0.3047 | 0.0934--0.7932 |
| linear LD | `1e-3` | 1124 | 1111 | 626 | 55.7% | 1.1353 | 0.3490--2.6748 |
| linear LD | `1e-4` | 648 | 633 | 612 | 94.4% | 5.2034 | 1.8544--11.8643 |

Among measured points only, the linear-LD win rates are 5.2%, 56.3%, and
96.7% for `1e-2`, `1e-3`, and `1e-4`. The difference between the two
denominators is entirely unresolved algebraic or VBM points, not a timeout win.
For this run there were 28 polar unsafe calls at the first two targets and 21
at `1e-4`; Cartesian itself had no unsafe calls. The common polar reasons were
certified seed support not active, a seed not lying on an active radius, and a
small number of odd physical-root certifications. The Cartesian method was
therefore used as the faster qualifying algebraic candidate where appropriate.

Relative to the handoff's previous production-grid medians, the new algebraic
median `R` is approximately 2.3--2.5 times larger for uniform `1e-2`/`1e-3`,
5.5 times larger for uniform `1e-4`, 2.3 times larger for linear `1e-2`, 2.1
times larger for linear `1e-3`, and 3.5 times larger for linear `1e-4`. This is
an indicative improvement because unresolved points make the measured subsets
slightly different; it is not a paired per-point speedup claim.

The handoff reproduction is implemented by:

- `tests/diagnostics/benchmark_algebraic_batch.cpp`, a native batch runner;
- `tests/diagnostics/recal2026/benchmark_algebraic_vs_vbm_dark.py`, which
  applies the corpus filter, runs the batch runner, performs the VBM ladder,
  and writes per-point JSON plus summaries;
- CMake target `benchmark_algebraic_batch`.

Example command:

```bash
LCBININT_RECAL_CORPUS=tests/diagnostics/results/recal2026/speed_discovery \
python tests/diagnostics/recal2026/benchmark_algebraic_vs_vbm_dark.py \
  --executable build/benchmark_algebraic_batch \
  --case-count 160 --profiles uniform linear \
  --targets 1e-2 1e-3 1e-4 --repeats 2 \
  --output /tmp/algebraic-vbm-full.json
```
