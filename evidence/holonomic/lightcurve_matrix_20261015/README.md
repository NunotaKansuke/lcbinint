# Public LightCurve performance matrix

## What was measured

Ten specified binary-lens trajectories, each with uniform intensity (`linear(0)`), linear limb-darkening coefficient 0.5 and 0.9, and atol=reltol=1e-3 and 1e-4: **60 configurations, 401 uniformly spaced epochs each**. Parameters and time ranges are in cases.json. This is a targeted matrix, not a random population or global certification.

The headline timing is the normal notebook call `curve.info(times, params)` versus `VBMicrolensing.BinaryLightCurve(params, times.tolist())`, at matched requested tolerances, VBM coordinates and linear-intensity coefficients. Constructors and parameter conversion are excluded as in the original notebook. Each process uses one pinned CPU and one BLAS/OpenMP thread. A cold call is recorded, followed by three repetitions with alternating backend order; medians are reported. CPU/native modules and exact hashes are in provenance.json. VBM version is 5.5. Native value library: Release, -O3 -DNDEBUG. The fused gradient module is separately built at O2.

`curve(times, params)` is also called normally and matches `info()['mu']` bitwise on all 60 configurations. The ordinary API currently obtains those values via the same rich diagnostic route. Gradient calls are separate single-call timings, with finiteness recorded; they are not included in the value/VBM speed ratios and are not gradient precision measurements.

The initial three-case pilot took 4.34 seconds; the remaining sequential matrix run took 94.63 seconds, with a 40-second timeout per worker. All 60 completed. Peak worker RSS was 54 MiB. Data generation and diagnostic reference checks are complete; scientific acceptance is qualified below.

## Results

Times below are milliseconds per 401-point curve. Ratio >1 means lcbinint is faster.

| Case | LD coefficient | atol=rtol | lcbinint | VBM | VBM / lcbinint |
|---|---:|---:|---:|---:|---:|
| Ordinary, all point routes | .5 | 1e-3 | .38 | .35 | .92 |
| Wide planetary caustic | .5 | 1e-3 | 9.99 | 21.91 | 2.19 |
| Wide planetary caustic | .5 | 1e-4 | 12.20 | 81.42 | 6.67 |
| Resonant | .5 | 1e-3 | 35.03 | 39.97 | 1.14 |
| Earlier broad source, rho=.05 | 0 | 1e-3 | 186.34 | 31.19 | .17 |
| Large source, rho=.5 | 0 | 1e-3 | 220.21 | 18.38 | .083 |
| Large source, rho=.5 | .5 | 1e-3 | 234.09 | 561.44 | 2.40 |
| Earlier broad source, rho=.05 | .9 | 1e-4 | 224.02 | 3076.99 | 13.73 |

lcbinint is faster in 43/60 selected configurations; median per-configuration speed ratio is 1.85. This unweighted median must not be interpreted as an expected speedup for an observing population. Main pattern: linear LD is relatively inexpensive for lcbinint, whereas VBM's uniform-source route is much faster on broad sources. Tiny sources and distant trajectories often use point/hex routes, so their very small timings do not benchmark the finite-source kernel alone.

All 24060 values are finite and report value convergence. All 168420 mapped derivative components are finite. **None of the 7887 integral-route epochs certifies global radial support**; the 16173 non-integral epochs account for the fully-converged count. Finite gradients and value convergence do not establish full solver or derivative accuracy. The existing flags were preserved.

The exact latest notebook setting with alpha=180 is included as `notebook_alpha180_rad`. The API interprets that number in radians; the benchmark preserves it and does not silently convert it to degrees.

## Accuracy: avoid treating VBM as truth

The comparison metric is `abs(A_lc-A_vbm)/max(atol,rtol*abs(A_vbm))`. It is an agreement diagnostic, not an error certificate for either solver. 54/60 configurations stay within one requested budget by this metric. Six broad-source LD configurations exceed it; the largest relative pairwise difference is 7.67% at t=.272.

Eight worst-case LD points were checked with tighter native/VBM settings and an independent LD integral constructed from **VBM uniform disks**, bypassing its built-in LD algorithm. Integration by parts gives

```
A_LD(rho,c) = [(1-c)*A_U(rho)
 + c*integral_0^1 (1-z^2)*A_U(rho*sqrt(1-z^2)) dz] / (1-c/3).
```

Gauss-Legendre orders 32, 64, 128 and 256 were compared. At these eight points:

- lcbinint and the annulus result differ by at most about 1.2e-6 relative.
- The annulus 128-to-256 change is at most about 1.9e-6 relative, so these are numerical consistency checks, not certified errors.
- Native and VBM **uniform outer disks** agree to about 1.1e-10 relative.
- Built-in VBM LD differs by up to 7.12% from this annulus reference; the pairwise 7.67% denominator uses the smaller VBM value. Near t=0, several discrepancies remain around 3e-4 even after tightening VBM to 1e-8.
- Native strict-tolerance checks frequently have value_converged=false; they are explicitly not treated as certified references.

These observations support VBM LD as the origin of the large disagreements at the checked points. They do not prove which solver dominates every untested epoch. Raw step/order data are in *_reference.json and reference_summary.csv. No issue or message was sent externally.

## Where improvement is worth pursuing

Headline costs above come from public APIs. Additional **diagnostic** stage profiling samples 73 integral epochs from eight configurations (five spread across the integral epochs plus up to five slow epochs each). It uses the same value helper and reproduces public values to within 4.9e-13 relative. Five repeats and per-epoch medians suppress isolated OS scheduling delays. Instrumentation timings and sampling favor difficult epochs, so they are not another full-curve benchmark. Nested timing counters must not be summed.

### 1. Large-source uniform performance: prepare fewer equivalent witnesses

In the nine sampled rho=.5 uniform epochs, seed preparation takes 3.75 ms and the radial kernel 1.74 ms. The kernel receives 5409 candidate seeds in total, about **601 per epoch**. This does not mean there are 601 physical images. It indicates many geometric witness candidates feeding a small number of occupied radial bands. Seed preparation is about 68% of the measured preparation-plus-kernel subset.

First candidate: reuse static caustic segment geometry over a curve and bundle candidates whose equivalence is established by the same support/radial interval witness. Preserve the original component evidence while reducing duplicate root/count work. Verify parity of support/count witnesses and route outcomes, including disk/caustic contact. Simply dropping seeds by distance or reusing a previous epoch without verification would not establish equivalent coverage.

This target has a useful cost ceiling: even eliminating the measured seed preparation entirely would only give about 3.2x on that sampled subset. It cannot by itself explain away the full 12x uniform-source gap. Reducing both preparation and radial-boundary planning is needed.

### 2. Resonant and wide uniform curves: boundary-event continuation

For the ten sampled resonant uniform epochs, planning is 2.01 of 2.92 ms of kernel time; the event timer is 1.76 ms. Wide-caustic uniform samples show a similar planning burden. A curve-scoped predictor for physical folds, accepted only after the same residual/bracket/count checks, is a plausible next experiment. Retain cold fallback at count changes and tangent/cusp cases. Source code currently constructs a new adaptive workspace per integral epoch; capacity reuse can also be measured separately, though allocation alone is too small to explain the gap.

### 3. rho=.05 broad source: radial integration work

On nine sampled uniform epochs, radial integration is 9.27 of 10.24 ms kernel time (about 79% after including 1.46 ms seed preparation). The samples use 2986 node evaluations and 14 integration calls, including five local topology repairs. This points to node/count work and repaired radial intervals, rather than LD arithmetic. The kernel already skips LD quadrature for c=0; removing that work again is not an improvement opportunity. Investigate remaining repeated operations during interval repair and radial node evaluation while preserving the accepted tolerance and topology checks. The phase data do not yet identify a single scalar operation responsible for all the cost.

### Precision and assurance priority

The highest-priority accuracy work remains global support verification and a wider independently checked curve set. All current integral support flags are false even when pairwise differences are tiny. The diagnostic/reference infrastructure added here can check future optimizations without relaxing those flags or the tolerance contract. No new kernel optimization was promoted during this evaluation task.

## Notebook and reproduction

New notebook: `example/compare-vbm/lcbinint_vbm_lightcurve_performance.ipynb`. It loads the complete table, allows selecting a curve, plots magnification and relative backend difference, and contains the ordinary public calls for recomputation. It leaves the user's existing notebook untouched. Read/plot cells were executed in ordinary Python and preview.png was rendered and visually inspected; a separate Jupyter UI session was not launched.

From this existing checkout:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python evidence/holonomic/lightcurve_matrix_20261015/benchmark.py pilot
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python evidence/holonomic/lightcurve_matrix_20261015/benchmark.py matrix
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python evidence/holonomic/lightcurve_matrix_20261015/reference.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python evidence/holonomic/lightcurve_matrix_20261015/analyze.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python evidence/holonomic/lightcurve_matrix_20261015/make_notebook.py
```

The matrix resumes by skipping existing per-case JSON files; use a fresh copy of this evidence directory or explicitly select outputs for a deliberate rerun. Per-case timeout/failure is recorded and not silently omitted. Inputs, raw arrays, individual timing repetitions, statuses, route counts, and source/module hashes are retained. The current source identity is HEAD plus the earlier uncommitted angular-shortcut fix; provenance.json hashes that header and the actual modules. The value extension is unchanged by that gradient fix. Results.csv is the main machine-readable table; summary.json and stage_summary.json preserve qualification counts and diagnostic stage aggregates.
