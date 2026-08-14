# Handoff: LightCurve API comparison with VBMicrolensing

Date: 2026-08-14 (JST)

This document records only the light-curve-level example benchmark in
`example/compare-vbm/`. It is separate from the pure-kernel and 12,800-epoch
diagnostic campaigns.

## Scope

The benchmark calls the public `lcbinint.LightCurve` API for a complete array
of epochs and compares it with VBMicrolensing's complete light-curve API:

- binary: `VBMicrolensing.BinaryLightCurve`
- triple: `VBMicrolensing.TripleLightCurve`

The timing therefore includes the light-curve API path, trajectory handling,
route selection, and the reusable native object/cache state. It is not a
single-epoch pure integration-kernel benchmark.

The speed ratio is

\[
R = t_{\rm VBM}/t_{\rm lcbinint},
\]

so `R > 1` means that lcbinint is faster.

## Reproduction

Run from the repository root after building the in-tree extension:

```bash
cmake --build build -j8

OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
PYTHONPATH=build \
python example/compare-vbm/quickstart_compare_vbm.py

OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
PYTHONPATH=build \
python example/compare-vbm/quickstart_compare_vbm_triple.py
```

Each script uses seven timed repetitions and reports the median. The scripts
also save plots beside themselves:

- `example/compare-vbm/quickstart_compare_vbm.png`
- `example/compare-vbm/quickstart_compare_vbm_triple.png`

## Benchmark settings

Both scripts use 400 equally spaced epochs from `t=-0.8` to `t=0.8`.

### Binary

```text
s=0.95, q=1e-2, rho=5e-3
t0=0, tE=1, u0=-1e-3, alpha=0.5
coordinates="vbm", nbin="auto"
lcbinint tol=1e-3, reltol=1e-3
```

Two source profiles are evaluated:

- uniform: `LimbDarkening.none()`
- linear limb darkening: `c=0.5`, `d=0`

For VBM, `Tol=1e-3`, `a1=0` or `0.5`, and `a2=0`. The VBM trajectory vector
uses the logarithmic parameter convention required by `BinaryLightCurve`.

The binary script does not call `curve.warmup()`. The first complete
`LightCurve` and `BinaryLightCurve` calls are included in the timed samples.

### Triple

```text
s=1.2, q=1e-2, q2=1e-3
sep2=1, ang=0.5, rho=5e-3
t0=0, tE=1, u0=0.05, alpha=0
coordinates="vbm", nbin="auto"
```

The same uniform and `c=0.5` linear limb-darkened profiles are used. VBM is
run with `Tol=1e-3` and its `Multipoly` root method when available.

The triple script performs one unmeasured priming call for each engine before
the seven timed calls. This is an object/cache warm-state comparison, but it is
not the explicit lcbinint `curve.warmup()` API.

Important: the triple script currently does not pass `reltol=1e-3` explicitly
to `lcbinint.Options`, nor does it set VBM's `RelTol` explicitly. It relies on
the respective defaults while setting VBM's absolute `Tol=1e-3`. This is fine
for this example smoke benchmark, but should be fixed before using the triple
numbers as a paper-grade matched-tolerance result.

## Results from the current run

All times below are milliseconds per epoch; the reported value is the median
of seven complete light-curve calls.

| lens/profile | lcbinint | VBM | `R=t_VBM/t_lcbinint` | interpretation |
|---|---:|---:|---:|---|
| binary, uniform | 0.41141 | 0.06449 | 0.1568 | VBM faster |
| binary, LD `c=0.5` | 0.51474 | 1.21060 | 2.3519 | lcbinint faster |
| triple, uniform | 39.4449 | 0.3107 | 0.0079 | VBM faster |
| triple, LD `c=0.5` | 36.3566 | 1.5152 | 0.0417 | VBM faster |

Equivalent qualitative summary:

- binary uniform: lcbinint is about 6.4 times slower;
- binary LD: lcbinint is about 2.35 times faster;
- triple uniform: lcbinint is about 127 times slower;
- triple LD: lcbinint is about 24 times slower.

These are single hand-picked configurations, not population-level win rates.

## Accuracy against VBM

The scripts compute epoch-wise

```text
abs(lcbinint - VBM) / max(abs(VBM), 1e-12)
```

and print the maximum, 99th percentile, median, and RMS.

| lens/profile | max | p99 | median | RMS |
|---|---:|---:|---:|---:|
| binary, uniform | `4.554e-4` | `5.152e-5` | `6.656e-16` | `2.702e-5` |
| binary, LD `c=0.5` | `3.362e-4` | `8.275e-5` | `6.656e-16` | `2.680e-5` |
| triple, uniform | `4.733e-4` | `4.000e-4` | `1.440e-5` | `8.762e-5` |
| triple, LD `c=0.5` | `4.733e-4` | `4.000e-4` | `2.151e-5` | `9.127e-5` |

The binary run explicitly requests a `1e-3` relative tolerance from lcbinint.
The triple accuracy numbers should be treated as diagnostic until the
explicit tolerance settings noted above are made symmetric.

## Current code state and provenance

The run was performed on branch `master` with repository HEAD:

```text
25bcd68 Improve light-curve VBM comparison example
```

The worktree was dirty. The native extension was rebuilt with
`cmake --build build -j8`; the in-tree extension used by the scripts had an
mtime of `2026-08-14 07:31:02 +0900`. Consequently, HEAD alone is not enough
to reproduce this exact run: preserve the working-tree changes or record a
new commit before treating these numbers as archival.

The current native automatic path used for this run has no source-plane
fallback. The explicit low-level source-plane APIs remain available, but the
two example scripts do not call them.

## Machine and software

- CPU: Intel Xeon Gold 6530
- sockets / cores: 2 sockets, 32 physical cores per socket
- timing: one CPU worker; `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
  `MKL_NUM_THREADS=1`
- Python: 3.10.19
- compiler: GCC 11.5.0
- VBMicrolensing: 5.5

## Follow-up before publication

1. Make the triple `reltol`/`RelTol` settings explicit and identical.
2. Decide whether the binary comparison is intended to include a cold first
   light-curve call or to use the same primed state as the triple comparison.
3. Repeat over a controlled parameter sample; these four rows are only an API
   smoke test and should not be quoted as a general speed conclusion.
4. Record the final commit and build identifier after the source-plane removal
   and any later LightCurve optimizations are finalized.
