# Controlled pure-kernel speed comparison (2026-08-13)

This report records the final controlled comparison of `lcbinint` and
VBMicrolensing for the cache-warm finite-source integration kernel. It is a
direct integrator comparison, not a production dispatcher win-rate estimate.

## Benchmark design

- 160 independent binary-lens configurations were generated with `s`, `q`,
  and `rho` independently log-uniform over `[0.2, 4]`, `[1e-4, 1]`, and
  `[3e-5, 1]`.
- One source position was selected in each of the five equal-width measured
  `d/rho` bins `[0, 0.4)`, `[0.4, 0.8)`, `[0.8, 1.2)`, `[1.2, 1.6)`, and
  `[1.6, 2]`. This gives 800 positions and 1,600 profile rows (800 uniform
  and 800 linear-LD rows).
- Each profile row is evaluated at both requested tolerances, giving 3,200
  profile-tolerance jobs. Each job contains four reference epochs. Therefore,
  each row of the overall table has 3,200 nominal epoch measurements, while
  the four profile/tolerance combinations together contain 12,800 nominal
  epoch measurements before unresolved points are removed.
- Both a uniform source and a linearly limb-darkened source with `c=0.5`
  were evaluated at requested relative tolerances `1e-3` and `1e-4`.
- `lcbinint` independently increased `Nbin` and selected the first run of
  three increasing resolutions whose relative spread satisfied the requested
  tolerance. Cartesian and polar candidates were both timed; the faster
  self-converged candidate was retained. Search time was excluded.
- VBMicrolensing was evaluated independently with the requested `RelTol`.
  Its value was not used to select `Nbin`; cross-engine value differences are
  retained only as diagnostics and do not determine timing eligibility.
- The speed ratio is

  ```text
  R = t_VBM / t_lcbinint
  ```

  Thus `R > 1` means that `lcbinint` is faster. Timings exclude object
  construction, trajectory handling, and warm-up, and use one physical CPU
  worker with `OMP_NUM_THREADS=1`.

## Hardware

All timing measurements were performed on an Intel Xeon Gold 6530 system
with two sockets and 32 physical cores per socket. Each timing sample used a
single CPU worker; no multi-threaded speedup was included.

## Overall result

There are 3,200 profile-tolerance jobs in total (`800 positions x 2 source
profiles x 2 requested tolerances`). Each job contributes four reference
epochs. Thus each row below has 3,200 nominal epoch measurements, and the
four rows together have 12,800 nominal measurements. The recorded total is
12,797 because three epochs remain unresolved by the `Nbin=400` cap.

| profile | `epsilon_rel` | measured epochs | lcbinint wins | VBM wins | unresolved epochs | timeout jobs | cross-engine `abs(Delta) > epsilon` | win rate | p10 `R` | p25 `R` | p50 `R` (median) | p75 `R` | p90 `R` | median `Nbin` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform | `1e-3` | 3,200 | 2 | 3,198 | 0 | 0 | 43 | 0.1% | 0.037 | 0.080 | 0.199 | 0.368 | 0.507 | 25 |
| uniform | `1e-4` | 3,199 | 0 | 3,199 | 1 | 0 | 62 | 0.0% | 0.012 | 0.047 | 0.105 | 0.185 | 0.280 | 75 |
| linear LD | `1e-3` | 3,200 | 1,740 | 1,460 | 0 | 0 | 108 | 54.4% | 0.197 | 0.459 | 1.132 | 2.038 | 3.100 | 25 |
| linear LD | `1e-4` | 3,198 | 1,899 | 1,299 | 2 | 0 | 186 | 59.4% | 0.143 | 0.479 | 1.432 | 3.058 | 5.368 | 75 |

There were no final job-level timeouts. Three measurements remained
unresolved because the lcbinint self-convergence search did not certify a
resolution by `Nbin=400`; they were kept explicitly rather than silently
discarded. Candidate-search timeouts are reported separately in the raw
results.

## Dependence on measured `d/rho`

The equal-width strata show the clearest separation for linear limb
darkening:

| profile | `epsilon_rel` | `d/rho` bin | measured epochs | lcbinint win rate | p10 `R` | p25 `R` | p50 `R` (median) | p75 `R` | p90 `R` |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| uniform | `1e-3` | `[0, 0.4)` | 640 | 0.0% | 0.018 | 0.043 | 0.081 | 0.162 | 0.223 |
| uniform | `1e-3` | `[0.4, 0.8)` | 640 | 0.0% | 0.027 | 0.058 | 0.110 | 0.190 | 0.295 |
| uniform | `1e-3` | `[0.8, 1.2)` | 640 | 0.2% | 0.047 | 0.113 | 0.208 | 0.364 | 0.480 |
| uniform | `1e-3` | `[1.2, 1.6)` | 640 | 0.2% | 0.069 | 0.195 | 0.352 | 0.499 | 0.614 |
| uniform | `1e-3` | `[1.6, 2]` | 640 | 0.0% | 0.130 | 0.254 | 0.349 | 0.459 | 0.563 |
| linear LD | `1e-3` | `[0, 0.4)` | 640 | 67.8% | 0.250 | 0.724 | 1.688 | 2.806 | 4.110 |
| linear LD | `1e-3` | `[0.4, 0.8)` | 640 | 65.5% | 0.185 | 0.684 | 1.503 | 2.561 | 3.694 |
| linear LD | `1e-3` | `[0.8, 1.2)` | 640 | 54.1% | 0.205 | 0.491 | 1.101 | 1.879 | 2.833 |
| linear LD | `1e-3` | `[1.2, 1.6)` | 640 | 45.8% | 0.165 | 0.381 | 0.848 | 1.751 | 2.528 |
| linear LD | `1e-3` | `[1.6, 2]` | 640 | 38.8% | 0.238 | 0.406 | 0.766 | 1.305 | 1.811 |
| linear LD | `1e-4` | `[0, 0.4)` | 640 | 71.9% | 0.250 | 0.771 | 2.781 | 5.535 | 8.053 |
| linear LD | `1e-4` | `[0.4, 0.8)` | 638 | 72.3% | 0.270 | 0.875 | 2.502 | 4.588 | 7.321 |
| linear LD | `1e-4` | `[0.8, 1.2)` | 640 | 61.9% | 0.125 | 0.503 | 1.512 | 2.721 | 4.138 |
| linear LD | `1e-4` | `[1.2, 1.6)` | 640 | 45.3% | 0.081 | 0.243 | 0.841 | 1.923 | 3.187 |
| linear LD | `1e-4` | `[1.6, 2]` | 640 | 45.6% | 0.136 | 0.392 | 0.856 | 1.604 | 2.318 |

The controlled sample therefore supports a conditional statement: linear
limb darkening and small measured `d/rho` are favorable to `lcbinint`, while
uniform sources remain strongly VBM-favorable in this kernel-only test. It
does not support a one-variable rule based only on `A_finite` or only on
`rho`.

The percentile table and the violin plot should be read together: the
distribution is broad and spans multiple decades, so the median alone does
not describe the typical spread. The violin plot uses a logarithmic `R` axis;
the thin and thick black marks show p10--p90 and p25--p75, respectively, and
the white point shows p50. Its x-bins are the four logarithmic `rho` bins
`[3e-5,1e-3)`, `[1e-3,1e-2)`, `[1e-2,1e-1)`, `>=1e-1`, the five equal-width
measured `d/rho` bins used by the benchmark, and four logarithmic `A_finite`
bins `[1,10)`, `[10,100)`, `[100,1000)`, `>=1000`.

## Figure and provenance

The paper-facing 2-by-3 scatter figure shows `R` against `rho`, measured
`d/rho`, and `A_finite`, with uniform and linear-LD profiles shown in blue and
red. The matching binned violin figure keeps the same axes, limits, panel
layout, and profile colours while replacing the point clouds with paired
within-bin distributions:

[Final 2-by-3 speed-ratio figure](figures/controlled_parameter_vs_R_2x3_profiles_20260813.pdf)

[Binned 2-by-3 speed-ratio violin plot](figures/controlled_speed_ratio_violin_20260813.pdf)

The full raw merged JSON is intentionally not committed because it is about
35 MB. The benchmark harness, the updated handoff, and the compact summary
above are the versioned provenance; the raw run remains in the local
diagnostics workspace.
