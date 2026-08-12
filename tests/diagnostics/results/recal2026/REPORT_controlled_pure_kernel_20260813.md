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
  `[1.6, 2]`. This gives 800 positions, 1,600 profile rows, and 3,200
  target/profile measurements before unresolved points are removed.
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

| profile | `epsilon_rel` | measured | lcbinint wins | VBM wins | unresolved | timeout jobs | cross-engine `abs(Delta) > epsilon` | win rate | median `R` | p10 | p90 | median `Nbin` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform | `1e-3` | 3,200 | 2 | 3,198 | 0 | 0 | 43 | 0.1% | 0.199 | 0.037 | 0.507 | 25 |
| uniform | `1e-4` | 3,199 | 0 | 3,199 | 1 | 0 | 62 | 0.0% | 0.105 | 0.012 | 0.280 | 75 |
| linear LD | `1e-3` | 3,200 | 1,740 | 1,460 | 0 | 0 | 108 | 54.4% | 1.132 | 0.197 | 3.100 | 25 |
| linear LD | `1e-4` | 3,198 | 1,899 | 1,299 | 2 | 0 | 186 | 59.4% | 1.432 | 0.143 | 5.368 | 75 |

There were no final job-level timeouts. Three measurements remained
unresolved because the lcbinint self-convergence search did not certify a
resolution by `Nbin=400`; they were kept explicitly rather than silently
discarded. Candidate-search timeouts are reported separately in the raw
results.

## Dependence on measured `d/rho`

The equal-width strata show the clearest separation for linear limb
darkening:

| profile | `epsilon_rel` | `d/rho` bin | lcbinint win rate | median `R` |
|---|---:|---|---:|---:|
| uniform | `1e-3` | `[0, 0.4)` | 0.0% | 0.081 |
| uniform | `1e-3` | `[0.4, 0.8)` | 0.0% | 0.110 |
| uniform | `1e-3` | `[0.8, 1.2)` | 0.2% | 0.208 |
| uniform | `1e-3` | `[1.2, 1.6)` | 0.2% | 0.352 |
| uniform | `1e-3` | `[1.6, 2]` | 0.0% | 0.349 |
| linear LD | `1e-3` | `[0, 0.4)` | 67.8% | 1.688 |
| linear LD | `1e-3` | `[0.4, 0.8)` | 65.5% | 1.503 |
| linear LD | `1e-3` | `[0.8, 1.2)` | 54.1% | 1.101 |
| linear LD | `1e-3` | `[1.2, 1.6)` | 45.8% | 0.848 |
| linear LD | `1e-3` | `[1.6, 2]` | 38.8% | 0.766 |
| linear LD | `1e-4` | `[0, 0.4)` | 71.9% | 2.781 |
| linear LD | `1e-4` | `[0.4, 0.8)` | 72.3% | 2.502 |
| linear LD | `1e-4` | `[0.8, 1.2)` | 61.9% | 1.512 |
| linear LD | `1e-4` | `[1.2, 1.6)` | 45.3% | 0.841 |
| linear LD | `1e-4` | `[1.6, 2]` | 45.6% | 0.856 |

The controlled sample therefore supports a conditional statement: linear
limb darkening and small measured `d/rho` are favorable to `lcbinint`, while
uniform sources remain strongly VBM-favorable in this kernel-only test. It
does not support a one-variable rule based only on `A_finite` or only on
`rho`.

## Figure and provenance

The paper-facing 2-by-3 figure shows `R` against `rho`, measured `d/rho`,
and `A_finite`, with the two source profiles shown as blue uniform and red
linear-LD points:

[Final 2-by-3 speed-ratio figure](figures/controlled_parameter_vs_R_2x3_profiles_20260813.pdf)

The full raw merged JSON is intentionally not committed because it is about
35 MB. The benchmark harness, the updated handoff, and the compact summary
above are the versioned provenance; the raw run remains in the local
diagnostics workspace.
