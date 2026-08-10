# Pure one-epoch finite-source kernel comparison

This is the cache-warm integrator comparison. It is deliberately not a
`LightCurve` total-call benchmark.

## Timing definition

- lcbinint uses the stored minimum Nbin from the warm-up/speed-discovery
  corpus for each requested epsilon; no Nbin search is mixed into the
  timing run.
- For lcbinint, two identical epochs are evaluated inside one native
  `_evaluate_preplanned` call. The first epoch builds the LensModel and
  caustic cache; only the second epoch's native `seconds` value is used.
- VBM is warmed once at `RelTol=target`, then the direct finite-source
  call wall time is measured.
- `R = t_VBM / t_LCB-in`; `R > 1` means lcbinint is faster.

## Conditions

- q filter: `0.0001`; d/rho filter: `< 2.01`.
- External tolerances: `1e-3` and `1e-4` only.
- Only rows whose production auto route was finite-source grid were
  selected; point-source/hexadecapole/source-plane rows are excluded.
- Build extension: `/rogue1_8/nunota/lcbinint/build/lcbinint/_lcbinint.cpython-310-x86_64-linux-gnu.so`.

## Overall result

| profile | target | jobs | points | measured | lcbinint wins | VBM wins | unresolved | lcbinint win rate | median R | p10 | p90 | median Nbin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform | `0.001` | 283 | 1132 | 1132 | 0 | 1132 | 0 | 0.0% | 0.091 | 0.018 | 0.195 | 24 |
| uniform | `0.0001` | 128 | 512 | 512 | 0 | 512 | 0 | 0.0% | 0.038 | 0.006 | 0.102 | 50 |
| linear | `0.001` | 281 | 1124 | 1122 | 693 | 429 | 2 | 61.8% | 1.351 | 0.195 | 3.036 | 16 |
| linear | `0.0001` | 162 | 648 | 646 | 431 | 215 | 2 | 66.7% | 1.576 | 0.350 | 6.560 | 50 |

## By d/rho

| profile | target | region | measured | lcbinint wins | VBM wins | unresolved | win rate | median R |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| uniform | `0.001` | inner | 692 | 0 | 692 | 0 | 0.0% | 0.085 |
| uniform | `0.001` | tangent | 312 | 0 | 312 | 0 | 0.0% | 0.093 |
| uniform | `0.001` | outer-near | 128 | 0 | 128 | 0 | 0.0% | 0.114 |
| uniform | `0.0001` | inner | 348 | 0 | 348 | 0 | 0.0% | 0.039 |
| uniform | `0.0001` | tangent | 112 | 0 | 112 | 0 | 0.0% | 0.029 |
| uniform | `0.0001` | outer-near | 52 | 0 | 52 | 0 | 0.0% | 0.038 |
| linear | `0.001` | inner | 682 | 462 | 220 | 2 | 67.7% | 1.542 |
| linear | `0.001` | tangent | 312 | 157 | 155 | 0 | 50.3% | 1.021 |
| linear | `0.001` | outer-near | 128 | 74 | 54 | 0 | 57.8% | 1.120 |
| linear | `0.0001` | inner | 427 | 331 | 96 | 1 | 77.5% | 1.977 |
| linear | `0.0001` | tangent | 160 | 75 | 85 | 0 | 46.9% | 0.914 |
| linear | `0.0001` | outer-near | 59 | 25 | 34 | 1 | 42.4% | 0.918 |
