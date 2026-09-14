# Current V2 pure-kernel vs existing VBM four-way comparison

The four speed conditions are VBM `RelTol=1e-3` and `RelTol=1e-4`, each for uniform (LD off) and linear (LD on, `c=0.5`). Only current V2 was evaluated in this step. Relative error is evaluated against the existing VBM `RelTol=1e-6` reference.

V2 timing covers `flux_value_integrate(64,u,pf,topo)` only. `LensParams`, `PrimaryFrame`, `classify_cells` (D14+topology), `epoch_value_blend`, input parsing, and output formatting are outside the timer; no `epoch_value` or full epoch is timed. The existing VBM timing is the warmed direct `BinaryMag`/`BinaryMagDark` kernel.

Non-OK V2 points remain in the machine-readable join and status counts. The paper-style runtime map keeps finite positive V2 outputs, while the p95 error map keeps all finite V2 outputs; hatched cells identify incomplete status coverage.

## Conditions

| profile | VBM timing target | rows | V2 OK | V2 p50 ms | V2 p95 ms | median VBM/V2 (finite positive) | median VBM/V2 (OK) | all p95 error | OK p95 error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| linear | 0.0001 | 3608 | 3366 (93.3%) | 0.378413 | 1.39084 | 3.56614 | 3.75566 | 0.000203094 | 0.000210659 |
| linear | 0.001 | 3608 | 3366 (93.3%) | 0.378413 | 1.39084 | 0.880061 | 0.917975 | 0.000203094 | 0.000210659 |
| uniform | 0.0001 | 3608 | 3366 (93.3%) | 0.269058 | 0.729896 | 0.472575 | 0.490555 | 0.000210843 | 0.000218694 |
| uniform | 0.001 | 3608 | 3366 (93.3%) | 0.269058 | 0.729896 | 0.271952 | 0.280544 | 0.000210843 | 0.000218694 |

## Reproduction

The V2 runner was compiled at the current branch HEAD and run as:

```bash
/usr/bin/c++ -O3 -DNDEBUG -std=gnu++17 -I/rogue1_8/nunota/lcbinint/src -O3 -march=native -funroll-loops -ffp-contract=fast -fno-math-errno /tmp/v2_pure_kernel_runner.cpp -o /tmp/v2_pure_kernel_runner -lquadmath
/tmp/v2_pure_kernel_runner /tmp/v2_current_input_unique.tsv /tmp/v2_pure_kernel_results.tsv
python3 /tmp/make_v2_vbm_pure_fair.py
```

## Figures

- `/tmp/lcbinint_v2_vbm_pure_fair_20260911/figures/q_rho_uniform_1e-3_v2_current_vbm1e-6.png`
- `/tmp/lcbinint_v2_vbm_pure_fair_20260911/figures/q_rho_uniform_1e-3_v2_current_vbm1e-6.pdf`
- `/tmp/lcbinint_v2_vbm_pure_fair_20260911/figures/q_rho_uniform_1e-4_v2_current_vbm1e-6.png`
- `/tmp/lcbinint_v2_vbm_pure_fair_20260911/figures/q_rho_uniform_1e-4_v2_current_vbm1e-6.pdf`
- `/tmp/lcbinint_v2_vbm_pure_fair_20260911/figures/q_rho_linear_1e-3_v2_current_vbm1e-6.png`
- `/tmp/lcbinint_v2_vbm_pure_fair_20260911/figures/q_rho_linear_1e-3_v2_current_vbm1e-6.pdf`
- `/tmp/lcbinint_v2_vbm_pure_fair_20260911/figures/q_rho_linear_1e-4_v2_current_vbm1e-6.png`
- `/tmp/lcbinint_v2_vbm_pure_fair_20260911/figures/q_rho_linear_1e-4_v2_current_vbm1e-6.pdf`
