# Current V2 pure-kernel vs existing VBM four-way comparison (HEAD `8a9f701217003e49ec95ec13ab97b15b7fd95886`)

The four speed conditions are VBM `RelTol=1e-3` and `RelTol=1e-4`, each for uniform (LD off) and linear (LD on, `c=0.5`). Only current V2 was evaluated in this step. Relative error is evaluated against the existing VBM `RelTol=1e-6` reference.

V2 timing covers the value-only `flux_value_integrate(64,u,pf,topo)` path; no gradient/Jacobian is requested or timed. `LensParams`, `PrimaryFrame`, `classify_cells` (D14+topology), `epoch_value_blend`, input parsing, and output formatting are outside the timer; no `epoch_value` or full epoch is timed. The existing VBM timing is the warmed direct `BinaryMag`/`BinaryMagDark` kernel.

Non-OK V2 points remain in the machine-readable join and status counts. The paper-style runtime map keeps finite positive V2 outputs, while the p95 error map keeps all finite V2 outputs; hatched cells identify incomplete status coverage.

## Conditions

| profile | VBM timing target | rows | V2 OK | V2 p50 ms | V2 p95 ms | median VBM/V2 (finite positive) | median VBM/V2 (OK) | all p95 error | OK p95 error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| linear | 0.0001 | 3608 | 3366 (93.3%) | 0.371947 | 1.39466 | 3.64365 | 3.80158 | 0.000203094 | 0.000210659 |
| linear | 0.001 | 3608 | 3366 (93.3%) | 0.371947 | 1.39466 | 0.895148 | 0.929863 | 0.000203094 | 0.000210659 |
| uniform | 0.0001 | 3608 | 3366 (93.3%) | 0.264211 | 0.727066 | 0.482318 | 0.500895 | 0.000210843 | 0.000218694 |
| uniform | 0.001 | 3608 | 3366 (93.3%) | 0.264211 | 0.727066 | 0.276988 | 0.287334 | 0.000210843 | 0.000218694 |

## Reproduction

The V2 runner was compiled at the current branch HEAD and run as:

```bash
/usr/bin/c++ -O3 -DNDEBUG -std=gnu++17 -I/rogue1_8/nunota/lcbinint/src -march=native -funroll-loops -ffp-contract=fast -fno-math-errno evidence/holonomic/v2_vbm_pure_kernel_phase92_20260911/v2_pure_kernel_runner.cpp -o /tmp/v2_pure_kernel_phase92_runner -lquadmath
/tmp/v2_pure_kernel_phase92_runner evidence/holonomic/v2_vbm_pure_kernel_phase92_20260911/input_snapshot.tsv /tmp/v2_pure_kernel_phase92_results.tsv
cp /tmp/v2_pure_kernel_phase92_results.tsv evidence/holonomic/v2_vbm_pure_kernel_phase92_20260911/v2_results.tsv
python3 evidence/holonomic/v2_vbm_pure_kernel_phase92_20260911/make_figures.py
```

## Figures

- `/rogue1_8/nunota/lcbinint/evidence/holonomic/v2_vbm_pure_kernel_phase92_20260911/figures/q_rho_uniform_1e-3_v2_current_vbm1e-6.png`
- `/rogue1_8/nunota/lcbinint/evidence/holonomic/v2_vbm_pure_kernel_phase92_20260911/figures/q_rho_uniform_1e-3_v2_current_vbm1e-6.pdf`
- `/rogue1_8/nunota/lcbinint/evidence/holonomic/v2_vbm_pure_kernel_phase92_20260911/figures/q_rho_uniform_1e-4_v2_current_vbm1e-6.png`
- `/rogue1_8/nunota/lcbinint/evidence/holonomic/v2_vbm_pure_kernel_phase92_20260911/figures/q_rho_uniform_1e-4_v2_current_vbm1e-6.pdf`
- `/rogue1_8/nunota/lcbinint/evidence/holonomic/v2_vbm_pure_kernel_phase92_20260911/figures/q_rho_linear_1e-3_v2_current_vbm1e-6.png`
- `/rogue1_8/nunota/lcbinint/evidence/holonomic/v2_vbm_pure_kernel_phase92_20260911/figures/q_rho_linear_1e-3_v2_current_vbm1e-6.pdf`
- `/rogue1_8/nunota/lcbinint/evidence/holonomic/v2_vbm_pure_kernel_phase92_20260911/figures/q_rho_linear_1e-4_v2_current_vbm1e-6.png`
- `/rogue1_8/nunota/lcbinint/evidence/holonomic/v2_vbm_pure_kernel_phase92_20260911/figures/q_rho_linear_1e-4_v2_current_vbm1e-6.pdf`
