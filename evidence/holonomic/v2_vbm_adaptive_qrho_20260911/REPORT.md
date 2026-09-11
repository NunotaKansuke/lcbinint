# Current V2 adaptive value-only vs existing VBM four-way comparison (HEAD `fd4b0da009f61230de7809b9595e4785098d68e9`)

The four speed conditions are VBM `RelTol=1e-3` and `RelTol=1e-4`, each for uniform (LD off) and linear (LD on, `c=0.5`). Only current V2 was evaluated in this step. Relative error is evaluated against the existing VBM `RelTol=1e-6` reference.

V2 timing covers the value-only adaptive `flux_adaptive_integrate(p,u,topo,cfg,workspace)` path. The radial node count is selected by the adaptive value contract `Eabs <= max(1e-16, RelTol * abs(mu))`; this is not a fixed-64 run. `classify_cells` (D14+topology), `LensParams`, input parsing, and output formatting are outside the timer, while adaptive setup, event handling, panel refinement, physics, estimator, and scheduler are inside. No gradient/Jacobian is requested or timed, and this remains the same pure-kernel boundary as the VBM timing rather than a full epoch. The existing VBM timing is the warmed direct `BinaryMag`/`BinaryMagDark` kernel.

Non-OK V2 points remain in the machine-readable join and status counts. The paper-style runtime map keeps finite positive V2 outputs. The p95 error map uses value-converged V2 values only; all-status finite-value error distributions remain in `summary.json`, and hatched cells identify incomplete value coverage.

The q-rho panels use q=geomspace(1e-4, 1, 13) and rho=geomspace(3e-5, 1, 13), giving 12 x 12 bins with a minimum population of 8 points per cell. This is the canonical filled-grid protocol from the runbook.

The error colorbar was expanded for small differences: cell p95 relative-error boundaries are 1e-8, 1e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, and 3e-4. The underlying values, reference, and p95 reducer are unchanged.

All V2 rows in the four benchmark conditions value-converged with `topology_status=OK`; no incomplete-value hatching is expected in the regenerated maps.

## Conditions

| profile | VBM timing target | rows | V2 value-converged | V2 p50 ms | V2 p95 ms | median VBM/V2 (finite positive) | median VBM/V2 (value-converged) | all-finite p95 error | converged p95 error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| linear | 0.0001 | 3608 | 3608 (100.0%) | 0.204459 | 0.547826 | 6.5937 | 6.5937 | 7.84223e-06 | 7.84223e-06 |
| linear | 0.001 | 3608 | 3608 (100.0%) | 0.168019 | 0.385226 | 1.98919 | 1.98919 | 9.34684e-06 | 9.34684e-06 |
| uniform | 0.0001 | 3608 | 3608 (100.0%) | 0.155332 | 0.362237 | 0.721356 | 0.721356 | 3.36594e-07 | 3.36594e-07 |
| uniform | 0.001 | 3608 | 3608 (100.0%) | 0.122137 | 0.260495 | 0.513567 | 0.513567 | 4.40694e-06 | 4.40694e-06 |

## Reproduction

The V2 runner was compiled at the current branch HEAD and run as:

```bash
/usr/bin/c++ -O3 -DNDEBUG -std=gnu++17 -I/rogue1_8/nunota/lcbinint/src -march=native -funroll-loops -ffp-contract=fast -fno-math-errno evidence/holonomic/v2_vbm_adaptive_qrho_20260911/v2_adaptive_value_runner.cpp -o /tmp/v2_adaptive_value_qrho_runner -lquadmath
/tmp/v2_adaptive_value_qrho_runner evidence/holonomic/v2_vbm_pure_kernel_20260911/input_snapshot.tsv /tmp/v2_adaptive_value_qrho_results.tsv 2> evidence/holonomic/v2_vbm_adaptive_qrho_20260911/v2_run.log
cp /tmp/v2_adaptive_value_qrho_results.tsv evidence/holonomic/v2_vbm_adaptive_qrho_20260911/v2_results.tsv
python3 evidence/holonomic/v2_vbm_adaptive_qrho_20260911/make_figures.py
```

## Figures

- `/rogue1_8/nunota/lcbinint/evidence/holonomic/v2_vbm_adaptive_qrho_20260911/figures/q_rho_uniform_1e-3_v2_adaptive_current_vbm1e-6.png`
- `/rogue1_8/nunota/lcbinint/evidence/holonomic/v2_vbm_adaptive_qrho_20260911/figures/q_rho_uniform_1e-3_v2_adaptive_current_vbm1e-6.pdf`
- `/rogue1_8/nunota/lcbinint/evidence/holonomic/v2_vbm_adaptive_qrho_20260911/figures/q_rho_uniform_1e-4_v2_adaptive_current_vbm1e-6.png`
- `/rogue1_8/nunota/lcbinint/evidence/holonomic/v2_vbm_adaptive_qrho_20260911/figures/q_rho_uniform_1e-4_v2_adaptive_current_vbm1e-6.pdf`
- `/rogue1_8/nunota/lcbinint/evidence/holonomic/v2_vbm_adaptive_qrho_20260911/figures/q_rho_linear_1e-3_v2_adaptive_current_vbm1e-6.png`
- `/rogue1_8/nunota/lcbinint/evidence/holonomic/v2_vbm_adaptive_qrho_20260911/figures/q_rho_linear_1e-3_v2_adaptive_current_vbm1e-6.pdf`
- `/rogue1_8/nunota/lcbinint/evidence/holonomic/v2_vbm_adaptive_qrho_20260911/figures/q_rho_linear_1e-4_v2_adaptive_current_vbm1e-6.png`
- `/rogue1_8/nunota/lcbinint/evidence/holonomic/v2_vbm_adaptive_qrho_20260911/figures/q_rho_linear_1e-4_v2_adaptive_current_vbm1e-6.pdf`
