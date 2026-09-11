# Current V2 adaptive value-only vs existing VBM four-way comparison (HEAD `6a8ec865fec675c2d37230eba5d701e8c138c3f1`)

The four speed conditions are VBM `RelTol=1e-3` and `RelTol=1e-4`, each for uniform (LD off) and linear (LD on, `c=0.5`). Only current V2 was evaluated in this step. Relative error is evaluated against the existing VBM `RelTol=1e-6` reference.

V2 timing covers the value-only adaptive `flux_adaptive_integrate(p,u,topo,cfg,workspace)` path. The radial node count is selected by the adaptive value contract `Eabs <= max(1e-16, RelTol * abs(mu))`; this is not a fixed-64 run. `classify_cells` (D14+topology), `LensParams`, input parsing, and output formatting are outside the timer, while adaptive setup, event handling, panel refinement, physics, estimator, and scheduler are inside. No gradient/Jacobian is requested or timed, and this remains the same pure-kernel boundary as the VBM timing rather than a full epoch. The existing VBM timing is the warmed direct `BinaryMag`/`BinaryMagDark` kernel.

Non-OK V2 points remain in the machine-readable join and status counts. The paper-style runtime map keeps finite positive V2 outputs. The p95 error map uses value-converged V2 values only; all-status finite-value error distributions remain in `summary.json`, and hatched cells identify incomplete value coverage.

The 636 non-converged rows have `topology_status=OK`, zero uncertain topology cells, and positive node/panel counts. They therefore passed the topology classifier and stopped later when an adaptive sample evaluator returned `reliable=false`; the current evaluator reports those internal subreasons through the aggregate `TopologyUnresolved` stop label.

The failure breakdown is now explicit. 624/636 rows (98.1%) belong to a geometry whose restored cell plan contains at least one `kFull` cell. In `adaptive_epoch.hpp`, `mapped_radius()` has no full-circle evaluator because it expects endpoint arcs and returns `reliable=false` for `kFull`/`kDegenerate`; the fixed V2 `radius_terms()` path does have `full_circle_terms()`. Thus these rows are an adaptive representation coverage gap, not failure of the nested value estimator. The remaining 12 rows are two tiny-source/near-axis geometries (`case_id=9, d_bin=2, epoch=7` and `case_id=92, d_bin=0`) at the stricter target or both targets; the current evaluator does not expose which of its arc reliability guards fired. The topology and arc diagnostic raw files are retained beside this report.

## Conditions

| profile | VBM timing target | rows | V2 value-converged | V2 p50 ms | V2 p95 ms | median VBM/V2 (finite positive) | median VBM/V2 (value-converged) | all-finite p95 error | converged p95 error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| linear | 0.0001 | 3608 | 3447 (95.5%) | 0.19169 | 0.502671 | 7.04916 | 6.58425 | 1.71596e-05 | 7.82458e-06 |
| linear | 0.001 | 3608 | 3451 (95.6%) | 0.155517 | 0.359159 | 2.07628 | 1.97227 | 1.98223e-05 | 9.21579e-06 |
| uniform | 0.0001 | 3608 | 3447 (95.5%) | 0.147985 | 0.357806 | 0.752549 | 0.736649 | 1.26045e-06 | 3.11226e-07 |
| uniform | 0.001 | 3608 | 3451 (95.6%) | 0.116584 | 0.260118 | 0.534911 | 0.52499 | 1.39167e-05 | 3.32113e-06 |

## Reproduction

The V2 runner was compiled at the current branch HEAD and run as:

```bash
/usr/bin/c++ -O3 -DNDEBUG -std=gnu++17 -I/rogue1_8/nunota/lcbinint/src -march=native -funroll-loops -ffp-contract=fast -fno-math-errno evidence/holonomic/v2_vbm_adaptive_phase92_20260911/v2_adaptive_value_runner.cpp -o /tmp/v2_adaptive_value_phase92_runner -lquadmath
/tmp/v2_adaptive_value_phase92_runner evidence/holonomic/v2_vbm_adaptive_phase92_20260911/input_snapshot.tsv /tmp/v2_adaptive_value_phase92_results.tsv 2> evidence/holonomic/v2_vbm_adaptive_phase92_20260911/v2_run.log
cp /tmp/v2_adaptive_value_phase92_results.tsv evidence/holonomic/v2_vbm_adaptive_phase92_20260911/v2_results.tsv
python3 evidence/holonomic/v2_vbm_adaptive_phase92_20260911/make_figures.py
```

## Figures

- `/rogue1_8/nunota/lcbinint/evidence/holonomic/v2_vbm_adaptive_phase92_20260911/figures/q_rho_uniform_1e-3_v2_adaptive_current_vbm1e-6.png`
- `/rogue1_8/nunota/lcbinint/evidence/holonomic/v2_vbm_adaptive_phase92_20260911/figures/q_rho_uniform_1e-3_v2_adaptive_current_vbm1e-6.pdf`
- `/rogue1_8/nunota/lcbinint/evidence/holonomic/v2_vbm_adaptive_phase92_20260911/figures/q_rho_uniform_1e-4_v2_adaptive_current_vbm1e-6.png`
- `/rogue1_8/nunota/lcbinint/evidence/holonomic/v2_vbm_adaptive_phase92_20260911/figures/q_rho_uniform_1e-4_v2_adaptive_current_vbm1e-6.pdf`
- `/rogue1_8/nunota/lcbinint/evidence/holonomic/v2_vbm_adaptive_phase92_20260911/figures/q_rho_linear_1e-3_v2_adaptive_current_vbm1e-6.png`
- `/rogue1_8/nunota/lcbinint/evidence/holonomic/v2_vbm_adaptive_phase92_20260911/figures/q_rho_linear_1e-3_v2_adaptive_current_vbm1e-6.pdf`
- `/rogue1_8/nunota/lcbinint/evidence/holonomic/v2_vbm_adaptive_phase92_20260911/figures/q_rho_linear_1e-4_v2_adaptive_current_vbm1e-6.png`
- `/rogue1_8/nunota/lcbinint/evidence/holonomic/v2_vbm_adaptive_phase92_20260911/figures/q_rho_linear_1e-4_v2_adaptive_current_vbm1e-6.pdf`
