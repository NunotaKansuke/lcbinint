# Seeded radial frontier vs saved VBMicrolensing benchmark (HEAD `82d7f236c80f913ac87fb5399cf7f1733743a425`)

The full 7,216-row q-rho corpus was evaluated for VBM speed targets `RelTol=1e-3` and `1e-4`, with uniform and linear limb darkening. The reference values are the saved VBM `RelTol=1e-6` values. Each condition has 3 repetitions.

The tested path is the current v29 seed provider plus `epoch_seeded_radial_frontier`. Every repetition creates fresh point-source and finite-source objects, builds the 1400-bin caustic branches before timing, then times the center-image roots and finite-source seed construction. The timed seed step scans the prepared caustic branches and solves any additional seed roots. It then times the seeded radial discovery and integration. D14 and `classify_cells` are not called. The reported full time is the median of paired `seed_ms + radial_ms`; caustic construction, object construction, input/output, and figure generation are excluded.

The existing VBM timing is the saved warmed direct `BinaryMag2` (uniform) / `BinaryMagDark` (linear) kernel timing. Fresh VBM 5.5 calls on the low-magnification tolerance sample reproduce the saved medians within 1%; the original timing-table generator was not retained. Thus this comparison adds seed construction to the current route and compares it with the direct VBM call.

The seed provider support flag and the frontier `coverage_certified` flag are retained. The frontier reports `coverage_certified=false`. `status=OK` means seed-provider support plus value convergence; it is not a radial-support coverage certificate. The original run had 20 internally OK rows above 1e-3. Investigation found that endpoint polishing could accept a different fold outside the count-changing radius bracket and truncate an image band. The bracket-local fallback and post-polish band checks are now in place; the corrected full rerun has zero internally OK rows above 1e-3. See `OUTLIER_DIAGNOSIS.md` for the trace and limits of this empirical check.

The four output figures each have six panels: three median speed-ratio maps and three p95 relative-error maps over q-rho, finite-source magnification-rho, and finite-source magnification-q. Bins use the established q-rho coverage edges and require at least eight points. Non-OK rows stay in raw and joined tables; finite positive outputs remain visible in runtime panels, while error panels use rows with seed support and value convergence.

## Conditions

| profile | target | rows | OK | full p50 ms | seed p50 ms | radial p50 ms | median row-wise R | accepted p95 error | accepted max error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| linear | 0.0001 | 3608 | 3561 (98.7%) | 0.117381 | 0.0086025 | 0.0937345 | 11.2121 | 7.92198e-06 | 0.000217776 |
| linear | 0.001 | 3608 | 3561 (98.7%) | 0.11277 | 0.0086025 | 0.0848995 | 2.8985 | 1.05588e-05 | 0.000283123 |
| uniform | 0.0001 | 3608 | 3563 (98.8%) | 0.105599 | 0.008485 | 0.0811275 | 1.25081 | 2.94249e-07 | 0.000270597 |
| uniform | 0.001 | 3608 | 3563 (98.8%) | 0.103632 | 0.008485 | 0.0751425 | 0.745263 | 2.84185e-06 | 0.000365176 |

## Worst internally accepted reference discrepancies

Across internally OK rows, the maximum pointwise relative error is 0.000365176; 0 rows exceed 1e-3 across 0 distinct case/epoch geometries. No internally OK row exceeds 1e-3 relative error against the saved VBM 1e-6 reference. The radial-support coverage flag remains false.

| case | epoch | profile | target | q | rho | seeded mu | VBM 1e-6 mu | relative error |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 155 | 23 | uniform | 0.001 | 0.000179347 | 0.0525218 | 10.620569 | 10.616692 | 0.000365176 |
| 10 | 7 | uniform | 0.001 | 0.000172132 | 8.11333e-05 | 12363.396 | 12366.75 | 0.000271199 |
| 65 | 15 | uniform | 0.001 | 0.0201171 | 0.950986 | 2.2798517 | 2.2793383 | 0.000225247 |
| 115 | 15 | linear | 0.0001 | 0.000760658 | 0.000422429 | 1580.8618 | 1581.2061 | 0.000217776 |
| 109 | 23 | uniform | 0.001 | 0.289268 | 0.946499 | 1.5121557 | 1.5118269 | 0.000217482 |
| 195 | 23 | uniform | 0.001 | 0.00549261 | 0.0163441 | 46.988599 | 46.979318 | 0.00019757 |
| 38 | 0 | linear | 0.001 | 0.0341314 | 0.000497307 | 1253.0227 | 1252.778 | 0.000195351 |
| 10 | 15 | linear | 0.001 | 0.000172132 | 8.11333e-05 | 10446.828 | 10448.769 | 0.000185744 |
| 38 | 7 | linear | 0.001 | 0.0341314 | 0.000497307 | 1241.2203 | 1241.0025 | 0.000175498 |
| 90 | 7 | uniform | 0.001 | 0.00245297 | 0.518362 | 1.8973564 | 1.8976857 | 0.000173532 |

## Artifacts

- Raw results: `/rogue1_8/nunota/lcbinint/experiments/seeded-boundary-events/evidence/holonomic/seeded_paper_qrho_20261014/raw_results.tsv`
- Joined results: `/rogue1_8/nunota/lcbinint/experiments/seeded-boundary-events/evidence/holonomic/seeded_paper_qrho_20261014/joined_v2_vbm.tsv`
- Internally OK rows with relative error >1e-3: `/rogue1_8/nunota/lcbinint/experiments/seeded-boundary-events/evidence/holonomic/seeded_paper_qrho_20261014/accepted_error_outliers.tsv`
- Diagnosis and pre-fix rows: `/rogue1_8/nunota/lcbinint/experiments/seeded-boundary-events/evidence/holonomic/seeded_paper_qrho_20261014/OUTLIER_DIAGNOSIS.md` and `/rogue1_8/nunota/lcbinint/experiments/seeded-boundary-events/evidence/holonomic/seeded_paper_qrho_20261014/accepted_error_outliers_pre_endpoint_fix.tsv`
- Six-panel figures: `/rogue1_8/nunota/lcbinint/experiments/seeded-boundary-events/evidence/holonomic/seeded_paper_qrho_20261014/figures`

## Reproduction

From this checkout, run `bash benchmarks/holonomic/build_seeded_paper_qrho.sh`, then `bash evidence/holonomic/seeded_paper_qrho_20261014/run_full.sh`, then `python3 evidence/holonomic/seeded_paper_qrho_20261014/make_figures.py`. The full-run script fixes the single-thread environment; the plot script keeps its Matplotlib cache inside this evidence directory.

