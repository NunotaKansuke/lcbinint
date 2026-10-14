# Seeded radial frontier vs saved VBMicrolensing benchmark (HEAD `f3547723b79828d9c501182746028e99f7bad116`)

The full 7,216-row q-rho corpus was evaluated for VBM speed targets `RelTol=1e-3` and `1e-4`, with uniform and linear limb darkening. The reference values are the saved VBM `RelTol=1e-6` values. Each condition has 3 repetitions.

The tested path is the current v29 seed provider plus `epoch_seeded_radial_frontier`. Every repetition creates fresh point-source and finite-source objects, builds the 1400-bin caustic branches before timing, then times the center-image roots and finite-source seed construction. The timed seed step scans the prepared caustic branches and solves any additional seed roots. It then times the seeded radial discovery and integration. D14 and `classify_cells` are not called. The reported full time is the median of paired `seed_ms + radial_ms`; caustic construction, object construction, input/output, and figure generation are excluded.

The existing VBM timing is the saved warmed direct `BinaryMag`/`BinaryMagDark` kernel timing. Thus this comparison adds seed construction to the current route and compares it with the original direct VBM call.

The seed provider support flag and the frontier `coverage_certified` flag are retained. The frontier reports `coverage_certified=false`. `status=OK` means seed-provider support plus value convergence; it is not a radial-support coverage certificate. Some internally OK rows have large discrepancies against VBM 1e-6, so the p95 maps must be read with the worst-case audit below.

The four output figures each have six panels: three median speed-ratio maps and three p95 relative-error maps over q-rho, finite-source magnification-rho, and finite-source magnification-q. Bins use the established q-rho coverage edges and require at least eight points. Non-OK rows stay in raw and joined tables; finite positive outputs remain visible in runtime panels, while error panels use rows with seed support and value convergence.

## Conditions

| profile | target | rows | OK | full p50 ms | seed p50 ms | radial p50 ms | median row-wise R | accepted p95 error | accepted max error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| linear | 0.0001 | 3608 | 3549 (98.4%) | 0.221144 | 0.040816 | 0.1555 | 5.78808 | 7.99924e-06 | 0.998124 |
| linear | 0.001 | 3608 | 3549 (98.4%) | 0.197313 | 0.040816 | 0.132206 | 1.58503 | 1.20451e-05 | 0.998124 |
| uniform | 0.0001 | 3608 | 3552 (98.4%) | 0.187912 | 0.0407845 | 0.118962 | 0.627258 | 3.3732e-07 | 0.998122 |
| uniform | 0.001 | 3608 | 3552 (98.4%) | 0.171859 | 0.0407845 | 0.107568 | 0.40166 | 6.4562e-06 | 0.998122 |

## Worst internally accepted reference discrepancies

Across internally OK rows, the maximum pointwise relative error is 0.998124; 20 rows exceed 1e-3 across 5 distinct case/epoch geometries. The internal OK flag did not catch these discrepancies. Full rows are in `accepted_error_outliers.tsv`.

| case | epoch | profile | target | q | rho | seeded mu | VBM 1e-6 mu | relative error |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 98 | 23 | linear | 0.0001 | 0.000251651 | 0.000303917 | 7.6775213 | 4091.4357 | 0.998124 |
| 95 | 23 | uniform | 0.001 | 0.0674464 | 0.000136848 | 67.880389 | 1717.7378 | 0.960483 |
| 26 | 15 | linear | 0.001 | 0.58013 | 0.000698504 | 1.4519809 | 33.253488 | 0.956336 |
| 77 | 23 | uniform | 0.0001 | 0.0755711 | 0.00245511 | 32.367488 | 71.289373 | 0.54597 |
| 35 | 15 | uniform | 0.001 | 0.0321436 | 0.00211197 | 45.158991 | 98.796337 | 0.542908 |
| 36 | 23 | uniform | 0.001 | 0.21262 | 0.30491 | 1.0738048 | 1.073203 | 0.000560693 |
| 145 | 7 | uniform | 0.001 | 0.0019944 | 0.0120618 | 1.0861074 | 1.0866961 | 0.000541716 |
| 203 | 15 | linear | 0.001 | 0.000109819 | 0.35409 | 1.8606435 | 1.8599437 | 0.000376282 |
| 155 | 23 | uniform | 0.001 | 0.000179347 | 0.0525218 | 10.620569 | 10.616692 | 0.000365176 |
| 201 | 7 | linear | 0.001 | 0.000950123 | 0.10833 | 5.0330672 | 5.0347136 | 0.000327026 |

## Artifacts

- Raw results: `/rogue1_8/nunota/lcbinint/experiments/seeded-boundary-events/evidence/holonomic/seeded_paper_qrho_20261013/raw_results.tsv`
- Joined results: `/rogue1_8/nunota/lcbinint/experiments/seeded-boundary-events/evidence/holonomic/seeded_paper_qrho_20261013/joined_v2_vbm.tsv`
- Internally OK rows with relative error >1e-3: `/rogue1_8/nunota/lcbinint/experiments/seeded-boundary-events/evidence/holonomic/seeded_paper_qrho_20261013/accepted_error_outliers.tsv`
- Six-panel figures: `/rogue1_8/nunota/lcbinint/experiments/seeded-boundary-events/evidence/holonomic/seeded_paper_qrho_20261013/figures`

## Reproduction

From this checkout, run `bash benchmarks/holonomic/build_seeded_paper_qrho.sh`, then `bash evidence/holonomic/seeded_paper_qrho_20261013/run_full.sh`, then `python3 evidence/holonomic/seeded_paper_qrho_20261013/make_figures.py`. The full-run script fixes the single-thread environment; the plot script keeps its Matplotlib cache inside this evidence directory.

