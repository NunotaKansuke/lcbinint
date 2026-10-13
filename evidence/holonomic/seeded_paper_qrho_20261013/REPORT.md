# Seeded radial frontier vs saved VBMicrolensing benchmark (HEAD `f3547723b79828d9c501182746028e99f7bad116`)

The full 7,216-row q-rho corpus was evaluated for VBM speed targets `RelTol=1e-3` and `1e-4`, with uniform and linear limb darkening. The reference values are the saved VBM `RelTol=1e-6` values. Each condition has 3 repetitions.

The tested path is the current v29 seed provider plus `epoch_seeded_radial_frontier`. Every repetition creates fresh point-source and finite-source objects, builds the 1400-bin caustic branches before timing, then times the center-image roots and finite-source seed construction. The timed seed step scans the prepared caustic branches and solves any additional seed roots. It then times the seeded radial discovery and integration. D14 and `classify_cells` are not called. The reported full time is the median of paired `seed_ms + radial_ms`; caustic construction, object construction, input/output, and figure generation are excluded.

The existing VBM timing is the saved warmed direct `BinaryMag2` (uniform) / `BinaryMagDark` (linear) kernel timing. Fresh VBM 5.5 calls on the low-magnification tolerance sample reproduce the saved medians within 1%; the original timing-table generator was not retained. Thus this comparison adds seed construction to the current route and compares it with the direct VBM call.

The seed provider support flag and the frontier `coverage_certified` flag are retained. The frontier reports `coverage_certified=false`. `status=OK` means seed-provider support plus value convergence; it is not a radial-support coverage certificate. The original run had 20 internally OK rows above 1e-3. Investigation found that endpoint polishing could accept a different fold outside the count-changing radius bracket and truncate an image band. The bracket-local fallback and post-polish band checks are now in place; the corrected full rerun has zero internally OK rows above 1e-3. See `OUTLIER_DIAGNOSIS.md` for the trace and limits of this empirical check.

The four output figures each have six panels: three median speed-ratio maps and three p95 relative-error maps over q-rho, finite-source magnification-rho, and finite-source magnification-q. Bins use the established q-rho coverage edges and require at least eight points. Non-OK rows stay in raw and joined tables; finite positive outputs remain visible in runtime panels, while error panels use rows with seed support and value convergence.

## Conditions

| profile | target | rows | OK | full p50 ms | seed p50 ms | radial p50 ms | median row-wise R | accepted p95 error | accepted max error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| linear | 0.0001 | 3608 | 3559 (98.6%) | 0.223042 | 0.041077 | 0.15627 | 5.75959 | 7.97301e-06 | 0.000217776 |
| linear | 0.001 | 3608 | 3559 (98.6%) | 0.199103 | 0.041077 | 0.132443 | 1.57448 | 1.18461e-05 | 0.000530261 |
| uniform | 0.0001 | 3608 | 3562 (98.7%) | 0.188389 | 0.041015 | 0.118231 | 0.628506 | 3.30964e-07 | 0.000270597 |
| uniform | 0.001 | 3608 | 3562 (98.7%) | 0.172675 | 0.041015 | 0.106318 | 0.402405 | 6.03681e-06 | 0.000560693 |

## Worst internally accepted reference discrepancies

Across internally OK rows, the maximum pointwise relative error is 0.000560693; 0 rows exceed 1e-3 across 0 distinct case/epoch geometries. No internally OK row exceeds 1e-3 relative error against the saved VBM 1e-6 reference. The radial-support coverage flag remains false.

| case | epoch | profile | target | q | rho | seeded mu | VBM 1e-6 mu | relative error |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 36 | 23 | uniform | 0.001 | 0.21262 | 0.30491 | 1.0738048 | 1.073203 | 0.000560693 |
| 145 | 7 | uniform | 0.001 | 0.0019944 | 0.0120618 | 1.0861074 | 1.0866961 | 0.000541716 |
| 203 | 15 | linear | 0.001 | 0.000109819 | 0.35409 | 1.8606435 | 1.8599437 | 0.000376282 |
| 155 | 23 | uniform | 0.001 | 0.000179347 | 0.0525218 | 10.620569 | 10.616692 | 0.000365176 |
| 201 | 7 | linear | 0.001 | 0.000950123 | 0.10833 | 5.0330672 | 5.0347136 | 0.000327026 |
| 33 | 7 | uniform | 0.001 | 0.000228545 | 0.055347 | 9.9285254 | 9.925688 | 0.000285873 |
| 10 | 7 | uniform | 0.001 | 0.000172132 | 8.11333e-05 | 12363.396 | 12366.75 | 0.000271199 |
| 87 | 15 | linear | 0.001 | 0.035732 | 0.132241 | 1.3951698 | 1.3948232 | 0.000248478 |
| 87 | 7 | uniform | 0.001 | 0.035732 | 0.132241 | 1.3924065 | 1.3920808 | 0.000233916 |
| 145 | 23 | linear | 0.001 | 0.0019944 | 0.0120618 | 1.1400538 | 1.1397909 | 0.000230663 |

## Artifacts

- Raw results: `/rogue1_8/nunota/lcbinint/experiments/seeded-boundary-events/evidence/holonomic/seeded_paper_qrho_20261013/raw_results.tsv`
- Joined results: `/rogue1_8/nunota/lcbinint/experiments/seeded-boundary-events/evidence/holonomic/seeded_paper_qrho_20261013/joined_v2_vbm.tsv`
- Internally OK rows with relative error >1e-3: `/rogue1_8/nunota/lcbinint/experiments/seeded-boundary-events/evidence/holonomic/seeded_paper_qrho_20261013/accepted_error_outliers.tsv`
- Diagnosis and pre-fix rows: `/rogue1_8/nunota/lcbinint/experiments/seeded-boundary-events/evidence/holonomic/seeded_paper_qrho_20261013/OUTLIER_DIAGNOSIS.md` and `/rogue1_8/nunota/lcbinint/experiments/seeded-boundary-events/evidence/holonomic/seeded_paper_qrho_20261013/accepted_error_outliers_pre_endpoint_fix.tsv`
- Six-panel figures: `/rogue1_8/nunota/lcbinint/experiments/seeded-boundary-events/evidence/holonomic/seeded_paper_qrho_20261013/figures`

## Reproduction

From this checkout, run `bash benchmarks/holonomic/build_seeded_paper_qrho.sh`, then `bash evidence/holonomic/seeded_paper_qrho_20261013/run_full.sh`, then `python3 evidence/holonomic/seeded_paper_qrho_20261013/make_figures.py`. The full-run script fixes the single-thread environment; the plot script keeps its Matplotlib cache inside this evidence directory.
