# Seeded q-rho benchmark rerun and six-panel figures

Rerun date: 2026-10-14; host rogue1; branch `experiment/seeded-boundary-events`; source commit `82d7f236c80f913ac87fb5399cf7f1733743a425`. This is the complete existing benchmark, run with the current LD product-quadrature source. The previous 2026-10-13 artifacts remain untouched.

The run uses the same 7,216 geometry/profile inputs, saved VBM 5.5 RelTol=1e-6 magnifications and timing table, requested targets 1e-3/1e-4, three repetitions, and one thread. It produced all 14,432 unique geometry/profile/target rows. The figure generator joined all rows to the saved VBM table without key or metadata mismatch.

## Run result

- Status OK: 14,248 / 14,432 (98.72%).
- Seed support unproven: 180; topology unresolved: 4.
- Value convergence: 14,428; coverage certified: 0. The solver does not claim certified radial support.
- Relative difference from the saved VBM value exceeds the requested 1e-4 in 3 rows; no accepted row exceeds 1e-3. These are the high-magnification reference discrepancies diagnosed in `../accuracy_three_v44/README.md`; the maps label the saved VBM reference explicitly.
- Against the prior full run, all 14,422 rows converged in both runs stayed within their own requested value tolerance. Exact row changes, status counts, and error summaries are in `post_run_audit.json`.

## Figures

The established generator produced four 2x3 figures, each with the runtime maps and p95 relative-difference maps for q-rho, A_fs-rho, and A_fs-q. I inspected all four PNGs at full output resolution; axes, labels, color bars, and panel spacing are readable and unclipped. PDF and PNG outputs are retained.

## Reproduction

From this checkout:

    bash evidence/holonomic/seeded_paper_qrho_20261014/build.sh
    bash evidence/holonomic/seeded_paper_qrho_20261014/run_full.sh
    bash evidence/holonomic/seeded_paper_qrho_20261014/rebuild_figures.sh

The build emits `build-seeded-events/bench_seeded_paper_qrho_v45`. The run uses one thread and writes a progress log. The figure script is the established 2026-10-13 generator with only its output-directory reproduction path updated. `source_sha256.txt` and `SHA256SUMS` capture the executable, current solver, inputs, reference table, scripts, result tables, audit, report, and figures.
