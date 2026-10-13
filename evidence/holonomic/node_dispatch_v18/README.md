# v18: eliminate duplicate radial-node dispatch and sample initialization

Changes remain in the existing `experiment/seeded-boundary-events` checkout. Shared master checkout and master seed provider unchanged. This pass addresses the redundant per-node entry work identified in v17; no tolerance, root certificate, endpoint gate, error ledger, or seed algorithm was relaxed.

## Changes

- An ordinary frontier node previously certified its quartic count, then recursively called `mapped_radius` with a classified local cell. The node now retains that same certified count and coefficient array and continues directly through the same root/arc/endpoint/integration path. The observed count is still recorded on both success and rejection.
- For value-only evaluation without active atlas hooks or angular adaptive/upgrade modes, the default-constructed sample remains pristine. Two redundant assignments of a fresh sample are skipped. Gradient, atlas, and angular-mode resets remain.
- Cell-dependent atlas and angular-upgrade entry hooks retain recursive dispatch. The optimization is deliberately scoped; unknown-count and degenerate failures retain their existing handling.
- Sample layout remains1040 bytes. A separate value-only storage redesign was not required for this measured win.

Rollback/reproduction macros: `HOLO_ADAPTIVE_LEGACY_NODE_DISPATCH` and `HOLO_ADAPTIVE_LEGACY_SAMPLE_INIT`. Defining both restores the previous dispatch and initialization paths. `adaptive_epoch_before.hpp` preserves the actual pre-pass header; saved baseline binary is `build-seeded-events/bench_seeded_before_v18`.

## Matched normal timing

Same87 selected `(id,u)` conditions, tolerance1e-4, live master seeds, fresh magnifiers/workspaces every repetition, one thread. Five repetitions per condition; mean of per-condition medians. Serial runs after all compilation completed. The total includes center/root preparation, master seeds, boundary planning, integration and small outer overhead; caustic construction is excluded.

| Stage | Before ms | After ms | Reduction |
|---|---:|---:|---:|
| Integration | 0.30244 | 0.27572 | 8.83% |
| Total excluding caustic construction | 0.51772 | 0.48797 | 5.75% |

The dispatch-only intermediate trial measured integration0.28351 ms,total0.49762 ms. The complete candidate is faster than that intermediate trial.

Reverse execution order: integration0.30356→0.27738 ms (8.62%); total0.52129→0.49199 ms (5.62%). Planning and seed medians fluctuate slightly, despite no change to their implementation. Per-stage medians need not add exactly to whole medians.

## Profile confirmation

Three instrumented repetitions of the same87 conditions, arithmetic means. Common sample/header time fell0.02274→0.00457 ms. Actual radial-point evaluation fell0.22417→0.19993 ms. Sample-vector allocation remained0.02158→0.02103 ms, showing that this pass removed node entry work rather than allocation.

Full instrumented integration0.35543→0.32868 ms. These values include profile timer overhead and must not replace uninstrumented acceptance timing. Future targets still include coordinate mapping/anchor/storage work and value-only state footprint. No claim that all the measured remainder is removable.

## Precision checks

-896 normal/fold/deeper conditions: maximum relative magnification change0 and unchanged node counts versus pre-pass v16.
-754 conditions with a resolved D14 reference: zero candidate failures; every discrepancy below the configured tolerance (maximum discrepancy/tolerance0.249).
-142 unresolved D14 reference conditions excluded from independent-reference acceptance. Before/after equality does not independently certify their physical accuracy or universal component completeness.
-1608 adaptive checks plus23 frontier checks, zero failures; final tests compiled with interval-sign and nearest-anchor audits.

## Reproduction

Run from `/rogue1_8/nunota/lcbinint/experiments/seeded-boundary-events`:

1. Build candidate with `benchmarks/holonomic/build_seeded_radial_master.sh` and tests with `evidence/holonomic/node_dispatch_v18/build_tests.sh`.
2. Build profile with `evidence/holonomic/node_dispatch_v18/build_profile.sh`.
3. Finish all compilation before accepted timing. Use `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BINARY INPUT OUTPUT REPS 0`. Input files: `evidence/holonomic/caustic_seeded_connection_v4/{initial,fold,deeper}_minimal_seeds.tsv`; the benchmark regenerates master seeds, not stored minimal seeds. Normal reps5, holdout3, profile3.
4. `before.tsv`, `flat.tsv`, `after.tsv`, `after_reverse.tsv`, `before_reverse.tsv` capture normal trials. `fold.tsv`, `deeper.tsv` capture final holdouts. Validation uses the current normal baseline plus the numerically identical accepted v16 fold/deeper outputs; those older files are used for values/node-count checks, not matched holdout timing claims.
5. `profile_before/after.log` and corresponding TSV contain the scoped timing comparison, with v17 profile layout.
6. `python evidence/holonomic/node_dispatch_v18/summarize.py` rebuilds summary/provenance and source snapshots. `manifest.json` records source hashes and checkout identity.

No commit, push, new worktree, or shared-master edit performed.
