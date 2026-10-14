# v19: constant-time comparison of observed image counts

Current experimental checkout and master seed provider retained. No tolerance, convergence gate, quadrature weights, or summation arithmetic changed.

## Change and equivalence

With `reuse_samples=true`, existing slots are retained and new sample indices only append within one refinement. Scan the same populated slots once at refinement entry, recording the minimum and maximum normalized known count (`max(0,local_crossings)`, excluding unknown-1 and unreliable samples). A new known count disagrees with a previous count iff at least one previous count exists and either extrema differs from it. If no disagreement is found, update the extrema with that new sample. This implements the same boolean predicate at the same evaluation point; it does not certify any unobserved radial interval or assume uniform topology.

The summary is local to a refinement call, so resumed work and changed panels are scanned again. Without sample reuse, slots can be overwritten: retain the original full scan in that mode. An early topology yield returns at the same point. Duplicate/repeated count comparisons change from O(N^2) to an initial O(N) scan plus O(1) per inserted point.

`HOLO_ADAPTIVE_COUNT_WITNESS_AUDIT` recomputes the original full scan at every comparison and aborts on a different predicate. `HOLO_ADAPTIVE_LEGACY_COUNT_SCAN` selects the original comparison (the preparatory witness scan remains unless optimized away). The actual saved pre-pass binary is `build-seeded-events/bench_seeded_before_v19`, with pre-pass header in `adaptive_radial_before.hpp`.

## Matched timing

Same87 normal `(id,u)` conditions,tolerance1e-4,5 repetitions, per-condition median then mean, one thread, fresh workspaces and magnifiers, live master seeds. All compilation completed before serial accepted timing.

| Region | Before ms | After ms | Reduction |
|---|---:|---:|---:|
| Integration | 0.27613 | 0.26508 | 4.00% |
| Seed/preparation/integration excluding caustic construction | 0.48864 | 0.47995 | 1.78% |

Reverse execution order: integration0.27878→0.26584 ms (4.64%); combined0.49407→0.48147 ms (2.55%). There are timing fluctuations in unaffected preparation stages; do not assign all whole-time differences to the comparison loop.

## Validation

-896 normal/fold/deeper conditions: magnifications and node counts identical to pre-pass values.
-896 full benchmark audit conditions: no mismatch between old scan and summarized predicate; interval-sign and nearest-anchor audits also enabled.
-1608 adaptive checks plus23 frontier checks: zero failures.
-754 resolved D14 reference conditions: zero candidate failures; discrepancy/tolerance at most0.249.
-142 unresolved D14 reference conditions remain excluded from independent-reference acceptance. Equality before/after is not proof of universal topology completeness or their physical accuracy.

## Reproduce

From `/rogue1_8/nunota/lcbinint/experiments/seeded-boundary-events`, build with `benchmarks/holonomic/build_seeded_radial_master.sh`, `build_tests.sh`, and `build_audit.sh` in this evidence directory. Wait until all compilation completes. Run `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BINARY INPUT OUTPUT REPS 0` with inputs from `evidence/holonomic/caustic_seeded_connection_v4/{initial,fold,deeper}_minimal_seeds.tsv`. Normal reps5, holdouts3, audits1. The benchmark ignores stored seeds and generates master seeds.

Raw before/after and reverse-order TSV files, holdout TSV files, audit TSV files and test logs are retained. `summarize.py` creates `summary.json`, source snapshots and the hash/provenance manifest. Holdout value/node comparisons reuse the numerically identical accepted v18 fold/deeper outputs; no contemporaneous holdout speedup is claimed from those earlier files.

## Sums: next investigation

See `summation_notes.md`. Finite-source magnification is a convolution at fixed lens geometry, but a map FFT requires creating that magnification map. The local quadrature value is already an O(N) dot product. The error estimator's O(N^2) coarse-to-fine interpolation can instead be expressed by sine transforms on the uniform node angles. That is an exact-arithmetic reformulation; floating-point error and small-panel overhead require separate measurement/validation. No transform or summation change is enabled in v19.
