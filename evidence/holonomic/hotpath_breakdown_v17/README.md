# v17: current v16 integration cost diagnosis

No numerical algorithm or tolerance changed. Added profile scopes compiled only with `HOLO_FRONTIER_COST_PROFILE`, and appended columns to the existing profile benchmark. Source snapshots and hashes are in `source/` and `manifest.json`.

## Measurement contract

Existing experimental checkout, current master seed provider. Same 87 selected `(id,u)` conditions at tolerance 1e-4, 3 repetitions, fresh workspaces and magnifier instances, single thread, compilation completed before each measurement. These are arithmetic means across all 261 instrumented repetitions, not medians. Timing calls add overhead: detailed integration 0.3541 ms vs uninstrumented v16 approximately0.299 ms. Use the profile to locate costs, not to claim exact production wall-time fractions or speedups.

The previous uninstrumented v16 matched result remains: seed0.10052 ms, boundary preparation0.11080 ms, integration0.29915 ms, combined excluding caustic construction0.51361 ms. Do not sum nested timer fields. Dynamic-count timer measures the Sturm count, not coefficient generation.

## Disjoint major integration components (instrumented)

| Region | Mean ms |
|---|---:|
| Actual radial-point evaluation | 0.22343 |
| Node-loop remainder (mapping, nearest anchor, storage, validation and local topology comparisons, plus timer overhead) | 0.05090 |
| Integration setup | 0.03702 |
| Sample-vector growth/copy | 0.02244 |
| Error estimator | 0.01677 |
| Scheduler | 0.00138 |
| Remaining controller overhead | 0.00218 |
| Total integration | 0.35411 |

The actual point evaluation contains:

| Nested/disjoint region | Mean ms |
|---|---:|
| Angular root/arc construction | 0.07676 |
| Of which root-pair solver (nested) | 0.04507 |
| Whole per-arc body, including polish and angular integration | 0.05866 |
| Sturm count | 0.02819 |
| Sample initialization/common header checks (sum over recursive calls) | 0.02274 |
| Node quartic coefficient construction | 0.00327 |
| Final sample assembly | 0.00490 |
| Point evaluator remainder | 0.02890 |

Within per-arc body: endpoint polish0.01859 ms; K-rule0.00908 ms; rescue angular integration0.01441 ms; representation/geometry/other arc overhead remains. These smaller times are nested in the body above.

## Findings and prioritized targets

1. **Point/controller overhead is material.** The node-loop remainder alone is approximately0.051 ms. Header/common setup costs about0.023 ms, and another0.029 ms remains in the point evaluator. Allocation adds0.022 ms. These regions need finer targeted measurements before assigning all their time to copying. Timers themselves account for part of the apparent remainder.
2. **There is a concrete redundant call structure.** For frontier cells with `n_crossings<0`, `mapped_radius` initializes common sample state, certifies the node's count, and recursively invokes the same function with a local classified cell. Common setup is repeated. Flattening this dispatch can preserve the same certificate, endpoint gates and error ledger. The sample object is1040 bytes, with derivative and experimental angular state even on a value-only run; size is evidence of storage scope, not proof every byte is copied on every path. Reworking value-only state is a second candidate, after dispatch.
3. **The LD fallback is measurable, not dominant overall.** In the 40 selected LD conditions (120 repetitions), the 64+32-point angular rescue averages16.525 calls and0.03134 ms per evaluation. K attempts162.9, successes141.425, disagreement rejections21.475. A call may try both charts before one rescue, so rejected attempts and rescue calls are not one-to-one. There are zero logged K-level vfloor/tmax/S2/B rejections here. Arc-pair rejection before entering K is not logged by those K counters; no conclusion about its frequency follows. Never relax the disagreement gate from this count alone. Investigate chart/deflation accuracy and small-arc handling with the same acceptance tolerance.
4. **Setup still spends0.03354 ms on event location/error processing.** Existing same-epoch memoization works, but initial precision work remains. This is smaller than the combined point overhead; do not introduce another broad certification layer to optimize it.
5. Error estimation0.01677 ms and scheduling0.00138 ms are lower-priority targets.

Uniform and LD subsets have different selected input counts (47 and40), so their whole-runtime means are not a controlled LD speed comparison.

## Validation and reproduction

The final instrumented192-condition output has zero changed magnifications and zero changed node counts relative to accepted v16. Reference benchmark reports zero failures on resolved D14 cases, with14 unresolved reference conditions excluded. This run diagnoses time distribution; it does not add a new universal accuracy or component coverage guarantee.

From the checkout root:

```
bash evidence/holonomic/hotpath_breakdown_v17/build_profile.sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 build-seeded-events/bench_seeded_profile_v17 evidence/holonomic/caustic_seeded_connection_v4/initial_minimal_seeds.tsv evidence/holonomic/hotpath_breakdown_v17/final.tsv 3 0 > evidence/holonomic/hotpath_breakdown_v17/final_stdout.log 2> evidence/holonomic/hotpath_breakdown_v17/final.log
python evidence/holonomic/hotpath_breakdown_v17/summarize.py
```

`profile.log/tsv` are the first coarse profile; `detail.log/tsv` split arc-body/tail/node-loop; `final.log/tsv` add header/coefficient/rescue measurements. All accepted timing runs started after compilation completed. Final PROFILE fields0–22 retain the earlier layout; new fields23 arc_body_ms,24 mapped_tail_ms,25 node_loop_ms,26 mapped_header_ms,27 node_coeff_ms,28 rescue_ms,29 rescue_calls,30 K_attempts,31 K_success,32 vfloor_reject,33 tmax_reject,34 S2_reject,35 B_reject,36 disagreement_reject.
