# v21: eliminate repeated fold-map trigonometry

The accepted change removes geometry-independent sine evaluations from the per-node radial coordinate map. Seed, lens/root solver, integration rule, error estimator, convergence gate and tolerances are unchanged. The existing experimental checkout is reused; shared master untouched.

## Algorithm and scope

A full normalized panel (`xl=-1,xr=1`) with folds at both endpoints uses the same dyadic Fejer abscissa x for each canonical slot. Its trigonometric factors depend only on x, independent of source, lens positions, mass ratio and radial interval endpoints. Cache255 slots of these factors once:

- double sin(pi/2 * min(t,1-t)) and sin(pi*t), t=(1+x)/2;
- the same extended-precision factors using the incumbent long-double constants/arithmetic.

At each node, retain the original multiplication order with the actual interval width and radius endpoints. Cache sine values only, not products involving geometry. Both the ordinary and explicit radial-offset arithmetic paths use their corresponding precision. H-subdivided panels whose normalized interval differs from[-1,1], one-sided folds, and other maps retain the original calculation.

The table is approximately12 KiB and initialized once per process, independently of lens geometry. It does not require successive epochs with identical lens configurations. First-use table construction is present; repeated median measurements describe steady evaluation after fixed numerical tables initialize, just as existing Fejer/FFT rule tables do.

`HOLO_ADAPTIVE_DISABLE_FOLD_MAP_CACHE` restores uncached calculations. `HOLO_ADAPTIVE_FOLD_MAP_AUDIT` evaluates the same node with both paths and aborts if radius, radial-offset remainder or Jacobian differs. This audit is disabled in accepted timing runs.

## Matched normal timing

Same87 selected `(id,u)` conditions,tolerance1e-4, live master seeds, new magnifiers/workspaces each repetition, one thread.5 repetitions per condition; arithmetic mean of per-condition medians. All compilation finished before serial accepted measurements. Combined time includes seeds, boundary preparation and integration, excludes caustic construction.

| Region | Before ms | After ms | Reduction |
|---|---:|---:|---:|
| Integration | 0.26331 | 0.24800 | 5.81% |
| Combined excluding caustic construction | 0.47641 | 0.46075 | 3.29% |

Reverse execution order: integration0.26297→0.24997 ms (4.94%); combined0.47628→0.46429 ms (2.52%). This is larger than the preceding interpolation pass's approximately1% whole-time improvement.

## Profile confirmation

Same87 conditions,3 instrumented repetitions, arithmetic means. Coordinate mapping0.02127→0.00597 ms, a71.9% reduction. Mean141.47 new radial nodes,128.37 eligible cached mappings (90.7%); all those used the extended-precision map because event remainders were nonzero. Physical point evaluation stayed0.20191→0.20134 ms. The node loop fell0.24566→0.22966 ms. These observations localize the gain to mapping rather than root work or changed adaptive points. Profile timers add overhead; use the uninstrumented comparison above for accepted runtime claims.

## Validation

-896 normal/fold/deeper conditions: radius/Jacobian audit agrees at every evaluated node; count-witness, interval-sign and nearest-anchor audits also enabled.
-896 conditions: magnifications and node counts identical to accepted v20.
-754 resolved D14 reference conditions: zero candidate failures; maximum discrepancy/tolerance0.249.
-142 unresolved reference conditions remain excluded from independent-reference acceptance. Before/after equality is not universal component completeness or independent physical verification for those cases.
-1608 adaptive plus23 frontier checks: zero failures; fold-map audit enabled.
-The existing43680 interpolation checks were rerun and passed; this pass does not change interpolation.

## Reproduce and artifacts

From `/rogue1_8/nunota/lcbinint/experiments/seeded-boundary-events`:

- Candidate build: `benchmarks/holonomic/build_seeded_radial_master.sh`.
- Tests and audit: `build_tests.sh`, `build_audit.sh` in this evidence directory.
- Profile builds: `build_profile_before.sh` (cache disabled), `build_profile_after.sh`.
- Finish all compilation before timing; use `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BINARY INPUT OUTPUT REPS 0`. Inputs `evidence/holonomic/caustic_seeded_connection_v4/{initial,fold,deeper}_minimal_seeds.tsv`; live master seeds regenerated. Normal5, holdout3, audit1, profile3 repetitions.
- Actual pre-pass binary: `build-seeded-events/bench_seeded_before_v21`; original adaptive header retained. Rebuilding the current source with the disable macro reconstructs the previous map computations.
- Raw matched/reverse-order, holdout, audit and profile TSV/log files retained. `summarize.py` produces summary, source snapshots and hash/checkout provenance.
- Holdout value/node comparisons use accepted v20 outputs, not historical timing for a holdout speedup claim.
- PROFILE adds columns44 radial_map_ms,45 map_calls,46 cached_calls,47 cached_extended_calls to the v20 layout.

Remaining larger integration costs are physical point/root/arc evaluation and sample allocation/storage. The repeated fold-map sine work has now been removed for the common full-panel case.
