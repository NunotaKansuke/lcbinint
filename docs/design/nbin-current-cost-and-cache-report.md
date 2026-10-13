# Current adaptive cost decomposition and mechanical optimization

## Result

The remaining overhead is partly removable without changing resolution selection,
acceptance, Bennett corrections or seeds. On the selected experimental Cartesian
adaptive route, replace generic double hashing with an inlined exact-coordinate
hash, transfer a single refined child's retained vectors instead of copying them,
and initialize the per-call lattice cache with 16384 rather than8192 slots.
These changes are now the adaptive context's defaults. The production finite-source
route and the opt-in status of local directional/row/strip/cap integration remain
unchanged. Diagnostic legacy flags retain the previous implementation.

Final matched measurements (`adaptive-cache-and-transfer-v3`), initial nbin16,
max512, absolute magnification tolerances, one thread, changing geometry on every
call including batch wrap, 21 rotated full batches per profile:

| tolerance | LD c | new / previous | time reduction | previous / fixed oracle | new / fixed oracle |
|---|---:|---:|---:|---:|---:|
|1e-3|0|0.972|2.8%|1.082|1.052|
|1e-3|0.5|0.967|3.3%|1.117|1.080|
|1e-3|1|0.957|4.3%|1.219|1.166|
|1e-4|0|0.871|12.9%|1.729|1.505|
|1e-4|0.5|0.895|10.5%|1.830|1.637|
|1e-4|1|0.901|9.9%|1.633|1.472|

The sum of the three profile medians changes:

- 1e-3: 0.121043s ->0.116903s, 3.42% shorter; fixed ratio1.134->1.095.
- 1e-4: 0.279112s ->0.247410s, 11.36% shorter; fixed ratio1.733->1.536.

These are sums of profile medians, not a median of a combined batch. The fixed
oracle is the reference-selected smallest passing global nbin among16..512,
with selection cost excluded. It is not the production automatic selector.
All full scalar timings include the normal seeds, integration, error/scheduling
and result construction; compilation and audits are excluded. Per-profile inputs
are matched, but the three brightness profile sets differ (27/25/22 cases).

## Complete refinement timing

The previous cost trace started after trying local vertical caps. It therefore
omitted successful caps and failed-cap work preceding an ordinary refinement.
The timer now starts before the cap attempt and records cap adoption as a trial.
Normal successful trials include lifting, filling, aggregation and footprint
checking. Diagnostic clocks are disabled when tracing is off.

`adaptive-current-cost-v2` measures the preceding selected route (legacy hash,
8192 slots, copied child vectors). For each actual adopted history, independent
truth is consulted after the run. The earliest stage for which that stage and
all later stages meet tolerance defines a retrospective stable passing stage.

| tolerance | LD c | refinement / full call | post-stable-pass refinement / full call |
|---|---:|---:|---:|
|1e-3|0|5.8%|1.4%|
|1e-3|0.5|6.8%|3.6%|
|1e-3|1|14.2%|4.9%|
|1e-4|0|65.4%|22.4%|
|1e-4|0.5|57.5%|9.8%|
|1e-4|1|54.5%|12.8%|

The first exploratory v1 trace gave24.5/9.8/13.4% at1e-4; the independently
repeated v2 measurements above are authoritative for the phase decomposition.
This is a retrospective opportunity indicator, not a guaranteed removable cost
or an operational stopping criterion. Earlier truth-passing stages may rely on
cancellation. No reference is consulted by the integration algorithm.

## Where the mechanical overhead lives

Native phase timers cover initial and recursive fills without double counting:

| 1e-4 profile | row exploration, brightness, Bennett and endpoint error | row aggregation and vertical error model | adaptive control, lifting, merging, footprint checks | other full-call work |
|---|---:|---:|---:|---:|
|uniform|63.7%|4.1%|2.9%|29.3%|
|LD c=.5|57.7%|3.5%|2.6%|36.1%|
|LD c=1|55.8%|3.0%|2.5%|38.7%|

Other work includes the original seeds, entry preparation, final result
construction and boundary between these timed regions. It is not all adaptive
bookkeeping. Endpoint error estimation is inside the first column; the second
column is not the total error-estimation cost. Tracing/full result conversion
adds0.1--0.5% in the v2 alternating batches.

A direct one-shot integration on the exact final mixed representation has not
been implemented: its retained interior integrals depend on earlier grids.
Thus these measurements do not identify an exact avoidable adaptation tax or
prove a speed limit. The hash/move ablations instead give a clean separation:
**every numerical result field and final representation stays identical**, so
their runtime differences are mechanical rather than extra resolution.

## Implemented changes and rejected alternatives

1. Cache keys remain the exact two floating coordinates. The inlined 64-bit
   hash canonicalizes signed zero and avalanches the combined bit patterns.
   Full-key equality still resolves collisions. No quantization or approximate
   equality is introduced, and entries remain isolated to a single call.
2. A single refined component transfers its five retained arrays and related
   state by ownership move. Multi-child merging retains the previous code and
   summation order. No coarse work is discarded or recomputed as a fallback.
3. Initial16384 slots modestly reduce resizing/allocation overhead. In this
   build a slot is32 bytes, so initial allocation is512KiB rather than256KiB;
   the existing growth/load-factor and map-budget policy remain unchanged.

The ablation matrix separates legacy hash, capacity alone, move alone, hash
alone and combinations. At1e-4 the hash accounts for most of the repeatable
benefit; move alone gives a smaller change. The first v1 matrix found the
selected combination about10% faster across all profiles; the final v3 matrix
above reproduces the improvement. Exact attribution beyond this is unwarranted:
allocator and hashing effects interact and wall timings vary.

Rejected: reducing initial capacity to256..4096 made calls slower; increasing
it universally to65536 gave about11--12% at1e-4 but worsened some1e-3 cases by
2--4%. We did not choose a tolerance- or geometry-specific capacity table.

Selected experimental integration still requires:

```
LCBININT_DIAGNOSTIC_DIRECTIONAL_ADAPTIVE=1
LCBININT_DIAGNOSTIC_LOCAL_ROW_INTERPOLATION=1
LCBININT_DIAGNOSTIC_LOCAL_BOUNDARY_STRIPS=1
LCBININT_DIAGNOSTIC_LOCAL_VERTICAL_CAPS=1
```

Diagnostic old behavior uses `LCBININT_DIAGNOSTIC_CACHE_CAPACITY=8192`,
`LCBININT_DIAGNOSTIC_LEGACY_CACHE_HASH=1` and
`LCBININT_DIAGNOSTIC_DISABLE_SINGLE_CHILD_MOVE=1`.

## Verification and scope

- Main148 + separate18 saved independent VBM/radial-LD references:166/166
  actual tolerance passes, unchanged estimated acceptance; no failure excluded.
- All six ablation modes and the selected defaults exactly match every returned
  numerical/result field, including magnification, error, adaptive history,
  maps, components, mixed grids and cap work. No acceptance policy changed.
- 60 matched budget comparisons include partial work and exceptions. Separate
  partial-cap budget regression confirms last adopted positive flux, retained
  work and repeat-call isolation.
- 148 square adaptive and547 fixed values/indicators exactly reproduce saved
  results;148 directional histories validated.
- Two native unit suites pass. Cache unit checks120000 entries through repeated
  growth, hits/misses, both hashes and initial capacities, signed zeros,
  neighboring representable values and extremely small/large finite keys.
- General mathematical certification remains unfinished: `certified=false`,
  `complete_error_model=false`.

## Remaining priority

Control-only micro-optimization cannot remove the remaining ~1.54 fixed-oracle
ratio. Continue with horizontal boundary/error allocation and the extra
resolution chosen by the estimator, especially LD. Stable reference-passing
histories expose cases worth examining, but their stages are not safe stopping
rules. Keep the shared uniform/LD framework and cheap estimator; do not replace
it with reference-driven stopping, global Richardson runs or heavy certificates.

Artifacts: `tests/diagnostics/results/recal2026/adaptive-current-cost-v1/`,
`adaptive-current-cost-v2/`, `adaptive-cache-and-transfer-v1/` (factor ablations),
`adaptive-cache-and-transfer-v2/` (unit/build logs and preserved wrapper failure)
and `adaptive-cache-and-transfer-v3/` (final measurements, accuracy and regressions).
The v2 wrapper attempted to index a fixed-route float as a dictionary; corrected
in v3, with the failed log retained. It was a harness error, not a numerical failure.

Reproduce into a new result directory, one thread:

```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LCBININT_PROBE_BUILD=$PWD/build-boundary-support /home/nunota/.miniconda3/envs/myenv/bin/python -m tests.diagnostics.recal2026.adaptive_cache_transfer_audit --output UNIQUE_RESULT_NAME
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LCBININT_PROBE_BUILD=$PWD/build-boundary-support /home/nunota/.miniconda3/envs/myenv/bin/python -m tests.diagnostics.recal2026.current_adaptive_cost --output UNIQUE_COST_NAME
```
