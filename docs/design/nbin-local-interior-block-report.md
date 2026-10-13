# Local interior grouping pilot

## Result and scope

Implemented an opt-in path in the experimental `binary_adaptive_flux` kernel. It preserves the existing row lattice, Bennett endpoint corrections, vertical row weights, component selection, and incremental map cache. This is **interior grouping within a row**, not yet a general mixed-resolution boundary-panel integrator. Whole-component refinement still chooses the vertical resolution.

On the existing independent-reference matrix (74 geometry/profile combinations, two absolute tolerances, 148 evaluations), every result met the estimated target and the actual target. A separate three-geometry, nine-profile matrix also passed both targets (18 evaluations). Neither is a population-wide accuracy guarantee. `certified=false` and `complete_error_model=false` remain unchanged.

Final matched timing, five alternating batches per profile and tolerance, one thread, scalar call including initial seeds and all integration/refinement work:

| Absolute target | Profile | Cases | Lattice-map reduction | Total time, new / baseline |
|---|---|---:|---:|---:|
| 1e-3 | uniform | 27 | 0% | 1.0015 |
| 1e-3 | linear LD c=0.5 | 25 | -0.033% | 1.0004 |
| 1e-3 | linear LD c=1 | 22 | -0.017% | 1.0022 |
| 1e-4 | uniform | 27 | 39.55% | 0.8587 |
| 1e-4 | linear LD c=0.5 | 25 | 9.87% | 1.0524 |
| 1e-4 | linear LD c=1 | 22 | 4.22% | 1.0239 |

Uniform scalar results are exactly identical to baseline. LD changes at 1e-4 were at most 1.77e-6 (c=0.5) and 2.40e-6 (c=1). Maximum actual errors were 9.85e-5 and 9.78e-5, respectively. All 148 calls passed; maximum error across 1e-3 calls was 8.71e-4. These absolute errors matter more than estimator ratios.

On the small separate matrix, pure-LD 1e-4 timing was 1.0756 times baseline. Consequently the LD speed cost is not resolved, and this path is **not enabled by default**.

## Implementation

`cartesian_run_fill.hpp` now accepts an optional interior-block callback and block size. A successful callback supplies the discrete row contribution and an error remainder in physical flux units. The fill advances across that interval, retaining the last two actual endpoint states required by Bennett. The remainder joins the existing row remainder and is weighted with the same outer weights. A rejected interval continues with the existing per-cell scan. Evaluations performed during rejection remain in the existing adaptive cache.

Uniform blocks contain 32 cells. A mean-value enclosure of the binary lens map and its derivative verifies that the complete image-plane segment maps inside the source disk. The contribution is then exactly the number of cells; no interior brightness evaluations are needed. No sampled-endpoint inference replaces this check.

Linear LD blocks contain 16 cells. Five shared samples at offsets 0, 4, 8, 12, 15 give the discrete sums of cubic and quartic Lagrange interpolants. The cubic sum is the retained contribution; the absolute difference supplies an additional positive error indicator. This is a local smooth-error estimator, **not a rigorous interpolation-error bound**. A disk-interior interval check also prevents skipping an unresolved outside gap. The check is performed only after the inexpensive moment test passes. All sample evaluations are cached; a rejected block uses the existing sum without restarting. Profiles with nonzero second LD coefficient keep per-cell summation.

Grouping starts only at nbin >=64. This avoids attempt overhead on the initial coarse stages; it is an execution-policy cutoff, not a change to tolerance acceptance. The local attempt allowance is `target_raw_flux / rho^2 * block_cells / 64`. It only determines whether to try grouping; the resulting residual is still added to final acceptance, with no guarantee flag changes.

## Rejected approaches and evidence

Artifacts preserve the earlier variants:

- v1: linear interpolation with enclosed second derivative; no LD blocks accepted, about 10% slower at 1e-4.
- v2: interval Taylor jets through fourth derivative; no LD blocks accepted, about 16–18% slower.
- v3: 32-cell moment estimator and direct interval enclosure; few LD blocks accepted, about 7% slower. A small unit compilation overlapped this pilot, so its timing is exploratory.
- v4: tighter mean-value enclosure; uniform map reduction improved, LD remained about 18% slower.
- v5: 16-cell LD blocks reduced maps, but still about 12–14% slower.
- v6: cheap-first rejection and avoiding coarse attempts reduced the LD penalty to approximately 2–5% on the main matrix.

The directly observed problem is that saved map evaluations do not cover the grouping, moment-sample/cache, and interval-test costs for LD. Most LD work still uses the ordinary discrete sum. This pilot does not establish a universal lower limit on LD speed, or measure each overhead independently.

## Next work

Keep this opt-in prototype and the unchanged default. The next design should retain **local row panels and their moments across refinement**, so it identifies error-bearing intervals from already available work and adds new samples only to those intervals. Merely trying blocks while rescanning a whole component pays the rejection cost repeatedly. Uniform boundary geometry should remain shared with LD; local vertical panels and coherent Bennett endpoint/outer weights still need implementation. Do not remove the inside check by assuming sampled endpoints imply a complete inside interval.

## Artifacts and reproduction

- `tests/diagnostics/recal2026/local_interior_block_audit.py`: matched accuracy/timing using saved VBM uniform and independent radial-LD references.
- `tests/diagnostics/recal2026/test_local_interior_blocks.cpp`: exact rectangle run extents/areas and propagation of block residuals.
- `tests/diagnostics/results/recal2026/local-interior-block-v6/`: final main matrix, timing samples, build/unit/ctest logs and source provenance.
- `tests/diagnostics/results/recal2026/local-interior-block-fresh-v1/`: separate reference matrix.
- The prior variants and rejected results remain in v1–v5 directories.

Build with project-local TMPDIR. Then run with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LCBININT_PROBE_BUILD=$PWD/build-boundary-support`:

```sh
/home/nunota/.miniconda3/envs/myenv/bin/python -m tests.diagnostics.recal2026.local_interior_block_audit --output UNIQUE_OUTPUT
```

The script alternates `LCBININT_DIAGNOSTIC_INTERIOR_BLOCKS=1` against the unchanged adaptive baseline inside the same loaded native module. Baseline is the current adaptive method, not fixed oracle selection or the empirical automatic selector. Neither commit nor production promotion is part of this pilot.
