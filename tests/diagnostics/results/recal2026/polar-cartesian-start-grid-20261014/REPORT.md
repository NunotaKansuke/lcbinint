# Cartesian start-grid / polar comparison — 2026-10-14

60 A-stratified inputs: 30 prior pilot inputs and 30 held-out inputs from lens cases absent from the pilot. Each profile has 30 inputs (15 pilot, 15 holdout). Holdout selection seed: 20261014. All finite and failed rows are retained. This is a geometry stress cohort, not an iid population sample.

One pinned CPU, three measured native calls per input/path, median. Centre roots and caustic/seed warmup excluded. Each Cartesian request gets a new adaptive lattice; targets 1e-3/1e-4, max bins512, map budget1,000,000. Fixed polar calls have no requested error target: their accuracy is assessed independently against saved VBM RelTol1e-6 magnifications. Their converged flag must not be treated as acceptance at either target. No numerical-convergence ladder is used to accept a runtime result.

## Main comparison

Default polar grid_ratio=4. Production after the two verified changes is used for polar. Speed columns include all finite inputs, even accuracy failures; they are not an accuracy-matched headline benchmark.

|method|profile|target|initial/fixed bins|actual pass|false accepts|median ms|
|---|---|---|---:|---:|---:|---:|
|cart8|uniform|0.001|8|28/30|2|1.6147|
|cart8|uniform|0.0001|8|24/30|4|8.9708|
|cart8|linear|0.001|8|28/30|1|1.6144|
|cart8|linear|0.0001|8|27/30|2|3.3428|
|cart16|uniform|0.001|16|27/30|2|1.8190|
|cart16|uniform|0.0001|16|28/30|1|6.7323|
|cart16|linear|0.001|16|28/30|1|3.5670|
|cart16|linear|0.0001|16|28/30|2|5.0204|
|polar-after|uniform|0.001|8|11/30|n/a|0.7753|
|polar-after|uniform|0.001|16|15/30|n/a|1.3188|
|polar-after|uniform|0.001|32|26/30|n/a|2.6059|
|polar-after|uniform|0.001|64|29/30|n/a|5.7782|
|polar-after|uniform|0.0001|8|1/30|n/a|0.7753|
|polar-after|uniform|0.0001|16|2/30|n/a|1.3188|
|polar-after|uniform|0.0001|32|8/30|n/a|2.6059|
|polar-after|uniform|0.0001|64|24/30|n/a|5.7782|
|polar-after|linear|0.001|8|14/30|n/a|0.8018|
|polar-after|linear|0.001|16|21/30|n/a|1.1769|
|polar-after|linear|0.001|32|29/30|n/a|2.3977|
|polar-after|linear|0.001|64|30/30|n/a|6.0605|
|polar-after|linear|0.0001|8|1/30|n/a|0.8018|
|polar-after|linear|0.0001|16|6/30|n/a|1.1769|
|polar-after|linear|0.0001|32|21/30|n/a|2.3977|
|polar-after|linear|0.0001|64|28/30|n/a|6.0605|

## High-A matched accuracy

A>=100 gives twelve inputs per profile. These ratios include only inputs where both Cartesian initial16 and the indicated fixed polar resolution meet the target. Excluded counts remain explicit; the subset cannot establish safe automatic routing.

|profile|target|polar bins|joint pass / inputs|median Cartesian16 / polar time|
|---|---|---:|---:|---:|
|uniform|0.001|8|4/12|2.647|
|uniform|0.001|16|5/12|2.514|
|uniform|0.001|32|11/12|2.431|
|uniform|0.001|64|11/12|0.996|
|uniform|0.0001|8|0/12|n/a|
|uniform|0.0001|16|1/12|2.465|
|uniform|0.0001|32|2/12|6.592|
|uniform|0.0001|64|9/12|1.042|
|linear|0.001|8|5/12|5.035|
|linear|0.001|16|8/12|4.166|
|linear|0.001|32|10/12|2.049|
|linear|0.001|64|11/12|1.050|
|linear|0.0001|8|1/12|4.834|
|linear|0.0001|16|2/12|3.421|
|linear|0.0001|32|8/12|2.063|
|linear|0.0001|64|11/12|1.051|

## Implemented changes and checks

1. Store angular integrals in the already sparse visited pages; reconstruct polar components only on occupied angular columns. Preserve ascending-phi union and accumulation order and the last-to-first cyclic adjacency. Sparse-only outputs matched before exactly for 180 normal calls and 180 forced-outer calls. Sixteen extra axis/wrap/junction inputs, using the complex-segment diagnostic switch, matched for all 48 calls. Cartesian outputs and stop/map/refinement fields matched before/after for all 120 target rows. Sparse-only paired median speedup is 1.013x: a small measured effect, not the main solution.
2. For linear LD, use the same exact sqrt profile as Cartesian instead of nearest-table quantization. Uniform results are unchanged. LD values change by at most about 2.46e-6 relative in the 8/16/32 comparison; measured runtime ratio is about 1.00. This eliminates a separate profile approximation but does not solve the dominant spatial-grid errors.

The ordinary production build was rebuilt. Existing Cartesian initial16 routing/defaults and polar grid_ratio4 remain. No automatic mixed route or new acceptance guarantee was promoted.

## Experiments that were not promoted

Angular ratio2/1 improves many accuracy failures, but increases work. Removing global component-count/seed-count gates from the polar outer rule while preserving clean-arc guards was tested separately in component-outer.cpp. It costs substantially more and gives little consistent accuracy benefit. The production guards remain. Guarded intermediate outputs are preserved with .guarded.tsv suffixes and are excluded from the reported unguarded experiment.

Initial Cartesian8 has paired median speedups 1.78/2.13x at 1e-3 (Uniform/LD), and 1.01/1.56x at 1e-4. Its false accepts and newly failing geometries prevent unconditional promotion. Polar32 is about 2x faster than Cartesian16 on accuracy-matched high-A subsets, but comparing to Cartesian8 removes much of this advantage. Polar64 is often needed for 1e-4 and gives little remaining speed advantage on the matched high-A subsets.

## Connecting the methods

A hard magnification threshold and blending two fluxes do not resolve discretization error. A practical mixed route needs the same relative-flux acceptance contract for both coordinates. Initial method/grid proposals can use already-solved image positions, lens-map Jacobian directions, source size, and the existing caustic geometry without extra root solves. Polar then needs an angular error indicator as well as its current radial indicator; existing fixed explicit-tolerance polar checks use a coarse/fine rerun and are unsuitable for the requested retained-work adaptive route. Implement adaptation around retained radial runs and angular panels, subdividing only the indicated coordinate, before enabling automatic switching. Within a chosen coordinate system all retained rays/runs can be reused; Cartesian and polar lattice values cannot generally be reused point-for-point because their nodes differ. Choose coordinates before filling to avoid paying for two full initial fills. These are design requirements, not an implemented or validated routing policy.

## Reproduction

selected.json, input.tsv, runner.cpp, the preserved native executables, provenance.json and sparse-build.json record selection/build/timing inputs. finite_source_magnifier.cpp.before is the baseline; .sparse and .analytic preserve subsequent stages. component-outer.cpp is the rejected experiment. Per-path TSV and logs remain alongside summary.json. Run python tests/diagnostics/results/recal2026/polar-cartesian-start-grid-20261014/analyze.py to regenerate this report. Compilation and numerical kernels were run sequentially, with no overlapping measurement workers. VBM timings were not used to select or accept a current result.
