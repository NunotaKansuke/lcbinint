# Cheap local LD saddle error model

## Result

Implemented a local `t log|t|` model for isolated Cartesian run junctions in the experimental overlay. No new lens maps, image roots, global caustic construction or runtime resolution ladder. Scalar magnification is unchanged. Production master remains unchanged; overlay is opt-in. The model is approximate, not a certified universal error bound.

Original82 fixed conditions:

| quantity | before | after |
|---|---:|---:|
| maximum indicator/actual error |2.431413|1.381703|
| >=1.1 |15|5|
| >=1.5 |3|0|
| >=2 |2|0|
| minimum ratio |.993650|.993650|

Across175 tested coefficient/resolution/input combinations (not175 independent geometries): >=1.1 falls26→13, >=1.5 falls11→3, >=2 falls8→1. Maximum3.874914→2.020360. Extra27 fixed resolutions include the remaining maximum atbins96. Nine unused geometry inputs plus two predeclared perturbed saddle geometries add66 combinations.

One new geometry reveals3.867% underestimation (s=1.0468104907,c=1,bins128). It is bit-identical with the saddle model disabled, so it is a pre-existing limitation newly discovered by broader validation. No new deficit exceeding the agreed1.1% threshold was introduced in these175 combinations. This does not establish a universal safety guarantee.

## Mathematical structure

Near a saddle of the source-depth function, `D(x,t)=x²-t`, integrating linear-LD brightness sqrt(max(D,0)) over a row produces a regular background plus `t log|t|`. The previous toy derivation and independent SciPy quadrature establish this basis; it is not an empirical correction factor.

At the worst actual lens junctions, eight retained rows were inspected. An offline phase scan fitted a regular quadratic plus tlog on six rows and predicted two held-out rows. At junction y=-15, polynomial held defect7.16e-8 vs tlog1.38e-9; at y17,5.75e-8 vs6.44e-10 (raw image flux units). The scan is diagnostic only and is not in the runtime path.

Runtime contact phase is obtained geometrically: the square of the retained gap width is locally quadratic. The existing three-row quadratic endpoint model estimates its root, and the fourth gap row provides the existing geometry diagnostic. No source-plane target magnification/reference enters this calculation.

The runtime eight-row LD flux is separated from the uniform-edge term: `F_LD = row_flux-(1-c)*width`. Six rows fit a regular quadratic plus tlog via a4x4 system; the nearest two contact rows are withheld. Max disagreement over all eight retained rows is kept as a local approximate remainder. The fitted singular quadrature defect is added with its sign. Its explained second differences are subtracted only from the four affected junction rows. The existing uniform-width singular model remains in place.

The whole-lattice tlog midpoint defect is computed from its analytic primitive over33 terms, with three smooth Euler–Maclaurin tail corrections. This is not a new numerical lens integral. A standalone C++ unit checks it against a1025-term long-double calculation at seven phases, synthetic quadratic+tlog fits for three amplitudes, pure regular limits and invalid phases.

Models are attempted only for isolated1↔2-run transitions with a complete retained eight-row footprint and four valid positive gap widths, and a resolved contact strictly between rows. Other regions retain the existing estimator. Nearby transitions, unresolved geometric phase or unresolved linear system retain the original local calculation. No global fallback restarts previous work.

The pointwise disagreement and geometric phase are numerical models, not interval enclosures. Further basis terms and contact-phase uncertainty can matter at coarse resolution; this is a remaining limitation, not a proven truncation bound.

## Generalization checks

- Original82:20 indicators change, magnifications unchanged; min unchanged.
- Additional27 bins96/192/384 combinations:6 indicators change; max3.875→2.020; min.993090.
- Nine previously unused saved input geometries:54 combinations. Model does not activate on these inputs; these check fallback behavior and reveal the pre-existing deficit. They do not prove generalization of the active saddle fit.
- Two predeclared perturbed saddle configurations (s,q,source position,radius changed):12 combinations,8 LD indicators improve. Examples3.263→1.103 and3.587→1.334. No deficit>1.1%. Independent uniform VBM areas and SciPy LD radial references use two settings; no quadrature warnings. Max reference setting difference4.08e-11.

There is no per-geometry fitted safety multiplier or selection based on reference error ratios. Local polynomial coefficients are computed from retained rows of each evaluation, as part of the numerical approximation.

## Cost

Same binary on/off switch (`LCBININT_DIAGNOSTIC_DISABLE_SADDLE`) with nine alternating paired batches, five changing pilot geometries, one thread; warmups excluded.

| LD c | bins128 on/off | bins512 on/off |
|---|---:|---:|
|.5|1.000802|1.000618|
|1|1.000126|1.000353|

Observed differences0.01–0.08% are within timing variation; no material slowdown was detected in this workload. Uniform controls vary0.25–0.36%, reinforcing that small differences are noise-scale. This is indicator-entry total time, not an isolated cost per successful saddle fit or adaptive-nbin end-to-end speed gain.

Unit core/boundary support pass2/2; standalone saddle checks pass. All175 candidate scalar magnifications match ordinary/disabled-model results, and no extra estimator maps are introduced. Experimental status remains certified=false, complete_error_model=false.

## Artifacts

`tests/diagnostics/results/recal2026/ld-junction-prototype-v1/` contains trace/signature, before/after source, hashes, analytic unit, build/unit logs, combined validation, new-geometry and perturbation ablations, predeclared perturbation inputs.

Data: `shared-flux-saddle-v1`, `shared-flux-saddle-resolution-v1`, `shared-flux-saddle-newgeometry-v1`/`v2`, `shared-flux-saddle-perturbed-v1`, `shared-flux-saddle-paired-v1`.

Scripts: `ld_junction_signature_audit.py`, `test_ld_saddle_model.cpp`, `ld_saddle_paired_audit.py`; existing fresh validator now accepts --output, --indices, --source while preserving its default behavior.

Reproduce with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LCBININT_PROBE_BUILD=$PWD/build-boundary-support and /home/nunota/.miniconda3/envs/myenv/bin/python. Build `cmake --build build-boundary-support --target lcbinint_python test_core -j 2`; run modules `shared_flux_exact_ld_audit`, `shared_flux_junction_resolution_audit`, `shared_flux_error_fresh_validation`, `ld_saddle_paired_audit` under tests.diagnostics.recal2026. Use exclusive new output names. Unit command `ctest --test-dir build-boundary-support -R 'unit_core|unit_boundary_support' --output-on-failure`. Standalone C++ compile uses TMPDIR inside the output directory, g++ -O2 -std=c++17 -I ., followed by the compiled test.
