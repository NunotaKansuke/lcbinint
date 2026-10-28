# Triple normal lightcurve pipeline: implementation and matched timing

Native static single-source triples with uniform/linear LD now use the ordinary default LightCurve API. No backend flag is supplied in the measurements. Point/hex/integral selection and actual lazy caustic-tree construction are included. Values and tangents share scalar routing, roots and support preparation. Detailed design and scope: IMPLEMENTATION.md; public documentation: docs/python/triple-algebraic.md.

## Validation

```json
{
  "matrix_status": "COMPLETE",
  "lens_configurations": 7,
  "native_epoch_results": 1008,
  "nonzero_status": 0,
  "invalid_gradients": 0,
  "values_above_requested": 0,
  "route_counts": {
    "0": 500,
    "7": 120,
    "1": 388
  },
  "python_regression": "27 passed",
  "ctest": "30/30 passed",
  "prior_error_counterexamples": "3/3 pass in both modes"
}
```

The Python regressions check all ten LightCurve parameter tangents on point/hex/integral routes, two step sizes where applicable, both coordinate conventions, a secondary-dominant mass case, zero radius, implicit/explicit backend equivalence and three captured scalar-error failures. The binary ordinary-call regression subset remains unchanged. No runtime derivative-error contract is introduced. The six earlier lens configurations were extended to seven, with two radii, U/LD and relative1e-3/1e-4. VBM SinglePoly and MultiPoly reference values use relative1e-7/absolute0. All raw values, flags, routes and differences are retained in matrix.json. Agreement is evidence for this matrix, not a universal precision certificate.

## Full curve, x=-2..2

Five independent persistent subprocesses, all CPU0, serial rotated/reversed round-robin requests. No backend flag in ordinary lcbinint magnification(). VBM TripleLightCurve SinglePoly/MultiPoly/NoPoly. Each constructor and first API call are cold. Two warmups and eight timed calls per engine; cold and changed-geometry times separate. Thread CPU milliseconds inside each worker exclude IPC/imports/serialization. Calls include actual lazy caustic-tree construction, routing, value/tangent and API conversion. No builds or validation concurrently. CPU frequency and external load uncontrolled. VBM Tol=0/RelTol=request, so its current corrquad<Tol point gate cannot pass; this is the requested relative-only contract, not VBM default Tol.

129 epochs, y=.15, rho=.03. Lenses (0,1,-.4+.6i), masses (.72,.20,.08). All times are milliseconds for the complete public-API call. VBM columns are values only. lcbinint gradient columns include ten analytic parameter derivatives.

| LD | rtol | SinglePoly | MultiPoly | NoPoly | lcbinint value | value+gradient | extra | cold value | cold gradient | routes PS/hex/integral |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|0.0|0.001|24.081|31.817|65.623|22.556|27.118|20.2%|30.871|35.352|66/30/33|
|0.0|0.0001|42.154|55.937|115.701|31.804|34.001|6.9%|40.177|41.931|14/82/33|
|0.5|0.001|70.879|96.358|218.475|23.739|36.013|51.7%|31.880|44.308|66/30/33|
|0.5|0.0001|274.459|375.786|885.861|36.104|42.828|18.6%|44.334|50.705|14/82/33|

## Caustic-focused interval, x=-.5..1.1

Five independent persistent subprocesses, all CPU0, serial rotated/reversed round-robin requests. No backend flag in ordinary lcbinint magnification(). VBM TripleLightCurve SinglePoly/MultiPoly/NoPoly. Each constructor and first API call are cold. Two warmups and eight timed calls per engine; cold and changed-geometry times separate. Thread CPU milliseconds inside each worker exclude IPC/imports/serialization. Calls include actual lazy caustic-tree construction, routing, value/tangent and API conversion. No builds or validation concurrently. CPU frequency and external load uncontrolled. VBM Tol=0/RelTol=request, so its current corrquad<Tol point gate cannot pass; this is the requested relative-only contract, not VBM default Tol.

129 epochs, y=.15, rho=.03. Lenses (0,1,-.4+.6i), masses (.72,.20,.08). All times are milliseconds for the complete public-API call. VBM columns are values only. lcbinint gradient columns include ten analytic parameter derivatives.

| LD | rtol | SinglePoly | MultiPoly | NoPoly | lcbinint value | value+gradient | extra | cold value | cold gradient | routes PS/hex/integral |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|0.0|0.001|28.750|39.712|102.656|49.351|61.123|23.9%|60.893|72.542|0/47/82|
|0.0|0.0001|52.534|72.160|184.861|53.627|62.665|16.9%|65.084|73.834|0/47/82|
|0.5|0.001|141.513|195.559|470.190|51.697|81.785|58.2%|63.141|93.276|0/47/82|
|0.5|0.0001|613.996|844.735|2038.597|61.187|83.438|36.4%|72.912|94.996|0/47/82|

## First-call totals for all engines

Constructor plus first public API call in fresh processes, milliseconds. These include geometry construction by the respective implementation.

| track | LD | rtol | SinglePoly cold | MultiPoly cold | NoPoly cold | value cold | gradient cold |
|---|---:|---:|---:|---:|---:|---:|---:|
|Full curve, x=-2..2|0.0|0.001|24.107|31.756|65.678|30.871|35.352|
|Full curve, x=-2..2|0.0|0.0001|42.527|55.752|116.135|40.177|41.931|
|Full curve, x=-2..2|0.5|0.001|70.888|96.383|218.356|31.880|44.308|
|Full curve, x=-2..2|0.5|0.0001|267.138|357.907|885.900|44.334|50.705|
|Caustic-focused interval, x=-.5..1.1|0.0|0.001|28.841|39.709|102.676|60.893|72.542|
|Caustic-focused interval, x=-.5..1.1|0.0|0.0001|52.664|72.236|184.890|65.084|73.834|
|Caustic-focused interval, x=-.5..1.1|0.5|0.001|141.963|195.560|470.229|63.141|93.276|
|Caustic-focused interval, x=-.5..1.1|0.5|0.0001|614.613|846.400|2039.666|72.912|94.996|

## Interpretation and boundaries

The full curve and caustic-focused interval are both reported: a large number of cheap epochs must not hide the remaining uniform/contact deficit. Derivative overhead is per whole curve and depends on its route mix. LD derivative overhead is not universally below30%. These are single-geometry timing examples; the seven-geometry matrix checks values/status/route coverage separately.

Cold time is constructor plus first normal API call in a fresh engine process. It includes the actual continuous caustic-enclosure tree and support work, not a prebuilt plotting polyline. Nearby-source clearance and critical-phase enclosures are shared with the eventual integral support proof, following binary. Full plotting polylines have a separate explicit API and are not artificially added as a fictitious fixed cost. Changed-geometry times are single calls with s=1.0001 and are retained separately, not claimed as a robust isolated construction-time estimate.

Every engine uses its own persistent subprocess and state; the parent sends requests serially in rotating/reversed order on CPU0. Warm medians use eight calls after two excluded calls. Timing inside each worker excludes imports, IPC and raw-result serialization. CPU frequency and external load are not controlled. No compilation or validation ran concurrently with these final timing series. VBM Tol=0 and RelTol=request implement the requested relative-only tolerance contract. Its current MultiMag2 corrquad<Tol gate cannot pass at Tol=0; this must not be described as VBM default settings.

Incomplete support/continuation still refuses shortcuts or reports failure. Derivatives are not finite-difference implementations: frozen physical roots use the implicit lens equation, integral tangents are carried with their own scalar integration. Finite differences here are offline tests only. CTest and finite gradient flags do not constitute global scientific certification.

## Reproduction

Build: cmake --build build-triple-native -j2 (Release, OpenMP OFF, JAX FFI OFF). Python: /home/nunota/.miniconda3/envs/myenv/bin/python. Tests: regression-final.log and ctest-final.log. Validate: OPENBLAS_NUM_THREADS=1 python evidence/triple_default_pipeline/validate_matrix.py. Benchmark sequentially: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python evidence/triple_default_pipeline/interleaved.py ; then the same command with --wide. Use the recorded interpreter and preserve the editable-hook filter/verified build path in these scripts.

before.diff and before-head.txt freeze the pre-existing work. before-patterson-stream.hpp preserves the earlier unrelated triple changes before this task modified its scalar error reduction. Source/header/native hashes, command and dirty identity are in provenance.json. No commit or push is part of this task.
