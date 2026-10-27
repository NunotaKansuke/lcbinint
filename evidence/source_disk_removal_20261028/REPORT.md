# Remove embedded algebraic source-disk quadrature

Checkout: `/rogue1_8/nunota/lcbinint/experiments/algebraic_boundary`, branch `experiment/algebraic_boundary`, baseline HEAD `59be25417ba30a2277e40dfadb32adf8fd06f670`. Baseline source and native module are preserved locally in `baseline-src/` and `baseline-native.so`. The pre-existing ten dirty JAX/backend documentation files were preserved. No worktree was created.

## Change

Deleted binary `source_disk_gradient.hpp`, its eager source quadrature, early/late repair calls, thread-local cover cache, and the fused-gradient repair callback. Binary gradients now always use the existing seeded image-plane radial integrator after ordinary point/hex routing. Existing radial precision/topology retries remain.

Deleted triple `smooth_source.hpp` and both strict-diagnostic entry points. The default value-mesh gradient and scalar paths already used image-plane integration. Keep the three public `smooth_source_*` diagnostics as false/zero compatibility fields. Replace the old source-product quadrature test with smooth image-plane strict/value-mesh regression checks, and add binary regressions for two configurations formerly repaired by source quadrature.

Explicit standalone/preplanned source-plane backends and independent JAX APIs are outside this algebraic-shortcut removal; they are not deleted. In particular, this does not modify the concurrent dirty JAX implementation.

## Validation

- Release native build, at most two compiler processes; project-local compiler temporary directory.
- CTest 30/30 after route removal. The subsequently added binary regression is also run separately (`core-regression.log`).
- Public triple physical derivative gate: 88/88 (`native-physical.json`). Independent canonical derivative gate including pure-LD identities: 240/240 (`triple-independent.json`). Gate unchanged: `1e-4 + 2e-4*abs(reference)` including reference uncertainty where provided.
- Binary public before/final comparison: 203 matched configurations, each scalar and gradient: 406 calls per version. Inputs cover six representative disks (uniform/LD, `rtol=1e-3/1e-4`), 64 deterministic random geometries (same tolerances), archived near-axis/edge configurations (also `1e-7`), and three extreme exact-contact epochs (also `1e-7`). No success/reliability flag regressed. 403/406 outputs and diagnostic strings are exactly unchanged. Three old source-repair gradient calls change to the radial chart; their values remain within the requested relative error and all three retain valid gradients.
- Independent binary derivative comparisons: 100 smooth-source components in `binary-fd.json`, 20 contact components in `binary-contact-reference-fd.json`, 20 former-repair components in `binary-used-fd.json` pass the unchanged gate including two-step FD uncertainty. Source-disk baseline is used only as a smooth-value oracle for the first group. Contact and former-repair references use scalar values and five-point stencils.
- Initial contact checks are retained, including inconclusive references: a 1%-radius difference step crossed a feature, and very strict `1e-8/1e-10` references sometimes did not report convergence. The final contact reference uses a smaller radius step and converged scalar `rtol=1e-7`; the acceptance gate and production requested tolerances are unchanged. These initial files must not be counted as successful certifications.
- Known extreme-contact unconverged/invalid outputs remain unchanged; deletion does not solve or conceal those existing limitations (`exact-before.json`, `exact-final.json`).

## Timing

`benchmark.py`, `timings.json`: one pinned CPU, thread CPU time, four warmups plus sixteen alternating samples, separate ELF namespaces and persistent per-condition/per-version state. Build and validation jobs finished before timing. Setup excluded; support preparation and integration included. Before/after share source, tolerances, compiler optimization and precision. No new VBM timing claim is made.

Triple naturally integrated population: 48 conditions, both scalar and default gradient (96 pairs). Every non-timing output, including all eight gradients, is bitwise unchanged. Median paired after/before ratios: scalar 1.0006, gradient 0.9990 (no meaningful speed change).

Binary old source-repair cases, gradient milliseconds:

| configuration | rtol | before | after |
|---|---:|---:|---:|
| random_20 | 1e-3 | 2.633 | 2.308 |
| random_20 | 1e-4 | 3.732 | 2.352 |
| random_61 | 1e-4 | 3.760 | 3.407 |

Across the 41 binary gradient timing conditions, median paired ratio is 0.990. Binary repeated scalar calls hit the existing epoch cache (about microseconds); those raw times are retained but are not evidence for fresh-point integral performance. Source removal does not change the scalar implementation.

Tradeoff: explicit triple strict-policy smooth LD now uses image-plane integration and is slower: at `X=.5,Y=.4,rho=.03`, limb .5, `rtol=1e-3`, 0.652 to 5.767 ms; at `1e-4`, 1.382 to 6.862 ms. Values and strict gradients converge. The ordinary default value-mesh policy is unaffected. This is a measured diagnostic-policy cost, not a universal performance statement.

## Reproduction and provenance

`binary.cpp` is the old source-enabled harness; `binary-final.cpp` uses only the new image-plane path. `without-source.cpp` is the preliminary baseline-only prototype disabling the three calls before production edits. Final comparisons use `binary-final.so` built against the actual edited native libraries, not that prototype. `check_binary.py`, `check_archived.py`, `check_random.py`, `check_exact_contact.py`, and FD scripts persist all cases including failures. `artifact-manifest.json` records source/library hashes. Native source build flags are Release `-O3 -DNDEBUG -std=gnu++17`; harnesses additionally use `-fno-fast-math -ffp-contract=off`.
