# Triple algebraic integration checkpoint — 2026-10-28

The default derivative route transports the value and eight canonical derivatives together, with refinement controlled by the value tolerance. Scalar calls execute no parameter-derivative work. The offline gradient gate remains `1e-4 + 2e-4*abs(reference)`, including independent-reference uncertainty.

This checkpoint adds retained local Patterson cell refinement, cached physical-boundary regularity witnesses, inline angular storage and contiguous radius caches, fixed-angle physical LD sensitivity transport, and shared hyperbolic nodes. It also adds scalar LD absolute roundoff budgeting for loose requests, mirrored angular trigonometry and shared eight-direction contractions, and an early exclusion of absent coordinate-hint maxima. Physical support completeness and pole-distance guards remain active. Strict local requests retain early precision promotion; a small-source rtol1e-9 convergence regression is fixed in `test_regular.cpp`.

## Validation

- Release native rebuild; CTest **30/30**.
- Independent scalar FD references: **160 canonical components**, plus **80 pure-LD identity components**, all within the unchanged gradient gate.
- Physical gradients through rebuilt public Python API and C ABI: **88/88** each.
- Broad comparator **120/120**, saved **86/86**, new holdout **24/24**, scalar **168/168**.
- Three previously converged strict small-source requests recover baseline convergence and quad counts. An older rho=.001,rtol1e-9 baseline nonconvergence is retained as a diagnostic, not claimed solved.

The five raw independent FD reference files and the new `triple_gradient_contract` CTest fixture preserve the derivative regression inputs. `gradient_valid` denotes usable output, not universal precision certification; default `gradient_precision_checked` and `gradient_reliable` remain false.

## Performance

[Final naturally routed VBM comparison](../triple_speed_comparison_ld_20261028/REPORT.md): one triple geometry, twelve source disks, U/linear LD u=.5, rtol1e-3/1e-4, atol0. All48 conditions enter integration under the normal router; no point/hex, source-disc shortcut or light-curve API is timed. VBM5.5 SinglePoly/MultiPoly/NoPoly are reported separately. Geometry construction is excluded and support preparation is included in lcbinint warm point timings.

Against the immediately preceding validated working-tree build, two matched paired runs show LD gradient time reductions of **4.3–5.3%** and scalar LD rtol1e-4 reductions of **1.4–1.5%**; scalar LD rtol1e-3 is effectively unchanged. Frequency is uncontrolled; use within-run paired changes rather than subtracting historical tables. The **30% derivative-overhead goal is not achieved**.

[Implementation notes and rejected trials](../triple_ld_final_cost/NOTES.md), [paired improvement data](../triple_ld_final_cost/improvement-summary.json), and [source provenance](../triple_ld_final_cost/source-manifest.json) describe the final selection. Native binaries, build products and full experimental snapshots remain local; saved hashes identify the measured artifacts. Some local-only exploratory artifacts referenced by the provenance/notes are intentionally not tracked. The older `selected-default-*.json` files describe the earlier checkpoint rather than the current implementation.
