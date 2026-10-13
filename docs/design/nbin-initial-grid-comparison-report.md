# Initial nbin comparison

Directional adaptive, default axis ratio4, retained-row reuse enabled. Initial16/32/64; max512; lattice map budget1e6. Same independent saved references as directional-adaptive-audit-v4. Uniform27, linear LD c=.5:25, c=1:22 geometries, two tolerances; 444 numerical calls. All cases retained. One thread; nine rotated alternating timing batches per profile. Geometry changes between adjacent calls. Full scalar calls, no compilation during timing. Fixed comparator is reference-informed minimum passing fixed nbin; selection cost excluded.

| tolerance | fixed oracle (s) | initial16 (s) | initial32 (s) | initial64 (s) |
|---|---:|---:|---:|---:|
| 0.001 | 0.107532 | 0.123224 | 0.151817 | 0.321132 |
| 0.0001 | 0.151262 | 0.312668 | 0.334065 | 0.464558 |

Times are sums of profile medians, not the median of a combined batch. Relative to initial16: initial32 +23.2% / +6.8%, initial64 +160.6% / +48.6% for tolerance1e-3 /1e-4.

At1e-3 all222 numerical calls pass. At1e-4 initial16 passes74/74, initial32 73/74, initial64 73/74. Initial32 false acceptance: uniform actual1.158125773192964e-4. Initial64 false acceptance: c=1 actual1.003162885275799e-4. Full inputs and indicators in false-accepts.json. No general accuracy certification is claimed.

At1e-4 total new lattice maps: initial16 2238161, initial32 2712511, initial64 4596740. History includes the initial record, so history lengths in summary are stage-record counts, not solely refinement counts. Starting finer reduces later stages but raises total classification work. This supports broad initial overresolution as a cost, and does not prove every individual geometry is slower. No per-geometry timing was collected. The precise mechanism of the changed-path false acceptances remains unresolved; retain them as regressions for the error model.

Artifacts: tests/diagnostics/results/recal2026/adaptive-initial-bins-v2/{summary.json,accuracy.json,false-accepts.json}; script tests/diagnostics/recal2026/adaptive_initial_bins_cost.py. Native and script SHA256 recorded in summary. Checkout feature/boundary-error-and-discriminant-support HEAD b51c00c566667fc2ed1fe640543219f480642002, dirty experimental overlay. Native implementation unchanged. First harness launch failed on duplicate initial_bins key before numerical matrix completion; preserved v1 directory and log. Corrected harness ran to completion in v2.

Conclusion: a uniformly higher starting nbin does not resolve the observed high-precision overhead. Keep initial16 for now. Next work should localize refinement and investigate changed-path error-model misses rather than increase the global starting grid.
