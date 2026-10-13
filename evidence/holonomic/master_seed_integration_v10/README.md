# Integral hot-path investigation

Accepted candidate: adaptive_radial.hpp current-panel anchor selection. Fejer sample slots are monotonically ordered in radius, so a fresh node uses the nearest reliable occupied slot on each side. Equal rounded radii retain ascending-slot tie breaking. Parent-panel search and non-fresh/reuse-disabled behavior retain the existing scan. HOLO_ADAPTIVE_SCAN_CURRENT_ANCHOR restores previous scan. HOLO_ADAPTIVE_ANCHOR_AUDIT compares the selected anchor pointer with the full scan and aborts on mismatch.

Regression binaries freshly compiled with anchor audit: shared adaptive 1608 checks / 0 failures; frontier 20 checks / 0 failures.

Rejected trials: stationary endpoint evaluation reuse did not give measurable gain; short vector compact storage did not improve on the neighbor search. Both default changes reverted. before.tsv/after.tsv overlapped compilation and are not accepted performance evidence. Initial fold.tsv/deeper.tsv overlapped a later compile and are correctness evidence only. Final *_final.tsv are serial, with all compilation complete.

Profiler instrumented run: on normal 87 conditions, arc processing ~0.112 ms (includes root-pair tracking ~0.057 ms), endpoint polish ~0.051 ms. Profiler boundaries are nested; these columns are not additive and clock instrumentation adds cost. setup/physical stats at the last adaptive invocation do not represent sums over all resumes. Do not sum them into full epoch cost.

No quadrature order, tolerance, endpoint acceptance or root-count check relaxed. Same seeds and current master library as v9. Production public routing unchanged.

Final reverse-order A/B (5 repeats): integral 0.40258 -> 0.39618 ms (-1.59%); whole excluding caustic 0.63257 -> 0.62610 ms (-1.02%), normal matched 87 cases. Earlier serial run -2.94% integral, -2.12% whole; treat gain as small (~1-3%), not a major speedup. 192 normal outputs and node counts bit-identical. Final complete input campaign 896 rows, 754 resolved D14 references, zero failures; 142 reference-unresolved rows excluded from accuracy acceptance.
