# Earlier difficulty gates: bounded negative result

2026-10-18, experiment/algebraic_boundary at e740210b plus the preserved previous-turn working tree. No earlier optimized changes were reverted. No new production change was retained in this investigation.

## Clarification of the population

The notebook contains354 hex epochs and47 integral epochs. The previously labelled simple37 are integral epochs for which the existing detailed analysis ultimately finds no need for special replay, not epochs already known easy at the API entrance. The ordinary401-point curve contains276 point,121 hex and only4 integral epochs. Its existing full API gradient/value ratio is1.231. Thus the expensive screen is already bypassed for the great majority of ordinary epochs. This does not establish that an even earlier cheap classifier for the remaining integral epochs is impossible.

## Attempts and outcome

Same public API timing protocol, inputs, LD=.5 and atol=rtol=1e-4 as evidence/geometry_screen_cache_20261018/after5 (notebook44.3585ms, ordinary5.8224ms). Each candidate uses9 alternating repetitions, one pinned CPU and one BLAS/OpenMP thread. Public call includes route/seed preparation; existing thread-local refined-cover behavior is unchanged unless noted. Historical/cross-run noise limits small differences.

1. Tighter interval squaring and direct Graeffe recurrence: notebook44.6423ms, ordinary5.8353ms. No useful gain; restored.
2. Early analytic rectangle certificate using interval D and D': root-free gaps require nonzero Re D; known-root rectangles require fixed-sign Re D' (univalence) plus a certified real sign-changing bracket. Notebook45.7829ms, ordinary5.9660ms. The sufficient certificate is too inconclusive to offset its own cost. Restored. Candidate source is rejected.hpp.
3. Deflate already witnessed real roots using enclosing root intervals and synthetic division; certify constant dominance of the quotient before general root counting. Notebook44.8238ms, ordinary5.8576ms. Simple37 costs12.1414ms versus11.8542ms before; no benefit. Restored. See ../fold_deflation_20261018/rejected.hpp and timing.log.
4. Expand eligibility of the existing smooth source-disk tangent chart to disks with certified caustic clearance greater than2rho. Do not change value-only integration. Coarse-cover-only result: notebook44.7522ms, ordinary5.8490ms. Allow a reusable512-interval enclosure at the entrance and reuse an already finer enclosure: notebook45.4716ms, ordinary6.1284ms. Additional preparation/queries offset potential savings. Restored. See ../early_smooth_disk_20261018/rejected.hpp and timing*.log. Repeated-call measurements can reuse the existing thread-local fine cover; therefore these numbers do not prove fresh-model preparation improvements.

## Closeout

All rejected production changes have been restored to the turn-start source snapshots. The returned implementation is the previously validated interval-first screen, with ordinary full-curve overhead around23% and simple-integral overhead around87%. No new performance improvement or universal classifier is claimed. The unresolved optimization is a cheap sufficient early distinction inside the integral population; neither late absence of a special replay nor successful ordinary-case timing establishes such a distinction by itself.

Restoration is checked by byte comparisons of the source headers and the restored native build checksum (restore-verification.json). No new tests are attributed to discarded candidate implementations. Previous validation remains the62 Python tests, unit_core and120-case parity documented in ../geometry_screen_cache_20261018/REPORT.md. No commit or push.
