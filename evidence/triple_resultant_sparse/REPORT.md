# Triple resultant: reuse the fixed lens algebra

Adopted implementation: polarize the sextic's Bezout matrix in the source variables and cache the lens-only coefficients. In the final matched contact/U/rtol=1e-3 stage measurement, support preparation fell **20.1%**, and total time fell **8.5%**. The requested 30% overall reduction is NOT achieved.

The current change has performance tradeoffs: first-eight-query contact streams are about 3–5% slower because they include basis construction; repeated ordinary regular/planet cases show about 1–2% slower timings. These are retained as limitations, not excluded from the results. This is a useful repeated-geometry optimization, not a universal speedup.

## What changed

Before clearing the sextic discriminant, its polynomial can be written

`Q = Q0 + X*Qx + Y*Qy + (rho²-X²-Y²)*Qdisk`.

The four Q polynomials depend only on lens masses and positions. The Bezout operation is quadratic in Q, so its real symmetric 5x5 matrix has ten source monomials. `LensBasis` constructs the four source-independent Q bases; `MatrixBasis` polarizes the Bezout matrix and retains its ten real polynomial coefficient blocks. A source query now combines these blocks, converts the entries into homogeneous Bernstein form, and computes the degree-28 determinant.

The same row/column permutation minimizes weighted convolution work: 5910 coefficient products versus 6714 in the previous determinant ordering. Applying the same permutation to rows and columns preserves determinant sign. The original source construction remains available for one-shot queries. After four actual resultant queries with identical `(m1,m2,a,bx,by)`, a thread-local geometry basis is built. Source position and rho may vary. Any lens-identity change destroys the old basis. Its heap storage is allocated only when needed.

Only coefficient formation uses the existing hardware extended significand. Integration, physical proofs and ordinary numerical work retain their existing precision and fallback rules. There is no new quad route, no tolerance relaxation, no exact-point flux cache and no cached completeness decision: every query still performs current resultant isolation and physical inclusion.

Production changes are limited to `support_resultant.hpp` and `test_resultant.cpp`. Existing unrelated dirty work was preserved. `source.patch` is relative to the source at the beginning of this turn; the baseline header was checked byte-for-byte against the previous accepted final-source archive.

## Main A/B measurement

CPU0, one active numerical worker, no simultaneous compilation. Both engines use persistent geometry, nine center-source warmups, then the same eight distinct nearby sources over four cycles, with balanced before/after order (32 pairs per condition). The C++ timer includes prepare, support and flux integration; frame conversion, Python dispatch, engine construction and I/O are excluded. The one-off basis construction is included in the separate fresh-process startup experiment below.

Final warm coefficient microbenchmark: 1.389x (about 28.0% less coefficient time). This is not whole-integrator speedup. 3480 coefficient-interval overlap checks are diagnostics, not an independent accuracy proof.

| Case | rtol | limb u | Before median ms | After median ms | Median paired time reduction |
|---|---:|---:|---:|---:|---:|
| asymmetric | 0.0001 | 0 | 1.0743 | 1.0182 | 2.0% |
| asymmetric | 0.0001 | 0.5 | 1.1846 | 1.1468 | 2.3% |
| asymmetric | 0.001 | 0 | 1.0100 | 0.9370 | 5.2% |
| asymmetric | 0.001 | 0.5 | 1.0422 | 0.9884 | 3.1% |
| collinear | 0.0001 | 0 | 0.7796 | 0.7615 | 3.6% |
| collinear | 0.0001 | 0.5 | 1.0819 | 1.0398 | 3.0% |
| collinear | 0.001 | 0 | 0.7785 | 0.7273 | 5.4% |
| collinear | 0.001 | 0.5 | 0.9015 | 0.8761 | 4.3% |
| contact | 0.0001 | 0 | 0.7576 | 0.7108 | 7.3% |
| contact | 0.0001 | 0.5 | 1.1601 | 1.0949 | 3.7% |
| contact | 0.001 | 0 | 0.6981 | 0.6312 | 8.7% |
| contact | 0.001 | 0.5 | 0.9008 | 0.8125 | 6.4% |
| contact_rotated | 0.0001 | 0 | 0.7925 | 0.7809 | 7.1% |
| contact_rotated | 0.0001 | 0.5 | 1.1550 | 1.1543 | 4.3% |
| contact_rotated | 0.001 | 0 | 0.7265 | 0.7031 | 7.4% |
| contact_rotated | 0.001 | 0.5 | 0.9016 | 0.8942 | 6.7% |
| far | 0.0001 | 0 | 0.0555 | 0.0568 | -1.9% |
| far | 0.0001 | 0.5 | 0.0621 | 0.0609 | -0.2% |
| far | 0.001 | 0 | 0.0376 | 0.0373 | 0.3% |
| far | 0.001 | 0.5 | 0.0408 | 0.0418 | -2.3% |
| large | 0.0001 | 0 | 0.9641 | 0.9227 | 5.8% |
| large | 0.0001 | 0.5 | 1.8457 | 1.7808 | 2.0% |
| large | 0.001 | 0 | 0.8129 | 0.7658 | 5.9% |
| large | 0.001 | 0.5 | 1.0845 | 1.0495 | 3.6% |
| planet | 0.0001 | 0 | 0.1008 | 0.1022 | -0.7% |
| planet | 0.0001 | 0.5 | 0.1056 | 0.1068 | -0.9% |
| planet | 0.001 | 0 | 0.0710 | 0.0742 | -1.5% |
| planet | 0.001 | 0.5 | 0.0660 | 0.0677 | -3.2% |
| regular | 0.0001 | 0 | 0.0948 | 0.0957 | -2.0% |
| regular | 0.0001 | 0.5 | 0.1033 | 0.1038 | -0.7% |
| regular | 0.001 | 0 | 0.0749 | 0.0766 | -1.3% |
| regular | 0.001 | 0.5 | 0.0866 | 0.0875 | 0.0% |
| symmetric | 0.0001 | 0 | 0.1260 | 0.1285 | -2.0% |
| symmetric | 0.0001 | 0.5 | 0.1774 | 0.1806 | 0.8% |
| symmetric | 0.001 | 0 | 0.0993 | 0.1014 | -2.2% |
| symmetric | 0.001 | 0.5 | 0.1116 | 0.1116 | 0.2% |

Percentages use the median paired ratio, so they need not equal ratios of the separate time medians. Full raw pairs are in paired-stream.json. Ordinary cases were repeated with 128 pairs in repeat-normal.json; the small regression persisted. A noinline/code-layout trial did not consistently remove it and was reverted. Its cause is unresolved; altered code generation is a hypothesis, not a demonstrated diagnosis.

## Matched VBM comparison

Separate five-engine run: before, current algebraic, Singlepoly, Multipoly and Nopoly; identical saved geometry, source coordinates, rho, limb and relative tolerances (atol/Tol zero). Five original-source warmups and 16 measured evaluations per engine, rotated/reversed engine order, one active worker. All numbers below are median milliseconds. All methods had finite outputs and were within the requested relative error against the saved tight references on this matrix. Agreement with that reference is not an independent proof that one engine is universally correct.

This is a different timing sequence from the A/B experiment above: long VBM LD calls change the surrounding workload. Do not mix its absolute times with the A/B medians or historical measurements. Its shorter before/after comparison has mixed results; it does not establish an improvement in every condition. Annuli and sample counters are retained raw; VBM reports annuli=0 here, so this field does not substantiate a claim that LD used one annulus.

| Case | rtol | u | Before | Algebraic | Singlepoly | Multipoly | Nopoly |
|---|---:|---:|---:|---:|---:|---:|---:|
| asymmetric | 0.0001 | 0 | 1.1243 | 1.0789 | 0.3633 | 0.5086 | 1.2275 |
| asymmetric | 0.0001 | 0.5 | 1.2497 | 1.1869 | 9.2429 | 12.5672 | 28.6114 |
| asymmetric | 0.001 | 0 | 1.2516 | 1.2197 | 0.2695 | 0.3603 | 0.7791 |
| asymmetric | 0.001 | 0.5 | 1.0843 | 1.0399 | 2.2065 | 2.9265 | 6.7135 |
| collinear | 0.0001 | 0 | 0.8907 | 0.8293 | 0.6457 | 0.8411 | 2.1104 |
| collinear | 0.0001 | 0.5 | 1.3167 | 1.2601 | 22.4809 | 27.8527 | 70.5656 |
| collinear | 0.001 | 0 | 0.8904 | 0.8698 | 0.4123 | 0.5401 | 1.2107 |
| collinear | 0.001 | 0.5 | 1.0737 | 1.0287 | 4.9993 | 6.6087 | 15.7372 |
| contact | 0.0001 | 0 | 0.8308 | 0.8084 | 0.9564 | 1.2763 | 2.8908 |
| contact | 0.0001 | 0.5 | 1.4457 | 1.3207 | 48.1329 | 66.9877 | 150.5172 |
| contact | 0.001 | 0 | 0.8184 | 0.8566 | 0.5776 | 0.7520 | 1.7211 |
| contact | 0.001 | 0.5 | 1.0119 | 0.9545 | 10.0749 | 12.9343 | 32.7180 |
| contact_rotated | 0.0001 | 0 | 0.9008 | 0.8310 | 0.9503 | 1.3105 | 3.0577 |
| contact_rotated | 0.0001 | 0.5 | 1.3150 | 1.3141 | 51.3425 | 69.3049 | 154.8435 |
| contact_rotated | 0.001 | 0 | 0.8248 | 0.8113 | 0.5217 | 0.7078 | 1.6792 |
| contact_rotated | 0.001 | 0.5 | 1.0605 | 1.0293 | 10.9870 | 13.8205 | 35.0993 |
| far | 0.0001 | 0 | 0.1018 | 0.1001 | 0.2678 | 0.3400 | 0.4688 |
| far | 0.0001 | 0.5 | 0.0783 | 0.0777 | 0.2082 | 0.2639 | 0.3738 |
| far | 0.001 | 0 | 0.0560 | 0.0566 | 0.1281 | 0.1735 | 0.2410 |
| far | 0.001 | 0.5 | 0.0766 | 0.0789 | 0.1795 | 0.2313 | 0.3232 |
| large | 0.0001 | 0 | 1.0293 | 1.0364 | 0.9011 | 1.1599 | 2.7034 |
| large | 0.0001 | 0.5 | 2.3127 | 2.5117 | 53.0632 | 72.3135 | 167.6826 |
| large | 0.001 | 0 | 0.9393 | 0.8869 | 0.4929 | 0.6355 | 1.4510 |
| large | 0.001 | 0.5 | 1.2510 | 1.2595 | 10.8975 | 13.8822 | 33.2217 |
| planet | 0.0001 | 0 | 0.1156 | 0.1171 | 0.2860 | 0.3748 | 0.8425 |
| planet | 0.0001 | 0.5 | 0.1360 | 0.1391 | 0.4772 | 0.6075 | 1.3654 |
| planet | 0.001 | 0 | 0.0838 | 0.0819 | 0.1634 | 0.2104 | 0.4889 |
| planet | 0.001 | 0.5 | 0.0831 | 0.0846 | 0.1617 | 0.2089 | 0.4674 |
| regular | 0.0001 | 0 | 0.1129 | 0.1155 | 0.1936 | 0.2757 | 0.4662 |
| regular | 0.0001 | 0.5 | 0.1211 | 0.1209 | 0.1962 | 0.2802 | 0.4774 |
| regular | 0.001 | 0 | 0.0933 | 0.0945 | 0.1117 | 0.1593 | 0.2757 |
| regular | 0.001 | 0.5 | 0.1054 | 0.1017 | 0.1276 | 0.1722 | 0.2989 |
| symmetric | 0.0001 | 0 | 0.1474 | 0.1461 | 0.2432 | 0.5903 | 0.4336 |
| symmetric | 0.0001 | 0.5 | 0.2041 | 0.2031 | 1.6762 | 4.1403 | 2.7982 |
| symmetric | 0.001 | 0 | 0.1174 | 0.1197 | 0.1339 | 0.3159 | 0.2317 |
| symmetric | 0.001 | 0.5 | 0.1394 | 0.1361 | 0.3967 | 0.9675 | 0.6694 |

## Where the Singlepoly gap moved

The separate stage comparison uses 64 nearby-source evaluations per engine and additive means, with the same balanced five-engine sequence (contact, U, rtol=1e-3). Only this table should be used for the additive decomposition.

| Mean ms | Before | After |
|---|---:|---:|
| Support preparation | 0.332166 | 0.265402 |
| Fallback | 0.000054 | 0.000054 |
| Flux integration | 0.487142 | 0.483921 |
| Total algebraic | 0.819362 | 0.749377 |
| Singlepoly, same sequence | 0.549295 | 0.549295 |
| Difference from Singlepoly | 0.270067 | 0.200082 |

Multipoly: 0.708287 ms; Nopoly: 1.627059 ms. The gap to Singlepoly fell by 25.9% in this measurement; it remains positive. Flux time is essentially unchanged, supporting that the gain came from preparation.

## Startup cost is not free

startup.json starts fresh processes (resetting the geometry-only TLS cache), uses no warmup, and measures the first eight distinct sources. Four trials per condition, balanced before/after order. A new PreparedRegularSupport alone would NOT reset this TLS cache, so the previous cold_paired.py protocol is not used as evidence of cold lens-basis cost.

The fourth actual resultant query constructs the basis. In contact, that query adds approximately 0.6–0.7 ms; the first-eight-query total is approximately 3–5% slower. Initial single-query times are noisy across four trials and show no consistent speedup. Long streams amortize this expense; short streams can lose. No claim is made that a one-shot point evaluation has become 8% faster.

| Contact u | rtol | First-eight paired speed ratio (before/after) |
|---|---:|---:|
| 0 | 0.0001 | 0.9611 |
| 0.5 | 0.0001 | 0.9724 |
| 0 | 0.001 | 0.9500 |
| 0.5 | 0.001 | 0.9607 |

## Completeness-cache research continued

The previous attempt independently widened the source variables through the degree-28 determinant; its enclosures were unusable. This turn also implemented a different prototype: prove one event tube for every source in a square neighborhood, subtract those tubes (including opposite-chart coverage), then exclude the remaining physical F/F_t domain. A source-aware physical exclusion bound handles poles; parameter-interval Krawczyk exclusions eliminate the remaining overestimation strips. Unresolved boxes and failed tubes remain failures, never acceptance.

At halfwidth 1e-3*rho, the prototype reaches zero unresolved cells in 6/9 geometries, including contact, collinear and asymmetric. Contact proves 12 tubes and completes the complement in 6132 evaluated cells, about 10.5 ms in that run. The simple direct-physical prototype from the preceding turn did not reach this point. Large fails to construct all tubes at that width; broader 1e-2*rho neighborhoods mostly fail. The tube prototype is not installed in PreparedRegularSupport: its construction cost and narrow reuse neighborhood do not meet the direct-point workload, and independent source-neighborhood/degeneracy validation is still required. No speedup is attributed to this research prototype.

The files tube-probe.cpp, tube-probe-krawczyk.jsonl and earlier intermediate outputs preserve this investigation. Reproduction links the existing native geometry object and magnification archive, as shown in the recorded build scripts; the physical source-interval Ray snapshot is in ../triple_source_certificate/physical_box.hpp.

## Validation and rejected variants

- 20 native C++ tests passed; public native API/invariance/gradient smoke tests passed.
- 87 unique exact Gaussian-rational reference coefficients passed in cold, warm and every lens-identity invalidation sequence: 2262 reference enclosure checks. Assertions build passed as well.
- All 36 paired conditions passed requested relative error and reported-error-plus-reference checks at the measured nearby coordinates.
- Saved heldout matrix: 120 conditions passed; LD extremes (u=0 and u=1, including grazing): 64 passed. No reference was recomputed or tolerance loosened.
- Both parameter conventions passed 40 analytic-gradient versus two-step finite-difference comparisons using the final native build. This is a bounded gradient check, not general inference certification.
- Generic sparse complex multiplication: 0.990x coefficient speed ratio, rejected.
- Reordered determinant alone: 1.063x; Q-basis-only prototype: 1.129x. The adopted implementation additionally caches the polarized matrix.
- Exact-zero skipping during matrix preparation did not improve its measured construction time, rejected. The noinline/code-layout trial did not consistently fix ordinary timing, rejected.
- No UBSan result is claimed: the installed compiler's UBSan runtime was already found missing in the preceding source-certificate experiment. This turn uses the successful assertions build and the native numerical checks.

## Remaining work

The total 30% target remains open. The next useful work is to reduce matrix-basis startup cost, avoid the ordinary-case regression, and make a source-neighborhood certificate cheap enough to amortize at realistic source spacing. The current measured improvement is a repeated-lens preparation gain; it does not solve the remaining Singlepoly uniform gap or all LD cost.

Raw inputs are linked to the prior preserved conditions/references. provenance.json identifies the source/build hashes, compiler, branch/HEAD and timing boundary. source.patch contains only this turn's two production-file changes. SHA256SUMS covers the retained local artifacts. No commit or push was made.
