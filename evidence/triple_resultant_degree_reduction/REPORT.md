# Triple algebraic: reduce preparation, image proofs and LD value cost

The final implementation improves several independent parts without changing requested tolerances, final acceptance criteria, default backend selection or parameter-derivative arithmetic. Compared with the previous accepted triple_resultant_sparse build, the paired nearby-source tests show roughly 4–7% less time in the regular fixture, 12–16% in far, 5–6% for contact LD and 6–7% for large LD. Contact uniform improves only about 1–3%. These are measured fixture results, not a universal speedup or full-fit certification. The 30% overall target remains open.

## Adopted changes

1. **Share fixed-lens polynomial products.** The old matrix-basis constructor repeatedly formed derivative products and then applied the real congruence. The generated exact contraction has 96 integer terms and only 24 distinct sextic coefficient products. Each product is formed once per source-basis pair and reused across all real matrix entries. Exact zero tails are inspected once on the fixed bases; uncertain zeros are retained. The isolated balanced constructor probe fell from 0.6764 to 0.4622 ms (about 32% less time). Only the existing hardware extended-significand coefficient tier is used.
2. **Use the root-count certificate immediately.** One Bernstein variation already establishes exactly one discriminant root in a radial cell. The planner now attempts current physical inclusion there, instead of splitting unconditionally to depth six. A failed proposal retains its Newton work for child cells. No event is accepted without the same current physical inclusion, and unresolved coverage still goes to the incumbent physical planner.
3. **Defer component construction until it is needed.** The projected planner first constructs global coverage. If unresolved bands remain, a lazy factory builds the partial image components and the physical fallback receives the already completed events and radial exclusions. A complete one-dimensional projection does not call the center-image solver. Tests assert both zero factory calls for complete projection and exactly one call for fallback. No completed projection work is discarded and rebuilt.
4. **Reuse normal-range magnitudes in the value kernel.** The double LD kernel already needs squared norms. Their square roots now supply the magnitudes used solely in roundoff bounds, with four-epsilon relative padding. Tiny/overflowing squared norms retain scaled hypot. The physical phi calculation is unchanged. The helper is independently checked against quad precision over 24022 cases, including extreme/subnormal inputs. Gradient-only work is not introduced into the value path.
5. **Reuse disjoint full-source image disks and their motion model.** A successful full-source contraction already encloses an entire image branch. When all retained disks are proven and disjoint from each other and omitted disks, the planner starts from those disks instead of proving eight new source-arc disks per image. It prepares the disk-invariant kappa/U/third-derivative bounds once. The motion enclosure arithmetic matches the old implementation at 46416 tested interval endpoints. If the existing cheap-motion validity test fails, the old initial-cover route remains. If a particular arc is unresolved, only that arc receives a narrower disk, validated by containment or a common certified image witness. Other accepted branches remain intact. The already computed motion enclosure is also passed to event refinement.

The full-source reuse gate is based on actual disjointness and the existing motion-bound validity criterion, not a benchmark-case name or fitted source-size threshold. There is no cached flux and no previous-query completeness decision. The existing caustic count/cache and seed validation still establish image coverage. Motion models live only within one source query; the fixed-lens coefficient cache keeps its exact lens identity and four-query activation policy.

## Numerical checks

- 20 native C++ tests passed, including event-count changes across source-limb contact, independent full 2D event coverage, cache invalidation, budget failures and component masks.
- Assertions builds of resultant and magnitude tests passed. The 87 unique exact rational resultant reference coefficients are tested on cold, warm and lens-invalidation paths (2262 enclosure checks).
- The term generator independently compares the integer table with the unsimplified Bezout/congruence construction on generic Gaussian-integer polynomials: 4950 exact coefficient equalities. Run `python tests/triple_algebraic/generate_resultant_matrix_terms.py --check`.
- 324 source/profile/tolerance conditions were checked for five consecutive calls each, including the fourth-call matrix-basis construction: 1620 direct point evaluations. All converged and passed requested relative error and reported-error-plus-reference checks. Maximum observed error was 0.4337 of the requested allowance. These checks use saved tight references; they are not universal precision certification.
- Separate saved heldout matrix: 120 checks passed. LD extremes including grazing cases: 64 passed. Public native API, translation/rotation invariance and gradient smoke tests passed. Both public parameter conventions passed 40 two-step finite-difference gradient comparisons.
- All three VBM methods were finite and within requested relative error against the saved reference matrix. This does not establish that any engine is universally correct or that a future discrepancy must belong to another engine.

## Direct point A/B benchmark

CPU0, one active numerical worker, no concurrent compilation. Same persistent lens geometry, source coordinates, rho, limb and tolerances; absolute tolerance zero. Nine original-source warmups, then eight distinct nearby sources over four cycles, with balanced before/after order: 32 pairs per condition. C++ timers include support preparation and integration; engine construction, frame conversion, Python dispatch and I/O are excluded. All timings below are median milliseconds. Reductions use the median paired ratio, so they need not equal ratios of separate medians.

| Case | rtol | u | Before ms | After ms | Paired time reduction |
|---|---:|---:|---:|---:|---:|
| asymmetric | 0.0001 | 0 | 1.4354 | 1.4040 | 4.7% |
| asymmetric | 0.0001 | 0.5 | 1.7012 | 1.5852 | 6.7% |
| asymmetric | 0.001 | 0 | 1.3750 | 1.3283 | 4.9% |
| asymmetric | 0.001 | 0.5 | 1.4478 | 1.3889 | 5.8% |
| collinear | 0.0001 | 0 | 1.1312 | 1.0553 | 8.0% |
| collinear | 0.0001 | 0.5 | 1.4901 | 1.3987 | 8.1% |
| collinear | 0.001 | 0 | 1.0390 | 0.9749 | 7.1% |
| collinear | 0.001 | 0.5 | 1.2333 | 1.1401 | 9.9% |
| contact | 0.0001 | 0 | 0.9578 | 0.9197 | 3.2% |
| contact | 0.0001 | 0.5 | 1.4253 | 1.3859 | 6.3% |
| contact | 0.001 | 0 | 0.9060 | 0.8991 | 1.1% |
| contact | 0.001 | 0.5 | 1.1104 | 1.0689 | 5.5% |
| contact_rotated | 0.0001 | 0 | 1.0067 | 0.9609 | 1.8% |
| contact_rotated | 0.0001 | 0.5 | 1.5889 | 1.4552 | 5.8% |
| contact_rotated | 0.001 | 0 | 0.9654 | 0.9298 | 2.4% |
| contact_rotated | 0.001 | 0.5 | 1.1963 | 1.1253 | 5.3% |
| far | 0.0001 | 0 | 0.0799 | 0.0674 | 15.9% |
| far | 0.0001 | 0.5 | 0.0851 | 0.0726 | 15.0% |
| far | 0.001 | 0 | 0.0555 | 0.0482 | 11.8% |
| far | 0.001 | 0.5 | 0.0572 | 0.0482 | 14.8% |
| large | 0.0001 | 0 | 1.2359 | 1.2256 | 2.5% |
| large | 0.0001 | 0.5 | 2.6076 | 2.4145 | 6.8% |
| large | 0.001 | 0 | 1.0977 | 1.0661 | 1.7% |
| large | 0.001 | 0.5 | 1.5105 | 1.4066 | 6.1% |
| planet | 0.0001 | 0 | 0.1465 | 0.1443 | 1.7% |
| planet | 0.0001 | 0.5 | 0.1533 | 0.1504 | 3.1% |
| planet | 0.001 | 0 | 0.0935 | 0.0905 | 4.0% |
| planet | 0.001 | 0.5 | 0.0989 | 0.0960 | 3.0% |
| regular | 0.0001 | 0 | 0.1391 | 0.1323 | 4.1% |
| regular | 0.0001 | 0.5 | 0.1502 | 0.1409 | 5.5% |
| regular | 0.001 | 0 | 0.1107 | 0.1048 | 6.3% |
| regular | 0.001 | 0.5 | 0.1185 | 0.1109 | 7.3% |
| symmetric | 0.0001 | 0 | 0.1877 | 0.1836 | 2.4% |
| symmetric | 0.0001 | 0.5 | 0.2457 | 0.2354 | 3.4% |
| symmetric | 0.001 | 0 | 0.1470 | 0.1486 | -0.4% |
| symmetric | 0.001 | 0.5 | 0.1602 | 0.1578 | 1.8% |

## Ordinary-case repeat check

The prior change had a small ordinary-case slowdown. The final regular/planet fixtures were repeated for 128 balanced pairs per condition. This run shows improvement or near-neutral behavior; the full table is retained rather than selecting only favorable tolerances. The separate symmetric fixture is close to neutral at rtol=1e-3, so no universal ordinary-case improvement is claimed.

| Case | rtol | u | Before ms | After ms | Paired time reduction |
|---|---:|---:|---:|---:|---:|
| planet | 0.0001 | 0 | 0.1416 | 0.1383 | 2.1% |
| planet | 0.0001 | 0.5 | 0.1578 | 0.1518 | 3.4% |
| planet | 0.001 | 0 | 0.0930 | 0.0917 | 2.6% |
| planet | 0.001 | 0.5 | 0.0950 | 0.0930 | 1.7% |
| regular | 0.0001 | 0 | 0.1359 | 0.1298 | 5.4% |
| regular | 0.0001 | 0.5 | 0.1505 | 0.1418 | 6.1% |
| regular | 0.001 | 0 | 0.1115 | 0.1019 | 7.8% |
| regular | 0.001 | 0.5 | 0.1136 | 0.1062 | 6.8% |

## Initial calls, with an explicit readiness barrier

The final startup experiment uses eight fresh-process pairs per condition, zero numerical warmups and eight distinct source coordinates. Both engines must send READY after library import and object creation before the first query is issued. Thus another worker's initialization cannot overlap the first integration timer. Library import and process creation are excluded; actual seed, caustic-tree and matrix-basis work inside each point call is included. New PreparedRegularSupport objects alone would not reset the TLS matrix cache, so fresh processes are required.

Earlier startup.json files without the readiness barrier are intermediate diagnostics; startup-ready.json is authoritative. The fourth actual resultant call still has a basis-construction expense, but it is smaller. The first-eight-query contact streams now improve about 4–7% relative to the previous accepted build. First-call times remain noisy and not every individual condition wins. This is not a claim that all one-shot calls are faster, or that setup cost has disappeared.

| Case | rtol | u | First query before ms | First query after ms | First-eight paired time reduction |
|---|---:|---:|---:|---:|---:|
| asymmetric | 0.0001 | 0 | 2.3874 | 2.2713 | 4.4% |
| asymmetric | 0.0001 | 0.5 | 2.8989 | 2.6081 | 5.5% |
| asymmetric | 0.001 | 0 | 2.4486 | 2.3078 | 5.4% |
| asymmetric | 0.001 | 0.5 | 2.6471 | 2.4911 | 5.1% |
| collinear | 0.0001 | 0 | 1.4419 | 1.3727 | 4.9% |
| collinear | 0.0001 | 0.5 | 2.0452 | 1.9513 | 5.8% |
| collinear | 0.001 | 0 | 1.1841 | 1.1042 | 5.3% |
| collinear | 0.001 | 0.5 | 1.7166 | 1.5935 | 6.6% |
| contact | 0.0001 | 0 | 1.2037 | 1.1692 | 3.8% |
| contact | 0.0001 | 0.5 | 1.9600 | 1.9070 | 5.7% |
| contact | 0.001 | 0 | 1.1614 | 1.1362 | 3.6% |
| contact | 0.001 | 0.5 | 1.3794 | 1.4076 | 7.5% |
| contact_rotated | 0.0001 | 0 | 1.2500 | 1.2069 | 5.2% |
| contact_rotated | 0.0001 | 0.5 | 1.7678 | 1.6468 | 7.2% |
| contact_rotated | 0.001 | 0 | 1.2358 | 1.1853 | 5.4% |
| contact_rotated | 0.001 | 0.5 | 1.5069 | 1.4310 | 7.6% |
| far | 0.0001 | 0 | 1.1973 | 1.1769 | 5.3% |
| far | 0.0001 | 0.5 | 1.2302 | 1.2119 | 6.5% |
| far | 0.001 | 0 | 1.2219 | 1.2228 | 4.2% |
| far | 0.001 | 0.5 | 1.2568 | 1.2268 | 5.4% |
| large | 0.0001 | 0 | 1.6885 | 1.5589 | 2.6% |
| large | 0.0001 | 0.5 | 2.9437 | 2.6522 | 7.9% |
| large | 0.001 | 0 | 1.4737 | 1.3902 | 4.3% |
| large | 0.001 | 0.5 | 2.0147 | 1.8517 | 7.5% |
| planet | 0.0001 | 0 | 1.1953 | 1.1817 | 0.5% |
| planet | 0.0001 | 0.5 | 1.2191 | 1.2401 | 1.1% |
| planet | 0.001 | 0 | 1.1056 | 1.1028 | 0.0% |
| planet | 0.001 | 0.5 | 1.1811 | 1.1732 | 1.0% |
| regular | 0.0001 | 0 | 1.2855 | 1.2664 | 2.9% |
| regular | 0.0001 | 0.5 | 1.0445 | 1.0441 | 2.7% |
| regular | 0.001 | 0 | 1.0270 | 1.0576 | 2.3% |
| regular | 0.001 | 0.5 | 1.2586 | 1.2441 | 4.8% |
| symmetric | 0.0001 | 0 | 0.9200 | 0.9295 | -0.5% |
| symmetric | 0.0001 | 0.5 | 0.9702 | 0.9984 | 1.0% |
| symmetric | 0.001 | 0 | 0.8777 | 0.8886 | 1.0% |
| symmetric | 0.001 | 0.5 | 0.8572 | 0.8795 | 0.6% |

## Matched VBM comparison: Singlepoly, Multipoly, Nopoly

Same five-engine protocol as the prior report: before/current algebraic and the three VBM methods, identical physical geometry, source, rho, u and relative tolerance, with absolute tolerance/Tol zero. Five original-source warmups followed by 16 measured fresh-source evaluations per engine, rotated/reversed engine order. Timers include each direct point integrator, with VBM SetLensGeometry outside the timer as before. Values below are median milliseconds. The workload differs from the two-engine A/B run, so do not combine these absolute times with those medians or historical timings.

VBM samples and annuli are saved in matched-vbm.json. Its reported annuli counter is zero here; that field does not prove that LD used one annulus. Both uniform and LD APIs are invoked directly with the requested profile.

| Case | rtol | u | Before | Algebraic | Singlepoly | Multipoly | Nopoly |
|---|---:|---:|---:|---:|---:|---:|---:|
| asymmetric | 0.0001 | 0 | 1.3389 | 1.2715 | 0.5203 | 0.7051 | 1.6184 |
| asymmetric | 0.0001 | 0.5 | 1.5927 | 1.4327 | 11.4609 | 16.6280 | 37.4488 |
| asymmetric | 0.001 | 0 | 1.1125 | 1.0426 | 0.2273 | 0.3010 | 0.6941 |
| asymmetric | 0.001 | 0.5 | 1.4386 | 1.3701 | 2.9008 | 3.7968 | 8.8150 |
| collinear | 0.0001 | 0 | 0.9326 | 0.8803 | 0.8199 | 1.0644 | 2.4720 |
| collinear | 0.0001 | 0.5 | 1.3446 | 1.2002 | 25.0300 | 32.4869 | 74.3068 |
| collinear | 0.001 | 0 | 0.7551 | 0.7510 | 0.3506 | 0.4359 | 1.0812 |
| collinear | 0.001 | 0.5 | 1.0884 | 0.9386 | 4.8147 | 6.1145 | 16.1481 |
| contact | 0.0001 | 0 | 0.9197 | 0.8979 | 1.1160 | 1.4579 | 3.4480 |
| contact | 0.0001 | 0.5 | 1.3942 | 1.3269 | 48.9062 | 66.3274 | 165.2027 |
| contact | 0.001 | 0 | 0.8968 | 0.8896 | 0.6709 | 0.8750 | 1.9876 |
| contact | 0.001 | 0.5 | 1.0628 | 1.0206 | 10.8770 | 14.6871 | 35.3368 |
| contact_rotated | 0.0001 | 0 | 1.1985 | 1.1120 | 1.2650 | 1.6519 | 4.0067 |
| contact_rotated | 0.0001 | 0.5 | 1.6590 | 1.5557 | 66.1443 | 85.3692 | 199.9591 |
| contact_rotated | 0.001 | 0 | 0.9298 | 0.8605 | 0.6120 | 0.7838 | 1.9078 |
| contact_rotated | 0.001 | 0.5 | 1.2625 | 1.1904 | 14.3189 | 18.6563 | 45.1688 |
| far | 0.0001 | 0 | 0.1036 | 0.0924 | 0.2764 | 0.3542 | 0.5076 |
| far | 0.0001 | 0.5 | 0.1119 | 0.0980 | 0.2871 | 0.3690 | 0.5188 |
| far | 0.001 | 0 | 0.0736 | 0.0621 | 0.1698 | 0.2219 | 0.3126 |
| far | 0.001 | 0.5 | 0.0766 | 0.0669 | 0.1773 | 0.2296 | 0.3241 |
| large | 0.0001 | 0 | 1.4015 | 1.3726 | 1.1870 | 1.5232 | 3.5762 |
| large | 0.0001 | 0.5 | 2.5076 | 2.3810 | 68.2286 | 89.2596 | 214.9993 |
| large | 0.001 | 0 | 1.0624 | 1.0314 | 0.5931 | 0.7539 | 1.7726 |
| large | 0.001 | 0.5 | 1.5985 | 1.4462 | 13.9343 | 18.4329 | 44.8194 |
| planet | 0.0001 | 0 | 0.1687 | 0.1654 | 0.4181 | 0.5384 | 1.2150 |
| planet | 0.0001 | 0.5 | 0.1740 | 0.1704 | 0.5800 | 0.7161 | 1.6829 |
| planet | 0.001 | 0 | 0.1088 | 0.1069 | 0.2217 | 0.2917 | 0.6500 |
| planet | 0.001 | 0.5 | 0.1166 | 0.1131 | 0.2299 | 0.2961 | 0.6512 |
| regular | 0.0001 | 0 | 0.1611 | 0.1536 | 0.2863 | 0.4051 | 0.6921 |
| regular | 0.0001 | 0.5 | 0.1408 | 0.1293 | 0.2252 | 0.3160 | 0.5431 |
| regular | 0.001 | 0 | 0.1248 | 0.1174 | 0.1556 | 0.2183 | 0.3707 |
| regular | 0.001 | 0.5 | 0.1356 | 0.1276 | 0.1623 | 0.2292 | 0.3976 |
| symmetric | 0.0001 | 0 | 0.2013 | 0.1991 | 0.3398 | 0.8330 | 0.6082 |
| symmetric | 0.0001 | 0.5 | 0.2464 | 0.2431 | 1.9865 | 5.0673 | 3.4490 |
| symmetric | 0.001 | 0 | 0.1655 | 0.1584 | 0.1850 | 0.4475 | 0.3222 |
| symmetric | 0.001 | 0.5 | 0.1500 | 0.1456 | 0.4373 | 1.1793 | 0.7523 |

## Remaining contact-uniform gap

A separate 64-evaluation, balanced five-engine run uses additive means for contact/u=0/rtol=1e-3. Only this table should be used for this stage decomposition.

| Mean ms | Before | After |
|---|---:|---:|
| Support preparation | 0.322441 | 0.303816 |
| Fallback | 0.000051 | 0.000058 |
| Flux integration | 0.596725 | 0.596811 |
| Total algebraic | 0.919217 | 0.900685 |
| Singlepoly | 0.662015 | 0.662015 |
| Difference from Singlepoly | 0.257202 | 0.238670 |

Multipoly: 0.862973 ms; Nopoly: 1.993050 ms. Support preparation fell 5.8% in this run. Uniform flux time is unchanged, as expected from a value-bound helper used only by LD. Contact uniform at 1e-3 still loses to Singlepoly; the total 30% reduction target remains unmet.

## Work counters

Center-source fifth-call counts at rtol=1e-3/u=0, comparing before/current point executables. Work counters only: executable diagnostic I/O timings are not used for speed claims.

| Case | Resultant cells before | After | Physical/branch boxes before | After |
|---|---:|---:|---:|---:|
| asymmetric | 256 | 256 | 156 | 144 |
| collinear | 133 | 81 | 54 | 32 |
| contact | 75 | 43 | 0 | 0 |
| contact_rotated | 75 | 43 | 0 | 0 |
| far | 0 | 0 | 24 | 16 |
| large | 79 | 39 | 0 | 0 |
| planet | 0 | 0 | 56 | 56 |
| regular | 0 | 0 | 42 | 36 |
| symmetric | 0 | 0 | 58 | 58 |

## Isolated LD check and rejected variants

magnitude-only/paired-profiles.json compares the value-magnitude helper against the immediately preceding global-first prototype, before the ordinary-cover changes. Uniform time is close to unchanged. LD time falls about 4–5% in contact and 7–9% in large, depending on source/profile. The final whole-change comparison is the A/B table above; those isolated percentages must not be added arithmetically to it.

The generic polynomial-tail normalization prototype was slower (degree-probe.json) and was reverted. The adopted fixed-basis code inspects tails once and shares symbolic products instead. A naive full-source-cover prototype made symmetric cases slower; the final implementation adds the existing cheap-model gate, shared disk preparation and per-arc tightening rather than introducing a symmetric-case special rule. Its source snapshot is retained as regular-full-cover-unrefined.hpp; intermediate timing observations are diagnostic, not final speed claims.

No source-neighborhood completeness cache from the earlier 10-ms research prototype is installed. No tolerance was loosened to obtain these gains. The coefficient tier, quad promotion policy, gradient-only arithmetic and failure gates retain their existing roles. Full mixed-trajectory/fit performance is not established by this direct-point benchmark.

## Reproduction and provenance

The canonical checkout, branch/HEAD, compiler, input/reference hashes, native module and baseline/candidate library hashes are recorded in provenance.json. The baseline kernel hash matches the prior accepted triple_resultant_sparse candidate. source.patch contains only this turn's eight source/test files and passes reverse apply-check against the current files. Unrelated dirty/untracked work was preserved; no commit or push was performed.

Build the configured native checkout with TMPDIR set to this directory's compiler-tmp and -j2, then run kernel-build.sh and build-point.sh. The existing scripts reproduce the paired, startup-ready, VBM, heldout, LD-extreme and API checks. Debug/assertion executables and logs are retained. SHA256SUMS covers the local artifacts; input/reference symlink targets have separate hashes in provenance.json. No UBSan pass is claimed because the installed runtime was unavailable in the preceding experiment.
