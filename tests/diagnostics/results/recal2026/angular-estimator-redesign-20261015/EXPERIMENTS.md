# Polar estimator redesign, 2026-10-15 (in progress)

Checkout: feature/boundary-error-and-discriminant-support @ ee4d1c4e; dirty baseline retained as before.inc, before-roots.hpp and baseline static libraries. All runs CPU 0, one thread; paired runs alternate baseline/candidate with identical 16 Cartesian / 2 Polar initial bins and radial allocation 4. Timing includes seed cache misses, excludes prepared roots/caustics/LD table.

## Findings so far

- Endpoint derivative interpolation without independent angular probes produced false accepted errors. Rejected.
- Positive degree-seven physical angular quadrature with parent nine-node interpolation passed 14,432 corpus conditions. Removing obsolete quartic observation errors improves estimator consistency, but alone produces only small timing gains.
- Exact algebraic LD endpoint weighting starts with three Gauss-Chebyshev U nodes plus four independently mapped physical witnesses. This is derived from the lens equation, not an ellipse image approximation. Endpoint Newton polishing is necessary: unpolished endpoints left a non-refinable factorization error. With polishing, full corpus passes.
- An image-aligned nonuniform initial angular mesh passed 882 independent conditions but falsely accepted geom 6218 at 1e-4 and worsened per-case median timing (LD 1e-3 speedup 0.916). Rejected and removed.
- v20: use at most eight cheap Newton steps to propose stationary cuts, then certify the entire sextic partition with the existing conservative Bernstein filter and physical endpoint signs. Completeness is not inferred from Newton convergence. Independent 882: no false accepts, one unmet fold fresh-1-0.0 at 1e-5 already present in the new physical-quadrature family. Paired 128: uniform 1e-3/1e-4 speedup 1.065/1.109; LD 1e-3/1e-4 1.085/1.103. Full corpus validation pending for this exact version.
- v21: factored cubic Bernstein certificate avoids expanded sextic cancellation. Random 4000 root-count checks: 2013 certified, zero count disagreements. Full 14,432 accepted, zero false accepts/exceptions. Independent results match v20. However long-double complex construction costs more than it saves: paired LD speedup 0.827/0.843. Rejected; snapshot retained.
- v22: test converting observation residuals using the absolute integral of the nine-node interpolation remainder. This is an empirical estimator, not a rigorous derivative bound; factor eight and topology guard retained. Validation pending.

No claim of a complete error certificate or of LD 1e-3 beating VBM is made. Rare unmet cases are retained explicitly; false accepts are not silently classified as accurate. All negative prototypes and references are retained in adjacent candidate snapshots and logs.

## Continued redesign (v23-v38)

- v23 shares lens denominators between the physical map and Newton Jacobian. Full corpus passes; little timing gain.
- v24 returns a positive degree-seven quadrature for transformed isolated caps, comparing that returned rule with degree-eight interpolation. Independent 882 pass; corpus one LD 1e-4 false accept (4591). An independent LD layer-cake calculation confirms reference 90.19210704652701; discrepancy remains just over 1e-4.
- v25 halves the initial angular partition count again (stride ratio/2). Full corpus passes; independent one unmet extreme fresh-1-0.0 at 1e-5. Paired LD 1e-3/1e-4 speedup 1.084/1.148.
- v26 reuses neighboring initial stencils. Full corpus passes, but timing fails to improve. Removed.
- v27 uses a double sign filter for stationary-quintic Bernstein coefficients, retaining long double on ambiguous signs and original root polishing. 10,000 random geometries / 12,776 intervals show no sign-count disagreement. Full corpus passes. Paired LD 1.131/1.198.
- v28 removes v26 and compares matched cap quadratures with factor two. Full corpus passes; paired LD 1.134/1.170.
- v29 starts weighted radial integration at one node plus two physical witnesses, then 3/7/15/... with four witnesses after refinement. Full corpus passes; independent one unmet extreme 1e-5. Fewer maps but little wall-clock gain. Paired LD 1.133/1.187.
- v30 replaces transformed-cap comparison with closed Newton-Cotes 9 plus a Stieltjes 13-node extension (returned degree 9, reference degree 17). Uses the same 13 physical locations count, with new non-nested locations. Full corpus passes; LD 1.154/1.207.
- v31 adds order refinement before interval bisection in ordinary panels. Full corpus passes; LD 1.152/1.227. Estimator/actual median on independent cases 61 at 1e-3 and 37 at 1e-4.
- v32 returns positive Boole 5 in ordinary panels, comparing against the existing nine physical columns and retaining every flood-fill / seed observation. Removes mandatory four additional comparison columns there. Full corpus passes; LD 1.185/1.277. Estimator/actual median falls to 28.7 / 11.7, versus about 150 / 143 initially. Independent one unmet extreme 1e-5, no false accept.
- v33 reuses the already known physical Newton residual as one bracket endpoint, evaluating only the opposite endpoint. The physical sign-changing bracket and its full uncertainty remain charged. Full corpus passes; LD 1.183/1.273.
- v34 directly inherits parent-interpolant child errors without independent parent-model error. Full corpus contains false accepts (all retained in adjudicated-summary.json). LD 1.189/1.301, including a false accept in the paired subset at 1e-4. Rejected: error model fails to cover the parent interpolant.
- v35 is a source-only proposal: retain Newton proposals after an inconclusive sextic filter. Not compiled independently.
- v36 combines that proposal with returned degree-seven Richardson-Boole 9 on ordinary panels. Independent known cases pass except the extreme unmet, but full corpus has false accepts at 1e-3 and 1e-4. Paired LD 1.201/1.300; gain insufficient for the observed accuracy loss. Rejected.
- v37 restores the verified v33 ordinary rule and retains the Newton proposals for independently checked stationary continuation. Full corpus passes. Timing pending.
- v38 (in preparation): geometry-local angular bands that prove a fixed stationary-root count using Fourier Bernstein coefficients and a Taylor angular bound. Radial coordinates are normalized by the actual direction length, so rounded sin/cos norms do not invalidate the argument. Unit comparison: 2,000 random geometries, 983 constructed bands, 4,915 in-band queries, no root-count mismatch, maximum stationary-root displacement 7.35e-15. Production integration will be checked against the full recursive root solver in shadow mode.

All paired numbers here are per-case median speedups over the frozen initial source, 128 cases, with failures retained. These exploratory timings are not the final larger-sample benchmark and do not establish an LD 1e-3 win over VBM.

## v38-v48 follow-up

- v38/v39/v40 stationary-root angular bands passed unit and corpus shadow checks, but slowed the common path. Removed from execution; the self-contained rejected prototype remains in band-prototype/.
- v41 removes unused LD derivative evaluations and simplifies the first-node weighted remainder algebraically. Full corpus passed.
- v42 uses a flat sorted angular observation index, fixed 3/7-node constants and stack storage. Full corpus passed.
- v43 selectively recomputes ambiguous sextic Bernstein coefficients in long double, reusing the original stationary solve if unresolved. Shadow corpus: 14,432 accepted, zero false accept or root-count disagreement. Independent 882: 881 accepted, one extreme 1e-5 unmet, zero false accepts. The unmatched extreme actually differs by 8.71e-6 but retains its unmet status. Matched final 256-case LD per-case median speedup 1.2623 / 1.3579 at 1e-3 / 1e-4. Direct retimed VBM/Polar on 128 LD: 0.8705 / 3.1578. Thus LD 1e-3 does not yet beat VBM.
- v44 reuses a non-nested physical observation in a positive seven-node angular extension. One corpus false accept at LD 1e-3; less than 1% additional median speedup versus v43. Rejected for the accuracy/complexity/performance tradeoff, with the counterexample retained.
- v45 admits initial corrected Simpson/Boole panels with factor 2 instead of 64 before their higher-order comparison. 57 false accepts, only about 2% incremental speedup. Rejected: smoothness cannot be inferred from those initial points.
- v46 adds connected-component coarse-grid peak witnesses to v45. Corrected build still contains many false accepts. The first build failed; audits inadvertently run against the preceding binary are explicitly quarantined as *-invalid-build and v46-invalid-run.json, excluded from evidence.
- v47 transports cuts already inside physical image intervals using the two boundary slopes, then proves the resulting complete sextic partition. Full corpus and independent checks pass with the same one 1e-5 unmet. Small timing change; no case-specific rescue.
- v48 bounds small LD contributions using 0 <= mu <= 1 on the physically solved radial interval. Returns the midpoint of [0, c*(r_end^2-r_start^2)/2] and charges its full half-width; skips weighted nodes only when that uncertainty fits the absolute radial allocation. The positive midpoint retains flood-fill support. Initial angular support sets the allocation density; final summed radial uncertainty remains in acceptance. Full 14,432 and independent 882 match v47. Final matched timing is pending; candidate is not selected solely from validation success.

All variants, including negative trials, are preserved. No claim of a mathematically complete error certificate is made. No ellipse image model, per-case rescue, or error-based timing winner is introduced.

## Selected closeout

v48 paired256 did not improve on v43: LD1.253 / 1.337 versus v43 1.262 / 1.358. Additional transported-cut and bounded-small-LD branches removed by restoring v43 snapshots. Rebuilt native exactly reproduces SHA 7cc97d0e14110fbde37867f2e35ea7c0ec493316d6670bdee776423f9c21017a. REPORT.md and final-identity.json identify the selected source and evidence. LD1e-3 VBM goal remains unmet.
