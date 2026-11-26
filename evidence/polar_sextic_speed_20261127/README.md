# Triple polar boundary coefficients and angular isolation

The finite-source solver integrates image arc lengths and brightness over radius. This change collects the primary-frame lens factors and the four angular harmonics directly into the seven real coefficients of the sextic in tan(theta/2). In each compact angular chart, a common strict sign of the power-coefficient prefix sums excludes a root by Abel summation. At the quadratic level of the derivative recursion we use interval enclosures of q/a and c/q. Ambiguous, coincident or edge-touching roots retain recursive isolation. The physical endpoint proofs share inverse squares and use exact power-of-two scaling of binary64 interval endpoints with overflow handling.

## Timing

Release builds on rogue1 from experiment/algebraic_boundary at 7038d6c2 plus working-tree changes. Starting extension SHA-256 e620107c93ab073aab7a0fd6d67a1b0d8afaee043e70ea306a8b38e2a7be5b54. Updated extension SHA-256 5bc6573cfa13f549dea876e4c3d81a45d48a724bbd1959a3f0333cb1511a846c. The public eight-mode driver in ../eight_mode_speed_20261127/bench.py asserts the native import path. Two offsets supply 16 fixtures per lens, with four two-epoch calls per fixture. Each row in timing.tsv records one paired repetition. Relative tolerance was 1e-4 and absolute tolerance zero. CPU 2 measured fixed lenses, CPU 11 measured changing s and q between calls. These are single-thread CPU timings for the public calls, excluding construction. Run 2 reversed the order.

For the independent VBM MultiPoly comparison we used the original three epochs of each of fixtures 568 and 889, both tolerances 1e-4, and the median of three paired warm elapsed totals. Uniform values took 17.82 ms before and 16.30 ms after, with VBM at 6.97 ms. At limb coefficient 0.5 the corresponding times were 18.64, 17.28 and 109.04 ms.

## Accuracy

All 7,072 archive evaluations succeeded and preserved status, support and validity flags (1,768 central epochs with uniform and archived source profiles, values and gradients). Maximum relative magnification change was 8.90e-8. Maximum per-epoch gradient change in infinity norm divided by max(1, the baseline gradient infinity norm) was 2.71e-5. Across the timed fixed and varying fixtures the maxima were 7.77e-11 for values and 1.06e-9 for gradients. Binary static, circular and Keplerian parity matched exactly in 468 epoch/mode evaluations.

An independent strict uniform VBM reference at 64 distinct timing inputs gave maximum relative magnification errors of 4.22e-5 for values and 2.67e-7 for values returned with gradients. The two 41-epoch contact tracks gave maximum error 4.65e-5 across both brightness profiles and both value modes. These are sampled accuracy measurements.

The largest archive derivative change occurred at fixture 1089, uniform dmu/dsep2. The archive-order result moved from -65670.7058 to -65668.9269. Independent VBM uniform finite differences at relative tolerance 1e-8 and steps 6.25e-8 and 3.125e-8 gave -65675.2396 and -65671.8396. Quadratic extrapolation gives -65670.7062. An isolated call to the new build gave -65670.7058 at relative tolerance 1e-4. The archive-order derivative thus lost approximately 2.71e-5 relative accuracy at this particular contact.

For 5,000 randomized radii and geometries, all 35,000 binary64 coefficient intervals enclosed the coefficients of the original factored implementation evaluated in binary128. Seven rebuilt C++ suites passed, including angular isolation, physical endpoint proofs, point signs, the resultant and regular support. The triple Python suite gave 259 passes and the same nine failures in test_triple_lens_point_source.py as the baseline.

The parent checkout contains build-accuracy-20261127/polar-jet with the baseline native package, measure.sh, validate.sh, archived comparisons, finite differences, coefficient enclosure check and build/test logs. The strict references and the original VBM finite-difference driver are in build-accuracy-20261127/uniform-search.
