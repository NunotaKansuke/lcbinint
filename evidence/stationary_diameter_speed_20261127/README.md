# Triple stationary-limb search timing

Measured on rogue1, CPU 2, one thread. The before build is the previous optimized experiment at build-accuracy-20261127/radial-speed/adopted. The after build is build-accuracy-20261127/algebraic. Both used the working tree based on 7038d6c2. SHA-256 of the Python native modules, before then after, was 01fc10968154065f03d13177e9bacde4dcc18af49f9657c6649eeb186b557c596 and cff43bf4bdee18df05c274feb06307f6522028ba32738dd358c45fbef7539760.

The driver was ../eight_mode_speed_20261127/bench.py with offsets 0 and 1, absolute tolerance zero and relative tolerance 1e-4. Each mode spans 16 fixtures, four two-epoch calls per fixture. timing.tsv gives mean thread CPU time per epoch for each of three runs. Speedup divides the before median by the after median. The driver's fixed order runs values before gradients and can warm the event cache. Compare like modes across builds.

All 512 triple epoch/mode evaluations had identical magnifications, gradients and reported flags across builds. A separate stratified sample of 48 triple curves (288 epoch/mode evaluations) had maximum relative magnification change 1.93e-14 and maximum gradient change 7.42e-14 in the per-epoch infinity norm divided by max(1, baseline gradient infinity norm). The sampled status, gradient validity, support and integration flags agreed. The largest magnification difference against VBM with Tol=1e-8 and RelTol=1e-6 was 3.97e-5 relative across 32 unique epoch/profile inputs.

The event-plan probe tested seven source positions for each of eight geometries. Completion and event counts agreed in all 56 comparisons. The physical subdivision box count fell from 58,724 to 45,011. The triple C++ event, surface, resultant and integrator tests passed, as did 98 Python triple regressions.

A separate public value-only comparison used cases 568 and 889 at their three original epochs and Tol=RelTol=1e-4 on both backends. For each profile we summed the two curve timings, using five warm repetitions per curve and three runs. Median six-epoch totals were 27.80 ms versus 6.93 ms (experiment versus VBM, uniform) and 27.80 ms versus 108.93 ms (linear limb darkening, coefficient 0.5).

Raw public timings, held-out outputs, independent references, event probes and test logs are under build-accuracy-20261127/projection-speed in the parent checkout.
