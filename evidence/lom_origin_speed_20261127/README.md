# Binary LOM radial-origin selection

The baseline is the experiment worktree at 7038d6c2e7 including the previous eight-mode optimizations. The native extension SHA-256 was 1880947a8cdf55d245182c9d02f6ce69027e7c3e9972c36ed441ef0d093b84eb before and de84a34609ea07e54c4ba1b2c67c2e25516d9ea6ac7e85df1fc3c0eb65d8db53 after.

Run bench.py BUILD from the parent checkout with the matching Python interpreter. The timing table reports elapsed milliseconds per epoch, median of five paired passes alternating order on CPU 2 with OMP and OpenBLAS threads set to one. Each row includes five mass ratios from 0.001 through 1000 and 23 epochs per ratio. Both absolute and relative tolerances are 1e-6. An untimed call warms the trajectory transform. The packaged records are the measured speeds. Full compared outputs and the two selector timing logs are in build-accuracy-20261127/lom-origin in the parent checkout.

On 1,800 independent lens geometries the median selector CPU time fell from 59.13 ms to 3.76 ms. Both selectors picked the heavy origin 718 times. All decisions agreed with full scans, including 420 additional steps bisecting the decision boundary. selector.cpp also checks the full numerical spread after a cached threshold-only query.

All 920 LOM epoch/mode results per pass matched for magnitude, all gradient entries and status fields. Of these, 548 had OK status and 372 retained COVERAGE_UNCERTIFIED. The eight static binary/triple modes matched on all 512 epoch/mode results of cohort 0 in the previous eight-mode benchmark.

The 30 tests in tests/regression/test_algebraic_boundary_orbit.py passed with the matching JAX FFI binding enabled for validation. They include finite differences, Kepler references, parallax, repeated and unsorted epochs, and separation crossing unity. On an additional independent uniform-source VBM Kepler comparison with the five benchmark mass ratios, 13 epochs per ratio, and tolerance 1e-7, the maximum relative value discrepancy was 7.21e-9. These sampled comparisons do not establish full image coverage.
