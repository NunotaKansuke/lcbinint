# What can be accelerated in the sums?

These are code-derived mathematical proposals, not implemented speedups.

## Finite-source magnification as a convolution

For fixed lens geometry and a normalized source brightness profile w, finite-source magnification is

mu_fs(zeta_s) = integral mu_ps(zeta_s + eta) w(eta) d^2 eta.

For a centered radial profile w, this is a convolution with the point-source magnification field. Fourier convolution is mathematically possible. Its practical cost includes constructing a sufficiently accurate magnification map, especially near caustics, before evaluating that convolution. A changed lens geometry changes the map. The current single-source seeded image-boundary integrator does not already possess this map; generating one solely for a single evaluation may cost much more than the current local integration. This observation does not establish that a Fourier map approach would be faster for this workload.

## Present weighted quadrature sums

`estimate` computes q = sum_k w_k f_k and separate inner/geometry/roundoff ledgers. This is a dot product per adaptive panel. It does not compute a family of translated outputs, so replacing one such O(N) sum by FFT convolution is not a computational win. Contiguous gathering and SIMD, or reusing unchanged coarse-level information, are more direct candidates. Compensated summation and the ledger semantics must be preserved or independently validated. Cache invalidation is needed for same-node angular upgrades and changed panel maps.

## Dense interpolation in the error estimator

The error estimator predicts each newly inserted odd-index value from all coarse even-index samples. It applies a dense, precomputed interpolation matrix and costs O(N^2) per level. The matrix is not simply Toeplitz in the physical radial coordinate, but the node angles are uniformly spaced and the samples are polynomial interpolation data at Chebyshev second-kind roots.

Let coarse M = 2^(level-1), theta_j = pi*j/M (j=1..M-1), f_j = f(cos(theta_j)). The same degree-(M-2) interpolant can be written

p(cos(theta)) = sum_(k=1..M-1) a_k sin(k*theta) / sin(theta),
a_k = (2/M) sum_(j=1..M-1) f_j sin(theta_j) sin(k*theta_j).

Thus a DST-I of f_j*sin(theta_j), followed by zero-padding the sine coefficients and evaluating at the finer angle grid, gives the same polynomial in exact arithmetic. This can reduce the prediction work to O(N log N), including via an FFT implementation of the sine transforms. The existing barycentric interpolation weights (-1)^j sin(theta_j)^2 in `nested_fejer2.hpp` are consistent with that polynomial.

This is a mathematical equivalence, not floating-point equivalence: division by small sin(theta) near the endpoints, transform roundoff, and cancellation can affect the detail estimate and hence the adaptive refinement decision. Compare predictions, detail norms, chosen nodes, resulting magnification and the existing independent reference contracts before accepting it. Most panels can be small (maximum current grid has255 interior samples), so transform/setup overhead can dominate. A measured size crossover or a fixed small transform is preferable to assuming FFT wins at every level.

## Priority

First profile the error estimator's dense prediction loop separately from quadrature and ledger sums, and inspect actual panel-level frequencies. The full estimator previously measured roughly0.017 ms, which includes all these regions: even eliminating it cannot save more than that measured stage under matched inputs. A contiguous value-only gather with the same compensated arithmetic is a bounded first trial; DST-I is the larger-grid mathematical option. No convolution or transform change is enabled in this pass.
