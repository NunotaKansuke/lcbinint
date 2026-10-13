#pragma once
// Gauss-Chebyshev 1st-kind rule, shared by the angular (n=64) and
// full-circle passes.  int_{-1}^{1} f dx ~ sum w_i f(x_i), with
// w_i = (pi/n) sqrt(1 - x_i^2) carrying the sqrt weight (jacobian._cheb1).
#include <array>
#include <cmath>
#include <vector>

namespace lcbinint::holonomic {

template <int N>
struct Cheb1 {
    std::array<double, N> x{};
    std::array<double, N> w{};
    Cheb1() {
        const double pi = 3.14159265358979323846;
        for (int i = 0; i < N; ++i) {
            double xi = std::cos((2.0 * (i + 1) - 1.0) * pi / (2.0 * N));
            x[i] = xi;
            w[i] = (pi / N) * std::sqrt(1.0 - xi * xi);
        }
    }
};

inline const Cheb1<64>& ang_rule() {
    static const Cheb1<64> r;
    return r;
}

}  // namespace lcbinint::holonomic

namespace lcbinint::holonomic {

// Runtime-sized GC-1st-kind rule (jacobian._cheb1) for the radial pass,
// where n_r is a parameter.  w carries sqrt(1 - x^2).
struct Cheb1Dyn {
    std::vector<double> x, w;
    explicit Cheb1Dyn(int n) : x(n), w(n) {
        const double pi = 3.14159265358979323846;
        for (int i = 0; i < n; ++i) {
            double xi = std::cos((2.0 * (i + 1) - 1.0) * pi / (2.0 * n));
            x[i] = xi;
            w[i] = (pi / n) * std::sqrt(1.0 - xi * xi);
        }
    }
};

}  // namespace lcbinint::holonomic
