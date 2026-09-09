#pragma once

// Taylor transport driven by the equation-derived connection in
// gm_connection.hpp.  If Z' = C(R) Z and
// C(Rc+h) = sum_j C_j h^j, the coefficient recurrence is
//
//   Z_{n+1} = 1/(n+1) sum_{j=0}^n C_j Z_{n-j}.
//
// This is the actual holonomic transport kernel.  It is intentionally kept
// separate from the existing K-rule / Chebyshev packet code in
// holonomic_ode_transport.hpp; no packet samples are used here.

#include <array>
#include <cmath>

#include "lcbinint/magnification/holonomic/gm_connection.hpp"

namespace lcbinint::holonomic {

template <int Order, class Scalar>
inline bool gm_taylor_transport(
    const GmConnectionJet<Order, Scalar>& connection,
    const std::array<Scalar, kGmEtaDim>& seed, double h,
    std::array<Scalar, kGmEtaDim>& result) {
    if (!connection.ok || !std::isfinite(h)) return false;

    std::array<std::array<Scalar, kGmEtaDim>, Order + 1> z{};
    z[0] = seed;
    for (int n = 0; n < Order; ++n) {
        for (int i = 0; i < kGmEtaDim; ++i) {
            Scalar sum = Scalar(0);
            for (int j = 0; j <= n; ++j)
                for (int k = 0; k < kGmEtaDim; ++k)
                    sum = sum + connection.C[i][k].c[j] * z[n - j][k];
            z[n + 1][i] = sum / Scalar(n + 1);
        }
    }

    result.fill(Scalar(0));
    double hp = 1.0;
    for (int n = 0; n <= Order; ++n) {
        for (int i = 0; i < kGmEtaDim; ++i)
            result[i] = result[i] + z[n][i] * Scalar(hp);
        hp *= h;
    }
    for (const Scalar& x : result)
        if (!gm_finite(x)) return false;
    return true;
}

template <int Order>
inline bool gm_taylor_transport_dual(
    const GmConnectionJet<Order, GmDual5>& connection,
    const std::array<GmDual5, kGmEtaDim>& seed, double h,
    std::array<GmDual5, kGmEtaDim>& result) {
    return gm_taylor_transport(connection, seed, h, result);
}

}  // namespace lcbinint::holonomic
