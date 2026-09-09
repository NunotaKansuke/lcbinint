#pragma once

// Residue-free six-dimensional view of the equation-derived GM kernel.
//
// This is an experiment on top of the true eta connection.  It keeps the
// physical basis
//
//   psi = (eta0, eta1, eta2, eta4-b1 eta3, eta5-b2 eta3, eta6-b3 eta3)
//
// and assembles C_psi = (W' + W C_eta)[:, (0,1,2,4,5,6)].  It deliberately
// does not use the K-rule / Chebyshev packet.  The wrapper is useful for
// measuring the six-dimensional algebra and its closure residual; a direct
// flux-priority companion basis is a separate, stronger experiment because
// C_psi itself is singular at a chart_p4 degree drop.

#include <array>
#include <cmath>
#include <limits>

#include "lcbinint/magnification/holonomic/gm_connection.hpp"

namespace lcbinint::holonomic {

constexpr int kGmPsiDim = 6;
constexpr std::array<int, kGmPsiDim> kGmPsiEtaIndex = {0, 1, 2, 4, 5, 6};

template <int Order, class Scalar>
struct GmPsiJet {
    using Series = GmSeries<Order, Scalar>;

    bool ok = false;
    bool finite = false;
    bool closure_ok = false;
    double closure_residual = std::numeric_limits<double>::infinity();
    std::array<Series, 3> b{};
    std::array<Series, 3> bprime{};
    std::array<std::array<Series, kGmPsiDim>, kGmPsiDim> C{};
    std::array<std::array<Series, kGmEtaDim>, kGmPsiDim> M{};
};

template <int Order, class Scalar>
inline GmPsiJet<Order, Scalar> gm_residue_free_jet(
    const GmConnectionJet<Order, Scalar>& eta) {
    using Series = GmSeries<Order, Scalar>;
    GmPsiJet<Order, Scalar> out;
    if (!eta.ok) return out;

    const Series q5 = eta.q.c[5];
    const Series q6 = eta.q.c[6];
    const Series q7 = eta.q.c[7];
    const Series q8 = eta.q.c[8];
    const Series two_q8 = q8 * Series(Scalar(2));
    const Series q8_2 = q8 * q8;
    const Series q8_3 = q8_2 * q8;
    const Series qR5 = eta.qR.c[5];
    const Series qR6 = eta.qR.c[6];
    const Series qR7 = eta.qR.c[7];
    const Series qR8 = eta.qR.c[8];

    auto quotient_derivative = [](const Series& numerator,
                                  const Series& numerator_R,
                                  const Series& denominator,
                                  const Series& denominator_R) {
        return (numerator_R * denominator - numerator * denominator_R) /
               (denominator * denominator);
    };

    const Series b1_bal = -q7 / two_q8;
    const Series b2_bal =
        -q6 / two_q8 + gm_series_scale_scalar(
            (q7 * q7) / q8_2, Scalar(3.0 / 8.0));
    const Series b3_bal =
        -q5 / two_q8 + gm_series_scale_scalar(
            (q7 * q6) / q8_2, Scalar(3.0 / 4.0)) -
        gm_series_scale_scalar((q7 * q7 * q7) / q8_3,
                               Scalar(5.0 / 16.0));
    const Series b1prime_bal = -gm_series_scale_scalar(
        quotient_derivative(q7, qR7, q8, qR8), Scalar(0.5));
    const Series b2prime_bal =
        -gm_series_scale_scalar(
            quotient_derivative(q6, qR6, q8, qR8), Scalar(0.5)) +
        gm_series_scale_scalar(
            quotient_derivative(q7 * q7, Series(Scalar(2)) * q7 * qR7,
                                q8_2, Series(Scalar(2)) * q8 * qR8),
            Scalar(3.0 / 8.0));
    const Series b3prime_bal =
        -gm_series_scale_scalar(
            quotient_derivative(q5, qR5, q8, qR8), Scalar(0.5)) +
        gm_series_scale_scalar(
            quotient_derivative(q7 * q6, qR7 * q6 + q7 * qR6,
                                q8_2, Series(Scalar(2)) * q8 * qR8),
            Scalar(3.0 / 4.0)) -
        gm_series_scale_scalar(
            quotient_derivative(q7 * q7 * q7,
                                Series(Scalar(3)) * q7 * q7 * qR7,
                                q8_3, Series(Scalar(3)) * q8_2 * qR8),
            Scalar(5.0 / 16.0));
    const std::array<Series, 3> b_bal = {b1_bal, b2_bal, b3_bal};
    const std::array<Series, 3> bp_bal = {
        b1prime_bal, b2prime_bal, b3prime_bal};

    // eta.C is expressed in the original t basis, while eta.q is held in
    // the fixed balanced chart used during its construction.  The residue
    // coefficients transform as b_raw,j = c^j b_bal,j.
    for (int i = 0; i < 3; ++i) {
        const double power = std::pow(eta.chart_scale, double(i + 1));
        out.b[i] = gm_series_scale_scalar(b_bal[i], Scalar(power));
        out.bprime[i] = gm_series_scale_scalar(bp_bal[i], Scalar(power));
    }

    for (int col = 0; col < kGmEtaDim; ++col) {
        for (int row = 0; row < 3; ++row) out.M[row][col] = eta.C[row][col];
        for (int row = 0; row < 3; ++row) {
            const int psi_row = row + 3;
            const int eta_row = kGmPsiEtaIndex[psi_row];
            out.M[psi_row][col] = eta.C[eta_row][col] -
                                   out.b[row] * eta.C[3][col];
            if (col == 3)
                out.M[psi_row][col] = out.M[psi_row][col] - out.bprime[row];
        }
    }

    double residual = 0.0;
    double scale = 1.0;
    for (int row = 0; row < kGmPsiDim; ++row) {
        const Series closure =
            out.M[row][3] + out.b[0] * out.M[row][4] +
            out.b[1] * out.M[row][5] + out.b[2] * out.M[row][6];
        residual = std::max(residual, gm_series_max_abs(closure));
        for (int col = 0; col < kGmEtaDim; ++col)
            scale = std::max(scale, gm_series_max_abs(out.M[row][col]));
        for (int col = 0; col < kGmPsiDim; ++col)
            out.C[row][col] = out.M[row][kGmPsiEtaIndex[col]];
    }
    out.closure_residual = residual / scale;

    out.finite = true;
    for (const auto& b : out.b)
        out.finite = out.finite && gm_series_finite(b);
    for (const auto& row : out.M)
        for (const Series& x : row) out.finite = out.finite && gm_series_finite(x);
    for (const auto& row : out.C)
        for (const Series& x : row) out.finite = out.finite && gm_series_finite(x);
    out.closure_ok = std::isfinite(out.closure_residual) &&
                     out.closure_residual <= gm_quality_limit(Scalar(0));
    out.ok = out.finite && out.closure_ok;
    return out;
}

template <int Order, class Scalar>
inline bool gm_psi_taylor_transport(
    const GmPsiJet<Order, Scalar>& connection,
    const std::array<Scalar, kGmPsiDim>& seed, double h,
    std::array<Scalar, kGmPsiDim>& result) {
    if (!connection.ok || !std::isfinite(h)) return false;
    std::array<std::array<Scalar, kGmPsiDim>, Order + 1> z{};
    z[0] = seed;
    for (int n = 0; n < Order; ++n) {
        for (int i = 0; i < kGmPsiDim; ++i) {
            Scalar sum = Scalar(0);
            for (int j = 0; j <= n; ++j)
                for (int k = 0; k < kGmPsiDim; ++k)
                    sum = sum + connection.C[i][k].c[j] * z[n - j][k];
            z[n + 1][i] = sum / Scalar(n + 1);
        }
    }
    result.fill(Scalar(0));
    double hp = 1.0;
    for (int n = 0; n <= Order; ++n) {
        for (int i = 0; i < kGmPsiDim; ++i)
            result[i] = result[i] + z[n][i] * Scalar(hp);
        hp *= h;
    }
    for (const Scalar& x : result)
        if (!gm_finite(x)) return false;
    return true;
}

template <int Order, class Scalar>
inline bool gm_eta_to_psi(
    const GmConnectionJet<Order, Scalar>& eta,
    const std::array<Scalar, kGmEtaDim>& eta_value,
    std::array<Scalar, kGmPsiDim>& psi_value) {
    const auto psi = gm_residue_free_jet(eta);
    if (!psi.ok) return false;
    psi_value = {eta_value[0], eta_value[1], eta_value[2],
                 eta_value[4] - psi.b[0].c[0] * eta_value[3],
                 eta_value[5] - psi.b[1].c[0] * eta_value[3],
                 eta_value[6] - psi.b[2].c[0] * eta_value[3]};
    return true;
}

}  // namespace lcbinint::holonomic
