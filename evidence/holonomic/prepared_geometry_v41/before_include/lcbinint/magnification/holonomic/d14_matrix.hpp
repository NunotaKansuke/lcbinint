#pragma once

// Research-only 2x2 matrix-polynomial representation of D14.
//
// With C=C3(v), G=G4(v), and Z=Z3(v), define
//
//   A = C^2 - 3 v G,
//   B = C G + 36 v Z,
//   E = G^2 + 12 C Z,
//   M(v) = [[2 E, B], [B, 2 A]].
//
// The exact identity is det(M) = 3 D14 / 4096.  The production solver keeps
// its existing scalar structured Aberth path.  This header is intentionally
// standalone so a QZ/linearization experiment cannot silently become a
// production dependency.

#include <array>

#include <quadmath.h>

#include "lcbinint/magnification/holonomic/d14_structure.hpp"

namespace lcbinint::holonomic {
namespace re_detail {

using D14MatrixQf = std::array<std::array<std::array<__float128, 2>, 2>, 9>;

inline std::array<__float128, 15> matrix_poly_mul(
    const std::array<__float128, 15>& x, int dx,
    const std::array<__float128, 15>& y, int dy) {
    std::array<__float128, 15> out{};
    for (int i = 0; i <= dx; ++i)
        for (int j = 0; j <= dy && i + j < 15; ++j)
            out[i + j] += x[i] * y[j];
    return out;
}

inline D14MatrixQf d14_matrix_build(const D14StructQf& s) {
    using q = __float128;
    std::array<q, 15> c{}, g{}, z{};
    for (int i = 0; i < 4; ++i) {
        c[i] = s.c3[i];
        z[i] = s.z3[i];
    }
    for (int i = 0; i < 5; ++i) g[i] = s.g4[i];

    const auto c2 = matrix_poly_mul(c, 3, c, 3);
    const auto cg = matrix_poly_mul(c, 3, g, 4);
    const auto g2 = matrix_poly_mul(g, 4, g, 4);
    const auto cz = matrix_poly_mul(c, 3, z, 3);

    std::array<q, 15> a{}, b{}, e{};
    for (int i = 0; i <= 6; ++i) {
        a[i] = c2[i];
        if (i >= 1) a[i] -= q(3) * g[i - 1];
    }
    for (int i = 0; i <= 7; ++i) {
        b[i] = cg[i];
        if (i >= 1) b[i] += q(36) * z[i - 1];
    }
    for (int i = 0; i <= 8; ++i) {
        e[i] = g2[i] + q(12) * cz[i];
    }

    D14MatrixQf out{};
    for (int i = 0; i < 9; ++i) {
        out[i][0][0] = q(2) * e[i];
        out[i][0][1] = b[i];
        out[i][1][0] = b[i];
        out[i][1][1] = q(2) * a[i];
    }
    return out;
}

inline std::array<__float128, 15> d14_matrix_determinant(
    const D14MatrixQf& m) {
    std::array<__float128, 15> lhs{}, rhs{};
    for (int i = 0; i < 9; ++i) {
        lhs[i] = m[i][0][0];
        rhs[i] = m[i][0][1];
    }
    std::array<__float128, 15> aa{}, bb{};
    for (int i = 0; i < 9; ++i) {
        aa[i] = m[i][1][1];
        bb[i] = m[i][1][0];
    }
    const auto first = matrix_poly_mul(lhs, 8, aa, 8);
    const auto second = matrix_poly_mul(rhs, 8, bb, 8);
    std::array<__float128, 15> out{};
    for (int i = 0; i < 15; ++i) out[i] = first[i] - second[i];
    return out;
}

inline __float128 d14_matrix_eval(const D14MatrixQf& m, __float128 v) {
    __float128 a00 = 0, a01 = 0, a10 = 0, a11 = 0;
    for (int k = 8; k >= 0; --k) {
        a00 = a00 * v + m[k][0][0];
        a01 = a01 * v + m[k][0][1];
        a10 = a10 * v + m[k][1][0];
        a11 = a11 * v + m[k][1][1];
    }
    return a00 * a11 - a01 * a10;
}

}  // namespace re_detail
}  // namespace lcbinint::holonomic
