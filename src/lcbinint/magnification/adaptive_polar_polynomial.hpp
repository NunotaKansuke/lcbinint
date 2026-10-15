#pragma once
#include <array>

namespace lcbinint::magnification::detail {

// Coefficients of |N(r)|^2-rho^2|D(r)|^2 along z=r*(ux+i*uy).
// Geometry invariants are shared by every angular column. Retain the actual
// norm of the supplied direction instead of assuming rounded sin/cos have
// norm exactly one, which matters for nearly paired source-limb roots.
struct PolarLimbPolynomial {
    long double separation, mass_sum, source_x, source_y, constant;
    long double quadratic_norm, cubic_projection, cubic_cosine, quartic_constant;

    PolarLimbPolynomial(long double a, long double m1, long double m2,
                        long double x, long double y, long double rho2)
        : separation(a), mass_sum(m1 + m2), source_x(x), source_y(y), constant(m2 * a)
    {
        const long double source_norm = x * x + y * y;
        const long double first_real = a * x - mass_sum;
        const long double first_imag = a * y;
        quadratic_norm = first_real * first_real + first_imag * first_imag - rho2 * a * a;
        cubic_projection = mass_sum - a * a;
        cubic_cosine = a * (mass_sum - source_norm) + constant + rho2 * a;
        quartic_constant = a * a + source_norm + 2.0L * first_real - rho2;
    }

    std::array<long double, 7> coefficients(long double ux, long double uy) const
    {
        const long double norm = ux * ux + uy * uy;
        const long double norm2 = norm * norm;
        const long double projection = source_x * ux + source_y * uy;
        const long double doubled_projection = source_x * (ux * ux - uy * uy)
            + 2.0L * source_y * ux * uy;
        return {
            constant * constant,
            2.0L * constant * (separation * projection - mass_sum * ux),
            norm * quadratic_norm - 2.0L * constant * (separation * norm + doubled_projection),
            2.0L * norm * (cubic_projection * projection + cubic_cosine * ux),
            norm2 * quartic_constant + 2.0L * separation * norm * doubled_projection,
            -2.0L * norm2 * (separation * ux + projection),
            norm2 * norm,
        };
    }
};

} // namespace lcbinint::magnification::detail
