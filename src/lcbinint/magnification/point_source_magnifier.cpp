#include "lcbinint/magnification/point_source_magnifier.hpp"

#include "SkowronGould.h"
#include "lcbinint/math/polynomial_roots.hpp"

#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace lcbinint::magnification {
namespace {

struct BinaryGeometry {
    Complex separation;
    Complex m1;
    Complex m2;
    Complex source;
};

struct StackCandidateImage {
    double residual = 0.0;
    double jacobian_determinant = 0.0;
};

struct PhysicalRootSelection {
    std::array<bool, 5> physical {};
};

struct FastComplex {
    double re = 0.0;
    double im = 0.0;
};

struct FastBinaryGeometry {
    double a = 0.0;
    double m1 = 0.0;
    double m2 = 0.0;
    double a2 = 0.0;
    double a3 = 0.0;
    double m2_2 = 0.0;
    FastComplex y;
};

struct FastBinaryConstants {
    double a = 0.0;
    double m1 = 0.0;
    double m2 = 0.0;
    double a2 = 0.0;
    double a3 = 0.0;
    double m2_2 = 0.0;
};

FastComplex operator+(FastComplex lhs, FastComplex rhs)
{
    return {lhs.re + rhs.re, lhs.im + rhs.im};
}

FastComplex operator-(FastComplex lhs, FastComplex rhs)
{
    return {lhs.re - rhs.re, lhs.im - rhs.im};
}

FastComplex operator*(FastComplex lhs, FastComplex rhs)
{
    return {lhs.re * rhs.re - lhs.im * rhs.im, lhs.re * rhs.im + lhs.im * rhs.re};
}

FastComplex operator*(double scale, FastComplex value)
{
    return {scale * value.re, scale * value.im};
}

FastComplex operator*(FastComplex value, double scale)
{
    return scale * value;
}

FastComplex operator-(double lhs, FastComplex rhs)
{
    return {lhs - rhs.re, -rhs.im};
}

FastComplex operator+(double lhs, FastComplex rhs)
{
    return {lhs + rhs.re, rhs.im};
}

FastComplex conj(FastComplex value)
{
    return {value.re, -value.im};
}

FastComplex sg_to_fast(::complex value)
{
    return {value.re, value.im};
}

::complex fast_to_sg(FastComplex value)
{
    return {value.re, value.im};
}

FastBinaryGeometry make_fast_vbm_geometry(double separation, double mass_ratio, SourcePosition source)
{
    const double s = std::abs(separation);
    const double q_input = std::abs(mass_ratio);
    const double q = q_input < 1.0 ? q_input : 1.0 / q_input;
    const double a = q_input < 1.0 ? -s : s;
    const double m1 = 1.0 / (1.0 + q);
    const double m2 = q * m1;
    const double a2 = a * a;
    return {a, m1, m2, a2, a2 * a, m2 * m2, {source.x + a * m1, source.y}};
}

FastBinaryConstants make_fast_vbm_constants(double separation, double mass_ratio)
{
    const double s = std::abs(separation);
    const double q_input = std::abs(mass_ratio);
    const double q = q_input < 1.0 ? q_input : 1.0 / q_input;
    const double a = q_input < 1.0 ? -s : s;
    const double m1 = 1.0 / (1.0 + q);
    const double m2 = q * m1;
    const double a2 = a * a;
    return {a, m1, m2, a2, a2 * a, m2 * m2};
}

FastBinaryGeometry make_fast_vbm_geometry(const FastBinaryConstants& constants, SourcePosition source)
{
    return {constants.a,
        constants.m1,
        constants.m2,
        constants.a2,
        constants.a3,
        constants.m2_2,
        {source.x + constants.a * constants.m1, source.y}};
}

BinaryGeometry make_vbm_geometry(double separation, double mass_ratio, SourcePosition source)
{
    const double s = std::abs(separation);
    const double q_input = std::abs(mass_ratio);
    const double q = q_input < 1.0 ? q_input : 1.0 / q_input;
    const Complex a = q_input < 1.0 ? Complex(-s, 0.0) : Complex(s, 0.0);
    const Complex m1 = 1.0 / (1.0 + q);
    const Complex m2 = q * m1;

    return {a, m1, m2, Complex(source.x, source.y) + a * m1};
}

void binary_polynomial_coefficients(const FastBinaryGeometry& geometry, std::array<::complex, 6>& coefficients)
{
    const double a = geometry.a;
    const double m2 = geometry.m2;
    const FastComplex y = geometry.y;
    const FastComplex yc = conj(y);
    const double a2 = geometry.a2;
    const double a3 = geometry.a3;
    const double c9 = a2 * geometry.m2_2;
    const double c10 = a * m2;
    const FastComplex c12 = FastComplex {a, 0.0} - yc;
    const FastComplex c13 = FastComplex {a, 0.0} + y;
    const FastComplex c14 = c13 + y;
    const FastComplex c15 = conj(c14);
    const FastComplex c16 = a * y;
    const FastComplex c17 = conj(c16);
    const FastComplex c18 = conj(c12);

    coefficients[0] = fast_to_sg(c9 * y);
    coefficients[1] = fast_to_sg(FastComplex {-c9, 0.0} +
                                 c10 * (FastComplex {a, 0.0} +
                                        (2.0 * c17 - FastComplex {2.0 + a2, 0.0}) * y));
    coefficients[2] = fast_to_sg(c10 * (FastComplex {1.0, 0.0} + c16 - 2.0 * yc * c13) -
                                 (c17 - FastComplex {1.0, 0.0}) * (c16 * c12 - c18));
    coefficients[3] = fast_to_sg(c10 * c15 +
                                 (FastComplex {a3, 0.0} + 2.0 * (1.0 + a2) * y -
                                     c17 * c14) *
                                     yc -
                                 a * c13);
    coefficients[4] = fast_to_sg(FastComplex {-c10, 0.0} -
                                 c12 * (yc * (c13 + FastComplex {a, 0.0}) -
                                        FastComplex {1.0, 0.0}));
    coefficients[5] = fast_to_sg(yc * c12);
}

double residual_squared(const FastBinaryGeometry& geometry, ::complex image)
{
    const double z_re = image.re;
    const double z_im = image.im;
    const double u1 = z_re - geometry.a;
    const double v1 = -z_im;
    const double d1 = u1 * u1 + v1 * v1;
    const double u2 = z_re;
    const double v2 = -z_im;
    const double d2 = u2 * u2 + v2 * v2;

    const double r_re = geometry.y.re - z_re +
        geometry.m1 * u1 / d1 + geometry.m2 * u2 / d2;
    const double r_im = geometry.y.im - z_im -
        geometry.m1 * v1 / d1 - geometry.m2 * v2 / d2;
    return r_re * r_re + r_im * r_im;
}

double jacobian_determinant(const FastBinaryGeometry& geometry, ::complex image)
{
    const double u1 = image.re - geometry.a;
    const double v1 = image.im;
    const double d1 = u1 * u1 + v1 * v1;
    const double d1_2 = d1 * d1;
    const double u2 = image.re;
    const double v2 = image.im;
    const double d2 = u2 * u2 + v2 * v2;
    const double d2_2 = d2 * d2;

    const double derivative_re =
        geometry.m1 * (u1 * u1 - v1 * v1) / d1_2 +
        geometry.m2 * (u2 * u2 - v2 * v2) / d2_2;
    const double derivative_im =
        -2.0 * geometry.m1 * u1 * v1 / d1_2 -
        2.0 * geometry.m2 * u2 * v2 / d2_2;
    return 1.0 - (derivative_re * derivative_re + derivative_im * derivative_im);
}

PhysicalRootSelection select_physical_roots(
    const FastBinaryGeometry& geometry,
    const std::array<::complex, 5>& roots)
{
    std::array<StackCandidateImage, 5> candidates;
    int worst1 = 0;
    int worst2 = 0;
    int worst3 = 0;
    for (std::size_t i = 0; i < roots.size(); ++i) {
        candidates[i].residual = residual_squared(geometry, roots[i]);
        const int index = static_cast<int>(i);
        if (i == 0) {
            worst1 = index;
        } else if (i == 1) {
            if (candidates[i].residual > candidates[static_cast<std::size_t>(worst1)].residual) {
                worst2 = worst1;
                worst1 = index;
            } else {
                worst2 = index;
            }
        } else if (i == 2) {
            if (candidates[i].residual > candidates[static_cast<std::size_t>(worst1)].residual) {
                worst3 = worst2;
                worst2 = worst1;
                worst1 = index;
            } else if (candidates[i].residual > candidates[static_cast<std::size_t>(worst2)].residual) {
                worst3 = worst2;
                worst2 = index;
            } else {
                worst3 = index;
            }
        } else if (candidates[i].residual > candidates[static_cast<std::size_t>(worst1)].residual) {
            worst3 = worst2;
            worst2 = worst1;
            worst1 = index;
        } else if (candidates[i].residual > candidates[static_cast<std::size_t>(worst2)].residual) {
            worst3 = worst2;
            worst2 = index;
        } else if (candidates[i].residual > candidates[static_cast<std::size_t>(worst3)].residual) {
            worst3 = index;
        }
    }

    constexpr double min_ratio2 = 1.0e-8;
    constexpr double absolute_gap2 = 1.0e-24;
    const bool three_images =
        candidates[static_cast<std::size_t>(worst2)].residual * min_ratio2 >
        candidates[static_cast<std::size_t>(worst3)].residual + absolute_gap2;
    PhysicalRootSelection selection;
    for (std::size_t i = 0; i < selection.physical.size(); ++i) {
        selection.physical[i] =
            !(three_images && (static_cast<int>(i) == worst1 || static_cast<int>(i) == worst2));
    }
    return selection;
}

PointSourceResult point_source_result_from_roots(
    const FastBinaryGeometry& geometry,
    const std::array<::complex, 5>& roots)
{
    const auto selection = select_physical_roots(geometry, roots);
    double magnification = 0.0;
    int image_count = 0;
    for (std::size_t i = 0; i < roots.size(); ++i) {
        if (!selection.physical[i]) {
            continue;
        }
        const double jacobian = jacobian_determinant(geometry, roots[i]);
        magnification += 1.0 / std::abs(jacobian);
        ++image_count;
    }
    return {magnification, image_count};
}

double max_physical_residual_squared(
    const FastBinaryGeometry& geometry,
    const std::array<::complex, 5>& roots)
{
    const auto selection = select_physical_roots(geometry, roots);
    double max_residual = 0.0;
    for (std::size_t i = 0; i < roots.size(); ++i) {
        if (selection.physical[i]) {
            max_residual = std::max(max_residual, residual_squared(geometry, roots[i]));
        }
    }
    return max_residual;
}

double derivative_error_indicator_from_roots(
    const FastBinaryGeometry& geometry,
    const std::array<::complex, 5>& roots,
    const PhysicalRootSelection& selection)
{
    double indicator = 0.0;
    const ::complex a(geometry.a, 0.0);
    const ::complex m1(geometry.m1, 0.0);
    const ::complex m2(geometry.m2, 0.0);
    for (std::size_t i = 0; i < roots.size(); ++i) {
        if (!selection.physical[i]) {
            continue;
        }
        const ::complex z = roots[i];
        const ::complex dza = z - a;
        const ::complex za2 = dza * dza;
        const ::complex zb2 = z * z;
        const ::complex j1 = m1 / za2 + m2 / zb2;
        const ::complex j1c = conj(j1);
        const double det_j = (1.0 - j1 * j1c).re;
        if (det_j == 0.0 || !std::isfinite(det_j)) {
            return std::numeric_limits<double>::infinity();
        }
        ::complex j2 = -2.0 * (m1 / (za2 * dza) + m2 / (zb2 * z));
        ::complex j3 = 6.0 * (m1 / (za2 * za2) + m2 / (zb2 * zb2));
        const double det_j2 = det_j * det_j;
        const ::complex j1c2 = j1c * j1c;
        j3 = j3 * j1c2;
        const double ob2 =
            (j2.re * j2.re + j2.im * j2.im) * (6.0 - 6.0 * det_j + det_j2);
        j2 = j2 * j2 * j1c2 * j1c;
        const double denominator = std::abs(det_j * det_j2 * det_j2);
        if (denominator == 0.0 || !std::isfinite(denominator)) {
            return std::numeric_limits<double>::infinity();
        }
        const double cq =
            0.5 * (std::abs(ob2 - 6.0 * j2.re - 2.0 * j3.re * det_j) +
                      3.0 * std::abs(j2.im)) /
            denominator;
        if (std::isfinite(cq)) {
            indicator += cq;
        } else {
            return std::numeric_limits<double>::infinity();
        }
    }
    return indicator;
}

PointSourceSafetyDiagnostic safety_diagnostic_from_roots(
    const FastBinaryGeometry& geometry,
    const std::array<::complex, 5>& roots)
{
    const auto selection = select_physical_roots(geometry, roots);
    PointSourceSafetyDiagnostic result;
    const ::complex a(geometry.a, 0.0);
    const ::complex m1(geometry.m1, 0.0);
    const ::complex m2(geometry.m2, 0.0);
    const ::complex source_conjugate(geometry.y.re, -geometry.y.im);

    for (std::size_t i = 0; i < roots.size(); ++i) {
        const ::complex z = roots[i];
        const ::complex dza = z - a;
        const ::complex za2 = dza * dza;
        const ::complex zb2 = z * z;
        const ::complex f1 = m1 / za2 + m2 / zb2;
        const ::complex f1c = conj(f1);
        const double jacobian = (1.0 - f1 * f1c).re;
        if (jacobian == 0.0 || !std::isfinite(jacobian)) {
            if (selection.physical[i]) {
                result.quadrupole_indicator = std::numeric_limits<double>::infinity();
                result.cusp_indicator = std::numeric_limits<double>::infinity();
            } else {
                result.ghost_indicator = std::numeric_limits<double>::infinity();
            }
            continue;
        }

        if (selection.physical[i]) {
            result.magnification += 1.0 / std::abs(jacobian);
            ++result.image_count;

            const ::complex f2 = -2.0 * (m1 / (za2 * dza) + m2 / (zb2 * z));
            const ::complex f3 = 6.0 * (m1 / (za2 * za2) + m2 / (zb2 * zb2));
            const double jacobian2 = jacobian * jacobian;
            const ::complex f1c2 = f1c * f1c;
            const ::complex f2_term = f2 * f2 * f1c2 * f1c;
            const ::complex f3_term = f3 * f1c2;
            const double real_term =
                (f2.re * f2.re + f2.im * f2.im) *
                    (6.0 - 6.0 * jacobian + jacobian2) -
                6.0 * f2_term.re - 2.0 * f3_term.re * jacobian;
            const double denominator =
                std::abs(jacobian * jacobian2 * jacobian2);
            if (denominator == 0.0 || !std::isfinite(denominator)) {
                result.quadrupole_indicator = std::numeric_limits<double>::infinity();
                result.cusp_indicator = std::numeric_limits<double>::infinity();
                continue;
            }
            result.quadrupole_indicator += 0.5 * std::abs(real_term) / denominator;
            result.cusp_indicator += 1.5 * std::abs(f2_term.im) / denominator;
            continue;
        }

        ++result.ghost_count;
        // Ghost-image safety test (MNRAS 479, 5157, eqs. 41--47).  The two rejected polynomial
        // roots are the ghost images available for free from the point-source
        // solve.  This estimates the inverse source-plane distance required
        // to drive either ghost image onto the critical curve.
        const ::complex f0 = -m1 / dza - m2 / z;
        const ::complex zhat = source_conjugate - f0;
        const ::complex zhat_a = zhat - a;
        const ::complex f1_hat = m1 / (zhat_a * zhat_a) + m2 / (zhat * zhat);
        const ::complex jhat = 1.0 - f1 * f1_hat;
        const ::complex zc = conj(z);
        const ::complex zc_a = zc - a;
        const ::complex f2_zc =
            -2.0 * (m1 / (zc_a * zc_a * zc_a) + m2 / (zc * zc * zc));
        const ::complex numerator_base = jhat * f2_zc * f1;
        const ::complex denominator = jacobian * jhat * jhat;
        const double denominator_abs = abs(denominator);
        if (denominator_abs == 0.0 || !std::isfinite(denominator_abs)) {
            result.ghost_indicator = std::numeric_limits<double>::infinity();
            continue;
        }
        const double indicator = abs(
            (numerator_base - conj(numerator_base) * f1_hat) / denominator);
        if (std::isfinite(indicator)) {
            result.ghost_indicator = std::max(result.ghost_indicator, indicator);
        } else {
            result.ghost_indicator = std::numeric_limits<double>::infinity();
        }
    }
    return result;
}

Complex lens_equation_residual(const BinaryGeometry& geometry, Complex image)
{
    const Complex zc = std::conj(image);
    return (geometry.source - image) + geometry.m1 / (zc - geometry.separation) + geometry.m2 / zc;
}

double jacobian_determinant(const BinaryGeometry& geometry, Complex image)
{
    const Complex dza = image - geometry.separation;
    const Complex derivative =
        geometry.m1 / (dza * dza) + geometry.m2 / (image * image);
    return 1.0 - std::norm(derivative);
}

using Polynomial = std::vector<Complex>;

// The ordinary triple solve is deliberately kept in double precision.  Its
// polynomial is degree ten, so a few physically valid roots can become too
// close for the double solver near a caustic.  ``long double`` is not a
// portable quadruple-precision type (on x86 it normally has a 64-bit
// significand), so the guarded retry below uses Boost's header-only binary
// quad type: 113 bits, the IEEE binary128 significand width.
using QuadReal = boost::multiprecision::cpp_bin_float_quad;

struct QuadComplex {
    QuadReal real = 0;
    QuadReal imag = 0;

    QuadComplex() = default;
    QuadComplex(double real_value) : real(real_value) {}
    QuadComplex(QuadReal real_value) : real(std::move(real_value)) {}
    QuadComplex(QuadReal real_value, QuadReal imag_value)
        : real(std::move(real_value)), imag(std::move(imag_value)) {}
};

QuadComplex operator+(const QuadComplex& lhs, const QuadComplex& rhs)
{
    return {lhs.real + rhs.real, lhs.imag + rhs.imag};
}

QuadComplex operator-(const QuadComplex& lhs, const QuadComplex& rhs)
{
    return {lhs.real - rhs.real, lhs.imag - rhs.imag};
}

QuadComplex operator-(const QuadComplex& value)
{
    return {-value.real, -value.imag};
}

QuadComplex operator*(const QuadComplex& lhs, const QuadComplex& rhs)
{
    return {lhs.real * rhs.real - lhs.imag * rhs.imag,
        lhs.real * rhs.imag + lhs.imag * rhs.real};
}

QuadComplex operator*(const QuadComplex& value, const QuadReal& scale)
{
    return {value.real * scale, value.imag * scale};
}

QuadComplex operator*(const QuadReal& scale, const QuadComplex& value)
{
    return value * scale;
}

QuadComplex operator/(const QuadComplex& value, const QuadReal& scale)
{
    return {value.real / scale, value.imag / scale};
}

QuadComplex operator/(const QuadComplex& lhs, const QuadComplex& rhs)
{
    const QuadReal denominator = rhs.real * rhs.real + rhs.imag * rhs.imag;
    return {(lhs.real * rhs.real + lhs.imag * rhs.imag) / denominator,
        (lhs.imag * rhs.real - lhs.real * rhs.imag) / denominator};
}

QuadComplex& operator+=(QuadComplex& lhs, const QuadComplex& rhs)
{
    lhs.real += rhs.real;
    lhs.imag += rhs.imag;
    return lhs;
}

QuadComplex& operator-=(QuadComplex& lhs, const QuadComplex& rhs)
{
    lhs.real -= rhs.real;
    lhs.imag -= rhs.imag;
    return lhs;
}

QuadComplex& operator*=(QuadComplex& lhs, const QuadComplex& rhs)
{
    lhs = lhs * rhs;
    return lhs;
}

QuadReal quad_abs(const QuadReal& value)
{
    return value < 0 ? -value : value;
}

QuadReal quad_abs(const QuadComplex& value)
{
    return boost::multiprecision::sqrt(
        value.real * value.real + value.imag * value.imag);
}

bool quad_isfinite(const QuadReal& value)
{
    return boost::math::isfinite(value);
}

bool quad_isfinite(const QuadComplex& value)
{
    return quad_isfinite(value.real) && quad_isfinite(value.imag);
}

using QuadPolynomial = std::vector<QuadComplex>;

QuadPolynomial quad_add_polynomial(
    const QuadPolynomial& lhs,
    const QuadPolynomial& rhs)
{
    QuadPolynomial out(std::max(lhs.size(), rhs.size()), QuadComplex(0.0));
    for (std::size_t i = 0; i < lhs.size(); ++i) out[i] += lhs[i];
    for (std::size_t i = 0; i < rhs.size(); ++i) out[i] += rhs[i];
    return out;
}

QuadPolynomial quad_subtract_polynomial(
    const QuadPolynomial& lhs,
    const QuadPolynomial& rhs)
{
    QuadPolynomial out(std::max(lhs.size(), rhs.size()), QuadComplex(0.0));
    for (std::size_t i = 0; i < lhs.size(); ++i) out[i] += lhs[i];
    for (std::size_t i = 0; i < rhs.size(); ++i) out[i] -= rhs[i];
    return out;
}

QuadPolynomial quad_multiply_polynomial(
    const QuadPolynomial& lhs,
    const QuadPolynomial& rhs)
{
    QuadPolynomial out(lhs.size() + rhs.size() - 1, QuadComplex(0.0));
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        for (std::size_t j = 0; j < rhs.size(); ++j) {
            out[i + j] += lhs[i] * rhs[j];
        }
    }
    return out;
}

QuadPolynomial quad_scale_polynomial(
    const QuadPolynomial& polynomial,
    QuadComplex scale)
{
    QuadPolynomial out = polynomial;
    for (auto& coefficient : out) coefficient *= scale;
    return out;
}

Polynomial trim_polynomial(Polynomial polynomial)
{
    while (polynomial.size() > 1 && std::abs(polynomial.back()) == 0.0) {
        polynomial.pop_back();
    }
    return polynomial;
}

Polynomial add_polynomial(const Polynomial& lhs, const Polynomial& rhs)
{
    Polynomial out(std::max(lhs.size(), rhs.size()), 0.0);
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        out[i] += lhs[i];
    }
    for (std::size_t i = 0; i < rhs.size(); ++i) {
        out[i] += rhs[i];
    }
    return trim_polynomial(std::move(out));
}

Polynomial subtract_polynomial(const Polynomial& lhs, const Polynomial& rhs)
{
    Polynomial out(std::max(lhs.size(), rhs.size()), 0.0);
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        out[i] += lhs[i];
    }
    for (std::size_t i = 0; i < rhs.size(); ++i) {
        out[i] -= rhs[i];
    }
    return trim_polynomial(std::move(out));
}

Polynomial multiply_polynomial(const Polynomial& lhs, const Polynomial& rhs)
{
    Polynomial out(lhs.size() + rhs.size() - 1, 0.0);
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        for (std::size_t j = 0; j < rhs.size(); ++j) {
            out[i + j] += lhs[i] * rhs[j];
        }
    }
    return trim_polynomial(std::move(out));
}

Polynomial scale_polynomial(const Polynomial& polynomial, Complex scale)
{
    Polynomial out = polynomial;
    for (auto& coefficient : out) {
        coefficient *= scale;
    }
    return trim_polynomial(std::move(out));
}

Polynomial product_without_lens(
    const std::array<Complex, 3>& lens_positions,
    std::size_t excluded)
{
    Polynomial out = {1.0};
    for (std::size_t i = 0; i < lens_positions.size(); ++i) {
        if (i == excluded) {
            continue;
        }
        out = multiply_polynomial(out, {-lens_positions[i], 1.0});
    }
    return out;
}

// The lens-equation polynomial depends on the source only through y and
// conj(y): each conjugate denominator is p(z)*yc + B_i(z) with geometry-only
// p, B_i, so the full degree-10 polynomial is
//   (z - y) * sum_k yc^k D_k(z)  -  sum_k yc^k E_k(z)
// with seven geometry-only basis polynomials D_0..3, E_0..2.  Building the
// basis costs the old full construction once per lens geometry; every
// subsequent source position only evaluates the cubic in yc (~50 complex
// multiplies instead of ~30 heap-allocating polynomial products).
struct TriplePolynomialBasis {
    bool valid = false;
    model::TripleLensGeometry geometry {};
    std::array<Polynomial, 4> d;
    std::array<Polynomial, 3> e;
};

bool same_triple_basis_geometry(
    const model::TripleLensGeometry& lhs,
    const model::TripleLensGeometry& rhs)
{
    for (std::size_t i = 0; i < lhs.lens_positions.size(); ++i) {
        if (lhs.lens_positions[i].x != rhs.lens_positions[i].x ||
            lhs.lens_positions[i].y != rhs.lens_positions[i].y ||
            lhs.masses[i] != rhs.masses[i]) {
            return false;
        }
    }
    return true;
}

const TriplePolynomialBasis& triple_polynomial_basis(
    const model::TripleLensGeometry& geometry)
{
    thread_local TriplePolynomialBasis cache;
    if (cache.valid && same_triple_basis_geometry(cache.geometry, geometry)) {
        return cache;
    }

    std::array<Complex, 3> lens_positions;
    for (std::size_t i = 0; i < lens_positions.size(); ++i) {
        lens_positions[i] = {
            geometry.lens_positions[i].x,
            geometry.lens_positions[i].y};
    }
    const std::array<double, 3>& m = geometry.masses;

    Polynomial p = {1.0};
    for (const auto& lens : lens_positions) {
        p = multiply_polynomial(p, {-lens, 1.0});
    }
    Polynomial s = {0.0};
    for (std::size_t j = 0; j < lens_positions.size(); ++j) {
        s = add_polynomial(
            s, scale_polynomial(product_without_lens(lens_positions, j), m[j]));
    }
    // conjugate_denominator_i(yc; z) = p(z) * yc + b[i](z)
    std::array<Polynomial, 3> b;
    for (std::size_t i = 0; i < b.size(); ++i) {
        b[i] = subtract_polynomial(
            s, scale_polynomial(p, std::conj(lens_positions[i])));
    }

    const Polynomial p2 = multiply_polynomial(p, p);
    const Polynomial p3 = multiply_polynomial(p2, p);
    const Polynomial b01 = multiply_polynomial(b[0], b[1]);
    const Polynomial b02 = multiply_polynomial(b[0], b[2]);
    const Polynomial b12 = multiply_polynomial(b[1], b[2]);

    cache.d[3] = p3;
    cache.d[2] = multiply_polynomial(
        p2, add_polynomial(add_polynomial(b[0], b[1]), b[2]));
    cache.d[1] = multiply_polynomial(
        p, add_polynomial(add_polynomial(b01, b02), b12));
    cache.d[0] = multiply_polynomial(b01, b[2]);

    cache.e[2] = scale_polynomial(p3, m[0] + m[1] + m[2]);
    cache.e[1] = multiply_polynomial(
        p2,
        add_polynomial(
            add_polynomial(
                scale_polynomial(add_polynomial(b[1], b[2]), m[0]),
                scale_polynomial(add_polynomial(b[0], b[2]), m[1])),
            scale_polynomial(add_polynomial(b[0], b[1]), m[2])));
    cache.e[0] = multiply_polynomial(
        p,
        add_polynomial(
            add_polynomial(
                scale_polynomial(b12, m[0]),
                scale_polynomial(b02, m[1])),
            scale_polynomial(b01, m[2])));

    cache.geometry = geometry;
    cache.valid = true;
    return cache;
}

Polynomial triple_polynomial_coefficients(
    const model::TripleLensGeometry& geometry,
    SourcePosition source)
{
    const auto& basis = triple_polynomial_basis(geometry);
    const Complex y(source.x, source.y);
    const Complex yc = std::conj(y);
    const Complex yc2 = yc * yc;
    const Complex yc3 = yc2 * yc;

    const auto coefficient_at = [](const Polynomial& polynomial, std::size_t j) {
        return j < polynomial.size() ? polynomial[j] : Complex(0.0);
    };
    std::array<Complex, 10> g {};
    std::array<Complex, 10> h {};
    for (std::size_t j = 0; j < g.size(); ++j) {
        g[j] = coefficient_at(basis.d[0], j) + yc * coefficient_at(basis.d[1], j) +
               yc2 * coefficient_at(basis.d[2], j) + yc3 * coefficient_at(basis.d[3], j);
        h[j] = coefficient_at(basis.e[0], j) + yc * coefficient_at(basis.e[1], j) +
               yc2 * coefficient_at(basis.e[2], j);
    }

    Polynomial coefficients(11, 0.0);
    for (std::size_t j = 0; j <= 10; ++j) {
        Complex value = j >= 1 ? g[j - 1] : Complex(0.0);
        if (j < g.size()) {
            value -= y * g[j] + h[j];
        }
        coefficients[j] = value;
    }
    return coefficients;
}

QuadPolynomial quad_product_without_lens(
    const std::array<QuadComplex, 3>& lens_positions,
    std::size_t excluded)
{
    QuadPolynomial out = {QuadComplex(1.0)};
    for (std::size_t i = 0; i < lens_positions.size(); ++i) {
        if (i == excluded) continue;
        out = quad_multiply_polynomial(
            out, {-lens_positions[i], QuadComplex(1.0)});
    }
    return out;
}

QuadComplex quad_coefficient_at(const QuadPolynomial& polynomial, std::size_t index)
{
    return index < polynomial.size() ? polynomial[index] : QuadComplex(0.0);
}

QuadPolynomial quad_triple_polynomial_coefficients(
    const model::TripleLensGeometry& geometry,
    SourcePosition source)
{
    std::array<QuadComplex, 3> lens_positions;
    for (std::size_t i = 0; i < lens_positions.size(); ++i) {
        lens_positions[i] = QuadComplex(
            QuadReal(geometry.lens_positions[i].x),
            QuadReal(geometry.lens_positions[i].y));
    }
    std::array<QuadReal, 3> masses;
    for (std::size_t i = 0; i < masses.size(); ++i) {
        masses[i] = QuadReal(geometry.masses[i]);
    }

    QuadPolynomial p = {QuadComplex(1.0)};
    for (const auto& lens : lens_positions) {
        p = quad_multiply_polynomial(p, {-lens, QuadComplex(1.0)});
    }
    QuadPolynomial s = {QuadComplex(0.0)};
    for (std::size_t j = 0; j < lens_positions.size(); ++j) {
        s = quad_add_polynomial(
            s,
            quad_scale_polynomial(
                quad_product_without_lens(lens_positions, j),
                QuadComplex(masses[j])));
    }

    std::array<QuadPolynomial, 3> b;
    for (std::size_t i = 0; i < b.size(); ++i) {
        const QuadComplex conjugate_lens(
            lens_positions[i].real, -lens_positions[i].imag);
        b[i] = quad_subtract_polynomial(
            s, quad_scale_polynomial(p, conjugate_lens));
    }

    const QuadPolynomial p2 = quad_multiply_polynomial(p, p);
    const QuadPolynomial p3 = quad_multiply_polynomial(p2, p);
    const QuadPolynomial b01 = quad_multiply_polynomial(b[0], b[1]);
    const QuadPolynomial b02 = quad_multiply_polynomial(b[0], b[2]);
    const QuadPolynomial b12 = quad_multiply_polynomial(b[1], b[2]);

    std::array<QuadPolynomial, 4> d;
    std::array<QuadPolynomial, 3> e;
    d[3] = p3;
    d[2] = quad_multiply_polynomial(
        p2,
        quad_add_polynomial(
            quad_add_polynomial(b[0], b[1]), b[2]));
    d[1] = quad_multiply_polynomial(
        p,
        quad_add_polynomial(
            quad_add_polynomial(b01, b02), b12));
    d[0] = quad_multiply_polynomial(b01, b[2]);

    const QuadReal total_mass = masses[0] + masses[1] + masses[2];
    e[2] = quad_scale_polynomial(p3, QuadComplex(total_mass));
    e[1] = quad_multiply_polynomial(
        p2,
        quad_add_polynomial(
            quad_add_polynomial(
                quad_scale_polynomial(
                    quad_add_polynomial(b[1], b[2]), QuadComplex(masses[0])),
                quad_scale_polynomial(
                    quad_add_polynomial(b[0], b[2]), QuadComplex(masses[1]))),
            quad_scale_polynomial(
                quad_add_polynomial(b[0], b[1]), QuadComplex(masses[2]))));
    e[0] = quad_multiply_polynomial(
        p,
        quad_add_polynomial(
            quad_add_polynomial(
                quad_scale_polynomial(b12, QuadComplex(masses[0])),
                quad_scale_polynomial(b02, QuadComplex(masses[1]))),
            quad_scale_polynomial(b01, QuadComplex(masses[2]))));

    const QuadComplex y(
        QuadReal(source.x), QuadReal(source.y));
    const QuadComplex yc(y.real, -y.imag);
    const QuadComplex yc2 = yc * yc;
    const QuadComplex yc3 = yc2 * yc;
    std::array<QuadComplex, 10> g {};
    std::array<QuadComplex, 10> h {};
    for (std::size_t j = 0; j < g.size(); ++j) {
        g[j] = quad_coefficient_at(d[0], j) +
            yc * quad_coefficient_at(d[1], j) +
            yc2 * quad_coefficient_at(d[2], j) +
            yc3 * quad_coefficient_at(d[3], j);
        h[j] = quad_coefficient_at(e[0], j) +
            yc * quad_coefficient_at(e[1], j) +
            yc2 * quad_coefficient_at(e[2], j);
    }

    QuadPolynomial coefficients(11, QuadComplex(0.0));
    for (std::size_t j = 0; j <= 10; ++j) {
        QuadComplex value = j >= 1 ? g[j - 1] : QuadComplex(0.0);
        if (j < g.size()) value -= y * g[j] + h[j];
        coefficients[j] = value;
    }
    return coefficients;
}

constexpr std::size_t kTriplePolynomialDegree = 10;

void quad_polynomial_value_and_derivative(
    const QuadPolynomial& coefficients,
    const QuadComplex& z,
    QuadComplex& value,
    QuadComplex& derivative)
{
    value = coefficients.back();
    derivative = QuadComplex(0.0);
    for (std::size_t index = coefficients.size() - 1; index > 0; --index) {
        derivative = derivative * z + value;
        value = value * z + coefficients[index - 1];
    }
}

QuadReal quad_polynomial_scale(
    const QuadPolynomial& coefficients,
    const QuadComplex& z)
{
    const QuadReal radius = quad_abs(z);
    QuadReal scale = 0;
    for (auto it = coefficients.rbegin(); it != coefficients.rend(); ++it) {
        scale = scale * radius + quad_abs(*it);
    }
    return scale;
}

bool quad_roots_are_usable(
    const QuadPolynomial& coefficients,
    const std::array<QuadComplex, kTriplePolynomialDegree>& roots)
{
    static const QuadReal residual_tolerance("1e-22");
    for (const auto& root : roots) {
        if (!quad_isfinite(root)) return false;
        QuadComplex value;
        QuadComplex derivative;
        quad_polynomial_value_and_derivative(
            coefficients, root, value, derivative);
        const QuadReal scale = std::max(
            QuadReal(1.0), quad_polynomial_scale(coefficients, root));
        if (quad_abs(value) > residual_tolerance * scale) return false;
    }

    // Residuals alone do not establish that the solver returned all ten
    // roots.  Reconstruct the monic polynomial from the candidate set as a
    // second, independent completeness check.
    std::array<QuadComplex, kTriplePolynomialDegree + 1> reconstructed {};
    reconstructed[0] = QuadComplex(1.0);
    for (std::size_t i = 0; i < roots.size(); ++i) {
        for (std::size_t k = i + 1; k > 0; --k) {
            reconstructed[k] = reconstructed[k - 1] -
                roots[i] * reconstructed[k];
        }
        reconstructed[0] = reconstructed[0] * -roots[i];
    }
    for (std::size_t k = 0; k <= kTriplePolynomialDegree; ++k) {
        const QuadReal scale = std::max(
            QuadReal(1.0), quad_abs(coefficients[k]));
        if (quad_abs(reconstructed[k] - coefficients[k]) >
            residual_tolerance * scale) {
            return false;
        }
    }
    return true;
}

bool run_quadruple_aberth(
    const QuadPolynomial& coefficients,
    std::array<QuadComplex, kTriplePolynomialDegree>& roots)
{
    static const QuadReal update_tolerance("1e-28");
    constexpr int maximum_iterations = 180;
    std::array<QuadComplex, kTriplePolynomialDegree> next;
    for (int iteration = 0; iteration < maximum_iterations; ++iteration) {
        QuadReal maximum_update = 0;
        for (std::size_t i = 0; i < roots.size(); ++i) {
            QuadComplex value;
            QuadComplex derivative;
            quad_polynomial_value_and_derivative(
                coefficients, roots[i], value, derivative);
            QuadComplex root_repulsion;
            for (std::size_t j = 0; j < roots.size(); ++j) {
                if (j == i) continue;
                const QuadComplex separation = roots[i] - roots[j];
                if (quad_abs(separation) == 0) return false;
                root_repulsion += QuadComplex(1.0) / separation;
            }
            QuadComplex denominator = derivative - value * root_repulsion;
            if (quad_abs(denominator) == 0) {
                denominator = derivative;
            }
            if (quad_abs(denominator) == 0) return false;
            const QuadComplex step = value / denominator;
            next[i] = roots[i] - step;
            if (!quad_isfinite(next[i])) return false;
            maximum_update = std::max(
                maximum_update,
                quad_abs(step) / std::max(QuadReal(1.0), quad_abs(next[i])));
        }
        roots = next;
        if (maximum_update <= update_tolerance &&
            quad_roots_are_usable(coefficients, roots)) {
            return true;
        }
    }
    return quad_roots_are_usable(coefficients, roots);
}

bool solve_triple_polynomial_quadruple(
    const model::TripleLensGeometry& geometry,
    SourcePosition source,
    const std::vector<Complex>& double_roots,
    std::array<Complex, kTriplePolynomialDegree>& output)
{
    if (double_roots.size() != kTriplePolynomialDegree) return false;
    const QuadPolynomial raw_coefficients =
        quad_triple_polynomial_coefficients(geometry, source);
    if (raw_coefficients.size() != kTriplePolynomialDegree + 1 ||
        quad_abs(raw_coefficients.back()) == 0) {
        return false;
    }
    const QuadComplex leading = raw_coefficients.back();
    QuadPolynomial coefficients = raw_coefficients;
    for (auto& coefficient : coefficients) coefficient = coefficient / leading;

    std::array<QuadComplex, kTriplePolynomialDegree> roots;
    bool finite_seeds = true;
    bool separated_seeds = true;
    for (std::size_t i = 0; i < roots.size(); ++i) {
        roots[i] = QuadComplex(
            QuadReal(double_roots[i].real()),
            QuadReal(double_roots[i].imag()));
        finite_seeds = finite_seeds && quad_isfinite(roots[i]);
        for (std::size_t j = 0; j < i; ++j) {
            const QuadReal scale = std::max(
                QuadReal(1.0), quad_abs(roots[i]));
            if (quad_abs(roots[i] - roots[j]) <
                QuadReal("1e-12") * scale) {
                separated_seeds = false;
            }
        }
    }
    if (!finite_seeds) return false;
    if (separated_seeds && run_quadruple_aberth(coefficients, roots)) {
        for (std::size_t i = 0; i < roots.size(); ++i) {
            output[i] = {
                roots[i].real.convert_to<double>(),
                roots[i].imag.convert_to<double>()};
        }
        return true;
    }

    // At a caustic the double solver can return coincident seeds.  Aberth's
    // repulsion term then has no direction, so retry from the Cauchy circle;
    // the polynomial itself, rather than the double roots, determines the
    // starting scale.
    QuadReal radius = 1.0;
    for (std::size_t i = 0; i < kTriplePolynomialDegree; ++i) {
        radius = std::max(radius, quad_abs(coefficients[i]));
    }
    radius += 1.0;
    const double pi = std::acos(-1.0);
    for (std::size_t i = 0; i < roots.size(); ++i) {
        const double angle = 2.0 * pi * (static_cast<double>(i) + 0.5) /
            static_cast<double>(roots.size());
        roots[i] = radius * QuadComplex(
            QuadReal(std::cos(angle)), QuadReal(std::sin(angle)));
    }
    if (!run_quadruple_aberth(coefficients, roots)) return false;
    for (std::size_t i = 0; i < roots.size(); ++i) {
        output[i] = {
            roots[i].real.convert_to<double>(),
            roots[i].imag.convert_to<double>()};
    }
    return true;
}

template <std::size_t Degree>
bool continue_triple_polynomial_roots(
    const Polynomial& coefficients,
    std::array<Complex, Degree>& roots)
{
    // The deflation solver's warm-start mode chooses the next root from the
    // smallest seed and changes the seed ordering while it deflates.  That
    // ordering is not stable near close roots, so continue the whole root set
    // simultaneously instead of letting one seed decide another root's
    // basin.  The Vieta/residual check below remains the completeness guard.
    if (coefficients.size() != Degree + 1) return false;
    std::array<Complex, Degree> current;
    std::array<Complex, Degree> next;
    for (std::size_t i = 0; i < Degree; ++i) {
        current[i] = roots[i];
        if (!std::isfinite(current[i].real()) ||
            !std::isfinite(current[i].imag())) {
            return false;
        }
    }
    constexpr int maximum_iterations = 24;
    constexpr double convergence_tolerance = 1.0e-9;
    for (int iteration = 0; iteration < maximum_iterations; ++iteration) {
        double maximum_update = 0.0;
        for (std::size_t i = 0; i < Degree; ++i) {
            const Complex z = current[i];
            Complex value = coefficients.back();
            Complex derivative = 0.0;
            for (std::size_t index = coefficients.size() - 1; index > 0; --index) {
                derivative = derivative * z + value;
                value = value * z + coefficients[index - 1];
            }
            Complex root_repulsion = 0.0;
            for (std::size_t j = 0; j < Degree; ++j) {
                if (j == i) continue;
                const Complex separation = z - current[j];
                if (std::abs(separation) < 1.0e-18) return false;
                root_repulsion += 1.0 / separation;
            }
            const Complex denominator = derivative - value * root_repulsion;
            if (!std::isfinite(denominator.real()) ||
                !std::isfinite(denominator.imag()) ||
                std::abs(denominator) == 0.0) {
                return false;
            }
            const Complex step = value / denominator;
            if (!std::isfinite(step.real()) || !std::isfinite(step.imag())) {
                return false;
            }
            next[i] = z - step;
            maximum_update = std::max(maximum_update, std::abs(step));
        }
        current = next;
        if (maximum_update <= convergence_tolerance) break;
    }
    for (std::size_t i = 0; i < Degree; ++i) {
        roots[i] = current[i];
    }
    for (const auto& root : current) {
        if (!std::isfinite(root.real()) || !std::isfinite(root.imag())) {
            return false;
        }
    }
    return true;
}

static SourcePosition polish_triple_image_root(
    const model::TripleLensGeometry& geometry,
    SourcePosition source,
    SourcePosition z)
{
    // 2D Newton on true lens equation f(x,y) = (x - sx - wx, y - sy - wy) = 0.
    // J_f = [[1-dxx, dxy], [dxy, 1+dxx]] where dxx = sum m*(dy^2-dx^2)/d4, dxy = sum m*2*dx*dy/d4.
    // 60 iterations covers the worst observed case (~36 for near-caustic spurious roots).
    // The tolerance is 1e-11 rather than 1e-14 because strongly demagnified images (|J|>>1)
    // cannot achieve source-plane residuals below |J|*eps_machine.
    constexpr int kMaxIter = 60;
    constexpr double kTol = 1.0e-11;
    for (int iter = 0; iter < kMaxIter; ++iter) {
        double sx = 0.0, sy = 0.0, dxx = 0.0, dxy = 0.0;
        for (std::size_t i = 0; i < geometry.lens_positions.size(); ++i) {
            const double dx = z.x - geometry.lens_positions[i].x;
            const double dy = z.y - geometry.lens_positions[i].y;
            const double d2 = dx * dx + dy * dy;
            const double d4 = d2 * d2;
            sx += geometry.masses[i] * dx / d2;
            sy += geometry.masses[i] * dy / d2;
            dxx += geometry.masses[i] * (dy * dy - dx * dx) / d4;
            dxy += geometry.masses[i] * 2.0 * dx * dy / d4;
        }
        const double fx = z.x - sx - source.x;
        const double fy = z.y - sy - source.y;
        if (std::abs(fx) + std::abs(fy) < kTol) { break; }
        const double j00 = 1.0 - dxx;
        const double j01 = dxy;
        const double j11 = 1.0 + dxx;
        const double det = j00 * j11 - j01 * j01;
        if (std::abs(det) < 1.0e-25) { break; }
        z.x -= (j11 * fx - j01 * fy) / det;
        z.y -= (j00 * fy - j01 * fx) / det;
    }
    return z;
}

static SourcePosition polish_triple_image_root_high_precision(
    const model::TripleLensGeometry& geometry,
    SourcePosition source,
    SourcePosition z)
{
    constexpr int kMaxIter = 100;
    static const QuadReal kTol("1e-30");
    static const QuadReal kDetFloor("1e-60");
    QuadReal zx(z.x);
    QuadReal zy(z.y);
    const QuadReal source_x(source.x);
    const QuadReal source_y(source.y);
    for (int iter = 0; iter < kMaxIter; ++iter) {
        QuadReal sx = 0.0;
        QuadReal sy = 0.0;
        QuadReal dxx = 0.0;
        QuadReal dxy = 0.0;
        for (std::size_t i = 0; i < geometry.lens_positions.size(); ++i) {
            const QuadReal lens_x(geometry.lens_positions[i].x);
            const QuadReal lens_y(geometry.lens_positions[i].y);
            const QuadReal mass(geometry.masses[i]);
            const QuadReal dx = zx - lens_x;
            const QuadReal dy = zy - lens_y;
            const QuadReal d2 = dx * dx + dy * dy;
            if (d2 == 0) {
                return z;
            }
            const QuadReal d4 = d2 * d2;
            sx += mass * dx / d2;
            sy += mass * dy / d2;
            dxx += mass * (dy * dy - dx * dx) / d4;
            dxy += mass * QuadReal(2.0) * dx * dy / d4;
        }
        const QuadReal fx = zx - sx - source_x;
        const QuadReal fy = zy - sy - source_y;
        if (quad_abs(fx) + quad_abs(fy) < kTol) break;
        const QuadReal j00 = 1.0 - dxx;
        const QuadReal j01 = dxy;
        const QuadReal j11 = 1.0 + dxx;
        const QuadReal det = j00 * j11 - j01 * j01;
        if (quad_abs(det) < kDetFloor || !quad_isfinite(det)) break;
        zx -= (j11 * fx - j01 * fy) / det;
        zy -= (j00 * fy - j01 * fx) / det;
    }
    if (!quad_isfinite(zx) || !quad_isfinite(zy)) return z;
    return {zx.convert_to<double>(), zy.convert_to<double>()};
}

double triple_residual_high_precision(
    const model::TripleLensGeometry& geometry,
    SourcePosition source,
    SourcePosition image)
{
    QuadReal sx(image.x);
    QuadReal sy(image.y);
    for (std::size_t i = 0; i < geometry.lens_positions.size(); ++i) {
        const QuadReal dx = QuadReal(image.x) -
            QuadReal(geometry.lens_positions[i].x);
        const QuadReal dy = QuadReal(image.y) -
            QuadReal(geometry.lens_positions[i].y);
        const QuadReal d2 = dx * dx + dy * dy;
        if (d2 == 0) return std::numeric_limits<double>::infinity();
        const QuadReal mass(geometry.masses[i]);
        sx -= mass * dx / d2;
        sy -= mass * dy / d2;
    }
    const QuadReal dx = sx - QuadReal(source.x);
    const QuadReal dy = sy - QuadReal(source.y);
    return boost::multiprecision::sqrt(dx * dx + dy * dy).convert_to<double>();
}

double triple_jacobian_determinant_high_precision(
    const model::TripleLensGeometry& geometry,
    SourcePosition image)
{
    QuadComplex derivative;
    for (std::size_t i = 0; i < geometry.lens_positions.size(); ++i) {
        const QuadComplex dz(
            QuadReal(image.x) - QuadReal(geometry.lens_positions[i].x),
            QuadReal(image.y) - QuadReal(geometry.lens_positions[i].y));
        if (quad_abs(dz) == 0) {
            return std::numeric_limits<double>::infinity();
        }
        derivative += QuadComplex(QuadReal(geometry.masses[i])) / (dz * dz);
    }
    const QuadReal determinant =
        QuadReal(1.0) - derivative.real * derivative.real -
        derivative.imag * derivative.imag;
    return determinant.convert_to<double>();
}

double triple_residual(
    const model::TripleLensGeometry& geometry,
    SourcePosition source,
    Complex image)
{
    const auto mapped = model::triple_lens_equation(
        geometry,
        {image.real(), image.imag()});
    const double dx = mapped.x - source.x;
    const double dy = mapped.y - source.y;
    return std::sqrt(dx * dx + dy * dy);
}

double triple_jacobian_determinant(
    const model::TripleLensGeometry& geometry,
    Complex image)
{
    Complex derivative = 0.0;
    for (std::size_t i = 0; i < geometry.lens_positions.size(); ++i) {
        const Complex lens(
            geometry.lens_positions[i].x,
            geometry.lens_positions[i].y);
        const Complex dz = image - lens;
        derivative += geometry.masses[i] / (dz * dz);
    }
    return 1.0 - std::norm(derivative);
}

template <std::size_t Degree, typename Coefficients>
bool polynomial_roots_are_usable(
    const Coefficients& coefficients,
    const std::array<Complex, Degree>& roots)
{
    if (coefficients.size() != Degree + 1) {
        return false;
    }
    constexpr double residual_tolerance = 1.0e-8;
    for (std::size_t i = 0; i < Degree; ++i) {
        const Complex z = roots[i];
        if (!std::isfinite(z.real()) || !std::isfinite(z.imag())) {
            return false;
        }
        Complex value = 0.0;
        double scale = 0.0;
        const double radius = std::abs(z);
        for (auto it = coefficients.rbegin(); it != coefficients.rend(); ++it) {
            value = value * z + *it;
            scale = scale * radius + std::abs(*it);
        }
        if (std::abs(value) > residual_tolerance * std::max(scale, 1.0)) {
            return false;
        }
    }

    // Individual residuals do not prove completeness: the solver could
    // return one valid root twice and omit another. Reconstruct the monic
    // polynomial from the complete root set and compare every coefficient.
    // Unlike a pair-distance test this permits genuinely close or repeated
    // roots near a caustic while still detecting a missing-root warm start.
    std::array<Complex, Degree + 1> reconstructed {};
    reconstructed[0] = 1.0;
    for (std::size_t i = 0; i < Degree; ++i) {
        const Complex root = roots[i];
        for (std::size_t k = i + 1; k > 0; --k) {
            reconstructed[k] = reconstructed[k - 1] - root * reconstructed[k];
        }
        reconstructed[0] *= -root;
    }
    const Complex leading = coefficients[Degree];
    if (!(std::isfinite(leading.real()) && std::isfinite(leading.imag())) ||
        std::abs(leading) == 0.0) {
        return false;
    }
    constexpr double coefficient_tolerance = 1.0e-7;
    for (std::size_t k = 0; k <= Degree; ++k) {
        const Complex expected = coefficients[k] / leading;
        if (std::abs(reconstructed[k] - expected) >
            coefficient_tolerance * std::max(std::abs(expected), 1.0)) {
            return false;
        }
    }
    return true;
}

template <std::size_t CoefficientCount, std::size_t Degree>
bool polynomial_roots_are_usable(
    const std::array<::complex, CoefficientCount>& coefficients,
    const std::array<::complex, Degree>& roots)
{
    std::array<Complex, CoefficientCount> converted_coefficients;
    std::array<Complex, Degree> converted_roots;
    for (std::size_t i = 0; i < CoefficientCount; ++i) {
        converted_coefficients[i] = {coefficients[i].re, coefficients[i].im};
    }
    for (std::size_t i = 0; i < Degree; ++i) {
        converted_roots[i] = {roots[i].re, roots[i].im};
    }
    return polynomial_roots_are_usable(
        converted_coefficients, converted_roots);
}

void apply_binary_image_predictor(
    const FastBinaryGeometry& previous_geometry,
    const FastBinaryGeometry& next_geometry,
    std::array<::complex, 5>& roots)
{
    // The analytic predictor assumes fixed lens positions and masses. Dynamic
    // geometries still use the retained roots as zero-order continuation
    // seeds and are protected by the ordinary validation/cold fallback.
    if (previous_geometry.a != next_geometry.a ||
        previous_geometry.m1 != next_geometry.m1 ||
        previous_geometry.m2 != next_geometry.m2) {
        return;
    }
    const Complex delta_source {
        next_geometry.y.re - previous_geometry.y.re,
        next_geometry.y.im - previous_geometry.y.im,
    };
    const double source_step = std::abs(delta_source);
    // Do not extrapolate across the end-to-start jump of a repeated light
    // curve. The predictor is intended for locally contiguous trajectory and
    // probe samples; zero-order continuation remains the safer large-step
    // seed.
    if (!(source_step > 0.0) || source_step > 0.25) {
        return;
    }

    const auto physical = select_physical_roots(previous_geometry, roots);
    constexpr double minimum_abs_jacobian = 1.0e-4;
    constexpr double maximum_image_step = 0.5;
    for (std::size_t i = 0; i < roots.size(); ++i) {
        // The two rejected roots of a three-image binary solution are roots of
        // the eliminated polynomial, not physical lens-equation images. The
        // local inverse lens Jacobian therefore applies only to physical roots.
        if (!physical.physical[i]) {
            continue;
        }
        const Complex z {roots[i].re, roots[i].im};
        const Complex zbar = std::conj(z);
        const Complex primary = zbar - previous_geometry.a;
        if (std::abs(primary) == 0.0 || std::abs(zbar) == 0.0) {
            continue;
        }
        const Complex kappa =
            previous_geometry.m1 / (primary * primary) +
            previous_geometry.m2 / (zbar * zbar);
        const double jacobian = 1.0 - std::norm(kappa);
        if (!std::isfinite(jacobian) ||
            std::abs(jacobian) < minimum_abs_jacobian) {
            continue;
        }
        const Complex delta_image =
            (delta_source - kappa * std::conj(delta_source)) / jacobian;
        if (!std::isfinite(delta_image.real()) ||
            !std::isfinite(delta_image.imag()) ||
            std::abs(delta_image) > maximum_image_step) {
            continue;
        }
        roots[i] = {
            roots[i].re + delta_image.real(),
            roots[i].im + delta_image.imag(),
        };
    }
}

} // namespace

PointSourceResult PointSourceMagnifier::binary_mag0(
    double separation,
    double mass_ratio,
    SourcePosition source) const
{
    return binary_mag0_impl(separation, mass_ratio, source, false);
}

PointSourceResult PointSourceMagnifier::binary_mag0_cached(
    double separation,
    double mass_ratio,
    SourcePosition source) const
{
    return binary_mag0_impl(separation, mass_ratio, source, true);
}

PointSourceDerivativeResult PointSourceMagnifier::binary_mag0_with_derivatives(
    double separation,
    double mass_ratio,
    SourcePosition source) const
{
    if (separation == 0.0 || mass_ratio <= 0.0) {
        return {};
    }

    const FastBinaryGeometry geometry = make_fast_vbm_geometry(separation, mass_ratio, source);
    std::array<::complex, 6> coefficients;
    std::array<::complex, 5> roots;
    binary_polynomial_coefficients(geometry, coefficients);
    cmplx_roots_gen(roots.data(), coefficients.data(), 5, true, false);
    const auto selection = select_physical_roots(geometry, roots);
    double magnification = 0.0;
    int image_count = 0;
    for (std::size_t i = 0; i < roots.size(); ++i) {
        if (!selection.physical[i]) {
            continue;
        }
        const double jacobian = jacobian_determinant(geometry, roots[i]);
        magnification += 1.0 / std::abs(jacobian);
        ++image_count;
    }
    const double derivative_indicator =
        derivative_error_indicator_from_roots(geometry, roots, selection);
    return {magnification, image_count, derivative_indicator};
}

PointSourceDerivativeResult PointSourceMagnifier::binary_mag0_with_derivatives_cached(
    double separation,
    double mass_ratio,
    SourcePosition source) const
{
    if (separation == 0.0 || mass_ratio <= 0.0) {
        return {};
    }
    const bool cache_matches =
        root_cache_valid_ &&
        root_cache_separation_ == separation &&
        root_cache_mass_ratio_ == mass_ratio &&
        root_cache_source_.x == source.x &&
        root_cache_source_.y == source.y;
    if (!cache_matches) {
        return binary_mag0_with_derivatives(separation, mass_ratio, source);
    }

    const FastBinaryGeometry geometry = make_fast_vbm_geometry(separation, mass_ratio, source);
    std::array<::complex, 5> roots;
    for (std::size_t i = 0; i < roots.size(); ++i) {
        roots[i] = {root_cache_roots_[i].x, root_cache_roots_[i].y};
    }
    const auto selection = select_physical_roots(geometry, roots);
    double magnification = 0.0;
    int image_count = 0;
    for (std::size_t i = 0; i < roots.size(); ++i) {
        if (!selection.physical[i]) {
            continue;
        }
        const double jacobian = jacobian_determinant(geometry, roots[i]);
        magnification += 1.0 / std::abs(jacobian);
        ++image_count;
    }
    const double derivative_indicator =
        derivative_error_indicator_from_roots(geometry, roots, selection);
    return {magnification, image_count, derivative_indicator};
}

PointSourceSafetyDiagnostic PointSourceMagnifier::binary_safety_diagnostic_cached(
    double separation,
    double mass_ratio,
    SourcePosition source) const
{
    if (separation == 0.0 || mass_ratio <= 0.0) {
        return {};
    }
    const bool cache_matches =
        root_cache_valid_ &&
        root_cache_separation_ == separation &&
        root_cache_mass_ratio_ == mass_ratio &&
        root_cache_source_.x == source.x &&
        root_cache_source_.y == source.y;
    if (!cache_matches) {
        (void)binary_mag0_impl(separation, mass_ratio, source, true);
    }

    const FastBinaryGeometry geometry = make_fast_vbm_geometry(
        separation, mass_ratio, source);
    std::array<::complex, 5> roots;
    for (std::size_t i = 0; i < roots.size(); ++i) {
        roots[i] = {root_cache_roots_[i].x, root_cache_roots_[i].y};
    }
    return safety_diagnostic_from_roots(geometry, roots);
}

PointSourceResult PointSourceMagnifier::binary_mag0_impl(
    double separation,
    double mass_ratio,
    SourcePosition source,
    bool use_root_cache) const
{
    if (separation == 0.0 || mass_ratio <= 0.0) {
        return {};
    }

    const FastBinaryGeometry geometry = make_fast_vbm_geometry(separation, mass_ratio, source);
    std::array<::complex, 6> coefficients;
    std::array<::complex, 5> roots;
    binary_polynomial_coefficients(geometry, coefficients);
    // Continuation remains useful for slowly changing orbital separation as
    // well as a static trajectory. The polynomial/Vieta and lens-residual
    // checks below make a geometry jump fail closed to the cold solver.
    const bool can_polish_from_cache =
        use_root_cache &&
        root_cache_valid_;
    bool used_cold_retry = false;
    if (can_polish_from_cache) {
        for (std::size_t i = 0; i < roots.size(); ++i) {
            roots[i] = {root_cache_roots_[i].x, root_cache_roots_[i].y};
        }
        apply_binary_image_predictor(
            make_fast_vbm_geometry(
                root_cache_separation_,
                root_cache_mass_ratio_,
                root_cache_source_),
            geometry,
            roots);
        cmplx_roots_gen(roots.data(), coefficients.data(), 5, true, true);
        if (!polynomial_roots_are_usable(coefficients, roots) ||
            max_physical_residual_squared(geometry, roots) > 1.0e-18) {
            cmplx_roots_gen(roots.data(), coefficients.data(), 5, true, false);
            used_cold_retry = true;
        }
    } else {
        cmplx_roots_gen(roots.data(), coefficients.data(), 5, true, false);
    }
    if (use_root_cache) {
        root_cache_valid_ = true;
        root_cache_separation_ = separation;
        root_cache_mass_ratio_ = mass_ratio;
        root_cache_source_ = source;
        for (std::size_t i = 0; i < roots.size(); ++i) {
            root_cache_roots_[i] = {roots[i].re, roots[i].im};
        }
    }

    auto result = point_source_result_from_roots(geometry, roots);
    result.root_candidate_count = static_cast<int>(roots.size());
    result.root_used_warm_start = can_polish_from_cache ? 1 : 0;
    result.root_used_cold_retry = used_cold_retry ? 1 : 0;
    result.root_max_residual = std::sqrt(
        max_physical_residual_squared(geometry, roots));
    return result;
}

void PointSourceMagnifier::binary_mag0_batch(
    double separation,
    double mass_ratio,
    const SourcePosition* sources,
    double* magnifications,
    std::size_t count) const
{
    if (sources == nullptr || magnifications == nullptr) {
        return;
    }
    if (separation == 0.0 || mass_ratio <= 0.0) {
        for (std::size_t i = 0; i < count; ++i) {
            magnifications[i] = 0.0;
        }
        return;
    }

    const FastBinaryConstants constants = make_fast_vbm_constants(separation, mass_ratio);
    std::array<::complex, 6> coefficients;
    std::array<::complex, 5> roots;
    bool warm = false;
    FastBinaryGeometry previous_geometry;
    for (std::size_t i = 0; i < count; ++i) {
        const FastBinaryGeometry geometry = make_fast_vbm_geometry(constants, sources[i]);
        binary_polynomial_coefficients(geometry, coefficients);
        // Batch sources are spatially contiguous (hexadecapole rings, ring
        // quadrature), so the previous sample's roots are excellent starting
        // points; fall back to a cold solve when polishing drifts.
        if (warm) {
            apply_binary_image_predictor(previous_geometry, geometry, roots);
            cmplx_roots_gen(roots.data(), coefficients.data(), 5, true, true);
            if (!polynomial_roots_are_usable(coefficients, roots) ||
                max_physical_residual_squared(geometry, roots) > 1.0e-18) {
                cmplx_roots_gen(roots.data(), coefficients.data(), 5, true, false);
            }
        } else {
            cmplx_roots_gen(roots.data(), coefficients.data(), 5, true, false);
            warm = true;
        }
        magnifications[i] = point_source_result_from_roots(geometry, roots).magnification;
        previous_geometry = geometry;
    }
}

std::vector<BinaryImage> PointSourceMagnifier::binary_images(
    double separation,
    double mass_ratio,
    SourcePosition source) const
{
    const auto candidates = binary_image_candidates(separation, mass_ratio, source);
    std::vector<BinaryImage> images;
    images.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        if (candidate.physical) {
            images.push_back({candidate.position, candidate.jacobian_determinant});
        }
    }
    return images;
}

std::vector<BinaryImageCandidate> PointSourceMagnifier::binary_image_candidates(
    double separation,
    double mass_ratio,
    SourcePosition source) const
{
    if (separation == 0.0 || mass_ratio <= 0.0) {
        return {};
    }

    const FastBinaryGeometry geometry = make_fast_vbm_geometry(separation, mass_ratio, source);
    std::array<::complex, 6> coefficients;
    std::array<::complex, 5> roots;
    binary_polynomial_coefficients(geometry, coefficients);
    // Use the previous centre/probe solve as a continuation seed even when a
    // dynamic binary changed separation; validation below decides whether it
    // was close enough and retries cold otherwise.
    const bool can_polish_from_cache =
        root_cache_valid_;
    if (can_polish_from_cache) {
        for (std::size_t i = 0; i < roots.size(); ++i) {
            roots[i] = {root_cache_roots_[i].x, root_cache_roots_[i].y};
        }
        apply_binary_image_predictor(
            make_fast_vbm_geometry(
                root_cache_separation_,
                root_cache_mass_ratio_,
                root_cache_source_),
            geometry,
            roots);
        cmplx_roots_gen(roots.data(), coefficients.data(), 5, true, true);
        if (!polynomial_roots_are_usable(coefficients, roots) ||
            max_physical_residual_squared(geometry, roots) > 1.0e-18) {
            cmplx_roots_gen(roots.data(), coefficients.data(), 5, true, false);
        }
    } else {
        cmplx_roots_gen(roots.data(), coefficients.data(), 5, true, false);
    }
    root_cache_valid_ = true;
    root_cache_separation_ = separation;
    root_cache_mass_ratio_ = mass_ratio;
    root_cache_source_ = source;
    for (std::size_t i = 0; i < roots.size(); ++i) {
        root_cache_roots_[i] = {roots[i].re, roots[i].im};
    }

    std::vector<BinaryImageCandidate> images;
    images.reserve(roots.size());
    for (const auto& root : roots) {
        images.push_back({{root.re, root.im},
            jacobian_determinant(geometry, root),
            std::sqrt(residual_squared(geometry, root)),
            false});
    }
    std::sort(images.begin(), images.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.residual < rhs.residual;
    });

    int physical_count = static_cast<int>(images.size());
    if (images.size() >= 5) {
        constexpr double min_ratio = 1.0e-4;
        constexpr double absolute_gap = 1.0e-12;
        if (images[3].residual * min_ratio > images[2].residual + absolute_gap) {
            physical_count = 3;
        }
    }
    for (std::size_t i = 0; i < images.size(); ++i) {
        images[i].physical = static_cast<int>(i) < physical_count;
    }
    return images;
}

SourcePosition PointSourceMagnifier::binary_lens_equation(
    double separation,
    double mass_ratio,
    SourcePosition image) const
{
    const BinaryGeometry geometry = make_vbm_geometry(separation, mass_ratio, {0.0, 0.0});
    const Complex z(image.x, image.y);
    const Complex zc = std::conj(z);
    const Complex shifted_source =
        z - geometry.m1 / (zc - geometry.separation) - geometry.m2 / zc;
    const Complex source = shifted_source - geometry.separation * geometry.m1;
    return {source.real(), source.imag()};
}

double triple_derivative_error_indicator(
    const model::TripleLensGeometry& geometry,
    const std::vector<TripleImageCandidate>& candidates)
{
    double indicator = 0.0;
    for (const auto& candidate : candidates) {
        if (!candidate.physical) {
            continue;
        }
        const Complex z(candidate.position.x, candidate.position.y);
        Complex j1(0.0, 0.0);
        Complex j2(0.0, 0.0);
        Complex j3(0.0, 0.0);
        for (int k = 0; k < 3; ++k) {
            const Complex zk(geometry.lens_positions[k].x, geometry.lens_positions[k].y);
            const Complex dz = z - zk;
            const Complex dz2 = dz * dz;
            const Complex dz3 = dz2 * dz;
            const Complex dz4 = dz3 * dz;
            const double mk = geometry.masses[k];
            j1 += mk / dz2;
            j2 += Complex(-2.0 * mk, 0.0) / dz3;
            j3 += Complex(6.0 * mk, 0.0) / dz4;
        }
        const Complex j1c = std::conj(j1);
        const double det_j = 1.0 - std::real(j1 * j1c);
        if (det_j == 0.0 || !std::isfinite(det_j)) {
            return std::numeric_limits<double>::infinity();
        }
        const Complex j1c2 = j1c * j1c;
        const Complex j3_mod = j3 * j1c2;
        const double j2_abs2 = std::norm(j2);
        const double det_j2 = det_j * det_j;
        const double ob2 = j2_abs2 * (6.0 - 6.0 * det_j + det_j2);
        const Complex j2_mod = j2 * j2 * j1c2 * j1c;
        const double denominator = std::abs(det_j * det_j2 * det_j2);
        if (denominator == 0.0 || !std::isfinite(denominator)) {
            return std::numeric_limits<double>::infinity();
        }
        const double cq =
            0.5 *
            (std::abs(ob2 - 6.0 * std::real(j2_mod) - 2.0 * std::real(j3_mod) * det_j) +
             3.0 * std::abs(std::imag(j2_mod))) /
            denominator;
        if (std::isfinite(cq)) {
            indicator += cq;
        } else {
            return std::numeric_limits<double>::infinity();
        }
    }
    return indicator;
}

bool triple_geometry_equals(
    const model::TripleLensGeometry& a,
    const model::TripleLensGeometry& b)
{
    for (int i = 0; i < 3; ++i) {
        if (a.lens_positions[i].x != b.lens_positions[i].x ||
            a.lens_positions[i].y != b.lens_positions[i].y ||
            a.masses[i] != b.masses[i]) {
            return false;
        }
    }
    return true;
}

PointSourceResult triple_diagnostics_from_candidates(
    const std::vector<TripleImageCandidate>& candidates)
{
    PointSourceResult diagnostics;
    diagnostics.root_candidate_count = static_cast<int>(candidates.size());
    for (const auto& candidate : candidates) {
        diagnostics.root_max_residual =
            std::max(diagnostics.root_max_residual, candidate.residual);
        if (candidate.residual > 1.0e-7) {
            ++diagnostics.root_polish_failure_count;
        }
        if (candidate.physical) {
            ++diagnostics.image_count;
        }
    }
    return diagnostics;
}

PointSourceResult PointSourceMagnifier::triple_mag0(
    const model::TripleLensGeometry& geometry,
    SourcePosition source) const
{
    const auto& candidates = triple_image_candidates_cached(geometry, source);
    double magnification = 0.0;
    int image_count = 0;
    for (const auto& candidate : candidates) {
        if (!candidate.physical) {
            continue;
        }
        magnification += 1.0 / std::abs(candidate.jacobian_determinant);
        ++image_count;
    }
    auto result = triple_candidate_cache_diagnostics_;
    result.magnification = magnification;
    result.image_count = image_count;
    return result;
}

PointSourceDerivativeResult PointSourceMagnifier::triple_mag0_with_derivatives(
    const model::TripleLensGeometry& geometry,
    SourcePosition source) const
{
    const bool cache_hit =
        triple_candidate_cache_valid_ &&
        triple_candidate_cache_source_.x == source.x &&
        triple_candidate_cache_source_.y == source.y &&
        triple_geometry_equals(triple_candidate_cache_geometry_, geometry);
    const std::vector<TripleImageCandidate>* candidates_ptr = nullptr;
    std::vector<TripleImageCandidate> fresh_candidates;
    if (cache_hit) {
        candidates_ptr = &triple_candidate_cache_;
    } else {
        fresh_candidates = triple_image_candidates(geometry, source);
        candidates_ptr = &fresh_candidates;
    }
    const auto& candidates = *candidates_ptr;
    double magnification = 0.0;
    int image_count = 0;
    for (const auto& candidate : candidates) {
        if (!candidate.physical) {
            continue;
        }
        magnification += 1.0 / std::abs(candidate.jacobian_determinant);
        ++image_count;
    }
    const double indicator = triple_derivative_error_indicator(geometry, candidates);
    return {magnification, image_count, indicator};
}

std::vector<TripleImage> PointSourceMagnifier::triple_images(
    const model::TripleLensGeometry& geometry,
    SourcePosition source) const
{
    const auto candidates = triple_image_candidates(geometry, source);
    std::vector<TripleImage> images;
    images.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        if (candidate.physical) {
            images.push_back({candidate.position, candidate.jacobian_determinant});
        }
    }
    return images;
}

std::vector<TripleImageCandidate> PointSourceMagnifier::triple_image_candidates(
    const model::TripleLensGeometry& geometry,
    SourcePosition source) const
{
    // A direct solve replaces the shared diagnostics below without replacing
    // the one-entry candidate cache.  Invalidate it up front so a later cache
    // lookup can never combine candidates from one source with diagnostics
    // from another.  The cached entry point marks the new pair valid only
    // after both have been produced.
    triple_candidate_cache_valid_ = false;
    const auto coefficients = triple_polynomial_coefficients(geometry, source);
    constexpr int kTripleDegree = 10;
    PointSourceResult diagnostics;

    // Polish each root on the true lens equation, then deduplicate.
    // Multiple spurious starting points can converge to the same physical image;
    // deduplication ensures each image is counted once.
    constexpr double kDedupTol2 = 1.0e-16;  // squared distance threshold
    std::vector<TripleImageCandidate> images;
    std::vector<Complex> solved_roots;
    const auto collect_candidates = [&](const Complex* roots, std::size_t count, bool high_precision) {
        images.clear();
        images.reserve(count);
        diagnostics.root_duplicate_count = 0;
        diagnostics.root_polish_failure_count = 0;
        diagnostics.root_max_residual = 0.0;
        if (high_precision) {
            diagnostics.root_used_high_precision = 1;
        }
        for (std::size_t k = 0; k < count; ++k) {
            const Complex root = roots[k];
            if (!std::isfinite(root.real()) || !std::isfinite(root.imag())) {
                ++diagnostics.root_polish_failure_count;
                continue;
            }
            const SourcePosition z0 {root.real(), root.imag()};
            const SourcePosition z_polished = high_precision
                ? polish_triple_image_root_high_precision(geometry, source, z0)
                : polish_triple_image_root(geometry, source, z0);
            bool is_dup = false;
            for (const auto& existing : images) {
                const double ddx = z_polished.x - existing.position.x;
                const double ddy = z_polished.y - existing.position.y;
                if (ddx * ddx + ddy * ddy < kDedupTol2) {
                    is_dup = true;
                    break;
                }
            }
            if (is_dup) {
                ++diagnostics.root_duplicate_count;
                continue;
            }
            const Complex z_cmplx {z_polished.x, z_polished.y};
            const double residual = high_precision
                ? triple_residual_high_precision(geometry, source, z_polished)
                : triple_residual(geometry, source, z_cmplx);
            diagnostics.root_max_residual = std::max(diagnostics.root_max_residual, residual);
            if (!std::isfinite(residual) || residual > 1.0e-7) {
                ++diagnostics.root_polish_failure_count;
            }
            images.push_back({z_polished,
                high_precision
                    ? triple_jacobian_determinant_high_precision(geometry, z_polished)
                    : triple_jacobian_determinant(geometry, z_cmplx),
                residual,
                false});
        }
        diagnostics.root_candidate_count = static_cast<int>(images.size());
    };

    if (static_cast<int>(coefficients.size()) == kTripleDegree + 1 &&
        std::abs(coefficients[kTripleDegree]) > 0.0) {
        // Full-degree fast path: solve directly with fixed-size buffers and
        // warm-start from the previous solve of the same geometry (probe and
        // light-curve sweeps move the source continuously, so the previous
        // roots are excellent starting points). Validate the polynomial roots
        // before lens-equation polishing and retry cold only when the warm
        // solve itself is incomplete. The physical lens equation normally has
        // 4, 6, 8, or 10 images, so counting deduplicated physical candidates
        // against the polynomial degree would retry every healthy warm solve.
        std::array<::complex, kTripleDegree + 1> sg_coefficients;
        for (int k = 0; k <= kTripleDegree; ++k) {
            sg_coefficients[static_cast<std::size_t>(k)] = {
                coefficients[static_cast<std::size_t>(k)].real(),
                coefficients[static_cast<std::size_t>(k)].imag()};
        }
        std::array<::complex, kTripleDegree> sg_roots;
        std::array<Complex, kTripleDegree> roots;
        bool continuation_failed = false;
        const auto run_solve = [&](bool use_starting_points) {
            if (use_starting_points) {
                continuation_failed = !continue_triple_polynomial_roots(
                    coefficients, triple_warm_roots_);
                if (continuation_failed) {
                    cmplx_roots_gen(
                        sg_roots.data(), sg_coefficients.data(),
                        kTripleDegree, true, false);
                    for (int k = 0; k < kTripleDegree; ++k) {
                        triple_warm_roots_[static_cast<std::size_t>(k)] = {
                            sg_roots[static_cast<std::size_t>(k)].re,
                            sg_roots[static_cast<std::size_t>(k)].im};
                    }
                }
            } else {
                cmplx_roots_gen(
                    sg_roots.data(), sg_coefficients.data(), kTripleDegree,
                    true, false);
                for (int k = 0; k < kTripleDegree; ++k) {
                    triple_warm_roots_[static_cast<std::size_t>(k)] = {
                        sg_roots[static_cast<std::size_t>(k)].re,
                        sg_roots[static_cast<std::size_t>(k)].im};
                }
            }
            solved_roots.clear();
            solved_roots.reserve(static_cast<std::size_t>(kTripleDegree));
            for (int k = 0; k < kTripleDegree; ++k) {
                roots[static_cast<std::size_t>(k)] =
                    triple_warm_roots_[static_cast<std::size_t>(k)];
                solved_roots.push_back(roots[static_cast<std::size_t>(k)]);
            }
        };
        const bool warm =
            triple_warm_valid_ &&
            same_triple_basis_geometry(triple_warm_geometry_, geometry);
        diagnostics.root_used_warm_start = warm ? 1 : 0;
        run_solve(warm);
        const bool polynomial_roots_valid =
            polynomial_roots_are_usable(coefficients, roots);
        if (warm && (continuation_failed || !polynomial_roots_valid)) {
            if (continuation_failed) {
                diagnostics.root_used_cold_retry = 1;
            } else {
                run_solve(false);
                diagnostics.root_used_cold_retry = 1;
            }
        }
        collect_candidates(roots.data(), roots.size(), false);
        triple_warm_geometry_ = geometry;
        triple_warm_valid_ = true;
    } else {
        const auto roots = math::PolynomialRootSolver().solve(coefficients);
        if (roots.status != math::RootSolverStatus::ok) {
            triple_candidate_cache_diagnostics_ = diagnostics;
            return {};
        }
        solved_roots.assign(roots.roots.begin(), roots.roots.end());
        collect_candidates(roots.roots.data(), roots.roots.size(), false);
    }
    std::sort(images.begin(), images.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.residual < rhs.residual;
    });

    constexpr double absolute_tolerance = 1.0e-7;
    constexpr double relative_gap = 1.0e-4;
    int physical_count = 0;
    for (const auto& image : images) {
        if (image.residual <= absolute_tolerance) {
            ++physical_count;
        }
    }
    if (physical_count == 0 && !images.empty()) {
        physical_count = 1;
    }
    for (std::size_t i = 1; i < images.size(); ++i) {
        if (images[i].residual * relative_gap > images[i - 1].residual + absolute_tolerance) {
            physical_count = std::max(physical_count, static_cast<int>(i));
            break;
        }
    }
    // A triple lens produces 4, 6, 8 or 10 images, so an odd classification
    // is inconsistent.  Resolve it toward the smaller residual jump: include
    // the next candidate when its residual is close to the last accepted one,
    // otherwise drop the worst accepted image.  Counts below 5 are left
    // alone: they indicate a degenerate solve where forcing parity would do
    // more harm than good.
    if (physical_count % 2 == 1 && physical_count >= 5) {
        const std::size_t last = static_cast<std::size_t>(physical_count) - 1;
        const bool can_extend = static_cast<std::size_t>(physical_count) < images.size();
        if (can_extend &&
            images[static_cast<std::size_t>(physical_count)].residual <=
                10.0 * (images[last].residual + absolute_tolerance)) {
            ++physical_count;
        } else {
            --physical_count;
        }
    }
    const bool suspicious =
        diagnostics.root_polish_failure_count > 0 ||
        physical_count < 4 ||
        physical_count > kTripleDegree ||
        physical_count % 2 != 0;
    if (suspicious && diagnostics.root_used_high_precision == 0) {
        diagnostics.root_needs_high_precision = 1;
        if (solved_roots.size() == kTripleDegree) {
            std::array<Complex, kTripleDegree> quad_roots;
            if (solve_triple_polynomial_quadruple(
                    geometry, source, solved_roots, quad_roots)) {
                diagnostics.root_used_quad_precision = 1;
                solved_roots.assign(quad_roots.begin(), quad_roots.end());
                // Keep the improved roots as the next continuation seed.  The
                // ordinary path still validates this seed with its double
                // polynomial/Vieta guard before using it.
                for (std::size_t i = 0; i < quad_roots.size(); ++i) {
                    triple_warm_roots_[i] = quad_roots[i];
                }
            }
        }
        collect_candidates(solved_roots.data(), solved_roots.size(), true);
        std::sort(images.begin(), images.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.residual < rhs.residual;
        });
        physical_count = 0;
        for (const auto& image : images) {
            if (image.residual <= absolute_tolerance) {
                ++physical_count;
            }
        }
        if (physical_count == 0 && !images.empty()) {
            physical_count = 1;
        }
        for (std::size_t i = 1; i < images.size(); ++i) {
            if (images[i].residual * relative_gap > images[i - 1].residual + absolute_tolerance) {
                physical_count = std::max(physical_count, static_cast<int>(i));
                break;
            }
        }
        if (physical_count % 2 == 1 && physical_count >= 5) {
            const std::size_t last = static_cast<std::size_t>(physical_count) - 1;
            const bool can_extend = static_cast<std::size_t>(physical_count) < images.size();
            if (can_extend &&
                images[static_cast<std::size_t>(physical_count)].residual <=
                    10.0 * (images[last].residual + absolute_tolerance)) {
                ++physical_count;
            } else {
                --physical_count;
            }
        }
    }
    for (std::size_t i = 0; i < images.size(); ++i) {
        images[i].physical = static_cast<int>(i) < physical_count;
    }
    auto final_diagnostics = triple_diagnostics_from_candidates(images);
    final_diagnostics.root_duplicate_count = diagnostics.root_duplicate_count;
    final_diagnostics.root_polish_failure_count = diagnostics.root_polish_failure_count;
    final_diagnostics.root_used_warm_start = diagnostics.root_used_warm_start;
    final_diagnostics.root_used_cold_retry = diagnostics.root_used_cold_retry;
    final_diagnostics.root_used_high_precision = diagnostics.root_used_high_precision;
    final_diagnostics.root_used_quad_precision = diagnostics.root_used_quad_precision;
    final_diagnostics.root_needs_high_precision = diagnostics.root_needs_high_precision;
    final_diagnostics.root_max_residual =
        std::max(final_diagnostics.root_max_residual, diagnostics.root_max_residual);
    triple_candidate_cache_diagnostics_ = final_diagnostics;
    return images;
}

const std::vector<TripleImageCandidate>&
PointSourceMagnifier::triple_image_candidates_cached(
    const model::TripleLensGeometry& geometry,
    SourcePosition source) const
{
    const bool cache_hit =
        triple_candidate_cache_valid_ &&
        triple_candidate_cache_source_.x == source.x &&
        triple_candidate_cache_source_.y == source.y &&
        triple_geometry_equals(triple_candidate_cache_geometry_, geometry);
    if (!cache_hit) {
        triple_candidate_cache_ = triple_image_candidates(geometry, source);
        triple_candidate_cache_valid_ = true;
        triple_candidate_cache_geometry_ = geometry;
        triple_candidate_cache_source_ = source;
    }
    return triple_candidate_cache_;
}

SourcePosition PointSourceMagnifier::triple_lens_equation(
    const model::TripleLensGeometry& geometry,
    SourcePosition image) const
{
    return model::triple_lens_equation(geometry, image);
}

} // namespace lcbinint::magnification
