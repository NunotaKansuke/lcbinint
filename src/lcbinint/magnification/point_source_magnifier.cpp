#include "lcbinint/magnification/point_source_magnifier.hpp"

#include "SkowronGould.h"
#include "lcbinint/math/polynomial_roots.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
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
    double max_residual_squared = 0.0;
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
        if (selection.physical[i]) {
            selection.max_residual_squared =
                std::max(selection.max_residual_squared, candidates[i].residual);
        }
    }
    return selection;
}

PointSourceResult point_source_result_from_roots(
    const FastBinaryGeometry& geometry,
    const std::array<::complex, 5>& roots,
    const PhysicalRootSelection* selection_hint = nullptr)
{
    const auto selection = selection_hint != nullptr
        ? *selection_hint : select_physical_roots(geometry, roots);
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
    return select_physical_roots(geometry, roots).max_residual_squared;
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
    const std::array<::complex, 5>& roots,
    std::array<bool, 5>* physical_cache,
    std::array<SourcePosition, 5>* derivative_cache,
    const std::array<bool, 5>* physical_hint = nullptr)
{
    const auto selection = physical_hint != nullptr
        ? PhysicalRootSelection{*physical_hint} : select_physical_roots(geometry, roots);
    *physical_cache = selection.physical;
    PointSourceSafetyDiagnostic result;
    const ::complex a(geometry.a, 0.0);
    const ::complex m1(geometry.m1, 0.0);
    const ::complex m2(geometry.m2, 0.0);
    const ::complex source_conjugate(geometry.y.re, -geometry.y.im);

    for (std::size_t i = 0; i < roots.size(); ++i) {
        const ::complex z = roots[i];
        const ::complex dza = z - a;
        // Form the reciprocal of each lens displacement once. All derivative
        // orders and the ghost map share these powers, instead of dividing
        // again by independently formed square/cube/fourth-power denominators.
        const ::complex inv_a = 1.0 / dza;
        const ::complex inv_z = 1.0 / z;
        const ::complex inv_a2 = inv_a * inv_a;
        const ::complex inv_z2 = inv_z * inv_z;
        const ::complex f1 = m1 * inv_a2 + m2 * inv_z2;
        const ::complex f2 = -2.0 * (m1 * inv_a2 * inv_a + m2 * inv_z2 * inv_z);
        const ::complex f1c = conj(f1);
        (*derivative_cache)[i] = {f1c.re, f1c.im};
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

            const ::complex f3 = 6.0 * (m1 * inv_a2 * inv_a2 + m2 * inv_z2 * inv_z2);
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
        const ::complex f0 = -m1 * inv_a - m2 * inv_z;
        const ::complex zhat = source_conjugate - f0;
        const ::complex zhat_a = zhat - a;
        const ::complex f1_hat = m1 / (zhat_a * zhat_a) + m2 / (zhat * zhat);
        const ::complex jhat = 1.0 - f1 * f1_hat;
        const ::complex f2_zc = conj(f2);
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
    constexpr int kMaxIter = 80;
    constexpr long double kTol = 1.0e-18L;
    long double zx = static_cast<long double>(z.x);
    long double zy = static_cast<long double>(z.y);
    const long double source_x = static_cast<long double>(source.x);
    const long double source_y = static_cast<long double>(source.y);
    for (int iter = 0; iter < kMaxIter; ++iter) {
        long double sx = 0.0L, sy = 0.0L, dxx = 0.0L, dxy = 0.0L;
        for (std::size_t i = 0; i < geometry.lens_positions.size(); ++i) {
            const long double lens_x = static_cast<long double>(geometry.lens_positions[i].x);
            const long double lens_y = static_cast<long double>(geometry.lens_positions[i].y);
            const long double mass = static_cast<long double>(geometry.masses[i]);
            const long double dx = zx - lens_x;
            const long double dy = zy - lens_y;
            const long double d2 = dx * dx + dy * dy;
            if (d2 == 0.0L) {
                return z;
            }
            const long double d4 = d2 * d2;
            sx += mass * dx / d2;
            sy += mass * dy / d2;
            dxx += mass * (dy * dy - dx * dx) / d4;
            dxy += mass * 2.0L * dx * dy / d4;
        }
        const long double fx = zx - sx - source_x;
        const long double fy = zy - sy - source_y;
        if ((fx < 0.0L ? -fx : fx) + (fy < 0.0L ? -fy : fy) < kTol) { break; }
        const long double j00 = 1.0L - dxx;
        const long double j01 = dxy;
        const long double j11 = 1.0L + dxx;
        const long double det = j00 * j11 - j01 * j01;
        const long double abs_det = det < 0.0L ? -det : det;
        if (abs_det < 1.0e-35L) { break; }
        zx -= (j11 * fx - j01 * fy) / det;
        zy -= (j00 * fy - j01 * fx) / det;
    }
    return {static_cast<double>(zx), static_cast<double>(zy)};
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
    static_assert(CoefficientCount == Degree + 1);
    // Same residual and complete-set/Vieta guards as the generic validator.
    // Evaluate coefficient magnitudes once, and compare squared norms after
    // multiplying the monic coefficient test by the leading coefficient.
    // This removes repeated hypot and complex division from every epoch/probe.
    const auto norm2 = [](FastComplex z) { return z.re*z.re + z.im*z.im; };
    std::array<double, CoefficientCount> magnitudes;
    for (std::size_t k = 0; k < CoefficientCount; ++k) {
        const auto c = sg_to_fast(coefficients[k]);
        if (!std::isfinite(c.re) || !std::isfinite(c.im)) return false;
        magnitudes[k] = std::sqrt(norm2(c));
    }
    for (const auto root : roots) {
        const auto z = sg_to_fast(root);
        if (!std::isfinite(z.re) || !std::isfinite(z.im)) return false;
        FastComplex value {};
        double scale = 0.0;
        const double radius = std::sqrt(norm2(z));
        for (std::size_t k = CoefficientCount; k-- > 0;) {
            value = value*z + sg_to_fast(coefficients[k]);
            scale = scale*radius + magnitudes[k];
        }
        const double allowance = 1.0e-8*std::max(scale, 1.0);
        if (!std::isfinite(scale) || !std::isfinite(norm2(value)) ||
            norm2(value) > allowance*allowance) return false;
    }
    std::array<FastComplex, CoefficientCount> reconstructed {};
    reconstructed[0] = {1.0, 0.0};
    for (std::size_t i = 0; i < Degree; ++i) {
        const auto root = sg_to_fast(roots[i]);
        for (std::size_t k = i+1; k > 0; --k) {
            reconstructed[k] = reconstructed[k-1] - root*reconstructed[k];
        }
        reconstructed[0] = FastComplex{} - root*reconstructed[0];
    }
    const auto leading = sg_to_fast(coefficients[Degree]);
    const double leading_norm = norm2(leading);
    if (!(leading_norm > 0.0 && std::isfinite(leading_norm))) return false;
    for (std::size_t k = 0; k < CoefficientCount; ++k) {
        const auto expected = sg_to_fast(coefficients[k]);
        const double difference = norm2(leading*reconstructed[k] - expected);
        if (!std::isfinite(difference) ||
            difference > 1.0e-14*std::max(norm2(expected), leading_norm)) return false;
    }
    return true;
}

// Same Adams stopping rule, final Newton step and fractional jumps as
// cmplx_newton_spec. A fixed degree lets the compiler unroll the common
// five-term Horner sweep; retain SG itself for unusual/failed iterations.
bool polish_binary_root_fixed_degree(
    const std::array<::complex, 6>& coefficients, ::complex& root)
{
    constexpr std::array<double, 10> fractions {
        0.64109297, 0.91577881, 0.25921289, 0.50487203, 0.08177045,
        0.13653241, 0.306162, 0.37794326, 0.04618805, 0.75132137};
    if (coefficients[5] == ::complex{}) return false;
    double stopping_squared = 0.0;
    for (int iteration = 1; iteration <= MAXIT; ++iteration) {
        ::complex p = coefficients[5], derivative {};
        if (iteration % 10 == 1) {
            double error = abs(p);
            const double radius = abs(root);
            for (int k = 4; k >= 0; --k) {
                derivative = p + derivative * root;
                p = coefficients[k] + p * root;
                error = radius * error + abs(p);
            }
            const double stopping = 2.0e-15*error;
            stopping_squared = stopping*stopping;
        } else {
            for (int k = 4; k >= 0; --k) {
                derivative = p + derivative * root;
                p = coefficients[k] + p * root;
            }
        }
        const double residual = abs2(p);
        if (residual == 0.0) return true;
        const bool near_root = residual < stopping_squared;
        if (near_root && (derivative == ::complex{} ||
                          residual < 0.01*stopping_squared)) return true;
        // The SG zero-derivative rescue is rare for continuation. Leave it
        // to the original routine, together with nonfinite iterates.
        if (derivative == ::complex{} || !std::isfinite(residual)) return false;
        const ::complex delta = p / derivative;
        const ::complex next = root - delta;
        if (next == root) return true;
        if (near_root) { root = next; return true; }
        root = iteration % 10 == 0
            ? root - fractions[(iteration/10-1)%10]*delta : next;
    }
    return false;
}

bool polish_binary_root_set(
    std::array<::complex, 6>& coefficients, std::array<::complex, 5>& roots)
{
    // With five retained roots there is no reason to deflate/re-factor the
    // polynomial just to update each source sample. Polish independently,
    // then require the same residual and complete-set/Vieta validation as
    // before. A coalesced/missing root or failed Newton step retries cold.
    for (auto& root : roots) {
        const auto starting_root = root;
        if (polish_binary_root_fixed_degree(coefficients, root)) continue;
        root = starting_root;
        int iterations = 0;
        bool succeeded = false;
        cmplx_newton_spec(coefficients.data(), 5, &root, iterations, succeeded);
        if (!succeeded) return false;
    }
    return true;
}

void apply_binary_image_predictor(
    const FastBinaryGeometry& previous_geometry,
    const FastBinaryGeometry& next_geometry,
    std::array<::complex, 5>& roots,
    const std::array<bool, 5>* physical_hint = nullptr,
    const std::array<SourcePosition, 5>* derivative_hint = nullptr)
{
    // The analytic predictor assumes fixed lens positions and masses. Dynamic
    // geometries still use the retained roots as zero-order continuation
    // seeds and are protected by the ordinary validation/cold fallback.
    if (previous_geometry.a != next_geometry.a ||
        previous_geometry.m1 != next_geometry.m1 ||
        previous_geometry.m2 != next_geometry.m2) {
        return;
    }
    const FastComplex delta_source {
        next_geometry.y.re - previous_geometry.y.re,
        next_geometry.y.im - previous_geometry.y.im,
    };
    const double source_step_squared =
        delta_source.re*delta_source.re + delta_source.im*delta_source.im;
    // Do not extrapolate across the end-to-start jump of a repeated light
    // curve. The predictor is intended for locally contiguous trajectory and
    // probe samples; zero-order continuation remains the safer large-step
    // seed.
    if (!(source_step_squared > 0.0) || source_step_squared > 0.25*0.25) {
        return;
    }

    const auto physical = physical_hint != nullptr
        ? PhysicalRootSelection{*physical_hint}
        : select_physical_roots(previous_geometry, roots);
    constexpr double minimum_abs_jacobian = 1.0e-4;
    constexpr double maximum_image_step = 0.5;
    for (std::size_t i = 0; i < roots.size(); ++i) {
        // The two rejected roots of a three-image binary solution are roots of
        // the eliminated polynomial, not physical lens-equation images. The
        // local inverse lens Jacobian therefore applies only to physical roots.
        if (!physical.physical[i]) {
            continue;
        }
        FastComplex kappa;
        if (derivative_hint != nullptr) {
            kappa = {(*derivative_hint)[i].x, (*derivative_hint)[i].y};
        } else {
            const double x = roots[i].re, y = roots[i].im;
            const double xa = x-previous_geometry.a;
            const double da = xa*xa+y*y, dz = x*x+y*y;
            if (!(da > 0.0 && dz > 0.0)) continue;
            const double ta = previous_geometry.m1/(da*da);
            const double tz = previous_geometry.m2/(dz*dz);
            kappa = {ta*(xa*xa-y*y)+tz*(x*x-y*y), 2.0*y*(ta*xa+tz*x)};
        }
        const double jacobian = 1.0 - (kappa.re*kappa.re+kappa.im*kappa.im);
        if (!std::isfinite(jacobian) ||
            std::abs(jacobian) < minimum_abs_jacobian) {
            continue;
        }
        // The real inverse Jacobian gives the same analytic predictor without
        // complex division or hypot. Its guard is still the same image-step
        // radius; the ordinary polynomial/lens checks validate the result.
        const double inverse_jacobian = 1.0/jacobian;
        const FastComplex delta_image {
            ((1.0-kappa.re)*delta_source.re-kappa.im*delta_source.im)*inverse_jacobian,
            ((1.0+kappa.re)*delta_source.im-kappa.im*delta_source.re)*inverse_jacobian,
        };
        if (!std::isfinite(delta_image.re) ||
            !std::isfinite(delta_image.im) ||
            delta_image.re*delta_image.re+delta_image.im*delta_image.im >
                maximum_image_step*maximum_image_step) {
            continue;
        }
        roots[i] = {
            roots[i].re + delta_image.re,
            roots[i].im + delta_image.im,
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
        // The safety pass also computes magnification and physical-image
        // count. Do not traverse/classify the same roots first for a scalar
        // result that this caller would immediately discard.
        (void)binary_mag0_impl(separation, mass_ratio, source, true, false);
    } else if (root_cache_safety_valid_) {
        return root_cache_safety_;
    }

    const FastBinaryGeometry geometry = make_fast_vbm_geometry(
        separation, mass_ratio, source);
    std::array<::complex, 5> roots;
    for (std::size_t i = 0; i < roots.size(); ++i) {
        roots[i] = {root_cache_roots_[i].x, root_cache_roots_[i].y};
    }
    root_cache_safety_ = safety_diagnostic_from_roots(
        geometry, roots, &root_cache_physical_, &root_cache_image_derivative_,
        root_cache_selection_valid_ ? &root_cache_physical_ : nullptr);
    root_cache_selection_valid_ = true;
    root_cache_safety_valid_ = true;
    return root_cache_safety_;
}

PointSourceResult PointSourceMagnifier::binary_mag0_impl(
    double separation,
    double mass_ratio,
    SourcePosition source,
    bool use_root_cache,
    bool evaluate_magnification) const
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
    PhysicalRootSelection selection;
    bool selection_valid = false;
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
            roots,
            root_cache_safety_valid_ ? &root_cache_physical_ : nullptr,
            root_cache_safety_valid_ ? &root_cache_image_derivative_ : nullptr);
        const bool polynomial_valid = polish_binary_root_set(coefficients, roots) &&
            polynomial_roots_are_usable(coefficients, roots);
        if (polynomial_valid) {
            selection = select_physical_roots(geometry, roots);
            selection_valid = selection.max_residual_squared <= 1.0e-18;
        }
        if (!selection_valid) {
            cmplx_roots_gen(roots.data(), coefficients.data(), 5, true, false);
            used_cold_retry = true;
        }
    } else {
        cmplx_roots_gen(roots.data(), coefficients.data(), 5, true, false);
    }
    if (!selection_valid) selection = select_physical_roots(geometry, roots);
    if (use_root_cache) {
        root_cache_safety_valid_ = false;
        root_cache_selection_valid_ = true;
        root_cache_physical_ = selection.physical;
        root_cache_valid_ = true;
        root_cache_separation_ = separation;
        root_cache_mass_ratio_ = mass_ratio;
        root_cache_source_ = source;
        for (std::size_t i = 0; i < roots.size(); ++i) {
            root_cache_roots_[i] = {roots[i].re, roots[i].im};
        }
    }

    if (!evaluate_magnification) return {};
    auto result = point_source_result_from_roots(geometry, roots, &selection);
    result.root_candidate_count = static_cast<int>(roots.size());
    result.root_used_warm_start = can_polish_from_cache ? 1 : 0;
    result.root_used_cold_retry = used_cold_retry ? 1 : 0;
    result.root_max_residual = std::sqrt(selection.max_residual_squared);
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
            if (!polish_binary_root_set(coefficients, roots) ||
                !polynomial_roots_are_usable(coefficients, roots) ||
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
    const bool same_cached_source = root_cache_valid_ &&
        root_cache_separation_ == separation &&
        root_cache_mass_ratio_ == mass_ratio &&
        root_cache_source_.x == source.x && root_cache_source_.y == source.y;
    // Use the previous centre/probe solve as a continuation seed even when a
    // dynamic binary changed separation; validation below decides whether it
    // was close enough and retries cold otherwise.
    const bool can_polish_from_cache =
        root_cache_valid_;
    if (same_cached_source) {
        // Image seeds requested after routing use the already validated
        // centre roots, even if hexadecapole probes were evaluated in between.
        for (std::size_t i = 0; i < roots.size(); ++i) {
            roots[i] = {root_cache_roots_[i].x, root_cache_roots_[i].y};
        }
    } else if (can_polish_from_cache) {
        binary_polynomial_coefficients(geometry, coefficients);
        for (std::size_t i = 0; i < roots.size(); ++i) {
            roots[i] = {root_cache_roots_[i].x, root_cache_roots_[i].y};
        }
        apply_binary_image_predictor(
            make_fast_vbm_geometry(
                root_cache_separation_,
                root_cache_mass_ratio_,
                root_cache_source_),
            geometry,
            roots,
            root_cache_safety_valid_ ? &root_cache_physical_ : nullptr,
            root_cache_safety_valid_ ? &root_cache_image_derivative_ : nullptr);
        if (!polish_binary_root_set(coefficients, roots) ||
            !polynomial_roots_are_usable(coefficients, roots) ||
            max_physical_residual_squared(geometry, roots) > 1.0e-18) {
            cmplx_roots_gen(roots.data(), coefficients.data(), 5, true, false);
        }
    } else {
        binary_polynomial_coefficients(geometry, coefficients);
        cmplx_roots_gen(roots.data(), coefficients.data(), 5, true, false);
    }
    if (!same_cached_source) {
        root_cache_safety_valid_ = false;
        root_cache_selection_valid_ = false;
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
            const double residual = triple_residual(geometry, source, z_cmplx);
            diagnostics.root_max_residual = std::max(diagnostics.root_max_residual, residual);
            if (!std::isfinite(residual) || residual > 1.0e-7) {
                ++diagnostics.root_polish_failure_count;
            }
            images.push_back({z_polished,
                triple_jacobian_determinant(geometry, z_cmplx),
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
