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
    const bool can_polish_from_cache =
        use_root_cache &&
        root_cache_valid_ &&
        root_cache_separation_ == separation &&
        root_cache_mass_ratio_ == mass_ratio;
    if (can_polish_from_cache) {
        for (std::size_t i = 0; i < roots.size(); ++i) {
            roots[i] = {root_cache_roots_[i].x, root_cache_roots_[i].y};
        }
        cmplx_roots_gen(roots.data(), coefficients.data(), 5, true, true);
        if (max_physical_residual_squared(geometry, roots) > 1.0e-18) {
            cmplx_roots_gen(roots.data(), coefficients.data(), 5, true, false);
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

    return point_source_result_from_roots(geometry, roots);
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
    for (std::size_t i = 0; i < count; ++i) {
        const FastBinaryGeometry geometry = make_fast_vbm_geometry(constants, sources[i]);
        binary_polynomial_coefficients(geometry, coefficients);
        // Batch sources are spatially contiguous (hexadecapole rings, ring
        // quadrature), so the previous sample's roots are excellent starting
        // points; fall back to a cold solve when polishing drifts.
        if (warm) {
            cmplx_roots_gen(roots.data(), coefficients.data(), 5, true, true);
            if (max_physical_residual_squared(geometry, roots) > 1.0e-18) {
                cmplx_roots_gen(roots.data(), coefficients.data(), 5, true, false);
            }
        } else {
            cmplx_roots_gen(roots.data(), coefficients.data(), 5, true, false);
            warm = true;
        }
        magnifications[i] = point_source_result_from_roots(geometry, roots).magnification;
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
    cmplx_roots_gen(roots.data(), coefficients.data(), 5, true, false);

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

PointSourceResult PointSourceMagnifier::triple_mag0(
    const model::TripleLensGeometry& geometry,
    SourcePosition source) const
{
    const auto candidates = triple_image_candidates(geometry, source);
    // Populate cache so triple_mag0_with_derivatives can reuse these candidates.
    triple_candidate_cache_valid_ = true;
    triple_candidate_cache_geometry_ = geometry;
    triple_candidate_cache_source_ = source;
    triple_candidate_cache_ = candidates;
    double magnification = 0.0;
    int image_count = 0;
    for (const auto& candidate : candidates) {
        if (!candidate.physical) {
            continue;
        }
        magnification += 1.0 / std::abs(candidate.jacobian_determinant);
        ++image_count;
    }
    return {magnification, image_count};
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
    const auto coefficients = triple_polynomial_coefficients(geometry, source);
    constexpr int kTripleDegree = 10;

    // Polish each root on the true lens equation, then deduplicate.
    // Multiple spurious starting points can converge to the same physical image;
    // deduplication ensures each image is counted once.
    constexpr double kDedupTol2 = 1.0e-16;  // squared distance threshold
    std::vector<TripleImageCandidate> images;
    const auto collect_candidates = [&](const Complex* roots, std::size_t count) {
        images.clear();
        images.reserve(count);
        for (std::size_t k = 0; k < count; ++k) {
            const Complex root = roots[k];
            if (!std::isfinite(root.real()) || !std::isfinite(root.imag())) {
                continue;
            }
            const SourcePosition z_polished = polish_triple_image_root(
                geometry, source, SourcePosition{root.real(), root.imag()});
            bool is_dup = false;
            for (const auto& existing : images) {
                const double ddx = z_polished.x - existing.position.x;
                const double ddy = z_polished.y - existing.position.y;
                if (ddx * ddx + ddy * ddy < kDedupTol2) {
                    is_dup = true;
                    break;
                }
            }
            if (is_dup) { continue; }
            const Complex z_cmplx {z_polished.x, z_polished.y};
            const double residual = triple_residual(geometry, source, z_cmplx);
            images.push_back({z_polished,
                triple_jacobian_determinant(geometry, z_cmplx),
                residual,
                false});
        }
    };

    if (static_cast<int>(coefficients.size()) == kTripleDegree + 1 &&
        std::abs(coefficients[kTripleDegree]) > 0.0) {
        // Full-degree fast path: solve directly with fixed-size buffers and
        // warm-start from the previous solve of the same geometry (probe and
        // light-curve sweeps move the source continuously, so the previous
        // roots are excellent starting points).  If polished roots collapse
        // onto each other under a warm start, one image may have been lost to
        // duplicate convergence: redo the solve cold.
        thread_local std::array<::complex, kTripleDegree> warm_roots;
        thread_local model::TripleLensGeometry warm_geometry;
        thread_local bool warm_valid = false;

        std::array<::complex, kTripleDegree + 1> sg_coefficients;
        for (int k = 0; k <= kTripleDegree; ++k) {
            sg_coefficients[static_cast<std::size_t>(k)] = {
                coefficients[static_cast<std::size_t>(k)].real(),
                coefficients[static_cast<std::size_t>(k)].imag()};
        }
        std::array<Complex, kTripleDegree> roots;
        const auto run_solve = [&](bool use_starting_points) {
            cmplx_roots_gen(
                warm_roots.data(), sg_coefficients.data(), kTripleDegree, true,
                use_starting_points);
            for (int k = 0; k < kTripleDegree; ++k) {
                roots[static_cast<std::size_t>(k)] = {
                    warm_roots[static_cast<std::size_t>(k)].re,
                    warm_roots[static_cast<std::size_t>(k)].im};
            }
        };
        const bool warm =
            warm_valid && same_triple_basis_geometry(warm_geometry, geometry);
        run_solve(warm);
        collect_candidates(roots.data(), roots.size());
        if (warm && images.size() < static_cast<std::size_t>(kTripleDegree)) {
            run_solve(false);
            collect_candidates(roots.data(), roots.size());
        }
        warm_geometry = geometry;
        warm_valid = true;
    } else {
        const auto roots = math::PolynomialRootSolver().solve(coefficients);
        if (roots.status != math::RootSolverStatus::ok) {
            return {};
        }
        collect_candidates(roots.roots.data(), roots.roots.size());
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
    for (std::size_t i = 0; i < images.size(); ++i) {
        images[i].physical = static_cast<int>(i) < physical_count;
    }
    return images;
}

SourcePosition PointSourceMagnifier::triple_lens_equation(
    const model::TripleLensGeometry& geometry,
    SourcePosition image) const
{
    return model::triple_lens_equation(geometry, image);
}

} // namespace lcbinint::magnification
