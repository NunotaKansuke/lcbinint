#include "lcbinint/magnification/point_source_magnifier.hpp"

#include "SkowronGould.h"

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

double distance2(SourcePosition lhs, SourcePosition rhs)
{
    const double dx = lhs.x - rhs.x;
    const double dy = lhs.y - rhs.y;
    return dx * dx + dy * dy;
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
        root_cache_mass_ratio_ == mass_ratio &&
        distance2(root_cache_source_, source) < 2.5e-3;
    if (can_polish_from_cache) {
        for (std::size_t i = 0; i < roots.size(); ++i) {
            roots[i] = {root_cache_roots_[i].x, root_cache_roots_[i].y};
        }
        cmplx_roots_gen(roots.data(), coefficients.data(), 5, true, true);
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
    double magnification = 0.0;
    int image_count = 0;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (three_images && (static_cast<int>(i) == worst1 || static_cast<int>(i) == worst2)) {
            continue;
        }
        const double jacobian = jacobian_determinant(geometry, roots[i]);
        magnification += 1.0 / std::abs(jacobian);
        ++image_count;
    }
    return {magnification, image_count};
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

} // namespace lcbinint::magnification
