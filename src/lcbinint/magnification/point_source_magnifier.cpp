#include "lcbinint/magnification/point_source_magnifier.hpp"

#include "lcbinint/math/polynomial_roots.hpp"

#include <algorithm>
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

struct CandidateImage {
    Complex z;
    double residual = 0.0;
};

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

std::vector<Complex> binary_polynomial_coefficients(const BinaryGeometry& geometry)
{
    const Complex a = geometry.separation;
    const Complex m1 = geometry.m1;
    const Complex m2 = geometry.m2;
    const Complex y = geometry.source;
    const Complex yc = std::conj(y);
    const Complex a2 = a * a;
    const Complex a3 = a2 * a;
    const Complex m2_2 = m2 * m2;

    std::vector<Complex> coefficients(6);
    coefficients[0] = a2 * m2_2 * y;
    coefficients[1] = a * m2 * (a * (m1 + y * (2.0 * yc - a)) - 2.0 * y);
    coefficients[2] =
        y * (1.0 - a3 * yc) - a * (m1 + 2.0 * y * yc * (1.0 + m2)) +
        a2 * (yc * (m1 - m2) + y * (1.0 + m2 + yc * yc));
    coefficients[3] =
        2.0 * y * yc + a3 * yc + a2 * (yc * (2.0 * y - yc) - m1) -
        a * (y + 2.0 * yc * (yc * y - m2));
    coefficients[4] = yc * (yc * (2.0 * a + y) - 1.0) - a * (yc * (2.0 * a + y) - m1);
    coefficients[5] = yc * (a - yc);
    return coefficients;
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

std::vector<CandidateImage> sorted_candidates(
    const BinaryGeometry& geometry,
    const std::vector<Complex>& roots)
{
    std::vector<CandidateImage> candidates;
    candidates.reserve(roots.size());
    for (const auto& root : roots) {
        candidates.push_back({root, std::abs(lens_equation_residual(geometry, root))});
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.residual < rhs.residual;
    });
    return candidates;
}

int physical_image_count(const std::vector<CandidateImage>& candidates)
{
    if (candidates.size() < 5) {
        return static_cast<int>(candidates.size());
    }

    constexpr double min_ratio = 1.0e-4;
    constexpr double absolute_gap = 1.0e-12;
    if (candidates[3].residual * min_ratio > candidates[2].residual + absolute_gap) {
        return 3;
    }
    return 5;
}

} // namespace

PointSourceResult PointSourceMagnifier::binary_mag0(
    double separation,
    double mass_ratio,
    SourcePosition source) const
{
    const auto images = binary_images(separation, mass_ratio, source);
    double magnification = 0.0;
    for (const auto& image : images) {
        magnification += 1.0 / std::abs(image.jacobian_determinant);
    }
    return {magnification, static_cast<int>(images.size())};
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

    const BinaryGeometry geometry = make_vbm_geometry(separation, mass_ratio, source);
    math::PolynomialRootSolver solver;
    const auto root_result = solver.solve(binary_polynomial_coefficients(geometry));
    if (root_result.status != math::RootSolverStatus::ok) {
        return {};
    }

    const auto candidates = sorted_candidates(geometry, root_result.roots);
    const int physical_count = physical_image_count(candidates);
    std::vector<BinaryImageCandidate> images;
    images.reserve(candidates.size());
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        const Complex z = candidates[i].z;
        images.push_back({{z.real(), z.imag()},
            jacobian_determinant(geometry, z),
            candidates[i].residual,
            static_cast<int>(i) < physical_count});
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
