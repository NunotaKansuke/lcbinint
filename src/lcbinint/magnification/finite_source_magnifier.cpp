#include "lcbinint/magnification/finite_source_magnifier.hpp"

#include "lcbinint/magnification/point_source_magnifier.hpp"
#include "lcbinint/math/polynomial_roots.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace lcbinint::magnification {
namespace {

constexpr double kHighMagnificationPolarThreshold = 10.0;
constexpr double kSqrtHalf = 0.70710678118654752440;
constexpr double kPi = 3.14159265358979323846;
constexpr int kMaxRefinementLevels = 3;
constexpr int kHexadecapoleEvaluations = 13;

struct QuadrupoleSafety {
    double error_estimate = 0.0;
    bool accepted = false;
};

struct BinaryQuadrupoleGeometry {
    Complex a;
    double m1 = 0.0;
    double m2 = 0.0;
    double q = 0.0;
    Complex source;
};

double source_distance(SourcePosition source)
{
    return std::hypot(source.x, source.y);
}

int estimate_cartesian_cost(const FiniteSourceSettings& settings)
{
    const int bins = settings.source_bins > 0 ? settings.source_bins : 1;
    return bins * bins * 16;
}

int estimate_polar_cost(const FiniteSourceSettings& settings)
{
    const int radial_bins = settings.source_bins > 0 ? settings.source_bins : 1;
    const int angular_bins = static_cast<int>(std::ceil(2.0 * M_PI * radial_bins / settings.grid_ratio));
    return radial_bins * angular_bins * 8;
}

double distance_squared(SourcePosition lhs, SourcePosition rhs)
{
    const double dx = lhs.x - rhs.x;
    const double dy = lhs.y - rhs.y;
    return dx * dx + dy * dy;
}

double point_segment_distance(SourcePosition point, SourcePosition start, SourcePosition end)
{
    const double dx = end.x - start.x;
    const double dy = end.y - start.y;
    const double length2 = dx * dx + dy * dy;
    if (length2 == 0.0) {
        return std::sqrt(distance_squared(point, start));
    }

    const double t = std::clamp(
        ((point.x - start.x) * dx + (point.y - start.y) * dy) / length2, 0.0, 1.0);
    return std::hypot(point.x - (start.x + t * dx), point.y - (start.y + t * dy));
}

bool tolerance_met(double error_estimate, double value, const FiniteSourceSettings& settings)
{
    const double absolute_tolerance = std::max(settings.tolerance, 0.0);
    if (error_estimate <= absolute_tolerance) {
        return true;
    }

    const double relative_tolerance = std::max(settings.relative_tolerance, 0.0);
    return relative_tolerance > 0.0 && error_estimate <= relative_tolerance * std::abs(value);
}

double limb_darkening_flux_factor(const FiniteSourceSettings& settings)
{
    return 1.0 - settings.limb_darkening_c / 3.0 - settings.limb_darkening_d / 5.0;
}

double limb_darkening_gamma(const FiniteSourceSettings& settings)
{
    const double denominator =
        15.0 - 5.0 * settings.limb_darkening_c - 3.0 * settings.limb_darkening_d;
    if (denominator == 0.0) {
        return 0.0;
    }
    return 10.0 * settings.limb_darkening_c / denominator;
}

double limb_darkening_lambda(const FiniteSourceSettings& settings)
{
    const double denominator =
        15.0 - 5.0 * settings.limb_darkening_c - 3.0 * settings.limb_darkening_d;
    if (denominator == 0.0) {
        return 0.0;
    }
    return 12.0 * settings.limb_darkening_d / denominator;
}

double source_surface_brightness(double normalized_radius2, const FiniteSourceSettings& settings)
{
    const double bounded_radius2 = std::clamp(normalized_radius2, 0.0, 1.0);
    const double mu = std::sqrt(std::max(0.0, 1.0 - bounded_radius2));
    return 1.0 - settings.limb_darkening_c * (1.0 - mu) -
           settings.limb_darkening_d * (1.0 - std::sqrt(mu));
}

double source_flux(double source_radius, const FiniteSourceSettings& settings)
{
    const double flux_factor = limb_darkening_flux_factor(settings);
    if (flux_factor <= 0.0 || !std::isfinite(flux_factor)) {
        return std::nan("");
    }
    return kPi * source_radius * source_radius * flux_factor;
}

std::vector<Complex> critical_curve_polynomial_coefficients(double separation, double mass_ratio, Complex phase)
{
    const double s = std::abs(separation);
    const double q_input = std::abs(mass_ratio);
    const double q = q_input < 1.0 ? q_input : 1.0 / q_input;
    const Complex a = q_input < 1.0 ? Complex(-s, 0.0) : Complex(s, 0.0);
    const Complex m1 = 1.0 / (1.0 + q);
    const Complex m2 = q * m1;
    const Complex a2 = a * a;

    return {
        m2 * a2,
        -2.0 * m2 * a,
        m1 + m2 - phase * a2,
        2.0 * phase * a,
        -phase,
    };
}

std::vector<SourcePosition> caustic_points_at_phase(
    const PointSourceMagnifier& point_magnifier,
    double separation,
    double mass_ratio,
    double phase_angle)
{
    const Complex phase = std::polar(1.0, phase_angle);
    math::PolynomialRootSolver solver;
    const auto roots = solver.solve(critical_curve_polynomial_coefficients(separation, mass_ratio, phase));
    if (roots.status != math::RootSolverStatus::ok) {
        return {};
    }

    std::vector<SourcePosition> points;
    points.reserve(roots.roots.size());
    for (const auto& root : roots.roots) {
        points.push_back(point_magnifier.binary_lens_equation(
            separation, mass_ratio, {root.real(), root.imag()}));
    }
    return points;
}

void append_tracked_caustic_points(
    std::vector<std::vector<SourcePosition>>& branches,
    std::vector<SourcePosition> points)
{
    if (points.size() != branches.size()) {
        return;
    }

    if (branches[0].empty()) {
        std::sort(points.begin(), points.end(), [](const auto& lhs, const auto& rhs) {
            return std::atan2(lhs.y, lhs.x) < std::atan2(rhs.y, rhs.x);
        });
        for (std::size_t i = 0; i < points.size(); ++i) {
            branches[i].push_back(points[i]);
        }
        return;
    }

    std::vector<bool> used(points.size(), false);
    for (auto& branch : branches) {
        const SourcePosition previous = branch.back();
        std::size_t best_index = 0;
        double best_distance2 = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < points.size(); ++i) {
            if (used[i]) {
                continue;
            }
            const double candidate_distance2 = distance_squared(previous, points[i]);
            if (candidate_distance2 < best_distance2) {
                best_distance2 = candidate_distance2;
                best_index = i;
            }
        }
        used[best_index] = true;
        branch.push_back(points[best_index]);
    }
}

double binary_caustic_distance(
    const PointSourceMagnifier& point_magnifier,
    double separation,
    double mass_ratio,
    SourcePosition source,
    int caustic_bins)
{
    const int bins = std::max(caustic_bins, 32);
    std::vector<std::vector<SourcePosition>> branches(4);
    for (int i = 0; i < bins; ++i) {
        const double phase_angle = 2.0 * kPi * static_cast<double>(i) / static_cast<double>(bins);
        append_tracked_caustic_points(
            branches, caustic_points_at_phase(point_magnifier, separation, mass_ratio, phase_angle));
    }

    double distance = std::numeric_limits<double>::infinity();
    for (const auto& branch : branches) {
        if (branch.size() < 2) {
            continue;
        }
        for (std::size_t i = 1; i < branch.size(); ++i) {
            distance = std::min(distance, point_segment_distance(source, branch[i - 1], branch[i]));
        }
        distance = std::min(distance, point_segment_distance(source, branch.back(), branch.front()));
    }
    return distance;
}

double image_radius(double source_radius, double determinant)
{
    const double abs_det = std::max(std::abs(determinant), 1.0e-8);
    return std::max(2.5 * source_radius / std::sqrt(abs_det), 2.0 * source_radius);
}

double hexadecapole_binary(
    const PointSourceMagnifier& point_magnifier,
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings)
{
    const double sineta[8] = {0.0, 1.0, 0.0, -1.0, kSqrtHalf, kSqrtHalf, -kSqrtHalf, -kSqrtHalf};
    const double coseta[8] = {1.0, 0.0, -1.0, 0.0, kSqrtHalf, -kSqrtHalf, -kSqrtHalf, kSqrtHalf};

    const double a0 = point_magnifier.binary_mag0(separation, mass_ratio, source).magnification;
    double a1_plus = 0.0;
    double a2_plus = 0.0;
    double a1_cross = 0.0;
    for (int i = 0; i < 4; ++i) {
        a1_plus += point_magnifier
                       .binary_mag0(separation, mass_ratio,
                           {source.x + source_radius * coseta[i],
                               source.y + source_radius * sineta[i]})
                       .magnification;
        a2_plus += point_magnifier
                       .binary_mag0(separation, mass_ratio,
                           {source.x + 0.5 * source_radius * coseta[i],
                               source.y + 0.5 * source_radius * sineta[i]})
                       .magnification;
        a1_cross += point_magnifier
                        .binary_mag0(separation, mass_ratio,
                            {source.x + source_radius * coseta[i + 4],
                                source.y + source_radius * sineta[i + 4]})
                        .magnification;
    }
    a1_plus = a1_plus / 4.0 - a0;
    a2_plus = a2_plus / 4.0 - a0;
    a1_cross = a1_cross / 4.0 - a0;

    const double a2rho2 = (16.0 * a2_plus - a1_plus) / 3.0;
    const double a4rho4 = (a1_plus + a1_cross) / 2.0 - a2rho2;
    const double gamma = limb_darkening_gamma(settings);
    const double lambda = limb_darkening_lambda(settings);
    return a0 + 0.5 * a2rho2 * (1.0 - 0.2 * gamma - lambda / 9.0) +
           a4rho4 / 3.0 * (1.0 - 11.0 * gamma / 35.0 - 7.0 * lambda / 39.0);
}

BinaryQuadrupoleGeometry make_quadrupole_geometry(
    double separation,
    double mass_ratio,
    SourcePosition source)
{
    const double s = std::abs(separation);
    const double q_input = std::abs(mass_ratio);
    const double q = q_input < 1.0 ? q_input : 1.0 / q_input;
    const Complex a = q_input < 1.0 ? Complex(-s, 0.0) : Complex(s, 0.0);
    const double m1 = 1.0 / (1.0 + q);
    const double m2 = q * m1;
    return {a, m1, m2, q, Complex(source.x, source.y) + a * m1};
}

Complex f0(const BinaryQuadrupoleGeometry& geometry, Complex z)
{
    return -geometry.m1 / (z - geometry.a) - geometry.m2 / z;
}

Complex f1(const BinaryQuadrupoleGeometry& geometry, Complex z)
{
    const Complex za = z - geometry.a;
    return geometry.m1 / (za * za) + geometry.m2 / (z * z);
}

Complex f2(const BinaryQuadrupoleGeometry& geometry, Complex z)
{
    const Complex za = z - geometry.a;
    return -2.0 * geometry.m1 / (za * za * za) - 2.0 * geometry.m2 / (z * z * z);
}

Complex f3(const BinaryQuadrupoleGeometry& geometry, Complex z)
{
    const Complex za = z - geometry.a;
    return 6.0 * geometry.m1 / (za * za * za * za) + 6.0 * geometry.m2 / (z * z * z * z);
}

double jacobian_from_f1(Complex derivative)
{
    return 1.0 - std::norm(derivative);
}

QuadrupoleSafety quadrupole_safety_test(
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    double tolerance,
    const std::vector<BinaryImageCandidate>& candidates)
{
    if (candidates.empty()) {
        return {};
    }

    const auto geometry = make_quadrupole_geometry(separation, mass_ratio, source);
    double correction_sum = 0.0;
    double ghost_max = 0.0;
    for (const auto& candidate : candidates) {
        const Complex z(candidate.position.x, candidate.position.y);
        const Complex dz1 = f1(geometry, z);
        const Complex dz2 = f2(geometry, z);
        const Complex dz3 = f3(geometry, z);
        const double j = jacobian_from_f1(dz1);
        if (!std::isfinite(j) || std::abs(j) < 1.0e-14) {
            return {std::numeric_limits<double>::infinity(), false};
        }

        if (candidate.physical) {
            const double j2 = j * j;
            const double j5 = j2 * j2 * j;
            const Complex term = 3.0 * std::pow(std::conj(dz1), 3) * dz2 * dz2 -
                                 (3.0 - 3.0 * j + 0.5 * j * j) * std::norm(dz2) +
                                 j * std::pow(std::conj(dz1), 2) * dz3;
            const double mu_q = std::abs(-2.0 * term.real() / j5);
            const double mu_c =
                std::abs((3.0 * std::pow(std::conj(dz1), 3) * dz2 * dz2).imag() / j5);
            correction_sum += mu_q + mu_c;
        } else {
            const Complex zwave = std::conj(geometry.source) - f0(geometry, z);
            const Complex j_wave = 1.0 - f1(geometry, z) * f1(geometry, zwave);
            if (std::abs(j_wave) < 1.0e-14) {
                ghost_max = std::numeric_limits<double>::infinity();
                continue;
            }
            const Complex j3 = j_wave * f2(geometry, std::conj(z)) * f1(geometry, z);
            const Complex mu_g =
                (j3 - std::conj(j3) * f1(geometry, zwave)) / (j * j_wave * j_wave);
            ghost_max = std::max(ghost_max, std::abs(mu_g));
        }
    }

    const double c_q = 2.0;
    const double c_g = 3.0;
    const double c_p = 4.0;
    const double correction_error = correction_sum * c_q *
                                    (source_radius * source_radius + 1.0e-4 * tolerance);
    const bool quadrupole_ok = correction_error < tolerance;
    const bool ghost_ok = (source_radius + 1.0e-3) * ghost_max * c_g < 1.0;
    bool planet_ok = true;
    if (geometry.q <= 1.0e-2 && std::abs(geometry.a) > 0.0) {
        const Complex planetary_caustic = 1.0 / geometry.a;
        const double safe_distance2 = std::norm(geometry.source - planetary_caustic);
        const double separation2 = std::norm(geometry.a);
        planet_ok = safe_distance2 > c_p *
                                     (source_radius * source_radius +
                                         9.0 * geometry.q / separation2);
    }

    return {correction_error, quadrupole_ok && ghost_ok && planet_ok};
}

double inverse_ray_cartesian_binary(
    const PointSourceMagnifier& point_magnifier,
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings,
    int bins)
{
    const auto images = point_magnifier.binary_images(separation, mass_ratio, source);
    if (images.empty()) {
        return std::nan("");
    }

    const int grid_bins = std::max(bins, 1);
    const double source_radius2 = source_radius * source_radius;
    const double total_source_flux = source_flux(source_radius, settings);
    if (!std::isfinite(total_source_flux)) {
        return std::nan("");
    }
    const bool uniform_source = settings.limb_darkening_c == 0.0 && settings.limb_darkening_d == 0.0;

    double image_flux = 0.0;
    for (const auto& image : images) {
        const double half_width = image_radius(source_radius, image.jacobian_determinant);
        const double step = 2.0 * half_width / static_cast<double>(grid_bins);
        const double cell_area = step * step;
        for (int ix = 0; ix < grid_bins; ++ix) {
            const double x = image.position.x - half_width + (ix + 0.5) * step;
            for (int iy = 0; iy < grid_bins; ++iy) {
                const double y = image.position.y - half_width + (iy + 0.5) * step;
                const auto mapped = point_magnifier.binary_lens_equation(separation, mass_ratio, {x, y});
                const double mapped_distance2 = distance_squared(mapped, source);
                if (mapped_distance2 <= source_radius2) {
                    image_flux += uniform_source ?
                                      cell_area :
                                      cell_area * source_surface_brightness(
                                                      mapped_distance2 / source_radius2, settings);
                }
            }
        }
    }
    return image_flux / total_source_flux;
}

double inverse_ray_polar_binary(
    const PointSourceMagnifier& point_magnifier,
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings,
    int bins)
{
    const auto images = point_magnifier.binary_images(separation, mass_ratio, source);
    if (images.empty()) {
        return std::nan("");
    }

    const int radial_bins = std::max(bins, 1);
    const int angular_bins = std::max(
        16, static_cast<int>(std::ceil(2.0 * kPi * radial_bins / settings.grid_ratio)));
    const double source_radius2 = source_radius * source_radius;
    const double total_source_flux = source_flux(source_radius, settings);
    if (!std::isfinite(total_source_flux)) {
        return std::nan("");
    }
    const bool uniform_source = settings.limb_darkening_c == 0.0 && settings.limb_darkening_d == 0.0;

    std::vector<double> cos_phi(static_cast<std::size_t>(angular_bins));
    std::vector<double> sin_phi(static_cast<std::size_t>(angular_bins));
    const double dphi = 2.0 * kPi / static_cast<double>(angular_bins);
    for (int iphi = 0; iphi < angular_bins; ++iphi) {
        const double phi = (iphi + 0.5) * dphi;
        cos_phi[static_cast<std::size_t>(iphi)] = std::cos(phi);
        sin_phi[static_cast<std::size_t>(iphi)] = std::sin(phi);
    }

    double image_flux = 0.0;
    for (const auto& image : images) {
        const double rmax = image_radius(source_radius, image.jacobian_determinant);
        const double dr = rmax / static_cast<double>(radial_bins);
        for (int ir = 0; ir < radial_bins; ++ir) {
            const double r_inner = ir * dr;
            const double r_outer = (ir + 1) * dr;
            const double r = 0.5 * (r_inner + r_outer);
            const double cell_area = 0.5 * (r_outer * r_outer - r_inner * r_inner) * dphi;
            for (int iphi = 0; iphi < angular_bins; ++iphi) {
                const double x = image.position.x + r * cos_phi[static_cast<std::size_t>(iphi)];
                const double y = image.position.y + r * sin_phi[static_cast<std::size_t>(iphi)];
                const auto mapped = point_magnifier.binary_lens_equation(separation, mass_ratio, {x, y});
                const double mapped_distance2 = distance_squared(mapped, source);
                if (mapped_distance2 <= source_radius2) {
                    image_flux += uniform_source ?
                                      cell_area :
                                      cell_area * source_surface_brightness(
                                                      mapped_distance2 / source_radius2, settings);
                }
            }
        }
    }
    return image_flux / total_source_flux;
}

double inverse_ray_binary(
    const PointSourceMagnifier& point_magnifier,
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings,
    FiniteSourceMethod method,
    int bins)
{
    if (method == FiniteSourceMethod::inverse_ray_polar) {
        return inverse_ray_polar_binary(
            point_magnifier, separation, mass_ratio, source, source_radius, settings, bins);
    }

    return inverse_ray_cartesian_binary(
        point_magnifier, separation, mass_ratio, source, source_radius, settings, bins);
}

FiniteSourceResult refined_inverse_ray_binary(
    const PointSourceMagnifier& point_magnifier,
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings,
    FiniteSourceDecision decision)
{
    int bins = std::max(settings.source_bins, 1);
    double coarse = inverse_ray_binary(
        point_magnifier, separation, mass_ratio, source, source_radius, settings, decision.method, bins);
    if (!std::isfinite(coarse)) {
        return {coarse, 0, decision, std::nan(""), 0, false};
    }

    double error_estimate = std::numeric_limits<double>::infinity();
    double previous_delta = std::numeric_limits<double>::quiet_NaN();
    for (int level = 1; level <= kMaxRefinementLevels; ++level) {
        bins *= 2;
        const double fine = inverse_ray_binary(
            point_magnifier, separation, mass_ratio, source, source_radius, settings, decision.method, bins);
        if (!std::isfinite(fine)) {
            return {fine, 0, decision, std::nan(""), level, false};
        }

        const double delta = std::abs(fine - coarse);
        error_estimate = delta;
        if (std::isfinite(previous_delta) && previous_delta > 0.0) {
            const double ratio = delta / previous_delta;
            if (std::isfinite(ratio) && ratio > 0.0 && ratio < 0.95) {
                error_estimate = std::max(delta, delta * ratio / (1.0 - ratio));
            } else {
                error_estimate = delta + previous_delta;
            }
        }
        if (tolerance_met(error_estimate, fine, settings)) {
            decision.reason += "; refined to requested tolerance";
            return {fine, 0, decision, error_estimate, level, true};
        }
        previous_delta = delta;
        coarse = fine;
    }

    decision.reason += "; refinement limit reached";
    return {coarse, 0, decision, error_estimate, kMaxRefinementLevels, false};
}

FiniteSourceResult fixed_inverse_ray_binary(
    const PointSourceMagnifier& point_magnifier,
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings,
    FiniteSourceDecision decision)
{
    const int bins = std::max(settings.source_bins, 1);
    const double magnification = inverse_ray_binary(
        point_magnifier, separation, mass_ratio, source, source_radius, settings, decision.method, bins);
    if (!std::isfinite(magnification)) {
        return {magnification, 0, decision, std::nan(""), 0, false};
    }
    return {magnification, 0, decision, 0.0, 0, true};
}

} // namespace

FiniteSourceMagnifier::FiniteSourceMagnifier(FiniteSourceSettings settings)
    : settings_(settings)
{
}

void FiniteSourceMagnifier::ensure_legacy_caustic_cache(double separation, double mass_ratio) const
{
    const int bins = std::max(settings_.caustic_bins, 32);
    const bool cache_matches = caustic_cache_valid_ &&
                               caustic_cache_bins_ == bins &&
                               caustic_cache_separation_ == separation &&
                               caustic_cache_mass_ratio_ == mass_ratio;
    if (!cache_matches) {
        const PointSourceMagnifier point_magnifier;
        caustic_cache_branches_.assign(4, {});
        caustic_cache_points_.clear();
        caustic_cache_points_.reserve(static_cast<std::size_t>(bins) * 4);
        caustic_cache_min_x_ = std::numeric_limits<double>::infinity();
        caustic_cache_max_x_ = -std::numeric_limits<double>::infinity();
        caustic_cache_min_y_ = std::numeric_limits<double>::infinity();
        caustic_cache_max_y_ = -std::numeric_limits<double>::infinity();
        for (int i = 0; i < bins; ++i) {
            const double phase_angle = 2.0 * kPi * static_cast<double>(i) /
                                       static_cast<double>(bins);
            auto points = caustic_points_at_phase(point_magnifier, separation, mass_ratio, phase_angle);
            for (const auto& point : points) {
                caustic_cache_points_.push_back(point);
                caustic_cache_min_x_ = std::min(caustic_cache_min_x_, point.x);
                caustic_cache_max_x_ = std::max(caustic_cache_max_x_, point.x);
                caustic_cache_min_y_ = std::min(caustic_cache_min_y_, point.y);
                caustic_cache_max_y_ = std::max(caustic_cache_max_y_, point.y);
            }
            append_tracked_caustic_points(caustic_cache_branches_, std::move(points));
        }
        const double width = std::max(caustic_cache_max_x_ - caustic_cache_min_x_, 1.0e-12);
        const double height = std::max(caustic_cache_max_y_ - caustic_cache_min_y_, 1.0e-12);
        caustic_cache_grid_step_x_ = width / static_cast<double>(caustic_cache_grid_size_);
        caustic_cache_grid_step_y_ = height / static_cast<double>(caustic_cache_grid_size_);
        caustic_cache_grid_.assign(
            static_cast<std::size_t>(caustic_cache_grid_size_ * caustic_cache_grid_size_), {});
        for (std::size_t i = 0; i < caustic_cache_points_.size(); ++i) {
            const auto& point = caustic_cache_points_[i];
            const int ix = std::clamp(static_cast<int>((point.x - caustic_cache_min_x_) /
                                           caustic_cache_grid_step_x_),
                0,
                caustic_cache_grid_size_ - 1);
            const int iy = std::clamp(static_cast<int>((point.y - caustic_cache_min_y_) /
                                           caustic_cache_grid_step_y_),
                0,
                caustic_cache_grid_size_ - 1);
            caustic_cache_grid_[static_cast<std::size_t>(iy * caustic_cache_grid_size_ + ix)]
                .push_back(static_cast<int>(i));
        }
        caustic_cache_valid_ = true;
        caustic_cache_separation_ = separation;
        caustic_cache_mass_ratio_ = mass_ratio;
        caustic_cache_bins_ = bins;
    }
}

double FiniteSourceMagnifier::legacy_binary_caustic_distance(
    double separation,
    double mass_ratio,
    SourcePosition source) const
{
    ensure_legacy_caustic_cache(separation, mass_ratio);

    double distance = std::numeric_limits<double>::infinity();
    for (const auto& branch : caustic_cache_branches_) {
        if (branch.size() < 2) {
            continue;
        }
        for (std::size_t i = 1; i < branch.size(); ++i) {
            distance = std::min(distance, point_segment_distance(source, branch[i - 1], branch[i]));
        }
        distance = std::min(distance, point_segment_distance(source, branch.back(), branch.front()));
    }
    return distance;
}

double FiniteSourceMagnifier::legacy_binary_sampled_caustic_distance(
    double separation,
    double mass_ratio,
    SourcePosition source,
    double search_radius) const
{
    ensure_legacy_caustic_cache(separation, mass_ratio);
    double distance2 = std::numeric_limits<double>::infinity();

    if (search_radius > 0.0 &&
        (source.x < caustic_cache_min_x_ - search_radius ||
            source.x > caustic_cache_max_x_ + search_radius ||
            source.y < caustic_cache_min_y_ - search_radius ||
            source.y > caustic_cache_max_y_ + search_radius)) {
        return std::numeric_limits<double>::infinity();
    }

    if (search_radius > 0.0 && !caustic_cache_grid_.empty()) {
        const int ix0 = std::clamp(static_cast<int>((source.x - search_radius - caustic_cache_min_x_) /
                                       caustic_cache_grid_step_x_),
            0,
            caustic_cache_grid_size_ - 1);
        const int ix1 = std::clamp(static_cast<int>((source.x + search_radius - caustic_cache_min_x_) /
                                       caustic_cache_grid_step_x_),
            0,
            caustic_cache_grid_size_ - 1);
        const int iy0 = std::clamp(static_cast<int>((source.y - search_radius - caustic_cache_min_y_) /
                                       caustic_cache_grid_step_y_),
            0,
            caustic_cache_grid_size_ - 1);
        const int iy1 = std::clamp(static_cast<int>((source.y + search_radius - caustic_cache_min_y_) /
                                       caustic_cache_grid_step_y_),
            0,
            caustic_cache_grid_size_ - 1);
        for (int iy = iy0; iy <= iy1; ++iy) {
            for (int ix = ix0; ix <= ix1; ++ix) {
                const auto& cell =
                    caustic_cache_grid_[static_cast<std::size_t>(iy * caustic_cache_grid_size_ + ix)];
                for (const int index : cell) {
                    const auto& point = caustic_cache_points_[static_cast<std::size_t>(index)];
                    const double dx = source.x - point.x;
                    const double dy = source.y - point.y;
                    const double candidate = dx * dx + dy * dy;
                    if (candidate < distance2) {
                        distance2 = candidate;
                    }
                }
            }
        }
        if (distance2 == std::numeric_limits<double>::infinity()) {
            return distance2;
        }
    } else {
        for (const auto& point : caustic_cache_points_) {
            const double dx = source.x - point.x;
            const double dy = source.y - point.y;
            const double candidate = dx * dx + dy * dy;
            if (candidate < distance2) {
                distance2 = candidate;
            }
        }
    }
    return std::sqrt(distance2);
}

FiniteSourceDecision FiniteSourceMagnifier::choose_binary_method(
    SourcePosition source,
    double source_radius,
    double point_source_magnification) const
{
    if (source_radius <= 0.0) {
        return {FiniteSourceMethod::point_source, 0, "zero source radius"};
    }

    const int polar_cost = estimate_polar_cost(settings_);
    const int cartesian_cost = estimate_cartesian_cost(settings_);
    if (settings_.inverse_ray_method == InverseRayMethod::cartesian) {
        return {FiniteSourceMethod::inverse_ray_cartesian, cartesian_cost, "user-selected cartesian inverse-ray"};
    }
    if (settings_.inverse_ray_method == InverseRayMethod::polar) {
        return {FiniteSourceMethod::inverse_ray_polar, polar_cost, "user-selected polar inverse-ray"};
    }

    if (point_source_magnification >= kHighMagnificationPolarThreshold ||
        source_distance(source) < 3.0 * source_radius) {
        return {FiniteSourceMethod::inverse_ray_polar, polar_cost, "high magnification or near source center"};
    }

    return {FiniteSourceMethod::inverse_ray_cartesian, cartesian_cost, "default inverse-ray fallback"};
}

const char* finite_source_method_name(FiniteSourceMethod method)
{
    switch (method) {
    case FiniteSourceMethod::point_source:
        return "point_source";
    case FiniteSourceMethod::hexadecapole:
        return "hexadecapole";
    case FiniteSourceMethod::inverse_ray_cartesian:
        return "inverse_ray_cartesian";
    case FiniteSourceMethod::inverse_ray_polar:
        return "inverse_ray_polar";
    default:
        return "unknown";
    }
}

FiniteSourceResult FiniteSourceMagnifier::binary_mag(
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    double point_source_magnification) const
{
    if (result_cache_valid_ && result_cache_separation_ == separation &&
        result_cache_mass_ratio_ == mass_ratio && result_cache_source_x_ == source.x &&
        result_cache_source_y_ == source.y && result_cache_source_radius_ == source_radius &&
        result_cache_point_magnification_ == point_source_magnification) {
        return result_cache_;
    }

    const auto cache_and_return = [&](FiniteSourceResult result) {
        result_cache_valid_ = true;
        result_cache_separation_ = separation;
        result_cache_mass_ratio_ = mass_ratio;
        result_cache_source_x_ = source.x;
        result_cache_source_y_ = source.y;
        result_cache_source_radius_ = source_radius;
        result_cache_point_magnification_ = point_source_magnification;
        result_cache_ = result;
        return result;
    };

    PointSourceMagnifier point_magnifier;
    if (source_radius <= 0.0) {
        auto decision = choose_binary_method(source, source_radius, point_source_magnification);
        const auto point = point_magnifier.binary_mag0(separation, mass_ratio, source);
        return cache_and_return({point.magnification, point.image_count, decision, 0.0, 0, true});
    }

    if (settings_.legacy_mode) {
        if (settings_.legacy_finite_mode <= 0) {
            FiniteSourceDecision decision {
                FiniteSourceMethod::point_source,
                0,
                "legacy smode=0 point-source",
            };
            return cache_and_return({point_source_magnification, 0, decision, 0.0, 0, true});
        }

        const double cached_point_threshold = 2.0 * settings_.legacy_kinji * source_radius;
        double caustic_distance = legacy_binary_sampled_caustic_distance(
            separation, mass_ratio, source, cached_point_threshold);
        if (!std::isfinite(caustic_distance) || caustic_distance >= cached_point_threshold) {
            FiniteSourceDecision decision {
                FiniteSourceMethod::point_source,
                settings_.caustic_bins * 4,
                "legacy cached caustic distance accepted point-source approximation",
            };
            return cache_and_return({point_source_magnification, 0, decision, 0.0, 0, true});
        }

        caustic_distance = legacy_binary_caustic_distance(separation, mass_ratio, source);
        if (!std::isfinite(caustic_distance)) {
            caustic_distance = source_distance(source);
        }

        if (caustic_distance > settings_.legacy_kinji * source_radius) {
            FiniteSourceDecision decision {
                FiniteSourceMethod::point_source,
                settings_.caustic_bins * 4,
                "legacy KINJI accepted point-source approximation",
            };
            return cache_and_return({point_source_magnification, 0, decision, 0.0, 0, true});
        }
        if (caustic_distance > settings_.legacy_hex * source_radius) {
            FiniteSourceDecision decision {
                FiniteSourceMethod::hexadecapole,
                settings_.caustic_bins * 4 + kHexadecapoleEvaluations,
                "legacy HEX accepted hexadecapole approximation",
            };
            return cache_and_return({hexadecapole_binary(point_magnifier, separation, mass_ratio, source, source_radius, settings_),
                0,
                decision,
                0.0,
                0,
                true});
        }

        auto decision = choose_binary_method(source, source_radius, point_source_magnification);
        if (settings_.legacy_finite_mode == 5 || settings_.legacy_finite_mode == 6) {
            decision.method = FiniteSourceMethod::inverse_ray_polar;
            decision.estimated_evaluations = estimate_polar_cost(settings_);
            decision.reason = "legacy smode selected polar inverse-ray";
        } else if (settings_.legacy_finite_mode == 3 || settings_.legacy_finite_mode == 4) {
            decision.method = FiniteSourceMethod::inverse_ray_cartesian;
            decision.estimated_evaluations = estimate_cartesian_cost(settings_);
            decision.reason = "legacy smode selected cartesian inverse-ray";
        } else {
            decision.reason = "legacy smode fell back to automatic finite-source strategy";
        }
        return cache_and_return(fixed_inverse_ray_binary(
            point_magnifier, separation, mass_ratio, source, source_radius, settings_, decision));
    }

    const auto candidates = point_magnifier.binary_image_candidates(separation, mass_ratio, source);
    const auto quadrupole_safety = quadrupole_safety_test(
        separation, mass_ratio, source, source_radius, settings_.tolerance, candidates);
    if (quadrupole_safety.accepted) {
        FiniteSourceDecision decision {
            FiniteSourceMethod::point_source,
            static_cast<int>(candidates.size()),
            "quadrupole safety test accepted point-source approximation",
        };
        return cache_and_return({point_source_magnification,
            static_cast<int>(std::count_if(candidates.begin(), candidates.end(), [](const auto& image) {
                return image.physical;
            })),
            decision,
            quadrupole_safety.error_estimate,
            0,
            true});
    }

    auto decision = choose_binary_method(source, source_radius, point_source_magnification);
    decision.reason += "; quadrupole safety test rejected point-source approximation";
    return cache_and_return(refined_inverse_ray_binary(
        point_magnifier, separation, mass_ratio, source, source_radius, settings_, decision));
}

} // namespace lcbinint::magnification
