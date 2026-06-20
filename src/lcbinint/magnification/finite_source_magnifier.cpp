#include "lcbinint/magnification/finite_source_magnifier.hpp"

#include "lcbinint/magnification/point_source_magnifier.hpp"
#include "lcbinint/math/polynomial_roots.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

namespace lcbinint::magnification {
namespace {

constexpr double kSqrtHalf = 0.70710678118654752440;
constexpr double kPi = 3.14159265358979323846;
constexpr int kHexadecapoleEvaluations = 13;
constexpr int kLimbDarkeningTableSize = 5000;
constexpr int kLegacyIndexOffset = 2000000;

struct BinaryLensMapper {
    Complex separation;
    double m1 = 0.0;
    double m2 = 0.0;
};

BinaryLensMapper make_binary_lens_mapper(double separation, double mass_ratio)
{
    const double s = std::abs(separation);
    const double q_input = std::abs(mass_ratio);
    const double q = q_input < 1.0 ? q_input : 1.0 / q_input;
    const Complex lens_separation = q_input < 1.0 ? Complex(-s, 0.0) : Complex(s, 0.0);
    const double m1 = 1.0 / (1.0 + q);
    const double m2 = q * m1;
    return {lens_separation, m1, m2};
}

double mapped_binary_lens_distance2(
    const BinaryLensMapper& mapper,
    double x,
    double y,
    SourcePosition source)
{
    const double a = mapper.separation.real();
    const double xa = x - a;
    const double den1 = xa * xa + y * y;
    const double den2 = x * x + y * y;
    const double mapped_x = x - mapper.m1 * xa / den1 - mapper.m2 * x / den2 - a * mapper.m1;
    const double mapped_y = y - mapper.m1 * y / den1 - mapper.m2 * y / den2;
    const double dx = mapped_x - source.x;
    const double dy = mapped_y - source.y;
    return dx * dx + dy * dy;
}

SourcePosition map_binary_lens_real(
    const BinaryLensMapper& mapper,
    double x,
    double y)
{
    const double a = mapper.separation.real();
    const double xa = x - a;
    const double den1 = xa * xa + y * y;
    const double den2 = x * x + y * y;
    return {
        x - mapper.m1 * xa / den1 - mapper.m2 * x / den2 - a * mapper.m1,
        y - mapper.m1 * y / den1 - mapper.m2 * y / den2,
    };
}

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
    const std::size_t n = branches.size();
    if (points.size() != n) {
        return;
    }

    if (branches[0].empty()) {
        std::sort(points.begin(), points.end(), [](const auto& lhs, const auto& rhs) {
            return std::atan2(lhs.y, lhs.x) < std::atan2(rhs.y, rhs.x);
        });
        for (std::size_t i = 0; i < n; ++i) {
            branches[i].push_back(points[i]);
        }
        return;
    }

    // Find the permutation of `points` that minimises total squared step length.
    // Greedy nearest-neighbour can swap inner/outer caustic branches when they
    // come close; the global optimum never makes a swap unless the two assignments
    // have identical total cost (branches genuinely coincide), avoiding spurious
    // long segments in the branch grid.  For n=4 this is 4!=24 permutations.
    std::vector<std::size_t> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::vector<std::size_t> best_perm = perm;
    double best_cost = std::numeric_limits<double>::infinity();
    do {
        double cost = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            cost += distance_squared(branches[i].back(), points[perm[i]]);
        }
        if (cost < best_cost) {
            best_cost = cost;
            best_perm = perm;
        }
    } while (std::next_permutation(perm.begin(), perm.end()));

    for (std::size_t i = 0; i < n; ++i) {
        branches[i].push_back(points[best_perm[i]]);
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

struct PolarBoundaryRow {
    double min_r = 0.0;
    double max_r = 0.0;
    double phi = 0.0;
};

struct PolarBoundaryScratch {
    std::vector<PolarBoundaryRow> rows;
    std::vector<std::vector<int>> row_indices_by_phi;
};

struct PolarMapCacheView {
    const std::vector<SourcePosition>* mapped_sources = nullptr;
    const std::vector<int>* radial_offsets = nullptr;
    int radial_offset_min_index = 0;
    int phi_bins = 0;
    double dr = 1.0;
};

struct LegacyImageAreaScratch {
    std::vector<double> xmin;
    std::vector<double> xmax;
    std::vector<double> ax;
    std::vector<double> y;
    std::vector<double> dys;
    std::unordered_map<int, std::vector<int>> row_indices;

    void ensure(std::size_t index)
    {
        if (xmin.size() <= index) {
            const std::size_t size = index + 1;
            xmin.resize(size);
            xmax.resize(size);
            ax.resize(size);
            y.resize(size);
            dys.resize(size);
        }
    }
};

double wrap_angle(double angle)
{
    while (angle < 0.0) {
        angle += 2.0 * kPi;
    }
    while (angle >= 2.0 * kPi) {
        angle -= 2.0 * kPi;
    }
    return angle;
}

double trace_polar_boundary_rows(
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings,
    const FiniteSourceMagnifier* finite_magnifier,
    const PolarMapCacheView* map_cache,
    double start_radius,
    double start_phi,
    double dphi,
    PolarBoundaryScratch& scratch)
{
    const double dr = source_radius / static_cast<double>(std::max(settings.source_bins, 1));
    const double source_radius2 = source_radius * source_radius;
    const bool uniform_source = settings.limb_darkening_c == 0.0 && settings.limb_darkening_d == 0.0;
    if (!uniform_source && finite_magnifier != nullptr) {
        finite_magnifier->ensure_limb_darkening_table();
    }
    const auto mapper = make_binary_lens_mapper(separation, mass_ratio);
    const int phi_bins = static_cast<int>(scratch.row_indices_by_phi.size());
    const int max_rows = std::max(64, phi_bins * 2 + 16);
    const int max_radial_steps = std::max(64, settings.source_bins * 12);

    double total_count = 0.0;
    double radius_origin = start_radius;
    double phi = wrap_angle(start_phi);
    for (int row = 0; row < max_rows; ++row) {
        const int phi_index = std::clamp(static_cast<int>(phi / std::abs(dphi)), 0, phi_bins - 1);
        const double cos_phi = std::cos(phi);
        const double sin_phi = std::sin(phi);

        double count = 0.0;
        double dz2 = std::numeric_limits<double>::infinity();
        double dz2_last = dz2;
        double max_r = radius_origin;
        double min_r = radius_origin;

        double radius = radius_origin;
        double direction = dr;
        for (int step = 0; step < max_radial_steps; ++step) {
            SourcePosition mapped;
            const int radial_index = static_cast<int>(std::floor(radius / dr));
            const bool use_cached_map =
                map_cache != nullptr && map_cache->mapped_sources != nullptr &&
                map_cache->radial_offsets != nullptr &&
                map_cache->phi_bins == phi_bins && map_cache->dr == dr &&
                radial_index >= map_cache->radial_offset_min_index &&
                radial_index < map_cache->radial_offset_min_index +
                    static_cast<int>(map_cache->radial_offsets->size()) &&
                (*map_cache->radial_offsets)[static_cast<std::size_t>(
                    radial_index - map_cache->radial_offset_min_index)] >= 0;
            if (use_cached_map) {
                const int row_offset = (*map_cache->radial_offsets)[static_cast<std::size_t>(
                    radial_index - map_cache->radial_offset_min_index)];
                const auto index =
                    static_cast<std::size_t>(row_offset) *
                        static_cast<std::size_t>(map_cache->phi_bins) +
                    static_cast<std::size_t>(phi_index);
                mapped = (*map_cache->mapped_sources)[index];
            } else {
                const SourcePosition image {radius * cos_phi, radius * sin_phi};
                mapped = map_binary_lens_real(mapper, image.x, image.y);
            }
            dz2_last = dz2;
            dz2 = distance_squared(mapped, source);

            if (dz2 <= source_radius2) {
                if (direction < 0.0 && count == 0.0) {
                    max_r = radius - direction;
                }
                const double brightness =
                    uniform_source ? 1.0 : finite_magnifier->legacy_limb_darkening_table_brightness(
                                               dz2 / source_radius2);
                count += brightness * radius;
            } else if (direction > 0.0) {
                if (dz2_last <= source_radius2) {
                    max_r = radius;
                } else if (total_count == 0.0 && row <= 1 && radius == radius_origin) {
                    radius += direction;
                    continue;
                }
                direction = -dr;
                radius = radius_origin;
                min_r = radius + direction;
            } else {
                if (dz2_last <= source_radius2) {
                    min_r = radius;
                }
                if (!scratch.rows.empty() && count == 0.0 &&
                    radius >= scratch.rows.back().min_r - direction) {
                    radius += direction;
                    continue;
                }

                if (count == 0.0) {
                    if (!scratch.rows.empty()) {
                        return total_count;
                    }
                } else {
                    for (const int index : scratch.row_indices_by_phi[static_cast<std::size_t>(phi_index)]) {
                        const auto& existing = scratch.rows[static_cast<std::size_t>(index)];
                        if (min_r + dr < existing.max_r && max_r - dr > existing.min_r) {
                            return total_count;
                        }
                    }
                    total_count += count;
                    scratch.row_indices_by_phi[static_cast<std::size_t>(phi_index)].push_back(
                        static_cast<int>(scratch.rows.size()));
                    scratch.rows.push_back({min_r, max_r, phi});
                }

                radius_origin = max_r - dr;
                phi = wrap_angle(phi + dphi);
                break;
            }
            radius += direction;
        }

        if (row > 0 && count == 0.0) {
            break;
        }
    }
    return total_count;
}

double inverse_ray_polar_boundary_binary(
    const PointSourceMagnifier& point_magnifier,
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings,
    const FiniteSourceMagnifier* finite_magnifier,
    const PolarMapCacheView* map_cache = nullptr,
    const std::vector<SourcePosition>* seed_positions = nullptr)
{
    std::vector<SourcePosition> image_positions;
    if (seed_positions != nullptr) {
        image_positions = *seed_positions;
    } else {
        const auto images = point_magnifier.binary_images(separation, mass_ratio, source);
        image_positions.reserve(images.size());
        for (const auto& image : images) {
            image_positions.push_back(image.position);
        }
    }
    if (image_positions.empty()) {
        return std::nan("");
    }

    const int source_bins = std::max(settings.source_bins, 1);
    const double dr = source_radius / static_cast<double>(source_bins);
    const int phi_bins = std::max(16, static_cast<int>(2.0 * kPi / (dr * settings.grid_ratio)));
    const double dphi = 2.0 * kPi / static_cast<double>(phi_bins);
    const double total_source_flux = source_flux(source_radius, settings);
    if (!std::isfinite(total_source_flux)) {
        return std::nan("");
    }

    std::vector<double> image_area_counts(image_positions.size(), 0.0);
    std::vector<bool> skip(image_positions.size(), false);
    double total_count = 0.0;
    for (std::size_t i = 0; i < image_positions.size(); ++i) {
        if (skip[i]) {
            continue;
        }

        const double image_radius_value = std::hypot(image_positions[i].x, image_positions[i].y);
        double image_phi = wrap_angle(std::atan2(image_positions[i].y, image_positions[i].x));
        const double grid_radius = std::floor(image_radius_value / dr) * dr + 0.5 * dr;
        image_phi = std::floor(image_phi / dphi) * dphi + 0.5 * dphi;

        PolarBoundaryScratch scratch;
        scratch.row_indices_by_phi.assign(static_cast<std::size_t>(phi_bins), {});
        double count = trace_polar_boundary_rows(
            separation, mass_ratio, source, source_radius, settings, finite_magnifier, map_cache,
            grid_radius, image_phi, dphi, scratch);
        if (!scratch.rows.empty()) {
            const double reverse_start_radius = scratch.rows.front().max_r;
            count += trace_polar_boundary_rows(
                separation, mass_ratio, source, source_radius, settings, finite_magnifier, map_cache,
                reverse_start_radius, image_phi - dphi, -dphi, scratch);
        }

        total_count += count;
        image_area_counts[i] = count;

        for (std::size_t j = 0; j < image_positions.size(); ++j) {
            if (j == i) {
                continue;
            }
            const double other_radius = std::hypot(image_positions[j].x, image_positions[j].y);
            const double other_phi = wrap_angle(std::atan2(image_positions[j].y, image_positions[j].x));
            const double other_grid_radius = std::floor(other_radius / dr) * dr;
            const double other_grid_phi = std::floor(other_phi / dphi) * dphi;
            for (const auto& row : scratch.rows) {
                const double row_phi = wrap_angle(row.phi - 0.5 * std::abs(dphi));
                const double delta_phi = std::abs(wrap_angle(other_grid_phi - row_phi));
                const double wrapped_delta_phi = std::min(delta_phi, 2.0 * kPi - delta_phi);
                if (wrapped_delta_phi <= 1.01 * std::abs(dphi) &&
                    other_grid_radius >= row.min_r - 1.01 * dr &&
                    other_grid_radius <= row.max_r + 1.01 * dr) {
                    if (j < i) {
                        total_count -= image_area_counts[j];
                    } else {
                        skip[j] = true;
                    }
                    break;
                }
            }
        }
    }

    const double image_flux = total_count * dr * dphi;
    return image_flux / total_source_flux;
}

double legacy_limb_brightness(
    double normalized_radius2,
    const FiniteSourceSettings& settings,
    const FiniteSourceMagnifier* finite_magnifier)
{
    if (settings.limb_darkening_c == 0.0 && settings.limb_darkening_d == 0.0) {
        return 1.0;
    }
    return finite_magnifier != nullptr ?
        finite_magnifier->limb_darkening_table_brightness(normalized_radius2) :
        source_surface_brightness(normalized_radius2, settings);
}

std::vector<SourcePosition> legacy_residual_selected_images(
    const PointSourceMagnifier& point_magnifier,
    double separation,
    double mass_ratio,
    SourcePosition source);

std::vector<SourcePosition> critical_sources_at_phase(
    const BinaryLensMapper& mapper,
    double separation,
    double mass_ratio,
    double phase_angle)
{
    math::PolynomialRootSolver solver;
    const auto root_result = solver.solve(critical_curve_polynomial_coefficients(
        separation, mass_ratio, std::polar(1.0, phase_angle)));
    if (root_result.status != math::RootSolverStatus::ok) {
        return {};
    }

    std::vector<SourcePosition> sources;
    sources.reserve(root_result.roots.size());
    for (const auto& root : root_result.roots) {
        sources.push_back(map_binary_lens_real(mapper, root.real(), root.imag()));
    }
    return sources;
}

double nearest_critical_source_distance2(
    const BinaryLensMapper& mapper,
    double separation,
    double mass_ratio,
    SourcePosition source,
    double phase_angle,
    SourcePosition* nearest_source)
{
    double best = std::numeric_limits<double>::infinity();
    const auto critical_sources =
        critical_sources_at_phase(mapper, separation, mass_ratio, phase_angle);
    for (const auto& critical_source : critical_sources) {
        const double distance2 = distance_squared(critical_source, source);
        if (distance2 < best) {
            best = distance2;
            if (nearest_source != nullptr) {
                *nearest_source = critical_source;
            }
        }
    }
    return best;
}

SourcePosition refine_nearest_critical_source(
    const BinaryLensMapper& mapper,
    double separation,
    double mass_ratio,
    SourcePosition source,
    double center_phase,
    double phase_step)
{
    double left = center_phase - phase_step;
    double right = center_phase + phase_step;
    constexpr double golden = 0.61803398874989484820;
    double x1 = right - golden * (right - left);
    double x2 = left + golden * (right - left);
    SourcePosition nearest1;
    SourcePosition nearest2;
    double f1 = nearest_critical_source_distance2(
        mapper, separation, mass_ratio, source, x1, &nearest1);
    double f2 = nearest_critical_source_distance2(
        mapper, separation, mass_ratio, source, x2, &nearest2);

    for (int iter = 0; iter < 32; ++iter) {
        if (f1 > f2) {
            left = x1;
            x1 = x2;
            f1 = f2;
            nearest1 = nearest2;
            x2 = left + golden * (right - left);
            f2 = nearest_critical_source_distance2(
                mapper, separation, mass_ratio, source, x2, &nearest2);
        } else {
            right = x2;
            x2 = x1;
            f2 = f1;
            nearest2 = nearest1;
            x1 = right - golden * (right - left);
            f1 = nearest_critical_source_distance2(
                mapper, separation, mass_ratio, source, x1, &nearest1);
        }
    }

    return f1 < f2 ? nearest1 : nearest2;
}

std::vector<SourcePosition> legacy_augmented_image_seeds(
    const PointSourceMagnifier& point_magnifier,
    const BinaryLensMapper& mapper,
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    double hint_caustic_dist = std::numeric_limits<double>::infinity())
{
    std::vector<SourcePosition> seeds;
    const auto point_images = point_magnifier.binary_images(separation, mass_ratio, source);
    seeds.reserve(5);
    for (const auto& image : point_images) {
        seeds.push_back(image.position);
    }
    if (seeds.size() >= 5 || source_radius <= 0.0) {
        return seeds;
    }
    // Do not skip the caustic scan based on hint_caustic_dist alone: the
    // computed caustic distance can be slightly over-estimated (e.g. when a
    // phantom wrap-around segment distorts the branch grid search), causing a
    // false early exit when the source disk just straddles the caustic.

    const double source_radius2 = source_radius * source_radius;
    const int samples = 1400;
    const double phase_step = 2.0 * kPi / static_cast<double>(samples);
    double best_distance2 = std::numeric_limits<double>::infinity();
    double best_phase = 0.0;
    constexpr int nskip = 40;

    // Phase 1: find the first set of extra seeds (existing interleaved scan).
    for (int kphi = 0; kphi < nskip && seeds.size() < 5; ++kphi) {
        for (int jphi = 0; jphi < samples / nskip && seeds.size() < 5; ++jphi) {
            const int sample = jphi * nskip + kphi;
            const double phi = phase_step * static_cast<double>(sample);
            const auto critical_sources =
                critical_sources_at_phase(mapper, separation, mass_ratio, phi);
            for (const auto& critical_source : critical_sources) {
                const double distance2 = distance_squared(critical_source, source);
                if (distance2 < best_distance2) {
                    best_distance2 = distance2;
                    best_phase = phi;
                }
                if (distance2 >= source_radius2 || distance2 <= 0.0) {
                    continue;
                }
                const double distance = std::sqrt(distance2);
                const double fraction = (source_radius - distance) / distance * 0.01;
                const SourcePosition probe_source {
                    critical_source.x + (critical_source.x - source.x) * fraction,
                    critical_source.y + (critical_source.y - source.y) * fraction,
                };
                const auto probe_images = legacy_residual_selected_images(
                    point_magnifier, separation, mass_ratio, probe_source);
                if (probe_images.size() <= seeds.size()) {
                    continue;
                }
                seeds.clear();
                seeds.reserve(probe_images.size());
                for (const auto& image : probe_images) {
                    seeds.push_back(image);
                }
                if (seeds.size() >= 5) {
                    break;
                }
            }
            if (seeds.size() >= 5) {
                break;
            }
        }
    }
    if (seeds.size() < 5 && best_distance2 < source_radius2 && best_distance2 > 0.0) {
        const SourcePosition critical_source = refine_nearest_critical_source(
            mapper, separation, mass_ratio, source, best_phase, phase_step);
        const double distance = std::sqrt(distance_squared(critical_source, source));
        if (distance < source_radius && distance > 0.0) {
            const double fraction = (source_radius - distance) / distance * 0.01;
            const SourcePosition probe_source {
                critical_source.x + (critical_source.x - source.x) * fraction,
                critical_source.y + (critical_source.y - source.y) * fraction,
            };
            const auto probe_images = legacy_residual_selected_images(
                point_magnifier, separation, mass_ratio, probe_source);
            if (probe_images.size() > seeds.size()) {
                seeds = probe_images;
            }
        }
    }

    // Phase 2: when the source disk straddles the caustic at multiple arcs
    // (e.g. source center outside caustic but disk overlaps at two separate
    // regions), Phase 1 only seeds the first crossing.  Scan all caustic
    // samples again and accumulate seeds from any additional crossings found.
    if (seeds.size() >= 5) {
        for (int kphi = 0; kphi < nskip; ++kphi) {
            for (int jphi = 0; jphi < samples / nskip; ++jphi) {
                const int sample = jphi * nskip + kphi;
                const double phi = phase_step * static_cast<double>(sample);
                const auto critical_sources =
                    critical_sources_at_phase(mapper, separation, mass_ratio, phi);
                for (const auto& critical_source : critical_sources) {
                    const double distance2 = distance_squared(critical_source, source);
                    if (distance2 >= source_radius2 || distance2 <= 0.0) {
                        continue;
                    }
                    const double distance = std::sqrt(distance2);
                    const double fraction = (source_radius - distance) / distance * 0.01;
                    const SourcePosition probe_source {
                        critical_source.x + (critical_source.x - source.x) * fraction,
                        critical_source.y + (critical_source.y - source.y) * fraction,
                    };
                    const auto probe_images = legacy_residual_selected_images(
                        point_magnifier, separation, mass_ratio, probe_source);
                    if (probe_images.size() <= 3) {
                        continue;
                    }
                    for (const auto& img : probe_images) {
                        bool is_dup = false;
                        for (const auto& s : seeds) {
                            if (distance_squared(img, s) < source_radius2) {
                                is_dup = true;
                                break;
                            }
                        }
                        if (!is_dup) {
                            seeds.push_back(img);
                        }
                    }
                }
            }
        }
    }
    return seeds;
}

std::vector<SourcePosition> legacy_residual_selected_images(
    const PointSourceMagnifier& point_magnifier,
    double separation,
    double mass_ratio,
    SourcePosition source)
{
    std::vector<SourcePosition> images;
    const auto candidates =
        point_magnifier.binary_image_candidates(separation, mass_ratio, source);
    images.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        if (candidate.residual * candidate.residual < 1.0e-12) {
            images.push_back(candidate.position);
        }
    }
    return images;
}

double legacy_imagearea0_binary(
    const BinaryLensMapper& mapper,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings,
    const FiniteSourceMagnifier* finite_magnifier,
    SourcePosition seed,
    double dy,
    int& yi,
    LegacyImageAreaScratch& scratch)
{
    double countx = 0.0;
    double countall = 0.0;
    double dz2 = 99999999.9;
    const double incr = std::abs(dy);
    const double inv_incr = 1.0 / incr;
    double dx = incr;
    SourcePosition image = seed;
    double x0 = seed.x;
    const double source_radius2 = source_radius * source_radius;
    const double inv_source_radius2 = 1.0 / source_radius2;
    int guard = 0;
    const int max_steps = std::max(100000, settings.source_bins * settings.source_bins * 2000);

    while (++guard < max_steps) {
        const double dz2_last = dz2;
        const double mapped_distance2 =
            mapped_binary_lens_distance2(mapper, image.x, image.y, source);
        dz2 = mapped_distance2;

        scratch.ensure(static_cast<std::size_t>(yi));
        if (mapped_distance2 <= source_radius2) {
            if (dx == -incr && countx == 0.0) {
                scratch.xmax[static_cast<std::size_t>(yi)] = image.x - dx;
            }
            const double normalized_radius2 = mapped_distance2 * inv_source_radius2;
            countx += legacy_limb_brightness(normalized_radius2, settings, finite_magnifier);
        } else {
            if (dx == incr) {
                if (dz2_last <= source_radius2) {
                    scratch.xmax[static_cast<std::size_t>(yi)] = image.x;
                }
                dx = -incr;
                image.x = x0;
                scratch.xmin[static_cast<std::size_t>(yi)] = image.x + dx;
            } else {
                if (dz2_last <= source_radius2) {
                    scratch.xmin[static_cast<std::size_t>(yi)] = image.x;
                }
                if (yi != 0 && countx == 0.0) {
                    scratch.ensure(static_cast<std::size_t>(yi - 1));
                    if (image.x >= scratch.xmin[static_cast<std::size_t>(yi - 1)] - dx) {
                        image.x += dx;
                        continue;
                    }
                }

                countall += countx;
                scratch.ax[static_cast<std::size_t>(yi)] = countx;
                scratch.y[static_cast<std::size_t>(yi)] = image.y;
                scratch.dys[static_cast<std::size_t>(yi)] = dy;
                if (countx == 0.0) {
                    scratch.dys[static_cast<std::size_t>(yi)] = -dy;
                    break;
                }

                const int row_key = static_cast<int>(image.y * inv_incr + kLegacyIndexOffset);
                auto& row_indices = scratch.row_indices[row_key];
                for (const int index : row_indices) {
                    const auto existing = static_cast<std::size_t>(index);
                    if (scratch.xmin[static_cast<std::size_t>(yi)] + incr < scratch.xmax[existing] &&
                        scratch.xmax[static_cast<std::size_t>(yi)] - incr > scratch.xmin[existing]) {
                        return countall - countx;
                    }
                }
                row_indices.push_back(yi);

                ++yi;
                scratch.ensure(static_cast<std::size_t>(yi));
                dx = incr;
                x0 = scratch.xmax[static_cast<std::size_t>(yi - 1)];
                image.x = x0 - dx;
                image.y += dy;
                countx = 0.0;
            }
        }
        image.x += dx;
    }

    return countall;
}

double legacy_imagearea4_binary(
    const PointSourceMagnifier& point_magnifier,
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings,
    const FiniteSourceMagnifier* finite_magnifier,
    const std::vector<SourcePosition>* precomputed_seeds = nullptr)
{
    if ((settings.limb_darkening_c != 0.0 || settings.limb_darkening_d != 0.0) &&
        finite_magnifier != nullptr) {
        finite_magnifier->ensure_limb_darkening_table();
    }

    const auto mapper = make_binary_lens_mapper(separation, mass_ratio);
    const auto computed_images = precomputed_seeds == nullptr ?
        legacy_augmented_image_seeds(point_magnifier, mapper, separation, mass_ratio, source, source_radius) :
        std::vector<SourcePosition> {};
    const auto& images = precomputed_seeds == nullptr ? computed_images : *precomputed_seeds;
    if (images.empty() || source_radius <= 0.0) {
        return std::nan("");
    }
    const double nbin = static_cast<double>(std::max(settings.source_bins, 1));
    const double incr = source_radius / nbin;
    const double incr2_margin = 0.5 * incr * 1.01;

    double area = 0.0;
    std::vector<double> areaimage(images.size(), 0.0);
    std::vector<int> overlap(images.size(), 0);

    for (std::size_t image_index = 0; image_index < images.size(); ++image_index) {
        if (overlap[image_index] == 1) {
            continue;
        }

        LegacyImageAreaScratch scratch;
        scratch.ensure(1);
        double area0 = 0.0;
        double areai = 0.0;
        double dy = incr;
        int yi = 0;

        const SourcePosition seed = images[image_index];
        scratch.xmin[0] = seed.x;
        scratch.xmax[0] = seed.x;
        areai = legacy_imagearea0_binary(
            mapper, source, source_radius, settings, finite_magnifier, seed, dy, yi, scratch);

        dy = -incr;
        scratch.ensure(static_cast<std::size_t>(yi));
        const SourcePosition lower_seed {scratch.xmax[0], seed.y + dy};
        scratch.xmin[static_cast<std::size_t>(yi)] = scratch.xmin[0];
        scratch.xmax[static_cast<std::size_t>(yi)] = scratch.xmax[0];
        scratch.y[static_cast<std::size_t>(yi)] = scratch.y[0];
        scratch.dys[static_cast<std::size_t>(yi)] = dy;
        ++yi;
        areai += legacy_imagearea0_binary(
            mapper, source, source_radius, settings, finite_magnifier, lower_seed, dy, yi, scratch);

        int nyi = yi;
        double areabound = 0.0;
        for (int row = 0; row < nyi; ++row) {
            scratch.ensure(static_cast<std::size_t>(row + 1));
            const double dxmax =
                scratch.xmax[static_cast<std::size_t>(row + 1)] -
                scratch.xmax[static_cast<std::size_t>(row)];
            const double dxmin =
                scratch.xmin[static_cast<std::size_t>(row + 1)] -
                scratch.xmin[static_cast<std::size_t>(row)];
            if (scratch.ax[static_cast<std::size_t>(row + 1)] > 0.0) {
                if (dxmax > 1.1 * incr) {
                    const SourcePosition extra_seed {
                        scratch.xmax[static_cast<std::size_t>(row + 1)],
                        scratch.y[static_cast<std::size_t>(row)]};
                    scratch.ensure(static_cast<std::size_t>(yi));
                    scratch.xmin[static_cast<std::size_t>(yi)] = scratch.xmax[static_cast<std::size_t>(row)];
                    scratch.xmax[static_cast<std::size_t>(yi)] = scratch.xmax[static_cast<std::size_t>(row + 1)];
                    dy = -scratch.dys[static_cast<std::size_t>(row)];
                    scratch.dys[static_cast<std::size_t>(yi)] = dy;
                    ++yi;
                    area0 = legacy_imagearea0_binary(
                        mapper, source, source_radius, settings, finite_magnifier, extra_seed, dy, yi, scratch);
                    areai += area0;
                    areabound += area0;
                    if (area0 <= 0.0) {
                        --yi;
                    }
                }
                if (dxmin > 1.1 * incr) {
                    const SourcePosition extra_seed {
                        scratch.xmin[static_cast<std::size_t>(row + 1)] - incr,
                        scratch.y[static_cast<std::size_t>(row + 1)]};
                    scratch.ensure(static_cast<std::size_t>(yi));
                    scratch.xmin[static_cast<std::size_t>(yi)] = scratch.xmin[static_cast<std::size_t>(row)];
                    scratch.xmax[static_cast<std::size_t>(yi)] = scratch.xmin[static_cast<std::size_t>(row + 1)];
                    dy = scratch.dys[static_cast<std::size_t>(row)];
                    scratch.dys[static_cast<std::size_t>(yi)] = dy;
                    ++yi;
                    area0 = legacy_imagearea0_binary(
                        mapper, source, source_radius, settings, finite_magnifier, extra_seed, dy, yi, scratch);
                    areai += area0;
                    areabound += area0;
                    if (area0 <= 0.0) {
                        --yi;
                    }
                }
                if (dxmin < -1.1 * incr) {
                    const SourcePosition extra_seed {
                        scratch.xmin[static_cast<std::size_t>(row)] - incr,
                        scratch.y[static_cast<std::size_t>(row)]};
                    scratch.ensure(static_cast<std::size_t>(yi));
                    scratch.xmin[static_cast<std::size_t>(yi)] = scratch.xmin[static_cast<std::size_t>(row + 1)];
                    scratch.xmax[static_cast<std::size_t>(yi)] = scratch.xmin[static_cast<std::size_t>(row)];
                    dy = -scratch.dys[static_cast<std::size_t>(row)];
                    scratch.dys[static_cast<std::size_t>(yi)] = dy;
                    ++yi;
                    area0 = legacy_imagearea0_binary(
                        mapper, source, source_radius, settings, finite_magnifier, extra_seed, dy, yi, scratch);
                    areai += area0;
                    areabound += area0;
                    if (area0 <= 0.0) {
                        --yi;
                    }
                }
            }
            if (row == nyi - 1 && areabound > 0.0 && yi > nyi) {
                nyi = yi;
            }
        }

        area += areai;
        areaimage[image_index] = areai;

        for (std::size_t other = 0; other < images.size(); ++other) {
            if (other == image_index) {
                continue;
            }
            const auto& position = images[other];
            for (int row = 0; row < nyi; ++row) {
                const auto row_index = static_cast<std::size_t>(row);
                if (scratch.ax[row_index] <= 0.0) {
                    continue;
                }
                if (position.y >= scratch.y[row_index] - incr2_margin &&
                    position.y <= scratch.y[row_index] + incr2_margin &&
                    position.x >= scratch.xmin[row_index] - incr2_margin &&
                    position.x <= scratch.xmax[row_index] + incr2_margin) {
                    if (other < image_index) {
                        area -= areaimage[other];
                    } else {
                        overlap[other] = 1;
                    }
                    break;
                }
            }
        }
    }

    const double scale =
        source_flux(source_radius, settings) / (source_radius * source_radius) * nbin * nbin;
    return area / scale;
}

FiniteSourceResult fixed_inverse_ray_binary(
    const PointSourceMagnifier& point_magnifier,
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings,
    const FiniteSourceMagnifier* finite_magnifier,
    FiniteSourceDecision decision,
    double caustic_distance = std::numeric_limits<double>::infinity())
{
    const auto mapper = make_binary_lens_mapper(separation, mass_ratio);
    auto seeds = legacy_augmented_image_seeds(
        point_magnifier, mapper, separation, mass_ratio, source, source_radius,
        caustic_distance);
    // Phase 3: find caustic crossings that fall in the gap between the last
    // phase sample and phi=2*pi (missed by uniform 1400-point sampling).
    if (seeds.size() < 5 && finite_magnifier != nullptr) {
        finite_magnifier->legacy_augment_seeds_from_branches(
            separation, mass_ratio, source, source_radius, seeds);
    }
    double magnification;
    if (decision.method == FiniteSourceMethod::inverse_ray_polar) {
        magnification = inverse_ray_polar_boundary_binary(
            point_magnifier, separation, mass_ratio, source, source_radius,
            settings, finite_magnifier, nullptr, &seeds);
    } else {
        magnification = legacy_imagearea4_binary(
            point_magnifier, separation, mass_ratio, source, source_radius,
            settings, finite_magnifier, &seeds);
    }
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

void FiniteSourceMagnifier::ensure_limb_darkening_table() const
{
    const bool cache_matches = limb_darkening_table_valid_ &&
                               limb_darkening_table_c_ == settings_.limb_darkening_c &&
                               limb_darkening_table_d_ == settings_.limb_darkening_d &&
                               static_cast<int>(limb_darkening_table_.size()) == kLimbDarkeningTableSize + 1;
    if (cache_matches) {
        return;
    }

    limb_darkening_table_.resize(kLimbDarkeningTableSize + 1);
    for (int i = 0; i <= kLimbDarkeningTableSize; ++i) {
        const double normalized_radius2 =
            static_cast<double>(i) / static_cast<double>(kLimbDarkeningTableSize);
        limb_darkening_table_[static_cast<std::size_t>(i)] =
            source_surface_brightness(normalized_radius2, settings_);
    }
    limb_darkening_table_valid_ = true;
    limb_darkening_table_c_ = settings_.limb_darkening_c;
    limb_darkening_table_d_ = settings_.limb_darkening_d;
}

double FiniteSourceMagnifier::limb_darkening_table_brightness(double normalized_radius2) const
{
    const double bounded = std::clamp(normalized_radius2, 0.0, 1.0);
    const double index = bounded * static_cast<double>(kLimbDarkeningTableSize);
    const int lower = static_cast<int>(index);
    if (lower >= kLimbDarkeningTableSize) {
        return limb_darkening_table_[static_cast<std::size_t>(kLimbDarkeningTableSize)];
    }
    const double fraction = index - static_cast<double>(lower);
    const double left = limb_darkening_table_[static_cast<std::size_t>(lower)];
    const double right = limb_darkening_table_[static_cast<std::size_t>(lower + 1)];
    return left + fraction * (right - left);
}

double FiniteSourceMagnifier::legacy_limb_darkening_table_brightness(double normalized_radius2) const
{
    const double bounded = std::clamp(normalized_radius2, 0.0, 1.0);
    const int index = std::min(
        static_cast<int>(bounded * static_cast<double>(kLimbDarkeningTableSize)),
        kLimbDarkeningTableSize);
    return limb_darkening_table_[static_cast<std::size_t>(index)];
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
        // Build branch-position grid for fast per-segment distance queries.
        caustic_cache_branch_grid_.assign(
            static_cast<std::size_t>(caustic_cache_grid_size_ * caustic_cache_grid_size_), {});
        // max_seg_len is used as a safety margin when bounding the grid search
        // radius in legacy_binary_caustic_distance.  The wrap-around segment
        // (from j=n-1 back to j=0) can be spuriously long when branch tracking
        // swaps inner/outer caustic components near a crossing phase.  Excluding
        // it keeps max_seg_len tight.  Correctness is preserved because the
        // wrap-around gap is still found via the "prev" check on the j=0 entry
        // of each branch in legacy_binary_caustic_distance.
        double max_seg2 = 0.0;
        for (int b = 0; b < static_cast<int>(caustic_cache_branches_.size()); ++b) {
            const auto& br = caustic_cache_branches_[static_cast<std::size_t>(b)];
            const int n = static_cast<int>(br.size());
            for (int j = 0; j < n; ++j) {
                const auto& pt = br[static_cast<std::size_t>(j)];
                const auto& next_pt = br[static_cast<std::size_t>((j + 1) % n)];
                // Exclude wrap-around (j == n-1) from max_seg_len to avoid
                // inflating the search radius with tracking-artifact segments.
                if (j < n - 1) {
                    const double seg2 = distance_squared(pt, next_pt);
                    if (seg2 > max_seg2) {
                        max_seg2 = seg2;
                    }
                }
                const int ix = std::clamp(
                    static_cast<int>((pt.x - caustic_cache_min_x_) / caustic_cache_grid_step_x_),
                    0, caustic_cache_grid_size_ - 1);
                const int iy = std::clamp(
                    static_cast<int>((pt.y - caustic_cache_min_y_) / caustic_cache_grid_step_y_),
                    0, caustic_cache_grid_size_ - 1);
                caustic_cache_branch_grid_[
                    static_cast<std::size_t>(iy * caustic_cache_grid_size_ + ix)]
                    .push_back({b, j});
            }
        }
        caustic_cache_max_seg_len_ = std::sqrt(max_seg2);

        caustic_cache_valid_ = true;
        caustic_cache_separation_ = separation;
        caustic_cache_mass_ratio_ = mass_ratio;
        caustic_cache_bins_ = bins;
    }
}

double FiniteSourceMagnifier::legacy_binary_caustic_distance(
    double separation,
    double mass_ratio,
    SourcePosition source,
    double hint_nearest_point_dist) const
{
    ensure_legacy_caustic_cache(separation, mass_ratio);

    // Obtain nearest caustic POINT distance as an upper bound on segment distance.
    // The caller often already has this from legacy_binary_sampled_caustic_distance;
    // if so, the hint skips this O(N) scan.
    double nearest_dist = hint_nearest_point_dist;
    if (!std::isfinite(nearest_dist)) {
        double nearest2 = std::numeric_limits<double>::infinity();
        for (const auto& pt : caustic_cache_points_) {
            const double dx = source.x - pt.x;
            const double dy = source.y - pt.y;
            const double d2 = dx * dx + dy * dy;
            if (d2 < nearest2) {
                nearest2 = d2;
            }
        }
        nearest_dist = std::sqrt(nearest2);
    }

    if (caustic_cache_branch_grid_.empty()) {
        return nearest_dist;
    }

    // Search the branch grid within nearest_dist + max_seg_len to catch all segments
    // whose distance to source could be less than nearest_dist.
    const double seg_radius = nearest_dist + caustic_cache_max_seg_len_;
    const int ix0 = std::clamp(
        static_cast<int>((source.x - seg_radius - caustic_cache_min_x_) /
                         caustic_cache_grid_step_x_),
        0, caustic_cache_grid_size_ - 1);
    const int ix1 = std::clamp(
        static_cast<int>((source.x + seg_radius - caustic_cache_min_x_) /
                         caustic_cache_grid_step_x_),
        0, caustic_cache_grid_size_ - 1);
    const int iy0 = std::clamp(
        static_cast<int>((source.y - seg_radius - caustic_cache_min_y_) /
                         caustic_cache_grid_step_y_),
        0, caustic_cache_grid_size_ - 1);
    const int iy1 = std::clamp(
        static_cast<int>((source.y + seg_radius - caustic_cache_min_y_) /
                         caustic_cache_grid_step_y_),
        0, caustic_cache_grid_size_ - 1);

    double distance = nearest_dist;
    for (int iy = iy0; iy <= iy1; ++iy) {
        for (int ix = ix0; ix <= ix1; ++ix) {
            for (const auto& ref : caustic_cache_branch_grid_[
                    static_cast<std::size_t>(iy * caustic_cache_grid_size_ + ix)]) {
                const auto& branch =
                    caustic_cache_branches_[static_cast<std::size_t>(ref.branch)];
                const int n = static_cast<int>(branch.size());
                if (n < 2) {
                    continue;
                }
                const int prev = (ref.pos > 0) ? ref.pos - 1 : n - 1;
                const int next = (ref.pos + 1) % n;
                distance = std::min(distance,
                    point_segment_distance(source,
                        branch[static_cast<std::size_t>(prev)],
                        branch[static_cast<std::size_t>(ref.pos)]));
                distance = std::min(distance,
                    point_segment_distance(source,
                        branch[static_cast<std::size_t>(ref.pos)],
                        branch[static_cast<std::size_t>(next)]));
            }
        }
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

void FiniteSourceMagnifier::ensure_legacy_polar_map_cache(
    double separation,
    double mass_ratio,
    double source_radius) const
{
    const int source_bins = std::max(settings_.source_bins, 1);
    const double dr = source_radius / static_cast<double>(source_bins);
    const int phi_bins = std::max(16, static_cast<int>(2.0 * kPi / (dr * settings_.grid_ratio)));
    const double dphi = 2.0 * kPi / static_cast<double>(phi_bins);
    const int radial_count = std::max(3 * source_bins, 1);
    const int radial_min_index = static_cast<int>(1.0 / dr) - radial_count / 2;
    const bool cache_matches =
        polar_map_cache_valid_ &&
        polar_map_cache_separation_ == separation &&
        polar_map_cache_mass_ratio_ == mass_ratio &&
        polar_map_cache_source_radius_ == source_radius &&
        polar_map_cache_source_bins_ == source_bins &&
        polar_map_cache_grid_ratio_ == settings_.grid_ratio &&
        polar_map_cache_phi_bins_ == phi_bins &&
        polar_map_cache_radial_offset_min_index_ == radial_min_index &&
        static_cast<int>(polar_map_cache_radial_offsets_.size()) == radial_count;
    if (cache_matches) {
        return;
    }

    const auto mapper = make_binary_lens_mapper(separation, mass_ratio);
    polar_map_cache_radial_offset_min_index_ = radial_min_index;
    polar_map_cache_radial_offsets_.resize(static_cast<std::size_t>(radial_count));
    polar_map_cache_.resize(static_cast<std::size_t>(radial_count) * static_cast<std::size_t>(phi_bins));
    for (int ir = 0; ir < radial_count; ++ir) {
        polar_map_cache_radial_offsets_[static_cast<std::size_t>(ir)] = ir;
        const double radius = (radial_min_index + ir) * dr + 0.5 * dr;
        for (int iphi = 0; iphi < phi_bins; ++iphi) {
            const double phi = (iphi + 0.5) * dphi;
            const SourcePosition image {radius * std::cos(phi), radius * std::sin(phi)};
            polar_map_cache_[static_cast<std::size_t>(ir) * static_cast<std::size_t>(phi_bins) +
                             static_cast<std::size_t>(iphi)] =
                map_binary_lens_real(mapper, image.x, image.y);
        }
    }

    polar_map_cache_valid_ = true;
    polar_map_cache_separation_ = separation;
    polar_map_cache_mass_ratio_ = mass_ratio;
    polar_map_cache_source_radius_ = source_radius;
    polar_map_cache_source_bins_ = source_bins;
    polar_map_cache_grid_ratio_ = settings_.grid_ratio;
    polar_map_cache_dr_ = dr;
    polar_map_cache_dphi_ = dphi;
    polar_map_cache_phi_bins_ = phi_bins;
}

FiniteSourceResult FiniteSourceMagnifier::legacy_polar_memory_binary_mag(
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    double caustic_distance) const
{
    const PointSourceMagnifier point_magnifier;
    FiniteSourceDecision decision {
        FiniteSourceMethod::inverse_ray_polar,
        estimate_polar_cost(settings_),
        "polar cached inverse-ray",
    };
    const auto mapper = make_binary_lens_mapper(separation, mass_ratio);
    const auto seeds = legacy_augmented_image_seeds(
        point_magnifier, mapper, separation, mass_ratio, source, source_radius);
    const auto point_images = point_magnifier.binary_images(separation, mass_ratio, source);
    const double sampled_caustic_distance = legacy_binary_sampled_caustic_distance(
        separation, mass_ratio, source, source_radius);
    const double polar_fallback_distance =
        std::max(settings_.hex_threshold, 1.0) * source_radius;
    if (seeds.size() > point_images.size() ||
        (std::isfinite(sampled_caustic_distance) && sampled_caustic_distance < polar_fallback_distance) ||
        (std::isfinite(caustic_distance) && caustic_distance < polar_fallback_distance)) {
        const double magnification = legacy_imagearea4_binary(
            point_magnifier, separation, mass_ratio, source, source_radius, settings_, this, &seeds);
        decision.method = FiniteSourceMethod::inverse_ray_cartesian;
        decision.reason = "polar mode used cartesian fallback for caustic-crossing";
        if (!std::isfinite(magnification)) {
            return {magnification, 0, decision, std::nan(""), 0, false};
        }
        return {magnification, 0, decision, 0.0, 0, true};
    }
    ensure_legacy_polar_map_cache(separation, mass_ratio, source_radius);
    const PolarMapCacheView cache_view {
        &polar_map_cache_,
        &polar_map_cache_radial_offsets_,
        polar_map_cache_radial_offset_min_index_,
        polar_map_cache_phi_bins_,
        polar_map_cache_dr_,
    };
    const double magnification = inverse_ray_polar_boundary_binary(
        point_magnifier, separation, mass_ratio, source, source_radius, settings_, this, &cache_view, &seeds);
    if (!std::isfinite(magnification)) {
        return {magnification, 0, decision, std::nan(""), 0, false};
    }
    return {magnification, 0, decision, 0.0, 0, true};
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
        const auto point = point_magnifier.binary_mag0(separation, mass_ratio, source);
        FiniteSourceDecision decision {FiniteSourceMethod::point_source, 0, "zero source radius"};
        return cache_and_return({point.magnification, point.image_count, decision, 0.0, 0, true});
    }

    if (settings_.finite_mode <= 0) {
        FiniteSourceDecision decision {FiniteSourceMethod::point_source, 0, "point-source mode"};
        return cache_and_return({point_source_magnification, 0, decision, 0.0, 0, true});
    }

    const double cached_point_threshold = 2.0 * settings_.kinji_threshold * source_radius;
    double caustic_distance = legacy_binary_sampled_caustic_distance(
        separation, mass_ratio, source, cached_point_threshold);
    // Fast PS exit when source is outside the caustic bounding box by at least
    // cached_point_threshold (= 2·kinji·ρ > kinji·ρ).  In this case
    // legacy_binary_sampled_caustic_distance returns ∞ via the bbox early-exit path,
    // and the true caustic distance is guaranteed to exceed kinji_threshold·ρ.
    // Without this check the code falls through to the expensive O(N) segment scan.
    if (!std::isfinite(caustic_distance) &&
        (source.x < caustic_cache_min_x_ - cached_point_threshold ||
            source.x > caustic_cache_max_x_ + cached_point_threshold ||
            source.y < caustic_cache_min_y_ - cached_point_threshold ||
            source.y > caustic_cache_max_y_ + cached_point_threshold)) {
        FiniteSourceDecision decision {
            FiniteSourceMethod::point_source,
            settings_.caustic_bins * 4,
            "source outside caustic bounding box",
        };
        return cache_and_return({point_source_magnification, 0, decision, 0.0, 0, true});
    }
    // Skip the segment-based distance check when even the longest cached branch
    // segment cannot bring a caustic point at distance `caustic_distance` within
    // `cached_point_threshold`.
    if (std::isfinite(caustic_distance) &&
        caustic_distance >= cached_point_threshold + caustic_cache_max_seg_len_) {
        FiniteSourceDecision decision {
            FiniteSourceMethod::point_source,
            settings_.caustic_bins * 4,
            "caustic distance accepted point-source approximation",
        };
        return cache_and_return({point_source_magnification, 0, decision, 0.0, 0, true});
    }

    caustic_distance = legacy_binary_caustic_distance(separation, mass_ratio, source, caustic_distance);
    if (!std::isfinite(caustic_distance)) {
        caustic_distance = source_distance(source);
    }
    if (caustic_distance > settings_.kinji_threshold * source_radius) {
        FiniteSourceDecision decision {
            FiniteSourceMethod::point_source,
            settings_.caustic_bins * 4,
            "caustic distance accepted point-source approximation",
        };
        return cache_and_return({point_source_magnification, 0, decision, 0.0, 0, true});
    }
    if (caustic_distance > settings_.hex_threshold * source_radius) {
        FiniteSourceDecision decision {
            FiniteSourceMethod::hexadecapole,
            settings_.caustic_bins * 4 + kHexadecapoleEvaluations,
            "caustic distance accepted hexadecapole approximation",
        };
        return cache_and_return({hexadecapole_binary(point_magnifier, separation, mass_ratio, source, source_radius, settings_),
            0,
            decision,
            0.0,
            0,
            true});
    }

    if (settings_.finite_mode == 2) {
        return cache_and_return(legacy_polar_memory_binary_mag(
            separation, mass_ratio, source, source_radius, caustic_distance));
    }
    FiniteSourceDecision decision {
        FiniteSourceMethod::inverse_ray_cartesian,
        estimate_cartesian_cost(settings_),
        "cartesian inverse-ray",
    };
    return cache_and_return(fixed_inverse_ray_binary(
        point_magnifier, separation, mass_ratio, source, source_radius, settings_, this, decision,
        caustic_distance));
}

void FiniteSourceMagnifier::legacy_augment_seeds_from_branches(
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    std::vector<SourcePosition>& seeds) const
{
    ensure_legacy_caustic_cache(separation, mass_ratio);
    if (seeds.size() >= 5 || caustic_cache_branch_grid_.empty()) return;

    const PointSourceMagnifier point_magnifier;
    const double seg_search = source_radius + caustic_cache_max_seg_len_;
    const int gs = caustic_cache_grid_size_;
    const int ix0 = std::clamp(
        static_cast<int>((source.x - seg_search - caustic_cache_min_x_) / caustic_cache_grid_step_x_),
        0, gs - 1);
    const int ix1 = std::clamp(
        static_cast<int>((source.x + seg_search - caustic_cache_min_x_) / caustic_cache_grid_step_x_),
        0, gs - 1);
    const int iy0 = std::clamp(
        static_cast<int>((source.y - seg_search - caustic_cache_min_y_) / caustic_cache_grid_step_y_),
        0, gs - 1);
    const int iy1 = std::clamp(
        static_cast<int>((source.y + seg_search - caustic_cache_min_y_) / caustic_cache_grid_step_y_),
        0, gs - 1);

    for (int iy = iy0; iy <= iy1 && seeds.size() < 5; ++iy) {
        for (int ix = ix0; ix <= ix1 && seeds.size() < 5; ++ix) {
            for (const auto& ref :
                 caustic_cache_branch_grid_[static_cast<std::size_t>(iy * gs + ix)]) {
                const auto& branch =
                    caustic_cache_branches_[static_cast<std::size_t>(ref.branch)];
                const int n = static_cast<int>(branch.size());
                if (n < 2) continue;
                const int next = (ref.pos + 1) % n;
                // Skip the wrap-around segment (last point back to first).  When
                // branch-tracking swaps occur, this segment is a phantom artifact of
                // length ~O(1) that does not correspond to any real caustic arc, yet
                // it can pass through the source disk and produce spurious seed updates.
                if (next == 0) continue;
                const SourcePosition p0 = branch[static_cast<std::size_t>(ref.pos)];
                const SourcePosition p1 = branch[static_cast<std::size_t>(next)];
                if (point_segment_distance(source, p0, p1) >= source_radius) continue;

                const double seg_dx = p1.x - p0.x;
                const double seg_dy = p1.y - p0.y;
                const double seg_len2 = seg_dx * seg_dx + seg_dy * seg_dy;
                const double t = seg_len2 > 0.0 ?
                    std::clamp(
                        ((source.x - p0.x) * seg_dx + (source.y - p0.y) * seg_dy) / seg_len2,
                        0.0, 1.0) :
                    0.0;
                const SourcePosition nearest {p0.x + t * seg_dx, p0.y + t * seg_dy};
                const double distance = std::sqrt(distance_squared(nearest, source));
                if (distance <= 0.0 || distance >= source_radius) continue;

                // Step 5% of source_radius past the nearest caustic segment point
                // toward the interior of the caustic. This is large enough to cross
                // the segment-to-true-caustic approximation error (one inter-sample
                // spacing) without landing too close to a fold caustic where two
                // merging images are nearly degenerate.
                const double step = source_radius * 0.05 / distance;
                const SourcePosition probe_source {
                    nearest.x + (nearest.x - source.x) * step,
                    nearest.y + (nearest.y - source.y) * step,
                };
                // Only use images from regions that actually overlap the source disk.
                // A probe outside the source disk belongs to a caustic region that
                // the disk does not straddle (e.g., a fold caustic tangent to the
                // disk edge) and its images would be seeds for the wrong area.
                if (distance_squared(probe_source, source) >= source_radius * source_radius) {
                    continue;
                }
                const auto probe_images = legacy_residual_selected_images(
                    point_magnifier, separation, mass_ratio, probe_source);
                if (probe_images.size() > seeds.size()) {
                    seeds = probe_images;
                }
            }
        }
    }
}

} // namespace lcbinint::magnification
