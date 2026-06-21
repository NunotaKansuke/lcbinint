#include "lcbinint/magnification/finite_source_magnifier.hpp"

#include "lcbinint/magnification/point_source_magnifier.hpp"
#include "lcbinint/math/polynomial_roots.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace lcbinint::magnification {
namespace {

constexpr double kHighMagnificationPolarThreshold = 10.0;
constexpr double kSqrtHalf = 0.70710678118654752440;
constexpr double kPi = 3.14159265358979323846;
constexpr int kMaxRefinementLevels = 3;
constexpr int kHexadecapoleEvaluations = 13;
constexpr int kLimbDarkeningTableSize = 5000;
constexpr int kLegacyIndexOffset = 2000000;
constexpr int kLocal7TileN = 8;
constexpr int kLocal7MinTileN = 2;
constexpr int kLocal7MaxSplitLevel = 8;
constexpr double kLocal7DetMin = 1.0e-5;
constexpr double kLocal7LambdaMin = 1.0e-5;
constexpr double kLocal7CenterTol = 1.0e-2;
constexpr double kLocal7MapErrMax = 2.0e-2;
constexpr double kLocal7DetVarMax = 2.5e-1;
constexpr double kLocal7AreaJacMax = 1.0e8;
constexpr double kLocal7AuditTol = 3.0e-2;
constexpr double kLocal7BoundaryJump = 1.1;
constexpr double kLocal7SpineAreaJacMin = 100.0;
constexpr double kLocal7SpineDetMax = 1.0e-2;
constexpr double kLocal7SpineMaxStepCells = 256.0;
constexpr double kLocal7SpineMinStepCells = 0.125;
constexpr int kLocal7SpineMaxPoints = 200000;
constexpr int kLocal7SpineMaxNormalSamples = 20000000;
constexpr int kLocal7SpineOutsideStop = 3;
constexpr double kLocal7SpineCurvatureMax = 0.55;
constexpr double kLocal7SpineNormalSubstep = 1.0;
constexpr double kLocal7SpineTargetTolCells = 2.0;
constexpr double kLocal7SpineFrameDetMin = 1.0e-9;
constexpr double kLocal7SpineFrameLambdaMin = 1.0e-9;
constexpr double kLocal7SpineFrameAreaJacMax = 1.0e12;
constexpr double kLocal7SpinePairDistanceCells = 50000.0;
constexpr double kLocal7SpineMaxRelativeArea = 1.0e8;

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

struct BinaryLensMapper {
    Complex separation;
    double m1 = 0.0;
    double m2 = 0.0;
};

struct Local7Frame {
    Complex za;
    SourcePosition wa;
    SourcePosition sa;
    double gamma_r = 0.0;
    double gamma_i = 0.0;
    double beta_r = 0.0;
    double beta_i = 0.0;
    double kappa_r = 0.0;
    double kappa_i = 0.0;
    double lambda_l = 0.0;
    double lambda_s = 0.0;
    double det_j = 0.0;
    double area_jac = 0.0;
    double e_lx = 0.0;
    double e_ly = 0.0;
    double e_sx = 0.0;
    double e_sy = 0.0;
    bool ok = false;
};

struct Local7Tile {
    double cx = 0.0;
    double cy = 0.0;
    double hs = 0.0;
    int ix0 = 0;
    int ix1 = 0;
    int iy0 = 0;
    int iy1 = 0;
    int level = 0;
};

struct Local7Stats {
    bool fallback = false;
    int nframes = 0;
    int ntiles = 0;
    int nsplit = 0;
    int naudit = 0;
    int fallback_reason = 0;
    double max_maperr = 0.0;
    double max_detvar = 0.0;
    double max_center_resid = 0.0;
    double max_audit_resid = 0.0;
};

struct Local7TimingStats {
    using Clock = std::chrono::steady_clock;

    Clock::time_point total_start = Clock::now();
    double seed_ms = 0.0;
    double frame_ms = 0.0;
    double safe_scan_ms = 0.0;
    double exact_scan_ms = 0.0;
    double check_ms = 0.0;
    long long exact_lens_evals = 0;
    long long derivative_step_samples = 0;
    long long exact_samples = 0;
    long long spine_points = 0;
    long long spine_normal_samples = 0;
    long long spine_exact_lens_evals = 0;
    int spine_branches = 0;
    int spine_fallbacks = 0;
    int spine_fallback_reason = 0;
    double spine_ms = 0.0;
    int caustic_born_branches = 0;
    int safe_branches = 0;
    int fallback_calls = 0;

    static double elapsed_ms(Clock::time_point start)
    {
        return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    }
};

struct Local7ScopedTimer {
    Local7TimingStats::Clock::time_point start;
    double* accumulator = nullptr;

    explicit Local7ScopedTimer(double* output)
        : start(Local7TimingStats::Clock::now()), accumulator(output)
    {
    }

    ~Local7ScopedTimer()
    {
        if (accumulator != nullptr) {
            *accumulator += Local7TimingStats::elapsed_ms(start);
        }
    }
};

double legacy_limb_brightness(
    double normalized_radius2,
    const FiniteSourceSettings& settings,
    const FiniteSourceMagnifier* finite_magnifier);
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

bool local7_derivatives_binary(
    Complex z,
    const BinaryLensMapper& mapper,
    double* gr,
    double* gi,
    double* br,
    double* bi,
    double* kr,
    double* ki)
{
    *gr = 0.0;
    *gi = 0.0;
    *br = 0.0;
    *bi = 0.0;
    *kr = 0.0;
    *ki = 0.0;

    const Complex lenses[2] = {mapper.separation, Complex(0.0, 0.0)};
    const double masses[2] = {mapper.m1, mapper.m2};
    for (int i = 0; i < 2; ++i) {
        const double dx = lenses[i].real() - z.real();
        const double dy = z.imag() - lenses[i].imag();
        const double r2 = dx * dx + dy * dy;
        if (r2 <= 1.0e-30 || !std::isfinite(r2)) {
            return false;
        }
        const double r4 = r2 * r2;
        const double r6 = r4 * r2;
        const double r8 = r4 * r4;
        const double dx2 = dx * dx;
        const double dy2 = dy * dy;
        const double dx3 = dx2 * dx;
        const double dy3 = dy2 * dy;
        const double dx4 = dx2 * dx2;
        const double dy4 = dy2 * dy2;
        const double mass = masses[i];

        *gr += mass * (dx2 - dy2) / r4;
        *gi += mass * (-2.0 * dx * dy) / r4;
        *br += 2.0 * mass * dx * (dx2 - 3.0 * dy2) / r6;
        *bi += 2.0 * mass * (dy3 - 3.0 * dx2 * dy) / r6;
        *kr += 6.0 * mass * (dx4 - 6.0 * dx2 * dy2 + dy4) / r8;
        *ki += 6.0 * mass * (-4.0 * dx3 * dy + 4.0 * dx * dy3) / r8;
    }
    return std::isfinite(*gr) && std::isfinite(*gi) && std::isfinite(*br) &&
           std::isfinite(*bi) && std::isfinite(*kr) && std::isfinite(*ki);
}

bool local7_make_frame(
    Complex za,
    SourcePosition source,
    const BinaryLensMapper& mapper,
    Local7Frame* frame)
{
    double gr = 0.0;
    double gi = 0.0;
    double br = 0.0;
    double bi = 0.0;
    double kr = 0.0;
    double ki = 0.0;
    if (!local7_derivatives_binary(za, mapper, &gr, &gi, &br, &bi, &kr, &ki)) {
        return false;
    }

    const double g = std::hypot(gr, gi);
    const double lambda_s = 1.0 + g;
    const double lambda_l = 1.0 - g;
    const double det_j = lambda_l * lambda_s;
    const double abs_det = std::abs(det_j);
    if (abs_det <= 0.0 || !std::isfinite(abs_det)) {
        return false;
    }

    const double phi = 0.5 * std::atan2(gi, gr);
    const SourcePosition wa = map_binary_lens_real(mapper, za.real(), za.imag());
    *frame = {};
    frame->za = za;
    frame->wa = wa;
    frame->sa = {wa.x - source.x, wa.y - source.y};
    frame->gamma_r = gr;
    frame->gamma_i = gi;
    frame->beta_r = br;
    frame->beta_i = bi;
    frame->kappa_r = kr;
    frame->kappa_i = ki;
    frame->lambda_l = lambda_l;
    frame->lambda_s = lambda_s;
    frame->det_j = det_j;
    frame->area_jac = 1.0 / abs_det;
    frame->e_lx = -std::sin(phi);
    frame->e_ly = std::cos(phi);
    frame->e_sx = std::cos(phi);
    frame->e_sy = std::sin(phi);
    frame->ok = std::isfinite(frame->area_jac);
    return frame->ok;
}

Complex local7_apply_inverse_linear(const Local7Frame& frame, double sx, double sy)
{
    const double dx = sx - frame.sa.x;
    const double dy = sy - frame.sa.y;
    const double xi = dx * frame.e_lx + dy * frame.e_ly;
    const double eta = dx * frame.e_sx + dy * frame.e_sy;
    return {
        (xi / frame.lambda_l) * frame.e_lx + (eta / frame.lambda_s) * frame.e_sx,
        (xi / frame.lambda_l) * frame.e_ly + (eta / frame.lambda_s) * frame.e_sy,
    };
}

Complex local7_apply_inverse_jacobian(const Local7Frame& frame, Complex residual)
{
    const double xi = residual.real() * frame.e_lx + residual.imag() * frame.e_ly;
    const double eta = residual.real() * frame.e_sx + residual.imag() * frame.e_sy;
    return {
        (xi / frame.lambda_l) * frame.e_lx + (eta / frame.lambda_s) * frame.e_sx,
        (xi / frame.lambda_l) * frame.e_ly + (eta / frame.lambda_s) * frame.e_sy,
    };
}

Complex local7_correct_quadratic(const Local7Frame& frame, Complex dz0)
{
    const Complex beta(frame.beta_r, frame.beta_i);
    const Complex residual = 0.5 * beta * std::conj(dz0) * std::conj(dz0);
    return dz0 - local7_apply_inverse_jacobian(frame, residual);
}

Complex local7_approx_image(const Local7Frame& frame, double sx, double sy)
{
    const Complex dz0 = local7_apply_inverse_linear(frame, sx, sy);
    return frame.za + local7_correct_quadratic(frame, dz0);
}

double local7_det_taylor(const Local7Frame& frame, Complex dz)
{
    const Complex gamma(frame.gamma_r, frame.gamma_i);
    const Complex beta(frame.beta_r, frame.beta_i);
    const Complex kappa(frame.kappa_r, frame.kappa_i);
    const Complex cdz = std::conj(dz);
    const Complex estimate = gamma + beta * cdz + 0.5 * kappa * cdz * cdz;
    return 1.0 - std::norm(estimate);
}

bool local7_can_use_derivative_steps(const Local7Frame& frame, double source_radius)
{
    const double abs_lambda_l = std::abs(frame.lambda_l);
    const double abs_lambda_s = std::abs(frame.lambda_s);
    const double abs_det = std::abs(frame.det_j);
    if (abs_lambda_l < 0.05 || abs_lambda_s < 0.05 || abs_det < 0.05 ||
        frame.area_jac < 1.5 || frame.area_jac > 20.0) {
        return false;
    }

    const double gamma_abs = std::hypot(frame.gamma_r, frame.gamma_i);
    const double beta_abs = std::hypot(frame.beta_r, frame.beta_i);
    const double dzmax = std::hypot(source_radius / abs_lambda_l, source_radius / abs_lambda_s);
    const double det_variation = 2.0 * gamma_abs * beta_abs * dzmax / abs_det;
    return std::isfinite(det_variation) && det_variation < 0.15;
}

bool local7_reanchor_tile(
    const Local7Frame& parent,
    const Local7Tile& tile,
    SourcePosition source,
    const BinaryLensMapper& mapper,
    double source_radius,
    Local7Frame* out,
    Local7Stats* stats)
{
    Complex zc = local7_approx_image(parent, tile.cx, tile.cy);
    SourcePosition mapped = map_binary_lens_real(mapper, zc.real(), zc.imag());
    Complex residual(mapped.x - (source.x + tile.cx), mapped.y - (source.y + tile.cy));
    double scaled_residual = std::abs(residual) / source_radius;
    stats->max_center_resid = std::max(stats->max_center_resid, scaled_residual);

    if (scaled_residual > kLocal7CenterTol) {
        Local7Frame center_frame;
        if (!local7_make_frame(zc, source, mapper, &center_frame) ||
            std::abs(center_frame.lambda_l) < kLocal7LambdaMin ||
            std::abs(center_frame.lambda_s) < kLocal7LambdaMin) {
            return false;
        }
        zc -= local7_apply_inverse_jacobian(center_frame, residual);
        mapped = map_binary_lens_real(mapper, zc.real(), zc.imag());
        residual = Complex(mapped.x - (source.x + tile.cx), mapped.y - (source.y + tile.cy));
        scaled_residual = std::abs(residual) / source_radius;
        stats->max_center_resid = std::max(stats->max_center_resid, scaled_residual);
        if (scaled_residual > kLocal7CenterTol) {
            return false;
        }
    }

    if (!local7_make_frame(zc, source, mapper, out)) {
        return false;
    }
    ++stats->nframes;
    return true;
}

bool local7_tile_trust(
    const Local7Frame& frame,
    const Local7Tile& tile,
    double source_radius,
    Local7Stats* stats)
{
    if (!frame.ok || std::abs(frame.lambda_l) < kLocal7LambdaMin ||
        std::abs(frame.lambda_s) < kLocal7LambdaMin ||
        std::abs(frame.det_j) < kLocal7DetMin || !std::isfinite(frame.area_jac) ||
        frame.area_jac > kLocal7AreaJacMax) {
        return false;
    }

    const double h = std::sqrt(2.0) * tile.hs;
    const double dzmax = std::hypot(
        h / std::abs(frame.lambda_l),
        h / std::abs(frame.lambda_s));
    const double gamma_abs = std::hypot(frame.gamma_r, frame.gamma_i);
    const double beta_abs = std::hypot(frame.beta_r, frame.beta_i);
    const double kappa_abs = std::hypot(frame.kappa_r, frame.kappa_i);
    const double map_error = kappa_abs * dzmax * dzmax * dzmax / 6.0;
    const double det_variation =
        2.0 * gamma_abs * beta_abs * dzmax / std::abs(frame.det_j);
    stats->max_maperr = std::max(stats->max_maperr, map_error / source_radius);
    stats->max_detvar = std::max(stats->max_detvar, det_variation);

    if (map_error / source_radius >= kLocal7MapErrMax || det_variation >= kLocal7DetVarMax) {
        return false;
    }
    if (std::abs(frame.det_j) < 6.0 * gamma_abs * beta_abs * dzmax) {
        return false;
    }
    return true;
}

bool local7_audit_tile(
    const Local7Frame& frame,
    const Local7Tile& tile,
    SourcePosition source,
    const BinaryLensMapper& mapper,
    double source_radius,
    Local7Stats* stats)
{
    const double candidates[5][2] = {
        {tile.cx, tile.cy},
        {tile.cx - tile.hs, tile.cy - tile.hs},
        {tile.cx + tile.hs, tile.cy - tile.hs},
        {tile.cx - tile.hs, tile.cy + tile.hs},
        {tile.cx + tile.hs, tile.cy + tile.hs},
    };
    const int audit_count = std::abs(frame.det_j) < 1.0e-3 ? 5 : 1;
    const double radius2 = source_radius * source_radius;
    bool audited = false;
    for (int i = 0; i < audit_count; ++i) {
        const double sx = candidates[i][0];
        const double sy = candidates[i][1];
        if (sx * sx + sy * sy > radius2) {
            continue;
        }
        audited = true;
        ++stats->naudit;
        const Complex approx = local7_approx_image(frame, sx, sy);
        const SourcePosition mapped = map_binary_lens_real(mapper, approx.real(), approx.imag());
        const double residual =
            std::hypot(mapped.x - (source.x + sx), mapped.y - (source.y + sy)) / source_radius;
        stats->max_audit_resid = std::max(stats->max_audit_resid, residual);
        if (!std::isfinite(residual) || residual > kLocal7AuditTol) {
            return false;
        }
    }
    return audited;
}

double local7_integrate_tile_samples(
    const Local7Frame& frame,
    const Local7Tile& tile,
    double source_radius,
    const FiniteSourceSettings& settings,
    const FiniteSourceMagnifier* finite_magnifier)
{
    const int bins = std::max(settings.source_bins, 1);
    const double step = source_radius / static_cast<double>(bins);
    const double radius2 = source_radius * source_radius;
    const double inv_radius2 = 1.0 / radius2;
    double count = 0.0;
    for (int ix = tile.ix0; ix <= tile.ix1; ++ix) {
        const double sx = static_cast<double>(ix) * step;
        for (int iy = tile.iy0; iy <= tile.iy1; ++iy) {
            const double sy = static_cast<double>(iy) * step;
            const double qld = (sx * sx + sy * sy) * inv_radius2;
            if (qld > 1.0) {
                continue;
            }
            const Complex dz0 = local7_apply_inverse_linear(frame, sx, sy);
            const Complex dz = local7_correct_quadratic(frame, dz0);
            const double det = local7_det_taylor(frame, dz);
            if (!std::isfinite(det) || std::abs(det) < kLocal7DetMin) {
                return std::nan("");
            }
            count += legacy_limb_brightness(qld, settings, finite_magnifier) / std::abs(det);
        }
    }
    return count;
}

bool local7_tile_intersects_source_disk(const Local7Tile& tile, double source_radius)
{
    const double xmin = tile.cx - tile.hs;
    const double xmax = tile.cx + tile.hs;
    const double ymin = tile.cy - tile.hs;
    const double ymax = tile.cy + tile.hs;
    const double nearest_x = xmin > 0.0 ? xmin : (xmax < 0.0 ? xmax : 0.0);
    const double nearest_y = ymin > 0.0 ? ymin : (ymax < 0.0 ? ymax : 0.0);
    return nearest_x * nearest_x + nearest_y * nearest_y <= source_radius * source_radius;
}

double local7_integrate_tile_recursive(
    const Local7Frame& parent,
    const Local7Tile& tile,
    SourcePosition source,
    const BinaryLensMapper& mapper,
    double source_radius,
    const FiniteSourceSettings& settings,
    const FiniteSourceMagnifier* finite_magnifier,
    Local7Stats* stats)
{
    if (stats->fallback) {
        return 0.0;
    }
    if (!local7_tile_intersects_source_disk(tile, source_radius)) {
        return 0.0;
    }
    ++stats->ntiles;

    Local7Frame frame;
    const bool reanchored =
        local7_reanchor_tile(parent, tile, source, mapper, source_radius, &frame, stats);
    const bool trusted = reanchored && local7_tile_trust(frame, tile, source_radius, stats) &&
                         local7_audit_tile(frame, tile, source, mapper, source_radius, stats);
    if (trusted) {
        const double count =
            local7_integrate_tile_samples(frame, tile, source_radius, settings, finite_magnifier);
        if (std::isfinite(count)) {
            return count;
        }
    }

    const int side_x = tile.ix1 - tile.ix0 + 1;
    const int side_y = tile.iy1 - tile.iy0 + 1;
    const int side = std::max(side_x, side_y);
    if (side <= kLocal7MinTileN || tile.level >= kLocal7MaxSplitLevel) {
        stats->fallback = true;
        stats->fallback_reason = trusted ? 7 : 6;
        return 0.0;
    }

    ++stats->nsplit;
    const int mid_x = (tile.ix0 + tile.ix1) / 2;
    const int mid_y = (tile.iy0 + tile.iy1) / 2;
    const int ranges[4][4] = {
        {tile.ix0, mid_x, tile.iy0, mid_y},
        {mid_x + 1, tile.ix1, tile.iy0, mid_y},
        {tile.ix0, mid_x, mid_y + 1, tile.iy1},
        {mid_x + 1, tile.ix1, mid_y + 1, tile.iy1},
    };
    const double step = source_radius / static_cast<double>(std::max(settings.source_bins, 1));
    double count = 0.0;
    for (const auto& range : ranges) {
        if (range[0] > range[1] || range[2] > range[3]) {
            continue;
        }
        Local7Tile child;
        child.ix0 = range[0];
        child.ix1 = range[1];
        child.iy0 = range[2];
        child.iy1 = range[3];
        child.level = tile.level + 1;
        child.cx = 0.5 * static_cast<double>(child.ix0 + child.ix1) * step;
        child.cy = 0.5 * static_cast<double>(child.iy0 + child.iy1) * step;
        child.hs = 0.5 * static_cast<double>(std::max(child.ix1 - child.ix0 + 1, child.iy1 - child.iy0 + 1)) * step;
        count += local7_integrate_tile_recursive(
            frame.ok ? frame : parent,
            child,
            source,
            mapper,
            source_radius,
            settings,
            finite_magnifier,
            stats);
        if (stats->fallback) {
            return 0.0;
        }
    }
    return count;
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
    const FiniteSourceMagnifier* finite_magnifier,
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
    const auto mapper = make_binary_lens_mapper(separation, mass_ratio);
    if (!uniform_source && finite_magnifier != nullptr) {
        finite_magnifier->ensure_limb_darkening_table();
    }

    double image_flux = 0.0;
    for (const auto& image : images) {
        const double half_width = image_radius(source_radius, image.jacobian_determinant);
        const double step = 2.0 * half_width / static_cast<double>(grid_bins);
        const double cell_area = step * step;
        const double x0 = image.position.x - half_width + 0.5 * step;
        const double y0 = image.position.y - half_width + 0.5 * step;
        for (int ix = 0; ix < grid_bins; ++ix) {
            const double x = x0 + ix * step;
            for (int iy = 0; iy < grid_bins; ++iy) {
                const double y = y0 + iy * step;
                const double mapped_distance2 = mapped_binary_lens_distance2(mapper, x, y, source);
                if (mapped_distance2 <= source_radius2) {
                    image_flux += uniform_source ?
                                      cell_area :
                                      cell_area * finite_magnifier->limb_darkening_table_brightness(
                                                      mapped_distance2 / source_radius2);
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
    const FiniteSourceMagnifier* finite_magnifier,
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
    const auto mapper = make_binary_lens_mapper(separation, mass_ratio);
    if (!uniform_source && finite_magnifier != nullptr) {
        finite_magnifier->ensure_limb_darkening_table();
    }

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
                const double mapped_distance2 = mapped_binary_lens_distance2(mapper, x, y, source);
                if (mapped_distance2 <= source_radius2) {
                    image_flux += uniform_source ?
                                      cell_area :
                                      cell_area * finite_magnifier->limb_darkening_table_brightness(
                                                      mapped_distance2 / source_radius2);
                }
            }
        }
    }
    return image_flux / total_source_flux;
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
        finite_magnifier->legacy_limb_darkening_table_brightness(normalized_radius2) :
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
    double source_radius)
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

    const double source_radius2 = source_radius * source_radius;
    const int samples = 1400;
    const double phase_step = 2.0 * kPi / static_cast<double>(samples);
    double best_distance2 = std::numeric_limits<double>::infinity();
    double best_phase = 0.0;
    constexpr int nskip = 40;
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
    if (seeds.size() < 5 && std::isfinite(best_distance2)) {
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

SourcePosition local7_image_from_uv(const Local7Frame& frame, double step_l, double step_s, SourcePosition uv)
{
    const double dl = uv.x * step_l;
    const double ds = uv.y * step_s;
    return {
        frame.za.real() + dl * frame.e_lx + ds * frame.e_sx,
        frame.za.imag() + dl * frame.e_ly + ds * frame.e_sy,
    };
}

SourcePosition local7_seed_to_uv(const Local7Frame& frame, double step_l, double step_s, SourcePosition seed)
{
    const double dx = seed.x - frame.za.real();
    const double dy = seed.y - frame.za.imag();
    return {
        (dx * frame.e_lx + dy * frame.e_ly) / step_l,
        (dx * frame.e_sx + dy * frame.e_sy) / step_s,
    };
}

struct Local7SpinePoint {
    SourcePosition image;
    SourcePosition source_offset;
    Local7Frame frame;
    double half_weight = 0.0;
};

struct Local7SpineEligibility {
    bool ok = false;
    std::size_t pair_index = 0;
    int reason = 0;
};

bool local7_is_spine_candidate(const Local7Frame& frame)
{
    return frame.ok &&
           std::isfinite(frame.area_jac) &&
           frame.area_jac >= kLocal7SpineAreaJacMin &&
           std::abs(frame.det_j) <= kLocal7SpineDetMax;
}

Local7SpineEligibility local7_spine_eligibility(
    const std::vector<SourcePosition>& seeds,
    const std::vector<int>& overlap,
    std::size_t image_index,
    const Local7Frame& frame,
    SourcePosition source,
    double source_step,
    const BinaryLensMapper& mapper,
    int caustic_born_branches)
{
    if (caustic_born_branches <= 0) {
        return {false, 0, 30};
    }
    if (!local7_is_spine_candidate(frame)) {
        return {false, 0, 31};
    }

    const auto nearest_partner = [&](std::size_t from_index, const Local7Frame& from_frame) {
        double best_distance = std::numeric_limits<double>::infinity();
        std::size_t best_index = seeds.size();
        int candidate_count = 0;
        for (std::size_t other = 0; other < seeds.size(); ++other) {
            if (other == from_index || overlap[other] == 1) {
                continue;
            }
            Local7Frame other_frame;
            if (!local7_make_frame(Complex(seeds[other].x, seeds[other].y), source, mapper, &other_frame) ||
                !local7_is_spine_candidate(other_frame) ||
                std::signbit(other_frame.det_j) == std::signbit(from_frame.det_j)) {
                continue;
            }
            const double image_distance =
                std::hypot(seeds[other].x - seeds[from_index].x, seeds[other].y - seeds[from_index].y);
            if (image_distance > kLocal7SpinePairDistanceCells * source_step) {
                continue;
            }
            ++candidate_count;
            if (image_distance < best_distance) {
                best_distance = image_distance;
                best_index = other;
            }
        }
        return std::pair<std::size_t, int> {best_index, candidate_count};
    };

    const auto [partner_index, partner_count] = nearest_partner(image_index, frame);
    if (partner_count < 1 || partner_index >= seeds.size()) {
        return {false, 0, 33};
    }
    if (partner_count > 1) {
        Local7Frame partner_frame;
        if (!local7_make_frame(Complex(seeds[partner_index].x, seeds[partner_index].y), source, mapper, &partner_frame)) {
            return {false, 0, 34};
        }
        const auto [mutual_index, mutual_count] = nearest_partner(partner_index, partner_frame);
        if (mutual_count < 1 || mutual_index != image_index) {
            return {false, 0, 35};
        }
    }
    if (partner_index < image_index) {
        return {false, 0, 36};
    }
    return {true, partner_index, 0};
}

double local7_spine_step(
    const Local7Frame& frame,
    double source_step,
    double source_radius)
{
    const double abs_lambda = std::max(std::abs(frame.lambda_l), kLocal7LambdaMin);
    double step = source_step / abs_lambda;
    const double beta_abs = std::hypot(frame.beta_r, frame.beta_i);
    if (beta_abs > 0.0 && std::isfinite(beta_abs)) {
        const double nonlinear_cap =
            2.0 * source_radius / (abs_lambda + std::sqrt(abs_lambda * abs_lambda + 2.0 * beta_abs * source_radius));
        if (std::isfinite(nonlinear_cap) && nonlinear_cap > 0.0) {
            step = std::min(step, nonlinear_cap);
        }
    }
    step = std::min(step, kLocal7SpineMaxStepCells * source_step);
    step = std::max(step, kLocal7SpineMinStepCells * source_step);
    return step;
}

bool local7_spine_frame_safe(const Local7Frame& frame)
{
    if (!frame.ok || !std::isfinite(frame.area_jac) ||
        std::abs(frame.lambda_l) < kLocal7SpineFrameLambdaMin ||
        std::abs(frame.lambda_s) < kLocal7SpineFrameLambdaMin ||
        std::abs(frame.det_j) < kLocal7SpineFrameDetMin ||
        frame.area_jac > kLocal7SpineFrameAreaJacMax) {
        return false;
    }
    return true;
}

bool local7_spine_try_step(
    const BinaryLensMapper& mapper,
    SourcePosition source,
    double source_radius,
    double source_step,
    SourcePosition current,
    SourcePosition current_source_offset,
    const Local7Frame& current_frame,
    double signed_step,
    Local7SpinePoint* output,
    Local7TimingStats* timing,
    int* fail_reason)
{
    double step = signed_step;
    const double min_abs_step = kLocal7SpineMinStepCells * source_step;
    int last_reason = 25;
    for (int attempt = 0; attempt < 12; ++attempt) {
        const SourcePosition candidate {
            current.x + step * current_frame.e_lx,
            current.y + step * current_frame.e_ly,
        };
        SourcePosition target_offset {
            current_source_offset.x + current_frame.lambda_l * step * current_frame.e_lx,
            current_source_offset.y + current_frame.lambda_l * step * current_frame.e_ly,
        };
        if (target_offset.x * target_offset.x + target_offset.y * target_offset.y > source_radius * source_radius) {
            last_reason = 20;
            step *= 0.5;
            if (std::abs(step) < min_abs_step) {
                break;
            }
            continue;
        }

        SourcePosition corrected = candidate;
        Local7Frame candidate_frame;
        SourcePosition mapped {};
        bool frame_ok = false;
        double residual_norm = std::numeric_limits<double>::infinity();
        for (int newton = 0; newton < 5; ++newton) {
            mapped = map_binary_lens_real(mapper, corrected.x, corrected.y);
            if (timing != nullptr) {
                ++timing->exact_lens_evals;
                ++timing->spine_exact_lens_evals;
            }
            frame_ok =
                local7_make_frame(Complex(corrected.x, corrected.y), source, mapper, &candidate_frame) &&
                local7_spine_frame_safe(candidate_frame);
            if (!frame_ok) {
                last_reason = newton == 0 ? 21 : 22;
                break;
            }
            const Complex residual(
                mapped.x - (source.x + target_offset.x),
                mapped.y - (source.y + target_offset.y));
            residual_norm = std::abs(residual);
            if (residual_norm <= kLocal7SpineTargetTolCells * source_step) {
                break;
            }
            const Complex dz = local7_apply_inverse_jacobian(candidate_frame, residual);
            double damping = 1.0;
            const double dz_abs = std::abs(dz);
            const double max_dz = 4.0 * std::max(std::abs(step), source_step);
            if (dz_abs > max_dz && dz_abs > 0.0) {
                damping = max_dz / dz_abs;
            }
            corrected.x -= damping * dz.real();
            corrected.y -= damping * dz.imag();
        }
        if (frame_ok) {
            const double dot =
                current_frame.e_lx * candidate_frame.e_lx + current_frame.e_ly * candidate_frame.e_ly;
            if (residual_norm <= kLocal7SpineTargetTolCells * source_step &&
                std::abs(dot) >= std::cos(kLocal7SpineCurvatureMax)) {
                *output = {corrected, target_offset, candidate_frame, 0.0};
                return true;
            }
            last_reason = residual_norm > kLocal7SpineTargetTolCells * source_step ? 23 : 24;
        }
        step *= 0.5;
        if (std::abs(step) < min_abs_step) {
            break;
        }
    }
    if (fail_reason != nullptr) {
        *fail_reason = last_reason;
    }
    return false;
}

bool local7_build_spine_direction(
    const BinaryLensMapper& mapper,
    SourcePosition source,
    double source_radius,
    double source_step,
    const Local7SpinePoint& seed,
    double sign,
    std::vector<Local7SpinePoint>* points,
    Local7TimingStats* timing,
    int* fail_reason)
{
    SourcePosition current = seed.image;
    SourcePosition current_source_offset = seed.source_offset;
    Local7Frame frame = seed.frame;
    int guard = 0;
    while (++guard < kLocal7SpineMaxPoints) {
        const double step = sign * local7_spine_step(frame, source_step, source_radius);
        Local7SpinePoint next;
        if (!local7_spine_try_step(
                mapper,
                source,
                source_radius,
                source_step,
                current,
                current_source_offset,
                frame,
                step,
                &next,
                timing,
                fail_reason)) {
            return true;
        }
        points->push_back(next);
        if (timing != nullptr) {
            ++timing->spine_points;
        }
        current = next.image;
        current_source_offset = next.source_offset;
        frame = next.frame;
        if (static_cast<int>(points->size()) > kLocal7SpineMaxPoints) {
            return false;
        }
    }
    return false;
}

double local7_spine_integrate_normals(
    const BinaryLensMapper& mapper,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings,
    const FiniteSourceMagnifier* finite_magnifier,
    const std::vector<Local7SpinePoint>& spine,
    double source_step,
    Local7TimingStats* timing,
    int* fallback_reason)
{
    const double radius2 = source_radius * source_radius;
    const double inv_radius2 = 1.0 / radius2;
    double area = 0.0;
    long long normal_samples = 0;
    for (std::size_t i = 0; i < spine.size(); ++i) {
        const auto& point = spine[i];
        const double tangent_weight = 2.0 * point.half_weight;
        if (!std::isfinite(tangent_weight) || tangent_weight <= 0.0) {
            *fallback_reason = 10;
            return std::nan("");
        }
        const double normal_step = std::min(
            std::max(source_step / std::max(std::abs(point.frame.lambda_s), kLocal7LambdaMin),
                kLocal7SpineMinStepCells * source_step),
            kLocal7SpineMaxStepCells * source_step) * kLocal7SpineNormalSubstep;
        const double cell_area = tangent_weight * normal_step;
        if (!std::isfinite(cell_area) || cell_area <= 0.0) {
            *fallback_reason = 11;
            return std::nan("");
        }

        for (int direction = -1; direction <= 1; direction += 2) {
            int outside = 0;
            for (int n = direction == -1 ? -1 : 0; ; n += direction) {
                const double offset = static_cast<double>(n) * normal_step;
                const SourcePosition image {
                    point.image.x + offset * point.frame.e_sx,
                    point.image.y + offset * point.frame.e_sy,
                };
                const SourcePosition mapped = map_binary_lens_real(mapper, image.x, image.y);
                if (timing != nullptr) {
                    ++timing->exact_lens_evals;
                    ++timing->spine_exact_lens_evals;
                }
                ++normal_samples;
                if (normal_samples > kLocal7SpineMaxNormalSamples) {
                    *fallback_reason = 12;
                    return std::nan("");
                }
                const double dx = mapped.x - source.x;
                const double dy = mapped.y - source.y;
                const double q = (dx * dx + dy * dy) * inv_radius2;
                if (q <= 1.0) {
                    outside = 0;
                    area += legacy_limb_brightness(q, settings, finite_magnifier) * cell_area;
                    if (timing != nullptr) {
                        ++timing->exact_samples;
                    }
                } else {
                    ++outside;
                    if (outside >= kLocal7SpineOutsideStop) {
                        break;
                    }
                }
                if (std::abs(offset) > 4.0 * source_radius / std::max(std::abs(point.frame.lambda_s), kLocal7LambdaMin)) {
                    *fallback_reason = 13;
                    return std::nan("");
                }
            }
        }
    }
    if (timing != nullptr) {
        timing->spine_normal_samples += normal_samples;
    }
    return area;
}

double local7_spine_area_binary(
    const BinaryLensMapper& mapper,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings,
    const FiniteSourceMagnifier* finite_magnifier,
    const Local7Frame& seed_frame,
    Local7TimingStats* timing,
    int* fallback_reason)
{
    *fallback_reason = 0;
    if (!local7_spine_frame_safe(seed_frame) || !local7_is_spine_candidate(seed_frame)) {
        *fallback_reason = 1;
        return std::nan("");
    }

    const int bins = std::max(settings.source_bins, 1);
    const double source_step = source_radius / static_cast<double>(bins);
    std::vector<Local7SpinePoint> minus_points;
    std::vector<Local7SpinePoint> plus_points;
    minus_points.reserve(1024);
    plus_points.reserve(1024);
    const Local7SpinePoint seed {
        {seed_frame.za.real(), seed_frame.za.imag()},
        seed_frame.sa,
        seed_frame,
        0.0,
    };
    if (!local7_build_spine_direction(
            mapper, source, source_radius, source_step, seed, -1.0, &minus_points, timing, fallback_reason) ||
        !local7_build_spine_direction(
            mapper, source, source_radius, source_step, seed, 1.0, &plus_points, timing, fallback_reason)) {
        *fallback_reason = 2;
        return std::nan("");
    }

    std::vector<Local7SpinePoint> spine;
    spine.reserve(minus_points.size() + plus_points.size() + 1);
    for (auto it = minus_points.rbegin(); it != minus_points.rend(); ++it) {
        spine.push_back(*it);
    }
    spine.push_back(seed);
    for (const auto& point : plus_points) {
        spine.push_back(point);
    }
    if (spine.size() < 3) {
        if (*fallback_reason == 0) {
            *fallback_reason = 3;
        }
        return std::nan("");
    }
    if (timing != nullptr) {
        ++timing->spine_points;
    }

    for (std::size_t i = 0; i < spine.size(); ++i) {
        double left_cross = 0.0;
        double right_cross = 0.0;
        double left_source = 0.0;
        double right_source = 0.0;
        if (i > 0) {
            const double dx = spine[i].image.x - spine[i - 1].image.x;
            const double dy = spine[i].image.y - spine[i - 1].image.y;
            left_cross = std::abs(dx * spine[i].frame.e_sy - dy * spine[i].frame.e_sx);
            left_source = std::hypot(spine[i].source_offset.x - spine[i - 1].source_offset.x,
                spine[i].source_offset.y - spine[i - 1].source_offset.y);
        }
        if (i + 1 < spine.size()) {
            const double dx = spine[i + 1].image.x - spine[i].image.x;
            const double dy = spine[i + 1].image.y - spine[i].image.y;
            right_cross = std::abs(dx * spine[i].frame.e_sy - dy * spine[i].frame.e_sx);
            right_source = std::hypot(spine[i + 1].source_offset.x - spine[i].source_offset.x,
                spine[i + 1].source_offset.y - spine[i].source_offset.y);
        }
        if (i == 0) {
            left_cross = right_cross;
            left_source = right_source;
        } else if (i + 1 == spine.size()) {
            right_cross = left_cross;
            right_source = left_source;
        }
        spine[i].half_weight = 0.25 * (left_cross + right_cross);
        if (!std::isfinite(spine[i].half_weight) || spine[i].half_weight <= 0.0 ||
            left_source > kLocal7SpineMaxStepCells * source_step * 2.0 ||
            right_source > kLocal7SpineMaxStepCells * source_step * 2.0) {
            *fallback_reason = 4;
            return std::nan("");
        }
    }

    const double area = local7_spine_integrate_normals(
        mapper,
        source,
        source_radius,
        settings,
        finite_magnifier,
        spine,
        source_step,
        timing,
        fallback_reason);
    const double total_source = source_flux(source_radius, settings);
    if (!std::isfinite(area) || area <= 0.0 ||
        !std::isfinite(total_source) || total_source <= 0.0 ||
        area / total_source > kLocal7SpineMaxRelativeArea) {
        if (*fallback_reason == 0) {
            *fallback_reason = 5;
        }
        return std::nan("");
    }
    return area;
}

double local7_imagearea0_binary(
    const BinaryLensMapper& mapper,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings,
    const FiniteSourceMagnifier* finite_magnifier,
    const Local7Frame& frame,
    double step_l,
    double step_s,
    SourcePosition seed_uv,
    double dv,
    int& yi,
    LegacyImageAreaScratch& scratch,
    Local7TimingStats* timing,
    bool derivative_steps)
{
    double countx = 0.0;
    double countall = 0.0;
    double dz2 = 99999999.9;
    double du = 1.0;
    SourcePosition uv = seed_uv;
    double u0 = seed_uv.x;
    const double step_ux = step_l * frame.e_lx;
    const double step_uy = step_l * frame.e_ly;
    const double step_vx = step_s * frame.e_sx;
    const double step_vy = step_s * frame.e_sy;
    const auto image_from_uv = [&]() {
        return SourcePosition {
            frame.za.real() + uv.x * step_ux + uv.y * step_vx,
            frame.za.imag() + uv.x * step_uy + uv.y * step_vy,
        };
    };
    SourcePosition image = image_from_uv();
    const double source_radius2 = source_radius * source_radius;
    const double inv_source_radius2 = 1.0 / source_radius2;
    int guard = 0;
    const int max_steps = std::max(100000, settings.source_bins * settings.source_bins * 4000);

    while (++guard < max_steps) {
        const double dz2_last = dz2;
        dz2 = mapped_binary_lens_distance2(mapper, image.x, image.y, source);
        if (timing != nullptr) {
            ++timing->exact_lens_evals;
        }

        scratch.ensure(static_cast<std::size_t>(yi));
        if (dz2 <= source_radius2) {
            if (du == -1.0 && countx == 0.0) {
                scratch.xmax[static_cast<std::size_t>(yi)] = uv.x - du;
            }
            countx += legacy_limb_brightness(dz2 * inv_source_radius2, settings, finite_magnifier);
            if (timing != nullptr) {
                if (derivative_steps) {
                    ++timing->derivative_step_samples;
                } else {
                    ++timing->exact_samples;
                }
            }
        } else {
            if (du == 1.0) {
                if (dz2_last <= source_radius2) {
                    scratch.xmax[static_cast<std::size_t>(yi)] = uv.x;
                }
                du = -1.0;
                uv.x = u0;
                image = image_from_uv();
                scratch.xmin[static_cast<std::size_t>(yi)] = uv.x + du;
            } else {
                if (dz2_last <= source_radius2) {
                    scratch.xmin[static_cast<std::size_t>(yi)] = uv.x;
                }
                if (yi != 0 && countx == 0.0) {
                    scratch.ensure(static_cast<std::size_t>(yi - 1));
                    if (uv.x >= scratch.xmin[static_cast<std::size_t>(yi - 1)] - du) {
                        uv.x += du;
                        image.x += du * step_ux;
                        image.y += du * step_uy;
                        continue;
                    }
                }

                countall += countx;
                scratch.ax[static_cast<std::size_t>(yi)] = countx;
                scratch.y[static_cast<std::size_t>(yi)] = uv.y;
                scratch.dys[static_cast<std::size_t>(yi)] = dv;
                if (countx == 0.0) {
                    scratch.dys[static_cast<std::size_t>(yi)] = -dv;
                    break;
                }

                const int row_key = static_cast<int>(std::llround(uv.y)) + kLegacyIndexOffset;
                auto& row_indices = scratch.row_indices[row_key];
                for (const int index : row_indices) {
                    const auto existing = static_cast<std::size_t>(index);
                    if (scratch.xmin[static_cast<std::size_t>(yi)] + 1.0 < scratch.xmax[existing] &&
                        scratch.xmax[static_cast<std::size_t>(yi)] - 1.0 > scratch.xmin[existing]) {
                        return countall - countx;
                    }
                }
                row_indices.push_back(yi);

                ++yi;
                scratch.ensure(static_cast<std::size_t>(yi));
                du = 1.0;
                u0 = scratch.xmax[static_cast<std::size_t>(yi - 1)];
                uv.x = u0 - du;
                uv.y += dv;
                image = image_from_uv();
                countx = 0.0;
            }
        }
        uv.x += du;
        image.x += du * step_ux;
        image.y += du * step_uy;
    }

    return countall;
}

double inverse_ray_local_binary(
    const PointSourceMagnifier& point_magnifier,
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings,
    const FiniteSourceMagnifier* finite_magnifier)
{
    if (source_radius <= 0.0 || separation == 0.0 || mass_ratio <= 0.0) {
        return std::nan("");
    }
    if ((settings.limb_darkening_c != 0.0 || settings.limb_darkening_d != 0.0) &&
        finite_magnifier != nullptr) {
        finite_magnifier->ensure_limb_darkening_table();
    }
    Local7TimingStats timing;

    const auto mapper = make_binary_lens_mapper(separation, mass_ratio);
    std::vector<SourcePosition> seeds;
    std::size_t point_image_count = 0;
    {
        Local7ScopedTimer timer(&timing.seed_ms);
        seeds = legacy_augmented_image_seeds(point_magnifier, mapper, separation, mass_ratio, source, source_radius);
        point_image_count = point_magnifier.binary_images(separation, mass_ratio, source).size();
    }
    if (seeds.empty()) {
        return std::nan("");
    }
    timing.caustic_born_branches =
        static_cast<int>(std::max<std::ptrdiff_t>(0, static_cast<std::ptrdiff_t>(seeds.size()) -
                                                        static_cast<std::ptrdiff_t>(point_image_count)));

    const int bins = std::max(settings.source_bins, 1);
    const double source_step = source_radius / static_cast<double>(bins);
    double area = 0.0;
    std::vector<double> areaimage(seeds.size(), 0.0);
    std::vector<int> overlap(seeds.size(), 0);

    for (std::size_t image_index = 0; image_index < seeds.size(); ++image_index) {
        if (overlap[image_index] == 1) {
            continue;
        }

        const SourcePosition seed = seeds[image_index];
        Local7Frame frame;
        {
            Local7ScopedTimer timer(&timing.frame_ms);
            if (!local7_make_frame(Complex(seed.x, seed.y), source, mapper, &frame) ||
                std::abs(frame.lambda_l) < kLocal7LambdaMin ||
                std::abs(frame.lambda_s) < kLocal7LambdaMin ||
                !std::isfinite(frame.area_jac)) {
                ++timing.fallback_calls;
                return std::nan("");
            }
        }

        const bool use_derivative_steps = seeds.size() < 5 && local7_can_use_derivative_steps(frame, source_radius);
        if (use_derivative_steps) {
            ++timing.safe_branches;
        }

        const Local7SpineEligibility spine_eligibility =
            use_derivative_steps ?
                Local7SpineEligibility {} :
                local7_spine_eligibility(
                    seeds,
                    overlap,
                    image_index,
                    frame,
                    source,
                    source_step,
                    mapper,
                    timing.caustic_born_branches);
        if (!use_derivative_steps && !spine_eligibility.ok &&
            timing.spine_branches == 0 && timing.spine_fallback_reason == 0) {
            timing.spine_fallback_reason = spine_eligibility.reason;
        }
        if (spine_eligibility.ok) {
            int spine_fallback_reason = 0;
            double spine_area = std::nan("");
            double pair_spine_area = 0.0;
            {
                Local7ScopedTimer timer(&timing.spine_ms);
                spine_area = local7_spine_area_binary(
                    mapper,
                    source,
                    source_radius,
                    settings,
                    finite_magnifier,
                    frame,
                    &timing,
                    &spine_fallback_reason);
            }
            if (std::isfinite(spine_area) && spine_area > 0.0) {
                ++timing.spine_branches;
                timing.spine_fallback_reason = 0;
                area += spine_area;
                areaimage[image_index] = spine_area;
                overlap[spine_eligibility.pair_index] = 1;
                continue;
            }
            ++timing.spine_fallbacks;
            timing.spine_fallback_reason = spine_fallback_reason;
        }

        const double step_l = use_derivative_steps ? source_step / frame.lambda_l :
                                                   std::copysign(source_step, frame.lambda_l);
        const double step_s = use_derivative_steps ? source_step / frame.lambda_s : source_step;
        const double image_cell_area = std::abs(step_l * step_s);
        if (!std::isfinite(image_cell_area) || image_cell_area <= 0.0) {
            return std::nan("");
        }

        LegacyImageAreaScratch scratch;
        scratch.ensure(1);
        double area0 = 0.0;
        double areai = 0.0;
        double dv = 1.0;
        int yi = 0;

        const SourcePosition seed_uv {0.0, 0.0};
        scratch.xmin[0] = seed_uv.x;
        scratch.xmax[0] = seed_uv.x;
        {
            Local7ScopedTimer timer(use_derivative_steps ? &timing.safe_scan_ms : &timing.exact_scan_ms);
            areai = local7_imagearea0_binary(
                mapper, source, source_radius, settings, finite_magnifier, frame, step_l, step_s, seed_uv, dv, yi,
                scratch, &timing, use_derivative_steps);
        }

        dv = -1.0;
        scratch.ensure(static_cast<std::size_t>(yi));
        const SourcePosition lower_seed_uv {scratch.xmax[0], seed_uv.y + dv};
        scratch.xmin[static_cast<std::size_t>(yi)] = scratch.xmin[0];
        scratch.xmax[static_cast<std::size_t>(yi)] = scratch.xmax[0];
        scratch.y[static_cast<std::size_t>(yi)] = scratch.y[0];
        scratch.dys[static_cast<std::size_t>(yi)] = dv;
        ++yi;
        {
            Local7ScopedTimer timer(use_derivative_steps ? &timing.safe_scan_ms : &timing.exact_scan_ms);
            areai += local7_imagearea0_binary(
                mapper, source, source_radius, settings, finite_magnifier, frame, step_l, step_s, lower_seed_uv, dv, yi,
                scratch, &timing, use_derivative_steps);
        }

        int nyi = yi;
        double areabound = 0.0;
        {
        const auto check_start = Local7TimingStats::Clock::now();
        const double safe_scan_before = timing.safe_scan_ms;
        const double exact_scan_before = timing.exact_scan_ms;
        for (int row = 0; row < nyi; ++row) {
            scratch.ensure(static_cast<std::size_t>(row + 1));
            const double dumax =
                scratch.xmax[static_cast<std::size_t>(row + 1)] -
                scratch.xmax[static_cast<std::size_t>(row)];
            const double dumin =
                scratch.xmin[static_cast<std::size_t>(row + 1)] -
                scratch.xmin[static_cast<std::size_t>(row)];
            if (scratch.ax[static_cast<std::size_t>(row + 1)] > 0.0) {
                if (dumax > kLocal7BoundaryJump) {
                    const SourcePosition extra_seed_uv {
                        scratch.xmax[static_cast<std::size_t>(row + 1)],
                        scratch.y[static_cast<std::size_t>(row)]};
                    scratch.ensure(static_cast<std::size_t>(yi));
                    scratch.xmin[static_cast<std::size_t>(yi)] = scratch.xmax[static_cast<std::size_t>(row)];
                    scratch.xmax[static_cast<std::size_t>(yi)] = scratch.xmax[static_cast<std::size_t>(row + 1)];
                    dv = -scratch.dys[static_cast<std::size_t>(row)];
                    scratch.dys[static_cast<std::size_t>(yi)] = dv;
                    ++yi;
                    {
                        Local7ScopedTimer timer(use_derivative_steps ? &timing.safe_scan_ms : &timing.exact_scan_ms);
                        area0 = local7_imagearea0_binary(
                            mapper, source, source_radius, settings, finite_magnifier, frame, step_l, step_s,
                            extra_seed_uv, dv, yi, scratch, &timing, use_derivative_steps);
                    }
                    areai += area0;
                    areabound += area0;
                    if (area0 <= 0.0) {
                        --yi;
                    }
                }
                if (dumin > kLocal7BoundaryJump) {
                    const SourcePosition extra_seed_uv {
                        scratch.xmin[static_cast<std::size_t>(row + 1)] - 1.0,
                        scratch.y[static_cast<std::size_t>(row + 1)]};
                    scratch.ensure(static_cast<std::size_t>(yi));
                    scratch.xmin[static_cast<std::size_t>(yi)] = scratch.xmin[static_cast<std::size_t>(row)];
                    scratch.xmax[static_cast<std::size_t>(yi)] = scratch.xmin[static_cast<std::size_t>(row + 1)];
                    dv = scratch.dys[static_cast<std::size_t>(row)];
                    scratch.dys[static_cast<std::size_t>(yi)] = dv;
                    ++yi;
                    {
                        Local7ScopedTimer timer(use_derivative_steps ? &timing.safe_scan_ms : &timing.exact_scan_ms);
                        area0 = local7_imagearea0_binary(
                            mapper, source, source_radius, settings, finite_magnifier, frame, step_l, step_s,
                            extra_seed_uv, dv, yi, scratch, &timing, use_derivative_steps);
                    }
                    areai += area0;
                    areabound += area0;
                    if (area0 <= 0.0) {
                        --yi;
                    }
                }
                if (dumin < -kLocal7BoundaryJump) {
                    const SourcePosition extra_seed_uv {
                        scratch.xmin[static_cast<std::size_t>(row)] - 1.0,
                        scratch.y[static_cast<std::size_t>(row)]};
                    scratch.ensure(static_cast<std::size_t>(yi));
                    scratch.xmin[static_cast<std::size_t>(yi)] = scratch.xmin[static_cast<std::size_t>(row + 1)];
                    scratch.xmax[static_cast<std::size_t>(yi)] = scratch.xmin[static_cast<std::size_t>(row)];
                    dv = -scratch.dys[static_cast<std::size_t>(row)];
                    scratch.dys[static_cast<std::size_t>(yi)] = dv;
                    ++yi;
                    {
                        Local7ScopedTimer timer(use_derivative_steps ? &timing.safe_scan_ms : &timing.exact_scan_ms);
                        area0 = local7_imagearea0_binary(
                            mapper, source, source_radius, settings, finite_magnifier, frame, step_l, step_s,
                            extra_seed_uv, dv, yi, scratch, &timing, use_derivative_steps);
                    }
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
        const double nested_scan_ms =
            (timing.safe_scan_ms - safe_scan_before) + (timing.exact_scan_ms - exact_scan_before);
        timing.check_ms += std::max(0.0, Local7TimingStats::elapsed_ms(check_start) - nested_scan_ms);
        }

        area += areai * image_cell_area;
        areaimage[image_index] = areai * image_cell_area;

        {
        Local7ScopedTimer check_timer(&timing.check_ms);
        const double margin = 0.5 * 1.01;
        for (std::size_t other = 0; other < seeds.size(); ++other) {
            if (other == image_index) {
                continue;
            }
            const SourcePosition other_uv = local7_seed_to_uv(frame, step_l, step_s, seeds[other]);
            for (int row = 0; row < nyi; ++row) {
                const auto row_index = static_cast<std::size_t>(row);
                if (scratch.ax[row_index] <= 0.0) {
                    continue;
                }
                if (other_uv.y >= scratch.y[row_index] - margin &&
                    other_uv.y <= scratch.y[row_index] + margin &&
                    other_uv.x >= scratch.xmin[row_index] - margin &&
                    other_uv.x <= scratch.xmax[row_index] + margin) {
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
    }

    const double total_source_flux = source_flux(source_radius, settings);
    if (!std::isfinite(total_source_flux) || total_source_flux <= 0.0) {
        return std::nan("");
    }
    const double magnification = area / total_source_flux;
    if (settings.verbosity >= 3) {
        std::fprintf(stderr,
            "#LOCAL7TIMING total_ms=%.6g seed_ms=%.6g frame_ms=%.6g safe_scan_ms=%.6g "
            "exact_scan_ms=%.6g spine_ms=%.6g check_ms=%.6g exact_lens_evals=%lld "
            "derivative_samples=%lld exact_samples=%lld spine_points=%lld spine_normal_samples=%lld "
            "spine_exact_lens_evals=%lld caustic_born_branches=%d safe_branches=%d "
            "spine_branches=%d spine_fallbacks=%d spine_fallback_reason=%d fallback_calls=%d "
            "nseed=%zu bins=%d rho=%.9g count=%.12g\n",
            Local7TimingStats::elapsed_ms(timing.total_start),
            timing.seed_ms,
            timing.frame_ms,
            timing.safe_scan_ms,
            timing.exact_scan_ms,
            timing.spine_ms,
            timing.check_ms,
            timing.exact_lens_evals,
            timing.derivative_step_samples,
            timing.exact_samples,
            timing.spine_points,
            timing.spine_normal_samples,
            timing.spine_exact_lens_evals,
            timing.caustic_born_branches,
            timing.safe_branches,
            timing.spine_branches,
            timing.spine_fallbacks,
            timing.spine_fallback_reason,
            timing.fallback_calls,
            seeds.size(),
            bins,
            source_radius,
            magnification);
    }
    return std::isfinite(magnification) ? magnification : std::nan("");
}

double inverse_ray_binary(
    const PointSourceMagnifier& point_magnifier,
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings,
    const FiniteSourceMagnifier* finite_magnifier,
    FiniteSourceMethod method,
    int bins)
{
    if (method == FiniteSourceMethod::inverse_ray_polar) {
        return inverse_ray_polar_binary(
            point_magnifier, separation, mass_ratio, source, source_radius, settings, finite_magnifier, bins);
    }

    return inverse_ray_cartesian_binary(
        point_magnifier, separation, mass_ratio, source, source_radius, settings, finite_magnifier, bins);
}

FiniteSourceResult refined_inverse_ray_binary(
    const PointSourceMagnifier& point_magnifier,
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings,
    const FiniteSourceMagnifier* finite_magnifier,
    FiniteSourceDecision decision)
{
    int bins = std::max(settings.source_bins, 1);
    double coarse = inverse_ray_binary(
        point_magnifier, separation, mass_ratio, source, source_radius, settings, finite_magnifier, decision.method, bins);
    if (!std::isfinite(coarse)) {
        return {coarse, 0, decision, std::nan(""), 0, false};
    }

    double error_estimate = std::numeric_limits<double>::infinity();
    double previous_delta = std::numeric_limits<double>::quiet_NaN();
    for (int level = 1; level <= kMaxRefinementLevels; ++level) {
        bins *= 2;
        const double fine = inverse_ray_binary(
            point_magnifier, separation, mass_ratio, source, source_radius, settings, finite_magnifier, decision.method, bins);
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
    const FiniteSourceMagnifier* finite_magnifier,
    FiniteSourceDecision decision)
{
    const int bins = std::max(settings.source_bins, 1);
    double magnification = std::nan("");
    if (decision.method == FiniteSourceMethod::inverse_ray_polar) {
        if (settings.legacy_mode) {
            const auto mapper = make_binary_lens_mapper(separation, mass_ratio);
            const auto seeds = legacy_augmented_image_seeds(
                point_magnifier, mapper, separation, mass_ratio, source, source_radius);
            magnification = inverse_ray_polar_boundary_binary(
                point_magnifier, separation, mass_ratio, source, source_radius, settings, finite_magnifier, nullptr, &seeds);
        } else {
            magnification = inverse_ray_polar_boundary_binary(
                point_magnifier, separation, mass_ratio, source, source_radius, settings, finite_magnifier);
        }
    } else if (decision.method == FiniteSourceMethod::inverse_ray_cartesian && settings.legacy_mode &&
               settings.legacy_finite_mode == 4) {
        magnification = legacy_imagearea4_binary(
            point_magnifier, separation, mass_ratio, source, source_radius, settings, finite_magnifier);
    } else if (decision.method == FiniteSourceMethod::inverse_ray_local && settings.legacy_mode &&
               settings.legacy_finite_mode == 7) {
        magnification = inverse_ray_local_binary(
            point_magnifier, separation, mass_ratio, source, source_radius, settings, finite_magnifier);
    } else {
        magnification = inverse_ray_binary(
            point_magnifier, separation, mass_ratio, source, source_radius, settings, finite_magnifier, decision.method, bins);
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
        "legacy smode=6 selected cached polar inverse-ray",
    };
    const auto mapper = make_binary_lens_mapper(separation, mass_ratio);
    const auto seeds = legacy_augmented_image_seeds(
        point_magnifier, mapper, separation, mass_ratio, source, source_radius);
    const auto point_images = point_magnifier.binary_images(separation, mass_ratio, source);
    const double sampled_caustic_distance = legacy_binary_sampled_caustic_distance(
        separation, mass_ratio, source, source_radius);
    const double polar_fallback_distance =
        std::max(settings_.legacy_hex, 1.0) * source_radius;
    if (seeds.size() > point_images.size() ||
        (std::isfinite(sampled_caustic_distance) && sampled_caustic_distance < polar_fallback_distance) ||
        (std::isfinite(caustic_distance) && caustic_distance < polar_fallback_distance)) {
        const double magnification = legacy_imagearea4_binary(
            point_magnifier, separation, mass_ratio, source, source_radius, settings_, this, &seeds);
        decision.method = FiniteSourceMethod::inverse_ray_cartesian;
        decision.reason = "legacy smode=6 used cartesian fallback for caustic-crossing seed set";
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
    case FiniteSourceMethod::inverse_ray_local:
        return "inverse_ray_local";
    default:
        return "unknown";
    }
}

FiniteSourceResult FiniteSourceMagnifier::legacy_binary_finite_mag_direct(
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    int legacy_finite_mode) const
{
    PointSourceMagnifier point_magnifier;
    FiniteSourceSettings direct_settings = settings_;
    direct_settings.legacy_mode = true;
    direct_settings.legacy_finite_mode = legacy_finite_mode;

    FiniteSourceDecision decision;
    switch (legacy_finite_mode) {
    case 5:
    case 6:
        decision = {
            FiniteSourceMethod::inverse_ray_polar,
            estimate_polar_cost(direct_settings),
            "direct legacy finite-source polar inverse-ray",
        };
        break;
    case 7:
        decision = {
            FiniteSourceMethod::inverse_ray_local,
            estimate_cartesian_cost(direct_settings),
            "direct local-coordinate inverse-ray",
        };
        break;
    case 3:
    case 4:
    default:
        decision = {
            FiniteSourceMethod::inverse_ray_cartesian,
            estimate_cartesian_cost(direct_settings),
            "direct legacy finite-source cartesian inverse-ray",
        };
        break;
    }

    return fixed_inverse_ray_binary(
        point_magnifier, separation, mass_ratio, source, source_radius, direct_settings, this, decision);
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
        if (settings_.legacy_finite_mode == 6) {
            return cache_and_return(legacy_polar_memory_binary_mag(
                separation, mass_ratio, source, source_radius, caustic_distance));
        }
        if (settings_.legacy_finite_mode == 5) {
            decision.method = FiniteSourceMethod::inverse_ray_polar;
            decision.estimated_evaluations = estimate_polar_cost(settings_);
            decision.reason = "legacy smode selected polar inverse-ray";
        } else if (settings_.legacy_finite_mode == 7) {
            decision.method = FiniteSourceMethod::inverse_ray_local;
            decision.estimated_evaluations = estimate_cartesian_cost(settings_);
            decision.reason = "legacy smode=7 selected local-coordinate inverse-ray";
        } else if (settings_.legacy_finite_mode == 3 || settings_.legacy_finite_mode == 4) {
            decision.method = FiniteSourceMethod::inverse_ray_cartesian;
            decision.estimated_evaluations = estimate_cartesian_cost(settings_);
            decision.reason = "legacy smode selected cartesian inverse-ray";
        } else {
            decision.reason = "legacy smode fell back to automatic finite-source strategy";
        }
        return cache_and_return(fixed_inverse_ray_binary(
            point_magnifier, separation, mass_ratio, source, source_radius, settings_, this, decision));
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
        point_magnifier, separation, mass_ratio, source, source_radius, settings_, this, decision));
}

} // namespace lcbinint::magnification
