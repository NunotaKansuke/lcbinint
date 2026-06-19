#include "lcbinint/model/orbital_motion.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace lcbinint::model {
namespace {

constexpr double kBranchEps = 1.0e-8;

double safe_sqrt(double value)
{
    return std::sqrt(std::max(0.0, value));
}

double dot(std::array<double, 3> lhs, std::array<double, 3> rhs)
{
    return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2];
}

std::array<double, 3> scale(std::array<double, 3> value, double factor)
{
    return {value[0] * factor, value[1] * factor, value[2] * factor};
}

std::array<double, 3> cross(std::array<double, 3> lhs, std::array<double, 3> rhs)
{
    return {
        lhs[1] * rhs[2] - lhs[2] * rhs[1],
        lhs[2] * rhs[0] - lhs[0] * rhs[2],
        lhs[0] * rhs[1] - lhs[1] * rhs[0],
    };
}

std::array<double, 3> normalize(std::array<double, 3> value)
{
    const double norm = safe_sqrt(dot(value, value));
    if (norm == 0.0 || !std::isfinite(norm)) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        return {nan, nan, nan};
    }
    return scale(value, 1.0 / norm);
}

double solve_kepler_newton(double mean_anomaly, double eccentricity)
{
    double eccentric_anomaly = mean_anomaly + eccentricity * std::sin(mean_anomaly);
    for (int i = 0; i < 10; ++i) {
        const double f = eccentric_anomaly - eccentricity * std::sin(eccentric_anomaly) -
                         mean_anomaly;
        const double fp = 1.0 - eccentricity * std::cos(eccentric_anomaly);
        eccentric_anomaly -= f / fp;
    }
    return eccentric_anomaly;
}

double orbital_reference_time(const LensParameters& params)
{
    return params.tfix != 0.0 ? params.tfix : params.t0;
}

} // namespace

OrbitalState circular_orbital_motion_3d(
    double time,
    double separation,
    double angle,
    double w1,
    double w2,
    double w3,
    double reference_time)
{
    const double w13_sq = w1 * w1 + w3 * w3;
    const double w13 = std::sqrt(w13_sq);
    const double w123 = std::sqrt(w13_sq + w2 * w2);

    double w_orb = w2;
    double inc = 0.0;
    double phi0 = 0.0;
    if (w13 > kBranchEps) {
        const double w3_eff = w3 > kBranchEps ? w3 : kBranchEps;
        w_orb = w3_eff * w123 / w13;
        const double cos_inc_arg = std::clamp(w2 * w3_eff / (w13 * w123), -1.0, 1.0);
        inc = std::acos(cos_inc_arg);
        phi0 = std::atan2(-w1 * w123, w3_eff * w13);
    }

    const double c_phi0 = std::cos(phi0);
    const double s_phi0 = std::sin(phi0);
    const double c_inc = std::cos(inc);
    const double s_inc = std::sin(inc);
    const double den0 = std::sqrt(c_phi0 * c_phi0 + c_inc * c_inc * s_phi0 * s_phi0);
    const double s_true = separation / den0;

    const double c_angle0 = std::cos(angle);
    const double s_angle0 = std::sin(angle);
    const double c_om = (c_phi0 * c_angle0 + c_inc * s_angle0 * s_phi0) / den0;
    const double s_om = (c_phi0 * s_angle0 - c_inc * c_angle0 * s_phi0) / den0;

    const double phi = w_orb * (time - reference_time) + phi0;
    const double c_phi = std::cos(phi);
    const double s_phi = std::sin(phi);
    const double den = std::sqrt(c_phi * c_phi + c_inc * c_inc * s_phi * s_phi);

    OrbitalState state;
    state.separation = s_true * den;
    const double sin_angle = (c_phi * s_om + c_inc * s_phi * c_om) / den;
    const double cos_angle = (c_phi * c_om - c_inc * s_phi * s_om) / den;
    state.angle = std::atan2(sin_angle, cos_angle);
    state.line_of_sight_separation = s_true * s_inc * s_phi;
    return state;
}

OrbitalState kepler_orbital_motion_3d(
    double time,
    double separation,
    double angle,
    double w1,
    double w2,
    double w3,
    double szs,
    double ar,
    double reference_time)
{
    ar += kBranchEps;
    const double smix = 1.0 + szs * szs;
    const double sqsmix = std::sqrt(smix);
    const double w11 = w1 * w1;
    const double w22 = w2 * w2;
    const double w33 = w3 * w3;
    const double w12 = w11 + w22;
    const double wt2 = w12 + w33;
    const double arm1 = ar - 1.0;
    const double arm2 = 2.0 * ar - 1.0;
    const double mean_motion = std::sqrt(wt2 / arm2 / smix) / ar;

    const auto z_axis = normalize({-szs * w2, szs * w1 - w3, w2});
    auto x_axis = std::array<double, 3>{
        -ar * w11 + arm1 * w22 - arm2 * szs * w1 * w3 + arm1 * w33,
        -arm2 * w2 * (w1 + szs * w3),
        arm1 * szs * w12 - arm2 * w1 * w3 - ar * szs * w33,
    };
    const double x_norm = safe_sqrt(dot(x_axis, x_axis));
    x_axis = scale(x_axis, 1.0 / x_norm);
    const double eccentricity = x_norm / (ar * sqsmix * wt2);
    const auto y_axis = cross(z_axis, x_axis);

    const double conu = (x_axis[0] + x_axis[2] * szs) / sqsmix;
    const double cos_e0 = std::clamp((conu + eccentricity) /
                                         (1.0 + eccentricity * conu),
        -1.0,
        1.0);
    const double sign = (y_axis[0] + y_axis[2] * szs) > 0.0 ? 1.0 : -1.0;
    const double e0 = std::acos(cos_e0) * sign;
    const double sin_e0 = safe_sqrt(1.0 - cos_e0 * cos_e0) * sign;
    const double t_peri = reference_time - (e0 - eccentricity * sin_e0) / mean_motion;
    const double semi_major_axis = ar * separation * sqsmix;

    const double mean_anomaly = mean_motion * (time - t_peri);
    const double eccentric_anomaly = solve_kepler_newton(mean_anomaly, eccentricity);
    const double cos_e = std::cos(eccentric_anomaly);
    const double sin_e = std::sin(eccentric_anomaly);
    const double r0 = semi_major_axis * (cos_e - eccentricity);
    const double r1 = semi_major_axis * safe_sqrt(1.0 - eccentricity * eccentricity) * sin_e;

    const double x0 = r0 * x_axis[0] + r1 * y_axis[0];
    const double x1 = r0 * x_axis[1] + r1 * y_axis[1];
    const double x2 = r0 * x_axis[2] + r1 * y_axis[2];

    OrbitalState state;
    state.separation = std::sqrt(x0 * x0 + x1 * x1);
    state.angle = angle + std::atan2(x1, x0);
    state.line_of_sight_separation = x2;
    return state;
}

OrbitalState orbital_state(const LensParameters& params, double time)
{
    const double reference_time = orbital_reference_time(params);
    switch (params.orbital_motion_mode) {
    case LCBI_ORBIT_CIRCULAR:
        return circular_orbital_motion_3d(
            time, params.sep, params.theta, params.g1, params.g2, params.g3,
            reference_time);
    case LCBI_ORBIT_KEPLER:
        return kepler_orbital_motion_3d(
            time, params.sep, params.theta, params.g1, params.g2, params.g3,
            params.lom_szs, params.lom_ar, reference_time);
    case LCBI_ORBIT_STATIC:
    default:
        return {params.sep, params.theta, 0.0};
    }
}

SourcePosition rotate_source_to_orbital_frame(SourcePosition source, double angle_delta)
{
    const double c = std::cos(angle_delta);
    const double s = std::sin(angle_delta);
    return {source.x * c + source.y * s, source.y * c - source.x * s};
}

} // namespace lcbinint::model
