#include "lcbinint/model/trajectory.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace lcbinint::model {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDegreeToRadian = kPi / 180.0;
constexpr double kReducedJulianDateOrigin = 2450000.0;
constexpr double kAuLightTravelDays = 0.005775518331436995;

struct EarthState {
    double time = 0.0;
    std::array<double, 3> position = {};
    std::array<double, 3> velocity = {};
};

struct SkyVector {
    double north = 0.0;
    double east = 0.0;
};

struct ProjectedEarthState {
    SkyVector position;
    SkyVector velocity;
};

std::array<double, 3> add(std::array<double, 3> lhs, std::array<double, 3> rhs)
{
    return {lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]};
}

std::array<double, 3> subtract(std::array<double, 3> lhs, std::array<double, 3> rhs)
{
    return {lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]};
}

std::array<double, 3> scale(std::array<double, 3> value, double factor)
{
    return {value[0] * factor, value[1] * factor, value[2] * factor};
}

double dot(std::array<double, 3> lhs, std::array<double, 3> rhs)
{
    return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2];
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
    const double norm = std::sqrt(dot(value, value));
    if (norm == 0.0 || !std::isfinite(norm)) {
        return {std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN()};
    }
    return scale(value, 1.0 / norm);
}

std::array<double, 3> sky_event_unit(double ra_degrees, double dec_degrees)
{
    const double ra = ra_degrees * kDegreeToRadian;
    const double dec = dec_degrees * kDegreeToRadian;
    const double cos_dec = std::cos(dec);
    return {std::cos(ra) * cos_dec, std::sin(ra) * cos_dec, std::sin(dec)};
}

std::array<double, 3> sky_east(double ra_degrees, double dec_degrees)
{
    const std::array<double, 3> earth_north = {0.0, 0.0, 1.0};
    return normalize(cross(earth_north, sky_event_unit(ra_degrees, dec_degrees)));
}

std::array<double, 3> sky_north(double ra_degrees, double dec_degrees)
{
    const auto event = sky_event_unit(ra_degrees, dec_degrees);
    return cross(event, sky_east(ra_degrees, dec_degrees));
}

std::string trim(std::string value)
{
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return "";
    }
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

std::vector<double> parse_horizons_numeric_fields(const std::string& line)
{
    std::vector<double> fields;
    std::stringstream stream(line);
    std::string token;
    int column = 0;
    while (std::getline(stream, token, ',')) {
        token = trim(token);
        if (token.empty()) {
            continue;
        }
        if (column == 1) {
            ++column;
            continue;
        }
        try {
            fields.push_back(std::stod(token));
        } catch (const std::invalid_argument&) {
        }
        ++column;
    }
    return fields;
}

const char* default_ephemeris_path()
{
    const char* override_path = std::getenv("LCBININT_EARTH_EPHEMERIS");
    if (override_path != nullptr && override_path[0] != '\0') {
        return override_path;
    }
#ifdef LCBININT_EARTH_EPHEMERIS_PATH
    return LCBININT_EARTH_EPHEMERIS_PATH;
#else
    return "";
#endif
}

std::vector<EarthState> load_earth_ephemeris()
{
    std::ifstream input(default_ephemeris_path());
    if (!input) {
        return {};
    }

    std::vector<EarthState> states;
    bool in_block = false;
    std::string line;
    while (std::getline(input, line)) {
        const std::string stripped = trim(line);
        if (!in_block) {
            if (stripped.rfind("$$SOE", 0) == 0) {
                in_block = true;
            }
            continue;
        }
        if (stripped.rfind("$$EOE", 0) == 0) {
            break;
        }
        const auto fields = parse_horizons_numeric_fields(stripped);
        if (fields.size() < 7) {
            continue;
        }
        states.push_back({fields[0],
            {fields[1], fields[2], fields[3]},
            {fields[4], fields[5], fields[6]}});
    }
    return states;
}

const std::vector<EarthState>& earth_ephemeris()
{
    static const std::vector<EarthState> states = load_earth_ephemeris();
    return states;
}

EarthState interpolate_earth_state(double time)
{
    const auto& states = earth_ephemeris();
    if (states.size() < 2 || time < states.front().time || time > states.back().time) {
        return {time,
            {std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN()},
            {std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN()}};
    }

    const auto upper = std::upper_bound(
        states.begin(), states.end(), time, [](double value, const EarthState& state) {
            return value < state.time;
        });
    const auto hi = upper == states.begin() ? states.begin() + 1 : upper;
    const auto lo = hi - 1;
    const double dt = hi->time - lo->time;
    const double weight = dt != 0.0 ? (time - lo->time) / dt : 0.0;
    return {time,
        add(lo->position, scale(subtract(hi->position, lo->position), weight)),
        add(lo->velocity, scale(subtract(hi->velocity, lo->velocity), weight))};
}

double parallax_time_offset(double reference_time)
{
    return reference_time < kReducedJulianDateOrigin ? kReducedJulianDateOrigin : 0.0;
}

bool has_annual_parallax(const LensParameters& params)
{
    return params.piEN != 0.0 || params.piEE != 0.0;
}

class EarthOrbitalParallaxProjector {
public:
    EarthOrbitalParallaxProjector(double ra_degrees, double dec_degrees, double reference_time)
        : reference_time_(reference_time)
        , time_offset_(parallax_time_offset(reference_time_))
        , ephemeris_reference_time_(reference_time_ + time_offset_)
        , sky_north_(sky_north(ra_degrees, dec_degrees))
        , sky_east_(sky_east(ra_degrees, dec_degrees))
        , event_unit_(sky_event_unit(ra_degrees, dec_degrees))
    {
        const double reference_emit_time = light_time_corrected_time(ephemeris_reference_time_);
        reference_state_ = project(interpolate_earth_state(reference_emit_time));
    }

    SkyVector offset(double time, double pi_north, double pi_east) const
    {
        const auto projected = displacement(time);
        return {
            pi_north * projected.north + pi_east * projected.east,
            -pi_east * projected.north + pi_north * projected.east,
        };
    }

private:
    SkyVector displacement(double time) const
    {
        const double ephemeris_time = time + time_offset_;
        const double emit_time = light_time_corrected_time(ephemeris_time);
        const auto state = project(interpolate_earth_state(emit_time));
        const double dt = ephemeris_time - ephemeris_reference_time_;
        return {
            state.position.north - reference_state_.position.north -
                reference_state_.velocity.north * dt,
            state.position.east - reference_state_.position.east -
                reference_state_.velocity.east * dt,
        };
    }

    double light_time_corrected_time(double observation_time) const
    {
        double emit_time = observation_time;
        for (int i = 0; i < 5; ++i) {
            const auto state = interpolate_earth_state(emit_time);
            const double light_time = dot(state.position, event_unit_) * kAuLightTravelDays;
            emit_time = observation_time - light_time;
        }
        return emit_time;
    }

    ProjectedEarthState project(const EarthState& state) const
    {
        return {
            {-dot(state.position, sky_north_), -dot(state.position, sky_east_)},
            {-dot(state.velocity, sky_north_), -dot(state.velocity, sky_east_)},
        };
    }

    double reference_time_ = 0.0;
    double time_offset_ = 0.0;
    double ephemeris_reference_time_ = 0.0;
    std::array<double, 3> sky_north_ = {};
    std::array<double, 3> sky_east_ = {};
    std::array<double, 3> event_unit_ = {};
    ProjectedEarthState reference_state_ = {};
};

void apply_xallarap_angular_velocity(
    const LensParameters& params, double time, double& tau, double& beta)
{
    if (!params.has_xallarap_angular_velocity()) {
        return;
    }
    const double phit = params.omega_xa * (time - params.t0);
    const double phi = params.phi_xa;
    const double s_inc = std::sin(params.inc_xa);
    const double disp0 = s_inc * (-std::cos(phi) + std::cos(phi + phit) + phit * std::sin(phi));
    const double disp1 = -phit * std::cos(phi) - std::sin(phi) + std::sin(phi + phit);
    tau  += params.xi_1 * disp0 + params.xi_2 * disp1;
    beta += params.xi_2 * disp0 - params.xi_1 * disp1;
}

double solve_kepler(double mean_anomaly, double eccentricity)
{
    double E = mean_anomaly + eccentricity * std::sin(mean_anomaly);
    for (int i = 0; i < 10; ++i) {
        const double f = E - eccentricity * std::sin(E) - mean_anomaly;
        const double fp = 1.0 - eccentricity * std::cos(E);
        const double delta = f / fp;
        E -= delta;
        if (std::abs(delta) < 1.0e-13) {
            break;
        }
    }
    return E;
}

// Returns the orbital position (x, y) in the orbital plane at time t,
// normalized to semi-major axis = 1. The mean anomaly at t=t0 is peri_xa.
std::array<double, 2> keplerian_position(
    double time, double t0, double period, double ecc, double peri)
{
    const double mean_anom = 2.0 * kPi / period * (time - t0) + peri;
    const double E = solve_kepler(mean_anom, ecc);
    const double cos_E = std::cos(E);
    const double sin_E = std::sin(E);
    const double r = 1.0 - ecc * cos_E;
    const double sqrt_1me2 = std::sqrt(std::max(0.0, 1.0 - ecc * ecc));
    const double cos_f = (cos_E - ecc) / r;
    const double sin_f = sqrt_1me2 * sin_E / r;
    return {r * cos_f, r * sin_f};
}

void apply_xallarap_orbital_elements(
    const LensParameters& params, double time, double& tau, double& beta)
{
    if (!params.has_xallarap_orbital_elements()) {
        return;
    }
    if (params.period_xa <= 0.0) {
        return;
    }
    const auto pos_t = keplerian_position(
        time, params.t0, params.period_xa, params.ecc_xa, params.peri_xa);
    const auto pos_0 = keplerian_position(
        params.t0, params.t0, params.period_xa, params.ecc_xa, params.peri_xa);
    const double dt_small = params.period_xa * 1.0e-7;
    const auto pos_dt = keplerian_position(
        params.t0 + dt_small, params.t0, params.period_xa, params.ecc_xa, params.peri_xa);
    const double vel_x0 = (pos_dt[0] - pos_0[0]) / dt_small;
    const double vel_y0 = (pos_dt[1] - pos_0[1]) / dt_small;
    const double elapsed = time - params.t0;
    const double dev_x = pos_t[0] - pos_0[0] - vel_x0 * elapsed;
    const double dev_y = pos_t[1] - pos_0[1] - vel_y0 * elapsed;
    const double s_inc = std::sin(params.inc_xa);
    const double disp0 = s_inc * dev_x;
    const double disp1 = dev_y;
    tau  += params.piEN_xa * disp0 + params.piEE_xa * disp1;
    beta += params.piEE_xa * disp0 - params.piEN_xa * disp1;
}

void apply_annual_parallax(const LensParameters& params, double time, double& tau, double& beta)
{
    if (!has_annual_parallax(params)) {
        return;
    }

    const double reference_time = params.tfix != 0.0 ? params.tfix : params.t0;
    struct ProjectorCache {
        bool valid = false;
        double ra = 0.0;
        double dec = 0.0;
        double reference_time = 0.0;
        std::unique_ptr<EarthOrbitalParallaxProjector> projector;
    };
    thread_local ProjectorCache cache;
    if (!cache.valid ||
        cache.ra != params.ra ||
        cache.dec != params.dec ||
        cache.reference_time != reference_time) {
        cache.ra = params.ra;
        cache.dec = params.dec;
        cache.reference_time = reference_time;
        cache.projector =
            std::make_unique<EarthOrbitalParallaxProjector>(params.ra, params.dec, reference_time);
        cache.valid = true;
    }
    const auto offset = cache.projector->offset(time, params.piEN, params.piEE);
    tau += offset.north;
    beta += offset.east;
}

} // namespace

SourcePosition Trajectory::source_position(
    double time, bool vbm_mode, lcbi_xallarap_param_type xallarap_type) const
{
    double tn = (time - params_.t0) / params_.tE;
    double beta = params_.umin;
    apply_annual_parallax(params_, time, tn, beta);
    if (xallarap_type == LCBI_XALLARAP_ANGULAR_VELOCITY) {
        apply_xallarap_angular_velocity(params_, time, tn, beta);
    } else if (xallarap_type == LCBI_XALLARAP_ORBITAL_ELEMENTS) {
        apply_xallarap_orbital_elements(params_, time, tn, beta);
    }

    const double costheta = std::cos(params_.theta);
    const double sintheta = std::sin(params_.theta);

    SourcePosition source;
    if (vbm_mode) {
        // VBM BinaryLightCurve convention: u1 = tau*cos(alpha) - u0*sin(alpha),
        // u2 = tau*sin(alpha) + u0*cos(alpha), where alpha = theta, u0 = umin.
        source.x = tn * costheta - beta * sintheta;
        source.y = tn * sintheta + beta * costheta;
    } else {
        source.x = beta * sintheta + tn * costheta;
        source.y = beta * costheta - tn * sintheta;
    }
    return source;
}

} // namespace lcbinint::model
