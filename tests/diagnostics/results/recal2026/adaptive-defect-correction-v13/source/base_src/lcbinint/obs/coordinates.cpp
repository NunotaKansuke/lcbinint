#include "lcbinint/obs/coordinates.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace lcbinint::obs {

namespace {

std::array<double, 3> three_point_derivative(
    double x0, double x1, double x2,
    const std::array<double, 3>& y0,
    const std::array<double, 3>& y1,
    const std::array<double, 3>& y2,
    double x)
{
    const double w0 = (2.0 * x - x1 - x2) / ((x0 - x1) * (x0 - x2));
    const double w1 = (2.0 * x - x0 - x2) / ((x1 - x0) * (x1 - x2));
    const double w2 = (2.0 * x - x0 - x1) / ((x2 - x0) * (x2 - x1));
    return {
        w0 * y0[0] + w1 * y1[0] + w2 * y2[0],
        w0 * y0[1] + w1 * y1[1] + w2 * y2[1],
        w0 * y0[2] + w1 * y1[2] + w2 * y2[2],
    };
}

std::vector<std::array<double, 3>> estimate_velocities(
    const std::vector<double>& times,
    const std::vector<std::array<double, 3>>& positions)
{
    std::vector<std::array<double, 3>> velocities(positions.size());
    if (positions.size() < 2 || times.size() != positions.size()) {
        return velocities;
    }
    if (positions.size() == 2) {
        const double dt = times[1] - times[0];
        const auto slope = std::array<double, 3>{
            (positions[1][0] - positions[0][0]) / dt,
            (positions[1][1] - positions[0][1]) / dt,
            (positions[1][2] - positions[0][2]) / dt,
        };
        velocities[0] = slope;
        velocities[1] = slope;
        return velocities;
    }

    velocities[0] = three_point_derivative(
        times[0], times[1], times[2], positions[0], positions[1], positions[2], times[0]);
    for (std::size_t i = 1; i + 1 < positions.size(); ++i) {
        velocities[i] = three_point_derivative(
            times[i - 1], times[i], times[i + 1],
            positions[i - 1], positions[i], positions[i + 1], times[i]);
    }
    const std::size_t last = positions.size() - 1;
    velocities[last] = three_point_derivative(
        times[last - 2], times[last - 1], times[last],
        positions[last - 2], positions[last - 1], positions[last], times[last]);
    return velocities;
}

} // namespace

Site::Site(
    std::vector<double> times,
    std::vector<std::array<double, 3>> positions)
    : kind_(SiteKind::space)
    , times_(std::move(times))
    , positions_(std::move(positions))
    , velocities_(estimate_velocities(times_, positions_))
{
}

Site Site::limited_to(
    double lower_time, double upper_time, std::size_t padding) const
{
    if (kind_ != SiteKind::space || !has_space_ephemeris()) {
        return *this;
    }
    if (!(lower_time < upper_time)) {
        throw std::invalid_argument(
            "space ephemeris time limit must satisfy lower < upper");
    }
    if (lower_time < times_.front() || upper_time > times_.back()) {
        throw std::invalid_argument(
            "Options.t_lim lies outside the space ephemeris table");
    }

    const auto first_upper =
        std::upper_bound(times_.begin(), times_.end(), lower_time);
    const std::size_t first_index =
        static_cast<std::size_t>(first_upper - times_.begin());
    const std::size_t start =
        first_index > padding ? first_index - padding : 0;

    const auto last_lower =
        std::lower_bound(times_.begin(), times_.end(), upper_time);
    const std::size_t last_index =
        static_cast<std::size_t>(last_lower - times_.begin());
    const std::size_t stop =
        std::min(times_.size(), last_index + padding + 1);

    Site limited(
        std::vector<double>(
            times_.begin() + static_cast<std::ptrdiff_t>(start),
            times_.begin() + static_cast<std::ptrdiff_t>(stop)),
        std::vector<std::array<double, 3>>(
            positions_.begin() + static_cast<std::ptrdiff_t>(start),
            positions_.begin() + static_cast<std::ptrdiff_t>(stop)));
    limited.velocities_ = std::vector<std::array<double, 3>>(
        velocities_.begin() + static_cast<std::ptrdiff_t>(start),
        velocities_.begin() + static_cast<std::ptrdiff_t>(stop));
    return limited;
}

bool Site::space_position(double time, std::array<double, 3>& position) const noexcept
{
    if (!has_space_ephemeris()) {
        return false;
    }
    auto upper = std::upper_bound(times_.begin(), times_.end(), time);
    std::size_t hi = 1;
    if (upper == times_.begin()) {
        hi = 1;
    } else if (upper == times_.end()) {
        hi = times_.size() - 1;
    } else {
        hi = static_cast<std::size_t>(upper - times_.begin());
    }
    const std::size_t lo = hi - 1;
    const double dt = times_[hi] - times_[lo];
    if (dt == 0.0) {
        position = positions_[lo];
        return true;
    }
    const double u = (time - times_[lo]) / dt;
    const double u2 = u * u;
    const double u3 = u2 * u;
    const double h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
    const double h10 = u3 - 2.0 * u2 + u;
    const double h01 = -2.0 * u3 + 3.0 * u2;
    const double h11 = u3 - u2;
    for (std::size_t i = 0; i < position.size(); ++i) {
        position[i] = h00 * positions_[lo][i] + h10 * dt * velocities_[lo][i]
            + h01 * positions_[hi][i] + h11 * dt * velocities_[hi][i];
    }
    return true;
}

} // namespace lcbinint::obs
