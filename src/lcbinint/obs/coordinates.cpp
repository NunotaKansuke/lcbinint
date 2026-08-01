#include "lcbinint/obs/coordinates.hpp"

#include <algorithm>
#include <stdexcept>

namespace lcbinint::obs {

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

    return Site(
        std::vector<double>(
            times_.begin() + static_cast<std::ptrdiff_t>(start),
            times_.begin() + static_cast<std::ptrdiff_t>(stop)),
        std::vector<std::array<double, 3>>(
            positions_.begin() + static_cast<std::ptrdiff_t>(start),
            positions_.begin() + static_cast<std::ptrdiff_t>(stop)));
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
    const double weight = dt == 0.0 ? 0.0 : (time - times_[lo]) / dt;
    for (std::size_t i = 0; i < position.size(); ++i) {
        position[i] = positions_[lo][i] + weight * (positions_[hi][i] - positions_[lo][i]);
    }
    return true;
}

} // namespace lcbinint::obs
