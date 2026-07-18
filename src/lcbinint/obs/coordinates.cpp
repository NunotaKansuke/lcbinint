#include "lcbinint/obs/coordinates.hpp"

#include <algorithm>

namespace lcbinint::obs {

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
