#include "lcbinint/model/trajectory.hpp"

#include <cmath>

namespace lcbinint::model {

SourcePosition Trajectory::source_position(double time) const
{
    const double tn = (time - params_.t0) / params_.tE;
    const double costheta = std::cos(params_.theta);
    const double sintheta = std::sin(params_.theta);

    SourcePosition source;
    source.x = params_.umin * sintheta + tn * costheta;
    source.y = params_.umin * costheta - tn * sintheta;
    return source;
}

} // namespace lcbinint::model
