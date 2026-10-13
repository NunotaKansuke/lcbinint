#include "lcbinint/model/lens_system.hpp"

#include <cmath>

namespace lcbinint::model {

LensSystem LensSystem::from_parameters(const LensParameters &params)
{
    LensSystem system;
    if (!params.is_triple()) {
        const double q = std::abs(params.q);
        const double m2 = q / (1.0 + q);
        const double m1 = 1.0 - m2;
        system.primary_ = LensBody{{-m2 * std::abs(params.sep), 0.0}, m1};
        system.secondary_ = LensBody{{m1 * std::abs(params.sep), 0.0}, m2};
        return system;
    }

    system.is_triple_ = true;
    const double q = std::abs(params.q);
    const double q2 = params.q2;
    const double eps2 = q / (1.0 + q + q2);
    const double eps3 = q2 / (1.0 + q + q2);
    const double eps4 = eps2 + eps3;
    const double eps1 = 1.0 - eps4;

    const double sep = std::abs(params.sep);
    const double sep2 = params.sep2;
    const double ang = params.ang;
    const double xx1 = -eps4 * sep;
    const double xx4 = eps1 * sep;
    const double xx2 = xx4 + eps3 / eps4 * sep2 * std::cos(ang);
    const double yy2 = eps3 / eps4 * sep2 * std::sin(ang);
    const double xx3 = xx4 - eps2 / eps4 * sep2 * std::cos(ang);
    const double yy3 = -eps2 / eps4 * sep2 * std::sin(ang);

    system.primary_ = LensBody{{xx1, 0.0}, eps1};
    system.secondary_ = LensBody{{xx2, yy2}, eps2};
    system.tertiary_ = LensBody{{xx3, yy3}, eps3};
    return system;
}

} // namespace lcbinint::model
