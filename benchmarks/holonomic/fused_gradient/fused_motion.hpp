#pragma once
#include <array>
namespace lcbinint::holonomic::fused_gradient {
struct Motion { std::array<double,5> radius{}, jacobian{}; };
inline thread_local const Motion* active_motion=nullptr;
struct MotionScope {
    const Motion* previous;
    explicit MotionScope(const Motion* motion):previous(active_motion){active_motion=motion;}
    ~MotionScope(){active_motion=previous;}
};
}
