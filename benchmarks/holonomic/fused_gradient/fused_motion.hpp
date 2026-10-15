#pragma once
#include <array>
namespace lcbinint::holonomic::fused_gradient {
// Scoped per-node scheduling: only value-adaptive child panels postpone
// tangents until their final geometry has been accepted.
inline thread_local bool defer_sample_tangents=false;
inline thread_local bool reliability_retry_used=false;
struct TangentScheduleScope {
    bool previous;
    explicit TangentScheduleScope(bool defer):previous(defer_sample_tangents){defer_sample_tangents=defer;}
    ~TangentScheduleScope(){defer_sample_tangents=previous;}
};
struct Motion { std::array<double,5> radius{}, jacobian{}; };
inline thread_local const Motion* active_motion=nullptr;
inline thread_local double radius_offset=0;
struct RadiusOffsetScope {
 double previous;
 explicit RadiusOffsetScope(double offset):previous(radius_offset){radius_offset=offset;}
 ~RadiusOffsetScope(){radius_offset=previous;}
};
struct MotionScope {
    const Motion* previous;
    explicit MotionScope(const Motion* motion):previous(active_motion){active_motion=motion;}
    ~MotionScope(){active_motion=previous;}
};
}
