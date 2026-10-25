#pragma once
namespace lcbinint::algebraic_boundary {
// Shared incumbent defaults: do not change accuracy policy independently in
// binary value, tangent, or triple controllers.
struct QuadratureErrorPolicy {
    static constexpr double difference_safety=2.0;
    static constexpr double weighted_detail_floor=0.125;
};
}
