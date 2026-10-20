#pragma once
#include <array>
#include <vector>
namespace lcbinint::algebraic_boundary {
struct RadialSupportPlan {
    bool completed = false;
    std::vector<double> radii;
    std::vector<int> crossings;
    // Inner lower/upper witnesses, followed by their exterior witnesses.
    std::vector<std::array<double,4>> brackets;
};
}
