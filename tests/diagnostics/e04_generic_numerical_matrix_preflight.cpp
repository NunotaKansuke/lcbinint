#include "lcbinint/magnification/algebraic_boundary.hpp"
#include "lcbinint/magnification/algebraic_critical_event_core.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>

namespace mg = lcbinint::magnification;
namespace ce = lcbinint::magnification::critical_events;

struct MatrixCase { const char* id; ce::CriticalEventInput input; };

int main()
{
    constexpr double alpha=.3, u0=-.2;
    const auto trajectory=[](double t) {
        return lcbinint::SourcePosition{t*std::cos(alpha)-u0*std::sin(alpha),
                                        t*std::sin(alpha)+u0*std::cos(alpha)};
    };
    const std::array<MatrixCase,20> cases{{
        {"regular_1",{{-.1737444200273406,.15560487302576048},.01,10,1.2}},
        {"regular_2",{{-.11642423067980426,.17333608542544085},.01,10,1.2}},
        {"regular_3",{{-.05910404133226791,.19106729782512122},.01,10,1.2}},
        {"regular_4",{{-.0017838519847315562,.2087985102248016},.01,10,1.2}},
        {"regular_5",{{.0555363373628048,.22652972262448196},.01,10,1.2}},
        {"signed_u0_1",{trajectory(-.12),.01,.1,1.2}},
        {"signed_u0_2",{trajectory(0),.01,.1,1.2}},
        {"signed_u0_3",{trajectory(.12),.01,.1,1.2}},
        {"q_exchange_low",{{.037,-.021},.007,.02,.9}}, {"q_exchange_high",{{-.037,-.021},.007,50,.9}},
        {"q_exchange_low_second_fixture_point",{{-.11,.04},.007,.02,.9}},
        {"q_exchange_high_second_fixture_point",{{.11,.04},.007,50,.9}},
        {"ld_uniform",{{.08,.03},.01,.001,1.1}}, {"ld_linear",{{.08,.03},.01,.001,1.1}},
        {"ld_square_root",{{.08,.03},.01,.001,1.1}}, {"near_caustic_existing",{{.1,.05},.01,.2,1}},
        {"low_q_close_existing",{{.002,.0007},.001,.001,1.1}},
        {"equal_axis_center",{{0,0},.01,1,1}}, {"equal_axis_plus",{{.05,0},.01,1,1}},
        {"equal_axis_minus",{{-.05,0},.01,1,1}}
    }};
    int successes=0;
    for(const auto& entry:cases) {
        const auto begin=std::chrono::steady_clock::now();
        const auto result=ce::enumerate_and_lift_critical_events(entry.input);
        const auto end=std::chrono::steady_clock::now();
        const double elapsed=std::chrono::duration<double,std::milli>(end-begin).count();
        successes+=result.success;
        std::printf("NUMERICAL_CASE id=%s x=%.17g y=%.17g rho=%.17g q=%.17g s=%.17g success=%d reason=%s events=%zu recursion=%d bisections=%d lift_candidates=%d elapsed_ms=%.3f\n",
                    entry.id,entry.input.source.x,entry.input.source.y,entry.input.rho,entry.input.q,entry.input.separation,
                    result.success?1:0,result.unsafe_reason.c_str(),result.events.size(),result.isolation_calls,
                    result.root_bisections,result.lift_candidates,elapsed);
    }
    const ce::CriticalEventInput collapsed{{0,0},.01,.001,1.1};
    ce::FrozenGeometry frozen; std::string reason;
    const bool frozen_ok=ce::freeze_positive_geometry(collapsed,&frozen,&reason);
    const mg::AlgebraicBoundaryLens lens{frozen.a,frozen.m1,frozen.m2};
    const auto intervals=mg::algebraic_boundary_intervals_at_radius(lens,collapsed.source,collapsed.rho,0.0,{});
    std::printf("COLLAPSED_RADIUS id=collapsed_polar_circle frozen=%d interval_success=%d reason=%s\n",
                frozen_ok?1:0,intervals.success?1:0,intervals.unsafe_reason.c_str());
    std::printf("NUMERICAL_MATRIX_SUMMARY success=%d/%zu collapsed_rejected=%d full_plan_not_evaluated=1\n",
                successes,cases.size(),(!intervals.success)?1:0);
    return 0; // This harness records numerical outcomes, including expected failures.
}
