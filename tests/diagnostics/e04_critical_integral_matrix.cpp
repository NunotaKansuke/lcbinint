#include "lcbinint/magnification/algebraic_boundary.hpp"
#include "lcbinint/magnification/algebraic_critical_event_core.hpp"

#include <array>
#include "lcbinint/magnification/algebraic_boundary_autodiff.hpp"
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
    // The JSON trajectory rows carry public q.  The matrix's `native_order`
    // requires the reciprocal native q used by the numerical core.
    const auto public_q_to_native=[](double q_public) { return 1.0/q_public; };
    const std::array<MatrixCase,20> cases{{
        {"regular_1",{{-.1737444200273406,.15560487302576048},.01,10,1.2}},
        {"regular_2",{{-.11642423067980426,.17333608542544085},.01,10,1.2}},
        {"regular_3",{{-.05910404133226791,.19106729782512122},.01,10,1.2}},
        {"regular_4",{{-.0017838519847315562,.2087985102248016},.01,10,1.2}},
        {"regular_5",{{.0555363373628048,.22652972262448196},.01,10,1.2}},
        {"signed_u0_1",{trajectory(-.12),.01,public_q_to_native(.1),1.2}},
        {"signed_u0_2",{trajectory(0),.01,public_q_to_native(.1),1.2}},
        {"signed_u0_3",{trajectory(.12),.01,public_q_to_native(.1),1.2}},
        {"q_exchange_low",{{.037,-.021},.007,.02,.9}}, {"q_exchange_high",{{-.037,-.021},.007,50,.9}},
        {"q_exchange_low_second_fixture_point",{{-.11,.04},.007,.02,.9}},
        {"q_exchange_high_second_fixture_point",{{.11,.04},.007,50,.9}},
        {"ld_uniform",{{.08,.03},.01,.001,1.1}}, {"ld_linear",{{.08,.03},.01,.001,1.1}},
        {"ld_square_root",{{.08,.03},.01,.001,1.1}}, {"near_caustic_existing",{{.1,.05},.01,.2,1}},
        {"low_q_close_existing",{{.002,.0007},.001,.001,1.1}},
        {"equal_axis_center",{{0,0},.01,1,1}}, {"equal_axis_plus",{{.05,0},.01,1,1}},
        {"equal_axis_minus",{{-.05,0},.01,1,1}}
    }};

    int successes=0, required_failures=0;
    for(const auto& entry:cases) {
        const auto& p=entry.input;
        mg::AlgebraicBoundarySettings settings;
        settings.absolute_tolerance=1e-4;settings.relative_tolerance=1e-4;
        settings.critical_events=true;
        const double c=std::string(entry.id)=="ld_linear"?.4:0.;
        const double d=std::string(entry.id)=="ld_square_root"?.3:0.;
        const auto result=mg::autodiff::experimental_algebraic_boundary_jacobian(
            {p.source.x,p.source.y,p.rho,p.q,p.separation},{1.,p.q<1.},c,d,{},settings,3e-4);
        const bool ok=result.success&&result.grad_reliable&&std::isfinite(result.magnification)&&
           result.value_error<=1e-4+1e-4*std::max(1.,std::abs(result.magnification));
        successes+=ok;
        if(std::string(entry.id).find("equal_axis")!=0&&!ok)++required_failures;
        std::printf("INTEGRAL id=%s ok=%d success=%d grad=%d A=%.17g ve=%.9g ge=%.9g bands=%d reason=%s J=",entry.id,ok,result.success,result.grad_reliable,result.magnification,result.value_error,result.grad_error,result.diagnostics.radial_band_count,result.unsafe_reason.c_str());
        for(auto v:result.jacobian) std::printf(" %.17g",v);
        std::printf("\n");std::fflush(stdout);
    }
    std::printf("INTEGRAL_SUMMARY success=%d/20 required_failures=%d\n",successes,required_failures);
    return required_failures?1:0;
}
