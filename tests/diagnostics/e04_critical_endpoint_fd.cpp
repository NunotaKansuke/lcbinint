#include "lcbinint/magnification/algebraic_critical_event_core.hpp"
#include "lcbinint/magnification/algebraic_boundary_autodiff.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>

namespace ce = lcbinint::magnification::critical_events;
namespace ad = lcbinint::magnification::autodiff;
using J = ad::ForwardJet<6>;
using Params = std::array<double, 5>;

ce::CriticalEventInput input(const Params& p) { return {{p[0],p[1]},p[2],p[3],p[4]}; }

int main()
{
    int failures = 0;
    const std::array<Params,4> cases{{
        {{.0555363373628048,.22652972262448196,.01,10,1.2}},
        {{.037,-.021,.007,.02,.9}},
        {{.08,.03,.01,.001,1.1}},
        {{.002,.0007,.001,.001,1.1}}
    }};
    for (std::size_t k=0;k<cases.size();++k) {
        const auto p=cases[k];
        const auto plan=ce::find_critical_events(input(p));
        if (!plan.success) { std::printf("ENDPOINT_FD_BASE_REJECT case=%zu reason=%s\n",k,plan.unsafe_reason.c_str()); ++failures; continue; }
        std::array<J,5> pj;
        for(std::size_t j=0;j<5;++j)pj[j]=J::variable(p[j],j);
        const auto geometry=ad::make_geometry<J>(pj,{1.0,p[3]<1.0});
        ce::FrozenGeometry frozen;
        std::string reason;
        if (!ce::freeze_positive_geometry(input(p),&frozen,&reason) ||
            geometry.a.value!=frozen.a || geometry.m1.value!=frozen.m1 ||
            geometry.m2.value!=frozen.m2 || geometry.zeta_x.value!=frozen.sx ||
            geometry.zeta_y.value!=frozen.sy || geometry.rho.value!=frozen.rho) ++failures;
        std::vector<Params> analytic(plan.events.size());
        for(std::size_t i=0;i<plan.events.size();++i) {
            const auto& e=plan.events[i];
            const auto form=ad::make_trig_form(geometry,J::variable(e.radius,5));
            const auto f=ad::trig_value(form,e.theta);
            const auto ft=ad::trig_first(form,e.theta);
            const auto ftt=ad::trig_second(form,e.theta);
            const double det=f.derivative[5]*ftt.value-ft.value*ft.derivative[5];
            for(std::size_t j=0;j<5;++j)
                analytic[i][j]=(-f.derivative[j]*ftt.value+ft.value*ft.derivative[j])/det;
        }
        double worst_best=0.0;
        for(std::size_t j=0;j<5;++j) {
            std::vector<double> best(plan.events.size(),INFINITY);
            for(double fraction:{1e-3,5e-4,2.5e-4}) {
                // Separation moves caustics: use the source-radius scale for
                // a local derivative, not an O(1) separation-scale step.
                const double h=fraction*(j==3?p[3]:p[2]);
                auto plus=p,minus=p;plus[j]+=h;minus[j]-=h;
                const auto a=ce::find_critical_events(input(plus));
                const auto b=ce::find_critical_events(input(minus));
                if(!a.success||!b.success||a.events.size()!=plan.events.size()||b.events.size()!=plan.events.size()) {
                    std::printf("ENDPOINT_FD_REJECT case=%zu param=%zu fraction=%.9g plus=%s minus=%s counts=%zu/%zu\n",k,j,fraction,a.unsafe_reason.c_str(),b.unsafe_reason.c_str(),a.events.size(),b.events.size());
                    ++failures;continue;
                }
                for(std::size_t i=0;i<plan.events.size();++i) {
                    const double fd=(a.events[i].radius-b.events[i].radius)/(2*h);
                    const double error=std::abs(fd-analytic[i][j])/(1+std::abs(analytic[i][j]));
                    best[i]=std::min(best[i],error);
                }
            }
            for(double error:best) {worst_best=std::max(worst_best,error);if(!(error<5e-5))++failures;}
        }
        std::printf("ENDPOINT_FD case=%zu events=%zu worst_best_scaled_error=%.9g\n",k,plan.events.size(),worst_best);
    }
    std::printf("ENDPOINT_FD_SUMMARY failures=%d\n",failures);
    return failures?1:0;
}
