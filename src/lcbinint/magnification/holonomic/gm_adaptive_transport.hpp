#pragma once
#include "lcbinint/magnification/holonomic/gm_physical_seed.hpp"
#include "lcbinint/magnification/holonomic/gm_sensitivity.hpp"
#include "lcbinint/magnification/holonomic/gm_direct_connection.hpp"
#include "lcbinint/magnification/holonomic/gm_taylor_transport.hpp"
#include <chrono>

namespace lcbinint::holonomic {
struct GmAdaptiveCost {
    int connections=0, connection_failed=0, steps=0, rejected=0;
    double connection_ms=0, transport_ms=0;
};

// Reconstruct connections, never physical seeds. A failed attempt does not
// mutate the accepted state. Both the full period state and flux are gated
// by the caller/reference harness; the embedded estimate here is local.
template<class Scalar>
inline bool gm_adaptive_advance(double& R, double target, double lo, double hi,
    const PrimaryFrame& pf, std::array<Scalar,7>& state, GmAdaptiveCost& cost) {
    using Clock=std::chrono::steady_clock;
    auto p=gm_sensitivity_params<Scalar>(pf);
    int budget=2048;
    while(R!=target && --budget>0){
        double distance=std::min(R-lo,hi-R);
        double step=std::copysign(std::min(std::fabs(target-R),0.35*distance),target-R);
        if(R+step==R || !(distance>0))return false;
        auto t0=Clock::now();
        auto jet=gm_direct_connection_jet<8,Scalar>(R,p,std::fabs(step));
        cost.connection_ms+=std::chrono::duration<double,std::milli>(Clock::now()-t0).count();
        ++cost.connections;
        if(!jet.ok){++cost.connection_failed;return false;}
        bool accepted=false;
        for(int retry=0;retry<24;++retry){
            std::array<Scalar,7> upper{},lower{};
            t0=Clock::now();
            // The initial scale is fixed for all rejected-step retries.
            const double scale=std::min(std::fabs(target-R),0.35*distance);
            bool ok=gm_taylor_transport(jet,state,step/scale,upper) &&
                gm_taylor_transport_prefix<6>(jet,state,step/scale,lower);
            double err=0;
            for(int k=0;k<7;++k)err=std::max(err,gm_state_error(upper[k],lower[k],state[k]));
            cost.transport_ms+=std::chrono::duration<double,std::milli>(Clock::now()-t0).count();
            if(ok && err<1e-12){state=upper;R=(step==target-R)?target:R+step;
                ++cost.steps;accepted=true;break;}
            ++cost.rejected;step*=0.5;
            if(R+step==R)break;
        }
        if(!accepted)return false;
    }
    return R==target;
}
} // namespace lcbinint::holonomic
