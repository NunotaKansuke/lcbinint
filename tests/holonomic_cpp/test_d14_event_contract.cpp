#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"

using namespace lcbinint::holonomic;

namespace {

struct Run {
    TopologyResult topology;
    AdaptiveResult value;
    V2Profile profile;
};

Run evaluate(const LensParams& params, bool event_contract) {
    if (event_contract)
        setenv("HOLO_D14_EVENT_CONTRACT_ACCEPT", "1", 1);
    else
        unsetenv("HOLO_D14_EVENT_CONTRACT_ACCEPT");
    Run run;
    {
        V2ProfileScope scope(run.profile);
        run.topology=classify_cells(PrimaryFrame::from(params),nullptr,nullptr,true);
    }
    AdaptiveConfig cfg;
    cfg.gradient_policy=GradientPolicy::None;
    cfg.with_jacobian=false;
    cfg.tol.mu_atol=1e-16;
    cfg.tol.mu_rtol=1e-4;
    cfg.preserve_radial_offset=true;
    AdaptiveWorkspace workspace;
    run.value=flux_adaptive_integrate(params,0.5,run.topology,cfg,workspace);
    return run;
}

bool same_event_structure(const TopologyResult& a,const TopologyResult& b) {
    if(a.status!=b.status||a.cells.size()!=b.cells.size()||
       a.events.size()!=b.events.size())return false;
    for(size_t i=0;i<a.events.size();++i) {
        if(a.events[i].kind!=b.events[i].kind||
           a.events[i].physically_real!=b.events[i].physically_real||
           std::fabs(a.events[i].radius-b.events[i].radius)>1e-8)return false;
    }
    return true;
}

} // namespace

int main() {
    int failures=0;
    auto check=[&](bool ok,const char* message){if(!ok){
        std::cerr<<"FAIL: "<<message<<'\n';++failures;}};

    // Corpus case 0/d=0/epoch=0: qf-warm misses the global step threshold,
    // while fourteen disjoint Rouche disks certify the retained root set.
    const LensParams accepted_params{
        1.1409565850100225,-4.0472265932659397,0.00017266345397718498,
        1.0/0.56332240832139935,0.2310772011499552,true};
    const Run oracle=evaluate(accepted_params,false);
    const Run accepted=evaluate(accepted_params,true);
    check(accepted.profile.d14_event_contract_attempts==1,
          "known candidate was not screened");
    check(accepted.profile.d14_event_contract_accepts==1,
          "known isolated root set was not accepted");
    check(oracle.profile.d14_qf_cold_calls==1&&
          accepted.profile.d14_qf_cold_calls==0,
          "event certificate did not replace the qf-cold restart");
    check(same_event_structure(oracle.topology,accepted.topology),
          "accepted event structure differs from qf-cold oracle");
    check(oracle.value.value_converged==accepted.value.value_converged&&
          oracle.value.stop==accepted.value.stop&&
          std::fabs(oracle.value.mu-accepted.value.mu)<1e-10,
          "accepted adaptive value differs from qf-cold oracle");

    // Corpus case 49/d=1/epoch=0 is the historical counterexample: a tiny
    // residual accompanies a wrong positive-root/event count.  The disjoint
    // disk contract must reject it and retain the incumbent qf-cold backstop.
    const LensParams rejected_params{
        -0.012715162713256456,0.30367111943384689,0.64030113599253125,
        1.0/0.0001084095167119894,0.47185016883470127,true};
    const Run rejected=evaluate(rejected_params,true);
    check(rejected.profile.d14_event_contract_attempts==1,
          "known ambiguous candidate was not screened");
    check(rejected.profile.d14_event_contract_accepts==0&&
          rejected.profile.d14_qf_cold_calls==1,
          "known wrong-topology candidate bypassed qf-cold");

    unsetenv("HOLO_D14_EVENT_CONTRACT_ACCEPT");
    if(failures==0)std::cout<<"D14 event-contract tests passed\n";
    return failures?1:0;
}
