#pragma once

#include <array>
#include <chrono>
#include <cstdlib>
#include <vector>

namespace lcbinint::magnification::adaptive_detail {

// Diagnostic sidecar: keep the request ABI and numerical decisions unchanged.
// stage,kind,seconds,new maps,error before/after,flux before/after.
struct RefinementWorkTrace {
    bool enabled=false;
    std::vector<std::array<double,8>> rows;
    void reset() {
        enabled=std::getenv("LCBININT_DIAGNOSTIC_REFINEMENT_WORK_TRACE")!=nullptr;
        rows.clear();
    }
    void initial(double maps,double error,double flux) {
        if(enabled)rows.push_back({0,-1,0,maps,error,error,flux,flux});
    }
};
inline RefinementWorkTrace& refinement_work_trace() {
    static thread_local RefinementWorkTrace trace;
    return trace;
}

template<class Sample> struct RefinementWorkTrial {
    RefinementWorkTrace& trace;
    Sample sample;
    std::array<double,3> before{};
    std::chrono::steady_clock::time_point start{};
    double kind=0;
    RefinementWorkTrial(RefinementWorkTrace& t,Sample s):trace(t),sample(s) {
        if(trace.enabled){before=sample();start=std::chrono::steady_clock::now();}
    }
    ~RefinementWorkTrial() {
        if(!trace.enabled)return;
        const auto after=sample();
        const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        trace.rows.push_back({double(trace.rows.size()),kind,seconds,after[0]-before[0],
            before[1],after[1],before[2],after[2]});
    }
};
}
