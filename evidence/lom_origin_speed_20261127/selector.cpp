#include "lcbinint/magnification/finite_source_magnifier.hpp"
#include <algorithm>
#include <ctime>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

using lcbinint::magnification::FiniteSourceMagnifier;

int main() {
    FiniteSourceMagnifier reference({}), selected({});
    std::vector<double> spreads;
    const auto start = std::clock();
    for (double q : {.0001,.001,.01,.1,.999999,1.000001,10.,1000.,10000.})
        for (int i=0;i<200;++i) {
            const double s = .25 + .0125*i;
            spreads.push_back(reference.binary_heavy_origin_spread(s,q));
        }
    const auto split = std::clock();
    std::size_t index=0,heavy=0;
    for (double q : {.0001,.001,.01,.1,.999999,1.000001,10.,1000.,10000.})
        for (int i=0;i<200;++i) {
            const double s=.25+.0125*i;
            const bool actual=selected.binary_uses_heavy_origin(s,q,{10.,10.},.001);
            if (actual != (spreads[index++]>=.03)) throw std::runtime_error("origin disagreement");
            heavy+=actual;
        }
    const auto stop = std::clock();
    // A cached early threshold witness must never escape as the full spread.
    for (double q : {.001,.1,1.001,1000.}) for (double s : {-.8,1.,1.2}) {
        selected.binary_uses_heavy_origin(s,q,{10.,10.},.001);
        if (selected.binary_heavy_origin_spread(s,q)!=reference.binary_heavy_origin_spread(s,q))
            throw std::runtime_error("incomplete spread returned by full query");
    }
    int near_threshold=0;
    for (double s : {.4,.6,.8,1.,1.2,1.5,2.0}) {
        double lo=.00001, hi=.9999;
        if (!(reference.binary_heavy_origin_spread(s,lo)<.03 &&
              reference.binary_heavy_origin_spread(s,hi)>=.03)) continue;
        for (int i=0;i<60;++i) {
            double q=(lo+hi)/2;
            const bool expected=reference.binary_heavy_origin_spread(s,q)>=.03;
            if (selected.binary_uses_heavy_origin(s,q,{10.,10.},.001)!=expected)
                throw std::runtime_error("origin disagreement near threshold");
            if (expected) hi=q; else lo=q;
            ++near_threshold;
        }
    }
    if (!near_threshold) throw std::runtime_error("missing threshold cases");
    std::printf("%d near-threshold checks; ", near_threshold);
    const double ms_per_tick=1000./CLOCKS_PER_SEC;
    std::printf("%zu geometries, %zu heavy, full %.2f ms, selector %.2f ms\n",
        index,heavy,ms_per_tick*(split-start),ms_per_tick*(stop-split));
}
