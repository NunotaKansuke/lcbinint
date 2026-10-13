#include <iostream>
#include <random>
#include "lcbinint/magnification/holonomic/quartic_fast_count.hpp"

using namespace lcbinint::holonomic;

int main() {
    int checks=0, failures=0, fast=0, high_precision_checked=0;
    auto check=[&](bool ok,const char* message) {
        ++checks;
        if(!ok){++failures;std::cerr<<message<<'\n';}
    };
    const std::array<std::array<double,5>,3> known{{
        {{1,0,0,0,1}},     // x^4+1: no real roots
        {{-1,0,0,0,1}},    // x^4-1: two real roots
        {{4,0,-5,0,1}}}}; // (x^2-1)(x^2-4): four real roots
    const int counts[]={0,2,4};
    for(int exponent=-1000;exponent<=1000;exponent+=25)
        for(int sign:{-1,1}) for(std::size_t j=0;j<known.size();++j) {
            auto p=known[j];for(auto& c:p)c=std::ldexp(sign*c,exponent);
            check(boundary_quartic_count(p)==counts[j],"known roots survive scaling/sign reversal");
        }
    for(auto p:std::array<std::array<double,5>,3>{{
        {{1,-4,6,-4,1}},{{1,0,-2,0,1}},{{0,0,-1,0,1}}}})
        check(invariant_quartic_count(p)==-1,"multiple roots require Sturm fallback");
    check(invariant_quartic_count({1,0,2,0,1})==0,
          "positive definite quartic has no real roots even with repeated complex roots");
    for(double c:{std::nextafter(1.0,0.0),std::nextafter(1.0,2.0)}) {
        const std::array<double,5> p{{c,0,-2,0,1}};
        check(invariant_quartic_count(p)==-1,"one-ulp split remains ambiguous to the fast signs");
        check(boundary_quartic_count(p)==(c<1?4:0),"Sturm resolves near-coincident roots");
    }
    check(invariant_quartic_count({0.125,-1,0,1,0})==-1,"projective chart crossing retains Sturm");
    check(boundary_quartic_count({0.125,-1,0,1,0})==4,"cubic plus infinity has four projective real roots");
    check(boundary_quartic_count({1,0,0,0,std::numeric_limits<double>::infinity()})==-1,"nonfinite input is rejected");
    check(boundary_quartic_count({0,0,0,0,0})==-1,"zero polynomial is rejected");
    {
        // Actual physical probe: a double Sturm chain reported two roots
        // with a large leading-margin screen. Exact rational Sturm gives zero.
        const std::array<double,5> p{{-3.1019337649664194,12.17726286383744,
            -30.545588616842519,123.04800387697598,-185.91161075698324}};
        check(boundary_quartic_count(p)==0,"physical false-positive Sturm probe has no real roots");
        const auto reference=quartic_sturm_detail::count_chain(
            quartic_sturm_detail::normalized<__float128>(p));
        check(reference.count==0,"binary128 independently confirms the exact-rational counterexample");
    }

    // An independent high-precision Sturm sequence validates decisions from
    // the invariant graphs, including cancellation and wide dynamic range.
    std::mt19937_64 rng(3101);
    std::uniform_real_distribution<double> value(-1,1);
    for(int j=0;j<200000;++j) {
        std::array<double,5> p;
        const double scale=std::pow(10.0,int(rng()%601)-300);
        for(auto& c:p)c=value(rng)*scale;
        if(j%10==0)p[rng()%5]=0;
        if(j%20==0)p={scale,0,-2*scale,0,scale};
        const int direct=invariant_quartic_count(p);
        if(direct<0)continue;
        ++fast;
        const auto counted=quartic_sturm_detail::count_chain(
            quartic_sturm_detail::normalized<__float128>(p));
        if(!quartic_sturm_detail::stable(counted,2))continue;
        ++high_precision_checked;
        check(direct==counted.count,"fast invariant count agrees with independent binary128 Sturm");
    }
    check(fast>100000 && high_precision_checked>100000,"validation exercises many accepted fast counts");
    std::cout<<checks<<" checks "<<failures<<" failures; "<<fast<<" fast, "
             <<high_precision_checked<<" independently checked\n";
    return failures!=0;
}
