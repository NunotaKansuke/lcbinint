#include "lcbinint/magnification/algebraic_boundary/triple/angular_topology.hpp"
#include <iostream>
#include <random>
#include "lcbinint/magnification/algebraic_boundary/triple/local_precision.hpp"
using namespace lcbinint::algebraic_boundary::triple;
void check(bool ok,const char* message) {if(!ok)throw std::runtime_error(message);}
int main() {
    {
        angular_detail::Polynomial<double> positive{};
        positive[0]=1;positive[1]=-.75;positive[3]=.5;
        check(angular_detail::excludes_unit_chart(positive,3,-1,1),"positive prefix sums");
        positive[1]=-2;
        check(!angular_detail::excludes_unit_chart(positive,3,-1,1),
              "sign change on the positive chart requires isolation");
        positive[1]=2;
        check(!angular_detail::excludes_unit_chart(positive,3,-1,1),
              "sign change on the negative chart requires isolation");
        angular_detail::Polynomial<double> p{};
        p[2]=1; p[1]=-1; p[0]=std::ldexp(1.,-48);
        std::vector<std::pair<double,double>> roots;
        check(angular_detail::quadratic_roots(p,-2,2,1e-12,roots) && roots.size()==2,
              "separated quadratic roots");
        const __float128 large=(1+sqrtq(1-4*(__float128)p[0].lo))/2;
        const __float128 small=(__float128)p[0].lo/large;
        check(roots[0].first<=small && small<=roots[0].second &&
              roots[1].first<=large && large<=roots[1].second,"stable small quadratic root");
        roots.clear();p[0]=.25;
        check(!angular_detail::quadratic_roots(p,-2,2,1e-12,roots) && roots.empty(),
              "multiple quadratic root requires isolation");
        p[0]=0;
        check(!angular_detail::quadratic_roots(p,0,2,1e-12,roots) && roots.empty(),
              "chart-edge root requires isolation");
    }

    Geometry p{.1,.15,.15,.2,.08,1.,-.4,.6};
    int nonempty=0,promotions=0;
    for(int k=0;k<60;++k) {
        const double R=.08+k*.037;
        auto t=angular_topology(R,p);
        if(!t.completed) {std::cerr<<R<<" "<<t.reason<<"\n";return 1;}
        promotions+=t.quad_promotions;
        for(auto arc:t.arcs)check(evaluate(R,.5*(arc.lo+arc.hi),p).phi>0,"arc interior");
        for(auto root:t.roots) {
            auto lo=local_precision::evaluate<false>(R,root.lo,p).phi,hi=local_precision::evaluate<false>(R,root.hi,p).phi;
            check(lo*hi<=0,"physical boundary bracket");
        }
        double length=0;for(auto a:t.arcs)length+=a.hi-a.lo;
        nonempty+=!t.arcs.empty();
        // Independent uniform-angle membership integration has a known error
        // <= 2pi * number_of_endpoints / N for simple crossing arcs.
        const int N=32768;int inside=0;const double pi=std::acos(-1.);
        for(int j=0;j<N;++j)inside+=evaluate(R,-pi*.5+2*pi*(j+.5)/N,p).phi>0;
        const double reference=inside*2*pi/N;
        check(std::abs(reference-length)<=2*pi*(t.roots.size()+1)/N+t.angular_measure_error,"independent angular measure");
    }
    check(nonempty>0,"nontrivial arcs tested");
    auto pole=angular_topology(1,p);
    check(pole.completed,"positive pole radius has regular physical boundary");
    // Tight endpoint tolerance forces local escalation; confirmed empty boxes
    // and previously accepted roots remain intact.
    auto fine=angular_topology(.93,p,{1e-13,8192,true});
    check(fine.completed,"quad refinement completed");

    auto disabled=angular_topology(.93,p,{1e-13,8192,false});
    check(disabled.completed || std::string(disabled.reason)=="precision_required","no silent low precision acceptance");
    {
        using I=angular_detail::I<double>;
        const std::array<double,6> exact{{-.875,-.5,-.25,.125,.375,.75}};
        angular_detail::Polynomial<double> polynomial{};
        polynomial[0]=I(1);
        int degree=0;
        for(double root:exact) {
            ++degree;
            for(int j=degree;j>=1;--j)
                polynomial[j]=polynomial[j-1]-I(root)*polynomial[j];
            polynomial[0]=-I(root)*polynomial[0];
        }
        std::vector<std::pair<double,double>> roots;
        int work=0;
        check(angular_detail::monotone_roots(polynomial,degree,-1,1,1e-9,roots,work,8192,true)
              && roots.size()==exact.size(),"certified Newton proposals retain all six crossings");
        for(std::size_t j=0;j<roots.size();++j)
            check(roots[j].first<=exact[j] && exact[j]<=roots[j].second &&
                  roots[j].second-roots[j].first<=1e-9,"Newton proposal brackets the known root");
    }
    auto empty=angular_topology(.05,p);
    check(empty.completed && empty.quad_promotions==0,"ordinary empty radius stays double");
    // A tiny disk centred on a mapped image creates a close pair of roots.
    // Uniform angle sampling can miss the arc entirely; the polynomial solver
    // must retain it without fixed-distance root merging.
    auto tiny=p;
    const double narrow_R=.94,narrow_theta=.37;
    auto z=std::polar(narrow_R,narrow_theta),source=z;
    auto lens=tiny.lenses();auto mass=tiny.masses();
    for(int j=0;j<3;++j)source-=mass[j]/std::conj(z-lens[j]);
    tiny.X=source.real();tiny.Y=source.imag();tiny.rho=1e-7;
    auto narrow=angular_topology(narrow_R,tiny,{1e-13,8192,true});
    check(narrow.completed && narrow.roots.size()>=2,"close real roots retained");
    auto narrow_disabled=angular_topology(narrow_R,tiny,{1e-13,8192,false});
    check(!narrow_disabled.completed && std::string(narrow_disabled.reason)=="precision_required",
          "ill-conditioned root count requires promotion");
    auto continued=continued_topology(narrow_R,tiny,narrow,{1e-15,8192,true});
    check(continued.completed && continued.roots.size()==narrow.roots.size(),
          "local promoted continuation retains complete crossing count");
    // Extended precision can certify this bracket before binary128.
    // Disabling both promotion stages below must still fail closed.
    if constexpr(std::numeric_limits<long double>::digits!=64)
        check(continued.quad_promotions>0,"cancellation-limited continued root promotes locally");
    for(auto root:continued.roots) {
        const auto a=local_precision::evaluate<false>(narrow_R,root.lo,tiny).phi;
        const auto b=local_precision::evaluate<false>(narrow_R,root.hi,tiny).phi;
        check(a*b<=0,"promoted continued root brackets physical boundary");
    }
    auto continued_double=continued_topology(narrow_R,tiny,narrow,{1e-15,8192,false});
    check(!continued_double.completed,"disabled local promotion fails closed");
    double narrow_length=0;for(auto arc:narrow.arcs)narrow_length+=arc.hi-arc.lo;
    check(narrow_length>0 && narrow_length<1e-5,"tiny angular arc recovered");
    const int rounding=std::fegetround();
    std::fesetround(FE_DOWNWARD);
    auto environment=angular_topology(narrow_R,tiny);
    std::fesetround(rounding);
    check(!environment.completed && std::string(environment.reason)=="floating_environment",
          "non-nearest rounding rejected");
    auto budget=angular_topology(narrow_R,tiny,{1e-13,1,true});
    check(!budget.completed,"box budget fails closed");
    std::cout<<"PASS: 60 radii, "<<nonempty<<" occupied, "<<promotions
             <<" local promotions; tight bracket quad boxes="<<fine.quad_boxes<<"\n";
}
