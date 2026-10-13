#pragma once
#include "lcbinint/magnification/critical_phase_enclosure.hpp"
#include "lcbinint/math/filtered_ball.hpp"

namespace lcbinint::magnification {
namespace critical_fast_detail {
using I=math::OutwardInterval;
using B=math::FilteredBall;
struct C {B x,y;};
inline I interval(B x) {return I(x.value)+I(-x.radius,x.radius);}
inline B ball(I x) {
    const double c=x.lo*.5+x.hi*.5;
    return B(c,std::max((I(c)-I(x.lo)).hi,(I(x.hi)-I(c)).hi));
}
inline C add(C a,C b) {return {a.x+b.x,a.y+b.y};}
inline C sub(C a,C b) {return {a.x-b.x,a.y-b.y};}
inline C mul(C a,C b) {return {a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x};}
inline C scale(C a,B b) {return {a.x*b,a.y*b};}
inline C inv(C a) {const B d=math::square(a.x)+math::square(a.y);return {a.x/d,-a.y/d};}
inline I norm(C a) {
    const I d=interval(math::square(a.x)+math::square(a.y));
    if (!d.finite()) return I(0,std::numeric_limits<double>::infinity());
    return I(std::max(0.0,math::down(std::sqrt(std::max(0.0,d.lo)))),math::up(std::sqrt(std::max(0.0,d.hi))));
}
struct Jet {C kappa,mapped;I derivative,d1,d2;double cap=0;};
struct ProposalCache {
    double s=0,q=0;
    std::array<SourcePosition,4> centres{};
    std::array<Jet,4> jets{};
    I m1,m2;
    bool ready=false;
};
inline ProposalCache& proposal_cache(double s,double qi,const std::array<SourcePosition,4>& roots) {
    static thread_local ProposalCache cache;
    bool same=cache.ready && cache.s==s && cache.q==qi;
    for (int j=0;j<4 && same;++j) same=roots[j].x==cache.centres[j].x && roots[j].y==cache.centres[j].y;
    if (same) return cache;
    cache.ready=true;cache.s=s;cache.q=qi;cache.centres=roots;
    const I q=qi<1 ? I(qi) : I(1)/I(qi);
    cache.m1=I(1)/(I(1)+q);cache.m2=q*cache.m1;
    const double a=qi<1 ? -std::abs(s) : std::abs(s);
    const B m1=ball(cache.m1),m2=ball(cache.m2),shift=ball(I(a)*cache.m1);
    for (int j=0;j<4;++j) {
        const C z{B(roots[j].x),B(roots[j].y)},za=sub(z,{B(a),B(0)});
        const C u=inv(za),v=inv(z),p=scale(mul(u,u),m1),t=scale(mul(v,v),m2);
        auto& jet=cache.jets[j];jet.kappa=add(p,t);
        jet.derivative=norm(scale(add(mul(p,u),mul(t,v)),B(-2)));
        jet.d1=norm(za);jet.d2=norm(z);jet.cap=std::min(jet.d1.lo,jet.d2.lo);
        for (int k=0;k<4;++k) if (k!=j)
            jet.cap=std::min(jet.cap,norm(sub(z,{B(roots[k].x),B(roots[k].y)})).lo);
        jet.cap=math::down(jet.cap*.25);
        jet.mapped=sub(sub(sub(z,scale({u.x,-u.y},m1)),scale({v.x,-v.y},m2)),{shift,B(0)});
    }
    return cache;
}
} // namespace critical_fast_detail

// Same Rouché inequality as the endpoint-interval reference. The FilteredBall
// centre jets enclose the entire outward mass/origin parameter box. Jets are
// shared by the full-phase and point-phase proofs; every acceptance still
// requires a positive outward interval margin, disjoint pole-free disks, and
// the quartic's complete root count. This is not an unvalidated error filter.
inline CriticalPhaseEnclosure binary_critical_phase_enclosure_fast(double s,double qi,int index,int bins,
    const std::array<SourcePosition,4>& roots,double fraction=1)
{
    using namespace critical_fast_detail;
    CriticalPhaseEnclosure out;
    if (!critical_enclosure_detail::supported_arithmetic() || !std::isfinite(s) || !std::isfinite(qi) ||
        s==0 || qi==0 || index<0 || bins<1 || index>=bins || !(fraction>=0 && fraction<=1)) return out;
    for (const auto& z:roots) if (!(std::isfinite(z.x) && std::isfinite(z.y))) return out;
    qi=std::abs(qi);
    auto& cache=proposal_cache(s,qi,roots);
    const auto phase=critical_enclosure_detail::unit_phase(index,bins);
    const C w{ball(phase.x),ball(phase.y)};
    const I reach=I(2)*critical_enclosure_detail::pi()/I(bins)*I(fraction);
    for (int j=0;j<4;++j) {
        const auto& p=cache.jets[j];const I residual=norm(sub(p.kappa,w));
        if (!(p.derivative.lo>0 && p.cap>0) || !p.derivative.finite() || !residual.finite()) return out;
        const double r=std::min(p.cap,(I(2)*(residual+reach)/I(p.derivative.lo)).hi);
        if (!(r>0 && std::isfinite(r))) return out;
        const I d1=p.d1-I(r),d2=p.d2-I(r);
        if (!(d1.lo>0 && d2.lo>0)) return out;
        const I h=I(6)*(cache.m1/math::square(math::square(d1))+cache.m2/math::square(math::square(d2)));
        const I margin=I(p.derivative.lo)*I(r)-residual-reach-h*I(r)*I(r)/I(2);
        if (!(margin.lo>0) || !margin.finite()) return out;
        out.radii[j]=r;out.margins[j]=margin.lo;
        // |Df| <= 1+|kappa(c)|+|kappa'(c)|r+sup|kappa''|r^2/2
        // throughout this disk. This is a bound on its full physical map.
        const I delta=I(r)*(I(1)+I(norm(p.kappa).hi)+I(p.derivative.hi)*I(r)+h*I(r)*I(r)/I(2));
        const I x=interval(p.mapped.x)+I(-delta.hi,delta.hi),y=interval(p.mapped.y)+I(-delta.hi,delta.hi);
        if (!(x.finite() && y.finite())) return out;
        out.source_lower[j]={x.lo,y.lo};out.source_upper[j]={x.hi,y.hi};
    }
    out.resolved=true;return out;
}
} // namespace lcbinint::magnification
