#pragma once

#include "lcbinint/math/outward_interval.hpp"
#include "lcbinint/types.hpp"
#include <array>
#include <cfenv>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace lcbinint::magnification {
namespace critical_enclosure_detail {
using I=math::OutwardInterval;
inline bool supported_arithmetic() {
#ifdef __FAST_MATH__
    return false;
#endif
    volatile double normal=std::numeric_limits<double>::min();
    volatile double subnormal=std::numeric_limits<double>::denorm_min();
    return std::numeric_limits<double>::is_iec559 && std::fegetround()==FE_TONEAREST &&
           normal*.5!=0 && subnormal+subnormal!=0;
}
struct C { I x,y; };
inline C add(C a,C b) { return {a.x+b.x,a.y+b.y}; }
inline C subtract(C a,C b) { return {a.x-b.x,a.y-b.y}; }
inline C multiply(C a,C b) { return {a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x}; }
inline C scale(C a,I b) { return {a.x*b,a.y*b}; }
inline C reciprocal(C a) {
    const I norm=math::square(a.x)+math::square(a.y);
    return {a.x/norm,-a.y/norm};
}
inline I magnitude(C a) {
    const I norm=math::square(a.x)+math::square(a.y);
    return {std::max(0.0,math::down(std::sqrt(std::max(0.0,norm.lo)))),
            math::up(std::sqrt(norm.hi))};
}
// This binary64 bracket contains mathematical pi. Trigonometric enclosures
// use finite Taylor sums and alternating-series remainders, not libm sin/cos.
inline I pi() { return {3.141592653589793,3.1415926535897936}; }
inline C unit_phase(int index,int bins) {
    // The rigorous phase enclosure depends only on the rational phase, not
    // on lens geometry or source. Preserve it across moving-lens epochs too.
    using Key=std::uint64_t;
    static thread_local std::unordered_map<Key,C> cache;
    const Key key=(Key(static_cast<std::uint32_t>(bins))<<32)|static_cast<std::uint32_t>(index);
    const auto found=cache.find(key);
    if (found!=cache.end()) return found->second;
    const int signed_index=index>bins/2 ? index-bins : index;
    const I angle=I(2)*pi()*I(signed_index)/I(bins), square=math::square(angle);
    I ct(1),st=angle,cosine=ct,sine=st;
    for (int n=1;n<=18;++n) {
        ct=-ct*square/I((2*n-1)*(2*n));
        st=-st*square/I((2*n)*(2*n+1));
        cosine=cosine+ct;sine=sine+st;
    }
    const double cr=(ct*square/I(37*38)).magnitude();
    const double sr=(st*square/I(38*39)).magnitude();
    const C result{cosine+I(-cr,cr),sine+I(-sr,sr)};
    if (cache.size()<65536) cache.emplace(key,result);
    return result;
}
} // namespace critical_enclosure_detail

struct CriticalPhaseEnclosure {
    bool resolved=false;
    std::array<double,4> radii{};
    // Positive margins certify four disjoint, pole-free root disks over the
    // entire phase interval. A failed interval is explicitly unresolved.
    std::array<double,4> margins{};
    std::array<SourcePosition,4> source_lower{},source_upper{};
};

inline CriticalPhaseEnclosure binary_critical_phase_enclosure(
    double separation,double mass_ratio,int phase_index,int bins,
    const std::array<SourcePosition,4>& roots,double phase_fraction=1.0)
{
    using namespace critical_enclosure_detail;
    CriticalPhaseEnclosure result;
    if (!supported_arithmetic())
        return result;
    if (!std::isfinite(separation) || !std::isfinite(mass_ratio) ||
        separation==0 || mass_ratio==0 || bins<1 || phase_index<0 || phase_index>=bins ||
        !(phase_fraction>=0 && phase_fraction<=1)) return result;
    const double qi=std::abs(mass_ratio);
    const I q=qi<1 ? I(qi) : I(1)/I(qi);
    const I m1=I(1)/(I(1)+q),m2=q*m1;
    const double a=qi<1 ? -std::abs(separation) : std::abs(separation);
    const C phase=unit_phase(phase_index,bins);
    const I phase_range=I(2)*pi()/I(bins)*I(phase_fraction);
    for (std::size_t j=0;j<4;++j) {
        if (!std::isfinite(roots[j].x) || !std::isfinite(roots[j].y)) return result;
        const C z{I(roots[j].x),I(roots[j].y)},za{subtract(z,{I(a),I(0)})};
        const C u=reciprocal(za),v=reciprocal(z);
        const C p=scale(multiply(u,u),m1),t=scale(multiply(v,v),m2);
        const C derivative=scale(add(multiply(p,u),multiply(t,v)),I(-2));
        const I slope=magnitude(derivative),residual=magnitude(subtract(add(p,t),phase));
        double cap=std::min(magnitude(z).lo,magnitude(za).lo);
        for (std::size_t k=0;k<4;++k) if (k!=j)
            cap=std::min(cap,magnitude(subtract(z,{I(roots[k].x),I(roots[k].y)})).lo);
        cap=math::down(cap*0.25);
        if (!(slope.lo>0 && cap>0) || !slope.finite() || !residual.finite()) return result;
        const double r=std::min(cap,(I(2)*(residual+phase_range)/I(slope.lo)).hi);
        if (!(r>0 && std::isfinite(r))) return result;
        const I d1=magnitude(za)-I(r),d2=magnitude(z)-I(r);
        if (!(d1.lo>0 && d2.lo>0)) return result;
        const I h=I(6)*(m1/math::square(math::square(d1))+
                       m2/math::square(math::square(d2)));
        const I margin=I(slope.lo)*I(r)-residual-phase_range-h*I(r)*I(r)/I(2);
        result.radii[j]=r;result.margins[j]=margin.lo;
        if (!(margin.lo>0) || !margin.finite()) return result;
    }
    for (std::size_t j=0;j<4;++j) {
        const double r=result.radii[j];
        const C z{I(roots[j].x)+I(-r,r),I(roots[j].y)+I(-r,r)};
        const C u=reciprocal(subtract(z,{I(a),I(0)})),v=reciprocal(z);
        const C conjugate_u{u.x,-u.y},conjugate_v{v.x,-v.y};
        const C mapped=subtract(subtract(subtract(z,scale(conjugate_u,m1)),
                                        scale(conjugate_v,m2)),{I(a)*m1,I(0)});
        if (!mapped.x.finite() || !mapped.y.finite()) return result;
        // Bound the physical disk using the map Jacobian norm 1+|kappa|.
        // This retains cancellation between the two lens deflections, which
        // separate rectangular reciprocal ranges otherwise lose.
        const C centre{I(roots[j].x),I(roots[j].y)};
        const C cu=reciprocal(subtract(centre,{I(a),I(0)})),cv=reciprocal(centre);
        const C cp=scale(multiply(cu,cu),m1),ct=scale(multiply(cv,cv),m2);
        const C kd=scale(add(multiply(cp,cu),multiply(ct,cv)),I(-2));
        const I d1=magnitude(subtract(centre,{I(a),I(0)}))-I(r),d2=magnitude(centre)-I(r);
        const I h2=I(6)*(m1/math::square(math::square(d1))+m2/math::square(math::square(d2)));
        const I delta=I(r)*(I(1)+I(magnitude(add(cp,ct)).hi)+I(magnitude(kd).hi)*I(r)+h2*I(r)*I(r)/I(2));
        const C fc=subtract(subtract(subtract(centre,scale({cu.x,-cu.y},m1)),
                                    scale({cv.x,-cv.y},m2)),{I(a)*m1,I(0)});
        const I x=fc.x+I(-delta.hi,delta.hi),y=fc.y+I(-delta.hi,delta.hi);
        result.source_lower[j]={std::max(mapped.x.lo,x.lo),std::max(mapped.y.lo,y.lo)};
        result.source_upper[j]={std::min(mapped.x.hi,x.hi),std::min(mapped.y.hi,y.hi)};
    }
    // Rouché gives one simple root in each disk for every unit phase in the
    // interval. Four disjoint disks exhaust the degree-four numerator.
    result.resolved=true;
    return result;
}
} // namespace lcbinint::magnification
