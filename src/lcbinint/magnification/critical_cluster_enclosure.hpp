#pragma once
#include "lcbinint/magnification/critical_phase_enclosure.hpp"
#include <vector>

namespace lcbinint::magnification {
struct CriticalClusterTube {
    SourcePosition centre,source_lower,source_upper;
    double radius=0;
    int roots=0;
};
// Near a topology transition a degree-two dominant Taylor term can enclose
// the merging roots, where a simple-root derivative certificate cannot.
inline std::vector<CriticalClusterTube> binary_critical_cluster_enclosure(
    double s,double qi,int index,int bins,const std::array<SourcePosition,4>& proposals,bool two_pairs=false)
{
    using namespace critical_enclosure_detail;
    if (s==0 || qi==0 || !std::isfinite(s) || !std::isfinite(qi) ||
        bins<1 || index<0 || index>=bins ||
        !supported_arithmetic()) return {};
    qi=std::abs(qi);
    const I q=qi<1 ? I(qi) : I(1)/I(qi),m1=I(1)/(I(1)+q),m2=q*m1;
    const double a=qi<1 ? -std::abs(s) : std::abs(s);
    const double reach=(I(2)*pi()/I(bins)).hi;
    const C w=add(unit_phase(index,bins),{I(-reach,reach),I(-reach,reach)});
    int first=0,second=1;double best=std::numeric_limits<double>::infinity();
    for (int j=0;j<4;++j) for (int k=0;k<j;++k) {
        const double d=std::hypot(proposals[j].x-proposals[k].x,proposals[j].y-proposals[k].y);
        if (d<best) {best=d;first=k;second=j;}
    }
    std::vector<std::pair<SourcePosition,int>> groups;
    groups.push_back({{.5*(proposals[first].x+proposals[second].x),
                       .5*(proposals[first].y+proposals[second].y)},2});
    if (two_pairs) {
        std::vector<int> remaining;
        for (int j=0;j<4;++j) if (j!=first && j!=second) remaining.push_back(j);
        groups.push_back({{.5*(proposals[remaining[0]].x+proposals[remaining[1]].x),
                           .5*(proposals[remaining[0]].y+proposals[remaining[1]].y)},2});
    } else for (int j=0;j<4;++j) if (j!=first && j!=second) groups.push_back({proposals[j],1});
    const auto fail=[&]() {
        return two_pairs ? std::vector<CriticalClusterTube>{} :
            binary_critical_cluster_enclosure(s,qi,index,bins,proposals,true);
    };
    std::vector<CriticalClusterTube> result;
    for (const auto& group:groups) {
        const C c{I(group.first.x),I(group.first.y)},ca=subtract(c,{I(a),I(0)});
        const C c2=multiply(c,c),ca2=multiply(ca,ca);
        const C p0=subtract(subtract(multiply(w,multiply(c2,ca2)),scale(c2,m1)),scale(ca2,m2));
        const C p1=subtract(multiply(w,scale(multiply(multiply(c,ca),add(c,ca)),I(2))),scale(add(scale(c,m1),scale(ca,m2)),I(2)));
        const C p2=subtract(multiply(w,add(add(c2,ca2),scale(multiply(c,ca),I(4)))),{m1+m2,I(0)});
        const C p3=scale(multiply(w,add(c,ca)),I(2));
        const std::array<I,5> coefficients={magnitude(p0),magnitude(p1),magnitude(p2),magnitude(p3),magnitude(w)};
        const double cap=math::down(.45*std::min(magnitude(c).lo,magnitude(ca).lo));
        if (!(cap>0) || !std::isfinite(cap)) return fail();
        double chosen=0;
        for (int exponent=24;exponent>=0;--exponent) {
            const double r=std::ldexp(cap,-exponent);
            I bound(0),power(1),dominant;
            for (int degree=0;degree<=4;++degree) {
                if (degree==group.second) dominant=I(coefficients[degree].lo)*power;
                else bound=bound+I(coefficients[degree].hi)*power;
                power=power*I(r);
            }
            if ((dominant-bound).lo>0) {chosen=r;break;}
        }
        if (!(chosen>0)) return fail();
        const C z{c.x+I(-chosen,chosen),c.y+I(-chosen,chosen)};
        const C u=reciprocal(subtract(z,{I(a),I(0)})),v=reciprocal(z);
        const C mapped=subtract(subtract(subtract(z,scale({u.x,-u.y},m1)),scale({v.x,-v.y},m2)),{I(a)*m1,I(0)});
        if (!mapped.x.finite() || !mapped.y.finite()) return fail();
        for (const auto& other:result) {
            const double gap=magnitude({I(group.first.x)-I(other.centre.x),I(group.first.y)-I(other.centre.y)}).lo;
            if (!(gap>(I(chosen)+I(other.radius)).hi)) return fail();
        }
        result.push_back({group.first,{mapped.x.lo,mapped.y.lo},{mapped.x.hi,mapped.y.hi},chosen,group.second});
    }
    // Disjoint Rouché disks containing 2+1+1 or 2+2 roots exhaust the quartic.
    return result;
}
} // namespace lcbinint::magnification
