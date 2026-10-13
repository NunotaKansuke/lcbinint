#pragma once

// Cheap, fail-open necessary-condition screens for the adaptive projective
// fold probe.  These checks do not promote an event: a possible event still
// goes through the incumbent __float128 P=P_t and D14 coincidence checks.

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "lcbinint/magnification/holonomic/radial_events.hpp"

namespace lcbinint::holonomic::adaptive_detail {

namespace projective_screen_detail {

struct Interval {
    double lo=0, hi=0;
};

inline double down(double x) {
    return std::nextafter(x,-std::numeric_limits<double>::infinity());
}
inline double up(double x) {
    return std::nextafter(x,std::numeric_limits<double>::infinity());
}
inline Interval point(double x) {
    return {x,x};
}
inline Interval add(Interval a,Interval b) {
    return {down(a.lo+b.lo),up(a.hi+b.hi)};
}
inline Interval neg(Interval a) { return {-a.hi,-a.lo}; }
inline Interval sub(Interval a,Interval b) { return add(a,neg(b)); }
inline Interval mul(Interval a,Interval b) {
    const double p[4]={a.lo*b.lo,a.lo*b.hi,a.hi*b.lo,a.hi*b.hi};
    const auto mm=std::minmax_element(p,p+4);
    return {down(*mm.first),up(*mm.second)};
}
inline Interval scale(Interval a,double x) {
    return mul(a,Interval{x,x});
}
inline Interval square(Interval a) {
    if(a.lo<=0 && a.hi>=0) return {0,up(std::max(a.lo*a.lo,a.hi*a.hi))};
    const double x=a.lo*a.lo,y=a.hi*a.hi;
    return {down(std::min(x,y)),up(std::max(x,y))};
}
inline Interval sqrt_interval(Interval a) {
    if(a.hi<0) return {1,0};
    const double l=std::sqrt(std::max(0.0,a.lo));
    const double h=std::sqrt(std::max(0.0,a.hi));
    return {l==0?0:down(l),up(h)};
}
inline bool finite(Interval a) {
    return std::isfinite(a.lo)&&std::isfinite(a.hi)&&a.lo<=a.hi;
}
inline bool excludes_zero(Interval a) { return a.hi<0 || a.lo>0; }
inline double magnitude(Interval a) {
    return std::max(std::fabs(a.lo),std::fabs(a.hi));
}

struct CubicFactor { Interval a,b,c; };

inline bool cubic_factors(const PrimaryFrame& pf,CubicFactor (&f)[2]) {
    const Interval a=point(pf.a),x=point(pf.X),m0=point(pf.m0);
    const Interval rho=point(pf.rho),ay=point(std::fabs(pf.Y));
    const Interval b2=mul(sub(rho,ay),add(rho,ay));
    if(!finite(b2)||b2.hi<0)return false;
    const Interval b=sqrt_interval({std::max(0.0,b2.lo),b2.hi});
    if(!finite(b))return false;
    for(int i=0;i<2;++i) {
        const Interval sb=i==0?neg(b):b;
        f[i].a=add(add(a,x),sb);
        f[i].b=sub(mul(a,add(x,sb)),point(1.0));
        f[i].c=neg(mul(a,m0));
        if(!finite(f[i].a)||!finite(f[i].b)||!finite(f[i].c))return false;
    }
    return true;
}

inline Interval cubic_value(const CubicFactor& f,double x) {
    const Interval xi{x,x};
    return add(mul(add(mul(add(xi,f.a),xi),f.b),xi),f.c);
}
inline Interval cubic_derivative(const CubicFactor& f,Interval x) {
    return add(add(scale(square(x),3.0),scale(mul(f.a,x),2.0)),f.b);
}

inline bool opposite_strict_signs(Interval a,Interval b) {
    return (a.hi<0 && b.lo>0)||(a.lo>0 && b.hi<0);
}

// Isolate a simple factor root near the existing double chart-event seed.
// The factor coefficients and every endpoint evaluation are outward widened.
// Failure is deliberately inconclusive: it never rejects the incumbent probe.
inline bool isolate_near_seed(const CubicFactor& f,double seed,
                              Interval& root,int& expansions) {
    if(!(seed>0.0)||!std::isfinite(seed))return false;
    const double c=seed;
    const double du=std::nextafter(seed,std::numeric_limits<double>::infinity())-seed;
    double width=std::max(8.0*du,
        8.0*std::numeric_limits<double>::epsilon()*(1.0+std::fabs(c)));
    const double width_limit=1e-6*(1.0+std::fabs(c));
    for(expansions=0;expansions<48 && width<=width_limit;++expansions,width*=2.0) {
        const double lo=std::max(0.0,down(c-width));
        const double hi=up(c+width);
        if(!(lo<hi)||!std::isfinite(lo)||!std::isfinite(hi))return false;
        const Interval fl=cubic_value(f,lo),fh=cubic_value(f,hi);
        const Interval deriv=cubic_derivative(f,{lo,hi});
        if(finite(fl)&&finite(fh)&&opposite_strict_signs(fl,fh)&&
           excludes_zero(deriv)) {
            root={lo,hi};
            return true;
        }
    }
    return false;
}

inline Interval p3_over_radius(Interval R,const PrimaryFrame& pf) {
    const Interval a=point(pf.a),m0=point(pf.m0),x=point(pf.X),
                   y=point(pf.Y),r2=square(R);
    const Interval A=add(mul(a,sub(m0,r2)),mul(x,r2));
    const Interval C=add(
        add(mul(R,add(sub(r2,point(1.0)),mul(a,x))),mul(x,r2)),
        add(neg(mul(a,m0)),mul(a,r2)));
    // For P(t;R)=p0+p1*t+p2*t^2+p3*t^3+p4*t^4,
    // p3 = 4 Y [R^2 C - A R(a+R)].
    return scale(mul(y,sub(mul(r2,C),mul(mul(A,R),add(a,R)))),4.0);
}

inline double reciprocal_coefficient_scale_upper(Interval R,
                                                  const PrimaryFrame& pf) {
    const Interval a=point(pf.a),m0=point(pf.m0),x=point(pf.X),
                   y=point(pf.Y),rho=point(pf.rho),r2=square(R);
    const Interval n0r=neg(mul(x,r2)),n0i=neg(mul(y,r2));
    const Interval n1r=mul(R,add(sub(r2,point(1.0)),mul(a,x)));
    const Interval n1i=mul(mul(R,a),y);
    const Interval n2r=mul(a,sub(m0,r2));
    const Interval c0r=add(add(n0r,n1r),n2r),c0i=add(n0i,n1i);
    const Interval c1r=scale(n0i,2.0),c1i=scale(sub(n2r,n0r),2.0);
    const Interval c2r=sub(sub(n1r,n0r),n2r),c2i=sub(n1i,n0i);
    const Interval dot00=add(square(c0r),square(c0i));
    const Interval dot01=add(mul(c0r,c1r),mul(c0i,c1i));
    const Interval dot02=add(mul(c0r,c2r),mul(c0i,c2i));
    const Interval dot11=add(square(c1r),square(c1i));
    const Interval dot12=add(mul(c1r,c2r),mul(c1i,c2i));
    const Interval dot22=add(square(c2r),square(c2i));
    const Interval rho2= square(rho);
    const Interval k=mul(rho2,r2),rm=sub(R,a),rp=add(R,a);
    const Interval bm=square(rm),bp=square(rp);
    const Interval p0=sub(mul(k,bm),dot00);
    const Interval p1=scale(dot01,-2.0);
    const Interval p2=sub(sub(mul(k,add(bm,bp)),dot11),scale(dot02,2.0));
    const Interval p3=scale(dot12,-2.0);
    const Interval p4=sub(mul(k,bp),dot22);
    const Interval coeffs[5]={p0,p1,p2,p3,p4};
    Interval sum=point(0.0);
    for(const Interval c:coeffs) {
        if(!finite(c))return std::numeric_limits<double>::infinity();
        sum=add(sum,{0.0,magnitude(c)});
    }
    return sum.hi;
}

inline bool outside_qf_projective_contact_gate(Interval p3,
                                                double coefficient_scale_hi) {
    if(!finite(p3)||!std::isfinite(coefficient_scale_hi)||
       !(coefficient_scale_hi>0.0))return false;
    const double threshold=up(8192.0*(double)FLT128_EPSILON*
                              coefficient_scale_hi);
    return p3.lo>threshold||p3.hi< -threshold;
}

} // namespace projective_screen_detail

enum class ProjectiveFastScreen {
    Disabled,
    Unresolved,
    NoD14Overlap,
    NoReciprocalContact,
    Possible
};

inline const char* projective_fast_screen_name(ProjectiveFastScreen r) {
    switch(r) {
        case ProjectiveFastScreen::Disabled: return "disabled";
        case ProjectiveFastScreen::Unresolved: return "unresolved_fail_open";
        case ProjectiveFastScreen::NoD14Overlap: return "no_d14_overlap";
        case ProjectiveFastScreen::NoReciprocalContact: return "p3_outside_qf_contact_tolerance";
        case ProjectiveFastScreen::Possible: return "possible_fail_open";
    }
    return "unknown";
}

struct ProjectiveFastScreenResult {
    ProjectiveFastScreen result=ProjectiveFastScreen::Unresolved;
    double radius_lo=std::numeric_limits<double>::quiet_NaN();
    double radius_hi=std::numeric_limits<double>::quiet_NaN();
    double p3_lo=std::numeric_limits<double>::quiet_NaN();
    double p3_hi=std::numeric_limits<double>::quiet_NaN();
    double p3_gate_hi=std::numeric_limits<double>::quiet_NaN();
    double nearest_d14_gap=std::numeric_limits<double>::infinity();
    double overlap_budget=0.0;
    int factor_roots=0;
    int expansions=0;
};

inline ProjectiveFastScreenResult projective_p4_fast_screen(
    const RadialEvent& chart_event,const std::vector<RadialEvent>& events,
    const PrimaryFrame& pf) {
    using namespace projective_screen_detail;
    ProjectiveFastScreenResult out;
    if(chart_event.kind!="chart_p4"||!(chart_event.radius>0.0))return out;

    CubicFactor factors[2];
    if(!cubic_factors(pf,factors))return out;
    Interval chart_root{0,0};
    bool have_root=false;
    for(const auto& factor:factors) {
        Interval root;
        int expansions=0;
        if(!isolate_near_seed(factor,chart_event.radius,root,expansions))continue;
        ++out.factor_roots;
        out.expansions=std::max(out.expansions,expansions);
        if(!have_root){chart_root=root;have_root=true;}
        else {chart_root.lo=std::min(chart_root.lo,root.lo);
              chart_root.hi=std::max(chart_root.hi,root.hi);}
    }
    if(!have_root||!finite(chart_root))return out;
    out.radius_lo=(double)chart_root.lo;
    out.radius_hi=(double)chart_root.hi;

    // Match the incumbent D14 radius policy, but use the entire chart-root
    // enclosure instead of a rounded center.  The D14 uncertainty itself is
    // existing metadata, not a formal interval certificate; add binary64
    // representation padding and fail open whenever that metadata is absent.
    double nearest_gap=std::numeric_limits<double>::infinity();
    double nearest_budget=0;
    bool has_usable_candidate=false,has_overlapping_candidate=false;
    for(const auto& candidate:events) {
        if((candidate.kind!="physical_real"&&candidate.kind!="physical_complex")||
           candidate.detail!="D14 real root"||candidate.precision_tier<2||
           !std::isfinite(candidate.radius)||candidate.radius_uncertainty<0.0||
           !std::isfinite(candidate.radius_lo)||
           !std::isfinite(candidate.radius_uncertainty))continue;
        has_usable_candidate=true;
        const Interval center=add(point(candidate.radius),point(candidate.radius_lo));
        const double gap=center.hi<chart_root.lo?chart_root.lo-center.hi:
                         (center.lo>chart_root.hi?center.lo-chart_root.hi:0.0);
        const double center_scale=1.0+std::max(std::fabs(center.lo),std::fabs(center.hi));
        const double chart_pad=(chart_root.hi-chart_root.lo)/2.0;
        const double representation_pad=16.0*std::numeric_limits<double>::epsilon()*center_scale;
        const double budget=8.0*(candidate.radius_uncertainty+chart_pad+representation_pad)+
            512.0*(double)FLT128_EPSILON*center_scale;
        if(gap<=budget)has_overlapping_candidate=true;
        if(gap<nearest_gap){nearest_gap=gap;nearest_budget=budget;}
    }
    out.nearest_d14_gap=(double)nearest_gap;
    out.overlap_budget=(double)nearest_budget;
    if(!has_usable_candidate||!has_overlapping_candidate) {
        out.result=ProjectiveFastScreen::NoD14Overlap;
        return out;
    }

    const double radius_pad=16.0*std::numeric_limits<double>::epsilon()*
        (1.0+std::max(std::fabs(chart_root.lo),std::fabs(chart_root.hi)));
    const Interval p3_radius{down(chart_root.lo-radius_pad),
                             up(chart_root.hi+radius_pad)};
    const Interval p3=p3_over_radius(p3_radius,pf);
    const double qscale=reciprocal_coefficient_scale_upper(p3_radius,pf);
    if(!finite(p3)||!std::isfinite(qscale))return out;
    out.p3_lo=(double)p3.lo;
    out.p3_hi=(double)p3.hi;
    const double p3_gate=up(8192.0*(double)FLT128_EPSILON*qscale);
    out.p3_gate_hi=p3_gate;
    // The incumbent qf contact certificate accepts |Q_u(0)|/sum|Q_k| <=
    // 8192 eps.  Reject only when the outward interval is wholly outside that
    // same tolerance; a simple p3!=0 test would be stricter than the incumbent.
    if(outside_qf_projective_contact_gate(p3,qscale)) {
        out.result=ProjectiveFastScreen::NoReciprocalContact;
        return out;
    }
    out.result=ProjectiveFastScreen::Possible;
    return out;
}

} // namespace lcbinint::holonomic::adaptive_detail
