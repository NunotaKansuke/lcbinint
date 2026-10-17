#pragma once

#include "lcbinint/magnification/algebraic_boundary/tangent/quartic_sturm.hpp"
#ifdef BOUNDARY_QUARTIC_COUNT_AUDIT
#include <cstdio>
#include <cstdlib>
#endif

namespace lcbinint::algebraic_boundary::tangent {

// Rees' real-quartic classification, evaluated only when its signs are
// separated from floating-point error. I,J follow Cremona & Fisher (2008):
// https://www.dpmms.cam.ac.uk/~taf1000/papers/quarticequiv.pdf
// Delta=4*I^3-J^2 is 27 times the usual discriminant. For Delta>0, four
// real roots require P<0 and D<0; otherwise there are no real roots.
// Delta<0 gives two real roots. Ambiguous signs retain Sturm. The separate
// definite-form test can prove zero real roots despite repeated complex roots.
inline int invariant_quartic_count(const std::array<double,5>& p) {
    double scale=0;
    for(double x:p) {
        if(!std::isfinite(x))return -1;
        scale=std::max(scale,std::fabs(x));
    }
    if(!(scale>0) || p[4]==0)return -1;
    // A cheap sufficient zero-root test before evaluating invariants.
    // Young's inequalities bound |b*x^3*y| by
    // |b|*(3*x^4+y^4)/4, and similarly for d*x*y^3.
    // A negative c*x^2*y^2 is bounded by |c|*(x^4+y^4)/2.
    // Strict positive remaining x^4/y^4 coefficients prove definiteness.
#ifndef BOUNDARY_QUARTIC_DISABLE_DEFINITE_TEST
    const double orientation=p[4]>0?1.0:-1.0;
    const double lead=orientation*p[4],tail=orientation*p[0];
    if(tail>0) {
        const double odd3=std::fabs(p[3]),odd1=std::fabs(p[1]);
        const double negative2=std::max(0.0,-orientation*p[2]);
        const double left=.75*odd3+.25*odd1+.5*negative2;
        const double right=.25*odd3+.75*odd1+.5*negative2;
        constexpr double guard=64*std::numeric_limits<double>::epsilon();
        const double floor=std::numeric_limits<double>::min();
        if(lead-left > guard*(lead+left)+floor &&
           tail-right > guard*(tail+right)+floor)return 0;
    }
#endif
    int exponent=0;
    std::frexp(scale,&exponent);
    if(exponent < -1022 || exponent > 1022)return -1;
    const double factor=std::ldexp(1.0,-exponent);
    std::array<double,5> c;
    for(int k=0;k<5;++k) {
        c[k]=p[k]*factor;
        // Normal power-of-two scaling is exact. Reject underflow/rounding
        // into the subnormal range rather than changing the polynomial.
        if(p[k]!=0 && !(std::fabs(c[k])>=std::numeric_limits<double>::min()))return -1;
    }
    const double a=c[4],b=c[3],cc=c[2],d=c[1],e=c[0];
    const double a2=a*a,b2=b*b,c2=cc*cc;
    const double i0=12*a*e,i1=-3*b*d,i2=c2;
    const double j0=72*a*cc*e,j1=9*b*cc*d,j2=-27*a*d*d;
    const double j3=-27*b2*e,j4=-2*c2*cc;
    const double I=(i0+i1)+i2;
    const double J=(((j0+j1)+j2)+j3)+j4;
    const double Is=std::fabs(i0)+std::fabs(i1)+std::fabs(i2);
    const double Js=std::fabs(j0)+std::fabs(j1)+std::fabs(j2)+std::fabs(j3)+std::fabs(j4);
    const double delta=4*I*I*I-J*J;
    // Forward errors are bounded by gamma_n times sums of absolute terms.
    // The invariant graphs have fewer than 128 rounding factors including
    // propagation through I^3,J^2 and their positive error scales. gamma_1024
    // safely dominates those factors and rounding of the bound itself.
    // The absolute floor dominates all gradual-underflow errors for these
    // bounded (|coefficient|<=1) graphs. No tolerance/geometry fit is used.
    constexpr double unit=0.5*std::numeric_limits<double>::epsilon();
    constexpr double gamma=1024*unit/(1-1024*unit);
    constexpr double floor=std::numeric_limits<double>::min();
    const double delta_error=gamma*(4*Is*Is*Is+Js*Js)+floor;
    if(delta < -delta_error)return 2;
    if(!(delta>delta_error))return -1;
    const double p0=8*a*cc,p1=-3*b2;
    const double P=p0+p1;
    const double p_error=gamma*(std::fabs(p0)+std::fabs(p1))+floor;
    if(P>p_error)return 0;
    const double d0=64*a2*a*e,d1=-16*a2*c2,d2=16*a*b2*cc;
    const double d3=-16*a2*b*d,d4=-3*b2*b2;
    const double D=(((d0+d1)+d2)+d3)+d4;
    const double d_error=gamma*(std::fabs(d0)+std::fabs(d1)+std::fabs(d2)+std::fabs(d3)+std::fabs(d4))+floor;
    if(D>d_error)return 0;
    if(P < -p_error && D < -d_error)return 4;
    return -1;
}

inline int boundary_quartic_count(const std::array<double,5>& p, bool* used_invariants=nullptr) {
#ifndef BOUNDARY_FRONTIER_STURM_ONLY
    const int direct=invariant_quartic_count(p);
    if(direct>=0) {
#ifdef BOUNDARY_QUARTIC_COUNT_AUDIT
        const auto reference=certify_quartic(p);
        if(reference.certified && reference.root_count!=direct) {
            std::fprintf(stderr,"QUARTIC_COUNT_DISAGREEMENT direct=%d sturm=%d tier=%d margin=%.17g coefficients=%.17g %.17g %.17g %.17g %.17g\n",
                direct,reference.root_count,reference.precision_tier,reference.sign_margin,p[0],p[1],p[2],p[3],p[4]);
        }
        if(!reference.certified)
            std::fprintf(stderr,"QUARTIC_COUNT_RECOVERY count=%d tier=%d margin=%.17g coefficients=%.17g %.17g %.17g %.17g %.17g\n",
                direct,reference.precision_tier,reference.sign_margin,p[0],p[1],p[2],p[3],p[4]);
#endif
        if(used_invariants)*used_invariants=true;
        return direct;
    }
#endif
    if(used_invariants)*used_invariants=false;
    const auto cert=certify_quartic(p);
    return cert.certified && !(cert.root_count&1)?cert.root_count:-1;
}

} // namespace lcbinint::algebraic_boundary::tangent
