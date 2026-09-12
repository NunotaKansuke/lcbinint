#pragma once
// Local certificate for the binary64 quartic used by the incumbent cold
// cross-check. This does not certify coefficient construction from physics.
#include <array>
#include <algorithm>
#include <cmath>
#include <cfenv>
#include <cstdint>
#include <cstring>
#include <limits>
namespace lcbinint::holonomic::local_bracket_detail {
static_assert(sizeof(double)==sizeof(std::uint64_t) && std::numeric_limits<double>::digits==53, "binary64 interval kernel required");
struct Interval { double lo,hi; bool ok=true; };
inline bool regular(double x){
    // Inspect bits: an FP comparison can see a subnormal input as zero
    // under DAZ, which must not silently change the polynomial certified.
    std::uint64_t bits;std::memcpy(&bits,&x,sizeof bits);
    bits&=0x7fffffffffffffffULL;
    const auto exponent=bits&0x7ff0000000000000ULL;
    return bits==0 || (exponent!=0 && exponent!=0x7ff0000000000000ULL);
}
inline double down(double x){
    const double v=std::nextafter(x,-std::numeric_limits<double>::infinity());
    return std::fabs(v)<std::numeric_limits<double>::min() ? -std::numeric_limits<double>::min():v;
}
inline double up(double x){
    const double v=std::nextafter(x,std::numeric_limits<double>::infinity());
    return std::fabs(v)<std::numeric_limits<double>::min() ? std::numeric_limits<double>::min():v;
}
inline Interval point(double x){return {x,x,regular(x)};}
inline Interval add(Interval a,Interval b){
    double l=a.lo+b.lo,h=a.hi+b.hi;
    bool ok=a.ok&&b.ok&&regular(l)&&regular(h)&&
        (l!=0||a.lo==-b.lo)&&(h!=0||a.hi==-b.hi);
    return {down(l),up(h),ok};
}
inline Interval mul(Interval a,Interval b){
    double lo=std::numeric_limits<double>::infinity(),hi=-lo;
    bool ok=a.ok&&b.ok;
    for(double x:{a.lo,a.hi})for(double y:{b.lo,b.hi}){
        const double p=x*y;
        ok=ok&&regular(p)&&(p!=0||x==0||y==0);
        lo=std::min(lo,p);hi=std::max(hi,p);
    }
    return {down(lo),up(hi),ok};
}
inline Interval value(const std::array<double,5>& p,Interval x){
    Interval y=point(p[4]);
    for(int k=3;k>=0;--k)y=add(mul(y,x),point(p[k]));
    return y;
}
inline Interval derivative(const std::array<double,5>& p,Interval x){
    Interval y=mul(point(4),point(p[4]));
    for(int k=3;k>=1;--k)y=add(mul(y,x),mul(point(k),point(p[k])));
    return y;
}
inline int sign(Interval a){return !a.ok?0:a.lo>0?1:a.hi<0?-1:0;}
// Disjoint intervals, opposite endpoint signs, and derivative of constant
// sign prove one distinct real root per interval. Bound theta error using
// |d(2 atan(t))/dt| <= 2, with ample margin inside the incumbent 1e-7 gate.
inline bool certify(const std::array<double,5>& p,
                    const std::array<double,4>& roots,int n){
#ifdef __FAST_MATH__
    return false;
#endif
    if((n!=2&&n!=4)||std::fegetround()!=FE_TONEAREST)return false;
    for(double c:p)if(!regular(c))return false;
    double previous_hi=-std::numeric_limits<double>::infinity();
    for(int i=0;i<n;++i){
        const double t=roots[i];if(!regular(t))return false;
        double radius=1e-9; // nominal radius; rounded offsets checked below
        if(i)radius=std::min(radius,0.125*(t-roots[i-1]));
        if(i+1<n)radius=std::min(radius,0.125*(roots[i+1]-t));
        double lo=t-radius,hi=t+radius;
        // The incumbent compares wrapped theta numerically, not circular
        // distance. Do not certify across its theta=0/2pi discontinuity.
        if(lo<=0.0&&hi>=0.0)return false;
        if(!(lo<t&&t<hi&&previous_hi<lo)||
           !regular(lo)||!regular(hi)||
           !(t-lo<2e-9&&hi-t<2e-9))return false;
        if(sign(value(p,point(lo)))*sign(value(p,point(hi)))!=-1)return false;
        if(!sign(derivative(p,{lo,hi,true})))return false;
        previous_hi=hi;
    }
    return true;
}
}
