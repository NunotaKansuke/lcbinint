#pragma once
#include "../d14_real_sturm.hpp"
#include <cstdint>
#include <cstring>
namespace lcbinint::algebraic_boundary::triple::interval_detail {
// The same outward neighbours as nextafter, without libm calls in every
// binary64 primitive. No arithmetic is performed on a NaN representation.
inline double neighbour(double x,bool upward) {
    static_assert(sizeof(double)==sizeof(std::uint64_t) &&
                  std::numeric_limits<double>::is_iec559,"requires IEEE binary64");
    std::uint64_t u;std::memcpy(&u,&x,sizeof(u));
    const auto mag=u&UINT64_C(0x7fffffffffffffff);
    if(mag>UINT64_C(0x7ff0000000000000))return x;
    if(u==(upward?UINT64_C(0x7ff0000000000000):UINT64_C(0xfff0000000000000)))return x;
    if(mag==0)u=upward?UINT64_C(1):UINT64_C(0x8000000000000001);
    else if(bool(u>>63)==upward)--u;
    else ++u;
    std::memcpy(&x,&u,sizeof(x));return x;
}
template<class T> inline T up(T x){return positive_detail::up(x);}
template<class T> inline T down(T x){return positive_detail::down(x);}
template<> inline double up(double x){return neighbour(x,true);}
template<> inline double down(double x){return neighbour(x,false);}
inline __float128 neighbour(__float128 x,bool upward) {
    static_assert(sizeof(__float128)==16 && __FLT128_MANT_DIG__==113,
                  "requires IEEE binary128");
    using U=unsigned __int128;
    constexpr U sign=U(1)<<127,inf=U(0x7fff)<<112;
    U u;std::memcpy(&u,&x,sizeof(u));const U magnitude=u&~sign;
    if(magnitude>inf || u==(upward?inf:sign|inf))return x;
    if(magnitude==0)u=upward?U(1):sign|U(1);
    else if(bool(u&sign)==upward)--u;
    else ++u;
    std::memcpy(&x,&u,sizeof(x));return x;
}
template<> inline __float128 up(__float128 x){return neighbour(x,true);}
template<> inline __float128 down(__float128 x){return neighbour(x,false);}
inline bool environment_ok(){return positive_detail::environment_ok();}
template<class T> struct Interval {
    T lo=0,hi=0;
    Interval()=default;Interval(double x):lo(x),hi(x){}
    Interval(T l,T h):lo(l),hi(h){}
    bool zero() const{return lo==0 && hi==0;}
    bool valid() const{return lo<=hi && lo>-T(INFINITY) && hi<T(INFINITY);}
    int sign() const{return !valid()?2:lo>0?1:hi<0?-1:zero()?0:2;}
};
template<class T> inline Interval<T> operator-(Interval<T> a){return {-a.hi,-a.lo};}
template<class T> inline Interval<T> operator+(Interval<T> a,Interval<T> b){
    if(a.zero())return b;
    if(b.zero())return a;
    return {down(a.lo+b.lo),up(a.hi+b.hi)};
}
template<class T> inline Interval<T> operator-(Interval<T> a,Interval<T> b){return a+(-b);}
template<class T> inline Interval<T> operator*(Interval<T> a,Interval<T> b){
    if(a.zero()||b.zero())return {};
    T l,h;
    if(a.lo>=0){
        if(b.lo>=0){l=a.lo*b.lo;h=a.hi*b.hi;}
        else if(b.hi<=0){l=a.hi*b.lo;h=a.lo*b.hi;}
        else{l=a.hi*b.lo;h=a.hi*b.hi;}
    }else if(a.hi<=0){
        if(b.lo>=0){l=a.lo*b.hi;h=a.hi*b.lo;}
        else if(b.hi<=0){l=a.hi*b.hi;h=a.lo*b.lo;}
        else{l=a.lo*b.hi;h=a.lo*b.lo;}
    }else{
        if(b.lo>=0){l=a.lo*b.hi;h=a.hi*b.hi;}
        else if(b.hi<=0){l=a.hi*b.lo;h=a.lo*b.lo;}
        else{l=std::min(a.lo*b.hi,a.hi*b.lo);h=std::max(a.lo*b.lo,a.hi*b.hi);}
    }
    return {down(l),up(h)};
}
}
