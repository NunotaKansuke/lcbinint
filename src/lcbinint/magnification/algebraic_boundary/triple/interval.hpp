#pragma once
#include "../d14_real_sturm.hpp"
#include <cstdint>
#include <cstring>
#if defined(__SSE2__)
#include <emmintrin.h>
#endif
namespace lcbinint::algebraic_boundary::triple::interval_detail {
using rounding::neighbour;
template<class T> inline T up(T x){return positive_detail::up(x);}
template<class T> inline T down(T x){return positive_detail::down(x);}
template<> inline double up(double x){return neighbour(x,true);}
template<> inline double down(double x){return neighbour(x,false);}
template<> inline __float128 up(__float128 x){return neighbour(x,true);}
template<> inline __float128 down(__float128 x){return neighbour(x,false);}
template<> inline long double up(long double x){return std::nextafter(x,std::numeric_limits<long double>::infinity());}
template<> inline long double down(long double x){return std::nextafter(x,-std::numeric_limits<long double>::infinity());}
inline bool environment_ok(){return positive_detail::environment_ok();}
template<class T> struct Interval {
    T lo=0,hi=0;
    Interval()=default;Interval(double x):lo(x),hi(x){}
    Interval(T l,T h):lo(l),hi(h){}
    bool zero() const{return lo==0 && hi==0;}
    bool valid() const{return lo<=hi && lo>-T(INFINITY) && hi<T(INFINITY);}
    int sign() const{return !valid()?2:lo>0?1:hi<0?-1:zero()?0:2;}
};
#if defined(__SSE2__)
// Both lanes are finite. In the lower lane we store the negated lower bound,
// so a step toward +infinity rounds both interval endpoints outward.
inline __m128d packed_up(__m128d value) {
    const __m128i one=_mm_set1_epi64x(1),bits=_mm_castpd_si128(value);
    const __m128i increment=_mm_sub_epi64(one,_mm_slli_epi64(_mm_srli_epi64(bits,63),1));
    const __m128d zero=_mm_cmpeq_pd(value,_mm_setzero_pd());
    return _mm_or_pd(_mm_and_pd(zero,_mm_castsi128_pd(one)),
        _mm_andnot_pd(zero,_mm_castsi128_pd(_mm_add_epi64(bits,increment))));
}
inline __m128d signed_endpoints(Interval<double> a) {
    return _mm_set_pd(a.hi,-a.lo);
}
inline Interval<double> endpoints(__m128d signed_bounds) {
    return {-_mm_cvtsd_f64(signed_bounds),
            _mm_cvtsd_f64(_mm_unpackhi_pd(signed_bounds,signed_bounds))};
}
#endif
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
