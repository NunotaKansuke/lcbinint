#pragma once
#include <limits>
#include <cmath>
#include "interval.hpp"
#if defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize ("no-fast-math","fp-contract=off")
#endif
namespace lcbinint::algebraic_boundary::triple::coefficient_ball_detail {
// Round-to-nearest error balls for support-coefficient formation only.
// epsilon conservatively bounds centre rounding. Positive radius expressions
// are inflated for their own rounding; min() covers gradual underflow.
// Overflow and NaN become an unbounded enclosure, never a certificate.
template<class T> struct Ball {
    using Scalar=T;
    T value=0,error=0;
    Ball()=default;
    Ball(double x):value(x){}
    Ball(T v,T e):value(v),error(e){}
    bool zero()const{return value==0 && error==0;}
    auto interval()const {
        using I=interval_detail::Interval<double>;
        if(!std::isfinite(value) || !std::isfinite(error) || error<0)return I(-INFINITY,INFINITY);
        constexpr T epsilon=std::numeric_limits<T>::epsilon();
        const T pad=(T(4)*epsilon*(std::abs(value)+error)+std::numeric_limits<T>::min())*(T(1)+T(16)*epsilon);
        const T lower=value-error-pad,upper=value+error+pad;
        double lo=double(lower),hi=double(upper);
        if(T(lo)>lower)lo=std::nextafter(lo,-INFINITY);
        if(T(hi)<upper)hi=std::nextafter(hi,INFINITY);
        return I(lo,hi);
    }
};
template<class T> Ball<T> operator-(Ball<T>a){return {-a.value,a.error};}
template<class T> Ball<T> operator+(Ball<T>a,Ball<T>b) {
    if(a.zero())return b;
    if(b.zero())return a;
    constexpr T epsilon=std::numeric_limits<T>::epsilon();
    const T value=a.value+b.value;
    const T radius=(a.error+b.error+epsilon*(std::abs(a.value)+std::abs(b.value))+
        std::numeric_limits<T>::min())*(T(1)+T(16)*epsilon);
    return {value,radius};
}
template<class T> Ball<T> operator-(Ball<T>a,Ball<T>b){return a+(-b);}
template<class T> Ball<T> operator*(Ball<T>a,Ball<T>b) {
    if(a.zero() || b.zero())return {};
    if(a.error==0 && a.value==1)return b;
    if(b.error==0 && b.value==1)return a;
    constexpr T epsilon=std::numeric_limits<T>::epsilon();
    const T value=a.value*b.value;
    const T radius=(std::abs(a.value)*b.error+std::abs(b.value)*a.error+a.error*b.error+
        epsilon*std::abs(value)+std::numeric_limits<T>::min())*(T(1)+T(32)*epsilon);
    return {value,radius};
}
}
#if defined(__GNUC__)
#pragma GCC pop_options
#endif
