#pragma once
#include "geometry.hpp"
#include "interval.hpp"
namespace lcbinint::algebraic_boundary::triple::surface_detail {
template<class T> struct RadialPolynomialT {
    using Scalar=interval_detail::Interval<T>;
    std::array<Scalar,9> c{};int degree=0;
    RadialPolynomialT()=default;
    RadialPolynomialT(double x){c[0]=Scalar(x);}
};
template<class T> RadialPolynomialT<T> operator-(RadialPolynomialT<T> a){for(auto& c:a.c)c=-c;return a;}
template<class T> RadialPolynomialT<T> operator+(RadialPolynomialT<T> a,const RadialPolynomialT<T>& b){
    a.degree=std::max(a.degree,b.degree);for(int j=0;j<=b.degree;++j)a.c[j]=a.c[j]+b.c[j];return a;
}
template<class T> RadialPolynomialT<T> operator-(RadialPolynomialT<T> a,RadialPolynomialT<T> b){return a+(-b);}
template<class T> RadialPolynomialT<T> operator*(const RadialPolynomialT<T>& a,const RadialPolynomialT<T>& b){
    RadialPolynomialT<T> out;out.degree=a.degree+b.degree;
    if(out.degree>8)throw std::logic_error("triple radial polynomial degree overflow");
    for(int i=0;i<=a.degree;++i)for(int j=0;j<=b.degree;++j)out.c[i+j]=out.c[i+j]+a.c[i]*b.c[j];
    return out;
}
using RadialPolynomial=RadialPolynomialT<double>;
}
