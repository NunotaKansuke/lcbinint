#pragma once
#include <array>
#include <algorithm>
#include <cmath>

namespace lcbinint::magnification::detail {
struct RetainedRowCore {
    double left=0,right=0,spacing=0,trapezoid_area=0;
    double derivative1=0,derivative3=0,uncertainty=0;
};
inline std::array<double,5> retained_core_derivatives(const std::array<double,7>& v,double h) {
    const double d1=(v[6]-9*v[5]+45*v[4]-45*v[2]+9*v[1]-v[0])/(60*h);
    const double d1low=(-v[5]+8*v[4]-8*v[2]+v[1])/(12*h);
    const double d3=(v[0]-8*v[1]+13*v[2]-13*v[4]+8*v[5]-v[6])/(8*h*h*h);
    const double d3low=(v[5]-2*v[4]+2*v[2]-v[1])/(2*h*h*h);
    const double d5=(-v[0]+4*v[1]-5*v[2]+5*v[4]-4*v[5]+v[6])/(2*h*h*h*h*h);
    return {d1,d3,std::abs(d1-d1low),std::abs(d3-d3low),d5};
}
inline std::array<double,2> retained_core_fine_equivalent(const RetainedRowCore& c,double spacing) {
    const double h2=c.spacing*c.spacing,f2=spacing*spacing;
    return {c.trapezoid_area-(h2-f2)*c.derivative1/12+
        (h2*h2-f2*f2)*c.derivative3/720,c.uncertainty};
}
// Four-point cubic midpoint from six retained equidistant samples. The two
// fourth differences estimate the first omitted term and its local variation.
// This is a local smoothness model, not a bound for arbitrary lens maps.
inline std::array<double,2> retained_row_midpoint(const std::array<double,6>& v) {
    const double midpoint=(-v[1]+9*v[2]+9*v[3]-v[4])/16;
    const double a=v[0]-4*v[1]+6*v[2]-4*v[3]+v[4];
    const double b=v[1]-4*v[2]+6*v[3]-4*v[4]+v[5];
    const double error=(3./128)*(std::max(std::abs(a),std::abs(b))+std::abs(a-b));
    return {midpoint,error};
}
}
