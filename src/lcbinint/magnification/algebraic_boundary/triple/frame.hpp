#pragma once
#include "geometry.hpp"
namespace lcbinint::algebraic_boundary::triple {
// Public physical coordinates: source_x,source_y,rho,m1,m2,
// lens0_x,lens0_y,lens1_x,lens1_y,lens2_x,lens2_y; m0=1-m1-m2.
// Positions may be translated/rotated arbitrarily; the primary is lens 0.
template<bool Gradient> struct Frame;
template<> struct Frame<false> {Geometry geometry;};
template<> struct Frame<true> {
    Geometry geometry;
    std::array<std::array<double,11>,8> jacobian{};
    std::array<double,11> pullback(const std::array<double,8>& gradient) const {
        std::array<double,11> out{};
        for(int i=0;i<8;++i)for(int j=0;j<11;++j)out[j]+=gradient[i]*jacobian[i][j];
        return out;
    }
};
template<bool Gradient=false> inline Frame<Gradient> primary_frame(
    const std::array<Complex,3>& lenses,Complex source,double rho,double m1,double m2) {
    for(auto z:lenses)if(!std::isfinite(z.real()) || !std::isfinite(z.imag()))
        throw std::invalid_argument("nonfinite triple position");
    const Complex d=lenses[1]-lenses[0];const double a=std::abs(d);
    if(!(a>0))throw std::invalid_argument("coincident primary and secondary");
    const Complex c=std::conj(d)/a,u=source-lenses[0],v=lenses[2]-lenses[0];
    const Complex canonical_source=c*u,canonical_third=c*v;
    Frame<Gradient> out;
    out.geometry={canonical_source.real(),canonical_source.imag(),rho,m1,m2,a,
                  canonical_third.real(),canonical_third.imag()};
    out.geometry.validate();
    if constexpr(Gradient)for(int j=0;j<11;++j) {
        Complex ds{},dl0{},dl1{},dl2{};
        if(j<2)ds=j?Complex(0,1):Complex(1,0);
        else if(j==5 || j==6)dl0=j==5?Complex(1,0):Complex(0,1);
        else if(j==7 || j==8)dl1=j==7?Complex(1,0):Complex(0,1);
        else if(j==9 || j==10)dl2=j==9?Complex(1,0):Complex(0,1);
        const Complex dd=dl1-dl0;
        const double da=std::real(std::conj(d)*dd)/a;
        const Complex dc=std::conj(dd)/a-c*(da/a);
        const Complex du=c*(ds-dl0)+dc*u,dv=c*(dl2-dl0)+dc*v;
        const std::array<double,8> column{du.real(),du.imag(),j==2?1.:0.,
            j==3?1.:0.,j==4?1.:0.,da,dv.real(),dv.imag()};
        for(int i=0;i<8;++i)out.jacobian[i][j]=column[i];
    }
    return out;
}
// Chain the physical-coordinate gradient to the existing lens parameters.
// Output order: q,s,q2,sep2,ang. Source coordinates are held fixed here.
inline std::array<double,5> lens_parameter_pullback(const std::array<double,11>& g,
    double s,double q,double q2,double sep2,double ang,bool vbm) {
    const double total=1+q+q2;
    const double dq1=(1+q2)/(total*total),dq2=-q2/(total*total);
    const double d21=-q/(total*total),d22=(1+q)/(total*total);
    std::array<double,5> out{g[3]*dq1+g[4]*dq2,0,g[3]*d21+g[4]*d22,0,0};
    std::array<std::array<Complex,3>,5> dz{};
    const Complex phase(std::cos(ang),std::sin(ang));
    if(vbm) {
        const double qshift=s/((1+q)*(1+q));
        dz[0]={qshift,qshift,qshift};
        dz[1]={q/(1+q),-1/(1+q),q/(1+q)};
        dz[3][2]=-phase;dz[4][2]=Complex(0,-1)*sep2*phase;
    } else {
        const double pair=q+q2;
        const Complex dq=-s/(total*total)-q2/(pair*pair)*sep2*phase;
        const Complex dthird=-s/(total*total)+q/(pair*pair)*sep2*phase;
        dz[0]={-s/(total*total),dq,dq};
        dz[1]={-pair/total,1/total,1/total};
        dz[2]={-s/(total*total),dthird,dthird};
        dz[3]={Complex{},q2/pair*phase,-q/pair*phase};
        dz[4]={Complex{},Complex(0,1)*q2/pair*sep2*phase,Complex(0,-1)*q/pair*sep2*phase};
    }
    for(int k=0;k<5;++k)for(int j=0;j<3;++j)
        out[k]+=g[5+2*j]*dz[k][j].real()+g[6+2*j]*dz[k][j].imag();
    return out;
}
} // namespace lcbinint::algebraic_boundary::triple
