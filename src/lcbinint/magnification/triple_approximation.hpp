#pragma once
#include "point_source_magnifier.hpp"
#include <array>
#include <cmath>
#include <complex>
#include <limits>
namespace lcbinint::magnification::triple_approximation {
// Physical coordinates match the integral pullback:
// source x,y,rho,m1,m2,lens0 x,y,lens1 x,y,lens2 x,y; m0=1-m1-m2.
template<bool Gradient> struct PointPacket { double mu=0; int images=0; bool valid=true; };
template<> struct PointPacket<true> {
    double mu=0; int images=0; bool valid=true,gradient_valid=true; std::array<double,11> gradient{};
};
template<bool Gradient> PointPacket<Gradient> point(
        const PointSourceMagnifier& solver,const model::TripleLensGeometry& geometry,
        SourcePosition source,int expected_images=0,bool continue_branches=false) {
    PointPacket<Gradient> out;
    // The value solve and the implicit tangent consume the same validated
    // roots. No polynomial is differentiated and no second root solve occurs.
    const auto& images=continue_branches?solver.triple_image_candidates_continued_cached(geometry,source,expected_images):
        solver.triple_image_candidates_cached(geometry,source,expected_images);
    for(const auto& image:images)if(image.physical) {
        out.mu+=1/std::abs(image.jacobian_determinant);++out.images;
        if constexpr(Gradient) {
            using C=std::complex<double>;
            std::array<C,3> v,v2,v3; C k{},third{};
            for(int j=0;j<3;++j) {
                v[j]=1./std::conj(triple_image_separation(geometry,image,j));
                v2[j]=v[j]*v[j];v3[j]=v2[j]*v[j];
                k+=geometry.masses[j]*v2[j];third+=geometry.masses[j]*v3[j];
            }
            const double det=1-std::norm(k);
            if(!std::isfinite(det)||det==0){out.gradient_valid=false;continue;}
            for(int j=0;j<11;++j) {
                if(j==2)continue;
                C b{},dk{};
                if(j<2)b=j==0?C(1,0):C(0,1);
                else if(j==3||j==4) {
                    const int lens=j-2;b=v[lens]-v[0];dk=v2[lens]-v2[0];
                } else {
                    const int lens=(j-5)/2;const C dl=(j-5)%2?C(0,-1):C(1,0);
                    b=geometry.masses[lens]*dl*v2[lens];
                    dk=2.*geometry.masses[lens]*dl*v3[lens];
                }
                const C dz=(b-k*std::conj(b))/det;
                dk-=2.*third*std::conj(dz);
                out.gradient[j]+=2*std::copysign(1.,det)*std::real(std::conj(k)*dk)/(det*det);
            }
        }
    }
    out.valid=out.valid&&std::isfinite(out.mu)&&out.mu>0&&
        (expected_images<=0 || out.images==expected_images);
    if constexpr(Gradient) {
        for(double g:out.gradient)out.gradient_valid=out.gradient_valid&&std::isfinite(g);
        if(!out.gradient_valid)out.gradient.fill(std::numeric_limits<double>::quiet_NaN());
    }
    return out;
}
template<bool Gradient> struct HexPacket:PointPacket<Gradient> {
    double error=std::numeric_limits<double>::infinity();
};
template<bool Gradient> HexPacket<Gradient> hex(
        const PointSourceMagnifier& solver,const model::TripleLensGeometry& geometry,
        SourcePosition source,double rho,double gamma,int expected_images) {
    const auto center=point<Gradient>(solver,geometry,source,expected_images);
    PointPacket<Gradient> axial,half,diagonal;
    constexpr double h=.7071067811865475244;
    const std::array<SourcePosition,4> axes{{{1,0},{0,1},{-1,0},{0,-1}}};
    const std::array<SourcePosition,4> diagonals{{{h,h},{-h,h},{-h,-h},{h,-h}}};
    const auto add=[&](auto& sum,SourcePosition direction,double scale) {
        auto p=point<Gradient>(solver,geometry,{source.x+rho*scale*direction.x,
            source.y+rho*scale*direction.y},expected_images,true);
        sum.mu+=p.mu;sum.valid=sum.valid&&p.valid&&p.images==expected_images;
        if constexpr(Gradient) {
            p.gradient[2]+=scale*(direction.x*p.gradient[0]+direction.y*p.gradient[1]);
            for(int j=0;j<11;++j)sum.gradient[j]+=p.gradient[j];
        }
    };
    for(int i=0;i<4;++i){add(axial,axes[i],1);add(half,axes[i],.5);add(diagonal,diagonals[i],1);}
    const auto combine=[&](double a,double b,double c,double d) {
        const double plus=b/4-a,small=c/4-a,cross=d/4-a;
        const double second=(16*small-plus)/3,fourth=(plus+cross)/2-second;
        return std::pair<double,double>{a+.5*second*(1-.2*gamma)+fourth/3*(1-11*gamma/35),
            fourth/3*(1-11*gamma/35)};
    };
    HexPacket<Gradient> out;const auto value=combine(center.mu,axial.mu,half.mu,diagonal.mu);
    out.mu=value.first;out.error=std::abs(value.second);
    out.valid=center.valid&&center.images==expected_images&&axial.valid&&half.valid&&diagonal.valid&&std::isfinite(out.mu)&&out.mu>0;
    if constexpr(Gradient)for(int j=0;j<11;++j) {
        out.gradient[j]=combine(center.gradient[j],axial.gradient[j],half.gradient[j],diagonal.gradient[j]).first;
        out.gradient_valid=out.gradient_valid&&std::isfinite(out.gradient[j]);
    }
    return out;
}
} // namespace lcbinint::magnification::triple_approximation
