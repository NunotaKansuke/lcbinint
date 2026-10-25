#pragma once
#include "geometry.hpp"
#include "interval.hpp"
#include <optional>
#include <algorithm>
namespace lcbinint::algebraic_boundary::triple::physical_detail {
template<class T> struct Jet {
    T v{},r{},t{},rt{},tt{};
    Jet()=default;
    Jet(double x):v(x) {}
    explicit Jet(T x,T R,T S):v(x),r(R),t(S) {}
};
template<class T> Jet<T> operator+(Jet<T> a,Jet<T> b) {
    Jet<T> c;c.v=a.v+b.v;c.r=a.r+b.r;c.t=a.t+b.t;
    c.rt=a.rt+b.rt;c.tt=a.tt+b.tt;return c;
}
template<class T> Jet<T> operator-(Jet<T> a) {
    Jet<T> c;c.v=-a.v;c.r=-a.r;c.t=-a.t;c.rt=-a.rt;c.tt=-a.tt;return c;
}
template<class T> Jet<T> operator-(Jet<T> a,Jet<T> b){return a+(-b);}
template<class T> Jet<T> operator*(Jet<T> a,Jet<T> b) {
    Jet<T> c;c.v=a.v*b.v;c.r=a.r*b.v+a.v*b.r;c.t=a.t*b.v+a.v*b.t;
    c.rt=a.rt*b.v+a.r*b.t+a.t*b.r+a.v*b.rt;
    c.tt=a.tt*b.v+T(2)*a.t*b.t+a.v*b.tt;return c;
}
template<class T> T square_value(T x){return x*x;}
template<class T> interval_detail::Interval<T> square_value(interval_detail::Interval<T> x) {
    if(x.lo>=0)return {x.lo==0?T(0):interval_detail::down(x.lo*x.lo),interval_detail::up(x.hi*x.hi)};
    if(x.hi<=0)return {x.hi==0?T(0):interval_detail::down(x.hi*x.hi),interval_detail::up(x.lo*x.lo)};
    return {T(0),interval_detail::up(std::max(x.lo*x.lo,x.hi*x.hi))};
}
template<class T> Jet<T> square_jet(Jet<T> a) {
    Jet<T> c;c.v=square_value(a.v);c.r=T(2)*a.v*a.r;c.t=T(2)*a.v*a.t;
    c.rt=T(2)*(a.v*a.rt+a.r*a.t);
    c.tt=T(2)*(a.v*a.tt+square_value(a.t));return c;
}
template<class T> bool positive_value(T x){return x>0;}
template<class T> bool positive_value(interval_detail::Interval<T> x){return x.valid() && x.lo>0;}
template<class T> T reciprocal_value(T x){return T(1)/x;}
template<class T> interval_detail::Interval<T> reciprocal_value(interval_detail::Interval<T> x){
    return {interval_detail::down(T(1)/x.hi),interval_detail::up(T(1)/x.lo)};
}
template<class T> Jet<T> reciprocal_jet(Jet<T> a) {
    Jet<T> out;out.v=reciprocal_value(a.v);const T square=out.v*out.v,cube=square*out.v;
    out.r=-a.r*square;out.t=-a.t*square;
    out.rt=T(2)*a.r*a.t*cube-a.rt*square;
    out.tt=T(2)*a.t*a.t*cube-a.tt*square;return out;
}
template<class T> std::optional<Jet<T>> physical(T radius,T coordinate,const Geometry& p,bool reciprocal_chart) {
    using J=Jet<T>;using C=coefficient_detail::C<J>;
    const J R(radius,T(1),T(0)),t(coordinate,T(0),T(1)),one(1);
    const J t_squared=square_jet(t),den=one+t_squared;
    if(!positive_value(den.v) || !positive_value(radius))return std::nullopt;
    const J inverse=reciprocal_jet(den),sign(reciprocal_chart?-1:1);
    const C unit{sign*(one-t_squared)*inverse,sign*J(2)*t*inverse};
    const C z=C{R}*unit;
    // At a fixed polar radius, z-m0/conj(z)=(R-m0/R)*unit.
    // This identity also holds for the R/t jets and their interval extensions.
    const J radial=R-(J(1)-J(p.m1)-J(p.m2))*reciprocal_jet(R);
    C g=C{radial}*unit-C{J(p.X),J(p.Y)};
    const std::array<C,3> lenses{C{},C{J(p.a)},C{J(p.bx),J(p.by)}};
    const std::array<J,3> masses{J(1)-J(p.m1)-J(p.m2),J(p.m1),J(p.m2)};
    for(int j=1;j<3;++j) {
        const C d=coefficient_detail::conj(z-lenses[j]);
        const J norm=square_jet(d.re)+square_jet(d.im);
        if(!positive_value(norm.v))return std::nullopt;
        const J inv=reciprocal_jet(norm);
        g=g-C{masses[j]}*coefficient_detail::conj(d)*C{inv};
    }
    return J(p.rho)*J(p.rho)-square_jet(g.re)-square_jet(g.im);
}
// Real scaling has no cross terms. In interval arithmetic, evaluating a
// generic complex product also evaluates identically zero cross terms.
template<class T> coefficient_detail::C<T> real_scale(coefficient_detail::C<T> z,T scale) {
    return {z.re*scale,z.im*scale};
}
template<class T> struct AngularEvaluation {T v,t;};
// A radial quadrature node uses the same primary term and source/lenses
// for every Newton iterate and every endpoint proof. Prepare them once.
template<class T> class Ray {
    using C=coefficient_detail::C<T>;
    static T orient(T value,bool chart){return chart?-value:value;}
    T radius_,radial_{};
    C source_;
    std::array<C,2> lenses_;
    std::array<T,2> masses_;
    bool valid_;
public:
    Ray(T radius,const Geometry& p):radius_(radius),source_{T(p.X),T(p.Y)},
        lenses_{C{T(p.a)},C{T(p.bx),T(p.by)}},masses_{T(p.m1),T(p.m2)},valid_(positive_value(radius)),rho_squared_(T(p.rho)*T(p.rho)) {
        if(valid_)radial_=radius-(T(1)-T(p.m1)-T(p.m2))*reciprocal_value(radius);
    }
    std::optional<AngularEvaluation<T>> angular(T coordinate,bool chart) const {
        const T one(1),t2=square_value(coordinate),den=one+t2;
        if(!valid_ || !positive_value(den))return std::nullopt;
        const T inv=reciprocal_value(den),inv2=inv*inv;
        const C unit{orient((one-t2)*inv,chart),orient(T(2)*coordinate*inv,chart)};
        const C z=real_scale(unit,radius_);
        const C tangent{orient(T(-4)*coordinate*inv2,chart),orient(T(2)*(one-t2)*inv2,chart)};
        C g=real_scale(unit,radial_)-source_,shear{};
        for(int j=0;j<2;++j) {
            const C d=coefficient_detail::conj(z-lenses_[j]);const T norm=square_value(d.re)+square_value(d.im);
            if(!positive_value(norm))return std::nullopt;
            const C inverse=real_scale(coefficient_detail::conj(d),reciprocal_value(norm));
            g=g-real_scale(inverse,masses_[j]);shear=shear+real_scale(inverse*inverse,masses_[j]);
        }
        const C dg=real_scale(tangent,radial_)+coefficient_detail::conj(real_scale(tangent,radius_))*shear;
        return AngularEvaluation<T>{rho_squared_-square_value(g.re)-square_value(g.im),T(-2)*(g.re*dg.re+g.im*dg.im)};
    }
    // Spatial event derivatives use the lens-map inverse powers already
    // needed for F and F_t. Build higher derivatives only if exclusion fails.
    std::optional<Jet<T>> event_jet(T coordinate,bool chart,bool exclude_first=false) const {
        const T one(1),t2=square_value(coordinate),den=one+t2;
        if(!valid_ || !positive_value(den))return std::nullopt;
        const T inv=reciprocal_value(den),inv2=inv*inv;
        const C unit{orient((one-t2)*inv,chart),orient(T(2)*coordinate*inv,chart)};
        const C tangent{orient(T(-4)*coordinate*inv2,chart),orient(T(2)*(one-t2)*inv2,chart)};
        const C z=real_scale(unit,radius_);
        C g=real_scale(unit,radial_)-source_,shear{};
        std::array<C,2> inverses;
        for(int j=0;j<2;++j) {
            const C d=coefficient_detail::conj(z-lenses_[j]);const T norm=square_value(d.re)+square_value(d.im);
            if(!positive_value(norm))return std::nullopt;
            const C v=real_scale(coefficient_detail::conj(d),reciprocal_value(norm));
            inverses[j]=v;g=g-real_scale(v,masses_[j]);shear=shear+real_scale(v*v,masses_[j]);
        }
        const C gt=real_scale(tangent,radial_)+coefficient_detail::conj(real_scale(tangent,radius_))*shear;
        Jet<T> out;out.v=rho_squared_-square_value(g.re)-square_value(g.im);
        out.t=T(-2)*(g.re*gt.re+g.im*gt.im);
        if constexpr(std::is_same_v<T,interval_detail::Interval<double>>) {
            if(exclude_first && (std::abs(out.v.sign())==1 || std::abs(out.t.sign())==1))return out;
        }
        C third{};for(int j=0;j<2;++j)third=third+real_scale(inverses[j]*inverses[j]*inverses[j],masses_[j]);
        const T inv3=inv2*inv;
        const C second{orient(T(-4)*(one-T(3)*t2)*inv3,chart),orient(T(-4)*coordinate*(T(3)-t2)*inv3,chart)};
        const T radial_r=one+(one-masses_[0]-masses_[1])*reciprocal_value(square_value(radius_));
        const C gr=real_scale(unit,radial_r)+coefficient_detail::conj(unit)*shear;
        const C grt=real_scale(tangent,radial_r)+coefficient_detail::conj(tangent)*shear-
            real_scale(coefficient_detail::conj(unit)*coefficient_detail::conj(tangent)*third,T(2)*radius_);
        const C gtt=real_scale(second,radial_)+coefficient_detail::conj(real_scale(second,radius_))*shear-
            real_scale(coefficient_detail::conj(tangent)*coefficient_detail::conj(tangent)*third,T(2)*radius_*radius_);
        out.r=T(-2)*(g.re*gr.re+g.im*gr.im);
        out.rt=T(-2)*(gr.re*gt.re+gr.im*gt.im+g.re*grt.re+g.im*grt.im);
        out.tt=T(-2)*(square_value(gt.re)+square_value(gt.im)+g.re*gtt.re+g.im*gtt.im);
        return out;
    }
    std::optional<T> value(T coordinate,bool chart) const {
        const T one(1),t2=square_value(coordinate),den=one+t2;
        if(!valid_ || !positive_value(den))return std::nullopt;
        const T inv=reciprocal_value(den);
        const C unit{orient((one-t2)*inv,chart),orient(T(2)*coordinate*inv,chart)};
        const C z=real_scale(unit,radius_);
        C g=real_scale(unit,radial_)-source_;
        for(int j=0;j<2;++j) {
            const C d=coefficient_detail::conj(z-lenses_[j]);const T norm=square_value(d.re)+square_value(d.im);
            if(!positive_value(norm))return std::nullopt;
            g=g-real_scale(coefficient_detail::conj(d),masses_[j]*reciprocal_value(norm));
        }
        return rho_squared_-square_value(g.re)-square_value(g.im);
    }
private:
    T rho_squared_;
};
template<class T> std::optional<AngularEvaluation<T>> angular(T radius,T coordinate,const Geometry& p,bool chart) {
    return Ray<T>(radius,p).angular(coordinate,chart);
}
template<class T> std::optional<T> boundary_value(T radius,T coordinate,const Geometry& p,bool chart) {
    return Ray<T>(radius,p).value(coordinate,chart);
}

}
