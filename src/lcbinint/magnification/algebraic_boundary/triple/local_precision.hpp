#pragma once
#include "geometry.hpp"
#include "physical_jet.hpp"
#include <quadmath.h>
namespace lcbinint::algebraic_boundary::triple {
namespace local_precision {
using Q=__float128;
using C=coefficient_detail::C<Q>;
inline C inverse(C z){Q s=z.re*z.re+z.im*z.im;return {z.re/s,-z.im/s};}
inline Q norm(C z){return z.re*z.re+z.im*z.im;}
struct Value {Q phi=0,dtheta=0;};
struct Gradient : Value {Q dR=0;std::array<Q,8> dp{};};
template<bool Derivatives,bool NeedTheta=Derivatives> auto evaluate(Q R,Q theta,const Geometry& p) {
    using Result=std::conditional_t<Derivatives,Gradient,Value>;
    const C z{R*cosq(theta),R*sinq(theta)};
    const C zb=coefficient_detail::conj(z);
    const std::array<C,3> lens{C{},C{Q(p.a)},C{Q(p.bx),Q(p.by)}};
    const std::array<Q,3> mass{Q(1)-Q(p.m1)-Q(p.m2),Q(p.m1),Q(p.m2)};
    C g=z-C{Q(p.X),Q(p.Y)},shear{};
    std::array<C,3> inv{};
    for(int j=0;j<3;++j){inv[j]=inverse(zb-coefficient_detail::conj(lens[j]));g=g-C{mass[j]}*inv[j];if constexpr(NeedTheta)shear=shear+C{mass[j]}*inv[j]*inv[j];}
    const Q rho2=Q(p.rho)*Q(p.rho);
    auto derivative=[&](C dg){return -Q(2)*(g.re*dg.re+g.im*dg.im)/rho2;};
    Result out;out.phi=1-norm(g)/rho2;
    if constexpr(NeedTheta)out.dtheta=derivative(C{0,1}*(z-zb*shear));
    if constexpr(Derivatives)out.dR=derivative(C{Q(1)/R}*(z+zb*shear));
    if constexpr(Derivatives)out.dp={derivative(C{-1}),derivative(C{0,-1}),
        Q(2)*norm(g)/(rho2*Q(p.rho)),derivative(inv[0]-inv[1]),derivative(inv[0]-inv[2]),
        derivative(C{-mass[1]}*inv[1]*inv[1]),derivative(C{-mass[2]}*inv[2]*inv[2]),
        derivative(C{0,mass[2]}*inv[2]*inv[2])};
    return out;
}
// Value-only high-precision fallback. A radius owns many endpoint Newton
// iterations and samples: retain the exact promoted geometry and primary term.
// The general evaluator above remains the derivative and reference route.
class ValueRay {
    Q radius_,radial_,rho2_;
    C source_;
    std::array<C,2> lenses_;
    std::array<Q,2> masses_;
public:
    ValueRay(double radius,const Geometry& p):radius_(radius),
        radial_(Q(radius)-(Q(1)-Q(p.m1)-Q(p.m2))/Q(radius)),
        rho2_(Q(p.rho)*Q(p.rho)),source_{Q(p.X),Q(p.Y)},
        lenses_{C{Q(p.a)},C{Q(p.bx),Q(p.by)}},masses_{Q(p.m1),Q(p.m2)} {}
    template<bool NeedTheta=false> Value evaluate(Q theta) const {
        Q sn,cs;sincosq(theta,&sn,&cs);
        const C z{radius_*cs,radius_*sn};
        C g{radial_*cs-source_.re,radial_*sn-source_.im},shear{};
        for(int j=0;j<2;++j) {
            const C inv=inverse(coefficient_detail::conj(z-lenses_[j]));
            // Avoid software-quad multiplications by the identically zero
            // imaginary part of a real mass.
            g.re-=masses_[j]*inv.re;g.im-=masses_[j]*inv.im;
            if constexpr(NeedTheta) {
                shear.re+=masses_[j]*(inv.re*inv.re-inv.im*inv.im);
                shear.im+=masses_[j]*(Q(2)*inv.re*inv.im);
            }
        }
        Value out;out.phi=1-norm(g)/rho2_;
        if constexpr(NeedTheta) {
            // The primary contribution is already in radial_.
            const C h{radial_*cs-(z.re*shear.re+z.im*shear.im),
                      radial_*sn-(z.re*shear.im-z.im*shear.re)};
            // Multiplication by i is exactly a component swap and negation.
            out.dtheta=-Q(2)*(-g.re*h.im+g.im*h.re)/rho2_;
        }
        return out;
    }
};
// Fixed-radius derivative kernel. Keep the general three-lens evaluator above
// as the independent reference; fold the central lens into radial constants.
class GradientRay {
    Q radius_,inverse_radius_,radial_,radial_derivative_,rho_,rho2_,derivative_scale_;
    C source_;
    std::array<C,2> lenses_;
    std::array<Q,2> masses_;
public:
    GradientRay(double radius,const Geometry& p):radius_(radius),
        inverse_radius_(Q(1)/Q(radius)),
        radial_(Q(radius)-(Q(1)-Q(p.m1)-Q(p.m2))/Q(radius)),
        radial_derivative_(Q(1)+(Q(1)-Q(p.m1)-Q(p.m2))/(Q(radius)*Q(radius))),
        rho_(p.rho),rho2_(Q(p.rho)*Q(p.rho)),derivative_scale_(-Q(2)/rho2_),source_{Q(p.X),Q(p.Y)},
        lenses_{C{Q(p.a)},C{Q(p.bx),Q(p.by)}},masses_{Q(p.m1),Q(p.m2)} {}
    Gradient evaluate(Q theta) const {
        Q sn,cs;sincosq(theta,&sn,&cs);
        const C unit{cs,sn},z{radius_*cs,radius_*sn},zb=coefficient_detail::conj(z);
        C g{radial_*cs-source_.re,radial_*sn-source_.im},shear{};
        std::array<C,2> inv{},square{};
        for(int j=0;j<2;++j) {
            inv[j]=inverse(zb-coefficient_detail::conj(lenses_[j]));
            square[j]=inv[j]*inv[j];
            g.re-=masses_[j]*inv[j].re;g.im-=masses_[j]*inv[j].im;
            shear.re+=masses_[j]*square[j].re;shear.im+=masses_[j]*square[j].im;
        }
        // One adjoint of phi supplies all first-order directions. Preserve
        // the original division in phi itself to avoid cancellation changes.
        const C adjoint{derivative_scale_*g.re,derivative_scale_*g.im};
        const auto derivative=[&](C dg){return adjoint.re*dg.re+adjoint.im*dg.im;};
        const C secondary=zb*shear;
        const C angular{radial_*cs-secondary.re,radial_*sn-secondary.im};
        Gradient out;const Q n=norm(g);out.phi=1-n/rho2_;
        out.dtheta=derivative(C{-angular.im,angular.re});
        out.dR=derivative(C{radial_derivative_*cs+inverse_radius_*secondary.re,
                              radial_derivative_*sn+inverse_radius_*secondary.im});
        const C primary{inverse_radius_*unit.re,inverse_radius_*unit.im};
        out.dp={-adjoint.re,-adjoint.im,Q(2)*n/(rho2_*rho_),
            derivative(primary-inv[0]),derivative(primary-inv[1]),
            derivative(C{-masses_[0]*square[0].re,-masses_[0]*square[0].im}),
            derivative(C{-masses_[1]*square[1].re,-masses_[1]*square[1].im}),
            derivative(C{-masses_[1]*square[1].im,masses_[1]*square[1].re})};
        return out;
    }
};
template<class Evaluate> bool refine_endpoint(double theta,const Geometry& p,Q& refined,Evaluate eval) {
    refined=theta;
    for(int j=0;j<20;++j) {
        const auto v=eval(refined);
        if(v.dtheta==0 || !finiteq(v.phi) || !finiteq(v.dtheta))return false;
        const Q step=v.phi/v.dtheta;refined-=step;
        // Keep the incumbent quad arithmetic floor below the binary64
        // angular enclosure already accounted for by the caller.
        const Q floor=128*ldexpq(Q(1),-112)*(1+fabsq(refined))/
            (Q(p.rho)*Q(p.rho)*fabsq(v.dtheta));
        if(fabsq(step)<Q(1e-29) || (floor<Q(1e-18) && fabsq(step)<floor))return true;
        if(fabsq(step)>Q(.1))return false;
    }
    return false;
}
inline bool endpoint(const ValueRay& ray,double theta,const Geometry& p,Q& refined) {
    return refine_endpoint(theta,p,refined,[&](Q t){return ray.evaluate<true>(t);});
}
inline bool endpoint(double R,double theta,const Geometry& p,Q& refined) {
    // Reuse the rational physical ray in the endpoint Newton iteration.
    // This avoids software-quad sine/cosine on every Newton step. The
    // original angular refinement remains the fallback, warm-started from
    // the retained rational proposal when it is local.
    const double tangent=std::tan(theta*.5);
    const bool chart=std::abs(tangent)>1;
    Q t=chart?-Q(1)/Q(tangent):Q(tangent);
    const physical_detail::Ray<Q> ray(Q(R),p);
    static const Q pi=acosq(Q(-1));
    bool converged=false;
    for(int k=0;k<16;++k) {
        const auto jet=ray.angular(t,chart);
        if(!jet || jet->t==0 || !finiteq(jet->v) || !finiteq(jet->t))break;
        const Q step=jet->v/jet->t;
        if(!finiteq(step) || fabsq(step)>Q(.1))break;
        t-=step;
        const Q floor=128*ldexpq(Q(1),-112)*(1+fabsq(t))/fabsq(jet->t);
        if(fabsq(step)<Q(1e-29) || (floor<Q(1e-18) && fabsq(step)<floor)){converged=true;break;}
    }
    Q proposal=(chart?pi:Q(0))+2*atanq(t);
    if(finiteq(proposal)) {
        proposal+=2*pi*roundq((Q(theta)-proposal)/(2*pi));
        if(fabsq(proposal-Q(theta))<Q(.1)) {
            refined=proposal;
            if(converged)return true;
            return refine_endpoint(double(proposal),p,refined,[&](Q angle){return evaluate<false,true>(Q(R),angle,p);});
        }
    }
    return refine_endpoint(theta,p,refined,[&](Q angle){return evaluate<false,true>(Q(R),angle,p);});
}
template<bool Gradient> struct Arc;
template<> struct Arc<false> {Q lo=0,hi=0;bool ready=false;};
template<> struct Arc<true> {
    Q lo=0,hi=0;bool ready=false;std::array<Q,8> dlo{},dhi{};
};
}
} // namespace lcbinint::algebraic_boundary::triple
