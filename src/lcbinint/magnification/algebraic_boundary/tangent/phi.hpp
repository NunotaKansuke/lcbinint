#pragma once

// Algebraic-boundary solver -- phi and its closed-form gradient.
// Direct port of historical Python reference: jacobian.py:phi_grad.
//
//   phi = 1 - |f(z) - zeta|^2 / rho^2,  z = R e^{i theta}
//   f(z) = z - m0/zbar - m1/(zbar - a),  m1 = 1 - m0,  zeta = X + iY
//
// Derivative order matches jacobian.py: (dphi/dX, dphi/dY, dphi/drho,
// dphi/dm0, dphi/da).
//
// Complex arithmetic is hand-rolled on real/imag pairs -- no
// std::complex (its C99 Annex-G inf/nan bookkeeping on every divide is
// ~4x slower and never needed here: zbar and zbar-a are bounded away
// from 0 on the integration cells).  This is the hottest routine in the
// solver (tens of thousands of calls per epoch).
//
// M8: three entry points share one core so each caller pays only for the
// derivatives it uses.
//   phi_grad       -- {phi, dphi/dtheta, dP[5]}  (arc endpoints only)
//   phi_val_dP     -- {phi, dP[5]}               (angular sqrt-phi quadrature)
//   phi_val_dtheta -- {phi, dphi/dtheta}         (Newton endpoint polish)

#include <array>
#include <algorithm>
#include <cmath>
#include <limits>
#include <quadmath.h>
#include "fused_motion.hpp"

#include "lcbinint/magnification/algebraic_boundary/tangent/lens_frame.hpp"

namespace lcbinint::algebraic_boundary::tangent {

struct PhiGrad {
    double phi;
    double dphi_dtheta;
    std::array<double, 5> dP;  // d/d(X, Y, rho, m0, a)
};

struct PhiValDP {
    double phi;
    std::array<double, 5> dP;
};

struct PhiValDtheta {
    double phi;
    double dphi_dtheta;
};

namespace phi_detail {
struct C {
    double re, im;
};
inline C operator+(C a, C b) { return {a.re + b.re, a.im + b.im}; }
inline C operator-(C a, C b) { return {a.re - b.re, a.im - b.im}; }
inline C operator*(C a, C b) {
    return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}
inline C operator/(C a, C b) {
    double d = 1.0 / (b.re * b.re + b.im * b.im);
    return {(a.re * b.re + a.im * b.im) * d, (a.im * b.re - a.re * b.im) * d};
}
inline C conj(C a) { return {a.re, -a.im}; }
inline C muli(C a) { return {-a.im, a.re}; }  // i * a
inline double re_prod(C a, C b) {  // Re(a * b)
    return a.re * b.re - a.im * b.im;
}

// Shared core: everything up to (and including) g, g2, phi, and the two
// inverse-conjugate reciprocals needed by every downstream derivative.
struct Core {
    C z, zb, g;
    C inv_zb, inv_zba;
    double m1, r2, inv_r2, g2, phi;
};
inline Core core(double R, double theta, const PrimaryFrame& pf) {
    Core c;
    double ct, st;
    __builtin_sincos(theta, &st, &ct);
    c.z = C{R * ct, R * st};
    c.zb = C{c.z.re, -c.z.im};
    c.m1 = 1.0 - pf.m0;
    const C zba = c.zb - C{pf.a, 0.0};
    c.inv_zb = C{1.0, 0.0} / c.zb;
    c.inv_zba = C{1.0, 0.0} / zba;
    const C f =
        c.z - C{pf.m0, 0.0} * c.inv_zb - C{c.m1, 0.0} * c.inv_zba;
    c.g = f - C{pf.X, pf.Y};
    c.r2 = pf.rho * pf.rho;
    c.inv_r2 = 1.0 / c.r2;
    c.g2 = c.g.re * c.g.re + c.g.im * c.g.im;
    c.phi = 1.0 - c.g2 * c.inv_r2;
    return c;
}

inline double dphi_dtheta_from(const Core& c, const PrimaryFrame& pf) {
    const C inv_zb2 = c.inv_zb * c.inv_zb;
    const C inv_zba2 = c.inv_zba * c.inv_zba;
    const C fzb = C{pf.m0, 0.0} * inv_zb2 + C{c.m1, 0.0} * inv_zba2;
    const C dg_dtheta = muli(c.z) - muli(c.zb * fzb);
    return -(2.0 * c.inv_r2) * re_prod(conj(c.g), dg_dtheta);
}

inline std::array<double, 5> dP_from(const Core& c, const PrimaryFrame& pf) {
    const double dphi_dX = (2.0 * c.inv_r2) * c.g.re;
    const double dphi_dY = (2.0 * c.inv_r2) * c.g.im;
    const double dphi_drho = 2.0 * c.g2 / (pf.rho * c.r2);
    const C df_dm0 = C{0.0, 0.0} - c.inv_zb + c.inv_zba;
    const double dphi_dm0 = -(2.0 * c.inv_r2) * re_prod(conj(c.g), df_dm0);
    const C df_da = C{-c.m1, 0.0} * (c.inv_zba * c.inv_zba);
    const double dphi_da = -(2.0 * c.inv_r2) * re_prod(conj(c.g), df_da);
    return {dphi_dX, dphi_dY, dphi_drho, dphi_dm0, dphi_da};
}
}  // namespace phi_detail

inline std::array<double,5> fused_material_phi_dp(const phi_detail::Core& c,
    double R,const PrimaryFrame& pf) {
    auto result=phi_detail::dP_from(c,pf);
    if(fused_gradient::com_tangent_replay&&fused_gradient::precise_tangent_replay) {
        using namespace phi_detail;
        const C velocity=C{1,0}+C{pf.m0,0}*(c.inv_zb*c.inv_zb)+
            C{c.m1,0}*(c.inv_zba*c.inv_zba);
        const double translation=-2*(c.g.re*velocity.re+c.g.im*velocity.im)*c.inv_r2;
        result[3]-=pf.a*translation;result[4]+=c.m1*translation;
    }
    if(const auto* motion=fused_gradient::active_motion) {
        using namespace phi_detail;
        const C radial=(c.z+C{pf.m0,0}*c.inv_zb+
            C{c.m1,0}*(c.inv_zba*c.inv_zba)*c.zb)/C{R,0};
        const double slope=-2.0*(c.g.re*radial.re+c.g.im*radial.im)*c.inv_r2;
        for(int j=0;j<5;++j)result[j]+=slope*motion->radius[j];
    }
    return result;
}

// A boundary square root can be much smaller than the two quantities whose
// difference defines phi. Re-evaluate that rare cancellation in binary128,
// preserving the radial low part; a genuinely negative result still fails.
inline bool thin_boundary_geometry(double R,const PrimaryFrame& pf) {
 return pf.rho<std::sqrt(std::sqrt(std::numeric_limits<double>::epsilon()))*
     std::max({1.0,std::fabs(R),std::fabs(pf.a),std::fabs(pf.X),std::fabs(pf.Y)});
}
inline bool precise_boundary_required(double R,const PrimaryFrame& pf) {
 return fused_gradient::precise_tangent_replay&&thin_boundary_geometry(R,pf);
}
template<bool WithParameterDerivatives=true>
inline PhiValDP fused_precise_phi(double R,double theta,const PrimaryFrame&pf,double* angular=nullptr,double* radial_slope=nullptr){
 using Q=__float128;struct Z{Q re,im;};
 auto add=[](Z a,Z b){return Z{a.re+b.re,a.im+b.im};};
 auto sub=[](Z a,Z b){return Z{a.re-b.re,a.im-b.im};};
 auto mul=[](Z a,Z b){return Z{a.re*b.re-a.im*b.im,a.re*b.im+a.im*b.re};};
 auto scale=[](Z a,Q s){return Z{a.re*s,a.im*s};};
 auto inverse=[](Z a){Q n=a.re*a.re+a.im*a.im;return Z{a.re/n,-a.im/n};};
 auto dot=[](Z a,Z b){return a.re*b.re+a.im*b.im;};
 const Q r=Q(R)+fused_gradient::radius_offset,m=pf.m0,n=1-m,rho=pf.rho;
 const Z z{r*cosq(Q(theta)),r*sinq(Q(theta))},zb{z.re,-z.im};
 const Z i=inverse(zb),j=inverse(sub(zb,{pf.a,0}));
 const Z g=sub(sub(sub(z,scale(i,m)),scale(j,n)),{pf.X,pf.Y});
 const Q norm=dot(g,g),r2=rho*rho;
 PhiValDP out;out.phi=double((r2-norm)/r2);
 if(angular) {
  const Z k=add(scale(mul(i,i),m),scale(mul(j,j),n));
  const Z product=mul(zb,k);
  const Z velocity{-z.im+product.im,z.re-product.re};
  *angular=double(-2*dot(g,velocity)/r2);
 }
 if constexpr(!WithParameterDerivatives) {
  if(radial_slope) {
   const Z radial=scale(add(add(z,scale(i,m)),scale(mul(mul(j,j),zb),n)),1/r);
   *radial_slope=double(-2*dot(g,radial)/r2);
  }
  return out;
 }
 std::array<Q,5> derivatives{{2*g.re/r2,2*g.im/r2,2*norm/(rho*r2),
          -2*dot(g,sub(j,i))/r2,2*n*dot(g,mul(j,j))/r2}};
 if(fused_gradient::com_tangent_replay&&fused_gradient::precise_tangent_replay) {
  const Z velocity=add({1,0},add(scale(mul(i,i),m),scale(mul(j,j),n)));
  const Q translation=-2*dot(g,velocity)/r2;
  derivatives[3]-=Q(pf.a)*translation;derivatives[4]+=n*translation;
 }
 if(radial_slope||fused_gradient::active_motion){
  const Z radial=scale(add(add(z,scale(i,m)),scale(mul(mul(j,j),zb),n)),1/r);
  const Q slope=-2*dot(g,radial)/r2;
  if(radial_slope)*radial_slope=double(slope);
  if(const auto*motion=fused_gradient::active_motion)
   for(int k=0;k<5;++k)derivatives[k]+=slope*motion->radius[k];
 }
 for(int k=0;k<5;++k)out.dP[k]=double(derivatives[k]);
 return out;
}

inline PhiGrad phi_grad(double R, double theta, const PrimaryFrame& pf) {
    if(precise_boundary_required(R,pf)) {
        double angular;
        const auto precise=fused_precise_phi(R,theta,pf,&angular);
        return {precise.phi,angular,precise.dP};
    }
    const phi_detail::Core c = phi_detail::core(R, theta, pf);
    PhiGrad out;
    out.phi = c.phi;
    out.dphi_dtheta = phi_detail::dphi_dtheta_from(c, pf);
    out.dP = fused_material_phi_dp(c, R, pf);
    return out;
}

inline PhiValDP phi_val_dP(double R, double theta, const PrimaryFrame& pf) {
    const phi_detail::Core c = phi_detail::core(R, theta, pf);
    if(c.phi<1e-8)return fused_precise_phi(R,theta,pf);
    return {c.phi, fused_material_phi_dp(c, R, pf)};
}

inline PhiGrad phi_grad_quadrature(double R,double theta,const PrimaryFrame& pf) {
    const auto core=phi_detail::core(R,theta,pf);
    if(core.phi<1e-8) {
        double angular;
        const auto precise=fused_precise_phi(R,theta,pf,&angular);
        return {precise.phi,angular,precise.dP};
    }
    return {core.phi,phi_detail::dphi_dtheta_from(core,pf),fused_material_phi_dp(core,R,pf)};
}

inline PhiValDtheta phi_val_dtheta(double R, double theta,
                                  const PrimaryFrame& pf) {
    if(precise_boundary_required(R,pf)) {
        double angular;
        const auto precise=fused_precise_phi<false>(R,theta,pf,&angular);
        return {precise.phi,angular};
    }
    const phi_detail::Core c = phi_detail::core(R, theta, pf);
    return {c.phi, phi_detail::dphi_dtheta_from(c, pf)};
}

// phi only, shared core (bit-identical to phi_val_dP(...).phi and
// phi_val_dtheta(...).phi).  The value lane's angular sqrt(phi)
// quadrature uses this so F_half matches the fused pass exactly while
// paying nothing for the unused dP / dphi_dtheta.
inline double phi_val(double R, double theta, const PrimaryFrame& pf) {
    const double value=phi_detail::core(R, theta, pf).phi;
    return value<1e-8?fused_precise_phi<false>(R,theta,pf).phi:value;
}

// phi only, straight from the lens equation (topology._phi) -- the
// independent evaluation used for arc mid-point sign tests.
inline double phi_lens(double R, double theta, const PrimaryFrame& pf) {
    using phi_detail::C;
    double ct, st;
    __builtin_sincos(theta, &st, &ct);
    const C z{R * ct, R * st};
    const C zb{z.re, -z.im};
    const C zba = zb - C{pf.a, 0.0};
    const C fz = z - C{pf.m0, 0.0} / zb - C{1.0 - pf.m0, 0.0} / zba;
    const C d = fz - C{pf.X, pf.Y};
    return 1.0 - (d.re * d.re + d.im * d.im) / (pf.rho * pf.rho);
}

}  // namespace lcbinint::algebraic_boundary::tangent
