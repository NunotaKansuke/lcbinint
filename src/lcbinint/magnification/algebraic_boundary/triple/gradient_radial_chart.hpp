#pragma once
#include "physical_jet.hpp"
#include "../poly_roots.hpp"
namespace lcbinint::algebraic_boundary::triple::shape_gradient_detail {
// Holomorphic extension of the existing physical event jet. Complex roots
// choose quadrature coordinates only; real support/count proofs remain primal.
namespace binary_complex_detail {
using Z=::lcbinint::algebraic_boundary::Cplx<double>;
inline Z operator-(Z z){return {-z.re,-z.im};}
inline Z operator-(double a,Z z){return {a-z.re,-z.im};}
inline Z operator*(double a,Z z){return {a*z.re,a*z.im};}
inline Z operator*(Z z,double a){return a*z;}
inline Z& operator+=(Z& a,Z b){a.re+=b.re;a.im+=b.im;return a;}
inline Z& operator-=(Z& a,Z b){a.re-=b.re;a.im-=b.im;return a;}
inline Z conjugate(Z z){return {z.re,-z.im};}
inline Z reciprocal(Z z){const double n=::lcbinint::algebraic_boundary::cabs2(z);if(n>32*std::numeric_limits<double>::min()&&std::isfinite(n))return ::lcbinint::algebraic_boundary::cdiv_fast(Z(1),z);return ::lcbinint::algebraic_boundary::crecip(z);}
inline Complex pack(Z z){return {z.re,z.im};}
inline physical_detail::Jet<Complex> event(Complex r,Complex t,const Geometry& p,bool chart){const Z radius(r.real(),r.imag()),coordinate(t.real(),t.imag());

 const Z one(1),i(0,1),inverse=reciprocal(one+coordinate*coordinate);
 const double sign=chart?-1.:1.;
 const Z w=sign*(one-coordinate*coordinate+2.*i*coordinate)*inverse;
 const Z wb=sign*(one-coordinate*coordinate-2.*i*coordinate)*inverse;
 const Z z=radius*w,zb=radius*wb,invR=reciprocal(radius);
 const Z radial=radius-(1-p.m1-p.m2)*invR;
 const Z radialR=one+(1-p.m1-p.m2)*invR*invR;
 Z g=radial*w-Z(p.X,p.Y),gb=radial*wb-Z(p.X,-p.Y);
 Z shear{},shearb{},third{},thirdb{};
 for(const auto& lens:std::array<std::pair<Z,double>,2>{{{Z(p.a),p.m1},{Z(p.bx,p.by),p.m2}}}) {
  const Z u=reciprocal(zb-conjugate(lens.first)),v=reciprocal(z-lens.first);
  g-=lens.second*u;gb-=lens.second*v;
  const Z su=lens.second*u*u,sv=lens.second*v*v;
  shear+=su;shearb+=sv;third+=su*u;thirdb+=sv*v;
 }
 const Z gr=radialR*w+wb*shear,grb=radialR*wb+w*shearb;
 const Z gt=i*(radial*w-zb*shear),gtb=-i*(radial*wb-z*shearb);
 const Z gtt=-radial*w-zb*shear+2.*zb*zb*third;
 const Z gttb=-radial*wb-z*shearb+2.*z*z*thirdb;
 const Z grt=i*(radialR*w-wb*shear+2.*zb*wb*third);
 const Z grtb=-i*(radialR*wb-w*shearb+2.*z*w*thirdb);
 const Z ft=-(gt*gb+g*gtb),theta_t=2.*inverse;
 physical_detail::Jet<Z> out;
 out.v=p.rho*p.rho-g*gb;out.r=-(gr*gb+g*grb);
 out.t=ft*theta_t;
 out.rt=-(grt*gb+gr*gtb+gt*grb+g*grtb)*theta_t;
 out.tt=-(gtt*gb+2.*gt*gtb+g*gttb)*theta_t*theta_t-ft*4.*coordinate*inverse*inverse;
 physical_detail::Jet<Complex> result;result.v=pack(out.v);result.r=pack(out.r);result.t=pack(out.t);result.rt=pack(out.rt);result.tt=pack(out.tt);return result;

}
}
inline physical_detail::Jet<Complex> complex_event(Complex radius,Complex coordinate,const Geometry& p,bool chart) {
 return binary_complex_detail::event(radius,coordinate,p,chart);
}
struct RadialComplexPair {double center,scale;std::array<double,8> center_motion{},scale_motion{};bool material=false;};
// Holomorphic stationary-root transport: F=0, F_t=0 implies R_p=-F_p/F_R.
// Conjugate-looking factors here are independent analytic factors, not a
// conjugation of the complex radius. This is the extension used by the root
// solver above, with the eight canonical real parameters differentiated.
inline bool complex_radial_motion(Complex R,Complex t,const Geometry& p,bool chart,RadialComplexPair& out){
 const Complex one(1),i(0,1),inv=std::conj(inverse_conjugate(one+t*t)),ri=std::conj(inverse_conjugate(R));const double sign=chart?-1.:1.;
 const Complex w=sign*(one-t*t+2.*i*t)*inv,wb=sign*(one-t*t-2.*i*t)*inv,z=R*w,zb=R*wb;
 const auto inverse=[](Complex v){return std::conj(inverse_conjugate(v));};
 const Complex u1=inverse(zb-Complex(p.a)),v1=inverse(z-Complex(p.a)),u2=inverse(zb-Complex(p.bx,-p.by)),v2=inverse(z-Complex(p.bx,p.by));
 const Complex radial=R-(1-p.m1-p.m2)*ri,g=radial*w-Complex(p.X,p.Y)-p.m1*u1-p.m2*u2,gb=radial*wb-Complex(p.X,-p.Y)-p.m1*v1-p.m2*v2;
 const auto f=complex_event(R,t,p,chart);const Complex invfr=inverse(f.r);
 const std::array<Complex,8> fp{g+gb,i*(gb-g),2*p.rho,-((w*ri-u1)*gb+g*(wb*ri-v1)),-((w*ri-u2)*gb+g*(wb*ri-v2)),p.m1*(u1*u1*gb+g*v1*v1),p.m2*(u2*u2*gb+g*v2*v2),-i*p.m2*(u2*u2*gb-g*v2*v2)};
 for(int j=0;j<8;++j){const Complex motion=-fp[j]*invfr;if(!std::isfinite(motion.real())||!std::isfinite(motion.imag()))return false;out.center_motion[j]=motion.real();out.scale_motion[j]=(R.imag()<0?-1:1)*motion.imag();}out.material=true;return true;
}
inline std::optional<RadialComplexPair> radial_complex_pair(double radius,double theta,double lo,double hi,const Geometry&p,bool with_motion=false) {
 const double width=hi-lo,pi=std::acos(-1.);const bool chart=std::abs(std::remainder(theta,2*pi))>pi*.5;
 const double t0=std::tan((theta-(chart?pi:0))*.5);
 Complex R(radius,width*.1),t(t0,0);bool converged=false;
 for(int iteration=0;iteration<32;++iteration) {
  const auto f=complex_event(R,t,p,chart);const auto det=f.r*f.tt-f.t*f.rt;
  // The two Newton right-hand sides share one determinant reciprocal.
  // inverse_conjugate retains scaled division outside the normal range.
  const auto inverse_det=std::conj(inverse_conjugate(det));
  const auto dr=(-f.v*f.tt+f.t*f.t)*inverse_det,dt=(-f.r*f.t+f.rt*f.v)*inverse_det;
  if(!std::isfinite(std::abs(dr)+std::abs(dt)) || std::abs(dr)>4*width || std::abs(dt)>4)return std::nullopt;
  R+=dr;t+=dt;
  if(R.real()<lo-width || R.real()>hi+width || std::abs(R.imag())>width)return std::nullopt;
  if(std::abs(dr)<1e-12*(1+std::abs(R)) && std::abs(dt)<1e-12*(1+std::abs(t))){converged=true;break;}
 }
 const auto f=complex_event(R,t,p,chart);
 if(!converged || std::abs(f.v)+std::abs(f.t)>1e-9*p.rho*p.rho)return std::nullopt;
 const double scale=std::abs(R.imag());
 if(R.real()<=lo || R.real()>=hi || scale<1e-10*width || scale>.25*width)return std::nullopt;
 RadialComplexPair out{R.real(),scale};if(with_motion)complex_radial_motion(R,t,p,chart,out);return out;
}
}
