#include <cmath>
#include <iomanip>
#include <iostream>
#include <array>
#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"
#include "lcbinint/magnification/holonomic/gm_sensitivity.hpp"
using namespace lcbinint::holonomic;
using namespace lcbinint::holonomic::adaptive_detail;
TopologyResult topo(double t,double u0,double alpha,double tE,double rho,double q,double s){double tau=t/tE,c=cos(alpha),z=sin(alpha);return classify_cells(PrimaryFrame::from(LensParams{tau*c-u0*z,tau*z+u0*c,rho,q,s,true}),nullptr,nullptr,true);}
template<class S> std::array<S,5> Pcoeff(DD rr,const GmParams<S>&p){S R(rr),R2=R*R;GmComplex<S> z(p.X,p.Y);GmComplex<S> n0(-p.X*R2,-p.Y*R2);GmComplex<S> n1=gm_complex_scale(z,p.a*R)+GmComplex<S>(R*(R2-S(1)),S(0));GmComplex<S> n2(p.a*(p.m0-R2),S(0));auto c0=n0+n1+n2;auto d=n2-n0;GmComplex<S> c1(-S(2)*d.im,S(2)*d.re);auto c2=n1-n0-n2;S k=p.rho*p.rho*R2,bm=(R-p.a)*(R-p.a),bp=(R+p.a)*(R+p.a);return {k*bm-gm_complex_norm(c0),-S(2)*gm_complex_re_cross(c0,c1),k*(bm+bp)-gm_complex_norm(c1)-S(2)*gm_complex_re_cross(c0,c2),-S(2)*gm_complex_re_cross(c1,c2),k*bp-gm_complex_norm(c2)};}
DD evald(const std::array<DD,5>&p,DD x){DD y=p[4];for(int i=3;i>=0;--i)y=y*x+p[i];return y;}
int main(){constexpr double u0=9.846481110931334e-7,alpha=.5,tE=10,rho=1e-6,q=1e7,s=.95;auto base=topo(0,u0,alpha,tE,rho,q,s);auto pf=PrimaryFrame::from(LensParams{-u0*sin(alpha),u0*cos(alpha),rho,q,s,true});std::cout<<std::setprecision(17);for(size_t i=0;i<base.events.size();++i){const auto&e=base.events[i];if(!e.physically_real||i<8||i>9)continue;auto loc=topology_event_estimate(e,pf,1e-3,true);DD rr=DD(loc.radius)+DD(loc.radius_lo),tt(loc.t_seed);LocalFoldQuantities<DD> f;bool ok=coupled_fold_newton<DD,false>(rr,tt,pf,8,1e-24,1e-14,nullptr,nullptr,nullptr,&f);auto params=gm_sensitivity_params<GmDDDual5>(pf);auto pc=Pcoeff(rr,params);std::array<DD,5> deriv[5];for(int j=0;j<5;++j)for(int k=0;k<5;++k)deriv[j][k]=pc[k].deriv[j];std::array<double,5> drdp{};for(int j=0;j<5;++j)drdp[j]=(double)(-evald(deriv[j],tt)/f.PR);double dt0=(-cos(alpha)*drdp[0]-sin(alpha)*drdp[1])/tE;std::cout<<"i="<<i<<" solve="<<ok<<" R="<<(double)rr<<" t="<<(double)tt<<" tlo="<<tt.lo<<" PR="<<(double)f.PR<<" dX="<<drdp[0]<<" dY="<<drdp[1]<<" dt0="<<dt0<<"\n";}}
