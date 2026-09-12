// Research only: correction error at saved binary64 candidates. No solve acceptance.
#include "lcbinint/magnification/holonomic/radial_events.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
using namespace lcbinint::holonomic;
using namespace lcbinint::holonomic::re_detail;
template<class T> struct Jet {
 Cplx<T> f,d;
 Jet(T x=T(0)):f(x,0),d(0,0){}
 Jet(Cplx<T> x,Cplx<T> dx):f(x),d(dx){}
};
template<class T> Jet<T> operator+(Jet<T> a,Jet<T> b){return {a.f+b.f,a.d+b.d};}
template<class T> Jet<T> operator-(Jet<T> a,Jet<T> b){return {a.f-b.f,a.d-b.d};}
template<class T> Jet<T> operator*(Jet<T> a,Jet<T> b){return {a.f*b.f,a.d*b.f+a.f*b.d};}
template<class T> Jet<T> factored(Cplx<T> z,const PrimaryFrame& p){
 using J=Jet<T>;
 J v(z,{1,0}),a(T(p.a)),m(T(p.m0)),x(T(p.X)),y(T(p.Y)),h(T(p.rho)*T(p.rho));
 J d=v-J(1),e=v-m,beta=x*x+y*y-h;
 J L=v*d+a*a*e,U=a*e*d+x*L+a*v*beta;
 J C=v*d*d+a*a*e*e+v*(v+a*a)*beta+J(2)*a*x*v*(J(3)*m-J(1)-J(2)*v);
 J G=U*U+y*y*L*L-J(4)*a*e*x*C-J(4)*a*a*e*e*v*(J(4)*x*x+y*y);
 J B=(a*m+(x-a)*v)*(a*m+(x-a)*v)+v*v*(y*y-h);
 J Z=a*a*(J(1)-m)*y*y*e*B;
 return J(4096)*((C*C-J(4)*v*G)*G*G+J(8)*C*(J(2)*C*C-J(9)*v*G)*Z-J(432)*v*v*Z*Z);
}
int main(int argc,char**argv){
 if(argc!=2)return 2;std::ifstream in(argv[1]);
 int id,root,stage;double max_identity_error=0;double time,y,rho,q,s,re,im,sep;
 std::cout<<std::setprecision(17)<<"sample root stage method correction_error scaled_error separation_ratio\n";
 while(in>>id>>root>>stage>>time>>y>>rho>>q>>s>>re>>im>>sep){
  auto p=PrimaryFrame::from(LensParams{time,y,rho,1/q,s,true});
  auto sq=d14_struct_build(p.a,p.m0,p.X,p.Y,p.rho);auto sd=d14_struct_cast<double>(sq);
  // Off-root complex point checks the alternative algebra and derivative,
  // avoiding division by a nearly zero value at the saved root.
  if(root==0 && stage==0){
   Cplx<__float128> test(0.137Q,0.239Q),f0,d0;d14_struct_eval(sq,test,f0,d0);
   auto f1=factored(test,p);
   double e0=double(hypotq(f0.re-f1.f.re,f0.im-f1.f.im)/(1+hypotq(f0.re,f0.im)));
   double e1=double(hypotq(d0.re-f1.d.re,d0.im-f1.d.im)/(1+hypotq(d0.re,d0.im)));
   max_identity_error=std::max(max_identity_error,std::max(e0,e1));
  }
  Cplx<__float128> vq(re,im),fq,dq;d14_struct_eval(sq,vq,fq,dq);auto wq=fq/dq;
  Cplx<double> z(re,im),f,d;auto coeff=d14_expanded_from_struct(sq);
  for(int method=0;method<3;++method){
   if(method==0){f={double(coeff.back()),0};d={0,0};for(int k=int(coeff.size())-2;k>=0;--k){d=d*z+f;f=f*z+Cplx<double>(double(coeff[k]),0);}}
   if(method==1)d14_struct_eval(sd,z,f,d);
   if(method==2){auto out=factored(z,p);f=out.f;d=out.d;}
   auto w=f/d;double err=double(hypotq((__float128)w.re-wq.re,(__float128)w.im-wq.im));
   std::cout<<id<<' '<<root<<' '<<stage<<' '<<method<<' '<<err<<' '<<err/(1+std::hypot(re,im))<<' '<<err/sep<<'\n';
  }
 }
 std::cerr<<std::setprecision(17)<<"off_root_qf_identity_max "<<max_identity_error<<"\n";
}
