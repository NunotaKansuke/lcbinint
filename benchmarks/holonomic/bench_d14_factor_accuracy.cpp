// Research only: correction error at saved binary64 candidates. No solve acceptance.
#include "lcbinint/magnification/holonomic/radial_events.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
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
// Mixed arithmetic diagnostic: blocks and discriminant combination separated.
template<class Out,class In> Cplx<Out> cv(Cplx<In> x){return {Out(x.re),Out(x.im)};}
template<class A,class B> Cplx<B> mixed_correction(const D14StructC<A>& s,Cplx<A> v){
 Cplx<A> ca,cpa,ga,gpa,za,zpa;
 horner_vd(s.c3,3,v,ca,cpa);horner_vd(s.g4,4,v,ga,gpa);horner_vd(s.z3,3,v,za,zpa);
 auto c=cv<B>(ca),cp=cv<B>(cpa),g=cv<B>(ga),gp=cv<B>(gpa),z=cv<B>(za),zp=cv<B>(zpa),vb=cv<B>(v);
 Cplx<B> two(B(2),0),four(B(4),0),nine(B(9),0),eight(B(8),0),k432(B(432),0),k864(B(864),0);
 auto cc=c*c,vg=vb*g,f=cc-four*vg,b=two*cc-nine*vg,gg=g*g,zz=z*z,vv=vb*vb;
 auto dh=f*gg+eight*(c*b*z)-k432*(vv*zz);
 auto ccp=c*cp,fp=two*ccp-four*g-four*(vb*gp),bp=four*ccp-nine*g-nine*(vb*gp);
 auto dp=fp*gg+two*(f*(g*gp))+eight*((cp*b+c*bp)*z+c*b*zp)-k864*(vb*zz)-k864*(vv*(z*zp));
 return dh/dp; // common exact factor 4096 cancels
}
struct TimingPoint {D14StructC<double> sd;D14StructC<D14Real> sr;Cplx<double> v;};
int timing_probe(const char* input){
 std::ifstream in(input);std::vector<TimingPoint> points;
 int id,root,stage;double time,y,rho,q,s,re,im,sep;
 while(in>>id>>root>>stage>>time>>y>>rho>>q>>s>>re>>im>>sep){
  if((id*17+root*3+stage)%97!=0)continue;
  auto p=PrimaryFrame::from(LensParams{time,y,rho,1/q,s,true});
  auto sq=d14_struct_build(p.a,p.m0,p.X,p.Y,p.rho);
  points.push_back({d14_struct_cast<double>(sq),d14_struct_cast<D14Real>(sq),{re,im}});
 }
 if(points.empty())return 3;
 using Clock=std::chrono::steady_clock;
 volatile double sink=0;
 std::cout<<std::setprecision(17)<<"repeat method points cycles ns_per_correction nonfinite\n";
 for(int rep=-1;rep<8;++rep){
  for(int order=0;order<2;++order){
   const int method=(order+(rep&1))%2;int bad=0;double sum=0;
   auto begin=Clock::now();
   for(int cycle=0;cycle<32;++cycle)for(const auto& p:points){
    Cplx<D14Real> w;
    if(method==0){Cplx<D14Real> f,d;d14_struct_eval(p.sr,Cplx<D14Real>(D14Real(p.v.re),D14Real(p.v.im)),f,d);w=f/d;}
    else w=mixed_correction<double,D14Real>(p.sd,p.v);
    double value=double(w.re);
    if(std::isfinite(value))sum+=value;else ++bad;
   }
   const double ns=std::chrono::duration<double,std::nano>(Clock::now()-begin).count()/(32*points.size());
   sink=sum;
   if(rep>=0)std::cout<<rep<<' '<<method<<' '<<points.size()<<" 32 "<<ns<<' '<<bad<<'\n';
  }
 }
 std::cerr<<"sink "<<sink<<"\n";return 0;
}
int main(int argc,char**argv){
 if(argc==3 && std::string(argv[2])=="timing")return timing_probe(argv[1]);
 if(argc<2 || argc>3)return 2;const int method_count=argc==3?std::atoi(argv[2]):3;if(method_count<3 || method_count>8)return 2;std::ifstream in(argv[1]);
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
  for(int method=0;method<method_count;++method){
   if(method==0){f={double(coeff.back()),0};d={0,0};for(int k=int(coeff.size())-2;k>=0;--k){d=d*z+f;f=f*z+Cplx<double>(double(coeff[k]),0);}}
   if(method==1)d14_struct_eval(sd,z,f,d);
   if(method==2){auto out=factored(z,p);f=out.f;d=out.d;}
   Cplx<__float128> result;
   if(method<3)result=cv<__float128>(f/d);
   if(method==3)result=mixed_correction<double,__float128>(sd,z);
   if(method==4)result=cv<__float128>(mixed_correction<__float128,double>(sq,vq));
   if(method==5){D14StructQf rounded;for(int k=0;k<4;++k){rounded.c3[k]=sd.c3[k];rounded.z3[k]=sd.z3[k];}for(int k=0;k<5;++k)rounded.g4[k]=sd.g4[k];Cplx<__float128> a,b;d14_struct_eval(rounded,vq,a,b);result=a/b;}
   if(method==6){auto sr=d14_struct_cast<D14Real>(sq);Cplx<D14Real> a,b;d14_struct_eval(sr,Cplx<D14Real>(D14Real(re),D14Real(im)),a,b);auto w=a/b;result={d14_to_qf(w.re),d14_to_qf(w.im)};}
   if(method==7){auto w=mixed_correction<double,D14Real>(sd,z);result={d14_to_qf(w.re),d14_to_qf(w.im)};}
   double err=double(hypotq(result.re-wq.re,result.im-wq.im));
   std::cout<<id<<' '<<root<<' '<<stage<<' '<<method<<' '<<err<<' '<<err/(1+std::hypot(re,im))<<' '<<err/sep<<'\n';
  }
 }
 std::cerr<<std::setprecision(17)<<"off_root_qf_identity_max "<<max_identity_error<<"\n";
}
