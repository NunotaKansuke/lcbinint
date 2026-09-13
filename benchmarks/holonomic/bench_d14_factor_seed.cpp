// Research candidate discovery: six simple roots + four split double roots.
// No completeness certificate; never connected to production routing.
#include "lcbinint/magnification/holonomic/radial_events.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <chrono>
using namespace lcbinint::holonomic;
using namespace lcbinint::holonomic::re_detail;
using Q=__float128;using Z=Cplx<Q>;using Clock=std::chrono::steady_clock;
static Z lift(Cplx<double> z){return {Q(z.re),Q(z.im)};}
static Z csqrt(Z z){Q r=hypotq(z.re,z.im);return {sqrtq(fmaxq(0,(r+z.re)/2)),copysignq(sqrtq(fmaxq(0,(r-z.re)/2)),z.im)};}
static void eval(const std::vector<Q>& c,Z z,Z& f,Z& d){f={c.back(),0};d={0,0};for(int i=int(c.size())-2;i>=0;--i){d=d*z+f;f=f*z+Z(c[i],0);}}
static std::vector<Z> lowroots(const std::vector<Q>& c,int& sweeps,bool high_precision){
 int n=int(c.size())-1;double scale=1;
 if(c[0]!=0 && c[n]!=0){double logscale=double(logq(fabsq(c[0]/c[n]))/n);if(std::isfinite(logscale))scale=std::exp(logscale);}
 if(!(scale>0)||!std::isfinite(scale))scale=1;
 std::vector<Q> b(n+1);Q mass=0,power=1;
 for(int i=0;i<=n;++i){b[n-i]=c[i]*power;mass=fmaxq(mass,fabsq(b[n-i]));power*=Q(scale);}
 if(high_precision){
  for(auto& x:b)x/=mass;
  auto out=aberth<Q>(b.data(),n,200,nullptr,1e-26Q,nullptr,&sweeps);
  for(auto& z:out)z=z*Z(Q(scale),0);
  return out;
 }
 std::vector<double> d(n+1);for(int i=0;i<=n;++i)d[i]=double(b[i]/mass);
 auto raw=aberth<double>(d.data(),n,80,nullptr,1e-12,nullptr,&sweeps);std::vector<Z> out;
 for(auto x:raw){Z z=lift(x)*Z(Q(scale),0);
  for(int k=0;k<3;++k){Z f,dp;eval(c,z,f,dp);Z step=f/dp;if(!finiteq(step.re)||!finiteq(step.im))break;z=z-step;}
  out.push_back(z);
 }return out;
}
int main(int argc,char**argv){
 if(argc<4||argc>5)return 2;const bool high_precision=argc==5 && std::string(argv[4])=="qf";std::ifstream in(argv[1]);std::ofstream roots(argv[2]),stats(argv[3]);
 roots<<std::setprecision(17)<<"sample stage root re im\n";
 stats<<std::setprecision(17)<<"sample low_sweeps factor_ms split_ms refine2_ms refine4_ms finite0 finite2 finite4 physical0 physical2 physical4 classify_ms\n";
 std::string line,profile;int sample=0,cid,config,db,epoch;double s,q,rho,x,y,time,u,X,ref;
 while(std::getline(in,line)){
  if(line.empty()||line[0]=='#')continue;std::istringstream l(line);
  if(!(l>>cid>>config>>profile>>db>>epoch>>s>>q>>rho>>x>>y>>time>>u>>X>>ref))return 3;
  auto p=PrimaryFrame::from(LensParams{time,y,rho,1/q,s,true});auto begin=Clock::now();
  auto sc=d14_struct_build(p.a,p.m0,p.X,p.Y,p.rho);
  std::vector<Q> F(7,0),G(sc.g4.begin(),sc.g4.end());
  for(int i=0;i<4;++i)for(int j=0;j<4;++j)F[i+j]+=sc.c3[i]*sc.c3[j];
  for(int i=0;i<5;++i)F[i+1]-=4*sc.g4[i];
  int fs=0,gs=0;auto fc=lowroots(F,fs,high_precision),gc=lowroots(G,gs,high_precision);auto after_factor=Clock::now();
  std::vector<Z> candidate=fc;
  for(auto z:gc){Z fv,fp,gv,gp,D,Dp;eval(F,z,fv,fp);eval(G,z,gv,gp);d14_struct_eval(sc,z,D,Dp);
   Z delta=csqrt((Z(0,0)-D)/(Z(4096,0)*fv*gp*gp));candidate.push_back(z+delta);candidate.push_back(z-delta);
  }
  auto after_split=Clock::now();int finite[3]{},physical[3]{};double times[2]{},classify_ms=0;
  const double W=std::hypot(p.X,p.Y)+p.rho,Rmax=0.5*(p.a+W+std::hypot(p.a-W,2.0))+1e-12;
  for(int stage=0;stage<3;++stage){
   if(stage){auto start=Clock::now();for(int it=0;it<2;++it)for(auto& z:candidate){Z D,Dp;d14_struct_eval(sc,z,D,Dp);Z step=D/Dp;if(finiteq(step.re)&&finiteq(step.im))z=z-step;}
    times[stage-1]=std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
   auto classify_start=Clock::now();
   for(auto z:candidate){
    if(finiteq(z.re)&&finiteq(z.im)&&z.re>0&&fabsq(z.im)<=1e-8Q*(1+fabsq(z.re))){
     double R=double(sqrtq(z.re));if(R>0 && R<Rmax && double_root_is_real(R,p))++physical[stage];
    }
   }
   classify_ms+=std::chrono::duration<double,std::milli>(Clock::now()-classify_start).count();
   for(int i=0;i<14;++i){auto z=candidate[i];finite[stage]+=finiteq(z.re)&&finiteq(z.im);roots<<sample<<' '<<stage*2<<' '<<i<<' '<<double(z.re)<<' '<<double(z.im)<<'\n';}
  }
  stats<<sample<<' '<<fs+gs<<' '<<std::chrono::duration<double,std::milli>(after_factor-begin).count()<<' '<<std::chrono::duration<double,std::milli>(after_split-after_factor).count()<<' '<<times[0]<<' '<<times[1]<<' '<<finite[0]<<' '<<finite[1]<<' '<<finite[2]<<' '<<physical[0]<<' '<<physical[1]<<' '<<physical[2]<<' '<<classify_ms<<'\n';++sample;
 }
}
