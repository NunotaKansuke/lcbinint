#include "lcbinint/magnification/holonomic/radial_events.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <chrono>
using namespace lcbinint::holonomic;
using namespace lcbinint::holonomic::re_detail;
int main(int argc,char**argv){
 if(argc!=2)return 2;std::ifstream in(argv[1]);std::string line;
 std::cout<<std::setprecision(17)<<"case config profile db epoch comparisons mismatch complex_us real_us\n";
 while(std::getline(in,line)){
 int id,config,db,ep;std::string profile;double s,q,rho,x,y,t,u,X,ref;
 std::istringstream row(line);if(!(row>>id>>config>>profile>>db>>ep>>s>>q>>rho>>x>>y>>t>>u>>X>>ref))continue;
 auto pf=PrimaryFrame::from(LensParams{t,y,rho,1/q,s,true});
 auto b=d14_struct_build((__float128)pf.a,(__float128)pf.m0,(__float128)pf.X,(__float128)pf.Y,(__float128)pf.rho);
 double complex_us=0,real_us=0;int mismatch=0;
 for(int k=0;k<32;++k){
  __float128 v=(__float128)(k+1)/8;Cplx<__float128> d,dp;__float128 a,ap;
  auto start=std::chrono::steady_clock::now();d14_struct_eval(b,Cplx<__float128>(v,0),d,dp);
  complex_us+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
  start=std::chrono::steady_clock::now();d14_struct_eval_real(b,v,a,ap);
  real_us+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
  mismatch+= !(a==d.re && ap==dp.re && d.im==0 && dp.im==0);
 }
 std::cout<<id<<' '<<config<<' '<<profile<<' '<<db<<' '<<ep<<" 32 "<<mismatch<<' '<<complex_us/32<<' '<<real_us/32<<'\n';
 }
}
