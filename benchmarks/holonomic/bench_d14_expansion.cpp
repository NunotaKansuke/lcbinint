#include "lcbinint/magnification/holonomic/radial_events.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <chrono>
using namespace lcbinint::holonomic;
int main(int argc,char**argv){
 if(argc!=2)return 2;
 std::ifstream in(argv[1]);std::string line;
 std::cout<<std::setprecision(17);
 while(std::getline(in,line)){
 int id,config,db,ep;std::string profile;double s,q,rho,x,y,t,u,X,ref;
 std::istringstream row(line);if(!(row>>id>>config>>profile>>db>>ep>>s>>q>>rho>>x>>y>>t>>u>>X>>ref))continue;
 auto pf=PrimaryFrame::from(LensParams{t,y,rho,1/q,s,true});
 auto block=re_detail::d14_struct_build((__float128)pf.a,(__float128)pf.m0,(__float128)pf.X,(__float128)pf.Y,(__float128)pf.rho);
 std::vector<__float128> c;std::array<double,21> times{};
 for(auto& elapsed:times){auto start=std::chrono::steady_clock::now();c=re_detail::d14_expanded_from_struct(block);elapsed=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();}
 std::sort(times.begin(),times.end());
 std::cout<<id<<' '<<config<<' '<<profile<<' '<<db<<' '<<ep<<' '<<times[10]<<' '<<c.size();
 char buf[128];for(auto a:c){quadmath_snprintf(buf,sizeof(buf),"%Qa",a);std::cout<<' '<<buf;}std::cout<<'\n';
 }
}
