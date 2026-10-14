#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include "lcbinint/magnification/holonomic/seeded_point_epoch.hpp"
using namespace lcbinint::holonomic;
int main(int argc,char**argv){if(argc!=2)return 2;std::ifstream in(argv[1]);std::string line;std::cout<<std::setprecision(17);
while(std::getline(in,line)){std::istringstream is(line);int id,support,n;LensParams p;double ms;if(!(is>>id>>p.xs>>p.ys>>p.rho>>p.q>>p.a>>support>>ms>>n))continue;if(id!=1014&&id!=1071)continue;auto pf=PrimaryFrame::from(p);
for(int i=0;i<n;++i){double x,y;is>>x>>y;point_seed_detail::Z z{x,y},partner=point_seed_detail::Z(pf.X,pf.Y)+pf.m0/std::conj(z)+(1-pf.m0)/std::conj(z-pf.a),mid=.5*(z+partner);
for(auto initial:{z,mid}){auto w=initial;bool ok=point_seed_detail::critical_witness(w,pf);auto s=point_seed_detail::critical_state(w,pf);std::cout<<id<<' '<<i<<' '<<initial<<' '<<w<<' '<<ok<<' '<<s.h<<' '<<s.g<<' '<<std::abs(s.mapped-point_seed_detail::Z(pf.X,pf.Y))/pf.rho<<'\n';}}
}}
