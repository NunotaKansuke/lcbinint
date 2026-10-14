// Exact exported cell/event plan audit, independent of adaptive integration.
#include "lcbinint/magnification/holonomic/prepared_geometry.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
using namespace lcbinint::holonomic;
int main(int argc,char**argv) {
 if(argc!=2)return 2;
 std::ifstream in(argv[1]);std::string line,last;PreparedEpochGeometry state;
 std::cout<<std::setprecision(17);
 while(std::getline(in,line)){
  int id,config,db,ep;std::string profile;double s,q,rho,x,y,t,u,X,ref;
  std::istringstream row(line);
  if(!(row>>id>>config>>profile>>db>>ep>>s>>q>>rho>>x>>y>>t>>u>>X>>ref))continue;
  std::string key=std::to_string(id)+"/"+std::to_string(config)+"/"+profile+"/"+std::to_string(db);
  if(key!=last){state=PreparedEpochGeometry{};last=key;}
  LensParams p{t,y,rho,1/q,s,true};auto pf=PrimaryFrame::from(p);
  for(int lane=0;lane<2;++lane){
   auto topo=lane?prepared_topology(pf,state,PreparedReuseConfig{},nullptr):classify_cells(pf,nullptr,nullptr,true);
   std::cout<<key<<' '<<ep<<' '<<lane<<" plan "<<int(topo.status)<<' '<<topo.r_max<<' '<<topo.cells.size()<<' '<<topo.events.size()<<'\n';
   for(auto& e:topo.events)std::cout<<key<<' '<<ep<<' '<<lane<<" event "<<e.radius<<' '<<e.kind<<' '<<e.physically_real<<' '<<e.radius_lo<<' '<<e.radius_uncertainty<<'\n';
   for(auto& c:topo.cells)std::cout<<key<<' '<<ep<<' '<<lane<<" cell "<<c.r_lo<<' '<<c.r_hi<<' '<<c.r_mid<<' '<<int(c.kind)<<' '<<c.n_crossings<<' '<<int(c.status)<<'\n';
  }
 }
}
