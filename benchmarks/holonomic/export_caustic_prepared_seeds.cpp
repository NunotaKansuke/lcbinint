#include <chrono>
#include <fstream>
#include <iomanip>
#include <limits>
#include "lcbinint/magnification/finite_source_magnifier.hpp"
#include "lcbinint/magnification/point_source_magnifier.hpp"
using namespace lcbinint;
using namespace lcbinint::magnification;
int main(int argc,char**argv){
 if(argc!=4)return 2;std::ifstream in(argv[1]);std::ofstream out(argv[2]),cost(argv[3]);if(!in||!out||!cost)return 2;
 out<<std::setprecision(17);cost<<std::setprecision(17)<<"id common_ms seed_incremental_ms seed_count support distance\n";
 int id,rows=0;double x,y,rho,q,s;
 while(in>>id>>x>>y>>rho>>q>>s){
  PointSourceMagnifier pm;FiniteSourceSettings cfg;cfg.caustic_bins=1400;FiniteSourceMagnifier fm(cfg);
  auto t=std::chrono::steady_clock::now();std::vector<SourcePosition> center;
  for(auto image:pm.binary_images(s,q,{x,y}))center.push_back(image.position);
  // Same-epoch common geometry, prepared for method selection. No reuse
  // from a preceding geometry/epoch; seed construction is measured separately.
  double distance=fm.binary_caustic_distance_for_source(s,q,{x,y});
  double common=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count();
  bool proven=false;t=std::chrono::steady_clock::now();
  auto seeds=fm.cached_binary_image_seeds(pm,s,q,{x,y},rho,distance,&center,&proven);
  double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count();
  out<<id<<' '<<x<<' '<<y<<' '<<rho<<' '<<q<<' '<<s<<' '<<proven<<' '<<ms<<' '<<seeds.size();
  for(auto z:seeds)out<<' '<<z.x+(q<1?s:0)<<' '<<z.y;out<<'\n';
  cost<<id<<' '<<common<<' '<<ms<<' '<<seeds.size()<<' '<<proven<<' '<<distance<<'\n';++rows;
 }return rows?0:2;
}
