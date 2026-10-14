#include <fstream>
#include <iomanip>
#include <chrono>
#include "lcbinint/magnification/holonomic/seeded_point_epoch.hpp"
using namespace lcbinint::holonomic;
int main(int argc,char**argv){
 if(argc!=3)return 2;std::ifstream in(argv[1]);std::ofstream out(argv[2]);if(!in||!out)return 2;out<<std::setprecision(17);
 int id,rows=0;LensParams p;
 while(in>>id>>p.xs>>p.ys>>p.rho>>p.q>>p.a){
  auto t=std::chrono::steady_clock::now();lcbinint::magnification::PointSourceMagnifier point;
  auto seeds=point_seed_proposals(p,point);
  double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count();
  out<<id<<' '<<p.xs<<' '<<p.ys<<' '<<p.rho<<' '<<p.q<<' '<<p.a<<" 0 "<<ms<<' '<<seeds.positions.size();
  for(auto z:seeds.positions)out<<' '<<z.re<<' '<<z.im;out<<'\n';++rows;
 }
 return rows?0:2;
}
