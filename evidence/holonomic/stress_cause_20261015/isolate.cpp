#include <cstdio>
#include <cstdlib>
#include <iostream>
#include "geometry.hpp"
int main(int argc,char**argv){using namespace lcbinint::holonomic_fused;
 PrimaryFrame pf{atof(argv[1]),atof(argv[2]),atof(argv[3]),0,atof(argv[4])};
 TopologyResult topo;double lo=atof(argv[5]),hi=atof(argv[6]);topo.cells.push_back({0,lo,hi,(lo+hi)/2,ArcKind::kArcs,4});bool complete=false;
 auto r=geometry_experiment::detect(pf,topo,{},&complete,7);std::cout<<complete<<" "<<r.size()<<"\n";
}
