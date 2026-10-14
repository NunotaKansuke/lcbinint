#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>
#include <limits>
#include "lcbinint/magnification/finite_source_magnifier.hpp"
#include "lcbinint/magnification/point_source_magnifier.hpp"
#include "lcbinint/magnification/probe_diagnostics.hpp"
using namespace lcbinint;using namespace lcbinint::magnification;
int main(){std::ifstream sel("evidence/holonomic/seeded_cartesian_cost_split_v8/selected.tsv");std::string line;std::getline(sel,line);int id,b;double u;std::cout<<"id u rep seed_ms certify_ms solves seeds root_ms offered extrema\n";
while(sel>>id>>u>>b){std::ifstream geo("evidence/holonomic/caustic_seeded_connection_v4/initial_minimal_seeds.tsv");int gid;double x,y,rho,q,a;while(std::getline(geo,line)){std::istringstream s(line);if(s>>gid>>x>>y>>rho>>q>>a && gid==id)break;}
for(int rep=0;rep<3;++rep){PointSourceMagnifier pm;FiniteSourceSettings settings;settings.caustic_bins=1400;FiniteSourceMagnifier fm(settings);std::vector<SourcePosition> center;for(auto im:pm.binary_images(a,q,{x,y}))center.push_back(im.position);double d=fm.binary_caustic_distance_for_source(a,q,{x,y});reset_probe_counters();auto start=std::chrono::steady_clock::now();bool proven;auto seeds=fm.cached_binary_image_seeds(pm,a,q,{x,y},rho,std::numeric_limits<double>::infinity(),&center,&proven);double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();auto c=probe_counters();std::cout<<id<<' '<<u<<' '<<rep<<' '<<ms<<' '<<c.certify_seconds*1000<<' '<<c.certified_solves<<' '<<seeds.size()<<' '<<c.certified_seconds*1000<<' '<<c.certified_offered<<' '<<c.certified_extrema<<'\n';}}}
