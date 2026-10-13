#include "lcbinint/magnification/finite_source_magnifier.hpp"
#include "lcbinint/magnification/point_source_magnifier.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <limits>
using namespace lcbinint;using namespace lcbinint::magnification;
int main(int argc,char**argv){std::ifstream in(argv[1]);std::string line;while(std::getline(in,line)){std::istringstream s(line);int id;double x,y,rho,q,a;if(!(s>>id>>x>>y>>rho>>q>>a))continue;PointSourceMagnifier pm;FiniteSourceSettings cfg;cfg.caustic_bins=1400;FiniteSourceMagnifier fm(cfg);std::vector<SourcePosition> center;for(auto im:pm.binary_images(a,q,{x,y}))center.push_back(im.position);fm.binary_caustic_distance_for_source(a,q,{x,y});bool proven=false;auto seed=fm.cached_binary_image_seeds(pm,a,q,{x,y},rho,std::numeric_limits<double>::infinity(),&center,&proven);std::cout<<id<<' '<<proven<<' '<<seed.size();for(auto z:seed)std::cout<<' '<<std::hexfloat<<z.x<<' '<<z.y;std::cout<<'\n';}}
