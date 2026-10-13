#include "lcbinint/magnification/finite_source_magnifier.hpp"
#include "lcbinint/magnification/point_source_magnifier.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <cmath>
using namespace lcbinint;using namespace lcbinint::magnification;
int main(int argc,char**argv){if(argc!=3)return 2;std::ifstream in(argv[1]);std::ofstream out(argv[2]);out<<"id\tsupport\tnseeds\tms\tcoordinates\n";std::string line;while(std::getline(in,line)){int id;double s,q,rho,x,y,c;std::istringstream ss(line);if(!(ss>>id>>s>>q>>rho>>x>>y>>c))continue;FiniteSourceSettings st;st.source_bins=16;st.caustic_bins=1400;st.grid_ratio=4;st.limb_darkening_c=c;FiniteSourceMagnifier mag(st);PointSourceMagnifier pm;SourcePosition src{x,y};auto ims=pm.binary_images(s,1/q,src);std::vector<SourcePosition>centres;for(auto im:ims)centres.push_back(im.position);mag.binary_caustic_branches(s,1/q);mag.binary_critical_curve_branches_cached(s,1/q);bool proven=false;auto start=std::chrono::steady_clock::now();auto seeds=mag.cached_binary_image_seeds(pm,s,1/q,src,rho,INFINITY,&centres,&proven);double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();out<<id<<'\t'<<proven<<'\t'<<seeds.size()<<'\t'<<std::defaultfloat<<std::setprecision(17)<<ms<<'\t';for(auto p:seeds)out<<std::hexfloat<<p.x<<','<<p.y<<';';out<<'\n';}}
