#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include "lcbinint/magnification/finite_source_magnifier.hpp"
#include "lcbinint/magnification/point_source_magnifier.hpp"
using namespace lcbinint; using namespace lcbinint::magnification;
using Clock=std::chrono::steady_clock;
int main(int argc,char**argv){
 if(argc<3||argc>5)return 2;std::ifstream in(argv[1]);std::ofstream out(argv[2]);if(!in||!out)return 2;
 out<<std::setprecision(17)<<"id u tol engine nbin mu relative_difference converged total_ms no_caustic_ms center_ms caustic_ms\n";
 std::string line;std::getline(in,line);std::vector<std::string> keys;{std::istringstream s(line);std::string k;while(s>>k)keys.push_back(k);}
 while(std::getline(in,line)){
  std::istringstream row(line);std::vector<std::string> v;std::string word;while(row>>word)v.push_back(word);
  auto field=[&](std::string k){return v.at(std::find(keys.begin(),keys.end(),k)-keys.begin());};
  if(field("d14_converged")!="1"||field("tol")!="0.0001")continue;
  int id=std::stoi(field("id")); double u=std::stod(field("u")),ref=std::stod(field("d14_mu"));
  // Geometry is read separately from the saved source input to avoid frame ambiguity.
  std::ifstream geometry(argc==5?argv[4]:"evidence/holonomic/caustic_seeded_connection_v4/initial_minimal_seeds.tsv");
  int input_id;double x,y,rho,q,a;bool found=false;while(std::getline(geometry,word)){std::istringstream g(word);if(g>>input_id>>x>>y>>rho>>q>>a && input_id==id){found=true;break;}}if(!found)return 2;
  std::vector<int> levels=(argc>=4&&std::string(argv[3])=="low")?std::vector<int>{16,32}:
   (argc>=4&&std::string(argv[3])=="pilot")?std::vector<int>{16,32,64,128}:std::vector<int>{64,128,256,512};
  for(int bins:levels)for(int engine:{1,2}){
   std::vector<double> times,noc,ct,cat;FiniteSourceResult result;
   for(int rep=0;rep<3;++rep){auto start=Clock::now();PointSourceMagnifier pm;FiniteSourceSettings cfg;cfg.caustic_bins=1400;cfg.limb_darkening_c=u;cfg.automatic_source_bins=false;FiniteSourceMagnifier fm(cfg);
    auto ms=[](Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();};
    auto stage=Clock::now();auto images=pm.binary_images(a,q,{x,y});std::vector<SourcePosition> center;double point=0;for(auto image:images){center.push_back(image.position);point+=1.0/std::abs(image.jacobian_determinant);}ct.push_back(ms(stage));
    stage=Clock::now();fm.binary_caustic_distance_for_source(a,q,{x,y});cat.push_back(ms(stage));
    result=fm.binary_mag_preplanned(a,q,{x,y},rho,point,engine==1?FiniteSourceMethod::inverse_ray_cartesian:FiniteSourceMethod::inverse_ray_polar,bins,&center,&pm);
    times.push_back(ms(start));noc.push_back(times.back()-cat.back());
   }
   auto median=[](std::vector<double> a){std::sort(a.begin(),a.end());return a[a.size()/2];};
   out<<id<<' '<<u<<" 0.0001 "<<engine<<' '<<bins<<' '<<result.magnification<<' '<<std::abs(result.magnification-ref)/std::abs(ref)<<' '<<result.converged<<' '<<median(times)<<' '<<median(noc)<<' '<<median(ct)<<' '<<median(cat)<<'\n';out.flush();
  }
 }
}
