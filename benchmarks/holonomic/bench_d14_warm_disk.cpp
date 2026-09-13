// Research only: independently correct previous roots and certify current disks.
#include "lcbinint/magnification/holonomic/d14_rouche.hpp"
#include "d14_tight_disk_probe.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
using namespace lcbinint::holonomic;
using namespace lcbinint::holonomic::re_detail;
using Clock=std::chrono::steady_clock;
int main(int argc,char**argv){
 if(argc<2||argc>3)return 2;const bool tight=argc==3&&std::string(argv[2])=="tight";std::ifstream in(argv[1]);int sample;double time,y,rho,q,s;
 std::cout<<std::setprecision(17)<<"sample stage prepare_ms correction_ms screen_ms interval_ms finite screened certified\n";
 while(in>>sample>>time>>y>>rho>>q>>s){
  std::vector<Cplx<D14Real>> roots(14);for(auto& z:roots){double re,im;if(!(in>>re>>im))return 3;z={D14Real(re),D14Real(im)};}
  auto p=PrimaryFrame::from(LensParams{time,y,rho,1/q,s,true});auto start=Clock::now();
  auto sc=d14_struct_build(p.a,p.m0,p.X,p.Y,p.rho);auto sr=d14_struct_cast<D14Real>(sc);
  auto desc=d14_expanded_from_struct(sc);std::reverse(desc.begin(),desc.end());
  const double prepare=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
  bool finite=true;double correction=0;
  for(int stage=0;stage<=2;++stage){
   if(stage){start=Clock::now();for(auto& z:roots){Cplx<D14Real> f,d;d14_struct_eval(sr,z,f,d);auto w=f/d;if(qfinite_(w.re)&&qfinite_(w.im))z=z-w;else finite=false;}
    correction+=std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
   std::vector<Cplx<qf>> candidate;for(auto z:roots)candidate.push_back({d14_to_qf(z.re),d14_to_qf(z.im)});
   start=Clock::now();bool screened=finite&&(tight?tight_fast_screen(desc,candidate):d14_rouche_fast_screen(desc,candidate));
   double screen_ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count(),interval_ms=0;bool certified=false;
   if(screened){start=Clock::now();auto cert=tight?tight_certificate(p,candidate):d14_rouche_certificate(p,candidate);certified=cert.certified;interval_ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
   std::cout<<sample<<' '<<stage<<' '<<prepare<<' '<<correction<<' '<<screen_ms<<' '<<interval_ms<<' '<<finite<<' '<<screened<<' '<<certified<<'\n';
  }
 }
}
