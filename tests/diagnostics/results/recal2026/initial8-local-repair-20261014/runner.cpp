#include "lcbinint/magnification/finite_source_magnifier.hpp"
#include "lcbinint/magnification/point_source_magnifier.hpp"
#ifdef ADAPTIVE
#include "lcbinint/magnification/probe_diagnostics.hpp"
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>
using namespace lcbinint;
using namespace lcbinint::magnification;
int main(int argc,char**argv){
 if(argc!=4)return 2;
 bool adaptive_mode=std::string(argv[3])=="adaptive";
#ifdef ADAPTIVE

#else
 if(adaptive_mode)return 3;
#endif
 std::ifstream input(argv[1]);std::ofstream out(argv[2]);
 out<<"sample_id\tmethod\tnbin\ttarget\tmagnification\tms\tconverged\trefinement_level\tmaps\trefinements\tindicator\tscale_ms\tstop_reason\tcoarse_guards\tcoarse_probes\tcap_adopted\tcap_trials\tcoarse_merges\treused_fine_rows\n"<<std::setprecision(17);
 std::string line;
 while(std::getline(input,line)){
  int key;double s,q,rho,x,y,c;std::istringstream ss(line);
  if(!(ss>>key>>s>>q>>rho>>x>>y>>c))continue;
  SourcePosition source{x,y};PointSourceMagnifier point;auto imgs=point.binary_images(s,1/q,source);
  double pm=0;std::vector<SourcePosition>centres;for(auto im:imgs){pm+=1/std::abs(im.jacobian_determinant);centres.push_back(im.position);}
  FiniteSourceSettings settings;settings.source_bins=16;if(const char* bins=std::getenv("LCBININT_BENCH_INITIAL_BINS"))settings.source_bins=std::stoi(bins);settings.caustic_bins=1400;settings.grid_ratio=4;settings.finite_mode=1;settings.kinji_threshold=0;settings.hex_threshold=0;settings.adaptive_hex_threshold=0;settings.limb_darkening_c=c;settings.limb_darkening_d=0;settings.automatic_source_bins=false;settings.max_source_bins=settings.source_bins;
  if(const char* ratio=std::getenv("LCBININT_BENCH_POLAR_RATIO"))settings.polar_grid_ratio=std::stod(ratio);
  FiniteSourceMagnifier mag(settings);
#ifdef ADAPTIVE
  if(adaptive_mode){
   auto call=[&](double tol){AdaptiveFluxRequest w;w.target=tol*3.141592653589793*rho*rho*(1-c/3);
#ifdef RELATIVE_SINGLE_PASS
    if(tol<1)w.relative_target=tol;
    if(const char* cap=std::getenv("LCBININT_CACHE_CAPACITY"))w.mapped.initial_capacity=std::stoul(cap);
#endif
    if(const char* rows=std::getenv("LCBININT_BENCH_START_GUARD_ROWS"))w.coarse_start_min_rows=std::stoi(rows);
    w.max_bins=512;w.budget=1000000;if(const char* budget=std::getenv("LCBININT_BENCH_MAP_BUDGET"))w.budget=std::stoull(budget);w.local_vertical_caps=true;w.defect_correction=true;
    auto*prior=adaptive_flux_request();bool old=lightweight_flux_error_mode();auto saved=probe_counters();adaptive_flux_request()=&w;lightweight_flux_error_mode()=true;
    try{auto r=mag.binary_mag(s,1/q,source,rho,pm,&centres,true,&point);if(!w.executed)throw std::runtime_error("adaptive kernel was not executed");adaptive_flux_request()=prior;lightweight_flux_error_mode()=old;probe_counters()=saved;return std::make_pair(r.magnification,std::move(w));}
    catch(...){adaptive_flux_request()=prior;lightweight_flux_error_mode()=old;probe_counters()=saved;throw;}
   };
   std::vector<double> scale_times;double scale=0;
#ifdef RELATIVE_SINGLE_PASS
    scale_times.assign(3,0);
#else
    call(1e100);
    for(int rep=0;rep<3;++rep){auto t0=std::chrono::steady_clock::now();auto a=call(1e100);scale=std::abs(a.first);scale_times.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count());}
    std::sort(scale_times.begin(),scale_times.end());
#endif
    for(double target:{1e-3,1e-4}){
     try{call(
#ifdef RELATIVE_SINGLE_PASS
     target
#else
     target*scale
#endif
     );std::vector<double>times;decltype(call(1.))r;
     for(int rep=0;rep<3;++rep){auto t0=std::chrono::steady_clock::now();r=call(
#ifdef RELATIVE_SINGLE_PASS
     target
#else
     target*scale
#endif
     );times.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count());}
     std::sort(times.begin(),times.end());out<<key<<"\tadaptive\t"<<settings.source_bins<<"\t"<<target<<'\t'<<r.first<<'\t'<<times[1]<<'\t'<<r.second.met<<"\t0\t"<<r.second.new_maps<<'\t'<<r.second.refinements<<'\t'<<r.second.acceptance_indicator/(3.141592653589793*rho*rho*(1-c/3))<<'\t'<<scale_times[1]<<'\t'<<r.second.stop_reason<<'\t'<<r.second.coarse_start_guards<<'\t'<<r.second.coarse_start_probe_calls<<'\t'<<r.second.cap_adopted<<'\t'<<r.second.cap_trials<<'\t'<<r.second.coarse_start_merges<<'\t'<<r.second.reused_cap_rows<<'\n';
     }catch(const std::exception&e){out<<key<<"\tadaptive\t16\t"<<target<<"\tnan\tnan\t0\t0\t0\t0\tnan\tnan\texception\t0\t0\t0\t0\n";std::cerr<<key<<' '<<target<<' '<<e.what()<<'\n';}
     out.flush();
    }
   continue;
  }
#endif
  for(auto method:{FiniteSourceMethod::inverse_ray_polar}){
   const char*name=method==FiniteSourceMethod::inverse_ray_cartesian?"cartesian":"polar";
   for(int bins:{8,16,32,64}){
    auto call=[&](){return mag.binary_mag_preplanned(s,1/q,source,rho,pm,method,bins,&centres,&point);};
    try{call();std::vector<double>times;FiniteSourceResult r;
     for(int rep=0;rep<3;++rep){auto t0=std::chrono::steady_clock::now();r=call();times.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count());}
     std::sort(times.begin(),times.end());out<<key<<'\t'<<name<<'\t'<<bins<<"\t0\t"<<r.magnification<<'\t'<<times[1]<<'\t'<<r.converged<<'\t'<<r.refinement_level<<"\t0\t0\t"<<r.error_estimate<<"\t0\tcompleted\n";out.flush();
     if(times[1]>500){std::cerr<<key<<' '<<name<<" cost_cap bins="<<bins<<'\n';break;}
    }catch(const std::exception&e){out<<key<<'\t'<<name<<'\t'<<bins<<"\t0\tnan\tnan\t0\t0\t0\t0\t0\t0\texception\t0\t0\t0\t0\n";out.flush();std::cerr<<key<<' '<<name<<' '<<bins<<' '<<e.what()<<'\n';}
   }
  }
 }
}
