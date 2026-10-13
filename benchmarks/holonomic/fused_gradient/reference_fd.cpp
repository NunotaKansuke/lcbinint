#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include "lcbinint/magnification/finite_source_magnifier.hpp"
#include "lcbinint/magnification/point_source_magnifier.hpp"
#include "lcbinint/magnification/holonomic/seeded_radial_frontier.hpp"
using namespace lcbinint::holonomic;
int main(int argc,char**argv){
 if(argc!=3)return 2;std::ifstream in(argv[1]);std::ofstream out(argv[2]);
 out<<std::setprecision(17)<<"case\tprofile\tdbin\tepoch\tparam\tstep\tfd\tmu_plus\tmu_minus\terror_plus\terror_minus\tconverged_plus\tconverged_minus\tsupport_plus\tsupport_minus\n";
 std::string line;while(std::getline(in,line)){
  std::istringstream is(line);int id,conf,db,ep;std::string profile;double s,q,rho,x,y,t,u,X,ref;
  if(!(is>>id>>conf>>profile>>db>>ep>>s>>q>>rho>>x>>y>>t>>u>>X>>ref))continue;
  const LensParams p{t,y,rho,1/q,s,true};
  lcbinint::magnification::PointSourceMagnifier pm;
  lcbinint::magnification::FiniteSourceSettings fs;fs.caustic_bins=1400;
  lcbinint::magnification::FiniteSourceMagnifier fm(fs);
  auto evaluate=[&](LensParams point){
   std::vector<lcbinint::SourcePosition> center;
   for(const auto& z:pm.binary_images(point.a,point.q,{point.xs,point.ys}))center.push_back(z.position);
   bool support=false;
   auto positions=fm.cached_binary_image_seeds(pm,point.a,point.q,{point.xs,point.ys},point.rho,std::numeric_limits<double>::infinity(),&center,&support);
   std::vector<Cplx<double>> seeds;for(auto z:positions)seeds.push_back({z.x+(point.q<1?point.a:0),z.y});
   AdaptiveConfig cfg;cfg.tol.mu_atol=1e-10;cfg.tol.mu_rtol=1e-8;
   AdaptiveWorkspace w;auto r=epoch_seeded_radial_frontier(point,u,seeds,cfg,w);
   return std::make_pair(r.integral,support);};
  for(int j=0;j<5;++j){
   const double scale=j==3?std::min(p.q,p.rho*(1+p.q)*(1+p.q)/std::max(p.a,p.rho)):p.rho;
   for(double fraction:{0.01,0.005,0.0025}){
    const double h=scale*fraction;auto a=p,b=p;
    auto displace=[&](LensParams& z,double delta){switch(j){case 0:z.xs+=delta;break;case 1:z.ys+=delta;break;case 2:z.rho+=delta;break;case 3:z.q+=delta;break;case 4:z.a+=delta;break;}};
    displace(a,h);displace(b,-h);const auto plus=evaluate(a),minus=evaluate(b);
    out<<id<<'\t'<<profile<<'\t'<<db<<'\t'<<ep<<'\t'<<j<<'\t'<<h<<'\t'<<(plus.first.mu-minus.first.mu)/(2*h)<<'\t'<<plus.first.mu<<'\t'<<minus.first.mu<<'\t'<<plus.first.estimated_abs_error_mu<<'\t'<<minus.first.estimated_abs_error_mu<<'\t'<<plus.first.value_converged<<'\t'<<minus.first.value_converged<<'\t'<<plus.second<<'\t'<<minus.second<<'\n';out.flush();
   }
  }
  std::cerr<<"finished "<<id<<" "<<profile<<" "<<db<<" "<<ep<<'\n';
 }
}
