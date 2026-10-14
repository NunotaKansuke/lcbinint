#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
using namespace lcbinint::holonomic;
int main(int argc,char**argv){
 if(argc!=2)return 2;std::ifstream in(argv[1]);std::string line;
 std::cout<<std::setprecision(17);
 while(std::getline(in,line)){
  LensParams p;double u,dummy;int bary;std::string name;
  std::istringstream s(line);if(!(s>>p.xs>>p.ys>>p.rho>>p.q>>p.a>>bary>>u>>dummy>>name))continue;p.barycentric=bary;
  for(double tol:{1e-3,1e-4}){
   AdaptiveConfig cfg;if(std::getenv("P65_SAFETY"))cfg.nested_difference_safety=1.;cfg.tol.mu_rtol=tol;cfg.gradient_policy=GradientPolicy::ValueFirst;
   cfg.value_first_gradient_node_budget=0;cfg.value_first_gradient_round_budget=0;
   AdaptiveWorkspace w;PreparedEpochGeometry state;
   (void)epoch_adaptive_prepared(p,u,cfg,w,state);
   auto r=epoch_adaptive_prepared(p,u,cfg,w,state);
   std::cout<<name<<' '<<u<<' '<<tol<<' '<<r.mu<<' '<<r.value_converged<<' '<<int(r.stop)<<' '<<r.stats.unique_nodes;
   for(int j=0;j<5;++j)std::cout<<' '<<r.grad_mu[j]<<' '<<int(r.grad_quality[j]);
   std::cout<<'\n';
  }
 }
}
