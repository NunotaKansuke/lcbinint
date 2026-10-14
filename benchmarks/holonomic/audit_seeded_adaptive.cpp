#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include "lcbinint/magnification/holonomic/seeded_boundary_events.hpp"
#include "lcbinint/magnification/holonomic/seeded_epoch.hpp"
using namespace lcbinint::holonomic;
int main(int argc,char**argv){
 if(argc!=3)return 2;
 std::ifstream in(argv[1]);std::ofstream out(argv[2]);std::string line;int failures=0,rows=0;
 if(!in||!out){std::cerr<<"cannot open audit input/output\n";return 2;}
 out<<std::setprecision(17)<<"id u tol seeded_converged d14_converged seeded_stop d14_stop seeded_mu d14_mu relative_difference seeded_nodes d14_nodes\n";
 while(std::getline(in,line)){
  std::istringstream is(line);int id,support,n;LensParams p;double seedms;
  if(!(is>>id>>p.xs>>p.ys>>p.rho>>p.q>>p.a>>support>>seedms>>n))continue;
  std::vector<Cplx<double>> seeds(n);for(auto&z:seeds)is>>z.re>>z.im;
  auto pf=PrimaryFrame::from(p);auto discovery=seeded_boundary_events(pf,seeds);
  auto candidate=seeded_boundary_cell_plan(pf,discovery),reference=classify_cells(pf,nullptr,nullptr,true);
  for(double u:{0.,.5})for(double tolerance:{1e-3,1e-4}){
   AdaptiveConfig cfg;cfg.tol.mu_rtol=tolerance;cfg.gradient_policy=GradientPolicy::None;
   AdaptiveWorkspace a,b;auto x=flux_adaptive_integrate(p,u,candidate,cfg,a),y=flux_adaptive_integrate(p,u,reference,cfg,b);
   double relative=std::fabs(x.mu-y.mu)/std::max(1e-30,std::fabs(y.mu));
   bool pass=x.value_converged&&y.value_converged&&std::isfinite(relative)&&relative<=tolerance;
   ++rows;if(!pass)++failures;
   out<<id<<' '<<u<<' '<<tolerance<<' '<<x.value_converged<<' '<<y.value_converged<<' '<<int(x.stop)<<' '<<int(y.stop)<<' '<<x.mu<<' '<<y.mu<<' '<<relative<<' '<<x.stats.unique_nodes<<' '<<y.stats.unique_nodes<<'\n';
  }
 }
 std::cerr<<"rows="<<rows<<" failed="<<failures<<'\n';return rows==0?2:(failures?1:0);
}
