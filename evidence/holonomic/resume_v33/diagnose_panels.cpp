#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include "lcbinint/magnification/holonomic/seeded_radial_frontier.hpp"
using namespace lcbinint::holonomic;
int main(int argc,char**argv){
 std::ifstream in(argv[1]);std::string line;
 while(std::getline(in,line)){
 std::istringstream is(line);int id,support,n;LensParams p;double ms;
 if(!(is>>id>>p.xs>>p.ys>>p.rho>>p.q>>p.a>>support>>ms>>n))continue;
 std::vector<Cplx<double>> seeds(n);for(auto&z:seeds)is>>z.re>>z.im;
 for(double tol:{1e-3,1e-4}){
 AdaptiveConfig cfg;cfg.tol.mu_rtol=tol;cfg.gradient_policy=GradientPolicy::None;cfg.collect_diagnostics=true;
 AdaptiveWorkspace w;auto x=epoch_seeded_radial_frontier(p,0,seeds,cfg,w,0);
 std::cout<<std::setprecision(17)<<"result "<<id<<' '<<tol<<' '<<x.integral.mu<<" budget "<<cfg.tol.budget(0,x.integral.mu)<<" nodes "<<x.integral.stats.node_evaluations<<" splits "<<x.integral.stats.splits<<'\n';
 for(const auto& h:x.integral.stats.refinement_history)std::cout<<"history "<<h.panel<<" level "<<h.level<<" nodes "<<h.node_evaluations<<" error "<<h.value_error<<" resolved "<<h.value_resolved<<"\n";
 for(size_t i=0;i<w.panels.size();++i){const auto& a=w.panels[i];
 std::cout<<"panel "<<i<<" active "<<a.active<<" weighted "<<a.weighted<<" parent "<<a.parent<<" level "<<a.level<<" R "<<a.map.a<<' '<<a.map.b<<" x "<<a.xl<<' '<<a.xr<<" q "<<a.q[0]<<" radial "<<a.radial[0]<<" inner "<<a.inner[0]<<" geometry "<<a.geometry[0]<<" event "<<a.event[0]<<" resolved "<<a.value_resolved<<'\n';
 }
 }
 }
}
