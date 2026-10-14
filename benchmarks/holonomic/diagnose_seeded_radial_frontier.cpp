#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include "lcbinint/magnification/holonomic/seeded_radial_frontier.hpp"
using namespace lcbinint::holonomic;
int main(int argc,char**argv) {
 if(argc!=3)return 2;std::ifstream in(argv[1]);std::string line;
 while(std::getline(in,line)) {
  std::istringstream is(line);int id,support,n;LensParams p;double ms;
  if(!(is>>id>>p.xs>>p.ys>>p.rho>>p.q>>p.a>>support>>ms>>n))continue;
  std::vector<Cplx<double>> seeds(n);for(auto&z:seeds)is>>z.re>>z.im;
  if(id!=std::atoi(argv[2]))continue;
  std::cout<<std::setprecision(17);const auto pf=PrimaryFrame::from(p);
  for(auto z:seeds) {
   double R=std::hypot(z.re,z.im);auto c=certify_quartic(seeded_boundary_detail::coeff(R,pf,false));
   std::cout<<"seed "<<R<<" cert "<<c.certified<<" count "<<c.root_count<<'\n';
  }
  AdaptiveConfig cfg;cfg.tol.mu_rtol=1e-4;cfg.collect_diagnostics=true;
  AdaptiveWorkspace w;auto x=epoch_seeded_radial_frontier(p,0,seeds,cfg,w);
  std::cout<<"stop "<<int(x.integral.stop)<<" probes "<<x.frontier.radius_probes<<'\n';
  for(auto c:x.topology.cells)std::cout<<"cell "<<c.r_lo<<' '<<c.r_hi<<'\n';
  for(auto d:x.integral.stats.sample_diagnostics)std::cout<<"reject "<<d.R<<' '<<int(d.reason)<<'\n';
  auto ref=classify_cells(pf,nullptr,nullptr,true);
  for(auto e:ref.events)if(e.physically_real)std::cout<<"reference fold "<<e.radius<<'\n';
 }
}
