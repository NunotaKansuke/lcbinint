// Cold arc-centre connection coverage; no transport acceptance claim.
#include <fstream>
#include "lcbinint/magnification/holonomic/gm_direct_connection.hpp"
#include <sstream>
#include <cstdio>
#include "lcbinint/magnification/holonomic/gm_physical_transport.hpp"
using namespace lcbinint::holonomic;
int main(int argc,char** argv){std::ifstream f(argc>1?argv[1]:"/tmp/bench_cases.tsv");if(!f)return 2;std::string l;int n=0,raw=0,scaled=0,pt=0,quad=0;while(std::getline(f,l)){std::istringstream s(l);LensParams p;double u,t;int b;std::string name;if(!(s>>p.xs>>p.ys>>p.rho>>p.q>>p.a>>b>>u>>t>>name)||u==0)continue;p.barycentric=b;auto pf=PrimaryFrame::from(p);auto topo=classify_cells(pf);for(auto c:topo.cells){if(c.kind!=ArcKind::kArcs)continue;double r=(c.r_lo+c.r_hi)/2,h=(c.r_hi-c.r_lo)/2;auto as=arc_intervals(r,pf);for(auto a:as.arcs){auto seed=gm_physical_period_seed(r,pf,a);if(!seed.ok)continue;auto z=seed.chart_pf;GmParams<DD> pp{DD(z.X),DD(z.Y),DD(z.rho),DD(z.m0),DD(z.a)};++n;auto x=gm_connection_jet<8,DD>(r,pp);auto y=gm_connection_jet<8,DD>(r,pp,h);auto v=gm_connection_jet<0,DD>(r,pp,h);auto q=gm_direct_connection_jet<8,__float128>(r,{z.X,z.Y,z.rho,z.m0,z.a},h);raw+=x.ok;scaled+=y.ok;pt+=v.ok;quad+=q.ok;if(!y.ok)printf("fail %s r %.17g h %.3g raw %.3g norm %.3g pivot %.3g q %.3g\n",name.c_str(),r,h,x.identity_residual,y.identity_residual,y.matrix_pivot_rel,q.identity_residual);}}}printf("N %d raw_DD %d scaled_DD %d scaled_point_DD %d direct_quad %d\n",n,raw,scaled,pt,quad);}
