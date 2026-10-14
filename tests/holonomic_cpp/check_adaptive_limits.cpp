#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"
#include <fstream>
#include <sstream>
#include <cstdio>
using namespace lcbinint::holonomic;
int main(int argc,char**argv){if(argc<2)return 2;std::ifstream f(argv[1]);std::string line;
    puts("name,u,rtol,stop,component,Q,budget,error,radial,inner,geometry,event,roundoff,nodes");
    while(std::getline(f,line)){LensParams p;double u,unused;int b;std::string name;std::istringstream s(line);if(!(s>>p.xs>>p.ys>>p.rho>>p.q>>p.a>>b>>u>>unused>>name))continue;p.barycentric=b;
        for(double tol:{1e-3,1e-4,1e-5,1e-6,1e-8}){AdaptiveConfig c;c.with_jacobian=true;c.tol.mu_rtol=tol;c.tol.grad_rtol.fill(tol);AdaptiveWorkspace w;auto r=epoch_adaptive(p,u,c,w);
            for(int j=0;j<6;++j){double q=j?r.grad_mu[j-1]:r.mu,e=j?r.estimated_abs_error_grad[j-1]:r.estimated_abs_error_mu;
                printf("%s,%g,%g,%s,%d,%.17g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%zu\n",name.c_str(),u,tol,adaptive_stop_name(r.stop),j,q,c.tol.budget(j,q),e,r.radial_error[j],r.inner_error[j],r.geometry_error[j],r.event_error[j],r.roundoff_error[j],r.stats.unique_nodes);}
        }
    }
}
