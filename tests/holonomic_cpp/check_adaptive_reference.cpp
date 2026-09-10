#include "adaptive_reference.hpp"
int main(int argc,char**argv){if(argc<2)return 2;std::ifstream f(argv[1]);std::string line;puts("name,u,rtol,stop,mu,error,reference128,reference256,reference_uncertainty,actual_difference,reference_usable,violation");
    while(std::getline(f,line)){LensParams p;double u,dummy;int b;std::string name;std::istringstream s(line);if(!(s>>p.xs>>p.ys>>p.rho>>p.q>>p.a>>b>>u>>dummy>>name))continue;p.barycentric=b;
        auto topo=classify_cells(PrimaryFrame::from(p));double lo=reference(p,u,topo,128),hi=reference(p,u,topo,256),unc=std::fabs(hi-lo);AdaptiveWorkspace w;
        for(double tol:{1e-3,1e-4,1e-5,1e-6,1e-8}){AdaptiveConfig c;c.tol.mu_rtol=tol;auto r=flux_adaptive_integrate(p,u,topo,c,w);double diff=std::fabs(r.mu-hi),T=c.tol.budget(0,r.mu);int usable=std::isfinite(lo)&&std::isfinite(hi)&&unc<T;int violation=usable?(r.value_converged&&diff>T+unc):-1;
            printf("%s,%g,%g,%s,%.17g,%.9g,%.17g,%.17g,%.9g,%.9g,%d,%d\n",name.c_str(),u,tol,adaptive_stop_name(r.stop),r.mu,r.estimated_abs_error_mu,lo,hi,unc,diff,usable,violation);fflush(stdout);}}
}
