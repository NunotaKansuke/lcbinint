// Independent lens-equation angular derivative reference; no GM connection.
#include <fstream>
#include <sstream>
#include <cstdio>
#include "lcbinint/magnification/holonomic/gm_coverage_epoch.hpp"
using namespace lcbinint::holonomic;
int main(int argc,char** argv){
    std::ifstream file(argc>1?argv[1]:"/tmp/bench_cases.tsv");if(!file)return 2;
    int nt=argc>2?std::atoi(argv[2]):512;if(nt<8)return 2;
    std::string line;
    holo_mv_transport_override()=1;holo_holonomic_transport_override()=1;
    printf("name,u,nt,mu_error,jac_error,ref_mu,ref_dx,ref_dy,ref_drho,ref_dq,ref_da\n");
    while(std::getline(file,line)){
        std::istringstream in(line);LensParams p;int b;double u,t;std::string name;
        if(!(in>>p.xs>>p.ys>>p.rho>>p.q>>p.a>>b>>u>>t>>name)||u==0)continue;
        p.barycentric=b;auto pf=PrimaryFrame::from(p);auto topo=classify_cells(pf);
        auto uniform=gm_coverage_epoch_from_topology<GmDDDual5>(p,0,64,topo);
        Cheb1Dyn rr(64);double Fh=0;std::array<double,5> dFh{};bool ok=true;
        for(auto c:topo.cells){
            if(c.kind==ArcKind::kEmpty)continue;if(c.kind!=ArcKind::kArcs){ok=false;continue;}
            double rc=(c.r_lo+c.r_hi)/2,rh=(c.r_hi-c.r_lo)/2*(1-2e-9);
            for(int n=0;n<64;++n){double r=rc+rh*rr.x[n],wr=rh*rr.w[n];
                auto arcs=arc_intervals(r,pf);
                for(auto arc:arcs.arcs){double hi=arc[1];if(hi<=arc[0])hi+=2*M_PI;
                    double mid=(arc[0]+hi)/2,half=(hi-arc[0])/2;
                    for(int j=0;j<nt;++j){double theta=(j+.5)*M_PI/nt,x=std::cos(theta),s=std::sin(theta);
                        auto g=phi_val_dP(r,mid+half*x,pf);
                        if(!(g.phi>0)){ok=false;continue;}
                        double sq=std::sqrt(g.phi),weight=wr*r*half*M_PI/nt*s;
                        Fh+=weight*sq;
                        for(int k=0;k<5;++k)dFh[k]+=weight*g.dP[k]/(2*sq);
                    }
                }
            }
        }
        double D=M_PI*p.rho*p.rho*(1-u/3),mu=((1-u)*uniform.F0+u*Fh)/D;
        auto grad=internal_to_user_jac(dFh,p);
        // Undo the uniform rho normalization before blending its F0 derivative.
        auto g0=uniform.grad_mu;g0[2]+=2*uniform.mu/p.rho;
        for(int k=0;k<5;++k)grad[k]=((1-u)*g0[k]*M_PI*p.rho*p.rho+u*grad[k])/D;
        grad[2]-=2*mu/p.rho;
        auto v2=epoch_jacobian(p,u,64);double err=0;
        for(int k=0;k<5;++k)err=std::max(err,std::fabs(grad[k]-v2.grad_mu[k])/(1+std::max(std::fabs(grad[k]),std::fabs(v2.grad_mu[k]))));
        printf("%s,%.1f,%d,%.6e,%.6e,%.17g",name.c_str(),u,nt,std::fabs(mu-v2.mu)/(1+std::fabs(v2.mu)),ok?err:1e300,mu);
        for(auto g:grad)printf(",%.17g",g);printf("\n");fflush(stdout);
    }
}
