#include <fstream>
#include <sstream>
#include <cstdio>
#include "lcbinint/magnification/holonomic/gm_coverage_epoch.hpp"
using namespace lcbinint::holonomic;
using Clock=std::chrono::steady_clock;
double rel(double a,double b){return std::fabs(a-b)/(1+std::max(std::fabs(a),std::fabs(b)));}
int main(int argc,char** argv){
    if(argc<2)return 2;std::ifstream f(argv[1]);if(!f)return 2;
    int nr=argc>2?std::atoi(argv[2]):64,epochs=argc>3?std::atoi(argv[3]):2;
    bool jac=argc>4?std::atoi(argv[4]):false;int limit=argc>5?std::atoi(argv[5]):108;
    if(nr<2||nr%2||epochs<1)return 2;
    holo_mv_transport_override()=1;holo_holonomic_transport_override()=1;
    std::string line;int cases=0;
    printf("mode,name,u,epoch,nr,gm_ms,v2_ms,gm_status,v2_status,mu_error,jac_error,topology_ms,seed_ms,connection_ms,transport_ms,arc_ms,reseed_ms,seeds,nodes,transported,connection_attempts,connection_failed,steps,rejected\n");
    while(cases<limit&&std::getline(f,line)){
        std::istringstream in(line);LensParams base;double u,t;int b;std::string name;
        if(!(in>>base.xs>>base.ys>>base.rho>>base.q>>base.a>>b>>u>>t>>name))continue;
        base.barycentric=b;++cases;
        for(bool warm:{false,true}){
            PreparedEpochGeometry v2cache;std::vector<Cplx<__float128>> gmroots;
            for(int epoch=0;epoch<epochs;++epoch){
                // Identical nonzero source displacement for cold and warm.
                LensParams p=base;p.xs+=epoch*base.rho*1e-3;p.ys+=epoch*base.rho*3e-4;
                auto t0=Clock::now();
                GmCoverageEpoch g=jac?gm_coverage_epoch<GmDDDual5>(p,u,nr,warm?&gmroots:nullptr):gm_coverage_epoch<DD>(p,u,nr,warm?&gmroots:nullptr);
                double gm_ms=std::chrono::duration<double,std::milli>(Clock::now()-t0).count();
                t0=Clock::now();double v,je=0;Status status;
                if(jac){auto incumbent=warm?epoch_jacobian_prepared(p,u,nr,v2cache,PreparedReuseConfig{}):epoch_jacobian(p,u,nr);
                    v=incumbent.mu;status=incumbent.status;
                    for(int j=0;j<5;++j)je=std::max(je,rel(g.grad_mu[j],incumbent.grad_mu[j]));
                }else{auto incumbent=warm?epoch_value_prepared(p,u,nr,v2cache,PreparedReuseConfig{}):epoch_value(p,u,nr);
                    v=incumbent.mu;status=incumbent.status;}
                double v2_ms=std::chrono::duration<double,std::milli>(Clock::now()-t0).count();
                printf("%s,%s,%.1f,%d,%d,%.6f,%.6f,%d,%d,%.6e,%.6e,%.6f,%.6f,%.6f,%.6f,%.6f,0,%d,%d,%d,%d,%d,%d,%d\n",warm?(epoch?"warm":"warm_init"):"cold",name.c_str(),u,epoch,nr,gm_ms,v2_ms,(int)g.status,(int)status,rel(g.mu,v),je,g.topology_ms,g.seed_ms,g.transport.connection_ms,g.transport.transport_ms,g.arc_ms,g.seeds,g.nodes,g.transported,g.transport.connections,g.transport.connection_failed,g.transport.steps,g.transport.rejected);fflush(stdout);
            }
        }
    }
}
