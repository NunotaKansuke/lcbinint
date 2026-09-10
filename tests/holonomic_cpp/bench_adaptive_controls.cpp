#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"
#include <fstream>
#include <sstream>
#include <cstdio>
using namespace lcbinint::holonomic;
// Fixed 63-node Fejer control on the same restored/mapped radial cells. This
// measures a resolution, and deliberately makes NO tolerance-success claim.
AdaptiveResult mapped_fixed(const LensParams& p,double u,bool jac,const TopologyResult& topo){
    AdaptiveResult r;auto pf=PrimaryFrame::from(p);std::vector<CellPlan> cells;
    if(!adaptive_detail::restore_physical_cuts(topo,pf,cells))return r;
    auto fam=re_detail::p_coeffs_in_R(pf.a,pf.m0,pf.X,pf.Y,pf.rho);
    auto fold=[&](double R){for(auto&e:topo.events)if(e.radius==R&&e.physically_real&&e.kind=="physical_real")return true;return false;};
    std::array<adaptive_detail::Sum,6> sums;const auto& rule=fejer_rule(6);
    for(auto& c:cells)if(c.kind!=ArcKind::kEmpty){
        FoldRadialMap map{c.r_lo,c.r_hi,fold(c.r_lo),fold(c.r_hi)};
        if(map.left)map.a=adaptive_detail::refine_event(map.a,pf,fam).radius;
        if(map.right)map.b=adaptive_detail::refine_event(map.b,pf,fam).radius;
        AdaptiveSample previous;bool seeded=false;
        for(size_t k=0;k<rule.x.size();++k){auto rr=map(rule.x[k]);
            auto v=adaptive_detail::mapped_radius(rr[0],rr[1],p,u,pf,c,jac,topo.from_warm_d14,seeded?&previous:nullptr);
            ++r.stats.node_evaluations;if(!v.reliable)return r;
            for(int j=0;j<(jac?6:1);++j)sums[j].add(rule.w[k]*v.value[j]);previous=std::move(v);seeded=true;}
    }
    r.mu=sums[0].get();for(int j=0;j<5;++j)r.grad_mu[j]=sums[j+1].get();r.numerical_status=Status::OK;return r;
}
int main(int argc,char**argv){if(argc<2)return 2;std::ifstream f(argv[1]);std::string line;(void)fejer_rule(8);
    holo_holonomic_transport_override()=1;holo_mv_transport_override()=1;holo_ode_transport_override()=0;
    puts("name,u,jac,warm,repeat,mode,mu,status,stop,evaluations,whole_ms,topology_ms");
    while(std::getline(f,line)){LensParams p;double u,dummy;int b;std::string name;std::istringstream s(line);if(!(s>>p.xs>>p.ys>>p.rho>>p.q>>p.a>>b>>u>>dummy>>name))continue;p.barycentric=b;
        for(bool jac:{false,true})for(bool warm:{false,true})for(int rep=0;rep<3;++rep)for(int ii=0;ii<3;++ii){
            int mode=rep%2?2-ii:ii;auto runp=p;PreparedEpochGeometry prepared;
            if(warm){(void)prepared_topology(PrimaryFrame::from(p),prepared,PreparedReuseConfig{},nullptr);runp.xs+=.01*p.rho;}
            auto start=adaptive_detail::Clock::now();auto topo=warm?prepared_topology(PrimaryFrame::from(runp),prepared,PreparedReuseConfig{},nullptr):classify_cells(PrimaryFrame::from(runp));double tm=adaptive_detail::ms(start);
            AdaptiveResult r;
            if(mode==0)r=mapped_fixed(runp,u,jac,topo);
            else{AdaptiveConfig c;c.with_jacobian=jac;c.fold_maps=mode==2;c.tol.mu_rtol=1e-4;c.tol.grad_rtol.fill(1e-4);AdaptiveWorkspace w;r=flux_adaptive_integrate(runp,u,topo,c,w);}
            double ms=adaptive_detail::ms(start);
            printf("%s,%g,%d,%d,%d,%s,%.17g,%d,%s,%zu,%.6f,%.6f\n",name.c_str(),u,jac,warm,rep,mode==0?"mapped_fixed63":mode==1?"adaptive_affine":"adaptive_fold",r.mu,int(r.numerical_status),mode?adaptive_stop_name(r.stop):"fixed_resolution",r.stats.node_evaluations,ms,tm);fflush(stdout);
        }
    }
}
