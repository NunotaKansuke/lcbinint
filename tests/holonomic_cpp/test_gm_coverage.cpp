#include <cstdio>
#include "lcbinint/magnification/holonomic/gm_coverage_epoch.hpp"
using namespace lcbinint::holonomic;
int main(){
    int fail=0;auto check=[&](bool ok,const char* msg){if(!ok){++fail;fprintf(stderr,"FAIL %s\n",msg);}};
    LensParams p{.2,1.0/7,.125,.5,1.2,false};auto pf=PrimaryFrame::from(p);
    double R=0;for(auto c:classify_cells(pf).cells)if(c.kind==ArcKind::kArcs&&c.r_hi-c.r_lo>.02){R=(c.r_lo+c.r_hi)/2;break;}auto as=arc_intervals(R,pf);check(as.kind==ArcKind::kArcs,"physical arc exists");
    for(auto arc:as.arcs){
        auto geom=gm_physical_period_seed(R,pf,arc,512);check(geom.ok,"geometry seed");if(!geom.ok)continue;
        auto cp=geom.chart_pf;
        auto ddp=gm_sensitivity_params<DD>(cp);
        auto old=gm_connection_jet<3,DD>(R,ddp,.001);
        auto direct=gm_direct_connection_jet<3,DD>(R,ddp,.001);
        check(old.ok&&direct.ok,"direct/old connection gates");
        double err=0;for(int i=0;i<7;++i)for(int j=0;j<7;++j)for(int k=0;k<4;++k)
            err=std::max(err,gm_abs_value(old.C[i][j].c[k]-direct.C[i][j].c[k])/(1+gm_abs_value(old.C[i][j].c[k])));
        check(err<1e-12,"direct Hermite vs modular reduction");
        std::array<GmDDDual5,7> state{};check(gm_seed_sensitivity(R,geom,state),"analytic physical seed");
        GmAdaptiveCost cost;double r=R,target=R+.001;
        check(gm_adaptive_advance(r,target,R-.01,R+.01,cp,state,cost),"analytic state transport");
        auto h=gm_physical_h(target,gm_sensitivity_params<GmDDDual5>(cp));GmDDDual5 fh(0);
        for(int k=0;k<7;++k)fh=fh+h[k]*state[k];fh=fh/GmDDDual5::variable(2,cp.rho);
        // Independent angular expression at perturbed parameters. FD is an
        // oracle only, never the implemented Jacobian or the transported seed.
        for(int j=0;j<5;++j){double prev=0;
            for(double delta:{2e-6,1e-6}){
                double vals[2]{};for(int side=0;side<2;++side){auto pert=cp;
                    double* ps[]={&pert.X,&pert.Y,&pert.rho,&pert.m0,&pert.a};*ps[j]+=(side?1:-1)*delta;
                    auto arcs=arc_intervals(target,pert);
                    double mid=gm_transport_detail::arc_midpoint(arc)-(geom.chart2?M_PI:0);
                    int ai=gm_transport_detail::nearest_arc(arcs,mid);
                    check(ai>=0&&gm_physical_arc_angular(target,pert,arcs.arcs[ai],512,vals[side]),"angular FD oracle");}
                double fd=(vals[1]-vals[0])/(2*delta),ad=(double)fh.deriv[j];
                double rel=std::fabs(fd-ad)/(1+std::fabs(ad));
                printf("param=%d step=%.1e analytic=%.12g fd=%.12g error=%.3e\n",j,delta,ad,fd,rel);
                check(rel<2e-6,"analytic sensitivity vs angular FD");prev=fd;
            }(void)prev;
        }
        auto invalid=state;double bad=R;
        check(!gm_adaptive_advance(bad,R+.001,R,R+.01,cp,invalid,cost),"boundary fails closed");
    }
    pf=PrimaryFrame::from(LensParams{.05,.02,.05,.3,.9,false});
    int crossed=0;
    for(auto event:radial_events(pf,nullptr)){
        if(event.kind!="chart_p4")continue;
        double delta=1e-5*std::max(1.0,event.radius),r0=event.radius-delta;
        auto arcs=arc_intervals(r0,pf);
        for(auto arc:arcs.arcs){
            auto seed=gm_physical_period_seed(r0,pf,arc,512);
            if(!seed.ok||!seed.chart2)continue;
            std::array<DD,7> state{};check(gm_seed_sensitivity(r0,seed,state),"cross-chart seed");
            double r=r0;GmAdaptiveCost cost;
            for(double target:{event.radius,event.radius+delta}){
                bool moved=gm_adaptive_advance(r,target,r0-10*delta,event.radius+11*delta,seed.chart_pf,state,cost);
                check(moved,"reflected chart crosses original p4 without reseed");
                auto h=gm_physical_h(target,gm_sensitivity_params<DD>(seed.chart_pf));DD value(0);
                for(int k=0;k<7;++k)value=value+h[k]*state[k];
                auto at=arc_intervals(target,seed.chart_pf);int ai=gm_transport_detail::nearest_arc(at,gm_transport_detail::arc_midpoint(arc)-M_PI);
                double ref=0;bool valid=ai>=0&&gm_physical_arc_angular(target,seed.chart_pf,at.arcs[ai],1024,ref);
                printf("cross R=%.17g gm=%.17g angular=%.17g valid=%d\n",target,(double)(value/DD(pf.rho)),ref,valid);
                check(valid&&std::fabs((double)(value/DD(pf.rho))-ref)/(1+std::fabs(ref))<1e-9,"physical flux across degree drop");
            }
            ++crossed;
        }
    }
    check(crossed>0,"at least one physical chart_p4 crossing exercised");
    // Independent original-lens angular reference (2048 nodes), recorded by
    // bench_gm_angular_jac in gm_angular_jac2048_phase3.csv. Its 512/2048
    // derivative convergence is 2.64e-11 for this same resonant geometry.
    auto epoch=gm_coverage_epoch<GmDDDual5>(LensParams{.05,.02,.05,.3,.9,false},.5,64);
    const double reference_mu=13.878583451858105;
    const std::array<double,5> reference_grad={166.72230372213033,-122.35613535034926,-85.536829288138392,-44.326432549978414,-2.9022822105981003};
    check(epoch.local_status==Status::OK&&epoch.nodes==epoch.transported,"whole epoch no-reseed coverage");
    check(epoch.status==Status::GRADIENT_UNRELIABLE,"uncertified analytic lane fails closed");
    check(std::fabs(epoch.mu-reference_mu)/(1+std::fabs(reference_mu))<1e-10,"whole epoch independent angular value");
    double whole_error=0;
    for(int j=0;j<5;++j)whole_error=std::max(whole_error,std::fabs(epoch.grad_mu[j]-reference_grad[j])/(1+std::fabs(reference_grad[j])));
    printf("whole analytic Jacobian angular-reference error=%.3e nodes=%d/%d\n",whole_error,epoch.transported,epoch.nodes);
    check(whole_error<1e-7,"whole epoch independent angular analytic Jacobian");
    printf("%s\n",fail?"FAIL":"PASS");return fail?1:0;
}
