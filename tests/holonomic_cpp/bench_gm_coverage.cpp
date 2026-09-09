// Coverage probe. References are measured AFTER transport and never feed it.
#include <fstream>
#include <sstream>
#include <cstdio>
#include "lcbinint/magnification/holonomic/gm_physical_transport.hpp"
#include "lcbinint/magnification/holonomic/gm_adaptive_transport.hpp"
using namespace lcbinint::holonomic;
template<class Scalar> int run(int argc,char** argv){
    std::ifstream f(argc>1?argv[1]:"/tmp/bench_cases.tsv");
    if(!f)return 2;
    int nr=argc>2?std::atoi(argv[2]):16;
    if(nr<2)return 2;
    std::string line;int total=0,epoch_ok=0,total_nodes=0,transported=0,accurate=0;
    GmAdaptiveCost all;double worst=0,jac_worst=0;int jac_nodes=0,jac_good=0;
    while(std::getline(f,line)){
        std::istringstream in(line);LensParams p;double u,t;int b;std::string name;
        if(!(in>>p.xs>>p.ys>>p.rho>>p.q>>p.a>>b>>u>>t>>name)||u==0)continue;
        p.barycentric=b;auto pf=PrimaryFrame::from(p);auto topo=classify_cells(pf);
        int nodes=0,success=0,good=0,seeds=0;double errmax=0;bool ok=topo.status==Status::OK;
        GmAdaptiveCost cost;
        Cheb1Dyn rule(nr);
        for(auto cell:topo.cells){
            if(cell.kind==ArcKind::kEmpty)continue;
            if(cell.kind!=ArcKind::kArcs){ok=false;continue;}
            double rc=(cell.r_lo+cell.r_hi)*0.5;
            double rh=(cell.r_hi-cell.r_lo)*0.5*(1-2e-9);
            auto arcs=arc_intervals(rc,pf);
            if(arcs.kind!=ArcKind::kArcs){ok=false;continue;}
            for(auto arc:arcs.arcs){
                auto seed=gm_physical_period_seed(rc,pf,arc,512);++seeds;
                if(!seed.ok){ok=false;nodes+=nr;continue;}
                for(int direction:{1,-1}){
                    double r=rc;std::array<Scalar,7> state{};
                    if(!gm_seed_sensitivity(rc,seed,state,512)){ok=false;nodes+=nr/2;continue;}
                    bool alive=true;
                    for(int j=0;j<nr;++j){
                        int n=direction>0?nr-1-j:j;
                        if(rule.x[n]*direction<=0)continue;
                        double target=rc+rh*rule.x[n];++nodes;
                        alive=alive&&gm_adaptive_advance(r,target,cell.r_lo,cell.r_hi,seed.chart_pf,state,cost);
                        if(!alive){ok=false;continue;}
                        ++success;
                        auto hc=gm_physical_h(target,gm_sensitivity_params<Scalar>(seed.chart_pf));Scalar value(0);
                        for(int k=0;k<7;++k)value=value+hc[k]*state[k];
                        value=value/gm_sensitivity_params<Scalar>(seed.chart_pf).rho;
                        double fh=(double)value;
                        auto node_arcs=arc_intervals(target,pf);
                        int ai=gm_transport_detail::nearest_arc(node_arcs,gm_transport_detail::arc_midpoint(arc));
                        if constexpr(std::is_same_v<Scalar,GmDDDual5>){
                            ++jac_nodes;double je=1e300;
                            if(ai>=0){auto geom=gm_physical_period_seed(target,pf,node_arcs.arcs[ai],8);
                                auto evaluate=[&](int count,GmDDDual5& reference){
                                    std::array<GmDDDual5,7> fresh{};
                                    if(!gm_seed_sensitivity(target,geom,fresh,count))return false;
                                    auto pp=gm_sensitivity_params<GmDDDual5>(geom.chart_pf);
                                    auto hh=gm_physical_h(target,pp);reference=GmDDDual5(0);
                                    for(int k=0;k<7;++k)reference=reference+hh[k]*fresh[k];reference=reference/pp.rho;
                                    return true;
                                };
                                auto difference=[&](const GmDDDual5& a,bool ca,const GmDDDual5& b,bool cb){
                                    double e=0;for(int k=0;k<5;++k){
                                        double av=(double)a.deriv[k]*(ca&&(k==0||k==1||k==4)?-1:1);
                                        double bv=(double)b.deriv[k]*(cb&&(k==0||k==1||k==4)?-1:1);
                                        e=std::max(e,std::fabs(av-bv)/(1+std::max(std::fabs(av),std::fabs(bv))));}return e;
                                };
                                GmDDDual5 reference;
                                if(evaluate(512,reference)){
                                    je=difference(value,seed.chart2,reference,geom.chart2);
                                    if(je>=1e-7){GmDDDual5 fine,finer;
                                        if(evaluate(2048,fine)&&evaluate(4096,finer)){
                                            double convergence=difference(fine,false,finer,false);
                                            double refined=difference(value,seed.chart2,finer,geom.chart2);
                                            printf("JAC_REFERENCE_REFINE %s R=%.17g coarse=%.3e refined=%.3e convergence=%.3e\n",name.c_str(),target,je,refined,convergence);
                                            je=convergence<1e-8?refined:1e300;
                                        }
                                    }
                                }
                            }
                            jac_worst=std::max(jac_worst,je);if(je<1e-7)++jac_good;else {ok=false;printf("JACFAIL %s R=%.17g err=%.3e\n",name.c_str(),target,je);}
                        }
                        double angular=0;
                        bool ref=ai>=0&&gm_physical_arc_angular(target,pf,node_arcs.arcs[ai],512,angular);
                        double err=ref?std::fabs(fh-angular)/(1+std::fabs(angular)):1e300;
                        if(ref && err>=1e-11){
                            double fine=0,finer=0;
                            auto reseed=gm_physical_period_seed(target,pf,node_arcs.arcs[ai],2048);
                            bool fine_ok=gm_physical_arc_angular(target,pf,node_arcs.arcs[ai],2048,fine) &&
                                gm_physical_arc_angular(target,pf,node_arcs.arcs[ai],4096,finer);
                            printf("REFERENCE_REFINE %s R=%.17g fh=%.17g angular512=%.17g angular2048=%.17g angular4096=%.17g reseed2048=%.17g\n",name.c_str(),target,fh,angular,fine,finer,reseed.fhalf_arc);
                            ref=fine_ok && std::fabs(fine-finer)/(1+std::fabs(finer))<1e-11;
                            err=ref?std::fabs(fh-finer)/(1+std::fabs(finer)):1e300;
                        }
                        errmax=std::max(errmax,err);
                        if(ref&&err<1e-9)++good;else ok=false;
                    }
                }
            }
        }
        ++total;if(ok&&nodes==good)++epoch_ok;
        total_nodes+=nodes;transported+=success;accurate+=good;worst=std::max(worst,errmax);
        all.connections+=cost.connections;all.connection_failed+=cost.connection_failed;
        all.steps+=cost.steps;all.rejected+=cost.rejected;
        all.connection_ms+=cost.connection_ms;all.transport_ms+=cost.transport_ms;
        printf("%s nodes=%d transported=%d accurate=%d seeds=%d err=%.3e conn=%d/%d steps=%d rejected=%d ms=%.3f/%.3f\n",name.c_str(),nodes,success,good,seeds,errmax,cost.connections-cost.connection_failed,cost.connections,cost.steps,cost.rejected,cost.connection_ms,cost.transport_ms);fflush(stdout);
    }
    printf("TOTAL cases=%d all_true_accurate=%d nodes=%d transported=%d accurate=%d worst=%.3e connections=%d/%d steps=%d rejected=%d connection_ms=%.3f transport_ms=%.3f\n",total,epoch_ok,total_nodes,transported,accurate,worst,all.connections-all.connection_failed,all.connections,all.steps,all.rejected,all.connection_ms,all.transport_ms);
    if constexpr(std::is_same_v<Scalar,GmDDDual5>)printf("ANALYTIC_JAC nodes=%d accurate=%d max_physical_reseed_error=%.3e\n",jac_nodes,jac_good,jac_worst);
    return 0;
}
int main(int argc,char** argv){return argc>3 && std::string(argv[3])=="jac"?run<GmDDDual5>(argc,argv):run<DD>(argc,argv);}
