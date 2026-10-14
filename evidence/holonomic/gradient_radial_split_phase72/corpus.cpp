#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace lcbinint::holonomic;
using Clock=std::chrono::steady_clock;

namespace {
struct Timed {
    AdaptiveResult result;
    double ms=0;
};
Timed cold_run(const LensParams& p,double u,const AdaptiveConfig& cfg,
               AdaptiveWorkspace& w) {
    const auto a=Clock::now();
    auto r=epoch_adaptive(p,u,cfg,w);
    return {std::move(r),std::chrono::duration<double,std::milli>(Clock::now()-a).count()};
}
Timed warm_run(const LensParams& p,double u,const AdaptiveConfig& cfg,
               AdaptiveWorkspace& w,PreparedEpochGeometry& state) {
    const auto a=Clock::now();
    auto r=epoch_adaptive_prepared(p,u,cfg,w,state);
    return {std::move(r),std::chrono::duration<double,std::milli>(Clock::now()-a).count()};
}
AdaptiveConfig config(double value_rtol,bool local) {
    AdaptiveConfig c;
    c.gradient_policy=GradientPolicy::ValueFirst;
    c.tol.mu_rtol=value_rtol;
    c.gradient_local_refinement=local;
    c.value_first_gradient_round_budget=4;
    c.value_first_gradient_node_budget=4096;
    return c;
}
void emit(int rep,int row,const std::string& name,double u,double rtol,
          const char* path,int split_level,const Timed& t) {
    const auto& r=t.result;
    std::cout<<rep<<'\t'<<row<<'\t'<<name<<'\t'<<u<<'\t'<<rtol<<'\t'
             <<path<<'\t'<<split_level<<'\t'<<t.ms<<'\t'<<r.value_converged<<'\t'
             <<adaptive_stop_name(r.value_stop_reason)<<'\t'
             <<adaptive_stop_name(r.gradient_stop_reason)<<'\t'<<r.mu<<'\t'
             <<r.stats.unique_nodes<<'\t'<<r.stats.node_evaluations<<'\t'
             <<r.stats.splits;
    for(int j=0;j<5;++j)
        std::cout<<'\t'<<r.grad_mu[j]<<'\t'<<r.estimated_abs_error_grad[j]
                 <<'\t'<<gradient_quality_name(r.grad_quality[j])
                 <<'\t'<<gradient_reason_name(r.grad_reason[j]);
    for(int j=0;j<6;++j)
        std::cout<<'\t'<<r.radial_error[j]<<'\t'<<r.inner_error[j]
                 <<'\t'<<r.geometry_error[j]<<'\t'<<r.event_error[j]
                 <<'\t'<<r.roundoff_error[j];
    std::cout<<'\n';
}
}

int main(int argc,char** argv) {
    if(argc!=2)return 2;
    std::ifstream in(argv[1]);
    std::cout<<"rep\trow\tname\tu\tvalue_rtol\tpath\th_split_min_level\tms\tvalue_converged\tvalue_stop\tgradient_stop\tmu\tnodes\tevals\tsplits";
    for(int j=0;j<5;++j)std::cout<<"\tgrad"<<j<<"\terror"<<j<<"\tquality"<<j<<"\treason"<<j;
    for(int j=0;j<6;++j)std::cout<<"\tradial"<<j<<"\tinner"<<j<<"\tgeometry"<<j<<"\tevent"<<j<<"\troundoff"<<j;
    std::cout<<'\n'<<std::setprecision(17);
    std::string line;int row=0;
    while(std::getline(in,line)) {
        ++row;
        LensParams p;double u,dummy;int bary;std::string name;
        std::istringstream ss(line);
        if(!(ss>>p.xs>>p.ys>>p.rho>>p.q>>p.a>>bary>>u>>dummy>>name))continue;
        p.barycentric=bary;
        for(double rtol:{1e-3,1e-4}) {
            AdaptiveConfig cfg[4]={config(rtol,false),config(rtol,true),
                                   config(rtol,true),config(rtol,true)};
            cfg[1].gradient_split_min_level=4;
            cfg[2].gradient_split_min_level=5;
            cfg[3].gradient_split_min_level=8; // estimator control; h-split only at max level
            AdaptiveWorkspace cold_w[4],warm_w[4];
            PreparedEpochGeometry warm_state[4];
            // Match the same-point warmed D14-root seed used by the prior Phase69 corpus.
            for(int a=0;a<4;++a)
                (void)epoch_adaptive_prepared(p,u,cfg[a],warm_w[a],warm_state[a]);
            for(int rep=0;rep<2;++rep) {
                for(const char* path:{"cold","warm"}) {
                    const bool warm=std::string(path)=="warm";
                    std::array<int,4> order{{0,1,2,3}};
                    const int rotate=(row+rep+(rtol==1e-4)+warm)%4;
                    std::rotate(order.begin(),order.begin()+rotate,order.end());
                    if((row+rep+warm)%2)std::swap(order[1],order[3]);
                    for(int arm:order) {
                        const int split_level=arm==0?0:(arm==1?4:(arm==2?5:8));
                        const Timed t=warm
                            ?warm_run(p,u,cfg[arm],warm_w[arm],warm_state[arm])
                            :cold_run(p,u,cfg[arm],cold_w[arm]);
                        emit(rep,row,name,u,rtol,path,split_level,t);
                    }
                }
            }
        }
    }
}
