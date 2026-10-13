#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include "lcbinint/magnification/finite_source_magnifier.hpp"
#include "lcbinint/magnification/point_source_magnifier.hpp"
#include "lcbinint/magnification/holonomic/seeded_radial_frontier.hpp"
#ifdef HOLO_FUSED_GRADIENT_EXPERIMENT
#include "evaluate.hpp"
#endif
using namespace lcbinint::holonomic;
using Clock=std::chrono::steady_clock;
static double elapsed(Clock::time_point start) {
    return std::chrono::duration<double,std::milli>(Clock::now()-start).count();
}
static double median(std::vector<double> values) {
    std::sort(values.begin(),values.end());return values[values.size()/2];
}
int main(int argc,char**argv) {
    if(argc<3)return 2;
    std::ifstream in(argv[1]);std::ofstream out(argv[2]);
    if(!in||!out)return 2;
    const int reps=argc>3?std::max(1,std::atoi(argv[3])):3;
    const int selection=argc>4?std::atoi(argv[4]):0;
    const bool fast_only=selection!=0;
    out<<std::setprecision(17)<<"case\tprofile\tdbin\tepoch\tmode\tsupport\tmu\tvalue_converged\tstop\tnodes\tseed_ms\tfrontier_ms\ttotal_ms\tgradient_met";
    for(int j=0;j<5;++j)out<<"\tg"<<j<<"\tge"<<j<<"\tgc"<<j;
    out<<'\n';
    std::string line;int count=0;
    while(std::getline(in,line)) {
        if(line.empty()||line[0]=='#')continue;
        std::istringstream input(line);int id,configuration,db,epoch;
        std::string profile;double separation,q,rho,x,y,time,u,X,reference;
        if(!(input>>id>>configuration>>profile>>db>>epoch>>separation>>q>>rho>>x>>y>>time>>u>>X>>reference))return 3;
        const LensParams p{time,y,rho,1/q,separation,true};
        std::array<std::vector<double>,4> times;
        std::array<SeededFrontierResult,4> results;
        std::vector<double> seed_times;
        bool support=false;
        for(int rep=0;rep<reps;++rep) {
            // Match the established benchmark: a fresh magnifier each rep,
            // caustic prebuild excluded, seed/index construction included.
            lcbinint::magnification::PointSourceMagnifier point;
            lcbinint::magnification::FiniteSourceSettings settings;
            settings.caustic_bins=1400;
            lcbinint::magnification::FiniteSourceMagnifier finite(settings);
            finite.binary_caustic_branches(p.a,p.q);
            const auto seed_start=Clock::now();
            std::vector<lcbinint::SourcePosition> center;
            for(const auto& image:point.binary_images(p.a,p.q,{p.xs,p.ys}))center.push_back(image.position);
            const auto positions=finite.cached_binary_image_seeds(point,p.a,p.q,{p.xs,p.ys},p.rho,
                std::numeric_limits<double>::infinity(),&center,&support);
            std::vector<Cplx<double>> seeds;seeds.reserve(positions.size());
            for(auto image:positions)seeds.push_back({image.x+(p.q<1?p.a:0),image.y});
            seed_times.push_back(elapsed(seed_start));
            const std::array<int,4> order=rep%2?std::array<int,4>{3,2,1,0}:std::array<int,4>{0,1,2,3};
            for(int mode:order) {
                if(fast_only&&mode!=0&&mode!=3)continue;
                if(selection==2&&mode!=0)continue;
                AdaptiveConfig cfg;cfg.tol.mu_rtol=1e-4;
                cfg.gradient_policy=mode==0?GradientPolicy::None:mode==1?GradientPolicy::Strict:GradientPolicy::ValueFirst;
                cfg.with_jacobian=mode!=0;
                if(mode==3) {
                    cfg.value_first_gradient_node_budget=0;
                    cfg.value_first_gradient_round_budget=0;
                }
                AdaptiveWorkspace workspace;
                const auto start=Clock::now();
#ifdef HOLO_FUSED_GRADIENT_EXPERIMENT
                if(mode==1||mode==3)results[mode]=fused_gradient::evaluate(p,u,seeds,cfg,workspace,
                    mode==3?fused_gradient::Refinement::ValueMesh:fused_gradient::Refinement::JointTolerance).epoch;
                else
#endif
                    results[mode]=epoch_seeded_radial_frontier(p,u,seeds,cfg,workspace);
                times[mode].push_back(elapsed(start));
            }
        }
        for(int mode=0;mode<4;++mode) {
            if(times[mode].empty())continue;
            const auto& result=results[mode].integral;
            const bool met=mode!=0&&std::all_of(result.grad_converged.begin(),result.grad_converged.end(),[](bool value){return value;});
            out<<id<<'\t'<<profile<<'\t'<<db<<'\t'<<epoch<<'\t'<<mode<<'\t'<<support<<'\t'<<result.mu<<'\t'<<result.value_converged<<'\t'<<adaptive_stop_name(result.stop)<<'\t'<<result.stats.node_evaluations<<'\t'<<median(seed_times)<<'\t'<<median(times[mode])<<'\t'<<median(seed_times)+median(times[mode])<<'\t'<<met;
            for(int j=0;j<5;++j)out<<'\t'<<result.grad_mu[j]<<'\t'<<result.estimated_abs_error_grad[j]<<'\t'<<result.grad_converged[j];
            out<<'\n';
        }
        out.flush();
        if(++count%64==0)std::cerr<<"completed "<<count<<'\n';
    }
    std::cerr<<"finished "<<count<<'\n';
}
