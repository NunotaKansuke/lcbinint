// Research benchmark for the D14-free Vmax primary lane.
//
// The plan is established outside the timed epoch evaluation by an
// independent trajectory warm-up.  Each timed call still performs the point
// image solve and the complete certified inverse-ray kernel.  This models the
// intended steady-state contract: reuse a validated numerical plan, never a
// previously computed magnification.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "lcbinint/magnification/finite_source_magnifier.hpp"
#include "lcbinint/magnification/point_source_magnifier.hpp"
#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"

using namespace lcbinint;
using namespace lcbinint::magnification;
using Clock = std::chrono::steady_clock;
using namespace lcbinint::holonomic;

struct Row {
    int cid=0,cfg=0,db=0,epoch=0,nbin=0;
    std::string profile,method;
    double target=0,s=0,q=0,rho=0,x=0,y=0,time=0,u=0,reference=0;
};

static bool parse(const std::string& line,Row& r) {
    if(line.empty()||line[0]=='#') return false;
    std::istringstream in(line);
    return bool(in>>r.cid>>r.cfg>>r.profile>>r.target>>r.db>>r.epoch
                  >>r.s>>r.q>>r.rho>>r.x>>r.y>>r.time>>r.u>>r.reference
                  >>r.method>>r.nbin);
}

static double median(std::vector<double> v) {
    const auto m=v.begin()+v.size()/2;
    std::nth_element(v.begin(),m,v.end());
    return *m;
}

int main(int argc,char**argv) {
    if(argc<3) {std::cerr<<"usage: bench_vmax_preplanned PLAN OUT [reps]\n";return 2;}
    const int reps=argc>3?std::max(1,std::atoi(argv[3])):3;
    std::ifstream input(argv[1]);std::ofstream output(argv[2]);
    if(!input||!output)return 2;
    std::vector<Row> rows;std::string line;
    while(std::getline(input,line)){Row r;if(parse(line,r))rows.push_back(r);}
    output<<"case_id configuration_id profile target d_bin_index epoch_index "
             "method nbin ms mu reference relerr converged point_images estimated_evaluations "
             "nested_error boundary_backstop\n"
          <<std::setprecision(17);
    PointSourceMagnifier point;
    // Separate magnifiers preserve the correct LD table and seed cache for
    // each physical profile while allowing trajectory-warm cache reuse.
    FiniteSourceSettings uniform_settings,linear_settings;
    uniform_settings.caustic_bins=1400;
    linear_settings.caustic_bins=1400;linear_settings.limb_darkening_c=0.5;
    FiniteSourceMagnifier uniform(uniform_settings),linear(linear_settings);
    volatile double sink=0;
    for(std::size_t k=0;k<rows.size();++k) {
        const Row&r=rows[k];
        auto& finite=r.profile=="linear"?linear:uniform;
        const auto mode=r.method=="polar"?FiniteSourceMethod::inverse_ray_polar:
                                             FiniteSourceMethod::inverse_ray_cartesian;
        std::vector<double> elapsed;FiniteSourceResult keep;int image_count=0;
        double nested_error=0;bool boundary_backstop=false;
        for(int rep=-1;rep<reps;++rep) {
            const auto t0=Clock::now();
            if(r.method=="adaptive") {
                AdaptiveConfig cfg;cfg.gradient_policy=GradientPolicy::None;
                cfg.with_jacobian=false;cfg.tol.mu_atol=1e-16;cfg.tol.mu_rtol=r.target;
                AdaptiveWorkspace workspace;
                const auto a=epoch_adaptive({r.time,r.y,r.rho,1/r.q,r.s,true},r.u,cfg,workspace);
                keep.magnification=a.mu;keep.converged=a.value_converged&&a.numerical_status==Status::OK;
                keep.decision=FiniteSourceDecision{
                    FiniteSourceMethod::source_plane_quadrature,
                    static_cast<int>(a.stats.node_evaluations),
                    "Vmax certified boundary/K-rule backstop"};
                image_count=0;
            } else {
                const auto images=point.binary_images(r.s,1/r.q,{r.time,r.y});
                double point_mu=0;std::vector<SourcePosition> seeds;seeds.reserve(images.size());
                for(const auto&i:images){point_mu+=1/std::abs(i.jacobian_determinant);seeds.push_back(i.position);}
                keep=finite.binary_mag_preplanned(r.s,1/r.q,{r.time,r.y},r.rho,point_mu,
                                                  mode,r.nbin,&seeds,&point);
                image_count=static_cast<int>(images.size());
                const int coarse_n=std::max(1,(r.nbin+1)/2);
                const auto coarse=finite.binary_mag_preplanned(
                    r.s,1/r.q,{r.time,r.y},r.rho,point_mu,mode,coarse_n,&seeds,&point);
                nested_error=std::abs(keep.magnification-coarse.magnification);
                const double budget=std::max(1e-16,r.target*std::abs(keep.magnification));
                // A conservative quarter-budget leaves room for the observed
                // non-monotone grid error.  Failure transfers to the current
                // equation-derived boundary/K-rule lane, preserving its
                // topology and value contract.
                if(!keep.converged||!coarse.converged||!std::isfinite(nested_error)||
                   nested_error>0.25*budget) {
                    AdaptiveConfig cfg;cfg.gradient_policy=GradientPolicy::None;
                    cfg.with_jacobian=false;cfg.tol.mu_atol=1e-16;cfg.tol.mu_rtol=r.target;
                    AdaptiveWorkspace workspace;
                    const auto a=epoch_adaptive(
                        {r.time,r.y,r.rho,1/r.q,r.s,true},r.u,cfg,workspace);
                    keep.magnification=a.mu;
                    keep.converged=a.value_converged&&a.numerical_status==Status::OK;
                    keep.decision=FiniteSourceDecision{
                        FiniteSourceMethod::source_plane_quadrature,
                        static_cast<int>(a.stats.node_evaluations),
                        "Vmax certified boundary/K-rule backstop"};
                    boundary_backstop=true;
                }
            }
            const auto t1=Clock::now();
            if(rep>=0)elapsed.push_back(std::chrono::duration<double,std::milli>(t1-t0).count());
            sink+=keep.magnification;
        }
        const double rel=std::abs(keep.magnification-r.reference)/std::max(1.0,std::abs(r.reference));
        output<<r.cid<<' '<<r.cfg<<' '<<r.profile<<' '<<r.target<<' '<<r.db<<' '
              <<r.epoch<<' '<<r.method<<' '<<r.nbin<<' '<<median(elapsed)<<' '
              <<keep.magnification<<' '<<r.reference<<' '<<rel<<' '<<keep.converged
              <<' '<<image_count<<' '<<keep.decision.estimated_evaluations<<' '
              <<nested_error<<' '<<boundary_backstop<<'\n';
        if((k+1)%512==0)std::cerr<<"progress "<<(k+1)<<'/'<<rows.size()<<'\n';
    }
    std::cerr<<"wrote "<<rows.size()<<" rows sink="<<sink<<'\n';
}
