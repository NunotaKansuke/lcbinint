#include "lcbinint/magnification/holonomic/fused_gradient.hpp"
#include "lcbinint/magnification/holonomic/seeded_radial_frontier.hpp"
#include "lcbinint/magnification/point_source_magnifier.hpp"
#include <iostream>

using namespace lcbinint::holonomic;
int main() {
    const LensParams p{.2,.1,.001,1.25,.0001,true};
    const auto invalid=epoch_fused_gradient(p,0,{});
    if(invalid.value_converged || invalid.gradient_precision_checked ||
       std::any_of(invalid.gradient_valid.begin(),invalid.gradient_valid.end(),[](bool v){return v;}))return 1;
    lcbinint::magnification::PointSourceMagnifier point;
    std::vector<Cplx<double>> value_seeds;
    std::vector<std::array<double,2>> gradient_seeds;
    for(const auto& image:point.binary_images(p.a,p.q,{p.xs,p.ys})) {
        const double x=image.position.x+(p.q<1?p.a:0),y=image.position.y;
        value_seeds.push_back({x,y});gradient_seeds.push_back({x,y});
    }
    const double r=std::hypot(p.xs,p.ys);
    const double slope=-8/(r*r*std::pow(r*r+4,1.5));
    const double laplacian=32*(r*r+1)/(r*r*r*std::pow(r*r+4,2.5));
    for(double u:{0.,.5}) {
        AdaptiveWorkspace workspace;AdaptiveConfig cfg;
        const auto value=epoch_seeded_radial_frontier(p,u,value_seeds,cfg,workspace);
        const auto gradient=epoch_fused_gradient(p,u,gradient_seeds);
        if(!gradient.value_converged || gradient.gradient_precision_checked ||
           std::fabs(value.integral.mu-gradient.mu)>1e-8+1e-4*std::fabs(value.integral.mu))return 2;
        const double moment=((1-u)/4+2*u/15)/((1-u)/2+u/3);
        const std::array<double,3> analytic{{slope*p.xs/r,slope*p.ys/r,p.rho*moment*laplacian/2}};
        for(int j=0;j<3;++j)if(!gradient.gradient_valid[j] ||
            std::fabs(gradient.grad_mu[j]-analytic[j])>1e-3*std::fabs(analytic[j]))return 3;
        std::cout<<"u="<<u<<" value="<<gradient.mu<<" gradient=";
        for(auto g:gradient.grad_mu)std::cout<<g<<' ';
        std::cout<<"precision_checked="<<gradient.gradient_precision_checked<<'\n';
    }
    std::cout<<"normal value and compiled gradient API coexist; analytic limit and invalid-input checks passed\n";
}
