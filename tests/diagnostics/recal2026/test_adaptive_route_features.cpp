#include "lcbinint/magnification/adaptive_route_selector.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
using namespace lcbinint::magnification;
int main() {
    PointSourceMagnifier point;int checks=0;
    for(double s:{.6,1.2,2.0})for(double q:{.01,.3})for(double x:{-.7,.2}) {
        const double y=.17,rho=.001;
        auto images=point.binary_images(s,q,{x,y});
        auto mirrored=point.binary_images(s,q,{x,-y});
        auto f=adaptive_detail::adaptive_route_features(s,q,rho,.5,1e-3,images);
        auto g=adaptive_detail::adaptive_route_features(s,q,rho,.5,1e-3,mirrored);
        for(int k=0;k<15;++k){assert(std::abs(f[k]-g[k])<1e-7);++checks;}
        double axis=0;
        for(auto image:images){
            auto z=image.position;const double h=1e-6;
            auto xp=point.binary_lens_equation(s,q,{z.x+h,z.y});
            auto xm=point.binary_lens_equation(s,q,{z.x-h,z.y});
            auto yp=point.binary_lens_equation(s,q,{z.x,z.y+h});
            auto ym=point.binary_lens_equation(s,q,{z.x,z.y-h});
            double a=(xp.x-xm.x)/(2*h),b=(yp.x-ym.x)/(2*h);
            double c=(xp.y-xm.y)/(2*h),d=(yp.y-ym.y)/(2*h),det=a*d-b*c;
            axis+=(std::hypot(d,c)+std::hypot(b,a))/std::abs(det);
        }
        assert(std::abs(std::pow(10.0,f[9])/axis-1)<1e-4);++checks;
        auto chosen=adaptive_detail::choose_adaptive_route(s,q,rho,.5,1e-3,images);
        assert(chosen.within_training_domain);++checks;
        auto unsupported=adaptive_detail::choose_adaptive_route(s,q,rho,.5,1e-5,images);
        assert(!unsupported.within_training_domain&&unsupported.polar);++checks;
    }
    std::cout<<checks<<" feature/selector checks passed\n";
}
