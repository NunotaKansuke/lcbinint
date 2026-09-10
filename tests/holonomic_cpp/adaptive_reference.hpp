#pragma once
// Independent quadrature audit: Gauss-Legendre radial rule, direct angular
// phi integration. Shares lens equations/topology, not Fejer or vK algebra.
#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"
#include <fstream>
#include <sstream>
#include <cstdio>
using namespace lcbinint::holonomic;
struct GL {std::vector<double>x,w;GL(int n):x(n),w(n){for(int i=0;i<(n+1)/2;++i){double z=std::cos(kPi*(i+.75)/(n+.5)),dp=0;for(int it=0;it<20;++it){double p=1,q=0;for(int k=1;k<=n;++k){double r=((2*k-1)*z*p-(k-1)*q)/k;q=p;p=r;}dp=n*(z*p-q)/(z*z-1);double d=p/dp;z-=d;if(std::fabs(d)<2e-16)break;}x[i]=-z;x[n-1-i]=z;w[i]=w[n-1-i]=2/((1-z*z)*dp*dp);}}};
// Independent reference repair uses qf Sturm isolation, never a sampled
// angular existence test. A failed certificate still leaves the reference NaN.
ArcSet reference_arcs(double R,const PrimaryFrame& pf) {
    auto arcs=arc_intervals(R,pf);
    if(arcs.kind!=ArcKind::kDegenerate)return arcs;
    auto q=boundary_quartic(R,pf);auto cert=certify_quartic(q.p);
    if(!cert.certified)return arcs;
    auto roots=sturm_isolate_real_roots(cert.chart_coeffs,cert.root_count);
    if(int(roots.size())!=cert.root_count)return arcs;
    std::vector<double> theta;
    for(double x:roots){double a=(cert.reciprocal?kPi:0)+2*std::atan(x);if(a<0)a+=kTwoPi;if(a>=kTwoPi)a-=kTwoPi;theta.push_back(a);}
    std::sort(theta.begin(),theta.end());return arc_set_from_root_thetas(R,pf,theta);
}
double reference(const LensParams&p,double u,const TopologyResult&t,int n){
    if(t.status!=Status::OK)return NAN;
    auto pf=PrimaryFrame::from(p);GL radial(n);Cheb1Dyn angular(2*n);adaptive_detail::Sum total;
    auto fold=[&](double R){for(auto&e:t.events)if(e.radius==R&&e.physically_real&&e.kind=="physical_real")return true;return false;};
    std::vector<double> cuts{0,t.r_max};
    for(const auto& c:t.cells){cuts.push_back(c.r_lo);cuts.push_back(c.r_hi);}
    for(const auto& e:t.events)if(e.physically_real&&e.kind=="physical_real"&&e.radius>0&&e.radius<t.r_max)cuts.push_back(e.radius);
    std::sort(cuts.begin(),cuts.end());cuts.erase(std::unique(cuts.begin(),cuts.end()),cuts.end());
    for(size_t ci=1;ci<cuts.size();++ci){
        double a=cuts[ci-1],b=cuts[ci];auto middle=quartic_topology(a+.5*(b-a),pf);
        if(!middle.certified)return NAN;
        if(middle.kind==ArcKind::kEmpty)continue;
        if(middle.kind!=ArcKind::kArcs)return NAN;
        FoldRadialMap map{a,b,fold(a),fold(b)};
        for(int k=0;k<n;++k){auto rr=map(radial.x[k]);double R=rr[0];auto arcs=reference_arcs(R,pf);if(arcs.kind==ArcKind::kDegenerate||arcs.kind==ArcKind::kFull)return NAN;double f0=0,fh=0;
            for(auto a:arcs.arcs){auto e=polish_endpoint(R,a[0],pf),l=polish_endpoint(R,a[1],pf);double te=e.theta,tl=l.theta;if(tl<=te)tl+=kTwoPi;double h=(tl-te)/2,m=te+h;f0+=R*(tl-te);
                if(u!=0)for(int j=0;j<2*n;++j){double ph=phi_val(R,m+h*angular.x[j],pf);if(ph>0)fh+=R*h*angular.w[j]*std::sqrt(ph);}}
            total.add(radial.w[k]*rr[1]*((1-u)*f0+u*fh)/(kPi*p.rho*p.rho*(1-u/3)));}}
    return total.get();
}
