#pragma once
#include "coupled_profile.hpp"
#include "fused_radius.hpp"
#include "fused_radial_rule.hpp"
namespace lcbinint::algebraic_boundary::triple::fused_detail {
// A spatial-only witness of the value boundary interpolant. Parameter
// derivatives never participate in this quadrature geometry criterion.
struct GuardRay {
 struct Spatial {double phi=0,dR=0,dtheta=0;};
 const Geometry& geometry;double radius,tolerance;std::size_t* promotions;Ray<double> ray;mutable std::unique_ptr<Ray<Q>> wide;
 GuardRay(const Geometry& p,double R,double gate,std::size_t* count):geometry(p),radius(R),tolerance(gate),promotions(count),ray(R,p){}
 Spatial endpoint(double theta)const{const auto q=ray.template sample<false>(theta);return {q.phi,q.r,q.t};}
 template<bool Parameters=false> Spatial sample(double theta,double& error,double angular_width=0)const {
  const auto q=ray.template sample<false>(theta);error=q.roundoff;
  if(error>.125*tolerance*(angular_width>0?std::abs(q.t)*angular_width:std::abs(q.phi))){if(!wide)wide=std::make_unique<Ray<Q>>(radius,geometry);const auto w=wide->template sample<false>(Q(theta));if(promotions)++*promotions;error=double(w.roundoff);return {double(w.phi),double(w.r),double(w.t)};}
  Spatial out{};out.phi=q.phi;out.dR=q.r;out.dtheta=q.t;return out;
 }
};
template<class Panel> bool boundary_resolved(const Panel& panel,const Geometry& p,bool with_limb,double value_gate=1e-6,std::size_t* promotions=nullptr) {
    LCBININT_COUPLED_SCOPE(witness);
    const bool weighted=panel.weighted_radial();const auto& map=panel.radial_map();
    const auto& h=radial_rule(panel.level(),weighted?1:0);const std::size_t n=h.x.size();
    using Curves=std::array<std::array<double,4>,3>;std::vector<Curves> values(n);
    const auto& first=panel.retained_radii()[1<<(8-panel.level())];if(!first)return false;
    std::size_t count=0;bool full=first->topology.roots.empty();
    using Intensity=std::array<std::array<double,2>,27>;
    std::vector<Intensity> intensities(with_limb?n:0);
    auto angles=fejer_rule(3).x;
    if(full){angles.push_back(-1);angles.push_back(1);}
    for(std::size_t k=0;k<n;++k) {
        const auto& seed=panel.retained_radii()[int(k+1)<<(8-panel.level())];
        if(!seed)return false;
        auto topology=seed->topology;seed->resolved_endpoint_geometry(topology);
        if(k==0){count=topology.arcs.size();full=topology.roots.empty();}
        if(count>3 || topology.arcs.size()!=count || full!=topology.roots.empty())return false;
        if(full && !with_limb)continue;
        const double x=h.x[k],shape=(1-x)*(1+x),scale=weighted?std::sqrt(shape):1;
        const auto mapped=weighted?FoldRadialMap{map.a,map.b,false,false}(x):map(x);
        GuardRay ray(p,seed->radius(),value_gate,promotions);
        std::size_t rotation=0;
        if(k && count>1) {
            double best=INFINITY;
            // Arc ordering is cyclic, not a fixed label. A seam crossing
            // permutes the arcs as well as adding a 2pi turn to one arc.
            for(std::size_t shift=0;shift<count;++shift) {
                double cost=0;
                for(std::size_t j=0;j<count;++j) {
                    const auto& arc=topology.arcs[(j+shift)%count];
                    const double d=std::remainder((arc.lo+arc.hi)*.5-values[k-1][j][0],2*std::acos(-1.));cost+=d*d;
                }
                if(cost<best){best=cost;rotation=shift;}
            }
        }
        for(std::size_t j=0;j<count;++j) {
            auto arc=topology.arcs[(j+rotation)%count];
            // Canonical arc ordering can rotate across the angular seam.
            // Use the first radius as a continuous anchor and match modulo 2pi.
            if(k) {
                const double target=values[k-1][j][0],mid=(arc.lo+arc.hi)*.5;
                const double shift=2*std::acos(-1.)*std::round((target-mid)/(2*std::acos(-1.)));
                arc.lo+=shift;arc.hi+=shift;
            }
            const double width=arc.hi-arc.lo;double lr=0,hr=0;
            if(!full) {
                const auto lo=seed->spatial_witness(arc.lo,INFINITY,0,promotions),hi=seed->spatial_witness(arc.hi,INFINITY,0,promotions);
                if(!(lo.t>0 && hi.t<0))return false;
                lr=-lo.r/lo.t;hr=-hi.r/hi.t;
            }
            const double mean=(arc.lo+arc.hi)*.5,half=width*.5;
            values[k][j]={mean,mapped[1]*(lr+hr)*.5,half/scale,
                mapped[1]*(hr-lr)*.5/scale+(weighted?half*x/(shape*scale):0)};
            if(with_limb)for(std::size_t a=0;a<angles.size();++a) {
                const double t=angles[a],theta=mean+half*t;
                const auto field=seed->spatial_witness(theta,value_gate,0,promotions);const double error=field.roundoff;
                if(!(field.phi>0 && error<1e-5*field.phi))return false;
                const double theta_x=mapped[1]*((lr+hr)*.5+t*(hr-lr)*.5);
                const double radial_shape=weighted?shape:1;
                const double root=std::sqrt(field.phi/radial_shape);
                intensities[k][j*angles.size()+a]={root,
                    ((field.r*mapped[1]+field.t*theta_x)/radial_shape+(weighted?2*x*field.phi/(shape*shape):0))/(2*root)};
            }
        }
    }
    if(full && !with_limb)return true;
    for(const auto& witness:h.witnesses) {
        const auto mapped=weighted?FoldRadialMap{map.a,map.b,false,false}(witness.x):map(witness.x);
        const double scale=weighted?std::sqrt((1-witness.x)*(1+witness.x)):1;GuardRay ray(p,mapped[0],value_gate,promotions);
        for(std::size_t j=0;j<count;++j) {
            double mean=0,half=0;
            for(std::size_t k=0;k<n;++k){const auto& c=values[k][j];mean+=witness.value[k]*c[0]+witness.derivative[k]*c[1];half+=witness.value[k]*c[2]+witness.derivative[k]*c[3];}
            half*=scale;if(!(half>0))return false;
            double unused=0;if(!(ray.sample<false>(mean,unused).phi>0))return false;
            if(with_limb)for(std::size_t a=0;a<angles.size();++a) {
                double predicted=0;
                for(std::size_t k=0;k<n;++k) {
                    const auto& c=intensities[k][j*angles.size()+a];
                    predicted+=witness.value[k]*c[0]+witness.derivative[k]*c[1];
                }
                const double radial_shape=weighted?(1-witness.x)*(1+witness.x):1;
                double error=0;const auto field=ray.sample<false>(mean+half*angles[a],error);
                if(!(field.phi>0 && std::abs(predicted*std::sqrt(radial_shape)-std::sqrt(field.phi))+error/std::sqrt(field.phi)<value_gate*std::sqrt(field.phi)))return false;
            }
            if(full)continue;
            for(int sign:{-1,1}) {
                double error=0;const auto jet=ray.sample<false>(mean+sign*half,error,2*half);
                // Match the endpoint implementation's geometric meaning:
                // residual / angular slope is an angular displacement.
                if(!(std::abs(jet.phi)+error<value_gate*std::abs(jet.dtheta)*2*half))return false;
            }
        }
    }
    return true;
}
}
