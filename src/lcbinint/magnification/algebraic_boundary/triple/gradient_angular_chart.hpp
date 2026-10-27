#pragma once
#include "local_precision.hpp"
#include "angular_stationarity.hpp"
#include "../nested_fejer2.hpp"
#include "gradient_quadrature.hpp"
namespace lcbinint::algebraic_boundary::triple::shape_gradient_detail {
// A local arithmetic guard, independent of the requested flux tolerance.
// Tight endpoint roots are handled separately. Only a sample whose estimated
// phi roundoff exceeds this fraction needs promoted tangent arithmetic.
inline constexpr double tangent_phi_roundoff=1e-5;
struct AngularChart {struct Node {double theta,jac,weight;};struct Minimum {double center,scale,phi=0,error=0;};std::vector<Minimum> minima;std::vector<Node> nodes;bool valid=true,algebraic_hint=false;};
inline void render_angular_chart(AngularChart& out,double lo,double hi,bool physical_ends,
        int endpoint_order=16,int interior_order=12) {
    const auto& minima=out.minima;out.nodes.clear();
    for(std::size_t index=0;index<minima.size();++index) {
        const auto minimum=minima[index];
        const double lower=index?(minima[index-1].center+minimum.center)*.5:lo;
        const double upper=index+1<minima.size()?(minimum.center+minima[index+1].center)*.5:hi;
        const double tlo=std::asinh((lower-minimum.center)/minimum.scale),thi=std::asinh((upper-minimum.center)/minimum.scale);
        if(!std::isfinite(thi-tlo) || thi-tlo>64){out.valid=false;return;}
        const int cells=std::max(1,int(std::ceil(thi-tlo)));
        if(cells>64){out.valid=false;return;}
        for(int cell=0;cell<cells;++cell) {
            const double a=tlo+(thi-tlo)*cell/cells,b=tlo+(thi-tlo)*(cell+1)/cells;
            const FoldRadialMap map{a,b,physical_ends&&index==0&&cell==0,
                physical_ends&&index+1==minima.size()&&cell+1==cells};
            // Physical ends retain the incumbent order. Interior cells in
            // the hyperbolic chart have no remaining square-root endpoint.
            const auto& rule=gauss_rule(map.left || map.right ? endpoint_order : interior_order);
            for(std::size_t k=0;k<rule.x.size();++k) {
                const double x=rule.x[k];
                const auto mapped=map(x);const double t=mapped[0];
                const double theta=minimum.center+minimum.scale*std::sinh(t),jac=minimum.scale*std::cosh(t)*mapped[1];
                out.nodes.push_back({theta,jac,rule.w[k]});
            }
        }
    }
    return;
}struct ChartIntegral {double flux=0;std::array<double,8> gradient{};bool valid=true;};
// A small positive angular minimum is a nearby complex root pair. As in
// binary tangent replay, resolve that pair with a separate coordinate chart
// instead of demanding relative error convergence of eight final gradients.
// theta=c+s*sinh(t) removes the narrow 1/sqrt(phi) peak of the shape derivative.
template<class Ray> auto discovery_slope(const Ray& ray,double theta,int)->decltype(ray.angular_slope(theta)){return ray.angular_slope(theta);}
template<class Ray> double discovery_slope(const Ray& ray,double theta,long){double error=0;return ray.template sample<false>(theta,error).dtheta;}
template<class Ray,class Work> std::optional<AngularChart> complex_arc_geometry(
    const Ray& ray,double R,const Geometry& p,double lo,double hi,bool physical_ends,
    Work& work,int endpoint_order=16,int interior_order=12,bool render=true,const AngularStationarity* stationary=nullptr) {

    if(ray.strictly_concave(lo,hi))return std::nullopt;
    std::vector<AngularChart::Minimum> minima;
    const auto append_minimum=[&](double left,double right){
        double a=left,b=right,c=(a+b)*.5,curvature=0,error=0;Evaluation<true> jet{};
        for(int iteration=0;iteration<24;++iteration){jet=ray.template sample<false>(c,error,&curvature);if(jet.dtheta<0)a=c;else b=c;const double next=c-jet.dtheta/curvature;if(std::abs(next-c)<1e-13*(1+std::abs(c)))break;c=std::isfinite(next)&&next>a&&next<b?next:(a+b)*.5;}
        jet=ray.template sample<false>(c,error,&curvature);double phi=jet.phi;
        if(!(phi>0)||error>1e-5*phi){const auto q=local_precision::evaluate<false>(local_precision::Q(R),local_precision::Q(c),p);phi=double(q.phi);error=64*std::numeric_limits<double>::epsilon()*std::abs(phi);++work.quad_samples;}
        if(phi>0&&curvature>0){const double scale=std::sqrt(2*phi/curvature);if(scale<(hi-lo)*.15&&std::isfinite(scale))minima.push_back({c,scale,phi,error});}
    };
    const auto certified=stationary?stationary->minima(ray,work,lo,hi):std::nullopt;
    if(certified){for(const auto& minimum:*certified)if(minimum[1]<(hi-lo)*.15)minima.push_back({minimum[0],minimum[1],minimum[2],minimum[3]});}
    else{const auto slope=[&](double theta){
        return discovery_slope(ray,theta,0);
        };double left=lo,previous=slope(lo);const int scans=32;
        for(int k=1;k<=scans;++k){const double right=lo+(hi-lo)*k/scans,current=slope(right);if(previous<0&&current>0)append_minimum(left,right);left=right;previous=current;}
    }
    if(minima.empty())return std::nullopt;
    AngularChart out;out.minima=std::move(minima);
    if(render)render_angular_chart(out,lo,hi,physical_ends,endpoint_order,interior_order);
    return out;
}

template<class Ray,class Work> std::optional<ChartIntegral> complex_arc_chart(
    const Ray& ray,double R,const Geometry& p,double lo,double hi,bool physical_ends,
    const std::array<double,8>& motion,Work& work,
    std::optional<AngularChart>* retained=nullptr) {
    std::optional<AngularChart> discovered;
    if(!retained)discovered=complex_arc_geometry(ray,R,p,lo,hi,physical_ends,work);
    auto& geometry=retained?*retained:discovered;
    if(!geometry)return std::nullopt;
    ChartIntegral out;out.valid=geometry->valid;if(!out.valid)return out;
    if(geometry->nodes.empty())render_angular_chart(*geometry,lo,hi,physical_ends);
    out.valid=geometry->valid;if(!out.valid)return out;
    for(const auto& node:geometry->nodes) {
        const double theta=node.theta,jac=node.jac,weight=node.weight;
        double error=0;const auto jet=ray.sample(theta,error);++work.evaluations;
        double root=0;std::array<double,8> dp{};
        if(jet.phi>0 && error<=tangent_phi_roundoff*jet.phi) {
            root=std::sqrt(jet.phi);
            for(int j=0;j<8;++j)dp[j]=jet.dp[j]+jet.dR*motion[j];
        } else {
            using Q=local_precision::Q;
            const auto q=local_precision::evaluate<true>(Q(R),Q(theta),p);++work.quad_samples;
            if(!(q.phi>0)){out.valid=false;return out;}
            root=double(sqrtq(q.phi));for(int j=0;j<8;++j)dp[j]=double(q.dp[j]+q.dR*Q(motion[j]));
        }
        const double value=R*root*jac;
        out.flux+=weight*value;
        for(int j=0;j<8;++j)out.gradient[j]+=weight*(motion[j]*root*jac+R*jac*dp[j]/(2*root));
    }
    return out;
}
} // namespace lcbinint::algebraic_boundary::triple::shape_gradient_detail
