#pragma once
#include "shape_transport.hpp"
#include "gradient_hermite_replay.hpp"
namespace lcbinint::algebraic_boundary::triple {
inline ShapeGradientResult replay_shape_gradient(const Geometry& p,double limb,
                                                 const ValueIntegralTrace& trace,const RadialEventPlan& support,int minimum_radial=6,int minimum_angular=4) {
    ShapeGradientResult out;
    struct EventMotion {double radius;Complex position;std::array<double,8> dp;};
    std::vector<EventMotion> events;
    for(auto event:support.events) {
        EventMotion record;record.radius=event.radius;
        // The primal already proved existence and uniqueness in this box.
        // Reuse arithmetic-floor boxes directly; replay needs the event's
        // velocity, not a second existence proof. Only wider enclosures need
        // the incumbent local refinement before evaluating that velocity.
        const double floor=128*std::numeric_limits<double>::epsilon();
        if((event.radius_hi-event.radius_lo>floor*(1+std::abs(event.radius)) ||
            event.coordinate_hi-event.coordinate_lo>floor*(1+std::abs(event.coordinate))) &&
           !refine_event_center(event,p,true)){out.reason="gradient_replay_event";return out;}
        const double theta=(event.reciprocal?std::acos(-1.):0)+2*std::atan(event.coordinate);
        const auto jet=evaluate<true>(event.radius,theta,p);++out.evaluations;
        if(!std::isfinite(jet.dR) || jet.dR==0)return out;
        for(int j=0;j<8;++j)record.dp[j]=-jet.dp[j]/jet.dR;
        record.position=std::polar(event.radius,theta);events.push_back(record);
    }
    std::unique_ptr<root_pair_detail::RadialCache<8>> pair_table;
    for(const auto& ptr:trace.panels) {
        const auto& panel=*ptr.primal;const auto& map=panel.radial_map();
        const bool weighted=panel.weighted_radial();
        std::vector<const EventMotion*> domain;
        for(const auto& e:events) {
            if(ptr.inclusion && std::abs(e.position-ptr.inclusion->center)>ptr.inclusion->radius)continue;
            bool excluded=false;for(const auto& ball:ptr.exclusions)if(std::abs(e.position-ball.center)<ball.radius)excluded=true;
            if(!excluded)domain.push_back(&e);
        }
        std::sort(domain.begin(),domain.end(),[](const auto*a,const auto*b){return a->radius<b->radius;});
        const auto velocity=[&](double R) {
            std::array<double,8> v{};
            const EventMotion*lo=nullptr,*hi=nullptr;
            for(const auto*e:domain) {
                if(std::abs(e->radius-R)<=32*std::numeric_limits<double>::epsilon()*(1+R))return e->dp;
                if(e->radius<R)lo=e;else {hi=e;break;}
            }
            if(lo && hi)for(int j=0;j<8;++j)v[j]=lo->dp[j]+(hi->dp[j]-lo->dp[j])*(R-lo->radius)/(hi->radius-lo->radius);
            return v;
        };
        const auto left=velocity(map.a),right=velocity(map.b);
        if(limb==0 && panel.full_annulus()) {
            const double pi=std::acos(-1.);
            out.uniform_flux+=pi*(map.b-map.a)*(map.a+map.b);
            for(int j=0;j<8;++j)out.uniform[j]+=2*pi*(map.b*right[j]-map.a*left[j]);
            continue;
        }
        std::array<double,8>slope;for(int j=0;j<8;++j)slope[j]=(right[j]-left[j])/(map.b-map.a);
        const bool reusable=panel.level()>=3 && panel.level()<=5;
        if(reusable)++out.geometry_checked_panels;
        // Reserve a small part of the requested VALUE budget for geometric
        // interpolation. No parameter-gradient norm selects this route.
        if(reusable && shape_gradient_detail::hermite_boundary_resolved(panel,p,limb>0,.025*trace.relative_tolerance)) {
            const auto& h=shape_gradient_detail::hermite_rule(panel.level(),weighted);
            ShapeGradientResult piece;bool valid=true;
            for(std::size_t k=0;k<h.x.size();++k) {
                const int slot=int(k+1)<<(8-panel.level());
                const auto& seed=panel.retained_radii()[slot];
                if(!seed){valid=false;break;}
                const double x=h.x[k],R=seed->radius(),shape=(1-x)*(1+x);
                const auto mapped=weighted?FoldRadialMap{map.a,map.b,false,false}(x):map(x);
                std::array<double,8> velocity;for(int j=0;j<8;++j)velocity[j]=left[j]+slope[j]*(R-map.a);
                shape_gradient_detail::HermiteRadius value;
                if(!shape_gradient_detail::hermite_radius(R,p,seed->topology,velocity,slope,value,piece,limb>0,std::max(seed->level(),minimum_angular),.05*trace.relative_tolerance)){valid=false;break;}
                const double jac=mapped[1];
                double jac_prime=0;
                if(!weighted) {
                    const double pi=std::acos(-1.);
                    if(map.left && map.right)jac_prime=pi*pi*(map.b-map.a)/8*std::cos(pi*(1+x)*.5);
                    else if(map.left)jac_prime=(map.b-map.a)*.5;
                    else if(map.right)jac_prime=-(map.b-map.a)*.5;
                }
                const double scale=weighted?1/std::sqrt(shape):1;
                const double a=h.value[k]*jac*scale+h.derivative[k]*scale*(jac_prime+(weighted?jac*x/shape:0));
                const double b=h.derivative[k]*jac*jac*scale;
                piece.uniform_flux+=a*value.flux+b*value.radial_flux;
                piece.limb_flux+=a*value.limb_flux+b*value.radial_limb_flux;
                for(int j=0;j<8;++j){piece.uniform[j]+=a*value.gradient[j]+b*value.radial_gradient[j];piece.limb[j]+=a*value.limb_gradient[j]+b*value.radial_limb_gradient[j];}
            }
            out.evaluations+=piece.evaluations;out.quad_samples+=piece.quad_samples;
            if(valid){++out.reused_gradient_panels;out.reused_gradient_radii+=h.x.size();out.uniform_flux+=piece.uniform_flux;out.limb_flux+=piece.limb_flux;for(int j=0;j<8;++j){out.uniform[j]+=piece.uniform[j];out.limb[j]+=piece.limb[j];}continue;}
        }
        const int level=std::max(panel.level(),minimum_radial);
        const auto& rule=shape_gradient_detail::gauss_rule(level==6?24:std::max(12,1<<(level-1)));
        // A second real fold just outside a band produces the same close-pair
        // boundary layer as in binary Gaussian replay. A sine map resolves
        // one fold, but not that pair: use its known geometric separation.
        double left_scale=INFINITY,right_scale=INFINITY;
        const double matching=64*std::numeric_limits<double>::epsilon()*(1+map.b);
        for(const auto* e:domain) {
            if(e->radius<map.a-matching)left_scale=std::min(left_scale,(map.a-e->radius)*.5);
            if(e->radius>map.b+matching)right_scale=std::min(right_scale,(e->radius-map.b)*.5);
        }
        bool left_event=false,right_event=false;
        for(const auto*e:domain){left_event|=std::abs(e->radius-map.a)<matching;right_event|=std::abs(e->radius-map.b)<matching;}
        const bool left_chart=left_event && left_scale<(map.b-map.a)*.125;
        const bool right_chart=right_event && right_scale<(map.b-map.a)*.125;
        std::vector<std::array<double,2>> nodes;
        const auto add_chart=[&](double a,double b,bool at_left,double scale,bool outer_fold) {
            // Removing the close-pair scale permits half the ordinary order
            // in each bounded chart cell, without a derivative-error loop.
            const auto& rule=shape_gradient_detail::gauss_rule(std::max(12,1<<(level-2)));
            const double length=b-a,limit=2*std::asinh(std::sqrt(length/(2*scale)));
            const int cells=std::max(1,int(std::ceil(limit)));
            for(int cell=0;cell<cells;++cell) {
                const FoldRadialMap tmap{limit*cell/cells,limit*(cell+1)/cells,false,outer_fold&&cell+1==cells};
                for(std::size_t k=0;k<rule.x.size();++k) {
                    const auto t=tmap(rule.x[k]);const double sh=std::sinh(t[0]*.5);
                    const double distance=2*scale*sh*sh;
                    nodes.push_back({at_left?a+distance:b-distance,rule.w[k]*scale*std::sinh(t[0])*t[1]});
                }
            }
        };
        if(left_chart && right_chart) {
            const double mid=map.a+(map.b-map.a)*.5;
            add_chart(map.a,mid,true,left_scale,false);add_chart(mid,map.b,false,right_scale,false);
        } else if(left_chart)add_chart(map.a,map.b,true,left_scale,map.right);
        else if(right_chart)add_chart(map.a,map.b,false,right_scale,map.left);
        else for(std::size_t k=0;k<rule.x.size();++k) {
            const auto mapped=weighted?FoldRadialMap{map.a,map.b,true,true}(rule.x[k]):map(rule.x[k]);
            nodes.push_back({mapped[0],rule.w[k]*mapped[1]});
        }
        std::vector<const RadiusWorkspace<false>*> seeds;
        for(const auto& radius:panel.retained_radii())if(radius)seeds.push_back(radius.get());
        if(seeds.empty()){out.reason="gradient_replay_empty_panel";return out;}
        std::sort(seeds.begin(),seeds.end(),[](const auto*a,const auto*b){return a->radius()<b->radius();});
        std::vector<shape_gradient_detail::RadialComplexPair> complex_pairs;
        for(std::size_t s=0;s<5;++s) {
            const auto* seed=seeds[(seeds.size()-1)*(s+2)/8];
            for(const auto& root:seed->topology.roots) {
                const auto pair=shape_gradient_detail::radial_complex_pair(seed->radius(),(root.lo+root.hi)*.5,map.a,map.b,p);
                if(!pair)continue;bool known=false;
                for(const auto& old:complex_pairs)if(std::abs(old.center-pair->center)<1e-6*(map.b-map.a))known=true;
                if(!known)complex_pairs.push_back(*pair);
            }
        }
        if(!complex_pairs.empty()) {
            const auto& rule=shape_gradient_detail::gauss_rule(std::max(12,1<<(level-2)));
            std::sort(complex_pairs.begin(),complex_pairs.end(),[](auto a,auto b){return a.center<b.center;});nodes.clear();
            for(std::size_t index=0;index<complex_pairs.size();++index) {
                const auto pair=complex_pairs[index];
                const double a=index?(complex_pairs[index-1].center+pair.center)*.5:map.a;
                const double b=index+1<complex_pairs.size()?(pair.center+complex_pairs[index+1].center)*.5:map.b;
                const double lower=std::asinh((a-pair.center)/pair.scale),upper=std::asinh((b-pair.center)/pair.scale);
                const int cells=std::max(1,int(std::ceil(upper-lower)));
                for(int cell=0;cell<cells;++cell) {
                    const FoldRadialMap tmap{lower+(upper-lower)*cell/cells,lower+(upper-lower)*(cell+1)/cells,
                        index==0&&cell==0&&map.left,index+1==complex_pairs.size()&&cell+1==cells&&map.right};
                    for(std::size_t k=0;k<rule.x.size();++k) {
                        const auto t=tmap(rule.x[k]);nodes.push_back({pair.center+pair.scale*std::sinh(t[0]),rule.w[k]*pair.scale*std::cosh(t[0])*t[1]});
                    }
                }
            }
        }
        // Carry a successful endpoint and its radial velocity into the next
        // quadrature sample. The primal anchors remain available for every
        // local repair; a failed attempt never advances this moving anchor.
        shape_gradient_detail::RadiusAnchor anchor;
        for(const auto& node:nodes) {
            double R=node[0];
            // Hyperbolic nodes can round into an endpoint's certified box.
            // Stay on the proven interior side rather than asking Newton to
            // continue a root on the wrong side of a rounded fold.
            if(panel.band_topology()) {
                if(R<=panel.safe_lo())R=std::nextafter(panel.safe_lo(),map.b);
                if(R>=panel.safe_hi())R=std::nextafter(panel.safe_hi(),map.a);
            }
            auto next=std::lower_bound(seeds.begin(),seeds.end(),R,[](const auto* s,double r){return s->radius()<r;});
            const RadiusWorkspace<false>* seed=next==seeds.end()?seeds.back():*next;
            if(next!=seeds.begin() && std::abs((*std::prev(next))->radius()-R)<std::abs(seed->radius()-R))seed=*std::prev(next);
            const double distance=std::abs(seed->radius()-R);
            shape_gradient_detail::RadiusResult value;
            const int angular_level=std::max(seed->level(),minimum_angular);
            std::array<double,8>motion;for(int j=0;j<8;++j)motion[j]=left[j]+slope[j]*(R-map.a);
            const bool retained_band=panel.band_topology() && R>panel.safe_lo() && R<panel.safe_hi();
            bool completed=false;
            if(distance==0 || retained_band)
                completed=shape_gradient_detail::radius(R,p,seed->topology,limb>0,angular_level,motion,value,out,seed->radius(),&anchor,&pair_table,.05*trace.relative_tolerance);
            if(!completed) {
                auto topology=continued_topology(R,p,seed->topology,ptr.angular_options(),-1,panel.coefficient_cache());
                // Match RadiusWorkspace's local repair: an inherited root
                // can become a poor Newton anchor near a close pair. Count
                // roots again only at this failed radius, keeping the rest.
                if(!topology.completed) {
                    topology=angular_topology(R,p,ptr.angular_options(),panel.coefficient_cache());
                    ++out.global_radii;
                } else ++out.continued_radii;
                if(!topology.completed){out.reason="gradient_replay_topology";return out;}
                value={};
                if(!shape_gradient_detail::radius(R,p,topology,limb>0,angular_level,motion,value,out,R,&anchor,&pair_table,.05*trace.relative_tolerance)) {
                    return out;
                }
            }
            const double w0=node[1],wh=w0;
            out.uniform_flux+=w0*value.f0;out.limb_flux+=wh*value.fh;
            for(int j=0;j<8;++j){out.uniform[j]+=w0*(value.uniform[j]+slope[j]*value.f0);out.limb[j]+=wh*(value.limb[j]+slope[j]*value.fh);}
        }
    }
    out.completed=true;out.reason="converged_value_replay";return out;
}
} // namespace lcbinint::algebraic_boundary::triple
