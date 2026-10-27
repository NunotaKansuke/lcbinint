#pragma once
#include "integration_types.hpp"
#include "radial_panel.hpp"
#include "value_trace.hpp"
namespace lcbinint::algebraic_boundary::triple {

#ifdef LCBININT_TRIPLE_MULTIRATE_ARCS
template<class T,class=void>struct has_branch_refinement:std::false_type{};
template<class T>struct has_branch_refinement<T,std::void_t<decltype(std::declval<T&>().branches_active())>>:std::true_type{};
#endif
template<bool Gradient=false,template<bool>class PanelWorkspace=RadialPanel> inline IntegrationResult<Gradient> integrate_coupled(
    const Geometry& p,double limb,IntegrationOptions options={},
    const RadialEventPlan* supplied_plan=nullptr,ValueIntegralTrace* trace=nullptr) {
    IntegrationResult<Gradient> out;p.validate();
    if(trace)trace->relative_tolerance=options.relative_tolerance;
    // Gradient instantiation carries values and first derivatives through
    // the SAME adaptive frontier. The controller observes values only for
    // all calls; no completed-primal / tangent-replay pass follows.
    const bool loose_value=options.relative_tolerance>=1e-4;
    // Root inclusion/count proofs are shared with values. Gradient endpoint
    // motion and its local precision repairs remain independently controlled.
    if constexpr(Gradient)out.normalized.df0.fill(std::numeric_limits<double>::quiet_NaN());
    if(!std::isfinite(limb) || limb<0 || limb>1 ||
       !std::isfinite(options.absolute_tolerance) || options.absolute_tolerance<0 ||
       !std::isfinite(options.relative_tolerance) || options.relative_tolerance<0 ||
       options.absolute_tolerance+options.relative_tolerance<=0) {
        out.reason="invalid_config";return out;
    }
    // A supplied plan is valid only for exactly this geometry/source/radius;
    // the owning epoch adapter, not a global mutable cache, enforces identity.
    out.support=supplied_plan?*supplied_plan:radial_event_plan(p,options.event_box_budget,
        (Gradient || trace)?0:options.relative_tolerance*.1);
    if(!out.support.completed){out.reason=out.support.reason;return out;}
    if constexpr(Gradient)if(!out.support.excluded_images.empty() || (Gradient && out.support.bounded_component_value!=0)) {
        out.reason="gradient_omitted_support";return out;
    }
    options.angular.exclusions=out.support.excluded_images.empty()?nullptr:&out.support.excluded_images;
    if(out.support.identity!=std::array<double,8>{p.X,p.Y,p.rho,p.m1,p.m2,p.a,p.bx,p.by}) {
        out.reason="support_plan_identity";return out;
    }
    if constexpr(!Gradient) {
        // Overlapping certified event-radius intervals need no resolved
        // numerical gap in a value integral. Use one cut for that cluster
        // and account for the complete annular uncertainty below. Distinct
        // angular events and all their enclosures remain in the plan.
        auto& events=out.support.events;
        std::sort(events.begin(),events.end(),[](auto a,auto b){return a.radius_lo<b.radius_lo;});
        for(std::size_t begin=0;begin<events.size();) {
            std::size_t end=begin+1;double lo=std::min(events[begin].radius_lo,events[begin].radius);
            double hi=std::max(events[begin].radius_hi,events[begin].radius);
            while(end<events.size() && std::min(events[end].radius_lo,events[end].radius)<=hi) {
                hi=std::max({hi,events[end].radius_hi,events[end].radius});++end;
            }
            if(end>begin+1) {
                const double cut=lo+(hi-lo)*.5;
                for(auto i=begin;i<end;++i) {
                    events[i].radius=cut;events[i].radius_lo=std::min(events[i].radius_lo,cut);
                    events[i].radius_hi=std::max(events[i].radius_hi,cut);
                }
            }
            begin=end;
        }
        std::sort(events.begin(),events.end(),[](auto a,auto b){return a.radius<b.radius;});
    }
    if(Gradient || trace){LCBININT_COUPLED_SCOPE(witness);for(auto& event:out.support.events) {
        {
            // A tangent needs a geometrically accurate fold location before
            // the primal panel map is built. Flux-only event boxes can be
            // wider: refining them after quadrature would move its boundary.
            const double floor=128*std::numeric_limits<double>::epsilon();
            if(event.radius_hi-event.radius_lo<=floor*(1+std::abs(event.radius)) &&
               event.coordinate_hi-event.coordinate_lo<=floor*(1+std::abs(event.coordinate)))continue;
            bool used_quad=false;
            if(!refine_event_center(event,p,options.relative_tolerance>=1e-4,&used_quad)){out.reason="gradient_event_precision";return out;}
            if(used_quad)++out.support.quad_refinements;
            else ++out.support.double_refinements;
        }
    }}

    out.support_complete=true;
    struct Panel {
        std::unique_ptr<PanelWorkspace<Gradient>> workspace;
        int radial_level=3,angular_level=4;
        double lo=0,hi=0;bool left_fold=true,right_fold=true;
        std::shared_ptr<SharedRadiusCache<Gradient>> shared;AngularOptions angular;
    };
    // Partition only before any quadrature. A full-source contraction ball
    // contains one entire image component and has no image boundary on its
    // edge. Two events per image permit the incumbent isolated-panel rule.
    // All resulting panels stay in this one controller and retain their work.
    std::vector<RadialEventPlan> component_plans;
    std::vector<const RadialEventPlan*> domains{&out.support};
    if(options.relative_tolerance>=1e-4 && options.separate_components &&
            out.support.excluded_images.empty() &&
            !out.support.component_images.empty()) {
        const auto& balls=out.support.component_images;
        bool valid=true;
        for(std::size_t i=0;i<balls.size();++i)for(std::size_t j=0;j<i;++j)
            if(std::abs(balls[i].center-balls[j].center)<=balls[i].radius+balls[j].radius+
               256*std::numeric_limits<double>::epsilon()*(1+std::abs(balls[i].center)+std::abs(balls[j].center)))valid=false;
        component_plans.resize(balls.size()+1);
        for(std::size_t i=0;i<component_plans.size();++i){auto&d=component_plans[i];d=out.support;d.events.clear();d.component_images.clear();if(i<balls.size())d.included_image=balls[i];else d.excluded_images=balls;}
        for(auto event:out.support.events) {
            // Use the certified rectangle centre, not the Newton proposal,
            // which need not lie inside the final contracted rectangle.
            const double rc=event.radius_lo+(event.radius_hi-event.radius_lo)*.5;
            const double tc=event.coordinate_lo+(event.coordinate_hi-event.coordinate_lo)*.5;
            const double theta=(event.reciprocal?std::acos(-1.):0)+2*std::atan(tc);
            const Complex z=std::polar(rc,theta);
            const double error=event.radius_hi-event.radius_lo+2*event.radius_hi*(event.coordinate_hi-event.coordinate_lo)+
                256*std::numeric_limits<double>::epsilon()*(1+event.radius_hi);
            std::size_t owner=balls.size();
            for(std::size_t i=0;i<balls.size();++i){double distance=std::abs(z-balls[i].center);if(distance+error<balls[i].radius)owner=i;else if(distance-error<=balls[i].radius)valid=false;}
            component_plans[owner].events.push_back(event);
        }
        for(std::size_t i=0;i<balls.size();++i)if(component_plans[i].events.size()!=2)valid=false;
        const auto remaining=component_plans.back().events.size();
        if(remaining%2)valid=false;
        if(valid){domains.clear();for(const auto&d:component_plans)if(!d.events.empty())domains.push_back(&d);out.component_domains=int(domains.size());}
        else component_plans.clear();
    }
    const double denominator=std::acos(-1.)*p.rho*p.rho*(1-limb/3);
    const double repair_absolute=options.absolute_tolerance*denominator/(out.support.rmax-out.support.rmin)*.05;
    // Allocate at most 40% of the local limb scale to the inner estimate.
    // The existing weighted global controller still checks the requested
    // flux tolerance; ceilings above six progressively tighten this budget.
    const double repair_relative=options.relative_tolerance*((loose_value && limb>0 && out.support.resultant_nodes>0)?.4:.05);
    std::vector<Panel> panels;
    auto coefficients=std::make_shared<AngularCoefficientCache>(p);
    coefficients->lens_context=out.support.lens_context;
    // Global event isolation signals enough boundary work to amortize the
    // point-sign polynomial early. Smooth component plans keep lazy setup.
    if(out.support.resultant_nodes>0)coefficients->point_sign_threshold=8;
    std::vector<std::shared_ptr<SharedRadiusCache<Gradient>>> all_shared;
    std::vector<std::unique_ptr<PanelWorkspace<Gradient>>> retained_parents;
    // Diagnostics do not control refinement. Traverse retained radii only
    // when returning a frontier result, rather than on every iteration.
    std::size_t event_motion_quad=0;
    auto update_counters=[&]() {
        out.value_evaluations=out.parameter_derivative_evaluations=0;out.quad_samples=event_motion_quad;
        out.continued_radii=out.global_radii=0;
        if constexpr(!std::is_same_v<PanelWorkspace<Gradient>,RadialPanel<Gradient>>) {
            for(const auto& panel:panels){out.value_evaluations+=panel.workspace->work().evaluations+panel.workspace->work().geometry_evaluations;out.parameter_derivative_evaluations+=panel.workspace->work().evaluations;out.quad_samples+=panel.workspace->work().quad_samples;out.global_radii+=panel.workspace->work().global_radii;out.continued_radii+=panel.workspace->work().continued_radii;}
            // Descendants share their parent's stream. Count each retained
            // context once; the context work is reported only by its owner.
            for(const auto& parent:retained_parents){out.value_evaluations+=parent->work().evaluations+parent->work().geometry_evaluations;out.parameter_derivative_evaluations+=parent->work().evaluations;out.quad_samples+=parent->work().quad_samples;out.global_radii+=parent->work().global_radii;out.continued_radii+=parent->work().continued_radii;}
        }
        if constexpr(Gradient){for(const auto& panel:panels)out.quad_samples+=panel.workspace->geometry_promotions();for(const auto& parent:retained_parents)out.quad_samples+=parent->geometry_promotions();}
        for(const auto& cache:all_shared)for(const auto& entry:cache->radii) {
            if(entry.second->continued_topology_used)++out.continued_radii;else ++out.global_radii;
            out.value_evaluations+=entry.second->value_evaluations;
            out.parameter_derivative_evaluations+=entry.second->gradient_evaluations;
            out.quad_samples+=entry.second->quad_samples;
        }
    };
    const double requested_endpoint_tolerance=options.angular.endpoint_tolerance;
    for(const auto* packet:domains) {
        const auto&domain=*packet;
        const bool material_motion=Gradient && options.moving_radial_gradients && (Gradient || limb>0 || domain.events.size()==2);
        // The automatic uniform endpoint budget belongs to this component;
        // an adjustment for a previous component must not leak into this one.
        options.angular.endpoint_tolerance=requested_endpoint_tolerance;
        options.angular.exclusions=domain.excluded_images.empty()?nullptr:&domain.excluded_images;
        options.angular.inclusion=domain.included_image?&*domain.included_image:nullptr;
        std::vector<double> cuts{domain.rmin};
        for(auto event:domain.events)cuts.push_back(event.radius);
        // Positive-mass poles are outside the image. Their radii are not
        // topology events: clearing denominators gives a strictly negative
        // boundary polynomial at a distinct lens pole. Only physical folds cut
        // the radial integral, preserving complete isolated weighted bands.
        if constexpr(Gradient)if(!material_motion) {
            // Derivative error control benefits from the incumbent extra
            // segmentation. These cuts do not burden the independent value path.
            std::vector<double> poles;
            for(auto lens:p.lenses()) {
                const double r=std::abs(lens);
                if(r>domain.rmin && r<domain.rmax)poles.push_back(r);
            }
            std::sort(poles.begin(),poles.end());double last=-INFINITY;
            for(double r:poles)if(r-last>64*std::numeric_limits<double>::epsilon()*r) {
                cuts.push_back(r);last=r;
            }
        }
        cuts.push_back(domain.rmax);std::sort(cuts.begin(),cuts.end());
        // Coincident events in radius are distinct angular events but one panel
        // boundary. Merge only overlapping enclosures, never by a fixed distance.
        std::vector<double> unique;
        for(double r:cuts)if(unique.empty() || r!=unique.back())unique.push_back(r);
        std::vector<std::array<double,8>> cut_motion;
        if constexpr(Gradient)cut_motion.resize(unique.size());
        bool moving=false;
        if constexpr(Gradient)if(material_motion) {
            LCBININT_COUPLED_SCOPE(witness);
            moving=true;
            for(std::size_t k=0;k<unique.size();++k) {
                int matches=0;
                for(const auto& event:domain.events)if(event.radius==unique[k]) {
                    if(++matches>1){moving=false;break;}
                    const double angle=(event.reciprocal?std::acos(-1.):0)+2*std::atan(event.coordinate);
                    const auto fast=fused_detail::Ray<double>(event.radius,p).sample(angle);
                    const double conditioning=std::max(1.,std::abs(event.radius*fast.r));
                    if(std::isfinite(fast.r)&&std::abs(fast.r)>1e-12*conditioning && fast.roundoff<1e-7 && std::abs(fast.phi)<1e-6 && std::abs(fast.t)<1e-6*conditioning){
                        for(int n=0;n<8;++n){cut_motion[k][n]=-fast.p[n]/fast.r;if(!std::isfinite(cut_motion[k][n]))moving=false;}
                        continue;
                    }
                    ++event_motion_quad;
                    using Q=local_precision::Q;
                    const Q theta=(event.reciprocal?acosq(Q(-1)):Q(0))+2*atanq(Q(event.coordinate));
                    const auto jet=local_precision::evaluate<true>(Q(event.radius),theta,p);
                    if(jet.dR==0 || !finiteq(jet.dR)){moving=false;break;}
                    for(int n=0;n<8;++n) {
                        cut_motion[k][n]=double(-jet.dp[n]/jet.dR);
                        if(!std::isfinite(cut_motion[k][n]))moving=false;
                    }
                }
            }
        }
        auto shared=std::make_shared<SharedRadiusCache<Gradient>>();shared->requested_relative_tolerance=options.relative_tolerance;
        shared->coefficients=coefficients;all_shared.push_back(shared);
        std::vector<bool> occupied(unique.size()-1);
        struct OrientedEvent {RadialEvent event;int change;double angle_lo,angle_hi;};
        std::vector<OrientedEvent> oriented;bool count_plan=true;int net_change=0;
        for(auto e:domain.events) {
            auto f=event_detail::physical(event_detail::I(e.radius_lo,e.radius_hi),
                event_detail::I(e.coordinate_lo,e.coordinate_hi),p,e.reciprocal);
            if(!f || std::abs(f->r.sign())!=1 || std::abs(f->tt.sign())!=1){count_plan=false;break;}
            const int change=f->r.sign()==f->tt.sign()?-2:2;
            const double pi=std::acos(-1.),padding=16*std::numeric_limits<double>::epsilon();
            const double angle_lo=(e.reciprocal?pi:0)+2*std::atan(e.coordinate_lo)-padding;
            const double angle_hi=(e.reciprocal?pi:0)+2*std::atan(e.coordinate_hi)+padding;
            for(const auto& old:oriented)if(e.radius_lo<=old.event.radius_hi && e.radius_hi>=old.event.radius_lo)
                for(int turn:{-1,0,1})if(angle_lo<=old.angle_hi+turn*2*pi && angle_hi>=old.angle_lo+turn*2*pi)
                    count_plan=false; // unresolved duplicate/overlap: retain global counting
            oriented.push_back({e,change,angle_lo,angle_hi});net_change+=change;
        }
        std::sort(oriented.begin(),oriented.end(),[](const auto& a,const auto& b){return a.event.radius<b.event.radius;});
        if(net_change!=0)count_plan=false;
        if(count_plan)for(std::size_t j=1;j<unique.size();++j) {
            const double mid=unique[j-1]+(unique[j]-unique[j-1])*.5;int count=0;
            for(const auto& e:oriented) {
                if(mid>=e.event.radius_lo && mid<=e.event.radius_hi)count_plan=false;
                if(mid>e.event.radius_hi)count+=e.change;
            }
            if(count<0 || count>6 || count%2)count_plan=false;
        }
        AngularTopology proposals;std::size_t event_index=0;int crossing_count=0;
        for(std::size_t j=1;j<unique.size();++j) {
            const double lo=unique[j-1],hi=unique[j];
            const double radius=lo+(hi-lo)*.5;
            bool proposal_valid=count_plan;
            if(count_plan)while(event_index<oriented.size() && oriented[event_index].event.radius<radius) {
                const auto& o=oriented[event_index++];crossing_count+=o.change;
                const auto& e=o.event;
                const double angle=(e.reciprocal?std::acos(-1.):0)+2*std::atan(e.coordinate);
                if(o.change>0) {
                    auto f=event_detail::physical(e.radius,e.coordinate,p,e.reciprocal);
                    const double half=f && f->tt!=0?std::sqrt(std::abs(-2*f->r*(radius-e.radius)/f->tt)):.01;
                    for(int side:{-1,1}) {
                        const double theta=(e.reciprocal?std::acos(-1.):0)+2*std::atan(e.coordinate+side*half);
                        proposals.roots.push_back({theta,theta});
                    }
                } else if(proposals.roots.size()>=2) {
                    for(int remove=0;remove<2;++remove) {
                        auto nearest=std::min_element(proposals.roots.begin(),proposals.roots.end(),[&](auto a,auto b) {
                            return std::abs(std::remainder((a.lo+a.hi)*.5-angle,2*std::acos(-1.)))<
                                   std::abs(std::remainder((b.lo+b.hi)*.5-angle,2*std::acos(-1.)));
                        });
                        proposals.roots.erase(nearest);
                    }
                } else proposal_valid=false;
            }
            // Zero certified crossings plus one negative membership witness
            // proves this whole band empty. Zero crossings alone can also
            // describe a full circle, so keep the membership test.
            if(proposal_valid && crossing_count==0 &&
                    angular_membership(radius,.123,p,options.angular)<0) {
                occupied[j-1]=false;proposals=AngularTopology{};continue;
            }
            auto workspace=std::make_shared<RadiusWorkspace<Gradient>>();
            if(!workspace->initialize(radius,p,options.angular,limb==0,repair_relative*.2,shared->coefficients.get(),
                    proposal_valid?&proposals:nullptr,proposal_valid?crossing_count:-1)) {
                out.reason=workspace->reason;return out;
            }
            occupied[j-1]=!workspace->topology.arcs.empty();
            if(count_plan) {
                if(!workspace->continued_topology_used && workspace->topology.roots.size()!=std::size_t(crossing_count))count_plan=false;
                proposals=workspace->topology;
            }
            shared->radii.emplace(radius,std::move(workspace));
        }
        if(limb==0 && options.automatic_endpoint_tolerance &&
                options.relative_tolerance>0 && options.angular.endpoint_tolerance==1e-12) {
            double occupied_measure=0;
            for(std::size_t j=1;j<unique.size();++j)if(occupied[j-1])
                occupied_measure+=(unique[j]-unique[j-1])*(unique[j]+unique[j-1]);
            // At most six endpoints, each contributing R*dtheta. Allocate at
            // most rtol/64 to this geometric uncertainty, using mu>=1 for
            // positive point lenses. The measured endpoint errors still enter
            // the ordinary flux controller; the requested flux tolerance stays
            // unchanged. Strict midpoint witnesses already computed are retained.
            if(occupied_measure>0)options.angular.endpoint_tolerance=std::max(
                options.angular.endpoint_tolerance,std::min(1e-6,
                    options.relative_tolerance*std::acos(-1.)*p.rho*p.rho/(192*occupied_measure)));
        }
        for(std::size_t j=1;j<unique.size();++j) {
            const double lo=unique[j-1],hi=unique[j];
            if(!(hi>lo) || !occupied[j-1])continue;
            // Material endpoint motion preserves the uniform fold weight. LD
            // keeps its existing map until its angular conditioning is resolved.
            const bool isolated=(!Gradient || (moving && (limb==0 || Gradient))) && (j==1 || !occupied[j-2]) &&
                (j+1==unique.size() || !occupied[j]);
            Panel panel;panel.lo=lo;panel.hi=hi;panel.shared=shared;panel.angular=options.angular;
            if(isolated){panel.radial_level=2;panel.angular_level=3;}
            // Start the independent value controller with a nested 1/3-point
            // estimate; failed estimates add nodes to the same workspace.
            if(loose_value || (Gradient && moving)){panel.radial_level=2;panel.angular_level=2;}
            panel.workspace=std::make_unique<PanelWorkspace<Gradient>>(p,lo,hi,true,true,options.angular,shared,repair_absolute,repair_relative,limb==0,isolated,(loose_value || Gradient)?limb:-1);
            if constexpr(!std::is_same_v<PanelWorkspace<Gradient>,RadialPanel<Gradient>>)panel.workspace->set_events(domain.events);
            if constexpr(Gradient)if(moving) {
                std::array<double,8> slope;
                for(int n=0;n<8;++n)slope[n]=(cut_motion[j][n]-cut_motion[j-1][n])/(hi-lo);
                panel.workspace->set_radial_motion(lo,cut_motion[j-1],slope);
                ++out.moving_radial_panels;
            }
            double safe_lo=lo,safe_hi=hi;
            for(auto event:domain.events) {
                if(event.radius==lo)safe_lo=std::max(safe_lo,event.radius_hi);
                if(event.radius==hi)safe_hi=std::min(safe_hi,event.radius_lo);
            }
            const double midpoint=lo+(hi-lo)*.5;
            if(midpoint>safe_lo && midpoint<safe_hi)
                panel.workspace->set_certified_band(shared->radii.at(midpoint)->topology,safe_lo,safe_hi);
            if(!panel.workspace->refine(panel.radial_level,panel.angular_level)) {
                out.reason=panel.workspace->reason;return out;
            }
            panels.push_back(std::move(panel));
        }
    }
    double event_error=out.support.bounded_component_error;
    for(auto event:out.support.events)
        event_error+=std::acos(-1.)*(event.radius_hi-event.radius_lo)*(event.radius_hi+event.radius_lo)/denominator;
    for(int iteration=0;iteration<=options.maximum_refinements;++iteration) {
        RadiusIntegral<Gradient> sum{};

        std::size_t worst=0;double worst_error=-1;
        for(std::size_t j=0;j<panels.size();++j) {
            const auto& value=panels[j].workspace->result();
            sum.f0+=value.f0;sum.fh+=value.fh;
            sum.angular_error+=value.angular_error;
            sum.endpoint_error+=value.endpoint_error;
            if constexpr(Gradient)for(int k=0;k<8;++k) {
                sum.df0[k]+=value.df0[k];sum.dfh[k]+=value.dfh[k];

            }
            double error=value.angular_error+value.endpoint_error;

            if(error>worst_error) {
                worst_error=error;worst=j;
            }

        }
        out.normalized=normalize(sum,p,limb);
        out.normalized.f0+=out.support.bounded_component_value;
        out.magnification=out.normalized.f0;
        out.error_estimate=out.normalized.angular_error+out.normalized.endpoint_error+event_error;
        out.converged=std::isfinite(out.magnification) && out.magnification>0 &&
            out.error_estimate<=options.absolute_tolerance+options.relative_tolerance*std::abs(out.magnification);
        if constexpr(Gradient)out.gradient_valid=out.converged && std::all_of(out.normalized.df0.begin(),out.normalized.df0.end(),[](double g){return std::isfinite(g);});
        if(out.converged && (!Gradient || out.gradient_valid)) {
            out.reason=Gradient?"converged_value_mesh":"converged";update_counters();
            if constexpr(!Gradient)if(trace)for(auto& panel:panels)trace->panels.emplace_back(std::move(panel.workspace));
            return out;
        }
        if(worst_error<0){out.reason="refinement_limit";update_counters();return out;}
        if(iteration==options.maximum_refinements){out.reason="refinement_budget";update_counters();return out;}
        auto& panel=panels[worst];
#ifdef LCBININT_TRIPLE_MULTIRATE_ARCS
        if constexpr(has_branch_refinement<PanelWorkspace<Gradient>>::value)if(panel.workspace->branches_active()){
            if(!panel.workspace->refine_branches(panel.angular_level)){out.reason=panel.workspace->reason;update_counters();return out;}++out.refinements;continue;
        }
#endif
        const bool radial_dominates=!loose_value?panel.radial_level<=panel.angular_level:
            panel.workspace->radial_error()>=panel.workspace->inner_error();
        if(panel.radial_level>=panel.workspace->split_level && (limb==0 ||
                ((Gradient || loose_value)?(radial_dominates || panel.angular_level>=7):panel.angular_level>=7))) {
            const double mid=panel.lo+(panel.hi-panel.lo)*.5;
            if(!(mid>panel.lo && mid<panel.hi)){out.reason="panel_resolution_limit";update_counters();return out;}
            Panel left,right;
            left.lo=panel.lo;left.hi=mid;left.left_fold=panel.left_fold;left.right_fold=false;
            right.lo=mid;right.hi=panel.hi;right.left_fold=false;right.right_fold=panel.right_fold;
            for(auto child:{&left,&right}) {
                child->shared=panel.shared;child->angular=panel.angular;
                child->radial_level=std::min(4,panel.workspace->split_level);child->angular_level=(Gradient || loose_value)?panel.angular_level:5;
                child->workspace=std::make_unique<PanelWorkspace<Gradient>>(p,child->lo,child->hi,
                    child->left_fold,child->right_fold,panel.angular,panel.shared,repair_absolute,repair_relative,limb==0,false,(loose_value || Gradient)?limb:-1);
                if constexpr(!std::is_same_v<PanelWorkspace<Gradient>,RadialPanel<Gradient>>)child->workspace->inherit_stream(*panel.workspace);
                if constexpr(Gradient)if(panel.workspace->has_radial_motion())child->workspace->set_radial_motion(panel.workspace->motion_origin(),
                    panel.workspace->motion_base(),panel.workspace->motion_slope());
                if(panel.workspace->band_topology())child->workspace->set_certified_band(
                    *panel.workspace->band_topology(),std::max(child->lo,panel.workspace->safe_lo()),
                    std::min(child->hi,panel.workspace->safe_hi()));
                if(!child->workspace->refine(child->radial_level,child->angular_level)) {
                    out.reason=child->workspace->reason;update_counters();return out;
                }
            }
            retained_parents.push_back(std::move(panel.workspace));
            panel=std::move(left);panels.push_back(std::move(right));
            ++out.h_splits;++out.refinements;continue;
        }
        if((limb==0 || radial_dominates) && panel.radial_level<8)++panel.radial_level;
        else if(panel.angular_level<10)++panel.angular_level;
        else ++panel.radial_level;
        ++out.refinements;
        if(!panel.workspace->refine(panel.radial_level,panel.angular_level)) {
            out.reason=panel.workspace->reason;update_counters();return out;
        }
    }
    update_counters();return out;
}
} // namespace lcbinint::algebraic_boundary::triple
