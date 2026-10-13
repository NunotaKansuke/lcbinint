#pragma once

// Research front end: seed-anchored radial marching, local angular topology,
// and the existing adaptive angular/Fejer kernel. No D14 enumeration.
// A successful quadrature is NOT a proof of radial-support completeness.
#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"
#include "lcbinint/magnification/holonomic/seeded_boundary_events.hpp"
#include "lcbinint/magnification/holonomic/quartic_fast_count.hpp"
#include <complex>

namespace lcbinint::holonomic {
namespace seeded_frontier_detail {
// Keep only new sample witnesses. An in-place sample correction is noticed
// whenever its radius or certified count changes. Invalid samples remain
// eligible on a later pass; the union is identical to repeated collection.
struct SampleWitnessCollector {
    std::vector<std::pair<double,int>> seen;
    void append(const std::vector<AdaptiveSample>& samples,
                std::vector<std::pair<double,int>>& probes) {
        seen.resize(samples.size(),{0.0,std::numeric_limits<int>::min()});
        for(std::size_t j=0;j<samples.size();++j) {
            const auto& sample=samples[j];
            if(!sample.reliable || sample.local_crossings==-1)continue;
            const std::pair<double,int> witness{sample.R,sample.local_crossings};
            if(witness!=seen[j]) {probes.push_back(witness);seen[j]=witness;}
        }
    }
};
inline void sort_unique_probes(std::vector<std::pair<double,int>>& probes,
                              std::size_t& sorted_prefix) {
    auto middle=probes.begin()+static_cast<std::ptrdiff_t>(sorted_prefix);
    std::sort(middle,probes.end());
    if(sorted_prefix && middle!=probes.end())
        std::inplace_merge(probes.begin(),middle,probes.end());
    probes.erase(std::unique(probes.begin(),probes.end()),probes.end());
    sorted_prefix=probes.size();
}
}

struct SeededFrontierStats {
    int seeds=0,skipped_seeds=0,radius_probes=0,bands=0;
    int image_start_attempts=0,image_start_accepted=0;
    int invariant_count_probes=0,sturm_count_probes=0;
    double planning_ms=0,integration_ms=0;
#ifdef HOLO_FRONTIER_COST_PROFILE
    double physical_ms=0,estimator_ms=0,scheduler_ms=0,setup_ms=0;
    double setup_event_ms=0,setup_panel_ms=0,allocation_ms=0;
    std::size_t allocation_count=0;
    std::size_t integration_calls=0;
#endif
    bool discovery_completed=false,coverage_certified=false;
    int local_repairs=0;
    std::size_t retained_nodes=0;
    std::size_t evaluated_nodes=0;
    const char* reason="not_started";
};
struct SeededFrontierResult {
    AdaptiveResult integral;
    SeededFrontierStats frontier;
    TopologyResult topology;
};

inline SeededFrontierResult epoch_seeded_radial_frontier(
    const LensParams& p,double u,const std::vector<Cplx<double>>& seeds,
    const AdaptiveConfig& cfg,AdaptiveWorkspace& workspace,
    bool midpoint_split=false) {
    SeededFrontierResult out;
    const auto start=adaptive_detail::Clock::now();
    const auto policy=cfg.effective_gradient_policy();
    if(!adaptive_detail::valid_epoch_config(p,u,cfg) || seeds.empty()) {
        out.frontier.reason="invalid_config";
        workspace.reset();out.integral=adaptive_detail::failure_result(
            AdaptiveStop::InvalidConfig,policy);return out;
    }
    if(cfg.require_bound) {
        out.frontier.reason="bound_unavailable";
        workspace.reset();out.integral=adaptive_detail::failure_result(
            AdaptiveStop::BoundUnavailable,policy);return out;
    }
    const auto pf=PrimaryFrame::from(p);
    auto stable_coeff=[&](double R,bool reciprocal=false) {
#ifdef HOLO_FRONTIER_LEGACY_PROBE_COEFFICIENTS
        const auto f=local_fold_quantities<long double>((long double)R,0.L,pf,reciprocal);
        return std::array<double,5>{(double)f.P,(double)f.Pt,(double)(f.Ptt/2),
            (double)(f.Psss/6),(double)(f.Pssss/24)};
#else
        // Count-only probes need the coefficients, not radial derivatives
        // or the angular derivative jet at zero. Keep extended arithmetic.
        const long double r=R,a=pf.a,m=pf.m0,X=pf.X,Y=pf.Y,rho=pf.rho;
        const long double r2=r*r,n0r=-X*r2,n0i=-Y*r2;
        const long double n1r=r*(r2-1.L+a*X),n1i=r*a*Y,n2r=a*(m-r2);
        const long double c0r=n0r+n1r+n2r,c0i=n0i+n1i;
        const long double c1r=2.L*n0i,c1i=2.L*(n2r-n0r);
        const long double c2r=n1r-n0r-n2r,c2i=n1i-n0i;
        const long double k=rho*rho*r2,bm=(r-a)*(r-a),bp=(r+a)*(r+a);
        std::array<double,5> c{{
            (double)(k*bm-(c0r*c0r+c0i*c0i)),
            (double)(-2.L*(c0r*c1r+c0i*c1i)),
            (double)(k*(bm+bp)-(c1r*c1r+c1i*c1i)-2.L*(c0r*c2r+c0i*c2i)),
            (double)(-2.L*(c1r*c2r+c1i*c2i)),
            (double)(k*bp-(c2r*c2r+c2i*c2i))}};
        if(reciprocal)return std::array<double,5>{c[4],-c[3],c[2],-c[1],c[0]};
        return c;
#endif
    };
    struct ImageStart {double radius; Cplx<double> image;};
    std::vector<ImageStart> image_starts;
    image_starts.reserve(seeds.size());
    std::vector<double> radii;
    for(const auto& z:seeds) {
        const double r=std::hypot(z.re,z.im);
        if(r>0 && std::isfinite(r)) {
            radii.push_back(r);
            image_starts.push_back({r,z});
        }
    }
    std::sort(radii.begin(),radii.end());
    radii.erase(std::unique(radii.begin(),radii.end()),radii.end());
    out.frontier.seeds=static_cast<int>(radii.size());
    std::vector<std::pair<double,int>> probes;
    std::size_t sorted_probe_prefix=0;
    seeded_frontier_detail::SampleWitnessCollector sample_witnesses;
    auto has_image=[&](double R) {
        V2ProfileTimer timer(&V2Profile::frontier_count_ms);
        ++out.frontier.radius_probes;
        if(!(R>0))return 0;
        bool direct=false;
        const int count=boundary_quartic_count(stable_coeff(R),&direct);
        if(direct)++out.frontier.invariant_count_probes;
        else ++out.frontier.sturm_count_probes;
        if(count<0)return -1;
        const bool full=count==0 && phi_lens(R,.371,pf)>0;
        probes.emplace_back(R,full?-2:count);
        return count>0 || full ? 1:0;
    };
    const double image_radius_bound=pf.a+std::hypot(pf.X,pf.Y)+pf.rho+2.;
    auto local_event=[&](double R,double lo,double hi,RadialEvent& event,bool image_start_only=false) {
        V2ProfileTimer timer(&V2Profile::frontier_event_ms);
        bool found=false;double nearest=std::numeric_limits<double>::infinity();
        std::vector<std::pair<double,bool>> proposals;
        auto attempt=[&](double coordinate,bool reciprocal,bool high_precision=false) {
            if(!std::isfinite(coordinate))return false;
            if(!high_precision && !image_start_only)proposals.emplace_back(coordinate,reciprocal);
            RadialEvent candidate;
            if(!high_precision) {
                if(!seeded_boundary_detail::fold(R,coordinate,reciprocal,pf,
                    image_radius_bound,candidate))return false;
            } else {
                __float128 rr=R,tt=coordinate;double residual=0,step=0;int iterations=0;
                const bool ok=reciprocal
                    ? adaptive_detail::coupled_fold_newton<__float128,true>(rr,tt,pf,
                        24,1e-28,1e-24,&iterations,&residual,&step)
                    : adaptive_detail::coupled_fold_newton<__float128,false>(rr,tt,pf,
                        24,1e-28,1e-24,&iterations,&residual,&step);
                if(!ok || !(rr>0 && rr<image_radius_bound))return false;
                adaptive_detail::EventLocation location;
                adaptive_detail::store_refined_radius(location,rr,step,2);
                candidate.radius=location.radius;candidate.radius_lo=location.radius_lo;
                candidate.radius_uncertainty=location.uncertainty;candidate.precision_tier=2;
                candidate.kind="physical_real";candidate.physically_real=true;
                if(reciprocal){candidate.fold_u_seed=(double)tt;candidate.fold_u_seed_valid=true;}
                else{candidate.fold_t_seed=(double)tt;candidate.fold_t_seed_valid=true;}
            }
            if(candidate.radius<lo || candidate.radius>hi)return false;
            // Clearing lens denominators introduces stationary solutions at
            // lens poles. Check the original lens equation before accepting
            // a polynomial fold. This is a proposal-residual test; endpoint
            // precision is still selected by the existing adaptive kernel.
            const long double m=reciprocal?candidate.fold_u_seed:candidate.fold_t_seed;
            const long double den=1+m*m;
            const std::complex<long double> direction(
                reciprocal?(m*m-1)/den:(1-m*m)/den,
                reciprocal?-2*m/den:2*m/den);
            const auto image=((long double)candidate.radius+candidate.radius_lo)*direction;
            const auto mapped=image-(long double)pf.m0/std::conj(image)-
                (1.L-pf.m0)/(std::conj(image)-(long double)pf.a);
            const long double distance=std::abs(mapped-std::complex<long double>(pf.X,pf.Y));
            const long double scale=std::max({1.L,(long double)candidate.radius,
                (long double)pf.rho,(long double)std::hypot(pf.X,pf.Y)});
            if(!std::isfinite(distance) || std::fabs(distance-pf.rho)>
                std::sqrt(std::numeric_limits<double>::epsilon())*scale)return false;
            if(std::fabs(candidate.radius-R)<nearest) {
                nearest=std::fabs(candidate.radius-R);event=candidate;found=true;
            }
            return true;
        };

        if(image_start_only) {
            const ImageStart* nearest_start=nullptr;
            double distance=std::numeric_limits<double>::infinity();
            for(const auto& start:image_starts)if(std::fabs(start.radius-R)<distance) {
                distance=std::fabs(start.radius-R);nearest_start=&start;
            }
            if(!nearest_start)return false;
            const auto& z=nearest_start->image;
            const double r=nearest_start->radius;
            const bool reciprocal=z.re<0;
            const double coordinate=reciprocal?-z.im/(r-z.re):z.im/(r+z.re);
            return attempt(coordinate,reciprocal);
        }
        const auto seed=adaptive_detail::double_event_estimate(R,pf,cfg.tol.mu_rtol,
#ifdef HOLO_FRONTIER_LEGACY_EVENT_SEED
            false
#else
            true
#endif
        );
        if(seed.t_seed_valid) {
            const bool reciprocal=std::fabs(seed.t_seed)>1;
            if(attempt(reciprocal?-1/seed.t_seed:seed.t_seed,reciprocal))return true;
            if(attempt(reciprocal?seed.t_seed:-1/seed.t_seed,!reciprocal))return true;
        }
        for(bool reciprocal:{false,true}) {
            const auto pc=stable_coeff(R,reciprocal);
            double derivative[4]={4*pc[4],3*pc[3],2*pc[2],pc[1]};
            int degree=3;
            while(degree>0 && derivative[0]==0.) {
                for(int j=0;j<degree;++j)derivative[j]=derivative[j+1];--degree;
            }
            if(!degree)continue;
            for(auto z:aberth<double>(derivative,degree,32)) {
                if(std::fabs(z.im)>1e-7*(1+std::fabs(z.re)))continue;
                attempt(z.re,reciprocal);
            }
        }
        if(!found) {
            // Use the actual angular interval as the local continuation
            // anchor when a stationary-polynomial seed does not converge.
            // This extra solve is confined to the unresolved endpoint.
            const auto cert=certify_quartic(stable_coeff(R));
            if(cert.certified && cert.root_count>0) {
                const auto angles=adaptive_detail::adaptive_sturm_thetas(cert);
                const auto arcs=arc_set_from_root_thetas(R,pf,angles);
                for(auto arc:arcs.arcs) {
                    const double t0=std::tan(arc[0]/2),t1=std::tan(arc[1]/2);
                    const double u0=-1/t0,u1=-1/t1;
                    const bool reciprocal=std::max(std::fabs(u0),std::fabs(u1))<
                        std::max(std::fabs(t0),std::fabs(t1));
                    attempt(reciprocal?.5*(u0+u1):.5*(t0+t1),reciprocal);
                }
            }
        }
        if(!found)for(auto proposal:proposals)
            if(attempt(proposal.first,proposal.second,true))break;
        return found;
    };
    struct SolvedEdge {double inside,outside;RadialEvent event;};
    std::vector<SolvedEdge> solved_edges;
    auto march=[&](double r,double initial_step,int direction,double& edge,
                   double& exterior,double& refined_exterior) {
#ifdef HOLO_FRONTIER_COST_PROFILE
        V2ProfileTimer march_timer(&V2Profile::frontier_march_ms);
#endif
        double step=std::max(1e-14*(1+r),initial_step),active=r;
        for(int i=0;i<128;++i) {
            const double candidate=std::max(0.,active+direction*step);
            const int state=has_image(candidate);
            if(state<0)return false;
            if(state==0) {
                double inside=active,outside=candidate;
#ifndef HOLO_FRONTIER_DISABLE_SOLVED_BRACKET
                // Newton locates the physical fold; count witnesses still
                // validate both sides inside the original march bracket.
                // No count or physical acceptance gate is weakened.
                RadialEvent fold;
                ++out.frontier.image_start_attempts;
                if(local_event(inside,std::min(inside,outside),
                               std::max(inside,outside),fold,true)) {
                    const double inner=.5*(inside+fold.radius);
                    const double outer=.5*(outside+fold.radius);
                    if(inner!=outer && has_image(inner)==1 && has_image(outer)==0) {
                        ++out.frontier.image_start_accepted;
                        solved_edges.push_back({inner,candidate,fold});
                        edge=inner;exterior=candidate;refined_exterior=outer;
                        return true;
                    }
                }
#endif
                for(int j=0;j<12;++j) {
                    double mid=.5*(inside+outside);int m=has_image(mid);
                    if(m<0)return false;
                    if(m)inside=mid;else outside=mid;
                }
                // Retain the original sign-changing bracket. Near an
                // ill-conditioned fold, coefficient rounding can shift the
                // final boolean bisection relative to the lens-map Newton.
                edge=inside;exterior=candidate;refined_exterior=outside;return true;
            }
            active=candidate;step*=2.;
        }
        return false;
    };
    struct Band {double r_lo,r_hi,outer_lo,outer_hi,near_outer_lo,near_outer_hi;};
    std::vector<Band> bands;
    for(double r:radii) {
        // Radial overlap is not image-component connectivity. Even a seed
        // whose band was visited contributes its cheap topology witness;
        // only the duplicate outward/inward march is skipped.
        const int state=has_image(r);
        bool covered=false;
        for(const auto& b:bands)if(r>=b.r_lo && r<=b.r_hi){covered=true;break;}
        if(covered){++out.frontier.skipped_seeds;continue;}
        if(state!=1) {
            out.frontier.reason="seed_radius_unresolved";
            out.integral=adaptive_detail::failure_result(AdaptiveStop::TopologyUnresolved,policy);
            return out;
        }
        double lo=r,hi=r,outer_lo=r,outer_hi=r,near_outer_lo=r,near_outer_hi=r;
        double initial_step=pf.rho;
        for(const auto& start:image_starts)if(start.radius==r) {
            const auto& z=start.image;
            const std::complex<double> image(z.re,z.im),e=image/r;
            const auto kappa=pf.m0/(image*image)+(1-pf.m0)/
                ((image-pf.a)*(image-pf.a));
            const double motion=std::abs(e+std::conj(kappa)*std::conj(e));
            if(motion>0 && std::isfinite(motion))initial_step=std::min(initial_step,.5*pf.rho/motion);
        }
        // Only outer support edges are refined. Interior notches are legal:
        // the angular kernel returns zero locally instead of a solidity scan.
        if(!march(r,initial_step,-1,lo,outer_lo,near_outer_lo) ||
           !march(r,initial_step,+1,hi,outer_hi,near_outer_hi) || !(hi>lo)) {
            out.frontier.reason="radial_march_unresolved";
            out.integral=adaptive_detail::failure_result(AdaptiveStop::TopologyUnresolved,policy);
            return out;
        }
        bands.push_back({lo,hi,outer_lo,outer_hi,near_outer_lo,near_outer_hi});
    }
    std::sort(bands.begin(),bands.end(),[](auto a,auto b){return a.r_lo<b.r_lo;});
    std::vector<Band> merged;
    for(auto b:bands) {
        if(merged.empty() || b.r_lo>merged.back().r_hi)merged.push_back(b);
        else if(b.r_hi>merged.back().r_hi) {
            merged.back().r_hi=b.r_hi;merged.back().outer_hi=b.outer_hi;
            merged.back().near_outer_hi=b.near_outer_hi;
        }
    }
    // These are quadrature anchors, not component-connectivity assertions.
    auto& topo=out.topology;
    // Reuse count changes already encountered by the march. No independent
    // dense radial scan: refine only intervals with observed topology change.
    auto repair_observed=[&](bool include_empty=false) {
#ifdef HOLO_FRONTIER_COST_PROFILE
    V2ProfileTimer repair_timer(&V2Profile::frontier_repair_ms);
#endif
    const std::size_t before=topo.events.size();
#ifdef HOLO_FRONTIER_REPEAT_WITNESSES
    std::sort(probes.begin(),probes.end());
    probes.erase(std::unique(probes.begin(),probes.end()),probes.end());
#else
    seeded_frontier_detail::sort_unique_probes(probes,sorted_probe_prefix);
#endif
    const auto& observed=probes;
    for(std::size_t i=1;i<observed.size();++i) {
        const int left_count=std::max(0,observed[i-1].second);
        const int right_count=std::max(0,observed[i].second);
        if(left_count==right_count)continue;
        // Exterior zero-count transitions are handled by the support-edge
        // solve. During quadrature, every newly observed count change can
        // refine its own interval, including full-circle transitions.
        if(!include_empty && (!left_count || !right_count))continue;
        double lo=observed[i-1].first,hi=observed[i].first;
        bool already_partitioned=false;
        for(const auto& old:topo.events)if(old.radius>=lo && old.radius<=hi)already_partitioned=true;
        if(already_partitioned)continue;
        const int left=left_count;
        for(int j=0;j<8;++j) {
            const double mid=.5*(lo+hi);
            bool direct=false;
            const int count=boundary_quartic_count(stable_coeff(mid),&direct);
            if(direct)++out.frontier.invariant_count_probes;
            else ++out.frontier.sturm_count_probes;
            ++out.frontier.radius_probes;
            if(count<0)break;
            if(count==left)lo=mid;else hi=mid;
        }
        RadialEvent event;
        bool seeded=false;
#ifndef HOLO_FRONTIER_LEGACY_IMAGE_START
        ++out.frontier.image_start_attempts;
        seeded=local_event(.5*(lo+hi),lo,hi,event,true);
        if(seeded)++out.frontier.image_start_accepted;
#endif
        if(seeded || local_event(.5*(lo+hi),lo,hi,event)) {
            bool known=false;
            for(const auto& old:topo.events)if(std::fabs(event.radius-old.radius)<=
                8*std::numeric_limits<double>::epsilon()*(1+event.radius))known=true;
            if(!known)topo.events.push_back(event);
        }
    }
    return topo.events.size()-before;
    };
    repair_observed();
    bool endpoints_resolved=true;
    for(auto b:merged) {
        // A coarse bracket locates the two exterior folds. Correct only
        // those endpoints, retaining split radii for the existing fold map.
        auto endpoint=[&](double R,double exterior,double refined_exterior) {
#ifdef HOLO_FRONTIER_COST_PROFILE
        V2ProfileTimer endpoint_timer(&V2Profile::frontier_endpoint_ms);
#endif
            RadialEvent e;
#ifndef HOLO_FRONTIER_DISABLE_SOLVED_BRACKET
            for(const auto& solved:solved_edges)
                if(solved.inside==R && solved.outside==exterior) {
                    e=solved.event;
                    for(const auto& old:topo.events)
                        if(std::fabs(e.radius-old.radius)<=8*std::numeric_limits<double>::epsilon()*(1+e.radius))return old.radius;
                    e.detail="seed frontier witnessed Newton fold";
                    topo.events.push_back(e);return e.radius;
                }
#endif
            // The bracket initializes Newton; acceptance uses the physical
            // stationary equations, not an empirical distance-to-bracket cut.
            // The original proposal selects a nearby stationary fold; the
            // image-seed shortcut additionally requires the refined bracket.
            bool seeded=false;
#ifndef HOLO_FRONTIER_LEGACY_IMAGE_START
            // Reuse the final bisection bracket for this shortcut. A wider
            // bracket remains available to the original fallback, since
            // coefficient rounding may shift a nearly singular boundary.
            // A seed proposal is accepted only within that bracket; otherwise
            // retain the original quartic proposal and local fallback.
            ++out.frontier.image_start_attempts;
            seeded=local_event(R,std::min(R,refined_exterior),std::max(R,refined_exterior),e,true);
            if(seeded)++out.frontier.image_start_accepted;
#endif
            if(!seeded && !local_event(R,std::min(R,exterior),
                                         std::max(R,exterior),e)) {
                // Incremental fallback: keep the bracket and add precision
                // only here, rather than rebuilding topology with D14.  A
                // fold elsewhere in the image disk is not a valid substitute
                // for the edge witnessed by this count-changing bracket.
                for(int j=0;j<16;++j) {
                    double mid=.5*(R+exterior);int state=has_image(mid);
                    if(state<0)break;
                    if(state)R=mid;else exterior=mid;
                }
                if(!local_event(R,std::min(R,exterior),
                                  std::max(R,exterior),e)) {
                    endpoints_resolved=false;return R;
                }
            }
            for(const auto& old:topo.events)if(std::fabs(e.radius-old.radius)<=
                8*std::numeric_limits<double>::epsilon()*(1+e.radius))return old.radius;
            e.detail="seed frontier exterior fold";topo.events.push_back(e);
            return e.radius;
        };
        b.r_lo=endpoint(b.r_lo,b.outer_lo,b.near_outer_lo);
        b.r_hi=endpoint(b.r_hi,b.outer_hi,b.near_outer_hi);
        bool retains_seed=false;
        for(double r:radii)if(r>=b.r_lo && r<=b.r_hi){retains_seed=true;break;}
        if(!(b.r_hi>b.r_lo) || !retains_seed) {
            endpoints_resolved=false;
            break;
        }
        if(b.r_lo>topo.r_max)
            topo.cells.push_back({static_cast<int>(topo.cells.size()),topo.r_max,b.r_lo,
                .5*(topo.r_max+b.r_lo),ArcKind::kEmpty,0,Status::OK});
        std::vector<double> cuts{b.r_lo,b.r_hi};
        for(const auto& e:topo.events)if(e.radius>b.r_lo && e.radius<b.r_hi)cuts.push_back(e.radius);
        if(midpoint_split)cuts.push_back(.5*(b.r_lo+b.r_hi));
        std::sort(cuts.begin(),cuts.end());
        cuts.erase(std::unique(cuts.begin(),cuts.end()),cuts.end());
        for(std::size_t i=1;i<cuts.size();++i)if(cuts[i]>cuts[i-1])
            topo.cells.push_back({static_cast<int>(topo.cells.size()),cuts[i-1],cuts[i],
                .5*(cuts[i-1]+cuts[i]),ArcKind::kArcs,-1,Status::OK});
        topo.r_max=std::max(topo.r_max,b.r_hi);
    }
    topo.status=topo.cells.empty() || !endpoints_resolved?Status::TOPOLOGY_UNCERTAIN:Status::OK;
    out.frontier.bands=static_cast<int>(merged.size());
    out.frontier.discovery_completed=topo.status==Status::OK;
    out.frontier.reason=endpoints_resolved?"local_discovery_completed":"endpoint_unresolved";
    out.frontier.planning_ms=adaptive_detail::ms(start);
    // Deliberately no global event repair or legacy restart. Only the existing
    // adaptive nodes are refined, with accepted samples reused by that kernel.
    AdaptiveConfig exploration_cfg=cfg;
#ifndef HOLO_FRONTIER_DISABLE_WEIGHTED_RADIAL
    if(u==0 && cfg.effective_gradient_policy()==GradientPolicy::None && !cfg.with_jacobian)
        exploration_cfg.weighted_radial=true;
#endif
#ifndef HOLO_FRONTIER_DISABLE_ANGULAR_VALUE_BUDGET
    if(u>0 && cfg.effective_gradient_policy()==GradientPolicy::None && !cfg.with_jacobian)
        exploration_cfg.angular_value_budget_fraction=0.05;
#endif
#ifndef HOLO_FRONTIER_DISABLE_WARM_SEED_TOLERANCE
    if(cfg.effective_gradient_policy()==GradientPolicy::None && !cfg.with_jacobian)
        exploration_cfg.warm_root_seed_tolerance=1e-10;
#endif
    exploration_cfg.yield_on_local_topology_change=true;
#ifndef HOLO_FRONTIER_DISABLE_ENDPOINT_BUDGET
    if(cfg.effective_gradient_policy()==GradientPolicy::None && !cfg.with_jacobian)
        exploration_cfg.uniform_endpoint_budget_fraction=0.01;
#endif
    // Geometry, tolerance and existing event records are immutable within
    // this epoch. Repairs only append events; keep earlier error estimates.
    std::vector<std::pair<double,adaptive_detail::EventLocation>> epoch_event_errors;
    // The fast weighted controller is selected only when every initial
    // occupied span has the required endpoint structure. Do not impose its
    // conservative ledger on unrelated complex spans for a speculative gain.
    if(exploration_cfg.weighted_radial) {
        for(std::size_t i=0;i<topo.cells.size();++i) {
            const auto& cell=topo.cells[i];
            if(cell.kind==ArcKind::kEmpty)continue;
            bool left=false,right=false,seen=false,isolated=true;
            for(const auto& event:topo.events)if(adaptive_detail::adaptive_physical_fold_event(event)) {
                left=left||event.radius==cell.r_lo;right=right||event.radius==cell.r_hi;
            }
            for(const auto& probe:probes)if(probe.first>cell.r_lo && probe.first<cell.r_hi) {
                seen=true;if(probe.second!=2){isolated=false;break;}
            }
            const bool outside_left=i==0||topo.cells[i-1].kind==ArcKind::kEmpty;
            const bool outside_right=i+1==topo.cells.size()||topo.cells[i+1].kind==ArcKind::kEmpty;
            if(cell.kind!=ArcKind::kArcs || !left || !right || !seen || !isolated ||
               !outside_left || !outside_right) {
                exploration_cfg.weighted_radial=false;break;
            }
        }
    }
    std::vector<std::pair<double,double>> weighted_spans;
    auto run_integral=[&](bool retain) {
        weighted_spans.clear();
        if(exploration_cfg.weighted_radial) {
            for(const auto& cell:topo.cells)if(cell.kind==ArcKind::kArcs) {
                bool seen=false,isolated=true;
                for(const auto& probe:probes)if(probe.first>cell.r_lo && probe.first<cell.r_hi) {
                    seen=true;if(probe.second!=2){isolated=false;break;}
                }
                if(seen && isolated)weighted_spans.emplace_back(cell.r_lo,cell.r_hi);
            }
            exploration_cfg.weighted_radial_spans=&weighted_spans;
        }

        auto result=flux_adaptive_integrate(p,u,topo,exploration_cfg,workspace,retain,
            cfg.collect_diagnostics?nullptr:&epoch_event_errors);
#ifdef HOLO_FRONTIER_COST_PROFILE
        out.frontier.allocation_ms+=result.stats.allocation_ms;
        out.frontier.allocation_count+=result.stats.allocation_count;
        out.frontier.physical_ms+=result.stats.physical_ms;
        out.frontier.estimator_ms+=result.stats.estimator_ms;
        out.frontier.scheduler_ms+=result.stats.scheduler_ms;
        out.frontier.setup_ms+=result.stats.setup_ms;
        out.frontier.setup_event_ms+=result.stats.setup_event_ms;
        out.frontier.setup_panel_ms+=result.stats.setup_panel_ms;
        ++out.frontier.integration_calls;
#endif
        return result;
    };
    auto integration_start=adaptive_detail::Clock::now();
    out.integral=run_integral(false);
    out.frontier.integration_ms+=adaptive_detail::ms(integration_start);
    // Integration nodes are exploration probes too. A newly observed count
    // change only splits its affected span; all unchanged panel samples and
    // boundary continuation state survive the additional adaptive call.
    for(std::size_t pass=0;pass<cfg.max_panels;++pass) {
        const auto repair_start=adaptive_detail::Clock::now();
#ifdef HOLO_FRONTIER_REPEAT_WITNESSES
        for(const auto& sample:workspace.samples)
            if(sample.reliable && sample.local_crossings!=-1)
                probes.emplace_back(sample.R,sample.local_crossings);
#else
        sample_witnesses.append(workspace.samples,probes);
#endif
        const auto added=repair_observed(true);
        if(!added) {
            out.frontier.planning_ms+=adaptive_detail::ms(repair_start);
            if(out.integral.stats.local_topology_yields>0) {
                // A count witness can be too close to a rounded fold for
                // local Newton to identify a new event. Preserve the partial
                // panel and obtain additional quadrature witnesses in place.
                exploration_cfg.yield_on_local_topology_change=false;
                integration_start=adaptive_detail::Clock::now();
                out.integral=run_integral(true);
                out.frontier.integration_ms+=adaptive_detail::ms(integration_start);
                out.frontier.retained_nodes+=out.integral.stats.reused_nodes;
                continue;
            }
            break;
        }
        exploration_cfg.yield_on_local_topology_change=true;
        std::vector<CellPlan> split;
        for(auto cell:topo.cells) {
            std::vector<double> cuts{cell.r_lo,cell.r_hi};
            for(const auto& e:topo.events)if(e.radius>cell.r_lo && e.radius<cell.r_hi)cuts.push_back(e.radius);
            std::sort(cuts.begin(),cuts.end());cuts.erase(std::unique(cuts.begin(),cuts.end()),cuts.end());
            for(std::size_t i=1;i<cuts.size();++i) {
                auto part=cell;part.r_lo=cuts[i-1];part.r_hi=cuts[i];
                part.r_mid=.5*(part.r_lo+part.r_hi);part.index=static_cast<int>(split.size());
                split.push_back(part);
            }
        }
        topo.cells=std::move(split);++out.frontier.local_repairs;
        out.frontier.planning_ms+=adaptive_detail::ms(repair_start);
        integration_start=adaptive_detail::Clock::now();
        auto repaired=run_integral(true);
        out.frontier.integration_ms+=adaptive_detail::ms(integration_start);
        out.frontier.retained_nodes+=repaired.stats.reused_nodes;
        out.integral=std::move(repaired);
    }
    out.frontier.evaluated_nodes=workspace.samples.size();
    out.integral.stats.topology_ms=out.frontier.planning_ms;
    return out;
}
} // namespace lcbinint::holonomic
